from src.agent.environment import batch_interact_environment
import numpy as np
import wandb
import os
import torch
import time
from typing import List, Dict
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.llms.utils import DummyDataset
from typing import Tuple
import copy

def dict_mean(dict_list):
    mean_dict = {}
    if len(dict_list) > 0:
        for key in dict_list[0].keys():
            mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict


class AskingTrainer:
    def __init__(self,
                 agent,
                 accelerator,
                 tokenizer,
                 critic_lr: float = 1e-3,
                 lm_lr: float = 1e-5,
                 grad_accum_steps: int = 8,
                 gamma: float = 0.9,
                 tau: float = 0.1,
                 epochs: int = 3,
                 max_grad_norm: float = 0.01,
                 actor_epochs: int = 3,
                 mode="ON"
                 ):
        """
        """
        super().__init__()
        self.agent = agent
        self.tokenizer = tokenizer
        self.lm_optimizer = torch.optim.Adam(agent.model.parameters(), lr=lm_lr)
        self.critic_optimizer = torch.optim.Adam(agent.critic.parameters(), lr=critic_lr)
        self.criterion = torch.nn.MSELoss()
        self.grad_accum_steps = grad_accum_steps
        self.actor_epochs = actor_epochs
        self.gamma = gamma
        self.epochs = epochs
        self.step = 0
        self.tau = tau
        self.max_grad_norm = max_grad_norm
        self.accelerator = accelerator
        self.critic_optimizer, self.lm_optimizer = self.accelerator.prepare(self.critic_optimizer, self.lm_optimizer)
        self.device = self.accelerator.unwrap_model(self.agent.model).device
        self.dtype = self.accelerator.unwrap_model(self.agent.model).dtype

    def critic_loss(self, observation, action, reward, next_observation, done, mc_return, **kwargs):
        reward = torch.Tensor(reward).to(device=self.device, dtype=self.dtype).flatten()
        done = torch.Tensor(done).to(device=self.device, dtype=self.dtype).flatten()
        v = self.agent.critic(observation, detach_model=False).flatten()

        # V value
        with torch.no_grad():
            v_next = self.agent.critic(next_observation, detach_model=False).flatten()

        td_error = self.criterion(v, reward + (1-done) * v_next * self.gamma)
        self.accelerator.backward(td_error)
        td_error, v, v_next = td_error.detach().cpu(), v.detach().cpu(), v_next.detach().cpu

        return {
            "td_error": td_error,
            "v.mean": torch.mean(v),
            "v.min": torch.min(v),
            "v.max": torch.max(v),
            "v.std": torch.std(v),
            "v_next.mean": torch.mean(v_next),
            "v_next.max": torch.max(v_next),
            "v_next.min": torch.min(v_next),
            "v_next.std": torch.std(v_next),
        }

    def actor_loss(self, observation, pi_action, advantage, **kwargs):
        action = pi_action
        log_prob = self.agent.get_log_prob(observation, action)
        advantage = torch.Tensor(advantage).to(device=self.device, dtype=self.dtype)
        # in the case where a baseline is used
        if isinstance(log_prob, Tuple):
            values, log_prob, mask = log_prob
            values = values.squeeze(-1)
            advantage = advantage.reshape(-1, 1).broadcast_to(values.size())
            value_loss = torch.mean(((advantage - values) * mask) ** 2)
            with torch.no_grad():
                residual_advantage = advantage - values
            pg_loss = -torch.mean(torch.sum(residual_advantage * log_prob * mask, dim=1))

        else:
            advantages = advantage.flatten()
            values = torch.zeros_like(advantages)
            residual_advantage = torch.zeros_like(advantages)
            pg_loss = -torch.mean(log_prob.flatten() * advantages)
            value_loss = torch.zeros_like(pg_loss)
        advantages = advantage.flatten()
        self.accelerator.backward(pg_loss + value_loss)
        advantages = advantages.detach().cpu()
        return {
            "pg.loss": pg_loss.detach().cpu().item(),
            "values.loss": value_loss.detach().cpu().item(),
            "values.mean": values.mean(),
            "values.max": torch.max(values),
            "values.min": torch.min(values),
            "values.std": torch.std(values),
            "advantages.mean": advantages.mean(),
            "advantages.max": torch.max(advantages),
            "advantages.min": torch.min(advantages),
            "advantages.std": torch.std(advantages),
            "residual_advantages.mean": residual_advantage.mean(),
            "residual_advantages.max": torch.max(residual_advantage),
            "residual_advantages.min": torch.min(residual_advantage),
            "residual_advantages.std": torch.std(residual_advantage)
        }

    def update(self, replay_buffer, no_update_actor=False):
        self.step += 1
        info = {}
        info_list = []
        with torch.autograd.set_detect_anomaly(True):
            for _ in range(self.epochs):
                data = [replay_buffer.sample(1) for _ in range(self.grad_accum_steps * replay_buffer.batch_size)]
                for d in data:
                    for k, v in d.items():
                        d[k] = v[0]
                dataloader = DataLoader(DummyDataset(data), batch_size=replay_buffer.batch_size)
                dataloader = self.accelerator.prepare(dataloader)
                self.critic_optimizer.zero_grad()
                for batch in tqdm(dataloader, disable=True):
                    info_list.append(self.critic_loss(**batch))
                self.accelerator.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                self.agent.soft_update_target_critic(tau=self.tau)
        info.update(dict_mean(info_list))

        # Update Actor
        info_list = []
        if not no_update_actor:
            print(">>>updating actor")
            action_bsize = 2 if 'mistral' in self.agent.policy_lm else replay_buffer.batch_size
            for _ in range(self.actor_epochs):
                data = [replay_buffer.sample(1) for _ in range(self.grad_accum_steps * replay_buffer.batch_size)]
                for d in data:
                    for k, v in d.items():
                        d[k] = v[0]
                dataloader = DataLoader(DummyDataset(data), batch_size=action_bsize, shuffle=False)
                dataloader = self.accelerator.prepare(dataloader)
                self.lm_optimizer.zero_grad()
                for batch in dataloader:
                    with torch.no_grad():
                        pi_action = self.agent.get_action(batch["observation"])
                        q1, q2, v1, v2 = self.agent.critic(batch["observation"], pi_action)
                        q = torch.minimum(q1, q2)
                        v = torch.minimum(v1, v2)
                        advantages = q - v
                    info_list.append(self.actor_loss(**batch, pi_action=pi_action, advantage=advantages))
                self.accelerator.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.lm_optimizer.step()
        info.update(dict_mean(info_list))
        return info

    def save(self, path):
        torch.save({
            'model_state_dict': self.accelerator.unwrap_model(self.agent.model).state_dict(),
            'critic_state_dict': self.accelerator.unwrap_model(self.agent.critic).state_dict(),
            'target_critic_state_dict': self.accelerator.unwrap_model(self.agent.target_critic).state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'lm_optimizer_state_dict': self.lm_optimizer.state_dict()}, path
        )

    def load(self, path):
        checkpoint = torch.load(path)
        self.agent.model.load_state_dict(checkpoint['model_state_dict'])
        self.agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.agent.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.lm_optimizer.load_state_dict(checkpoint['lm_optimizer_state_dict'])
        return self.agent


class TrainerLoop:
    def __init__(
            self,
            env,
            eval_env,
            agent,
            tokenizer,
            accelerator,
            rollout_size: int = 50,
            eval_size: int = 1,
            batch_size: int = 2,
            capacity: int = 500000,
            iterations: int = 10,
            epochs: int = 3,
            grad_accum_steps: int = 1,
            env_idx: int = None,
            critic_lr: float = 1e-4,
            lm_lr: float = 1e-5,
            actor_epochs: int = 3,
            agent_type: str = "asking",
            gamma: float = 0.9,
            tau: float = 0.1,
            max_grad_norm: float = 0.01,
            do_sample: bool = False,
            temperature: float = 2.0,

            use_wandb: bool = False,
            env_load_path: str = '',
            save_path: str = None,
            save_freq: int = 25,
            eval_freq: int = 25,
            decode_f: callable = lambda x: x,
            **kwargs
    ):
        for key, value in locals().items():
            if key != 'self':
                setattr(self, key, value)

    def run(self, agent_policy: List, env, mode="ON"):
        """
        if len(agent_policy) > 1, default as multi-agents mode and the first one is the Supervisor agent
        """
        if len(agent_policy) > 1:
            supervisor = agent_policy[0]
            agents = agent_policy[1:]
            if mode == "ON":
                self.multi_agent_on_policy_run(supervisor, agents, env)
            else:
                self.multi_agent_off_policy_run(supervisor, agents, env)
        elif len(agent_policy) == 1:
            agent = agent_policy[0]
            if mode == "ON":
                self.single_agent_on_policy_run(agent, env)
            else:
                self.single_agent_off_policy_run(agent, env)
        else:
            print("Invalid agent Input.")

    def single_agent_on_policy_run(self, agent, env):
        print(">>> Start Running Single Agent Off Policy Loop <<<<")
        trainer = None
        if agent.get('agent_type').lower() == "asking":
            trainer = AskingTrainer(
                agent=agent,
                accelerator=self.accelerator,
                tokenizer=self.tokenizer,
                critic_lr=self.critic_lr,
                lm_lr=self.lm_lr,
                gamma=self.gamma,
                tau=self.tau,
                epochs=self.epochs,
                actor_epochs=self.actor_epochs,
                grad_accum_steps=self.grad_accum_steps,
                max_grad_norm=self.max_grad_norm
            )

        all_trajectories = []
        if not trainer:
            return

        replay_buffer = None
        if self.accelerator.is_main_process:
            if os.path.exists(os.path.join(self.save_path, 'trainer.pt')):
                print(">>>>> Loading from checkpoint <<<")
                trainer.load(os.path.join(self.save_path, 'trainer.pt'))
                all_trajectories = torch.load(os.path.join(self.save_path, 'trajectories.pt'))
                replay_buffer = torch.load(os.path.join(self.save_path, 'replay_buffer.pt'))
            else:
                print("Creating new checkpoint directory")
                os.makedirs(self.save_path, exist_ok=True)

        agent.prepare()
        # main training loop
        print(">>>start iterations")
        for i in tqdm(range(self.iterations)):
            print(">>>Interacting with Environment")
            if self.accelerator.is_main_process:
                trajectories = batch_interact_environment(
                    agent=agent,
                    env=env,
                    num_trajectories=self.rollout_size,
                    env_idx=self.env_idx,
                    use_tqdm=False,
                    decode_f=self.decode_f
                )
                info = {"rollout.mean": np.mean([d[0]["trajectory_reward"] for d in trajectories]),
                        "rollout.max": np.max([d[0]["trajectory_reward"] for d in trajectories]),
                        "rollout.min": np.min([d[0]["trajectory_reward"] for d in trajectories])}

                all_trajectories += trajectories
                data = sum(trajectories, [])
                for t in data:
                    replay_buffer.insert(**t)
                info.update({"rollout.reward.mean": np.mean([d["reward"] for d in data]),
                             "rollout.reward.max": np.max([d["reward"] for d in data]),
                             "rollout.reward.min": np.min([d["reward"] for d in data])})
                print(">>> Saving Replay Buffer")
                torch.save(replay_buffer, os.path.join(self.save_path, 'replay_buffer.pt'))
                torch.save(all_trajectories, os.path.join(self.save_path, 'trajectories.pt'))
                print(">>> Saved Replay Buffer")
                time.sleep(15)
            else:
                info = {}

            self.accelerator.wait_for_everyone()
            all_trajectories = torch.load(os.path.join(self.save_path, 'trajectories.pt'))
            replay_buffer = torch.load(os.path.join(self.save_path, 'replay_buffer.pt'))

            print(">>>>>>>>>>>>>>> Update Policy <<<<<<<<<<<<<<<<<<")
            # update TD error, Critic, Actor
            info.update(trainer.update(replay_buffer, no_update_actor=True))

            if self.use_wandb and self.accelerator.is_main_process:
                wandb.log(info)

            if (i + 1) % self.save_freq == 0 and self.save_path is not None and self.accelerator.is_main_process:
                print("Saving")
                trainer.save(os.path.join(self.save_path, 'trainer.pt'))
                torch.save(replay_buffer, os.path.join(self.save_path, 'replay_buffer.pt'))

    def single_agent_off_policy_run(self, agent, env):
        print(">>> Start Running Single Agent Off Policy Loop <<<<")
        trainer = None
        if agent.get('agent_type').lower() == "asking":
            trainer = AskingTrainer(
                agent=agent,
                accelerator=self.accelerator,
                tokenizer=self.tokenizer,
                critic_lr=self.critic_lr,
                lm_lr=self.lm_lr,
                gamma=self.gamma,
                tau=self.tau,
                epochs=self.epochs,
                actor_epochs=self.actor_epochs,
                grad_accum_steps=self.grad_accum_steps,
                max_grad_norm=self.max_grad_norm
            )

        all_trajectories = []
        if not trainer:
            return

        replay_buffer = None
        if self.accelerator.is_main_process:
            if os.path.exists(os.path.join(self.save_path, 'trainer.pt')):
                print(">>>>> Loading from checkpoint <<<")
                trainer.load(os.path.join(self.save_path, 'trainer.pt'))
                all_trajectories = torch.load(os.path.join(self.save_path, 'trajectories.pt'))
                replay_buffer = torch.load(os.path.join(self.save_path, 'replay_buffer.pt'))
            else:
                print("Creating new checkpoint directory")
                os.makedirs(self.save_path, exist_ok=True)

        agent.prepare()
        # main training loop
        print(">>>start iterations")
        for i in tqdm(range(self.iterations)):
            print(">>>Interacting with Environment")
            if self.accelerator.is_main_process:
                trajectories = batch_interact_environment(
                    agent=agent,
                    env=env,
                    num_trajectories=self.rollout_size,
                    env_idx=self.env_idx,
                    use_tqdm=False,
                    decode_f=self.decode_f
                )
                info = {"rollout.mean": np.mean([d[0]["trajectory_reward"] for d in trajectories]),
                        "rollout.max": np.max([d[0]["trajectory_reward"] for d in trajectories]),
                        "rollout.min": np.min([d[0]["trajectory_reward"] for d in trajectories])}

                if (i + 1) % self.eval_freq == 0:
                    old_sample = agent.do_sample
                    agent.do_sample = False
                    eval_trajectories = batch_interact_environment(
                        agent=agent,
                        env=self.eval_env,
                        num_trajectories=max(self.eval_size, self.eval_env.bsize),
                        env_idx=self.env_idx,
                        use_tqdm=False,
                        decode_f=self.decode_f
                    )

                    agent.do_sample = old_sample
                    info.update({"eval_rollout.mean": np.mean([d[0]["trajectory_reward"] for d in eval_trajectories]),
                                 "eval_rollout.max": np.max([d[0]["trajectory_reward"] for d in eval_trajectories]),
                                 "eval_rollout.min": np.min([d[0]["trajectory_reward"] for d in eval_trajectories]), })
                all_trajectories += trajectories
                data = sum(trajectories, [])
                for t in data:
                    replay_buffer.insert(**t)
                info.update({"rollout.reward.mean": np.mean([d["reward"] for d in data]),
                             "rollout.reward.max": np.max([d["reward"] for d in data]),
                             "rollout.reward.min": np.min([d["reward"] for d in data])})
                print(">>> Saving Replay Buffer")
                torch.save(replay_buffer, os.path.join(self.save_path, 'replay_buffer.pt'))
                torch.save(all_trajectories, os.path.join(self.save_path, 'trajectories.pt'))
                print(">>> Saved Replay Buffer")
                time.sleep(15)
            else:
                info = {}

            self.accelerator.wait_for_everyone()
            all_trajectories = torch.load(os.path.join(self.save_path, 'trajectories.pt'))
            replay_buffer = torch.load(os.path.join(self.save_path, 'replay_buffer.pt'))

            print("=========>>>>>> Training <<<<<<<<========")
            info.update(trainer.update(replay_buffer, no_update_actor=(i < self.warmup_iter)))

            if self.use_wandb and self.accelerator.is_main_process:
                wandb.log(info)

            if (i + 1) % self.save_freq == 0 and self.save_path is not None and self.accelerator.is_main_process:
                print("Saving")
                trainer.save(os.path.join(self.save_path, 'trainer.pt'))
                torch.save(replay_buffer, os.path.join(self.save_path, 'replay_buffer.pt'))

    def multi_agent_on_policy_run(self, supervisor, agents, env):
        pass

    def multi_agent_off_policy_run(self, supervisor, agents, env):
        pass


def train_loop(
        env,
        eval_env,
        agent,
        tokenizer,
        accelerator,
        warmup_iter: int = 20,
        rollout_size: int = 50,
        eval_size: int = 1,
        batch_size: int = 2,
        capacity: int = 500000,
        iterations: int = 10,
        epochs: int = 3,
        grad_accum_steps: int = 1,
        env_idx: int = None,
        do_sample: bool = False,
        temperature: float = 2.0,
        critic_lr: float = 1e-4,
        lm_lr: float = 1e-5,
        gamma: float = 0.9,
        tau: float = 0.1,
        use_wandb: bool = False,
        env_load_path: str = '',
        actor_epochs: int = 3,
        max_grad_norm: float = 0.01,
        save_path: str = None,
        save_freq: int = 25,
        eval_freq: int = 25,
        agent_type: str = "asking",
        decode_f: callable = lambda x: x,
        **kwargs
):
    print(">>> Start running Loop <<<<")
    trainer = None
    if agent_type.lower() == "asking":
        trainer = AskingTrainer(
            agent=agent,
            accelerator=accelerator,
            tokenizer=tokenizer,
            critic_lr=critic_lr,
            lm_lr=lm_lr,
            gamma=gamma,
            tau=tau,
            epochs=epochs,
            actor_epochs=actor_epochs,
            grad_accum_steps=grad_accum_steps,
            max_grad_norm=max_grad_norm
        )

    all_trajectories = []
    if not trainer:
        return

    replay_buffer = None
    if accelerator.is_main_process:
        if os.path.exists(os.path.join(save_path, 'trainer.pt')):
            print(">>>>> Loading from checkpoint <<<")
            trainer.load(os.path.join(save_path, 'trainer.pt'))
            all_trajectories = torch.load(os.path.join(save_path, 'trajectories.pt'))
            replay_buffer = torch.load(os.path.join(save_path, 'replay_buffer.pt'))
        else:
            print("Creating new checkpoint directory")
            os.makedirs(save_path, exist_ok=True)

    agent.prepare()
    # main training loop
    print(">>>start iterations")
    for i in tqdm(range(iterations)):
        print(">>>Interacting with Environment")
        if accelerator.is_main_process:
            trajectories = batch_interact_environment(
                agent=agent,
                env=env,
                num_trajectories=rollout_size,
                env_idx=env_idx,
                use_tqdm=False,
                decode_f=decode_f
            )
            info = {"rollout.mean": np.mean([d[0]["trajectory_reward"] for d in trajectories]),
                    "rollout.max": np.max([d[0]["trajectory_reward"] for d in trajectories]),
                    "rollout.min": np.min([d[0]["trajectory_reward"] for d in trajectories])}

            if (i + 1) % eval_freq == 0:
                old_sample = agent.do_sample
                agent.do_sample = False
                eval_trajectories = batch_interact_environment(
                    agent=agent,
                    env=eval_env,
                    num_trajectories=max(eval_size, eval_env.bsize),
                    env_idx=env_idx,
                    use_tqdm=False,
                    decode_f=decode_f
                )

                agent.do_sample = old_sample
                info.update({"eval_rollout.mean": np.mean([d[0]["trajectory_reward"] for d in eval_trajectories]),
                             "eval_rollout.max": np.max([d[0]["trajectory_reward"] for d in eval_trajectories]),
                             "eval_rollout.min": np.min([d[0]["trajectory_reward"] for d in eval_trajectories]), })
            all_trajectories += trajectories
            data = sum(trajectories, [])
            for t in data:
                replay_buffer.insert(**t)
            info.update({"rollout.reward.mean": np.mean([d["reward"] for d in data]),
                         "rollout.reward.max": np.max([d["reward"] for d in data]),
                         "rollout.reward.min": np.min([d["reward"] for d in data])})
            print(">>> Saving Replay Buffer")
            torch.save(replay_buffer, os.path.join(save_path, 'replay_buffer.pt'))
            torch.save(all_trajectories, os.path.join(save_path, 'trajectories.pt'))
            print(">>> Saved Replay Buffer")
            time.sleep(15)
        else:
            info = {}

        accelerator.wait_for_everyone()
        all_trajectories = torch.load(os.path.join(save_path, 'trajectories.pt'))
        replay_buffer = torch.load(os.path.join(save_path, 'replay_buffer.pt'))

        print("=========>>>>>> Training <<<<<<<<========")
        info.update(trainer.update(replay_buffer, no_update_actor=(i < warmup_iter)))

        if use_wandb and accelerator.is_main_process:
            wandb.log(info)

        if (i + 1) % save_freq == 0 and save_path is not None and accelerator.is_main_process:
            print("Saving")
            trainer.save(os.path.join(save_path, 'trainer.pt'))
            torch.save(replay_buffer, os.path.join(save_path, 'replay_buffer.pt'))

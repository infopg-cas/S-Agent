from src.algorithms import train_loop
from src.llms.llevel.finetuning.AskingAgent import AskingAgent
from src.agent.environment import BatchHotpotEnv
import wandb
import torch
import transformers
from accelerate import Accelerator
from datetime import timedelta
from accelerate import InitProcessGroupKwargs

transformers.logging.set_verbosity_error()

CONFIG_NAME = "AskingAgent"


def main(config):
    print(">>> Configuration file: " + CONFIG_NAME + "<<<")
    try:
        from huggingface_hub import login
        login(token=config['huggingface_token'])
    except:
        print(">>> Huggingface token not found.")

    accelerator = Accelerator(InitProcessGroupKwargs(timeout=timedelta(18000)))
    device = accelerator.device

    # load environment
    if config['env_name'] == "hotpot":
        print(">>> Using Hotpot env")
        env = BatchHotpotEnv(
            env_load_path=config['env_load_path'],
            device=device,
            cache_dir=config['cache_dir']
        )
        eval_env = env
    else:
        raise NotImplementedError("Environment not implemented.")

    decode_f = lambda x: x

    if config['agent_type'].lower() == "asking":
        print(">>> Using Asking agent")
        agent = AskingAgent(
            device=device,
            accelerator=accelerator,
            temperature=config['temperature'],
            do_sample=config['do_sample'],
            policy_lm=config['policy_lm'],
            # critic_lm=config['critic_lm'],
            cache_dir=config['cache_dir'],
            max_new_tokens=config['max_new_tokens'],
            eos_str='\n')
    else:
        raise NotImplementedError("Agent not implemented.")

    if config['checkpoint_path'] is not None:
        state_dict = torch.load(config['checkpoint_path'], map_location=device)['model_state_dict']

    if config['use_wandb'] and accelerator.is_main_process:
        wandb.login(key=config.wandb_key)
        wandb.init(project=config.project_name, name=config.run_name, config=dict(config))

    train_loop(
        env=env,
        agent=agent,
        tokenizer=agent.tokenizer,
        eval_env=eval_env,
        accelerator=accelerator,
        decode_f=decode_f,
        **config)


if __name__ == "__main__":
    from Tokens import HUGGING_FACE

    config = {
        "huggingface_token": HUGGING_FACE,
        "env_name": "hotpot",
        "agent_type": "asking",
        "use_wandb": False,
        "checkpoint_path": None,
        "env_load_path": "~/.model",
        "cache_dir": "/Users/zhilinhe/Desktop/hhhhzl/WorkGetBetter/AI-agent/S-Agent/cache",
        "policy_lm": 'gpt2',
        'temperature': 0.05,
        'do_sample': True,
        'critic_lm': None,
        'max_new_tokens': 1024,
        'use_lora': True,
        'save_path': "/Users/zhilinhe/Desktop/hhhhzl/WorkGetBetter/AI-agent/S-Agent/save"
    }
    main(config)

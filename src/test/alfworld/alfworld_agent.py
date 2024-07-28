import pprint
from src.test.hotpot.configs import REDIS_SETTINGS
from src.utils.redis_tools import RedisWrapper
from src.agent.agents.general import GeneralAgent, GeneralAgentGroup, GroupAgentTree
from src.agent.agents.alfworld_agent import AlfworldAgent
from src.test.hotpot.configs import HOTPOT_TOOLS, HOTPOT_PROMPT, HOTPOT_LLM
import alfworld
import alfworld.agents.environment
import yaml

def create_group(group_name, description):
    hotpot_group = GeneralAgentGroup(
        group_name=group_name,
        description=description
    )
    return hotpot_group


def create_agent(agent_name, description, llm, prompt, tools, env, config):
    h_agent = AlfworldAgent(
        agent_name=agent_name,
        agent_description=description,
        llm=llm,
        actions=tools,
        template=prompt,
        cache=RedisWrapper(REDIS_SETTINGS=REDIS_SETTINGS, setting_name='tasks'),
        env=env,
        config=config
    )
    return h_agent


def single_agent(env, config):
    team = GroupAgentTree()
    group_name = 'Alfworld Task Group'
    agent_name = 'Alfworld Agent'

    # 2. define the group
    hotpot_group = create_group(group_name, 'A group of agents to solve the Alfworld Tasks.')

    # 3. create an agent and add to team
    h_agent = create_agent(agent_name, "Expert Agent in Alfworld Tasks.", HOTPOT_LLM, HOTPOT_PROMPT, {}, env=env, config=config)
    team.stray_agents[agent_name] = h_agent

    # 4. add group to the team
    team.add_root({"group_name": group_name, "metadata": hotpot_group})

    # 5. add agent to the group
    res, msg = team.add_agent_to_group(
        agent=h_agent,
        group_name=group_name
    )
    if not res:
        return False, msg, None

    return True, "OK.", team

if __name__ == "__main__":
    import random
    with open('base_config.yaml') as reader:
        config = yaml.safe_load(reader)

    split = "eval_out_of_distribution"
    env = alfworld.agents.environment.AlfredTWEnv(config, train_eval=split)
    env = env.init_env(batch_size=1)

    NUM_GAMEFILES = len(env.gamefiles)
    for n in range(1):
        random.seed(config["general"]["random_seed"])
        ob, info = env.reset()
        ob = "\n".join(ob[0].split("\n\n")[1:])
        scene_observation, task_description = ob.split("\n")
        name = "/".join(info["extra.gamefile"][0].split("/")[-3:-1])

        res, msg, team = single_agent(env, config)
        agent_name = 'Alfworld Agent'
        group_name = 'Alfworld Task Group'
        if res:
            group = team.find_node("group_name", group_name).metadata
            if group:
                agent = group.agent_pools[agent_name]
                agent.run_agent(scene_observation, task_description, name)




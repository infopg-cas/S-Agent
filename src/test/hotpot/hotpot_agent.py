import pprint
from src.test.hotpot.configs import REDIS_SETTINGS
from src.utils.redis_tools import RedisWrapper
from src.agent.agents.general import GeneralAgent, GeneralAgentGroup, GroupAgentTree
from src.agent.agents.hotpot_agent import HotpotAgent
from src.test.hotpot.configs import HOTPOT_TOOLS, HOTPOT_PROMPT, HOTPOT_LLM


def create_group(group_name, description):
    hotpot_group = GeneralAgentGroup(
        group_name=group_name,
        description=description
    )
    return hotpot_group


def create_agent(agent_name, description, llm, prompt, tools):
    h_agent = HotpotAgent(
        agent_name=agent_name,
        agent_description=description,
        llm=llm,
        actions=tools,
        template=prompt,
        cache=RedisWrapper(REDIS_SETTINGS=REDIS_SETTINGS, setting_name='tasks')
    )
    return h_agent


def single_agent():
    team = GroupAgentTree()
    group_name = 'Hotpot Q&A'
    agent_name = 'Hotpot Agent'

    # 2. define the group
    hotpot_group = create_group(group_name, 'A group of agents to solve the hotpot Q&A questions.')

    # 3. create an agent and add to team

    h_agent = create_agent(agent_name, "Expert Agent in Hotpot Q&A question.", HOTPOT_LLM, HOTPOT_PROMPT, HOTPOT_TOOLS)
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

    return True,"OK.", team




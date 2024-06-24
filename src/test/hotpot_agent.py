from src.agent.agents.general import GeneralAgent, GeneralAgentGroup, GroupAgentTree
from src.llms.hlevel import OpenAiLLM
from src.agent.agents.hotpot_agent import HotpotAgent
from src.agent.tools.base import Tool
from src.agent.tools.wiki_tool import search_wiki, lookup
import json

if __name__ == "__main__":
    from Tokens import OPEN_KEY

    # 1. define the team tree
    team = GroupAgentTree()
    group_name = 'Hotpot Q&A'
    agent_name = 'Hotpot Agent'

    # 2. define the group
    hotpot_group = GeneralAgentGroup(
        group_name=group_name,
        description='A group of agents to solve the hotpot Q&A questions.'
    )

    # "2. Name: 'lookup', which returns the next sentence containing keyword in the current passage.\n" \

    # 3. create an agent and add to team
    prompt = "You are a Hotpot Q&A question agent, you job is to:\n" \
             "1. Answer the question and provide the correct answer. \n" \
             "Follow the guidance by humans.\n" \
             "You have a tool library\n:" \
             "1. Name: 'search_wiki', which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.\n" \
             "2. Name: 'lookup', which returns the next sentence containing keyword in the current passage.\n"\
             "Restrictions\n: " \
             "1. Do not call actions that not defined in the tool library.\n" \
             "2. You have to response short but clean.\n" \
             "Follow this to use a tool: create_group => create_agent.\n"

    h_agent = HotpotAgent(
        agent_name=agent_name,
        agent_description="Expert Agent in Hotpot Q&A question.",
        llm=OpenAiLLM(api_key=OPEN_KEY),
        actions={
            "search_wiki": Tool(
                name='search_wiki',
                description='searches the exact entity on Wikipedia returns the first paragraph if it exists. If not, it will return some similar entities to search.',
                func= search_wiki
            ),
            "lookup": Tool(
                name='loopup',
                description='returns the next sentence containing keyword in the current passage.',
                func=lookup
            )
        },
        template=prompt
    )

    team.stray_agents[agent_name] = h_agent

    # 4. add group to the team
    team.add_root({"group_name": group_name, "metadata": hotpot_group})

    # 5. add agent to the group
    res, msg = team.add_agent_to_group(
        agent=h_agent,
        group_name=group_name
    )
    print(msg)
    h_agent.run_agent("Musician and satirist Allie Goertz wrote a song about the 'The Simpsons' character Milhouse, who Matt Groening named after who?")

from src.agent.agents.general import GeneralAgent, GeneralAgentGroup, GroupAgentTree
from typing import Tuple
import re
from src.agent.planning import AskIsWhatALlYouNeed
from src.llms.hlevel import OpenAiLLM
from src.agent.tools.base import Tool
import json


class PlanningAgent(GeneralAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.planning_stra = AskIsWhatALlYouNeed(self)
        self.planning_graph = self.planning_stra.get_planning_graph()

    def get_nodes_args(self, pointer, *args, **kwargs):
        def memory_args(*args, **kwargs):
            memory = kwargs.get('memory')
            plan_record = kwargs.get('plan_record')
            return (memory.get('l'), memory.get('s'), plan_record["memory"] + 1)

        def belief_args(*args, **kwargs):
            plan_record = kwargs.get('plan_record')
            return (self.perception_env(), plan_record["belief"] + 1)

        def action_args(*args, **kwargs):
            plan_record = kwargs.get('plan_record')
            tool_name = kwargs.get('tool_name')
            return (self.actions[tool_name], plan_record["action"] + 1)

        def ask_args(*args, **kwargs):
            plan_record = kwargs.get('plan_record')
            return (plan_record["ask"] + 1,)

        POINTER_CONFIG = {
            "memory": memory_args,
            "belief": belief_args,
            "ask": ask_args,
            "action": action_args
        }

        if pointer in POINTER_CONFIG:
            return POINTER_CONFIG[pointer](*args, **kwargs)
        else:
            plan_record = kwargs.get('plan_record')
            return (plan_record[pointer] + 1,)

    def append_message(self, role, msg):
        if role in ["user", "system", "assistant"]:
            self.messages.append({"role": role, "content": msg})
        else:
            raise "No this type of role"

    def recall_memory(self):
        pass

    def run_agent(self, query):
        """
        a question as input
        """
        try:
            tool_name = None
            self.append_message('system', self.prompt_template + "Question: " + query + '\n')
            n_calls, n_bad_calls = 0, 0
            plan_record = {}
            for key in self.planning_graph:
                plan_record[key] = 0

            pointer = 'memory'
            while self.planning_graph[pointer] != 'SINK' and max(plan_record.values()) < 8 and n_bad_calls < 10:
                func = getattr(self.planning_stra, pointer)
                args = self.get_nodes_args(pointer, plan_record=plan_record, memory=self.memory, tool_name=tool_name)
                res, response = func(*args)

                if not res:
                    n_bad_calls += 1
                    continue

                # call back => for actions
                if pointer == 'action':
                    res, payload = self.actions[tool_name].func(**response)
                    if not res:
                        continue

                    self.append_message('user', str(payload))

                if pointer not in plan_record:
                    plan_record[pointer] = 1
                else:
                    plan_record[pointer] += 1

                # detach
                if type(self.planning_graph[pointer]) == list and len(self.planning_graph[pointer]) > 1:
                    for func, condition in self.planning_graph[pointer]:
                        if condition in response:
                            pointer = func
                            if pointer == 'action':
                                import re
                                import json
                                tool_name = json.loads(re.search(r'\{.*\}', response).group()).get('tool_name', '')
                            break
                else:
                    pointer, condition = self.planning_graph[pointer][0]

            if self.planning_graph[pointer] == 'SINK':
                func = getattr(self.planning_stra, pointer)
                res, response = func(query)
                pprint.pprint(self.trajectory)
            elif max(plan_record.values()) >= 8:
                print("max iterations")
                pprint.pprint(self.trajectory)
            elif n_bad_calls >= 10:
                print("max number of bad calls")
                pprint.pprint(self.trajectory)
        except:
            pprint.pprint(self.trajectory)


if __name__ == "__main__":
    from Tokens import OPEN_KEY

    # 1. define the team tree
    team = GroupAgentTree()

    # 2. define the group
    planning_group = GeneralAgentGroup(
        group_name='Meta Group',
        description='An Administrative group that has the highest decision level and manage the Worker Group.'
    )

    # 3. create an agent and add to team
    prompt = "You are a Planning agent, you job is to:\n" \
             "1. Split the task description query (input) to several subtasks. \n" \
             "2. Create a new agent for this subtask, and define the flow between agents. \n" \
             "3. Add the agent that created to the team.\n" \
             "Follow the guidance by humans.\n" \
             "You have a tool library\n:" \
             "1. Name: 'create_group', which create a new agent group.\n" \
             "2. Name: 'create_agent', which create an new agent.\n" \
             "Restrictions\n: " \
             "1. Do not call actions that not defined in the tool library.\n" \
             "2. You have to response short but clean.\n" \
             "Follow this to use a tool: create_group => create_agent.\n"
    p_agent = PlanningAgent(
        agent_name='planning agent',
        agent_description="Expert Agent in planning the task & subtasks for the team.",
        llm=OpenAiLLM(api_key=OPEN_KEY),
        actions={
            "create_group": Tool(
                name='create_group_tool',
                description='Create a new group in the team.',
                func=team.create_group
            ),
            "create_agent": Tool(
                name='create_agent_tool',
                description='Create a agent in the group.',
                func=team.create_agent
            ),
            "add_agent_to_group": Tool(
                name='add_agent_to_group',
                description='Add a agent to a group.',
                func=team.add_agent_to_group
            ),
            "add_group_to_group": Tool(
                name='add_group_to_group tool',
                description='Add a group to a group as subgroup.',
                func=team.add_group_to_group
            )
        },
        template=prompt
    )
    team.stray_agents['planning agent'] = p_agent

    # 4. add group to the team
    team.add_root({"group_name": 'Meta Group', "metadata": planning_group})

    # 5. add agent to the group
    res, msg = team.add_agent_to_group(
        agent=p_agent,
        group_name='Meta Group'
    )
    print(msg)
    import pprint

    # team.mac_env.get_group_info('meta group', sub_team)

    # print(planning_group.agent_organ_graph)

    # print(p_agent.perception_env())
    p_agent.run_agent(
        "Can you help me to grade the subjective questionnaire, I have all my questionnaire in my database?")

    sub_team = team.find_node_by_attribute(team.roots, 'group_name', 'Meta Group')
    print(team.mac_env.get_group_info('Meta Group', sub_team))

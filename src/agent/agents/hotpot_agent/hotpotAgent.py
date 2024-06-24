from src.agent.agents.general import GeneralAgent, GeneralAgentGroup, GroupAgentTree
from typing import Tuple
import re
from src.agent.planning import AskIsWhatALlYouNeed
import pprint


class HotpotAgent(GeneralAgent):
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
            while self.planning_graph[pointer] != 'SINK' and max(plan_record.values()) < 20 and n_bad_calls < 10:
                func = getattr(self.planning_stra, pointer)
                args = self.get_nodes_args(pointer, plan_record=plan_record, memory=self.memory, tool_name=tool_name)
                res, response = func(*args)
                print(response)

                if not res:
                    n_bad_calls += 1
                    continue

                # call back => for actions
                if pointer == 'action':
                    res, payload = self.actions[tool_name].func(**response)
                    if not res:
                        continue
                    self.append_message('user', str(payload))

                if type(response) == str and response[:3].lower() == 'ask':
                    question = response.split(":")[-1]
                    print(question)
                    human_input = input("Provide your Response/Guidence: ")
                    self.append_message('user', str(human_input))


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
        except Exception as e:
            print(str(e))
            pprint.pprint(self.trajectory)
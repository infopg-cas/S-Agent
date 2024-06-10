from src.agent.agents.general import GeneralAgent
from typing import Tuple
import re
from src.agent.planning import AskIsWhatALlYouNeed
from src.llms.hlevel import OpenAiLLM


class PlanningAgent(GeneralAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.planning_stra = AskIsWhatALlYouNeed(self)
        self.planning_graph = self.planning_stra.get_planning_graph()

    def get_nodes_args(self, pointer, *args, **kwargs):
        def memory_args(*args, **kwargs):
            memory = kwargs.get('memory')
            plan_record = kwargs.get('plan_record')
            return (memory['l'], memory['s'], plan_record["memory"] + 1)

        def belief_args(*args, **kwargs):
            plan_record = kwargs.get('plan_record')
            return ("", "", plan_record["belief"] + 1)

        def ask_args(*args, **kwargs):
            plan_record = kwargs.get('plan_record')
            return (plan_record["ask"] + 1,)

        POINTER_CONFIG = {
            "memory": memory_args,
            "belief": belief_args,
            "ask": ask_args,
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
        self.append_message('system', self.prompt_template)
        n_calls, n_bad_calls = 0, 0
        plan_record = {}
        for key in self.planning_graph:
            plan_record[key] = 0

        pointer = 'SOURCE'
        while self.planning_graph[pointer] != 'SINK' and max(plan_record.values()) < 8 and n_bad_calls < 10:
            func = getattr(self.planning_stra, pointer)
            args = self.get_nodes_args(pointer, plan_record=plan_record, memory=self.memory)
            res, response = func(*args)

            if not res:
                n_bad_calls += 1
                continue

            plan_record[pointer] += 1

            # detach
            if type(self.planning_graph[pointer]) == list and len(self.planning_graph[pointer] > 1):
                for func, condition in self.planning_graph[pointer]:
                    if condition in response:
                        pointer = func

        if self.planning_graph[pointer] == 'SINK':
            print("finish")
        elif max(plan_record.values()) >= 8:
            print("max iterations")
        elif n_bad_calls >= 10:
            print("max number of bad calls")

            # self.recall_memory()
            # res, response = self.planning_stra.memory(self.memory['long memory'], self.memory['short memory'], 0)
            #
            # if not res:
            #     return
            # self.prompt_template += response + "Question: " + query + '\n'
            # self.append_message('user', self.prompt_template)
            # for i in range(1, 8):
            #     res, response = self.planning_stra.belief(team_info="", team_detail="", iteration=i)
            #     if not res:
            #         return
            #
            #     res, response = self.planning_stra.think(i)
            #     if not res:
            #         return
            #
            #     if "ask" in response.lower():
            #         res, response = self.planning_stra.ask(i)
            #         # wait for other
            #     else:
            #         res, response = self.planning_stra.action(i)
            #
            #     res, response = self.planning_stra.observation(i)
            #     if not res:
            #         return
            #
            #     res, response = self.planning_stra.reflection(i)
            #     if not res:
            #         return
            #
            #     if "do" in response.lower():
            #         continue
            #     elif "finish" in response.lower():
            #         res, response = self.planning_stra.finish()
            #         break


if __name__ == "__main__":
    from Tokens import OPEN_KEY

    prompt = "You are a Planning agent, you job is to:\n" \
             "1. Split the task description query (input) to several subtasks. \n" \
             "2. Create a new agent for this subtask, and define the flow between agents. \n" \
             "3. Add the agent that created to the team.\n" \
             "I want to follow the guidance by humans.\n" \
             "You have a tool library\n:" \
             "1. Name: 'create_group', which create a new agent group.\n" \
             "2. Name: 'create_agent', which create an new agent.\n" \
             "Restrictions\n: " \
             "1. Do not call actions that not defined in the tool library.\n" \

    agent = PlanningAgent(
        agent_name='reAct',
        llm=OpenAiLLM(api_key=OPEN_KEY),
        actions={
            "create_group": None,
            "create_agent": None,
        },
        template=prompt
    )
    agent.run_agent("What's the weather today?")

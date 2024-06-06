from src.agent.agents.general import GeneralAgent
from typing import Tuple
import re
from src.agent.planning import AskIsWhatALlYouNeed


class PlanningAgent(GeneralAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.planning_stra = AskIsWhatALlYouNeed(self)
        self.planning_graph = self.planning_stra.get_planning_graph()

    def append_message(self, role, msg):
        if role in ["user", "system", "agent"]:
            self.messages.append({"role": role, "content": msg})
        else:
            raise "No this type of role"

    def run_agent(self, query):
        """
        a question as input
        """
        n_calls, n_bad_calls = 0, 0
        plan_record = {}
        for key in self.planning_graph:
            plan_record[key] = 0

        pointer = 'SOURCE'
        while self.planning_graph[pointer] != 'SINK' and max(plan_record.values()) < 8 and n_bad_calls < 10:
            func = getattr(self.planning_stra, pointer)

            if pointer == "memory":
                res, response = func(self.memory['l'], self.memory['s'], plan_record[pointer] + 1)
            elif pointer == "belief":
                res, response = func("", "", plan_record[pointer] + 1)
            else:
                if pointer == "ask":
                    res, response = func(plan_record[pointer] + 1)
                    # wait
                else:
                    res, response = func(plan_record[pointer] + 1)

            if not res:
                n_bad_calls += 1
                return

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




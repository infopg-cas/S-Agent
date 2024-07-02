from src.agent.agents.general import GeneralAgent, GeneralAgentGroup, GroupAgentTree
from typing import Tuple, Dict
from src.agent.planning import AskIsWhatALlYouNeed
import pprint
from src.eval.eval_plannings import EvaluatePlanning
import re
import json


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
        n_calls, n_bad_calls = 0, 0
        plan_record = {}
        try:
            tool_name = None
            self.append_message('system', self.prompt_template + "Question: " + query + '\n')

            for key in self.planning_graph.keys():
                plan_record[key] = 0
            pointer = 'memory'
            while self.planning_graph[pointer] != 'SINK' and max(plan_record.values()) < 50 and n_bad_calls < 15:
                func = getattr(self.planning_stra, pointer)
                args = self.get_nodes_args(pointer, plan_record=plan_record, memory=self.memory, tool_name=tool_name)
                res, response = func(*args)

                if not res:
                    n_bad_calls += 1
                    continue

                n_calls += 1

                # call back => for actions
                if pointer == 'action':
                    try:
                        res, payload = self.actions[tool_name].func(**response)
                        if not res:
                            continue
                        self.append_message('user', str(payload))
                    except:
                        continue

                if type(response) == str and response[:3].lower() == 'ask':
                    question = response.split(":")[-1]
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
                                try:
                                    import re
                                    import json
                                    tool_name = json.loads(re.search(r'\{.*\}', response).group()).get('tool_name', '')
                                except Exception as e:
                                    print(109, 'Tools failed', str(e))
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
            pprint.pprint(self.trajectory)

    def process_task(
            self,
            task: Dict,
            is_update=True
    ):
        status = task.get("status", None)
        question = task['task'][0]
        if status is None:
            return False, "NOT CORRECT PROCESS TYPE", "Finished"

        self.messages = task.get("messages", [])
        self.trajectory = task.get("traj", [])
        plan_record = task.get("planning_status", {})
        pointer = task['pointer']
        n_calls, n_bad_calls = 0, 0

        if status == 0:
            task['status'] = 1
            self.append_message('system', self.prompt_template + "Question: " + question + '\n')
            for key in self.planning_graph.keys():
                plan_record[key] = 0
        else:
            n_calls, n_bad_calls = task['n_calls'], task['n_bad_calls']
        try:
            tool_name = None
            while self.planning_graph[pointer] != 'SINK' and max(plan_record.values()) < 50 and n_bad_calls < 15:
                func = getattr(self.planning_stra, pointer)
                args = self.get_nodes_args(pointer, plan_record=plan_record, memory=self.memory, tool_name=tool_name)
                res, response = func(*args)
                print(160, response)

                if not res:
                    n_bad_calls += 1
                    self.messages = self.messages[0:-1]
                    continue
                n_calls += 1

                # call back => for actions
                if pointer == 'action':
                    try:
                        res, payload = self.actions[tool_name].func(**response)
                        if not res:
                            continue
                        self.append_message('user', str(payload))
                    except:
                        continue

                if type(response) == str and response[:3].lower() == 'ask':
                    question = response.split(":")[-1]
                    pointer, condition = self.planning_graph[pointer][0]
                    task['step'] += 1
                    task['msg_status'] = 1
                    task['pointer'] = pointer
                    task['planning_status'] = plan_record
                    break

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
                                tool_name = json.loads(re.search(r'\{.*\}', response).group()).get('tool_name', '')
                            break

                else:
                    pointer, condition = self.planning_graph[pointer][0]

                task['step'] += 1
                task['pointer'] = pointer
                task['planning_status'] = plan_record
                task['n_calls'] = n_calls
                task['n_bad_calls'] = n_bad_calls
                if is_update:
                    self.update_state(task)

            task['pointer'] = pointer
            task['planning_status'] = plan_record
            task['n_calls'] = n_calls
            task['n_bad_calls'] = n_bad_calls

            if self.planning_graph[pointer] == 'SINK':
                func = getattr(self.planning_stra, pointer)
                res, response = func(question)
                if res:
                    task['answer'], task['status'], task['done'] = response, 2, True
                    task['step'] += 1
                    task['metrics'] = EvaluatePlanning().get_metrics(task)
                    task['reward'] = EvaluatePlanning().get_reward(task)
            elif max(plan_record.values()) >= 50:
                task['error'], task['status'] = "max iterations", 3
            elif n_bad_calls >= 15:
                task['error'], task['status'] = "max number of bad calls", 3

            if is_update:
                self.update_state(task)

        except Exception as e:
            task['step'] += 1
            task['error'] = str(e)
            task['status'] = 3
            task['pointer'] = pointer
            task['planning_status'] = plan_record
            if is_update:
                self.update_state(task)
        finally:
            if task['msg_status'] == 1:
                print("Wait for other users to response.")
            print(f"Finish process task {task['id']}, {'Done' if task['done'] else 'Not Finished. '}{'Error: ' + task['error'] if task['error'] else ''}")
            # pprint.pprint(task)

    def update_state(self, task):
        try:
            task['traj'] = self.trajectory
            task['messages'] = self.messages

            if task['msg_status'] == 1:
                self.cache.list_push(f"human_process", *[task], side='l')
            elif task['done']:
                self.cache.list_push(f"done_process", *[task], side='l')
                self.cache.delete(f"{task['task_name']}:{task['id']}")
            elif task['error'] is not None:
                self.cache.list_push(f"error_process", *[task], side='l')
            else:
                self.cache.set(f"{task['task_name']}:{task['id']}", task)
        except:
            pass

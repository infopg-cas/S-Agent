import math
import re
from abc import ABC

from src.agent.agents.base import AgentBase, AgentGroupBase, EnvironmentBase
from typing import Union, Any, Dict, Optional, Tuple
import pprint
from queue import Queue

def parse_function_call(input_str):
    match = re.match(r"(\w+)\((\d+)\)", input_str)
    if match:
        function_name = match.group(1)
        function_parameter = int(match.group(2))
        return function_name, function_parameter
    else:
        return None


class GeneralAgent(AgentBase):
    def __init__(
            self,
            agent_name: str,
            template: Any,
            llm: Any,
            actions: Dict = None,
            agent_description: str = None,
            memory: Dict = None,
    ):
        self.agent_name: str = agent_name
        self.description: str = agent_description
        self.prompt_template = template
        self.llm = llm
        self.memory = memory
        self.plugins_map: Dict = {}
        self.actions = actions
        self.actions_async = {}
        self.trajectory = Queue

    def agent_name(self) -> str:
        return self.agent_name

    def description(self) -> str:
        return self.description

    def prompt_template(self):
        return self.prompt_template

    # @property
    # def llm(self) -> Optional[LLMBase]:
    #     return self.llm

    # @llm.setter
    # def llm(self, llm_client: LLMBase):
    #     if llm_client is None or not isinstance(llm_client, LLMBase):
    #         raise Exception("Invalid llm client.")
    #     self.llm = llm_client

    def run_agent(self, query, *args, **kwargs):
        """
        a question as input
        """
        self.prompt_template += "Question: " + query + '\n'
        n_calls, n_bad_calls = 0, 0
        done = False
        obs, r, traj = None, None, []
        for i in range(1, 8):
            n_calls += 1
            start = self.prompt_template + f"Thought {i}: "
            messages = [
                {"role": "system", "content": ""},
                {"role": "user", "content": start}
            ]
            thought_action = self.llm.chat_completion_text(messages=messages)['content']
            if 'Finish:' in thought_action:
                self.prompt_template += f"Thought {i}: {thought_action}"
                thought, finish = thought_action.strip().split(f"\nFinish:")
                traj.append(f"Thought {i}: {thought}\n")
                traj.append(f"Finish: {finish}")
                break
            try:
                thought, action = thought_action.strip().split(f"\nAction {i}: ")
                action_name, action_parameter = parse_function_call(action)
                self.prompt_template += f"Thought {i}: {thought}\nAction {i}: {action}\n"
                traj.append(f"Thought {i}: {thought}\n")
                traj.append(f"Action {i}: {action}\n")

            except:
                n_bad_calls += 1
                n_calls += 1
                thought = thought_action.strip().split('\n')[0]
                self.prompt_template += f"Thought {i}: {thought}\nAction {i}:"
                messages = [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": self.prompt_template}
                ]
                action = self.llm.chat_completion_text(messages=messages)['content']
                print(action)
                action_name, action_parameter = parse_function_call(action)
                self.prompt_template += f"{action}\n"
                traj.append(f"Thought {i}: {thought}\n")
                traj.append(f"Action {i}: {action}\n")

            obs, r, done = self.process_action(
                action_name=action_name,
                action_parameter=action_parameter
            )
            obs = obs.replace('\\n', "")
            self.prompt_template += f"Observation {i}: {obs}\n"
            traj.append(f"Observation {i}: {obs}\n")
            messages = [
                {"role": "system", "content": ""},
                {"role": "user", "content": self.prompt_template}
            ]

        if not done:
            obs, r, done = self.process_action("finish", None)
        return obs, r, traj

    def process_action(
            self,
            action_name,
            action_parameter
    ) -> Tuple[str, float, bool]:
        reward = 0
        status = False
        if "finish" in action_name:
            return f"Episode finished, reward = {0}\n", 0, True

        if action_name not in self.actions.keys():
            obs = f"Invalid action: {action_name}"
            return obs, 0, False

        # self.actions[action_name] 为function函数的闭包
        try:
            action_result = self.actions[action_name](int(action_parameter))
            obs = str(action_result)
        except:
            obs = f"Can't use the tool, try to solve other ways."

        # To do: 将 action_result 转化为obs
        return obs, reward, status

    def express_information(self):
        pass

    def finetune_trajectory(self):
        pass

    def inference(self):
        pass

    def pass_information(self):
        pass

    def recall_memory(self):
        pass


class GeneralAgentGroup(AgentGroupBase, ABC):
    def __init__(
            self,
            n_agents: int,
            work_flow: Dict,
    ):
        self.agent_number = n_agents
        self.environment: Union[None] = None
        self.organ_structure: Dict = {}
        self.group_chat_pool = Queue
        self.worker_thread_pool = []


class GeneralEnv(EnvironmentBase, ABC):
    def __init__(self, work_flow):
        self.agent_basic_info = {}
        self.group_goal = []
        self.group_agent_goal = {}
        self.workflow = work_flow
        self.group_traject = Queue
        self.agent_work_info = {}

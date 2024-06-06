import math
import re
from abc import ABC
from abc import ABC, abstractmethod
from src.agent.agents.base import AgentBase, AgentGroupBase, EnvironmentBase
from typing import Union, Any, Dict, Optional, Tuple
import pprint
from queue import Queue

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
        self.messages = []
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

    def run_agent(self, query):
        pass

    def process_action(self, action_name, action_parameter):
        pass

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
        self.external_env = {}

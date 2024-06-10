import math
import re
from abc import ABC
from abc import ABC, abstractmethod
from src.agent.agents.base import AgentBase, AgentGroupBase, EnvironmentBase, AgentTeamBase
from typing import Union, Any, Dict, Optional, Tuple, Type
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
            group_name: str,
            add_human_as_default: bool = True
    ):
        self.group_name = group_name
        self.total_agent_numbers = 0
        self.total_group_numbers = 0
        self.total_staff_number = 0  # user_numbers + agents_numbers
        self.total_staff_number += 1 if add_human_as_default else 0

        self.upstream_group = None
        self.downstream_group = None
        self.group_pool: Dict[str:Dict] = {}
        self.environment: Union[None, Type[GeneralEnv]] = None
        self.organ_structure: Dict = {}
        self.group_chat_pool = Dict[str: Queue] = {'main': Queue()}
        self.worker_thread_pool = []

    def create_self_instance(self, **kwargs):
        return self.__class__(**kwargs)

    def add_agent_to_group(self, agent: Type[GeneralAgent], group_name: str) -> Tuple[bool,str]:
        if group_name in self.group_pool:
            self.group_pool[group_name][agent.agent_name] = agent
            return True, f"Success to add {agent.agent_name} into the group {group_name} in {self.group_name}"
        else:
            return False, f"Failed to add {agent.agent_name} into the group {group_name} in {self.group_name} since there is no such group, suggest to create the group first"

    def create_agent(self, *args, **kwargs) -> Tuple[bool,  Union[str, 'GeneralAgent']]:
        agent_name = kwargs.get('agent_name', None)
        llm = kwargs.get('llm', None)
        actions = kwargs.get('actions', {})
        prompt = kwargs.get('prompt', None)
        if not agent_name or not llm or not prompt:
            missing_params = [param for param in ["agent_name", "llm", "prompt"] if not locals()[param]]
            return False, f"Failed to create agent because parameters {', '.join(missing_params)} are missing."

        agent = GeneralAgent(
            agent_name=agent_name,
            llm=llm,
            actions=actions,
            template=prompt)
        return True, agent

    def create_group(self, group_name: str, add_human_as_default: bool = False) -> Tuple[bool, Union[str, 'GeneralAgentGroup']]:
        if not group_name:
            return False, "Failed to create group because the group name is missing."

        group = GeneralAgentGroup(
            group_name=group_name,
            add_human_as_default=add_human_as_default
        )
        return True, group

    def add_group_to_group(self, group_name):
        pass


class GeneralEnv(EnvironmentBase, ABC):
    def __init__(self, work_flow):
        self.agent_basic_info = {}
        self.group_goal = []
        self.group_agent_goal = {}
        self.workflow = work_flow
        self.group_traject = Queue
        self.agent_work_info = {}
        self.external_env = {}

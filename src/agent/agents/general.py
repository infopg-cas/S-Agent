import math
import re
from abc import ABC
from abc import ABC, abstractmethod
from src.agent.agents.base import AgentBase, AgentGroupBase, EnvironmentBase, AgentTeamBase
from typing import Union, Any, Dict, Optional, Tuple, Type, List
from src.utils.tree_structure import TreeNode, Tree
import pprint
from queue import Queue
import weakref

class GeneralAgent(AgentBase):
    def __init__(
            self,
            agent_name: str,
            template: Any,
            llm: Any,
            actions: Dict = None,
            agent_description: str = None,
            memory: Dict = {},
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
        self.trajectory = []
        self.upper_pointer = None

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

    def perception_env(self):
        group = self.upper_pointer() if self.upper_pointer else None
        if hasattr(group, 'metadata'):
            metadata = group.metadata
            team = metadata.upper_pointer() if metadata and metadata.upper_pointer() else None
            if team:
                sub_team = team.find_node_by_attribute(team.roots, 'group_name', 'meta group')
                team.mac_env.get_group_info('meta group', sub_team)
            else:
                print("Tree function cannot be called. Group or Tree is not set.")
        else:
            print("Right Group Pointer")


class GroupAgentTree(Tree):
    def __init__(self):
        super().__init__()
        self.stray_agents = {}
        self.stray_groups = {}
        self.group_pool = {}
        self.mac_env: Union[None, Type[GeneralEnv()]] = GeneralEnv()

    def add_agent_to_group(
            self, agent: Type[GeneralAgent],
            group_name: str
    ) -> Tuple[bool, str]:
        if agent.agent_name in self.stray_agents:
            group = self.find_node("group_name", group_name)
            if group:
                group.metadata.add_agent(agent)
                agent.upper_pointer = weakref.ref(group)
                return True, f"Success to add {agent.agent_name} into the group {group_name}."
            else:
                return True, f"Fail to add {agent.agent_name} into the group {group_name} since there is no such group, suggest to create the group first. "
        else:
            return False, f"Failed to add {agent.agent_name} into the group {group_name} since there is no such agent in isolate pool, might added already."

    def create_agent(
            self,
            *args,
            **kwargs
    ) -> Tuple[bool, Union[str, 'GeneralAgent']]:
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
        self.stray_agents[agent_name] = agent
        return True, "Success to create the agent."

    def create_group(
            self,
            group_name: str,
            add_human_as_default: bool = False
    ) -> Tuple[bool, Union[str, 'GeneralAgentGroup']]:
        group = self.find_node("group_name", group_name)
        if group:
            return False, "Failed to create group because the group name already exists, suggest to change another."

        group = GeneralAgentGroup(
            group_name=group_name,
            add_human_as_default=add_human_as_default
        )
        self.stray_groups[group_name] = group
        return True, "Success to create the group."

    def add_group_to_group(
            self,
            father_group_name,
            children_group_name
    ) -> Tuple[bool, str]:
        if not self.find_node_by_attribute(self.roots, "group_name", father_group_name):
            return False, f"Cannot add {father_group_name} to the team because the group does not exist, suggest to create it first."
        if children_group_name not in self.stray_groups:
            return False, f"Cannot add {children_group_name} to the team because the agent does not exist, suggest to create it first."

        self.add_node(
            parent_attribute="group_name",
            parent_value=father_group_name,
            child_attributes={"group_name": children_group_name, "metadata": self.stray_groups[children_group_name]}
        )
        del self.stray_groups[children_group_name]
        return True, f"Success to add the group {children_group_name} to {father_group_name}. "


class GeneralAgentGroup(AgentGroupBase, ABC):
    def __init__(
            self,
            group_name: str,
            add_human_as_default: bool = True
    ):
        self.group_name = group_name
        self.total_agent_numbers = 0
        self.total_user_numbers = 1 if add_human_as_default else 0
        self.total_staff_number = self.total_user_numbers + self.total_agent_numbers  # user_numbers + agents_numbers

        self.upstream_group = None
        self.downstream_group = None
        self.agent_organ_graph: Dict[Type[GeneralAgent], List] = {}
        self.group_pool: Dict[str: Dict] = {}
        # self.group_env: Union[None, Type[GeneralEnv()]] = GeneralEnv(group_pointer=self.group_name)

        self.group_chat_pool: Dict[str: Queue] = {'main': Queue()}
        self.worker_thread_pool = []
        self.upper_pointer = None

    def create_self_instance(self, **kwargs):
        return self.__class__(**kwargs)

    def add_agent(self, agent: Type[GeneralAgent]):
        self.agent_organ_graph[agent] = []
        self.total_agent_numbers += 1
        self.total_staff_number += 1
        # self.group_env.update_env()

    def group_eval(self):
        pass

    def group_finetune(self):
        pass

    def start_task(self):
        pass


class GeneralEnv(EnvironmentBase, ABC):
    def __init__(
            self,
            group_pointer,
            agent_work_flow=None,
            group_structure=None,
    ):
        self.agent_basic_info = {}
        self.group_goal = []
        self.group_agent_goal = {}
        self.workflow = agent_work_flow
        self.group_traject = Queue
        self.agent_work_info = {}
        self.external_env = {}
        self.group_structure_pointer = group_pointer

    def get_group_info(self, group_pointer, root): # pass in the root pointer for the get
        # get information in group_Structure
        pprint.pprint(root)
        return "group_info"

    def get_agent_info(self):
        # get information through work_flow
        return "agent_info"

    def get_env_info(self):
        return "env_info"

    def update_env(self):
        pass

    def get_group_memory(self):
        pass

    def init_env(self):
        pass

    def reset_env(self):
        pass

from abc import ABC, abstractmethod
from typing import Union, Any, Dict, Optional
from src.llms.hlevel.base import LLMBase
# from src.agent.prompts import PromptTemplate



class AgentBase(ABC):
    @abstractmethod
    def inference(self, *args, **kwargs):
        pass

    @abstractmethod
    def process_action(self, *args, **kwargs):
        pass

    @abstractmethod
    def express_information(self, *args, **kwargs):
        pass

    @abstractmethod
    def pass_information(self, *args, **kwargs):
        pass

    @abstractmethod
    def recall_memory(self):
        pass

    @abstractmethod
    def finetune_trajectory(self, *args, **kwargs):
        pass


class AgentGroupBase(ABC):
    """Group Execution Level Constructor"""
    @abstractmethod
    def start_task(self):
        pass

    @abstractmethod
    def group_finetune(self):
        pass

    @abstractmethod
    def group_eval(self):
        pass


class EnvironmentBase(ABC):
    """Environment Information Level Constructor"""
    @abstractmethod
    def get_env_info(self):
        """general background for the task"""
    @abstractmethod
    def get_group_info(self):
        """general background for the group agents as summary"""
    @abstractmethod
    def get_agent_info(self):
        """general background for the group agent"""
    @abstractmethod
    def get_group_memory(self):
        """get the group memory"""
    @abstractmethod
    def init_env(self):
        """initialize the env"""
    def update_env(self):
        """update the env"""
    @abstractmethod
    def reset_env(self):
        """reset the env"""

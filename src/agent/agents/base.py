from abc import ABC, abstractmethod
from typing import Union, Any, Dict, Optional
from src.llms.hlevel.base import LLMBase
# from src.agent.prompts import PromptTemplate
from queue import Queue


class AgentBase(ABC):
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
    def __init__(
            self,
            n_agents: int,
            work_flow: Dict,
    ):
        self.agent_number = n_agents
        self.workflow = work_flow
        self.environment: Union[None] = None
        self.organ_structure: Dict = {}
        self.group_chat_pool = Queue

    @abstractmethod
    def start_task(self):
        pass

    @abstractmethod
    def generate_group_knowledge(self):
        pass

    @abstractmethod
    def group_finetune(self):
        pass

    @abstractmethod
    def group_eval(self):
        pass

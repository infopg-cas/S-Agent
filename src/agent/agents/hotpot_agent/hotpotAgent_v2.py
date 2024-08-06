import random

from src.agent.agents.general import GeneralAgent
from typing import Tuple, Dict
from src.agent.environment import HotpotEnv
from src.agent.planning import AskIsWhatALlYouNeed


class HotpotAgent(GeneralAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.planning_stra = AskIsWhatALlYouNeed(self)
        self.planning_graph = self.planning_stra.get_planning_graph()
        self.env = HotpotEnv()

    def run(self,
            task: Dict,
            is_update=True):
        env = HotpotEnv(group_pointer=None, agent_work_flow=None, group_structure=None, question='',
                        key='', max_steps=6)
        actions = [
            "memory long_term short_term",
            "belief team_info iteration",
            "think iteration",
            "action tool iteration",
            "observation iteration",
            "ask iteration",
            "reflection iteration",
            "finish query"
        ]
        env.reset()
        action = random.choice(actions)
        observation, reward, done, info = env.step(action)
        print(observation, reward, done, info)
        print("Training completed.")
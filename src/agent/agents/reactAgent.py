import math
import re
from src.agent.agents.base import AgentBase
from typing import Tuple, Dict
from src.llms.hlevel import OpenAiLLM


def parse_function_call(input_str):
    match = re.match(r"(\w+)\((\d+)\)", input_str)
    if match:
        function_name = match.group(1)
        function_parameter = int(match.group(2))
        return function_name, function_parameter
    else:
        return None


class ReActAgent(AgentBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run_agent(self, query, *args, **kwargs):
        """
        a question as input
        """
        self.prompt_template += "Question: " + query + '\n'
        n_calls, n_bad_calls = 0, 0
        done = False
        r, traj = None, []
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

        pprint.pprint(self.prompt_template)
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
        action_result = self.actions[action_name](int(action_parameter))

        # To do: 将 action_result 转化为obs
        obs = str(action_result)
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


def calculate_square(x):
    return round(math.sqrt(x), 2)


def add_two(x, y):
    return x + y


if __name__ == "__main__":
    # open = OpenAiLLM(api_key='')
    # print(open.get_models())
    import pprint

    prompt = "You are a math teach agent, you job is to solve the problem for the user. " \
             "You have a tool library\n:" \
             "1. Name: 'calculate_square', which calculate the square root.\n" \
             "2. Name: 'add_two', which adds two numbers.\n" \
             "I want to you generate the answer step by step, you will move one step each time. you have 4 steps options: Think, Action, Observation, Finish:\n" \
             "For action step, you will tell me the action tool name that you want to call, and I will call it and give you the result in Observation." \
             "When you get the answer, give the result in Finish step.\n" \
             "Restrictions\n: " \
             "1. You have to process only one step each time.\n" \
             "2. If you cannot process it, use the Finish step, and write 'Can't Help'." \
             "Some examples are:\n" \
             "Question: What's square root of 4?\n" \
             "Thought 1: I need to use tool calculate to compute the square root of 4.\n" \
             "Action 1: calculate(4).\n" \
             "Observation 1: 2." \
             "Finish: The answer for square root of 4 is 2.\n"

    agent = ReActAgent(
        agent_name='reAct',
        llm=open,
        actions={
            "calculate_square": calculate_square,
            "add_two": add_two
        },
        template=prompt
    )
    obs, r, j = agent.run_agent("What's the difference of square root of 3 and the square root of 25?")
    pprint.pprint(j)

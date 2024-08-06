import concurrent
import re
import string
import os
from typing import Tuple, Optional
import gym
from transformers import T5Tokenizer, T5ForConditionalGeneration

from src.agent.agents.general import GeneralEnv, GroupAgentTree
from collections import Counter
from src.agent.planning import process_response
from src.llms.hlevel import OpenAiLLM
api_key = os.getenv('OPENAI_API_KEY')
class MessageStore:
    def __init__(self):
        self.messages = []
        self.trajectory = []

    def append_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def get_latest_message(self) -> str:
        return self.messages[-1].get('content') if self.messages else ""

    def add_trajectory(self, content: str):
        self.trajectory.append(content)

class HotpotEnv(GeneralEnv, gym.Env):
    def __init__(self,
                 group_pointer=None,
                 agent_work_flow=None,
                 group_structure=None,
                 question: str = '',
                 key: str = '',
                 max_steps: int = 6,
                 ):
        GeneralEnv.__init__(self, group_pointer, agent_work_flow, group_structure)
        gym.Env.__init__(self)

        self.question = question
        self.key = key
        self.max_steps = max_steps
        self.messages = []
        self.reset()
        self.message_store = MessageStore()
        self.api_key = api_key
        print(self.question, self.key,23211)

    def memory(self, long_term, short_term) -> Tuple[bool, str]:
        try:
            prompt = f"""
            There are some examples from your long term memory and short memory for the task: \n
            Long Term: {long_term}.
            Short Term: {short_term}.
            """
            prompt = f"Finish:{prompt}"
            self.message_store.append_message('user', prompt)
            return True, prompt
        except Exception as e:
            return False, f"{str(e)}"

    def belief(self, team_info, iteration) -> Tuple[bool, str]:
        try:
            prompt = f"""
                Based on the Team Info, give a belief of the current status of team task and team members.
                You need to focus on the macro scope of the team task, and micro scope of the skills, status, and progress of each team agent in the team.
                If there is no other team member, which means you are the only one in the team. \n 
                Team Info: \n
                {team_info}
                Belief {iteration}: 
            """
            self.message_store.append_message("user", prompt)
            response = OpenAiLLM(api_key=self.api_key).chat_completion_text(messages=self.message_store.messages)['content']
            if f"Belief {iteration}:" in response:
                response = response.split(":")[-1].strip()

            self.message_store.append_message('assistant', f"{response}.")
            self.message_store.add_trajectory(f"Belief {iteration}: {response}.")
            return True, f"Belief {iteration}: {response}."
        except Exception as e:
            return False, f"{str(e)}"

    def think(self, iteration) -> Tuple[bool, str]:
        try:
            prompt = f"""Think about what to act first, if you know which tool to use to process the task, 
                    return by start with "I want to act" and then give the tool you want to ask in json key_value pair, then give a short reasoning. 
                    (Example: I want to act - {'{"tool_name": "action_tool_name"}'}) - short reasoning here.... \n
                    If you think there is no tools for you, or you think there is gap for you to process the task, or you seem unsuccessful by using the tools,  
                    just return 'I want to ask'\n. 
                    """ + f"Thought {iteration}:"
            self.message_store.append_message("user", prompt)
            response = OpenAiLLM(api_key=self.api_key).chat_completion_text(messages=self.message_store.messages)['content']
            if f"Thought {iteration}:" in response:
                response = response.split(":")[-1].strip()
            self.message_store.append_message('assistant', f"{response}")
            self.message_store.add_trajectory(f"Thought {iteration}: {response}")
            return True, f"Thought {iteration}: {response}"
        except Exception as e:
            return False, f"{str(e)}"

    def action(self, tool, iteration) -> Tuple[bool, str]:
        from pydantic import BaseModel, create_model, Field

        def generate_dynamic_class(tool):
            import inspect
            parameters = inspect.signature(tool.func).parameters

            fields = {}
            for name, param in parameters.items():
                if param.annotation != inspect.Parameter.empty:
                    annotation = param.annotation
                else:
                    annotation = str

                default_value = param.default if param.default is not inspect.Parameter.empty else ...
                if isinstance(default_value, type):
                    fields[name] = (annotation, Field(default_factory=lambda: default_value))
                else:
                    fields[name] = (annotation, default_value)

            return create_model(
                'DynamicClass',
                **fields,
            )

        try:
            DynamicClass = generate_dynamic_class(tool)
            format = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": DynamicClass.model_json_schema()
                }
            ]
            prompt = f"""
                For Action state, you will tell me the arguments in a JSON format by the schema that I give you, and I will call it and give you the result.
                Only Return One Action state for each time and only return the arguments in one single json not nested.\n
                The JSON should be double quotes.\n
                Action {iteration}: 
                """

            self.message_store.append_message('user', prompt)
            response = OpenAiLLM(api_key=self.api_key).chat_completion_json(messages=self.message_store.messages, function_format=format)['content']
            if f"Action {iteration}:" in response:
                response = response.split(":")[-1].strip()

            res, response = process_response(response)
            if not res:
                return False, response
            self.message_store.append_message('assistant', f"{response}")
            self.message_store.add_trajectory(f"Action {iteration}: {response}")
            return True, response
        except Exception as e:
            return False, f"{str(e)}"

    def observation(self, iteration) -> Tuple[bool, str]:
        try:
            prompt = f"""Observation {iteration}: {self.message_store.get_latest_message()}."""
            self.message_store.add_trajectory(prompt)
            return True, prompt
        except Exception as e:
            return False, f"{str(e)}"

    def ask(self, iteration) -> Tuple[bool, str]:
        try:
            prompt = f"""Based on your belief of the team, generate a question or query, and choose which team member to ask." \
                     "Give the response like '@[Name of the team member]:[Your Question]'\n
                     Ask {iteration}: """
            self.message_store.append_message('user', prompt)
            response = OpenAiLLM(api_key=self.api_key).chat_completion_text(messages=self.message_store.messages)['content']
            if f"Ask {iteration}:" in response:
                response = response.split(":")[-1].strip()
            self.message_store.append_message('assistant', response)
            self.message_store.add_trajectory(f"Ask {iteration}: " + response)
            return True, f"Ask {iteration}: {response}"
        except Exception as e:
            return False, f"{str(e)}"

    def reflection(self, iteration) -> Tuple[bool, str]:
        try:
            prompt = f"Try to do self-reflection on the answer provide above. " \
                     "If you think the answer is enough to finish your task, then just return 'Correct answer, Finish'. " \
                     "If you think the answer is Not enough to finish or wrong, just return 'Not end, do again'." \
                     "If you are not sure about the answer, but willing to ask other team member to check for you, just return 'Ask for check.'" \
                     "Don't return an another answer.\n" \
                     f"Reflection {iteration}: "
            self.message_store.append_message('user', prompt)
            response = OpenAiLLM(api_key=self.api_key).chat_completion_text(messages=self.message_store.messages)['content']
            if f"Reflection {iteration}:" in response:
                response = response.split(":")[-1].strip()

            self.message_store.append_message('assistant', response)
            self.message_store.add_trajectory(f"Reflection {iteration}: {response}")
            return True, f"Reflection {iteration}: {response}"
        except Exception as e:
            return False, f"{str(e)}"

    def finish(self, query) -> Tuple[bool, str]:
        try:
            prompt = f"Based on the content, give your answer to the initial question {query}. " \
                     f"Give the answer in phrase not a sentence. \n" \
                     f"Finish Answer: "
            self.message_store.append_message("user", prompt)
            response = OpenAiLLM(api_key=self.api_key).chat_completion_text(messages=self.message_store.messages)['content']
            if f"Finish Answer:" in response:
                response = response.split(":")[-1].strip()
            self.message_store.append_message('assistant', response)
            self.message_store.add_trajectory(f"Finish Answer: {response}")
            self.answer = response
            if self.is_correct() > 0.7:
                self.terminated = True
            return True, f"Finish Answer: {response}"
        except Exception as e:
            return False, str(e)


    def reset(self):
          self.curr_step = 0
          self.terminated = False
          self.answer = ''

    def append_message(self, role, msg):
        if role in ["user", "system", "assistant"]:
            self.messages.append({"role": role, "content": msg})
        else:
            raise ValueError("No such type of role")

    def step(self, raw_action: str):
        if self.terminated:
            return None, 0, True, {}
        try:
            action_type, *arguments = raw_action.split()
            print(f'action_type1: {action_type}')
        except Exception as e:
            print(e)
            return None, 0, True, {"error": str(e)}
        action_map = {
            "memory": self.memory,
            "belief": self.belief,
            "think": self.think,
            "action": self.action,
            "observation": self.observation,
            "ask": self.ask,
            "reflection": self.reflection,
            "finish": self.finish
        }
        if action_type in action_map:
            print(f'action_type2 {action_type}')
            func = action_map[action_type]
            print(f'arguments: {arguments}')
            res, observation = func(*arguments)
        else:
            raise ValueError(f"Unknown action type: {action_type}")

        reward = self.is_correct()
        done = self.is_terminated()
        truncated = self.is_truncated()

        info = {
            "truncated": truncated,
            "step": self.curr_step
        }

        return observation, reward, done, info

    def is_correct(self) -> float:
        return EM(self.question, self.key)

    def is_terminated(self) -> bool:
        return self.terminated

    def is_truncated(self) -> bool:
        return self.curr_step >= self.max_steps

class BatchHotpotEnv():
    def __init__(
            self,
            env_load_path: str,
            cache_dir: str,
            device,
            max_conversation_length: int = 20,
            bsize: int = 32,
    ):
        self.env_list = [HotpotEnv(max_conversation_length) for _ in range(bsize)]
        self.bsize = bsize
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
        self.model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small", cache_dir=cache_dir).to(device)
        # self.model.load_state_dict(torch.load(env_load_path)['model_state_dict'])

    def generate_answers(self, questions):
        curr_words = [env.curr_word[0].lower() for env in self.env_list]
        inputs = [f"The object is {curr_word}." + question for curr_word, question in zip(curr_words, questions)]
        encoder_ids = self.tokenizer(inputs, padding=True, return_tensors='pt').to(self.model.device)
        return self.tokenizer.batch_decode(
            self.model.generate(
                input_ids=encoder_ids['input_ids'],
                attention_mask=encoder_ids['attention_mask'],
                max_new_tokens=16, do_sample=False), skip_special_tokens=True
        )

    def reset(self, idx: Optional[int] = None):
        return [env.reset(idx) for env in self.env_list]

    def step(self, questions):
        answers = self.generate_answers(questions)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            jobs = [executor.submit(env._step, q, a) for env, q, a in zip(self.env_list, questions, answers)]
            results = [job.result() for job in jobs]
        return results
def parse_action(string):
    pattern = r'^(\w+)\[(.+)\]$'
    match = re.match(pattern, string)

    if match:
        action_type = match.group(1)
        argument = match.group(2)
        return action_type, argument

    else:
        return None, None
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    print(f'normalize: {normalized_prediction,normalized_ground_truth}')

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def EM(answer, key) -> float:
    pred = normalize_answer(answer)
    gt = normalize_answer(key)
    return f1_score(pred, gt)
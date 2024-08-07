import random
from typing import Tuple, Dict, Tuple, List, Union
from src.exceptions import Config
import json
import re


def process_response(response):
    if isinstance(response, str):
        try:
            json_data = json.loads(response)
            return True, json_data
        except:
            pass
        json_match = re.search(r'(\{.*\})', response)

        if json_match:
            json_str = json_match.group(1)

            # Attempt to handle single quotes in JSON by replacing them with double quotes
            try:
                # json_str = re.sub(r'(?<!\\)\'', '"', json_str)
                # json_str = re.sub(r'\\\'', "'", json_str)
                # json_str = json_str.replace("'", '"', 1).replace("'", '"', 1)
                json_data = json.loads(json_str)
                return True, json_data
            except json.JSONDecodeError as e:
                return False, f"JSON decoding error: {str(e)}"
        else:
            return False, "No JSON found in the text."
    elif type(response) == dict:
        return True, response


class AskIsWhatALlYouNeed:
    # Planning Class
    # 1. Memory
    # Long Term
    # Short Term
    # Real Time
    # 2. Belief
    # Micro
    # Macro
    # 3. Thought
    # 4. Action
    # 5. Observation
    # 6. Self-Reflection
    # 7. Ask

    def __init__(self, agent, action_space, task):
        self.agent = agent
        self.action_space = action_space
        self.task = task  # hotpot, alfworld

    def get_planning_graph(self):
        # [str, str | List[Tuple[str, Union[str, None]]]]
        graph: Dict = {
            "memory": [("belief", None)],
            "belief": [("think", None)],
            "think": [("action", 'want to act'), ("ask", "want to ask")],
            "action": [("observation", None)],
            "ask": [("observation", None)],
            "observation": [("reflection", None)],
            "reflection": [("think", "again"), ("ask", "ask"), ("finish", "Correct answer, Finish")],
            "finish": "SINK"
        }
        return graph

    def memory(self, long_term, short_term, iteration) -> Tuple[bool, str]:
        try:
            prompt = f"""
            There are some examples from your long term memory and short memory for the task: \n
            Long Term: {long_term}.
            Short Term: {short_term}.
            """
            self.agent.append_message('user', f"Finish:{prompt}")
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
            self.agent.append_message("user", prompt)
            response = self.agent.llm.chat_completion_text(messages=self.agent.messages)['content']
            if f"Belief {iteration}:" in response:
                response = response.split(":")[-1].strip()

            self.agent.append_message('assistant', f"{response}.")
            self.agent.trajectory.append(f"Belief {iteration}: {response}.")
            return True, f"Belief {iteration}: {response}."
        except Exception as e:
            return False, f"{str(e)}"

    def think(self, iteration) -> Tuple[bool, str]:
        try:
            if self.task == "alfworld":
                prompt = f"""This about what to act first, if you know how to process the task, 
                        return by start with "I want to act" and then give the action from the action space: [{",".join(self.action_space)}], then give a short reasoning.
                        (Example: I want to act - <<one action from action space>> - short reasoning here.... Action you gave should EXACT MATCH with item in the action space\n
                        If you think there is no action you can take, or you think there is gap for you to process the task, or you seem unsuccessful by using the action,  
                        just return 'I want to ask'\n. 
                        """ + f"Thought {iteration}:"
            else:
                prompt = f"""This about what to act first, if you know which tool to use to process the task, 
                        return by start with "I want to act" and then give the tool you want to use in json key_value pair, then give a short reasoning. 
                        (Example: I want to act - {'{"tool_name": "action_tool_name"}'}) - short reasoning here.... \n
                        If you think there is no tools for you, or you think there is gap for you to process the task, or you seem unsuccessful by using the tools,  
                        just return 'I want to ask'\n. 
                        """ + f"Thought {iteration}:"
            self.agent.append_message("user", prompt)
            response = self.agent.llm.chat_completion_text(messages=self.agent.messages)['content']
            if f"Thought {iteration}:" in response:
                response = response.split(":")[-1].strip()
            reasoning = response.split("-")[-1]
            self.agent.append_message('assistant', f"{reasoning}")
            self.agent.trajectory.append(f"Thought {iteration}: {response}")
            return True, f"Thought {iteration}: {response}"
        except Exception as e:
            return False, f"{str(e)}"

    def action(self, iteration, tool=None, action=None) -> Tuple[bool, str]:
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
                __config__=Config
            )

        try:
            if tool:
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

                self.agent.append_message('user', prompt)
                response = self.agent.llm.chat_completion_json(messages=self.agent.messages, function_format=format)['content']
                if f"Action {iteration}:" in response:
                    response = response.split(":")[-1].strip()
                res, response = process_response(response)
                if not res:
                    return False, response
                self.agent.append_message('assistant', f"{response}")
                self.agent.trajectory.append(f"Action {iteration}: {response}")
                return True, response
            elif action:
                if len(action) > max([len(x) for x in self.action_space]):
                    for i in self.action_space:
                        if i in action:
                            action = i
                            break
                elif action not in self.action_space:
                    action = random.choices(self.action_space)
                prompt = f"""For Action state, you will tell me the detail of the how you will act.\n
                Ignore the instruction in the previous step. Only return the action content.
                Example:\n
                Input: put; 
                Your response shoule be like: tomato in the microwave
                Action {iteration}: {action} """

                self.agent.append_message('user', prompt)
                response = self.agent.llm.chat_completion_text(messages=self.agent.messages, max_tokens=15)['content']
                if f"Action {iteration}:" in response:
                    response = response.split(":")[-1].strip()
                if action in response:
                    response = response.replace(action, "")

                self.agent.append_message('assistant', f"{response}")
                self.agent.trajectory.append(f"Action {iteration}: {action} {response}")
                return True, f"{action} {response}"
        except Exception as e:
            return False, f"{str(e)}"

    def observation(self, iteration) -> Tuple[bool, str]:
        try:
            prompt = f"""Observation {iteration}: {self.agent.messages[-1].get('content')}"""
            self.agent.trajectory.append(prompt)
            return True, prompt
        except Exception as e:
            return False, f"{str(e)}"

    def ask(self, iteration) -> Tuple[bool, str]:
        try:
            prompt = f"""Based on your belief of the team, generate a question or query, and choose which team member to ask." \
                     "Give the response like '@[Name of the team member]:[Your Question]'\n
                     Ask {iteration}: """
            self.agent.append_message('user', prompt)
            response = self.agent.llm.chat_completion_text(messages=self.agent.messages)['content']
            if f"Ask {iteration}:" in response:
                response = response.split(":")[-1].strip()
            self.agent.append_message('assistant', response)
            self.agent.trajectory.append(f"Ask {iteration}: " + response)
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
            self.agent.append_message('user', prompt)
            response = self.agent.llm.chat_completion_text(messages=self.agent.messages)['content']
            if f"Reflection {iteration}:" in response:
                response = response.split(":")[-1].strip()
            self.agent.append_message('assistant', response)
            self.agent.trajectory.append(f"Reflection {iteration}: {response}")
            return True, f"Reflection {iteration}: {response}"
        except Exception as e:
            return False, f"{str(e)}"

    def finish(self, query=None) -> Tuple[bool, str]:
        try:
            if self.task == "alfworld":
                prompt = f"Based on the content, give a short summary to this task. " \
                         f"Finish Answer: "
            else:
                prompt = f"Based on the content, give your answer to the initial question {query}. " \
                         f"Give the answer in phrase not a sentence. \n" \
                         f"Finish Answer: "
            self.agent.append_message("user", prompt)
            response = self.agent.llm.chat_completion_text(messages=self.agent.messages, max_tokens=100)['content']
            if f"Finish Answer:" in response:
                response = response.split(":")[-1].strip()
            self.agent.append_message('assistant', response)
            self.agent.trajectory.append(f"Finish Answer: {response}")
            return True, f"Finish Answer: {response}"
        except Exception as e:
            return False, str(e)


if __name__ == "__main__":
    response = """{'entity': "Arthur's Magazine"}"""
    success, result = process_response(response)
    if success:
        print("Parsed JSON:", result)
    else:
        print("Error:", result)

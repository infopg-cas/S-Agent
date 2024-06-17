from typing import Tuple, Dict, Tuple, List, Union
from pydantic import BaseModel, create_model

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

    def __init__(self, agent):
        self.agent = agent

    def get_planning_graph(self):
        graph: Dict[str, str | List[Tuple[str, Union[str, None]]]] = {
            "memory": [("belief", None)],
            "belief": [("think", None)],
            "think": [("action", 'I want to act'), ("ask", "I want to Ask")],
            "action": [("observation", None)],
            "ask": [("observation", None)],
            "observation": [("reflection", None)],
            "reflection": [("belief", "again"), ("ask", "ask"), ("finish", "Correct answer, Finish")],
            "finish": "SINK"
        }
        return graph

    def memory(self, long_term, short_term, iteration) -> Tuple[bool, str]:
        try:
            prompt = f"""
                There are some examples from your long term memory and short memory for the task: 
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
            """
            self.agent.append_message("user", prompt)
            response = self.agent.llm.chat_completion_text(messages=self.agent.messages)['content']
            self.agent.append_message('assistant', f"Belief {iteration}: {response}.")
            self.agent.trajectory.append(f"Belief {iteration}: {response}.")
            return True, f"Belief: {response}."
        except Exception as e:
            return False, f"{str(e)}"

    def think(self, iteration) -> Tuple[bool, str]:
        try:
            prompt = f"""This about what to act first, if you know which tool to use to process the task, 
                    return by start with "I want to act" and then give the tool you want to ask in json key_value pair, then give a short reasoning. 
                    (Example: I want to act - {'{"tool_name": "tool_for_act"}'}) - short reasoning here.... \n
                    If you think there is no tools for you, or you think there is gap for you to process the task, 
                    just return 'I want to Ask'. 
                    """ + f"Thought {iteration}:"
            self.agent.append_message("user", prompt)
            response = self.agent.llm.chat_completion_text(messages=self.agent.messages)['content']
            self.agent.append_message('assistant', f"Thought {iteration}: {response}")
            self.agent.trajectory.append(f"Thought {iteration}: {response}")
            return True, response
        except Exception as e:
            return False, f"{str(e)}"

    def action(self, tool, iteration) -> Tuple[bool, str]:
        def generate_dynamic_class(tool):
            import inspect
            parameters = inspect.signature(tool.func).parameters

            fields = {}
            for name, param in parameters.items():
                if name == 'args' or name == 'kwargs':
                    continue

                if param.annotation != inspect.Parameter.empty:
                    annotation = param.annotation
                else:
                    annotation = str

                default_value = param.default if param.default is not inspect.Parameter.empty else ...
                fields[name] = (annotation, default_value)

            DynamicClass = create_model(
                'DynamicClass',
                **fields,
                __config__=type('Config', (), {'arbitrary_types_allowed': True})
            )
            return DynamicClass

        try:
            DynamicClass = generate_dynamic_class(tool)
            format = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": DynamicClass.schema()
                }
            ],
            prompt = f"""
                For Action state, you will tell me the parameters in a json format by the detail that I give you, and I will call it and give you the result. 
                """
            self.agent.append_message('user', prompt)
            response = self.agent.llm.chat_completion_json(messages=self.agent.messages, function_format=format)['content']
            self.agent.append_message('assistant', f"Action {iteration}: {response}")
            self.agent.trajectory.append(f"Action {iteration}: {response}")
            return True, response
        except Exception as e:
            print(str(e))
            return False, f"{str(e)}"

    def observation(self, iteration) -> Tuple[bool, str]:
        try:
            prompt = f"""Observation {iteration}:{self.agent.messages[-1]}."""
            self.agent.append_message('user', prompt)
            self.agent.trajectory.append(prompt)
            return True, prompt
        except Exception as e:
            return False, f"{str(e)}"

    def ask(self, iteration) -> Tuple[bool, str]:
        try:
            prompt = "Based on your belief of the team, generate a question or query, and choose which team member to ask." \
                     "Give the response like [Name of the team member]:[Question]\n"
            self.agent.append_message('user', prompt + f"Ask {iteration}")
            response = self.agent.llm.chat_completion_text(messages=self.agent.messages)['content']
            self.agent.append_message('assistant', f"Ask {iteration}-" + response)
            self.agent.trajectory.append(f"Ask {iteration}" + response)
            return True, f"Ask {iteration}" + response
        except Exception as e:
            return False, f"{str(e)}"

    def reflection(self, iteration) -> Tuple[bool, str]:
        try:
            prompt = "Try to do self-reflection on the answer provide above. " \
                     "If you think the answer is correct, then just return 'Correct answer, Finish'. " \
                     "If you think the answer is wrong or not enough to finish, just return 'Not end, do again'." \
                     "If you are not sure about the answer, but willing to ask other team member to check for you, just return 'Ask for check.'" \
                     "Don't return another answers."
            self.agent.append_message('user', prompt)
            response = self.agent.llm.chat_completion_text(messages=self.agent.messages)['content']
            self.agent.append_message('assistant', f"Reflection {iteration}:{response}")
            self.agent.trajectory.append(f"Reflection {iteration}:{response}")
            return True, response
        except Exception as e:
            return False, f"{str(e)}"

    def finish(self, query) -> Tuple[bool, str]:
        try:
            prompt = f"Based on the content, conclude your answer to the initial question {query}."
            self.agent.append_message("user", prompt)
            response = self.agent.llm.chat_completion_text(messages=self.agent.messages)['content']
            self.agent.append_message('assistant', f"Finish Answer:{response}")
            self.agent.trajectory.append(f"Finish Answer:{response}")
            return True, response
        except Exception as e:
            return False, str(e)

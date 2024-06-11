import math
from src.agent.agents.reactAgent import ReActAgent
from src.llms.hlevel import OpenAiLLM


# ==================== Tool library =================#
def calculate_square(x):
    return round(math.sqrt(x), 2)


def add_two(x, y):
    return x + y


def subtract(x, y):
    return x - y


if __name__ == "__main__":
    from Tokens import OPEN_KEY
    import pprint

    # "3. Name: 'subtract', which subtracts two numbers.\n" \
    # "3. If there is no tools for you to choose, give the answer by yourself.\n"
    # "1. You don't generate the Observation step, user will generate it.\n"
    # "4. If there is no tools for you to choose, give the answer by yourself.\n" \


    prompt = "You are a math teach agent, you job is to solve the problem for the user. " \
             "You have a tool library\n:" \
             "1. Name: 'calculate_square', which calculate the square root.\n" \
             "2. Name: 'add_two', which adds two numbers.\n" \
             "I want to you generate the answer step by step, you will move one step each time. You have 4 Action options: Think, Action, Observation, Finish:\n" \
             "For action step, you will tell me the action tool name that you want to call, and I will call it and give you the result in Observation." \
             "When you get the answer, give the result in Finish step.\n" \
             "Restrictions\n: " \
             "1. You have to process only one step each time.\n" \
             "2. If you cannot process it, use the Finish step, and write 'Can't Help'.\n" \
             "3. Do not call actions that not defined in the tool library.\n" \
             "Some examples are:\n" \
             "Question: What's square root of 4?\n" \
             "Thought 1: I need to use tool calculate to compute the square root of 4.\n" \
             "Action 1: calculate(4).\n" \
             "Observation 1: 2." \
             "Finish: The answer for square root of 4 is 2.\n"

    agent = ReActAgent(
        agent_name='reAct',
        llm=OpenAiLLM(api_key=OPEN_KEY),
        actions={
            "calculate_square": calculate_square,
            "add_two": add_two,
            # "subtract": subtract
        },
        template=prompt
    )
    cobs, r, j = agent.run_agent("What's the difference of square root of 3 and the square root of 259?")
    # obs, r, j = agent.run_agent("What's the weather today?")
    pprint.pprint(j)

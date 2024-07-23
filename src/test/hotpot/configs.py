from src.agent.tools.base import Tool
from src.agent.tools.wiki_tool import search_wiki, lookup
from Tokens import OPEN_KEY
from src.llms.hlevel import OpenAiLLM, OllamaLLM

REDIS_SETTINGS = {
  'tasks': {
    'host': '127.0.0.1',
    'port': 19783,
    'db': 1,
    'password': 'test'
  }
}

HOTPOT_EXAMPLE = """
Belief 1: As the only agent in the Hotpot Q&A group, my task is to answer the question about who the 'The Simpsons' character Milhouse was named after by Matt Groening. Currently, there is no progress on this task as I have not yet used any tools to find the answer. The supervisor is available for any questions. I am equipped to handle this task due to my ability to use the 'search_wiki' and 'lookup' tools from the tool library..
I want to act - {"tool_name": "search_wiki", "entity": "Milhouse"} - I believe the Wikipedia page for the character Milhouse from 'The Simpsons' will have information on who Matt Groening named the character after.
Action 1: {'entity': 'Milhouse'}
Observation 1: Milhouse Mussolini Van Houten[1] is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening. Milhouse is Bart Simpson's best friend in Mrs. Krabappel's fourth grade class at Springfield Elementary School. He is insecure, gullible, and is often led into trouble by Bart, who takes advantage of his friend's naïveté. Milhouse is a regular target for school bully Nelson Muntz and his friends Jimbo Jones, Dolph Starbeam and Kearney Zzyzwicz..
Reflection 1: Not end, do again.
Thought 2: I want to act - {"tool_name": "search_wiki", "entity": "Matt Groening"} - Since the information about Milhouse didn't provide the answer, I believe the Wikipedia page for Matt Groening may have information on who he named the character Milhouse after.
Action 2: {'entity': 'Matt Groening'}
Observation 2: Matthew Abram Groening (/ˈɡreɪnɪŋ/ GRAY-ning; born February 15, 1954)[1] is an American cartoonist, writer, producer, and animator. He is best known as the creator of the comic strip Life in Hell (1977–2012) and the television series The Simpsons (1989–present), Futurama (1999–2003, 2008–2013, 2023–present),[2] and Disenchantment (2018–2023). The Simpsons is the longest-running U.S. primetime television series in history and the longest-running U.S. animated series and sitcom...
Reflection 2: Not end, do again.
Thought 3: I want to ask - I have tried to find the information using the 'search_wiki' tool on both 'Milhouse' and 'Matt Groening' but did not find the answer to the question. I need guidance on how to proceed.
Ask 1 @supervisor: I have tried to find the information using the 'search_wiki' tool on both 'Milhouse' and 'Matt Groening' but did not find the answer to the question. Can you guide me on how to proceed?
 I have tried to find the information using the 'search_wiki' tool on both 'Milhouse' and 'Matt Groening' but did not find the answer to the question. Can you guide me on how to proceed?
Observation 3: Search for Milhouse and try to use look up tool to look up some keywords.
Reflection 3: Not end, do again.
Thought 4: Thought 8: I want to act - {"tool_name": "search_wiki", "entity": "Milhouse"} - The supervisor suggested to search for 'Milhouse' again on Wikipedia and then use the 'lookup' tool. I will start with the 'search_wiki' tool.
Action 4: {'entity': 'Milhouse'}
Observation 4: Milhouse Mussolini Van Houten[1] is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening. Milhouse is Bart Simpson's best friend in Mrs. Krabappel's fourth grade class at Springfield Elementary School. He is insecure, gullible, and is often led into trouble by Bart, who takes advantage of his friend's naïveté. Milhouse is a regular target for school bully Nelson Muntz and his friends Jimbo Jones, Dolph Starbeam and Kearney Zzyzwicz..
Reflection 4: Not end, do again.
Thought 5: I want to act - {"tool_name": "lookup", "keyword": "named after"} - Now that I have the 'Milhouse' page from Wikipedia, I will use the 'lookup' tool with the keyword 'named after' as suggested by the supervisor.
Action 5: {'keyword': 'named after'}
Observation 9:(Result 1 / 1) [6][7] Milhouse was named after U.S. president Richard Milhous Nixon..
Reflection 9: Correct answer, Finish.
Finish Answer: president Richard Milhous Nixon.
"""

HOTPOT_LLM = OpenAiLLM(api_key=OPEN_KEY,model='gpt-4o', temperature=0.05)
# HOTPOT_LLM = OllamaLLM(model='llama2',ollama_host="http://gcn008.csns.ihep.ac.cn",ollama_port=60000, temperature=0.05)

HOTPOT_PROMPT = f"""You are a Hotpot Q&A question agent, you job is to:\n 
             Answer the question and provide the correct answer. \n
             You have a tool library\n:" 
             1. Name: 'search_wiki', which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.\n
             2. Name: 'lookup', which returns the next sentence containing keyword in the current passage.\n 
             Restrictions\n:
             1. Do not call actions that not defined in the tool library.\n 
             2. You have to response short but clean.\n
             Example:\n
             {HOTPOT_EXAMPLE}\n
             """


HOTPOT_TOOLS = {
    "search_wiki": Tool(
        name='search_wiki',
        description='searches the exact entity on Wikipedia returns the first paragraph if it exists. If not, it will return some similar entities to search.',
        func=search_wiki
    ),
    "lookup": Tool(
        name='lookup',
        description='returns the next sentence containing keyword in the current passage.',
        func=lookup
    )
}
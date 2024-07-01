from src.agent.tools.base import Tool
from src.agent.tools.wiki_tool import search_wiki, lookup
from Tokens import OPEN_KEY
from src.llms.hlevel import OpenAiLLM

REDIS_SETTING = {

}

HOTPOT_LLM = OpenAiLLM(api_key=OPEN_KEY)

HOTPOT_PROMPT = "You are a Hotpot Q&A question agent, you job is to:\n" \
             "Answer the question and provide the correct answer. \n" \
             "Follow the guidance by humans.\n" \
             "You have a tool library\n:" \
             "1. Name: 'search_wiki', which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.\n" \
             "2. Name: 'lookup', which returns the next sentence containing keyword in the current passage.\n" \
             "Restrictions\n: " \
             "1. Do not call actions that not defined in the tool library.\n" \
             "2. You have to response short but clean.\n" \

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
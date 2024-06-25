import os
import ast
import chainlit as cl

from src.agent.agents.db_agent.configs.model_config import MODEL_NAME
from src.agent.agents.db_agent.datasource.db_conn_info import DBConfig
from src.agent.agents.db_agent.configs.db_config import DB_TYPE, DB_NAME, DB_HOST, DB_PORT, DB_USER, DB_PWD
from src.agent.agents.db_agent.rag.similarity.table_similarity import TableSimilarityFinder
from src.llms.hlevel import OpenAiLLM


@cl.on_message
async def main(message: cl.Message):
    model_name = MODEL_NAME
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        cl.send_message("环境变量 OPENAI_API_KEY 未设置")
        return

    DB_Example = DBConfig(
        db_type=DB_TYPE,
        db_name=DB_NAME,
        db_host=DB_HOST,
        db_port=DB_PORT,
        db_user=DB_USER,
        db_pwd=DB_PWD
    )

    # Read table structure from DB
    table_structure = DB_Example.get_table_structure()
    table_descriptions = DB_Example.format_table_descriptions(table_structure)
    related_tables = DB_Example.get_foreign_keys()
    foreign_keys = DB_Example.format_foreign_keys(related_tables)
    finder = TableSimilarityFinder(model_name, table_descriptions, foreign_keys, "", MODEL_NAME)
    similar_tables = finder.find_similar_table(message.content)
    incomplete_tables = []
    table_selection_prompt = "You are a table selection agent. Your job is to find the most relevant tables.\n" \
                             "A user's query asking for specific information.\n" \
                             "A dictionary of foreign keys representing relationships between tables.\n" \
                             "A dictionary of tables with similarity scores indicating their relevance to the query.\n" \
                             "The incomplete results list. \n" \
                             "Your task is to determine which tables are most relevant to answering the user's query based on the provided foreign key relationships and similarity scores.\n" \
                             "Here are the details: \n" \
                             "User's query: \n" \
                             f"{message.content} \n" \
                             "Foreign keys: \n" \
                             f"{foreign_keys} \n" \
                             "Similarity scores for tables: \n" \
                             f"{similar_tables} \n" \
                             "Incomplete table names: \n" \
                             f"{str(incomplete_tables)} \n" \
                             "Please give me the complete table names and add them to the incomplete tables list. Return the complete table names list directly without any additional response or explanation. \n" \
                             "Examples \n" \
                             "User's query: \n" \
                             "What is the number of students enrolled in the China History lesson? \n" \
                             "Foreign keys \n" \
                             "{'studentlesson': ['student', 'lesson']} \n" \
                             "Similarity scores for tables: \n" \
                             "('studentlesson': 0.85,'lesson': 0.78,'student': 0.76} \n" \
                             "Incomplete table names: \n" \
                             "['studentlesson'] \n" \
                             "Answer: \n" \
                             "['studentlesson', 'lesson']"

    messages = [
        {"role": "system", "content": table_selection_prompt},
        {"role": "user", "content": ""}
    ]

    tables = OpenAiLLM(api_key=openai_api_key).chat_completion_text(messages=messages)['content']
    print('seleted tables: ',tables)
    tables_list = ast.literal_eval(tables)

    create_statements = DB_Example.get_create_table_sql(tables_list)
    sql_generation_prompt = f"""
        Below are SQL table schemas paired with an instruction that describes a task. Using valid SQLite, write a response that appropriately completes the request for the provided tables.
        If the instruction is invalid or cannot be completed with the provided tables, please return an appropriate error message.
        ### Instruction: {message.content}
        ### Input: {create_statements}
        ### Response:
    """

    sql_response_generation_message = [
        {"role": "system", "content": sql_generation_prompt},
        {"role": "user", "content": ""}
    ]

    answer = OpenAiLLM(api_key=openai_api_key).chat_completion_text(messages=sql_response_generation_message)['content']
     # is valid or not
    def is_valid_sql(answer):
        sql_keywords = ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP"]
        return any(keyword in answer.upper() for keyword in sql_keywords)
    if is_valid_sql(answer):
        result = DB_Example.execute_query(answer)
        if not result:
            result = "No relevant content found."
    else:
        result = "Invalid input. Please provide a valid instruction."

    await cl.Message(
        content=f"{result}",
    ).send()

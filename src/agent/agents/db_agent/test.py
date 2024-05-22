from src.agent.agents.db_agent.configs.model_config import MODEL_NAME
from src.agent.agents.db_agent.datasource.db_conn_info import DBConfig
from src.agent.agents.db_agent.configs.db_config import DB_TYPE, DB_NAME, DB_HOST, DB_PORT, DB_USER, DB_PWD
from src.agent.agents.db_agent.rag.similarity.table_similarity import TableSimilarityFinder

if __name__ == "__main__":
    model_name = MODEL_NAME

    DB_Example = DBConfig(
        db_type=DB_TYPE,
        db_name=DB_NAME,
        db_host=DB_HOST,
        db_port=DB_PORT,
        db_user=DB_USER,
        db_pwd=DB_PWD
    )
    table_structure = DB_Example.get_table_structure()
    table_descriptions = DB_Example.format_table_descriptions(table_structure)
    foreign_keys = {
        "iep_user": ["iep_user_role"],
    }
    query_instruction_for_retrieval = ""
    repo_id = MODEL_NAME

    finder = TableSimilarityFinder(model_name, table_descriptions, foreign_keys, query_instruction_for_retrieval, repo_id)
    # input user's question
    query = ""
    similar_tables = finder.find_similar_table(query)

    print("similar tables:")
    for table, similarity in similar_tables.items():
        print(f"{table}: {similarity:.4f}")

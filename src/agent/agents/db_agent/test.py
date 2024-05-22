from src.agent.agents.db_agent.configs.model_config import MODEL_NAME
from src.agent.agents.db_agent.rag.similarity.table_similarity import TableSimilarityFinder

if __name__ == "__main__":
    model_name = MODEL_NAME
    table_descriptions = {
        "Table 1": "Contains user data including names, emails, and addresses.",
        "Table 2": "Includes transaction records with timestamps and amounts.",
        "Table 3": "Stores product details such as product names, categories, and prices."
    }
    foreign_keys = {
        "Table 1": ["Table 2"],
        "Table 3": []
    }
    query_instruction_for_retrieval = ""
    repo_id = MODEL_NAME

    finder = TableSimilarityFinder(model_name, table_descriptions, foreign_keys, query_instruction_for_retrieval, repo_id)

    query = "I need information about user."
    similar_tables = finder.find_similar_table(query)

    print("Most similar tables:")
    for table, similarity in similar_tables.items():
        print(f"{table}: {similarity:.4f}")

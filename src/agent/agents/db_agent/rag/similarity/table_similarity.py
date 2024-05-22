
from src.agent.agents.db_agent.rag.embedding.embeddings import DefaultEmbedding
from src.agent.agents.db_agent.rag.similarity.rerank import SimilarityReranker


class TableSimilarityFinder:
    def __init__(self, model_name, table_descriptions, foreign_keys, query_instruction_for_retrieval, repo_id, weight=0.1):
        self.encoder = DefaultEmbedding(model_name, query_instruction_for_retrieval, repo_id)
        self.reranker = SimilarityReranker(table_descriptions, foreign_keys, weight)

    def find_similar_table(self, query):
        """
        Finds the most similar table based on the query.
        """
        query_vec = self.encoder.encode_queries(query)
        similar_tables = self.reranker.rerank(query_vec, lambda texts: self.encoder.encode(texts)[0])
        return similar_tables

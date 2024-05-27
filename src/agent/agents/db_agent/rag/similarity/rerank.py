import torch

class SimilarityReranker:
    def __init__(self, table_descriptions, foreign_keys, weight=0.05):
        """
        Parameters:
        - table_descriptions (dict): Descriptions of the tables.
        - foreign_keys (dict): Relationships between tables.
        - weight (float): Weight for adjusting similarity based on foreign keys.
        """
        self.table_descriptions = table_descriptions
        self.foreign_keys = foreign_keys
        self.weight = weight

    def calculate_similarity(self, query_vec, encode_fn):
        """
        Calculates cosine similarity between the query vector and table descriptions.
        """
        similarities = {}
        for table_name, table_description in self.table_descriptions.items():
            table_vec = encode_fn([table_name + " " + table_description])
            similarity = torch.nn.functional.cosine_similarity(query_vec.unsqueeze(0), table_vec, dim=1)
            similarities[table_name] = similarity.item()
        return similarities

    def adjust_similarity_for_foreign_keys(self, similarities):
        """
        Adjusts similarity scores based on foreign key relationships.
        """
        adjusted_similarities = similarities.copy()
        for table_name, related_tables in self.foreign_keys.items():
            for related_table in related_tables:
                if related_table in similarities:
                    adjusted_similarities[related_table] += self.weight * similarities[table_name]
        return adjusted_similarities

    def rerank(self, query_vec, encode_fn):
        """
        Reranks tables based on the query.
        """
        similarities = self.calculate_similarity(query_vec, encode_fn)
        adjusted_similarities = self.adjust_similarity_for_foreign_keys(similarities)
        sorted_similarities = dict(sorted(adjusted_similarities.items(), key=lambda item: item[1], reverse=True))
        return sorted_similarities

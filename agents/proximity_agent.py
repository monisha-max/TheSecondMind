import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class ProximityAgent:
    """
    Uses FAISS to store and retrieve semantically similar cases based on embeddings.
    The embeddings are produced by a SentenceTransformer.
    """
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.memory_store = [] 
        self.index = None
        self.dim = 384 

    def add_memory(self, memory_id, content, metadata=None):
        embedding = self.embedding_model.encode(content)
        vector = np.array([embedding]).astype('float32')
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(vector)
        record = {
            "id": memory_id,
            "content": content,
            "embedding": embedding,
            "metadata": metadata
        }
        self.memory_store.append(record)

    def retrieve_similar_cases(self, query, top_k=3):
        if self.index is None or self.index.ntotal == 0:
            return []
        query_embedding = self.embedding_model.encode(query)
        query_vector = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_vector, top_k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.memory_store):
                record = self.memory_store[idx]
                similarity = 1 / (1 + distances[0][i])
                results.append({
                    "id": record["id"],
                    "content": record["content"],
                    "similarity": similarity,
                    "metadata": record.get("metadata")
                })
        return results

    def rerank_with_llm(self, query, retrieved_cases, llm_reranker=None):
        if llm_reranker:
            return llm_reranker(query, retrieved_cases)
        return retrieved_cases

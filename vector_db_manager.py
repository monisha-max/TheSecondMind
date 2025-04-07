import os
import openai
import pinecone

class VectorDBManager:
    """
    Manages interactions with a vector database (Pinecone in this example).
    Demonstrates how to embed text (using OpenAI) and store/retrieve it in Pinecone.
    """

    def __init__(self, index_name="my_vector_index", environment="us-west1-gcp"):
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
        pinecone_api_key = os.environ.get("PINECONE_API_KEY", "YOUR_PINECONE_API_KEY")

        openai.api_key = self.openai_api_key
        pinecone.init(api_key=pinecone_api_key, environment=environment)

        self.index_name = index_name
        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(self.index_name, dimension=1536) 
        self.index = pinecone.Index(self.index_name)

    def embed_text(self, text):
        """
        Uses OpenAI's text-embedding-ada-002 to embed the text into a 1536-d vector.
        """
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response["data"][0]["embedding"]

    def upsert_text(self, doc_id, text):
        """
        Embed the text and upsert it into Pinecone with a given doc_id.
        """
        embedding = self.embed_text(text)
        self.index.upsert([(doc_id, embedding, {})])

    def query_similar(self, text, top_k=3):
        """
        Query the index for text similar to the given query.
        Returns top_k results.
        """
        query_embedding = self.embed_text(text)
        results = self.index.query(query_embedding, top_k=top_k, includeMetadata=True)
        return results

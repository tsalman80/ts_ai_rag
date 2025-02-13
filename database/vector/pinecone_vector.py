import os
import pinecone
from langchain_community.vectorstores import Pinecone


class PineconeVectorDB:
    """
    Embeds the texts using Pinecone embeddings and stores them in a Pinecone vector database.
    """

    @staticmethod
    def embeddings_on_pinecone(embeddings, texts):
        """
        Embeds the texts using Pinecone embeddings and stores them in a Pinecone vector database.

        Args:
            embeddings: Embeddings object
            texts: List of texts to embed
        Returns:
            vector_store: Pinecone vector database
        """
        pinecone.init(
            api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV")
        )

        vector_store = Pinecone.from_documents(
            texts,
            embeddings,
            index_name=os.getenv("PINECONE_INDEX"),
        )

        return vector_store

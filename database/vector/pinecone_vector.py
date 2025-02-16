import os
from config import PINECONE_API_KEY, PINECONE_INDEX
from pinecone import Pinecone as PineconeClient
from langchain_pinecone import PineconeVectorStore


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

        pc = PineconeClient(api_key=PINECONE_API_KEY)
        vector_store = PineconeVectorStore(
            index=pc.Index(PINECONE_INDEX), embedding=embeddings
        )
        
        vs = vector_store.from_documents(texts, embeddings, index_name=PINECONE_INDEX)

        retriever = vs.as_retriever()
        return retriever

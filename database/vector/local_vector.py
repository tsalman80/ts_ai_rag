import os
from pathlib import Path
from langchain_community.vectorstores import Chroma

# local vector store directory
LOCAL_VECTOR_STORE_DIR = (
    Path(__file__).resolve().parents[2].joinpath("data", "vector_store")
)

print(LOCAL_VECTOR_STORE_DIR)


class LocalVectorDB:
    """
    Embeds the texts using OpenAI embeddings and stores them in a local vector database.
    """

    @staticmethod
    def embeddings_on_local_vectordb(embeddings, texts):
        """
        Embeds the texts using OpenAI embeddings and stores them in a local vector database.

        Args:
            embeddings: Embeddings object
            texts: List of texts to embed
        Returns:
            retriever: Retriever object
        """

        vector_store = Chroma.from_documents(
            texts, embeddings, persist_directory=LOCAL_VECTOR_STORE_DIR.as_posix()
        )

        retriever = vector_store.as_retriever(search_kwargs={"k": 7})
        return retriever

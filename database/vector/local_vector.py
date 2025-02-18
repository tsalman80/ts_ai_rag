import os
from pathlib import Path
from langchain_community.vectorstores import Chroma
from config import LOCAL_VECTOR_STORE_DIR
import streamlit as st


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

        # metadata = {"session_id": st.session_state.session_id}
        vector_store = Chroma.from_documents(
            texts,
            embeddings,
            persist_directory=LOCAL_VECTOR_STORE_DIR,
        )

        retriever = vector_store.as_retriever(search_kwargs={"k": 7})
        return retriever

from langchain_chroma import Chroma
from config import LOCAL_VECTOR_STORE_DIR, OPENAI_API_KEY
from langchain_openai.embeddings import OpenAIEmbeddings


class LocalVectorDB:
    """
    Embeds the texts using OpenAI embeddings and stores them in a local vector database.
    """

    @staticmethod
    def embeddings_on_local_vectordb(texts):
        """
        Embeds the texts using OpenAI embeddings and stores them in a local vector database.

        Args:
            embeddings: Embeddings object
            texts: List of texts to embed
        Returns:
            retriever: Retriever object
        """
        embeddings = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            model="text-embedding-3-small",
        )

        # metadata = {"session_id": st.session_state.session_id}
        vector_store = Chroma.from_documents(
            texts,
            embeddings,
            persist_directory=LOCAL_VECTOR_STORE_DIR,
        )
        
        # Save to disk
        vector_store.persist()


        retriever = vector_store.as_retriever(search_kwargs={"k": 7})
        return retriever

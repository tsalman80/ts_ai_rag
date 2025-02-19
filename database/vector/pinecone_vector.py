import os
from config import PINECONE_API_KEY, PINECONE_INDEX, OPENAI_API_KEY
from pinecone import Pinecone as PineconeClient
from langchain_pinecone import PineconeVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings


class PineconeVectorDB:
    """
    Embeds the texts using Pinecone embeddings and stores them in a Pinecone vector database.
    """

    @staticmethod
    def embeddings_on_pinecone(texts):
        """
        Embeds the texts using Pinecone embeddings and stores them in a Pinecone vector database.

        Args:
            embeddings: Embeddings object
            texts: List of texts to embed
        Returns:
            vector_store: Pinecone vector database
        """

        embeddings = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            model="text-embedding-3-small",
        )

        pc = PineconeClient(api_key=PINECONE_API_KEY)
        vector_store = PineconeVectorStore(
            index=pc.Index(PINECONE_INDEX), embedding=embeddings
        )
        vs = vector_store.from_documents(texts, embeddings, index_name=PINECONE_INDEX)
        vs.persist()

        # retriever = vs.as_retriever(search_kwargs={"filter": {"session_id": st.session_state.session_id}})
        return vs.as_retriever()

import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma


def embeddings_on_local_vectordb(texts):
    """
    Embeds the texts using OpenAI embeddings and stores them in a local vector database.

    """

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    vector_store = Chroma.from_documents(
        texts, embeddings, persist_directory=LOCAL_VECTOR_STORE_DIR.as_posix()
    )

    vector_store.persist()
    retriever = vector_store.as_retriever(search_kwargs={"k": 7})
    return retriever

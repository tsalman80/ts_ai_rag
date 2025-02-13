import os
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings

def embeddings_on_pinecone(texts):
    """
    Embeds the texts using Pinecone embeddings and stores them in a Pinecone vector database.
    """
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV")
    )

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    vector_store = Pinecone.from_documents(
        texts,
        embeddings,
        index_name=os.getenv("PINECONE_INDEX"),
    )

    return vector_store

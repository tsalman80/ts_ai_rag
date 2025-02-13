import os
from pathlib import Path
import streamlit as st

# temporary directory
TMP_DIR = Path(__file__).resolve().parent.joinpath("data", "tmp")

# local vector store directory
LOCAL_VECTOR_STORE_DIR = (
    Path(__file__).resolve().parent.joinpath("data", "vector_store")
)


from langchain_community.document_loaders import DirectoryLoader


def load_documents():
    """
    Loads PDF documents from the temporary directory.

    Returns:
        documents: List of loaded document objects
    """

    loader = DirectoryLoader(TMP_DIR, glob="**/*.pdf")
    documents = loader.load()
    return documents


from langchain_text_splitters import CharacterTextSplitter

def split_documents(documents):
    """
    Splits the documents into chunks of text.

    Args:
        documents: List of loaded document objects
    Returns:
        chunks: List of chunks of text
    """

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(documents)
    return chunks


from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.llms.openai import OpenAIChat


def query_llm(retriever, query):
    """
    Queries the LLM with the given query and returns the response.
    """

    llm = OpenAIChat(model="gpt-3.5-turbo", temperature=0)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm, retriever, return_source_documents=True
    )

    result = qa_chain(
        {"question": query, "chat_history": st.session_state.get("chat_history", [])}
    )
    return result["answer"]


def page_main():
    """
    Main page of the application.
    """
    st.title("AI RAG")
    st.write(
        "This is a simple chatbot that uses a vector database to store and retrieve documents."
    )

    with st.sidebar:
        st.session_state.vector_store = st.text_input("OpenAI API Key", type="password")
        st.session_state.pinecone_db = st.toggle("Use Pinecone Vector DB", value=False)

        st.session_state.source_documents = st.file_uploader(
            "Upload PDF documents",
            type="pdf",
            accept_multiple_files=True,
        )

from pathlib import Path
from documents import DocumentProcessor


import time
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI

import os
import tempfile
from database.vector.local_vector import LocalVectorDB
from database.vector.pinecone_vector import PineconeVectorDB
from documents.loader import DocumentLoader
from documents.splitter import DocumentSplitter
from documents import DocumentProcessor
import streamlit as st
from config import OPENAI_API_KEY, TEMP_DIR

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub

from streamlit import runtime


st.set_page_config(layout="wide")


def query_llm(retriever, query):
    """
    Queries the LLM with the given query and returns the response.
    """

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)

    condense_question_system_template = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    condense_question_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", condense_question_system_template),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, condense_question_prompt
    )

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three to five sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "<context> {context} </context>"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )

    combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)
    retrieval_chain = create_retrieval_chain(
        history_aware_retriever, combine_docs_chain
    )
    result = retrieval_chain.invoke(
        {
            "input": query,
            "chat_history": st.session_state.get("chat_history", []),
        }
    )

    return result["answer"]


def create_temp_dir():
    """Create a temporary directory."""
    tmp_dir = tempfile.mkdtemp(dir=TEMP_DIR)
    return tmp_dir


def save_files(uploaded_files):
    if not uploaded_files or st.session_state["has_documents_uploaded"]:
        return

    status_bar = None
    status_bar = st.progress(0, text="Loading documents...")

    # write the uploaded documents to the temporary directory
    tmp_dir = create_temp_dir()
    DocumentProcessor.write_documents(tmp_dir, st.session_state.source_documents)
    status_bar.progress(0.25, text="Documents loaded")

    # load the documents from the temporary directory
    documents = DocumentLoader.load(tmp_dir)
    status_bar.progress(0.5, text="Documents split")

    # split the documents into chunks
    chunks = DocumentSplitter.chunk(documents)
    status_bar.progress(0.75, text="Documents split")

    # embed the chunks
    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY,
        model="text-embedding-3-small",
    )
    status_bar.progress(0.9, text="Documents embedded")

    # embed the chunks on the vector database
    retriever = None
    if st.session_state.pinecone_db:
        retriever = PineconeVectorDB.embeddings_on_pinecone(embeddings, chunks)
    else:
        retriever = LocalVectorDB.embeddings_on_local_vectordb(embeddings, chunks)
    st.session_state["retriever"] = retriever

    status_bar.progress(100, text="Ready to chat!")

    st.session_state["has_documents_uploaded"] = True

    if status_bar:
        time.sleep(1)
        status_bar.empty()


def chat_interface():
    retriever = st.session_state["retriever"]

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask me anything about the uploaded documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.markdown("")

            # Query the LLM
            response = query_llm(retriever, prompt)
            message_placeholder.empty()
            message_placeholder.markdown(response + " ")

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.chat_history = st.session_state.messages


def page_main():
    """
    Main page of the application.
    """
    if "has_documents_uploaded" not in st.session_state:
        st.session_state["has_documents_uploaded"] = False

    st.title("AI RAG")
    st.write(
        "This is a simple chatbot that uses a vector database to store and retrieve documents."
    )

    st.caption("Upload your documents and start chatting with the AI!")

    with st.sidebar:
        st.title("Settings")
        st.session_state.pinecone_db = st.toggle("Use Pinecone Vector DB", value=False)
        selected_files = st.file_uploader(
            "Upload PDF documents",
            type="pdf",
            accept_multiple_files=True,
        )

        for file in selected_files:
            if not DocumentProcessor.is_valid_file_size(file):
                st.toast(f":red[File size is too large!] {file.name}")
                selected_files.remove(file)

        st.session_state.source_documents = selected_files
        button_load_file = st.button("Load documents", use_container_width=True)

    if button_load_file:
        if not st.session_state.source_documents:
            st.toast(":red[No documents selected!]")
        else:
            save_files(st.session_state.source_documents)

    if st.session_state["has_documents_uploaded"]:
        chat_interface()


def get_session_id():
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    ctx = get_script_run_ctx()
    return ctx.session_id


if __name__ == "__main__":
    # st.session_state["session_id"] = get_session_id()
    page_main()

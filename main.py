from langchain.globals import set_verbose, set_debug

set_verbose(True)
set_debug(True)
import time
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI

import os
import tempfile
from database.vector.local_vector import LocalVectorDB
from database.vector.pinecone_vector import PineconeVectorDB
from documents.loader import DocumentLoader
from documents.splitter import DocumentSplitter
import streamlit as st


import dotenv

dotenv.load_dotenv()

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub


def query_llm(retriever, query):
    """
    Queries the LLM with the given query and returns the response.
    """

    llm = ChatOpenAI(
        model="gpt-4o-mini", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY")
    )

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
        "don't know. Use three sentences maximum and keep the "
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


def write_documents(uploaded_files):
    """Write uploaded documents to a temporary directory."""
    temp_dir = tempfile.mkdtemp(dir="data/tmp")
    for file in uploaded_files:
        file_path = os.path.join(temp_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getvalue())
    return temp_dir


def page_main():
    """
    Main page of the application.
    """
    st.title("AI RAG")
    st.write(
        "This is a simple chatbot that uses a vector database to store and retrieve documents."
    )

    st.write("Upload your documents and start chatting with the AI!")

    if "has_documents_uploaded" not in st.session_state:
        st.session_state["has_documents_uploaded"] = False

    with st.sidebar:
        st.session_state.pinecone_db = st.toggle("Use Pinecone Vector DB", value=False)

        st.session_state.source_documents = st.file_uploader(
            "Upload PDF documents",
            type="pdf",
            accept_multiple_files=True,
            help="Upload PDF documents to be used for review",
        )

        status_bar = None
        if (
            st.session_state.source_documents
            and not st.session_state["has_documents_uploaded"]
        ):
            status_bar = st.progress(0, text="Loading documents...")

            # write the uploaded documents to the temporary directory
            temp_dir = write_documents(st.session_state.source_documents)
            status_bar.progress(0.25, text="Documents loaded")

            # load the documents from the temporary directory
            documents = DocumentLoader.load_documents(temp_dir)
            status_bar.progress(0.5, text="Documents split")

            # split the documents into chunks
            chunks = DocumentSplitter.split_documents(documents)
            status_bar.progress(0.75, text="Documents embedded")
            # embed the chunks
            embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
            status_bar.progress(0.9, text="Documents embedded")
            # embed the chunks on the vector database
            if st.session_state.pinecone_db:
                retriever = PineconeVectorDB.embeddings_on_pinecone(embeddings, chunks)
            else:
                retriever = LocalVectorDB.embeddings_on_local_vectordb(
                    embeddings, chunks
                )

            status_bar.progress(100, text="Ready to chat!")

            st.session_state["has_documents_uploaded"] = True
            st.session_state["retriever"] = retriever

        if status_bar:
            time.sleep(1)
            status_bar.empty()

    if st.session_state["has_documents_uploaded"]:
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
                message_placeholder = st.empty()
                response = ""

                # Query the LLM
                response = query_llm(retriever, prompt)

                message_placeholder.markdown(response + " ")

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.chat_history = st.session_state.messages


if __name__ == "__main__":
    page_main()

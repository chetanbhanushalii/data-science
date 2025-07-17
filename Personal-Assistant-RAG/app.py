

# RAG Streamlit App using OpenAI + FAISS (Local Vector Store)

import os
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings # OpenAI API integration -- using OpenAI for both the LLM and text embeddings
from langchain_community.document_loaders import PyPDFLoader # PDF document loader from LangChain Community
from langchain.text_splitter import RecursiveCharacterTextSplitter # Text splitter to break documents into manageable chunks
from langchain.vectorstores import FAISS # FAISS for local vector storage -- FAISS stores embedded document chunks for fast similarity search.
from langchain.chains import RetrievalQA # Retrieval-based QA chain that uses the retriever to find relevant chunks and the LLM to generate answers
from tempfile import NamedTemporaryFile # NamedTemporaryFile for handling temporary files

# Setting up Streamlit page with a wide layout and a title
st.set_page_config(page_title="RAG Assistant", layout="wide")
st.title("RAG-based PDF Q&A App")

# Sidebar for file upload
with st.sidebar:
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"]) # File uploader for PDF documents
    openai_api_key = st.text_input("Enter OpenAI API Key", type="password") # Input for OpenAI API key

# Load and embed document
if uploaded_file and openai_api_key:
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success("PDF uploaded successfully!")
    
    # Load & split
    loader = PyPDFLoader(file_path) # Load PDF document using LangChain's PyPDFLoader
    pages = loader.load_and_split() # Load and split the PDF into pages
    
    # Split documents into manageable chunks (800 tokens with 150 overlap) which helps improve retrieval accuracy.
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150) #
    docs = splitter.split_documents(pages) 

    # Embedding and FAISS vector store
    embeddings = OpenAIEmbeddings(api_key=openai_api_key) # Using OpenAI for text embeddings
    db = FAISS.from_documents(docs, embeddings) # Create FAISS vector store from document chunks
    retriever = db.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 relevant chunks for a query

    # Build QA chain
    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4") # Using OpenAI's GPT-4 model for question answering
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever) # Create a retrieval-based QA chain

    st.subheader("Ask a question about the document") # Input section for user queries
    user_query = st.text_input("Your Question:") # Text input for user question

    if user_query:
        with st.spinner("Thinking..."):
            response = qa_chain.run(user_query)
            st.success("Answer:")
            st.write(response)
else:
    st.info("👈 Upload a PDF and enter your OpenAI key to begin.")


# === File: requirements.txt ===
# streamlit
# langchain
# langchain-openai
# openai
# faiss-cpu
# python-dotenv
# pypdf

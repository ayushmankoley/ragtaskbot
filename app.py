import streamlit as st
import pandas as pd
import os
from langchain_google_genai import GoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Streamlit layout setup
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="centered")
st.title("🔍 RAG Chatbot using Gemini + LangChain")

# Load components only once (cached)
@st.cache_resource(show_spinner="Loading model and preparing documents...")
def setup_rag_pipeline():
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

    # Load CSV and convert to documents
    df = pd.read_csv("organizations-100.csv")
    documents = []
    for _, row in df.iterrows():
        content = " ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
        documents.append(content)

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.create_documents(documents)

    # Embed documents
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)

    # Gemini LLM and prompt
    llm = GoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20")
    template = """
    Use the following context to answer the question. If you don't know the answer based on the context,
    just say you don't know, don't try to make up an answer.

    Context: {context}

    Question: {question}

    Answer:
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain

qa_chain = setup_rag_pipeline()

# Chat interface
user_input = st.chat_input("Ask a question about the organizations dataset...")
if user_input:
    st.chat_message("user").markdown(user_input)

    with st.chat_message("assistant"):
        response = qa_chain.invoke(user_input)
        st.markdown(response['result'])

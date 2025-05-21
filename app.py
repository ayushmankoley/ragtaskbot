# Import libraries
import pandas as pd
import os
from langchain_google_genai import GoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Set up Gemini API
os.environ["GOOGLE_API_KEY"] = "AIzaSyDomENrYIfkA5bsnQFIyNoWGpFe0clRl20"

# Data Loading
df = pd.read_csv('/content/organizations-100.csv')
documents = []
for i, row in df.iterrows():
    content = " ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
    documents.append(content)

# Text Splitting
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
chunks = text_splitter.create_documents(documents)

# Vector Store Setup
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)

# RAG Pipeline Setup
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

# Chatbot Interface
print("RAG Chatbot ready. Type 'exit' to end the conversation.")
while True:
    user_input = input("\nQuestion: ")
    if user_input.lower() == 'exit':
        print("Goodbye!")
        break
    response = qa_chain.invoke(user_input)
    print("\nAnswer:", response['result'])
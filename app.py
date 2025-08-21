import os
from transformers import pipeline
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
import streamlit as st

# =============================
# Setup LLM
# =============================
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",   # or flan-t5-base if memory issues
    device=-1
)
llm = HuggingFacePipeline(pipeline=pipe)

# =============================
# PDF Upload & Vectorstore
# =============================
@st.cache_resource
def load_vectorstore(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# =============================
# Product Finder Function
# =============================
def product_finder(query, vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})
    docs_found = retriever.get_relevant_documents(query)
    context = docs_found[0].page_content if docs_found else "No context found."
    
    template = """
You are an AI assistant that extracts PRODUCT INFORMATION.

Given the following product specifications:

{context}

And the user query:
{query}

Identify the **most relevant product** and return the details in JSON format:
{{
  "Application": "...",
  "Wattage": "...",
  "Luminous Flux": "..."
}}
If information is missing, put "Not specified".
"""
    prompt = PromptTemplate(input_variables=["context", "query"], template=template)
    final_prompt = prompt.format(context=context, query=query)
    return llm(final_prompt)

# =============================
# Streamlit UI
# =============================
st.title("💡 Product Finder AI")
st.write("Upload a product catalog PDF and search for products by application, wattage, luminous flux.")

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    
    st.success("PDF uploaded and processed ✅")
    vectorstore = load_vectorstore("temp.pdf")
    
    query = st.text_input("Enter your product search query")
    
    if st.button("Search") and query:
        result = product_finder(query, vectorstore)
        st.json(result)

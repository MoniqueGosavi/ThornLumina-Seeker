# =============================================================================
# FILE: app.py (Main Streamlit Application)
# =============================================================================

import os
import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
import json
import tempfile
import warnings
warnings.filterwarnings("ignore")

# Configure page
st.set_page_config(
    page_title="ThornLumina Seeker",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 3rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .main-title {
        font-size: 4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #fbbf24, #f59e0b, #d97706);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        text-shadow: 0 0 30px rgba(251, 191, 36, 0.3);
    }
    
    .main-subtitle {
        font-size: 1.4rem;
        color: #cbd5e1;
        font-weight: 400;
        opacity: 0.9;
    }
    
    .feature-container {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(15px);
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    .upload-section {
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.1), rgba(245, 158, 11, 0.1));
        border: 2px dashed #fbbf24;
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        margin: 2rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.15), rgba(245, 158, 11, 0.15));
        border-color: #f59e0b;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #fbbf24, #f59e0b);
        color: #0f0f23;
        border: none;
        border-radius: 12px;
        padding: 1rem 2.5rem;
        font-weight: 700;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(251, 191, 36, 0.4);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(251, 191, 36, 0.6);
    }
    
    .search-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .result-container {
        background: rgba(15, 15, 35, 0.8);
        border: 1px solid #374151;
        border-radius: 16px;
        padding: 2rem;
        margin-top: 2rem;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        color: #e2e8f0;
        font-size: 1.1rem;
        padding: 0.8rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #fbbf24;
        box-shadow: 0 0 0 2px rgba(251, 191, 36, 0.2);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Section headers */
    .section-title {
        font-size: 1.8rem;
        font-weight: 600;
        color: #fbbf24;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================
# Setup LLM with error handling
# =============================
@st.cache_resource
def initialize_llm():
    try:
        with st.spinner("🧠 Loading AI models..."):
            pipe = pipeline(
                "text2text-generation",
                model="google/flan-t5-base",
                device=-1,
                max_length=512,
                temperature=0.3,
                do_sample=True
            )
            return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        st.error(f"❌ Error initializing LLM: {e}")
        return None

# =============================
# PDF Upload & Vectorstore
# =============================
@st.cache_resource
def load_vectorstore(pdf_content, filename):
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("📄 Reading PDF content...")
        progress_bar.progress(25)
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_content)
            tmp_path = tmp_file.name
        
        # Load and process PDF
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        
        status_text.text("✂️ Splitting documents into chunks...")
        progress_bar.progress(50)
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        # Split documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        docs = splitter.split_documents(documents)
        
        status_text.text("🧮 Creating embeddings...")
        progress_bar.progress(75)
        
        # Create embeddings and vectorstore
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = FAISS.from_documents(docs, embeddings)
        
        status_text.text("✅ Search index ready!")
        progress_bar.progress(100)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return vectorstore, len(docs), len(documents)
    
    except Exception as e:
        st.error(f"❌ Error processing PDF: {e}")
        return None, 0, 0

# =============================
# Product Finder Function
# =============================
def product_finder(query, vectorstore, llm):
    try:
        retriever = vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 5}
        )
        docs_found = retriever.get_relevant_documents(query)
        
        if not docs_found:
            return {"error": "No relevant products found in the catalog"}
        
        # Combine context from multiple chunks
        context = "\n\n".join([doc.page_content for doc in docs_found[:3]])
        
        template = """
You are ThornLumina's AI product specialist. Analyze the product catalog and find the best match.

Product Catalog Context:
{context}

Customer Query: {query}

Find the most relevant lighting product and return ONLY valid JSON:
{{
  "product_name": "exact product name from catalog",
  "application": "intended use case",
  "wattage": "power consumption",
  "luminous_flux": "light output in lumens",
  "voltage": "operating voltage",
  "additional_features": "special characteristics",
  "recommendation_score": "confidence level (1-10)"
}}

Use "Not specified" for missing information.
"""
        
        prompt = PromptTemplate(
            input_variables=["context", "query"], 
            template=template
        )
        final_prompt = prompt.format(context=context, query=query)
        
        result = llm(final_prompt)
        
        # Clean and parse result
        result = result.strip()
        if result.startswith('```'):
            result = result.split('```')[1].split('```')[0]
        if result.startswith('json'):
            result = result[4:].strip()
        
        try:
            parsed_result = json.loads(result)
            return parsed_result
        except json.JSONDecodeError:
            return {
                "raw_response": result,
                "context_chunks": len(docs_found),
                "note": "AI response format needs adjustment"
            }
            
    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}

# =============================
# Main UI
# =============================

# Header
st.markdown("""
<div class="main-header">
    <h1 class="main-title">💡 ThornLumina Seeker</h1>
    <p class="main-subtitle">AI-Powered Lighting Product Discovery Engine</p>
</div>
""", unsafe_allow_html=True)

# Initialize LLM
llm = initialize_llm()

if llm is None:
    st.error("❌ Failed to initialize AI models. Please refresh the page.")
    st.stop()

# Features overview
st.markdown("""
<div class="feature-container">
    <h3 class="section-title">🌟 Features</h3>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem; margin-top: 1.5rem;">
        <div style="background: rgba(251, 191, 36, 0.1); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(251, 191, 36, 0.2);">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">📄</div>
            <strong style="color: #fbbf24;">PDF Catalog Processing</strong><br>
            <span style="color: #cbd5e1; font-size: 0.9rem;">Upload lighting catalogs, datasheets, and specifications</span>
        </div>
        <div style="background: rgba(251, 191, 36, 0.1); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(251, 191, 36, 0.2);">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔍</div>
            <strong style="color: #fbbf24;">Natural Language Search</strong><br>
            <span style="color: #cbd5e1; font-size: 0.9rem;">Ask questions like "20W LED for office use"</span>
        </div>
        <div style="background: rgba(251, 191, 36, 0.1); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(251, 191, 36, 0.2);">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">⚡</div>
            <strong style="color: #fbbf24;">Instant Results</strong><br>
            <span style="color: #cbd5e1; font-size: 0.9rem;">Fast AI-powered product matching with specs</span>
        </div>
        <div style="background: rgba(251, 191, 36, 0.1); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(251, 191, 36, 0.2);">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">📊</div>
            <strong style="color: #fbbf24;">Structured Output</strong><br>
            <span style="color: #cbd5e1; font-size: 0.9rem;">Clean JSON with all product specifications</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# File upload section
st.markdown("""
<div class="upload-section">
    <div style="font-size: 3rem; margin-bottom: 1rem;">📂</div>
    <h3 style="color: #fbbf24; margin-bottom: 1rem; font-size: 1.8rem;">Upload Lighting Catalog</h3>
    <p style="color: #94a3b8; font-size: 1.1rem;">Upload PDF catalogs containing lighting products, specifications, and datasheets</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
    help="Upload lighting catalogs, product datasheets, or specification documents"
)

# Process uploaded file
if uploaded_file is not None:
    st.markdown("""
    <div class="feature-container">
        <h3 class="section-title">🔄 Processing Catalog</h3>
    </div>
    """, unsafe_allow_html=True)
    
    pdf_content = uploaded_file.read()
    vectorstore, num_chunks, num_pages = load_vectorstore(pdf_content, uploaded_file.name)
    
    if vectorstore:
        st.success(f"✅ **{uploaded_file.name}** processed successfully!")
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 Pages Processed", num_pages)
        with col2:
            st.metric("🧩 Text Chunks", num_chunks)
        with col3:
            st.metric("🔍 Search Ready", "Yes" if vectorstore else "No")
        
        # Search interface
        st.markdown("""
        <div class="feature-container">
            <h3 class="section-title">🔍 Search Products</h3>
            <div class="search-container">
        """, unsafe_allow_html=True)
        
        # Search input
        col1, col2 = st.columns([4, 1])
        with col1:
            query = st.text_input(
                "🔎 Enter your search query",
                placeholder="e.g., LED bulb 15W for office, outdoor floodlight 2000 lumens, energy efficient lamp",
                help="Describe the lighting product you're looking for using natural language"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Spacing
            search_clicked = st.button("🚀 SEARCH", use_container_width=True)
        
        # Search examples
        st.markdown("""
        <div style="background: rgba(251, 191, 36, 0.1); border-radius: 12px; padding: 1.5rem; margin: 1.5rem 0; border: 1px solid rgba(251, 191, 36, 0.2);">
            <strong style="color: #fbbf24; font-size: 1.1rem;">💡 Example Searches:</strong><br><br>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; margin-top: 1rem;">
                <span style="color: #cbd5e1;">• "20W LED bulb for residential lighting"</span>
                <span style="color: #cbd5e1;">• "High efficiency fluorescent lamp 4000K"</span>
                <span style="color: #cbd5e1;">• "Outdoor floodlight with 3000 lumens"</span>
                <span style="color: #cbd5e1;">• "Energy saving bulb for office space"</span>
                <span style="color: #cbd5e1;">• "Dimmable LED with warm white light"</span>
                <span style="color: #cbd5e1;">• "Industrial lighting fixture IP65 rated"</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div></div>", unsafe_allow_html=True)
        
        # Search results
        if search_clicked and query:
            st.markdown("""
            <div class="feature-container">
                <h3 class="section-title">📋 Search Results</h3>
            </div>
            """, unsafe_allow_html=True)
            
            with st.spinner("🤖 ThornLumina AI is analyzing your catalog..."):
                result = product_finder(query, vectorstore, llm)
            
            if "error" in result:
                st.error(f"❌ {result['error']}")
            else:
                # Display results beautifully
                if isinstance(result, dict) and "raw_response" not in result:
                    # Clean structured result
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown("""
                        <div class="result-container">
                            <h4 style="color: #fbbf24; margin-bottom: 1.5rem;">🎯 Product Match</h4>
                        """, unsafe_allow_html=True)
                        
                        for key, value in result.items():
                            if value and value != "Not specified":
                                icon = "💡" if "name" in key else "⚡" if "watt" in key else "🌟" if "lumen" in key else "🔧"
                                st.markdown(f"""
                                <div style="margin-bottom: 1rem; padding: 0.8rem; background: rgba(251, 191, 36, 0.1); border-radius: 8px;">
                                    <strong style="color: #fbbf24;">{icon} {key.replace('_', ' ').title()}:</strong><br>
                                    <span style="color: #e2e8f0; font-size: 1.1rem;">{value}</span>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("""
                        <div class="result-container">
                            <h4 style="color: #fbbf24; margin-bottom: 1.5rem;">📄 Raw JSON Output</h4>
                        """, unsafe_allow_html=True)
                        st.json(result)
                        st.markdown("</div>", unsafe_allow_html=True)
                
                else:
                    # Fallback display for raw responses
                    st.markdown("""
                    <div class="result-container">
                        <h4 style="color: #fbbf24; margin-bottom: 1.5rem;">🤖 AI Response</h4>
                    """, unsafe_allow_html=True)
                    
                    if "raw_response" in result:
                        st.code(result["raw_response"], language="json")
                    else:
                        st.json(result)
                    
                    st.markdown("</div>", unsafe_allow_html=True)

else:
    # Welcome message when no file uploaded
    st.markdown("""
    <div class="feature-container">
        <div style="text-align: center; padding: 2rem;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🌟</div>
            <h3 style="color: #fbbf24; margin-bottom: 1rem;">Welcome to ThornLumina Seeker</h3>
            <p style="color: #cbd5e1; font-size: 1.2rem; line-height: 1.6; max-width: 600px; margin: 0 auto;">
                Upload your lighting product catalog to unlock the power of AI-driven product discovery. 
                Our advanced search engine understands natural language queries and finds the perfect lighting solutions.
            </p>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1.5rem; margin-top: 3rem; max-width: 800px; margin-left: auto; margin-right: auto;">
                <div style="background: rgba(251, 191, 36, 0.1); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(251, 191, 36, 0.2);">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">🏢</div>
                    <strong style="color: #fbbf24;">Commercial</strong><br>
                    <span style="color: #94a3b8; font-size: 0.9rem;">Office & retail lighting</span>
                </div>
                <div style="background: rgba(251, 191, 36, 0.1); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(251, 191, 36, 0.2);">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">🏠</div>
                    <strong style="color: #fbbf24;">Residential</strong><br>
                    <span style="color: #94a3b8; font-size: 0.9rem;">Home lighting solutions</span>
                </div>
                <div style="background: rgba(251, 191, 36, 0.1); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(251, 191, 36, 0.2);">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">🌃</div>
                    <strong style="color: #fbbf24;">Outdoor</strong><br>
                    <span style="color: #94a3b8; font-size: 0.9rem;">Street & landscape lighting</span>
                </div>
                <div style="background: rgba(251, 191, 36, 0.1); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(251, 191, 36, 0.2);">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">🏭</div>
                    <strong style="color: #fbbf24;">Industrial</strong><br>
                    <span style="color: #94a3b8; font-size: 0.9rem;">Heavy-duty lighting</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Sidebar with instructions
with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem;">
        <h3 style="color: #fbbf24;">📖 How to Use</h3>
        
        <div style="background: rgba(255, 255, 255, 0.05); padding: 1rem; border-radius: 10px; margin: 1rem 0;">
            <strong style="color: #cbd5e1;">Step 1:</strong><br>
            <span style="color: #94a3b8;">Upload your PDF catalog</span>
        </div>
        
        <div style="background: rgba(255, 255, 255, 0.05); padding: 1rem; border-radius: 10px; margin: 1rem 0;">
            <strong style="color: #cbd5e1;">Step 2:</strong><br>
            <span style="color: #94a3b8;">Enter your search query</span>
        </div>
        
        <div style="background: rgba(255, 255, 255, 0.05); padding: 1rem; border-radius: 10px; margin: 1rem 0;">
            <strong style="color: #cbd5e1;">Step 3:</strong><br>
            <span style="color: #94a3b8;">Get AI-powered results</span>
        </div>
        
        <hr style="border-color: rgba(255, 255, 255, 0.1); margin: 2rem 0;">
        
        <h4 style="color: #fbbf24;">💡 Search Tips</h4>
        <ul style="color: #94a3b8; font-size: 0.9rem;">
            <li>Include wattage for power requirements</li>
            <li>Specify application (office, home, outdoor)</li>
            <li>Mention lumens for brightness needs</li>
            <li>Include color temperature if important</li>
            <li>Specify mounting type if relevant</li>
        </ul>
        
        <hr style="border-color: rgba(255, 255, 255, 0.1); margin: 2rem 0;">
        
        <div style="text-align: center; color: #64748b; font-size: 0.8rem;">
            <p>🚀 <strong>ThornLumina Seeker v1.0</strong></p>
            <p>Powered by AI & Machine Learning</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 4rem; padding: 2rem; border-top: 1px solid rgba(255, 255, 255, 0.1);">
    <p style="color: #64748b; margin: 0; font-size: 1rem;">
        Built with ❤️ using <strong>Streamlit</strong>, <strong>LangChain</strong> & <strong>Hugging Face</strong>
    </p>
    <p style="color: #475569; margin: 0.5rem 0 0 0; font-size: 0.9rem;">
        ThornLumina Seeker - Illuminating Product Discovery
    </p>
</div>
""", unsafe_allow_html=True)


       

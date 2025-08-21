import streamlit as st
import tempfile
import os
import json
import PyPDF2
import re

# Configure page
st.set_page_config(
    page_title="ThornLumina Seeker",
    page_icon="💡",
    layout="wide"
)

# Custom CSS
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
    }
    
    .main-subtitle {
        font-size: 1.4rem;
        color: #cbd5e1;
        font-weight: 400;
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
    
    .stButton > button {
        background: linear-gradient(135deg, #fbbf24, #f59e0b);
        color: #0f0f23;
        border: none;
        border-radius: 12px;
        padding: 1rem 2.5rem;
        font-weight: 700;
        font-size: 1.1rem;
        box-shadow: 0 4px 20px rgba(251, 191, 36, 0.4);
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

# Simple PDF text extraction function
def extract_text_from_pdf(pdf_file):
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_path = tmp_file.name
        
        # Extract text using PyPDF2
        text = ""
        with open(tmp_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        
        # Clean up temp file
        os.unlink(tmp_path)
        return text
    
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

# Simple product search function
def simple_product_search(query, pdf_text):
    try:
        # Simple keyword-based search
        query_lower = query.lower()
        text_lower = pdf_text.lower()
        
        # Extract product information using regex patterns
        patterns = {
            "wattage": r'(\d+\.?\d*)\s*w(?:att)?',
            "lumens": r'(\d+)\s*(?:lm|lumen)',
            "voltage": r'(\d+(?:-\d+)?)\s*v(?:olt)?',
            "application": r'(?:for|application|use)[\s:]+([\w\s]+?)(?:\.|,|\n)',
        }
        
        results = {}
        
        # Find product mentions near the query terms
        lines = pdf_text.split('\n')
        relevant_lines = []
        
        for line in lines:
            if any(word in line.lower() for word in query_lower.split()):
                relevant_lines.extend([line] + lines[max(0, lines.index(line)-1):lines.index(line)+2])
        
        relevant_text = ' '.join(relevant_lines)
        
        # Extract specifications
        for spec, pattern in patterns.items():
            match = re.search(pattern, relevant_text, re.IGNORECASE)
            if match:
                results[spec] = match.group(1) if len(match.groups()) > 0 else match.group(0)
            else:
                results[spec] = "Not specified"
        
        # Try to find product name
        product_lines = [line.strip() for line in relevant_lines if line.strip() and len(line.strip()) < 100]
        results["product_name"] = product_lines[0] if product_lines else "Product found"
        results["query_matched"] = query
        results["confidence"] = "Medium" if len(relevant_lines) > 2 else "Low"
        
        return results
    
    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}

# Main UI
st.markdown("""
<div class="main-header">
    <h1 class="main-title">💡 ThornLumina Seeker</h1>
    <p class="main-subtitle">AI-Powered Lighting Product Discovery Engine</p>
</div>
""", unsafe_allow_html=True)

# Features
st.markdown("""
<div class="feature-container">
    <h3 style="color: #fbbf24; margin-bottom: 1rem;">🌟 Features</h3>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem;">
        <div style="background: rgba(251, 191, 36, 0.1); padding: 1.5rem; border-radius: 12px;">
            <strong style="color: #fbbf24;">📄 PDF Processing</strong><br>
            <span style="color: #cbd5e1;">Upload lighting catalogs and datasheets</span>
        </div>
        <div style="background: rgba(251, 191, 36, 0.1); padding: 1.5rem; border-radius: 12px;">
            <strong style="color: #fbbf24;">🔍 Smart Search</strong><br>
            <span style="color: #cbd5e1;">Natural language product queries</span>
        </div>
        <div style="background: rgba(251, 191, 36, 0.1); padding: 1.5rem; border-radius: 12px;">
            <strong style="color: #fbbf24;">⚡ Instant Results</strong><br>
            <span style="color: #cbd5e1;">Fast product specification extraction</span>
        </div>
        <div style="background: rgba(251, 191, 36, 0.1); padding: 1.5rem; border-radius: 12px;">
            <strong style="color: #fbbf24;">📊 Clean Output</strong><br>
            <span style="color: #cbd5e1;">Structured JSON results</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# File upload
st.markdown("""
<div style="background: linear-gradient(135deg, rgba(251, 191, 36, 0.1), rgba(245, 158, 11, 0.1)); 
     border: 2px dashed #fbbf24; border-radius: 20px; padding: 2rem; text-align: center; margin: 2rem 0;">
    <h3 style="color: #fbbf24;">📂 Upload Product Catalog</h3>
    <p style="color: #94a3b8;">Upload PDF containing lighting products and specifications</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Process PDF
    with st.spinner("📄 Processing PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_file)
    
    if pdf_text:
        st.success(f"✅ Successfully processed: {uploaded_file.name}")
        
        # Show file stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 File Size", f"{len(uploaded_file.getvalue())/1024:.1f} KB")
        with col2:
            st.metric("📝 Characters", f"{len(pdf_text):,}")
        with col3:
            st.metric("📋 Lines", f"{len(pdf_text.splitlines()):,}")
        
        # Search interface
        st.markdown("""
        <div class="feature-container">
            <h3 style="color: #fbbf24; margin-bottom: 1rem;">🔍 Search Products</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([4, 1])
        with col1:
            query = st.text_input(
                "Enter your search query",
                placeholder="e.g., 20W LED bulb, outdoor floodlight, office lighting"
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            search_clicked = st.button("🔍 SEARCH", use_container_width=True)
        
        # Example queries
        st.markdown("""
        <div style="background: rgba(251, 191, 36, 0.1); border-radius: 12px; padding: 1rem; margin: 1rem 0;">
            <strong style="color: #fbbf24;">💡 Try these examples:</strong><br>
            <span style="color: #cbd5e1;">
            • "20W LED" • "outdoor lighting" • "high efficiency" • "office bulb" • "2000 lumens"
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        # Search results
        if search_clicked and query:
            with st.spinner("🔍 Searching catalog..."):
                result = simple_product_search(query, pdf_text)
            
            st.markdown("""
            <div class="feature-container">
                <h3 style="color: #fbbf24; margin-bottom: 1rem;">📋 Search Results</h3>
            </div>
            """, unsafe_allow_html=True)
            
            if "error" in result:
                st.error(f"❌ {result['error']}")
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🎯 Product Information:**")
                    for key, value in result.items():
                        if key != "query_matched":
                            st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                
                with col2:
                    st.markdown("**📄 Raw JSON:**")
                    st.json(result)

else:
    # Welcome screen
    st.markdown("""
    <div class="feature-container">
        <div style="text-align: center; padding: 2rem;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🌟</div>
            <h3 style="color: #fbbf24;">Welcome to ThornLumina Seeker</h3>
            <p style="color: #cbd5e1; font-size: 1.2rem; max-width: 600px; margin: 0 auto;">
                Upload your lighting product catalog to start searching for products using natural language.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 3rem; padding: 2rem; color: #64748b;">
    <p>🚀 <strong>ThornLumina Seeker v1.0</strong> - Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)

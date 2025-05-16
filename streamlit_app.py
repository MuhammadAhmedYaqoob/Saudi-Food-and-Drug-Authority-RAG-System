import streamlit as st
from src.config import RAGS_DIR
from src.loader import load_multiple_pdfs
from src.retriever import build_faiss, query_semantic
from src.generator import answer
from src.graph_indexer import build_graph
import os
from pathlib import Path

st.set_page_config(page_title="Saudi Food and Drug Authority Assistant", layout="centered")

# Sidebar for administration
with st.sidebar:
    st.header("Administration")
    
    st.subheader("Knowledge Base")
    
    # Check if directory exists
    if not RAGS_DIR.exists():
        st.warning(f"Directory not found: {RAGS_DIR}")
        if st.button("Create Directory"):
            RAGS_DIR.mkdir(exist_ok=True)
            st.success(f"Created directory: {RAGS_DIR}")
    
    # List PDF files
    pdf_files = list(RAGS_DIR.glob("*.pdf")) if RAGS_DIR.exists() else []
    if pdf_files:
        st.success(f"Found {len(pdf_files)} PDF files:")
        for pdf in pdf_files:
            st.info(f"• {pdf.name}")
    else:
        st.error(f"No PDF files found in {RAGS_DIR}")
        st.info("Please upload PDF files to the 'rags' directory")
    
    st.subheader("Index Management")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Build Graph Index"):
            with st.spinner("Building knowledge graph..."):
                build_graph()
            st.success("Graph index built!")
    
    with col2:
        if st.button("Build FAISS Index"):
            with st.spinner("Building semantic index..."):
                build_faiss()
            st.success("FAISS index built!")

# Main interface
def main():
    # Header with branding
    st.title("🇸🇦 Saudi Food and Drug Authority Assistant")
    st.markdown("""
    Ask questions about food, medications, treatments, and regulations from the Saudi Food and Drug Authority's documentation.
    """)
    
    # Language toggle
    lang_col1, lang_col2 = st.columns(2)
    with lang_col1:
        st.write("Select Language / اختر اللغة")
    with lang_col2:
        language = st.radio("", ["English", "العربية"], horizontal=True, label_visibility="collapsed")
    
    # Query input
    if language == "English":
        query_placeholder = "Enter your question about food, drugs, or regulations..."
        search_text = "Search" 
    else:  # Arabic
        query_placeholder = "أدخل سؤالك حول الغذاء أو الدواء أو اللوائح..."
        search_text = "بحث"
    
    query = st.text_input("", placeholder=query_placeholder, label_visibility="collapsed")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search_button = st.button(search_text, type="primary")
    with col2:
        k = st.slider("", min_value=1, max_value=10, value=5, label_visibility="collapsed")
    
    if search_button and query:
        with st.spinner("Searching knowledge base..."):
            # Retrieve relevant contexts
            contexts = query_semantic(query, k=k)
            
            if not contexts:
                if language == "English":
                    st.warning("No relevant information found. Please try a different question or rebuild the indices.")
                else:
                    st.warning("لم يتم العثور على معلومات ذات صلة. يرجى تجربة سؤال مختلف أو إعادة بناء الفهارس.")
                return
            
            # Generate answer
            with st.spinner("Generating answer..."):
                response = answer(query, contexts)
        
        # Display answer
        st.markdown("### Answer / الإجابة")
        st.markdown(response)
        
        # Show debug info (can be removed in production)
        with st.expander("Debug Information (Admin Only)", expanded=False):
            st.subheader("Retrieved Contexts")
            for i, doc in enumerate(contexts):
                st.markdown(f"### Source {i+1}: [{doc.get('source_file', 'unknown')}] - Page {doc.get('page', 'N/A')}")
                st.info(f"Language: {doc.get('language', 'unknown')}, Score: {doc.get('score', 'N/A')}")
                
                if doc.get("is_table", False):
                    st.markdown("**Table Content:**")
                    rows = doc["text"].split("\n")
                    for row in rows:
                        st.text(row)
                else:
                    st.markdown(doc["text"])
                
                st.markdown("---")

if __name__ == "__main__":
    main()
# Saudi Food and Drug Authority RAG System

A bilingual (Arabic/English) RAG (Retrieval-Augmented Generation) chatbot system for the Saudi Food and Drug Authority. This system combines miniRAG's heterogeneous graph-based retrieval with coRAG-style generation to provide accurate answers from multiple source documents.

## Features

- **Bilingual Support**: Full Arabic and English language processing 
- **Multiple Document Sources**: Processes all PDFs from a designated folder
- **Heterogeneous Knowledge Graph**: Creates cross-document and cross-language connections
- **Advanced Retrieval**: Uses semantic search enhanced with graph-based heuristics
- **Concise Responses**: Generates direct, citation-free answers in the query's language
- **Interactive UI**: Streamlit interface with language switching support
- **Graph Visualization**: View entity relationships and document connections

## Setup Instructions

### 1. Environment Setup

- Python version 3.10.7 (used during development)

```bash
# Clone repository (or download files)
git clone https://github.com/yourusername/sfda-rag.git
cd sfda-rag

# Create and activate virtual environment
python -m venv .venv
# For Windows
.venv\Scripts\activate
# For macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install SpaCy language models
python -m spacy download en_core_web_sm
python -m spacy download xx_ent_wiki_sm
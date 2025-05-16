from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    print("[WARNING] OpenAI API key not found. Set OPENAI_API_KEY in your environment or .env file.")

# PDFs directory (multiple files)
RAGS_DIR = Path("rags")

# Directory structure
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
GRAPH_DIR = DATA_DIR / "graphs"
FAISS_DIR = DATA_DIR / "faiss"

# OpenAI Models
OPENAI_EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536  # Dimensions for OpenAI embeddings

# Generation settings
LLM_MODEL = "gpt-4.1-mini"
MAX_TOKENS = 256
TEMPERATURE = 0.1

# Language settings
SUPPORTED_LANGUAGES = ["en", "ar"]  # English and Arabic
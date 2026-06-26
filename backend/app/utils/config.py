"""
config.py — Application-wide configuration
============================================
Single source of truth for all filesystem paths and shared constants.
Import from here instead of resolving paths locally in each module.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load env variables from root folder
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

# ---------------------------------------------------------------------------
# Host and Ports
# ---------------------------------------------------------------------------
FASTAPI_HOST = os.getenv("FASTAPI_HOST", "localhost")
FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", "8000"))

CHROMA_HOST = os.getenv("CHROMA_HOST")
CHROMA_PORT = os.getenv("CHROMA_PORT")
if CHROMA_PORT:
    CHROMA_PORT = int(CHROMA_PORT)

# ---------------------------------------------------------------------------
# Root paths
# ---------------------------------------------------------------------------
# This file lives at backend/app/config.py, so:
APP_DIR = Path(__file__).resolve().parent.parent  # backend/app/
BACKEND_DIR = APP_DIR.parent  # backend/

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
DATA_DIR = BACKEND_DIR / "data"  # backend/data/
PDF_DIR = DATA_DIR / "pdf"  # backend/data/pdf/  — place PDFs here
CHROMA_DIR = (
    DATA_DIR / "chroma_db"
)  # backend/data/chroma_db/  (chroma.sqlite3 lives here)

# ---------------------------------------------------------------------------
# Chroma / vector store
# ---------------------------------------------------------------------------
COLLECTION_NAME = "bank_terms"

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
LLM_REPO_ID = "openai/gpt-oss-120b"
LLM_TOP_K = int(os.getenv("LLM_TOP_K", "5"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
LLM_STREAM = os.getenv("LLM_STREAM", "False").lower() in ("true", "1", "t")
LLM_MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "2048"))

# ---------------------------------------------------------------------------
# PostgreSQL
# ---------------------------------------------------------------------------
POSTGRE_DB_URI = os.getenv("POSTGRE_DB_URI")

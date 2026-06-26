"""
store.py — Document ingestion script
=====================================
Loads every PDF found in `DATA_DIR`, splits them into chunks, embeds them
using HuggingFace, and persists them into the local Chroma SQLite database
at `CHROMA_DIR`.

Usage (from backend/app/):
    python -m rag.store
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma

from utils.config import CHROMA_DIR, COLLECTION_NAME, PDF_DIR
from rag.loader import load_pdf
from rag.chunking import recursive_split
from rag.embedder import get_embeddings

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def collect_pdfs(directory: str | Path) -> list[str]:
    """Return a sorted list of PDF paths inside *directory*."""
    pdf_dir = Path(directory)
    if not pdf_dir.exists():
        raise FileNotFoundError(
            f"PDF source directory not found: {pdf_dir}\n"
            "Create it and place your PDF files inside."
        )
    pdfs = sorted(pdf_dir.glob("**/*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found under {pdf_dir}")
    return [str(p) for p in pdfs]


def build_local_store(embeddings) -> Chroma:
    """Open (or create) the persistent Chroma collection."""
    import chromadb
    from utils.config import CHROMA_HOST, CHROMA_PORT

    if CHROMA_HOST and CHROMA_PORT:
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        return Chroma(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
        )
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )


# ---------------------------------------------------------------------------
# Main ingestion pipeline
# ---------------------------------------------------------------------------


def run_ingestion(
    pdf_dir: Path = PDF_DIR,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> None:
    """
    Full ingestion pipeline:
        1. Collect all PDFs from *pdf_dir*
        2. Load each PDF into LangChain Documents
        3. Split documents into chunks
        4. Embed and persist into Chroma (SQLite)
    """
    logger.info("=== Starting ingestion ===")
    logger.info("Source directory : %s", pdf_dir)
    logger.info("Chroma directory : %s", CHROMA_DIR)

    # 1. Discover PDFs
    pdf_paths = collect_pdfs(pdf_dir)
    logger.info("Found %d PDF file(s).", len(pdf_paths))

    # 2. Load all PDFs
    all_docs = []
    for path in pdf_paths:
        logger.info("Loading: %s", os.path.basename(path))
        pages = load_pdf(path)
        all_docs.extend(pages)
    logger.info("Total pages loaded: %d", len(all_docs))

    # 3. Chunk
    logger.info(
        "Splitting into chunks (size=%d, overlap=%d)...", chunk_size, chunk_overlap
    )
    chunks = recursive_split(all_docs, chunk_size=chunk_size, overlap=chunk_overlap)
    logger.info("Total chunks produced: %d", len(chunks))

    # 4. Embed & persist
    logger.info("Embedding and writing to Chroma at %s ...", CHROMA_DIR)
    embeddings = get_embeddings()
    store = build_local_store(embeddings)
    store.add_documents(chunks)
    logger.info("=== Ingestion complete — %d chunks stored. ===", len(chunks))


if __name__ == "__main__":
    run_ingestion()

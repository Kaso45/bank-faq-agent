import logging
from langchain_community.document_loaders import PyPDFLoader

logger = logging.getLogger(__name__)


def load_pdf(file_path: str):
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        logger.info(f"Successfully load {len(pages)}")
        return pages
    except Exception as e:
        logger.error(f"Error loading PDF: {e}")
        raise

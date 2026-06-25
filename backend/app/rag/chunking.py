import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def recursive_split(docs: list[Document], chunk_size: int = 500, overlap: int = 100):
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=overlap
        )
        texts = splitter.split_documents(docs)
        logger.info(f"Successfully split {len(texts)} chunks.")
        return texts
    except Exception as e:
        logger.error(f"Error while chunking: {e}")
        raise

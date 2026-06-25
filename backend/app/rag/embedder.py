import logging
import os
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

from utils.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)
load_dotenv()


def get_embeddings():
    if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        raise EnvironmentError(
            "HUGGINGFACEHUB_API_TOKEN is not set. "
            "Add it to your .env file before starting the server."
        )
    try:
        embeddings = HuggingFaceEndpointEmbeddings(model=EMBEDDING_MODEL)
        logger.info("Successfully loaded embeddings.")
        return embeddings
    except Exception as e:
        logger.error(f"Error getting embeddings: {e}")
        raise

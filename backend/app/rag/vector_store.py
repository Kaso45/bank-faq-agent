import logging
from uuid import uuid4
from langchain_chroma import Chroma
from langchain_core.documents import Document

from utils.config import CHROMA_DIR, CHROMA_HOST, CHROMA_PORT

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, collection_name: str, embeddings) -> None:
        import chromadb

        if CHROMA_HOST and CHROMA_PORT:
            client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            self.store = Chroma(
                client=client,
                collection_name=collection_name,
                embedding_function=embeddings,
            )
        else:
            self.store = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=CHROMA_DIR,
            )

    def add_docs(self, documents: list[Document]):
        try:
            uuids = [str(uuid4()) for _ in range(len(documents))]
            self.store.add_documents(documents=documents, ids=uuids)
        except Exception as e:
            logger.exception("Error adding documents: %s", e)
            raise

    def update_docs(self, documents: list[Document], document_ids: list[str]):
        try:
            self.store.update_documents(ids=document_ids, documents=documents)
        except Exception as e:
            logger.exception("Error updating documents: %s", e)
            raise

    def delete_docs(self, documents_ids: list[str]):
        try:
            self.store.delete(ids=documents_ids)
        except Exception as e:
            logger.exception("Error deleting documents: %s", e)
            raise

    def search(self, user_query: str, top_k: int = 10):
        try:
            results = self.store.similarity_search(user_query, k=top_k)
            return results
        except Exception as e:
            logger.exception("Error searching documents: %s", e)
            raise

    def get_retriever(self, search_type: str = "similarity", top_k: int = 10):
        """Transform vector store into retriever

        Args:
            search_type (str, optional): search type that the Retriever should perform. Can be 'similarity' (default), 'mmr', or 'similarity_score_threshold'
            top_k (int, optional): _description_. Defaults to 10.

        Returns:
            _type_: _description_
        """
        return self.store.as_retriever(
            search_type=search_type, search_kwargs={"k": top_k}
        )

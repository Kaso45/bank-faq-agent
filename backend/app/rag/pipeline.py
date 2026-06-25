from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace

from rag.vector_store import VectorStore


def build_rag_chain(
    llm: ChatHuggingFace,
    vector_store: VectorStore,
    prompt: ChatPromptTemplate,
    search_type: str = "similarity",
    top_k: int = 10,
):
    """Assemble the full RAG retrieval chain.

    Args:
        llm: A LangChain-compatible LLM (e.g. from llm/llm.py)
        vector_store: Your VectorStore instance
        prompt: A ChatPromptTemplate with {context} and {input} variables

    Returns:
        A runnable chain that accepts {"input": user_query}
    """
    retriever = vector_store.get_retriever(search_type=search_type, top_k=top_k)
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)

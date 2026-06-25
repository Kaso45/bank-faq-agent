"""
knowledge_base_search tool — searches the internal Seven Bank knowledge base
(backed by the RAG vector store) for policy, fee, and product information.

The tool is created via a factory function so that the pre-built rag_chain
(instantiated once at server startup in main.py lifespan) is closed over and
reused for every tool call — no redundant re-instantiation occurs.

Usage (in main.py lifespan):
    from tools.knowledge_base_search import make_knowledge_base_tool

    rag_chain = build_rag_chain(llm, vector_store, prompt)
    kb_tool = make_knowledge_base_tool(rag_chain)

    # Then pass kb_tool to your LangChain agent's tool list
    agent = create_tool_calling_agent(llm, tools=[kb_tool, web_search], prompt=agent_prompt)
"""

import logging
from typing import Annotated

from langchain_core.tools import tool
from langchain_core.vectorstores import VectorStoreRetriever

logger = logging.getLogger(__name__)


def make_retriever_tool(
    retriever: VectorStoreRetriever,
):

    @tool
    def retriever_tool(
        query: Annotated[
            str, "The question or topic to look up in the Seven Bank knowledge base."
        ],
    ) -> str:
        """Search the internal Seven Bank knowledge base.

        Use this tool first before falling back to web search. It contains
        authoritative information about Seven Bank's products, services,
        account rules, fees, and policies sourced from official documents.
        """
        logger.info("search_knowledge_base tool called | query=%r", query)

        docs = retriever.invoke(query)

        if not docs:
            return "No relevant information found."

        results = []
        for i, doc in enumerate(docs):
            results.append(f"Document {i + 1}: {doc.page_content}")

        return "\n\n".join(results)

    return retriever_tool

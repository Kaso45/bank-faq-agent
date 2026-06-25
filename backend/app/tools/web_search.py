"""
web_search tool — searches the internet for additional information about
Seven Bank using the ddgs (Dux Distributed Global Search) metasearch library.

Usage (standalone):
    from app.tools.search import web_search
    results = web_search.invoke("Seven Bank ATM fees")

Usage (with LangChain agent):
    from app.tools.search import web_search
    agent = create_tool_calling_agent(llm, [web_search], prompt)
"""

import logging
from typing import Annotated

from ddgs import DDGS
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SEVEN_BANK_SITE_HINT = "Seven Bank"   # appended to every query to bias results
_DEFAULT_MAX_RESULTS = 5
_DEFAULT_REGION = "ja-jp"              # Seven Bank is a Japanese bank; prefer ja-jp
_DEFAULT_TIMELIMIT = None              # no time restriction by default


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

@tool
def web_search(
    query: Annotated[str, "The search query. Focus on Seven Bank products, services, fees, or policies."],
    max_results: Annotated[int, "Maximum number of web results to return (1–10)."] = _DEFAULT_MAX_RESULTS,
) -> str:
    """Search the internet for up-to-date information about Seven Bank.

    Use this tool when the knowledge base does not contain enough information
    to answer a question, or when the user asks about recent news, current
    fees, branch/ATM locations, or other time-sensitive details.

    Returns a formatted string containing the titles, URLs, and body snippets
    of the top search results.
    """
    # Always anchor the query to Seven Bank so results stay on-topic
    enriched_query = f"{_SEVEN_BANK_SITE_HINT} {query}"

    logger.info("web_search tool called | query=%r | max_results=%d", enriched_query, max_results)

    try:
        results = DDGS().text(
            query=enriched_query,
            region=_DEFAULT_REGION,
            safesearch="moderate",
            timelimit=_DEFAULT_TIMELIMIT,
            max_results=max(1, min(max_results, 10)),  # clamp to [1, 10]
            backend="auto",
        )
    except Exception as exc:
        logger.error("web_search failed: %s", exc)
        return f"Web search failed: {exc}"

    if not results:
        return "No web results found for the given query."

    # Format results into a readable string for the LLM
    formatted = []
    for i, item in enumerate(results, start=1):
        title = item.get("title", "No title")
        href = item.get("href", "")
        body = item.get("body", "No description available.")
        formatted.append(f"[{i}] {title}\nURL: {href}\nSnippet: {body}")

    return "\n\n".join(formatted)

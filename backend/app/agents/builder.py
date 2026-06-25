from llm.llm import init_chat_model
from tools.retriever_tool import make_retriever_tool
from tools.web_search import web_search
from graph.graph import build_graph


def create_faq_agent(vector_store, checkpointer):
    """
    Builds the LangGraph agent for the FAQ system.
    """
    # Initialize the LLM once
    llm = init_chat_model()

    # Define tools
    vector_store_retriever = vector_store.get_retriever()
    retriever_tool = make_retriever_tool(vector_store_retriever)
    tools = [retriever_tool, web_search]

    # Build and return the compiled graph
    agent = build_graph(llm, checkpointer, tools)
    return agent

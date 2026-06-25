from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import ToolNode
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.tools import BaseTool

from .state import State
from .nodes import AgentNode, tools_continue


def build_graph(llm: BaseChatModel, checkpointer: PostgresSaver, tools: list[BaseTool]):
    graph = StateGraph(State)  # type: ignore

    agent_node = AgentNode(llm, tools)
    tool_node = ToolNode(tools)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent", tools_continue, {"continue": "tools", "end": END}
    )
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=checkpointer)

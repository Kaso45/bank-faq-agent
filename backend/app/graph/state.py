from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

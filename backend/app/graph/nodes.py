from langchain_core.messages import SystemMessage

from .state import State
from prompts.prompts import AGENT_SYSTEM_PROMPT


class AgentNode:
    def __init__(self, llm, tools: list):
        self.llm = llm.bind_tools(tools)

    def __call__(self, state: State):
        messages = state["messages"]
        system_message = SystemMessage(content=AGENT_SYSTEM_PROMPT)
        
        call_messages = [system_message] + [m for m in messages if not isinstance(m, SystemMessage)]
        
        response = self.llm.invoke(call_messages)
        return {"messages": [response]}


def tools_continue(state: State):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    return "continue"

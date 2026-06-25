from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace
from dotenv import load_dotenv

from utils.config import (
    LLM_REPO_ID,
    LLM_TOP_K,
    LLM_TEMPERATURE,
    LLM_STREAM,
    LLM_MAX_NEW_TOKENS,
)

load_dotenv()


def _call_llm():
    llm = HuggingFaceEndpoint(
        repo_id=LLM_REPO_ID,
        max_new_tokens=LLM_MAX_NEW_TOKENS,
        top_k=LLM_TOP_K,
        temperature=LLM_TEMPERATURE,
        streaming=LLM_STREAM,
        provider="auto",
    )  # pyright: ignore[reportCallIssue]

    return llm


def init_chat_model():
    llm = _call_llm()
    model = ChatHuggingFace(llm=llm)

    return model

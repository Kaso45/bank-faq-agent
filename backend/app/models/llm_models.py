from typing import Optional
from pydantic import BaseModel


class LLMRequest(BaseModel):
    conversation_id: Optional[int]
    prompt: str
    retriever_top_k: Optional[int] = (
        5  # number of documents fetched from the vector store
    )


class LLMResponse(BaseModel):
    answer: str

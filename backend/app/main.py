import logging
from uuid import uuid4
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

from utils.config import COLLECTION_NAME, POSTGRE_DB_URI
from models.llm_models import LLMResponse, LLMRequest
from rag.vector_store import VectorStore
from rag.embedder import get_embeddings
from agents.builder import create_faq_agent

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

vector_store: VectorStore | None = None
agent = None
connection_pool = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store, agent, connection_pool
    logger.info("Loading vector store (collection: %s)...", COLLECTION_NAME)
    embeddings = get_embeddings()
    vector_store = VectorStore(COLLECTION_NAME, embeddings)
    logger.info("VectorStore ready.")

    # initialize postgres connection pool for checkpointer
    connection_pool = ConnectionPool(
        conninfo=POSTGRE_DB_URI,
        max_size=20,
        kwargs={"autocommit": True, "prepare_threshold": 0}
    )
    
    checkpointer = PostgresSaver(connection_pool)
    checkpointer.setup()

    # build agent
    agent = create_faq_agent(vector_store, checkpointer)

    yield

    if connection_pool:
        connection_pool.close()
    vector_store = None


app = FastAPI(lifespan=lifespan)


@app.post("/generate", response_model=LLMResponse)
def generate_answer(request: LLMRequest):
    if agent is None:
        raise HTTPException(
            status_code=503,
            detail="RAG chain not initialised — server is still starting up.",
        )

    thread_id = (
        str(request.conversation_id)
        if request.conversation_id is not None
        else str(uuid4())
    )

    response = agent.invoke(
        {"messages": [HumanMessage(content=request.prompt)]},
        {"configurable": {"thread_id": thread_id}},
    )
    return LLMResponse(answer=response["messages"][-1].content)


@app.get("/threads")
def get_threads():
    if connection_pool is None:
        raise HTTPException(status_code=503, detail="Database connection not ready")
    
    with connection_pool.connection() as conn:
        with conn.cursor() as cur:
            # LangGraph PostgresSaver creates a `checkpoints` table.
            # We fetch distinct thread_ids.
            try:
                cur.execute("SELECT DISTINCT thread_id FROM checkpoints")
                rows = cur.fetchall()
                return {"threads": [r[0] for r in rows if r[0] is not None]}
            except Exception as e:
                logger.error(f"Error fetching threads: {e}")
                return {"threads": []}

@app.get("/history/{thread_id}")
def get_history(thread_id: str):
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not ready")
    
    config = {"configurable": {"thread_id": thread_id}}
    state = agent.get_state(config)
    messages = state.values.get("messages", [])
    
    formatted_messages = []
    for m in messages:
        if getattr(m, "type", "") == "human":
            formatted_messages.append({"role": "user", "content": m.content})
        elif getattr(m, "type", "") == "ai":
            if m.content and str(m.content).strip():
                formatted_messages.append({"role": "assistant", "content": m.content})
            
    return {"messages": formatted_messages}


if __name__ == "__main__":
    import uvicorn
    from backend.app.utils.config import FASTAPI_HOST, FASTAPI_PORT

    uvicorn.run("main:app", host=FASTAPI_HOST, port=FASTAPI_PORT, reload=True)

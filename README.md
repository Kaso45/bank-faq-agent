# Bank Term FAQ Agent

A conversational AI agent built to answer frequently asked questions regarding banking terms and services. The project consists of a FastAPI backend using LangGraph/LangChain, a Streamlit frontend for user interaction, a Chroma vector database for Retrieval-Augmented Generation (RAG), and a PostgreSQL database for checkpointer state management.

## Prerequisites

- Python 3.10+
- PostgreSQL
- [Chroma](https://docs.trychroma.com/)
- API Keys (e.g., HuggingFace API key)

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd bank-term-faq-agent
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # On Windows
   .venv\Scripts\activate
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Copy the provided `.env.example` file to create your own `.env` file:
   ```bash
   cp .env.example .env
   ```
   Fill in the missing values in `.env` (like `HUGGINGFACEHUB_API_TOKEN` and your PostgreSQL credentials).

## Running the Application

To fully launch the application, you need to run several components simultaneously. Follow these steps in order:

### 1. Activate the Local PostgreSQL DB
Ensure your local PostgreSQL service is running and that the database specified in your `.env` file (`POSTGRE_DB_URI`) has been created. 
- If you are running PostgreSQL locally as a service, start it via your system's service manager or PostgreSQL App.
- If you are using Docker, you can start it with a command similar to:
  ```bash
  docker run --name some-postgres -e POSTGRES_PASSWORD=mypassword -p 5432:5432 -d postgres
  ```

### 2. Run the Chroma Database Locally
Start the ChromaDB server so the agent can retrieve knowledge for the RAG system:
```bash
chroma run --host localhost --port 8001 --path backend\data\chroma_db
```

### 3. Ingest Documents into the Vector Database
Before the agent can answer questions, you need to populate the vector database with the FAQ documents. Ensure the Chroma DB server is running (from step 2), then execute the ingestion script from the `backend/app` directory:
```bash
cd backend/app
python -m rag.store
cd ../..
```

### 4. Run the FastAPI App on Local
Start the backend FastAPI server, which manages the LangGraph agent and API endpoints:
```bash
uvicorn backend.app.main:app --reload
# or if using the fastapi cli:
# fastapi dev backend/app/main.py
```
*The backend should now be running (typically accessible at http://localhost:8000).*

### 5. Run the Streamlit App
Start the Streamlit frontend to interact with the FAQ Agent:
```bash
streamlit run frontend/app.py
```
*The Streamlit interface should open automatically in your browser (typically accessible at http://localhost:8501).*

## Project Structure

- `backend/`: Contains the FastAPI application, LangGraph agent definitions, and database schemas.
- `frontend/`: Contains the Streamlit user interface (`app.py`).
- `tests/`: Contains test cases for the agent and API.

## Citation

The FAQ data and reference documents used to power the agent's knowledge base are sourced from **Seven Bank**.

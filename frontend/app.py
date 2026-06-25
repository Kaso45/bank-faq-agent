import streamlit as st
import requests
import random
import os

# Configuration
FASTAPI_HOST = os.getenv("FASTAPI_HOST", "localhost")
FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", "8000"))
API_BASE = f"http://{FASTAPI_HOST}:{FASTAPI_PORT}"
API_URL = f"{API_BASE}/generate"
THREADS_URL = f"{API_BASE}/threads"
HISTORY_URL = f"{API_BASE}/history"

# Streamlit Page Config
st.set_page_config(page_title="Bank Term FAQ Agent", page_icon="🏦")
st.title("🏦 Bank Term FAQ Agent")
st.write("Ask questions about bank terminology and get answers from our RAG agent.")

# Initialize session state variables
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = random.randint(1, 1000000)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Fetch available threads from backend
try:
    res = requests.get(THREADS_URL)
    if res.status_code == 200:
        threads = res.json().get("threads", [])
    else:
        threads = []
except Exception:
    threads = []

# Callback for when a new thread is selected
def on_thread_change():
    selected = st.session_state.thread_selectbox
    if selected == "New Conversation":
        st.session_state.conversation_id = random.randint(1, 1000000)
        st.session_state.messages = []
    else:
        # Pydantic model expects int, so we try to parse it if possible
        try:
            st.session_state.conversation_id = int(selected)
        except ValueError:
            st.session_state.conversation_id = selected
            
        try:
            res = requests.get(f"{HISTORY_URL}/{selected}")
            if res.status_code == 200:
                st.session_state.messages = res.json().get("messages", [])
            else:
                st.session_state.messages = []
        except Exception:
            st.session_state.messages = []

# Sidebar for selecting past conversations
with st.sidebar:
    st.header("Conversations")
    
    options = ["New Conversation"] + threads
    
    # Try to find the index of the current conversation to set as default
    current_str = str(st.session_state.conversation_id)
    try:
        index = options.index(current_str)
    except ValueError:
        index = 0
        
    st.selectbox(
        "Select Conversation", 
        options=options, 
        index=index,
        key="thread_selectbox",
        on_change=on_thread_change
    )

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question (e.g. What is a Certificate of Deposit?)"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Prepare payload for backend API
    payload = {
        "conversation_id": st.session_state.conversation_id,
        "prompt": prompt
    }
    
    # Make request to backend
    with st.spinner("Thinking..."):
        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()
            data = response.json()
            answer = data.get("answer", "No answer received from the agent.")
        except requests.exceptions.RequestException as e:
            answer = f"⚠️ Error communicating with backend: {e}"

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(answer)
        
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})

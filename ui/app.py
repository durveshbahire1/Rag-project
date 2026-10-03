"""
app.py (Streamlit UI)
----------------------
A simple chat interface that calls our FastAPI backend's /ask endpoint
and displays the answer along with which source documents it used.


"""

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄")
st.title("📄 Document Q&A (RAG)")
st.caption("Ask questions about the documents that have been ingested into the knowledge base.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay previous messages in the chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption("Sources: " + ", ".join(msg["sources"]))

question = st.chat_input("Ask a question about your documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant chunks and generating an answer..."):
            try:
                response = requests.post(
                    f"{API_URL}/ask",
                    json={"question": question, "top_k": 3},
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()
                answer = data["answer"]
                sources = data["sources"]
            except Exception as e:
                answer = f"Error calling the API: {e}"
                sources = []

            st.markdown(answer)
            if sources:
                st.caption("Sources: " + ", ".join(sources))

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )

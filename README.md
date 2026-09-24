# Document Q&A — a Retrieval-Augmented Generation (RAG) system

A full-stack RAG application: upload documents, ask questions about them
in natural language, get answers grounded in the source text with
citations — instead of relying on a language model's memorized
training data.

## Why this project

Most companies today are building internal RAG systems over their own
documents (support tickets, internal wikis, contracts, research). This
project reproduces that architecture end-to-end: ingestion pipeline,
vector search, an LLM-backed API, and a chat UI.

## Architecture

```
 documents (.txt)
       |
       v
 [ingestion/chunker.py]   splits text into overlapping chunks
       |
       v
 [ingestion/ingest.py]    embeds each chunk (sentence-transformers)
       |                  and stores it in ChromaDB
       v
 [db/chroma_store/]       persistent vector database
       ^
       |  similarity search
       |
 [api/rag.py]             embeds the user's question, retrieves the
       |                  top-k most similar chunks, sends them to
       |                  an LLM as context
       v
 [api/main.py]            FastAPI endpoints: /ask, /health
       ^
       |  HTTP requests
       |
 [ui/app.py]              Streamlit chat interface with source citations
```

## Tech stack

| Layer            | Choice                          | Why |
|-------------------|----------------------------------|-----|
| Embeddings        | sentence-transformers (MiniLM)   | Free, runs locally, no API key needed |
| Vector database   | ChromaDB                         | Embedded, zero server setup for local dev |
| Generation LLM    | OpenAI gpt-4o-mini               | Cheap, fast, good enough to prove the concept |
| Backend API       | FastAPI                          | Async, auto-generated docs, industry standard |
| Frontend          | Streamlit                        | Fast to build a real chat interface |
| Deployment        | Docker + Render/Railway          | Free-tier friendly |

## Setup

1. Clone the repo and create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and add your OpenAI API key.

3. Add your own documents (`.txt` files) to `data/`, or use the
   included `sample_docs.txt`.

4. Ingest the documents into the vector database:
   ```
   python ingestion/ingest.py
   ```

5. Start the API:
   ```
   uvicorn api.main:app --reload
   ```

6. In a second terminal, start the UI:
   ```
   streamlit run ui/app.py
   ```

7. Open the Streamlit URL it prints and start asking questions.

## What I'd improve with more time (v2 ideas)

- Swap ChromaDB for Postgres + pgvector to show a production-grade,
  concurrent-safe database instead of an embedded local store
- Support PDF/DOCX ingestion, not just plain text
- Add retrieval evaluation (e.g. does the correct chunk get retrieved
  for a held-out set of test questions)
- Stream the LLM's response token-by-token instead of waiting for the
  full answer
- Add a re-ranking step after initial retrieval to improve precision

## Credits

Ingestion pipeline structure informed by patterns in open-source RAG
reference projects such as `serkanyasr/agentic_rag_project` and
`F4k3r22/RAG-FastAPI-Server` — code here is written from scratch.

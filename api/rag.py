"""
rag.py
------
The core RAG logic, separate from the API layer so it's testable and
reusable (e.g. you could call this from a CLI script too).

Two responsibilities:
  1. retrieve()  -- given a question, find the most relevant chunks
  2. generate_answer() -- given a question + retrieved chunks, ask an
     LLM to produce a grounded answer, citing which chunks it used
"""

import os

import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()


DB_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "chroma_store")
COLLECTION_NAME = "documents"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"



# Loaded once at import time -- loading the embedding model is slow,
# so we don't want to reload it on every request.
_embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
_chroma_client = chromadb.PersistentClient(path=DB_PATH)
_collection = _chroma_client.get_or_create_collection(name=COLLECTION_NAME)


_openai_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def retrieve(question: str, top_k: int = 3) -> list[dict]:
    """
    Embed the question and find the top_k most similar chunks in
    the vector database. Returns each chunk's text, source file,
    and similarity distance (lower distance = more similar).
    """
    question_embedding = _embedding_model.encode([question]).tolist()

    results = _collection.query(
        query_embeddings=question_embedding,
        n_results=top_k,
    )

    chunks = []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, dist in zip(documents, metadatas, distances):
        chunks.append(
            {
                "text": doc,
                "source": meta.get("source"),
                "position": meta.get("position"),
                "distance": dist,
            }
        )

    return chunks


def generate_answer(question: str, retrieved_chunks: list[dict]) -> dict:
    """
    Build a prompt from the retrieved chunks and ask the LLM to answer
    using ONLY that context. Returns the answer plus the sources used,
    so the frontend can show citations.
    """
    if not retrieved_chunks:
        return {
            "answer": "I couldn't find anything relevant in the documents to answer that.",
            "sources": [],
        }

    context_block = "\n\n".join(
        f"[Source: {c['source']}, chunk {c['position']}]\n{c['text']}"
        for c in retrieved_chunks
    )

    system_prompt = (
        "You are a helpful assistant that answers questions using ONLY "
        "the provided context. If the context does not contain the "
        "answer, say you don't have enough information -- do not make "
        "things up. When you use information from the context, mention "
        "which source it came from."
    )

    user_prompt = f"Context:\n{context_block}\n\nQuestion: {question}"

    response = _openai_client.chat.completions.create(
        model="openai/gpt-oss-120b",  # cheap + fast; swap for a stronger model if needed
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,  # low temperature: we want grounded, consistent answers
    )

    answer_text = response.choices[0].message.content

    sources = sorted({c["source"] for c in retrieved_chunks})

    return {"answer": answer_text, "sources": sources}


def answer_question(question: str, top_k: int = 3) -> dict:
    """Convenience wrapper: retrieve then generate, in one call."""
    chunks = retrieve(question, top_k=top_k)
    result = generate_answer(question, chunks)
    result["retrieved_chunks"] = chunks  # useful for debugging / transparency
    return result

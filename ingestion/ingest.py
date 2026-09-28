"""
ingest.py
---------
Full ingestion pipeline:

  raw file  ->  chunk_text()  ->  embed each chunk  ->  store in ChromaDB

WHY AN EMBEDDING MODEL?
An embedding model converts text into a vector of numbers (e.g. 384
numbers) such that texts with SIMILAR MEANING end up as vectors that
are close together in that 384-dimensional space. This is what lets
us later search by meaning ("how do I cancel my order") and match a
chunk that never uses those exact words ("Refunds can be requested
within 30 days of purchase...").

WHY sentence-transformers/all-MiniLM-L6-v2?
It's small (~80MB), fast, runs on CPU with no API key, and is a
solid general-purpose baseline. Good enough to prove the concept;
swap in a stronger model later if you want to show you understand
the tradeoffs (quality vs. speed vs. cost).

WHY CHROMADB?
It's an embedded vector database -- no separate server to run, it
just persists to a local folder. Perfect for learning and for a
portfolio demo. In production you'd likely use pgvector (Postgres)
or a managed vector DB (Pinecone, Weaviate) for scale/concurrency.
"""

import os
import sys

import chromadb
from sentence_transformers import SentenceTransformer

# Make the sibling `chunker.py` importable when running this file directly
sys.path.append(os.path.dirname(__file__))
from chunker import chunk_text

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "chroma_store")
COLLECTION_NAME = "documents"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def get_chroma_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return collection


def ingest_file(filepath: str, embedding_model: SentenceTransformer):
    """Read one text file, chunk it, embed each chunk, store in ChromaDB."""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    source_name = os.path.basename(filepath)
    chunks = chunk_text(text, source=source_name)

    if not chunks:
        print(f"  [skip] {source_name} produced no chunks (empty file?)")
        return 0

    # Embed all chunks in one batch call -- much faster than one at a time
    chunk_texts = [c.text for c in chunks]
    embeddings = embedding_model.encode(chunk_texts).tolist()

    collection = get_chroma_collection()
    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings,
        documents=chunk_texts,
        metadatas=[{"source": c.source, "position": c.position} for c in chunks],
    )

    print(f"  [ok] {source_name}: {len(chunks)} chunks ingested")
    return len(chunks)


def ingest_directory(data_dir: str):
    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' (first run downloads it)...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    total = 0
    files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]

    if not files:
        print(f"No .txt files found in {data_dir}. Add some documents and re-run.")
        return

    print(f"Found {len(files)} file(s) to ingest:")
    for filename in files:
        filepath = os.path.join(data_dir, filename)
        total += ingest_file(filepath, embedding_model)

    print(f"\nDone. {total} total chunks stored in ChromaDB at {DB_PATH}")


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    ingest_directory(data_dir)

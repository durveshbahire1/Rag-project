"""
main.py
-------
The web server. Exposes the RAG pipeline as an HTTP API so a frontend
(Streamlit, React, curl, Postman -- anything) can use it.

Run locally with:
    uvicorn api.main:app --reload

Then visit http://127.0.0.1:8000/docs for interactive API docs
(FastAPI generates this automatically from the type hints below --
worth mentioning in an interview, it's a big reason teams pick FastAPI).
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.rag import answer_question

app = FastAPI(title="RAG Q&A API")

# Allow the Streamlit/React frontend (running on a different port) to
# call this API from the browser. In production you'd restrict this
# to your actual frontend's domain instead of "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    top_k: int = 3


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/health")
def health():
    """Simple liveness check -- useful for deployment platforms and monitoring."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """
    Main endpoint: takes a question, runs retrieval + generation,
    returns the answer along with which source documents were used.
    """
    result = answer_question(request.question, top_k=request.top_k)
    return AskResponse(answer=result["answer"], sources=result["sources"])

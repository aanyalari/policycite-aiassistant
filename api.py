from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from citation.pipeline import answer_with_citations
from citation.schemas import CitationTrace
from rag_core import MedicalRAG

app = FastAPI(title="Medical RAG API", version="2.0")

#Local dev CORS 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = MedicalRAG()


class Query(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(query: Query):
    result = rag.answer(query.question)
    return {
        "answer": result.answer,
        "sources": result.sources,
        "confidence": result.confidence,
        "normalized_query": result.normalized_query,
    }


@app.post("/ask_cited", response_model=CitationTrace)
def ask_cited(query: Query):
    return answer_with_citations(query.question, rag)


#uvicorn api:app --reload

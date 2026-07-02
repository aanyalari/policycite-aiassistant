import json
from pathlib import Path
from time import perf_counter
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from citation.pipeline import answer_with_citations
from citation.schemas import CitationTrace
from rag_core import AnswerTrace, MedicalRAG, doc_key

app = FastAPI(title="Medical RAG API", version="2.0")

ROOT = Path(__file__).resolve().parent
PRESERVED_DEMO_ARTIFACT = ROOT / "evaluation/results/20260701T210613Z.json"
PRESERVED_DEMO_CASE_ID = "policy-003"

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


class BaselineSnapshot(BaseModel):
    answer: str
    sources: List[str]
    confidence: float
    normalized_query: str
    latency_ms: float


class ReplayProvenance(BaseModel):
    artifact: str
    case_id: str
    recorded_at: str
    description: str


class ReplayDemoResponse(BaseModel):
    question: str
    baseline: BaselineSnapshot
    policycite: CitationTrace
    provenance: ReplayProvenance


class LiveBaselineResponse(BaseModel):
    question: str
    baseline: BaselineSnapshot
    policycite: CitationTrace
    baseline_mode: str
    description: str


def _load_preserved_demo_case():
    artifact = json.loads(PRESERVED_DEMO_ARTIFACT.read_text(encoding="utf-8"))
    try:
        result = next(
            item
            for item in artifact["results"]
            if item["id"] == PRESERVED_DEMO_CASE_ID
        )
    except (KeyError, StopIteration) as exc:
        raise ValueError(
            f"Preserved demo case {PRESERVED_DEMO_CASE_ID!r} is unavailable"
        ) from exc
    return artifact, result


def _answer_trace_from_preserved_case(result, rag_instance):
    output = result["baseline"]["output"]
    generation_ids = result["policycite_rag"]["output"].get(
        "generation_context_ids", []
    )
    documents_by_id = {
        doc_key(document): document for document in rag_instance.corpus_docs
    }
    missing_ids = [
        generation_id
        for generation_id in generation_ids
        if generation_id not in documents_by_id
    ]
    if missing_ids:
        raise ValueError(
            "Preserved generation passages are missing from the current corpus: "
            + ", ".join(missing_ids)
        )
    return AnswerTrace(
        answer=output["answer"],
        sources=output["sources"],
        confidence=output["confidence"],
        normalized_query=output["normalized_query"],
        retrieved_docs=[documents_by_id[item] for item in generation_ids],
    )


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


@app.post("/demo/live", response_model=LiveBaselineResponse)
def audit_live_original_baseline(query: Query):
    """Generate with the frozen original RAG, then audit that same answer."""
    started = perf_counter()
    trace = rag.answer_with_trace_original(query.question)
    baseline_latency_ms = (perf_counter() - started) * 1000
    cited = answer_with_citations(
        query.question,
        rag,
        answer_trace=trace,
    )
    return LiveBaselineResponse(
        question=query.question,
        baseline=BaselineSnapshot(
            answer=trace.answer,
            sources=trace.sources,
            confidence=trace.confidence,
            normalized_query=trace.normalized_query,
            latency_ms=baseline_latency_ms,
        ),
        policycite=cited,
        baseline_mode="original_pre_correction",
        description=(
            "Live answer generated with the frozen pre-correction RAG behavior; "
            "the current PolicyCite pipeline audits the exact same answer."
        ),
    )


@app.post(
    "/demo/replay/policy-003",
    response_model=ReplayDemoResponse,
)
def replay_preserved_rag_failure():
    """Re-audit one genuine, preserved RAG answer with current PolicyCite."""
    try:
        artifact, result = _load_preserved_demo_case()
        trace = _answer_trace_from_preserved_case(result, rag)
        cited = answer_with_citations(
            result["question"],
            rag,
            answer_trace=trace,
        )
        output = result["baseline"]["output"]
        return ReplayDemoResponse(
            question=result["question"],
            baseline=BaselineSnapshot(
                answer=output["answer"],
                sources=output["sources"],
                confidence=output["confidence"],
                normalized_query=output["normalized_query"],
                latency_ms=output["latency_ms"],
            ),
            policycite=cited,
            provenance=ReplayProvenance(
                artifact=PRESERVED_DEMO_ARTIFACT.relative_to(ROOT).as_posix(),
                case_id=result["id"],
                recorded_at=artifact["created_at"],
                description=(
                    "Baseline answer generated by the project RAG model during "
                    "the preserved evaluation run; re-audited live by the current "
                    "PolicyCite pipeline."
                ),
            ),
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"The preserved RAG replay could not be loaded: {exc}",
        ) from exc


#uvicorn api:app --reload

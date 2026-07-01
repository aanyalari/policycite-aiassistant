"""Strict statement-level attribution against retrieved policy passages."""

from typing import List

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from llm_provider import get_attribution_llm

from .schemas import (
    AttributionVerdict,
    CitedStatement,
    EvidenceCitation,
)


class _SelectedEvidence(BaseModel):
    candidate_id: str
    evidence_excerpt: str


class _AttributionDecision(BaseModel):
    verdict: AttributionVerdict
    selected_evidence: List[_SelectedEvidence] = Field(default_factory=list)
    reason: str = ""


_SYSTEM_PROMPT = """You are a strict healthcare-policy evidence judge.
Use only the supplied candidate passages. A statement is SUPPORTED only when
the passages, considered together, support every material part of it. Dates,
quantities, exclusions, modality, scope, and qualifiers are material. Missing,
partial, irrelevant, or conflicting evidence means NOT_SUPPORTED.

Select only candidate IDs supplied below. Evidence excerpts must be short,
exact, contiguous text copied from their selected passages. Do not use external
knowledge and do not invent source names, pages, IDs, or excerpts.
"""


def _candidate_id(doc: Document) -> str:
    return str(doc.metadata.get("chunk_id") or "").strip()


def _format_request(statement: str, evidence_docs: List[Document]) -> str:
    candidates = []
    for doc in evidence_docs:
        candidates.append(
            f"CANDIDATE ID: {_candidate_id(doc)}\nPASSAGE:\n{doc.page_content.strip()}"
        )
    return (
        f"STATEMENT:\n{statement.strip()}\n\n"
        "CANDIDATE PASSAGES:\n\n" + "\n\n---\n\n".join(candidates)
    )


def _not_supported(statement: str, reason: str) -> CitedStatement:
    return CitedStatement(
        text=statement.strip(),
        verdict=AttributionVerdict.NOT_SUPPORTED,
        reason=reason,
    )


def judge_attribution(
    statement: str,
    evidence_docs: List[Document],
    llm=None,
) -> CitedStatement:
    """Return a strict binary verdict with application-validated citations."""
    text = (statement or "").strip()
    usable_docs = [
        doc
        for doc in evidence_docs
        if _candidate_id(doc) and (doc.page_content or "").strip()
    ]
    if not text:
        return _not_supported(text, "No statement was supplied.")
    if not usable_docs:
        return _not_supported(text, "No usable evidence was retrieved.")

    judge = llm or get_attribution_llm()
    try:
        structured_judge = judge.with_structured_output(_AttributionDecision)
        decision = structured_judge.invoke(
            [
                ("system", _SYSTEM_PROMPT),
                ("human", _format_request(text, usable_docs)),
            ]
        )
        if isinstance(decision, dict):
            decision = _AttributionDecision(**decision)
    except Exception as exc:
        return _not_supported(text, f"Attribution judge failed: {exc}")

    docs_by_id = {_candidate_id(doc): doc for doc in usable_docs}
    citations = []
    seen = set()
    for selected in decision.selected_evidence:
        doc = docs_by_id.get(selected.candidate_id)
        excerpt = selected.evidence_excerpt.strip()
        if (
            doc is None
            or not excerpt
            or excerpt not in doc.page_content
            or selected.candidate_id in seen
        ):
            continue
        seen.add(selected.candidate_id)
        metadata = doc.metadata
        internal_page = metadata.get("page")
        display_page = internal_page + 1 if isinstance(internal_page, int) else None
        source = metadata.get("source_file") or metadata.get("source") or ""
        citations.append(
            EvidenceCitation(
                citation_id=selected.candidate_id,
                source=str(source),
                page=display_page,
                evidence_excerpt=excerpt,
                retrieval_score=metadata.get("retrieval_score"),
            )
        )

    if decision.verdict == AttributionVerdict.SUPPORTED and not citations:
        return _not_supported(
            text,
            "The judge returned no valid citation for a supported statement.",
        )

    if decision.verdict == AttributionVerdict.NOT_SUPPORTED:
        citations = []

    return CitedStatement(
        text=text,
        verdict=decision.verdict,
        citations=citations,
        reason=decision.reason or None,
    )

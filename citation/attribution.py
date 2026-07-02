"""Strict statement-level attribution against retrieved policy passages."""

import re
from typing import List

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from llm_provider import get_attribution_llm

from .schemas import (
    AttributionVerdict,
    CitedStatement,
    EvidenceCitation,
)


_LEXICAL_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "is", "it", "of", "on", "or", "that", "the", "their", "this",
    "to", "was", "were", "with", "according", "provided", "sources",
}


class _AttributionDecision(BaseModel):
    verdict: AttributionVerdict
    citation_ids: List[str] = Field(default_factory=list)
    reason: str = ""


_SYSTEM_PROMPT = """You are a strict healthcare-policy evidence judge.
Use only the supplied candidate passages. A statement is SUPPORTED only when
the passages, considered together, support every material part of it. Dates,
quantities, exclusions, modality, scope, and qualifiers are material. Missing,
partial, irrelevant, or conflicting evidence means NOT_SUPPORTED.

For SUPPORTED, return every passage used in the top-level citation_ids list.
For NOT_SUPPORTED, return an empty citation_ids list. Select only candidate IDs
supplied below. Do not use external knowledge and do not invent source names,
pages, or IDs.
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


def _best_exact_excerpt(statement: str, passage: str) -> str:
    """Select the most statement-relevant sentence without model-generated text."""
    sentences = [
        value.strip()
        for value in re.split(r"(?<=[.!?])\s+", passage.strip())
        if value.strip()
    ]
    if not sentences:
        return passage.strip()

    statement_tokens = set(re.findall(r"[a-z0-9]+", statement.lower()))
    return max(
        sentences,
        key=lambda value: len(
            statement_tokens & set(re.findall(r"[a-z0-9]+", value.lower()))
        ),
    )


def _content_tokens(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in _LEXICAL_STOPWORDS
    }


def _support_score(statement: str, passage: str) -> float:
    statement_tokens = _content_tokens(statement)
    if not statement_tokens:
        return 0.0
    passage_tokens = _content_tokens(passage)
    statement_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", statement))
    passage_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", passage))
    if not statement_numbers.issubset(passage_numbers):
        return 0.0
    return len(statement_tokens & passage_tokens) / len(statement_tokens)


def _citation_from_doc(statement: str, doc: Document) -> EvidenceCitation:
    metadata = doc.metadata
    internal_page = metadata.get("page")
    display_page = internal_page + 1 if isinstance(internal_page, int) else None
    source = metadata.get("source_file") or metadata.get("source") or ""
    return EvidenceCitation(
        citation_id=_candidate_id(doc),
        source=str(source),
        page=display_page,
        evidence_excerpt=_best_exact_excerpt(statement, doc.page_content),
        retrieval_score=metadata.get("retrieval_score"),
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

    best_lexical_doc = max(
        usable_docs, key=lambda doc: _support_score(text, doc.page_content)
    )
    best_lexical_score = _support_score(text, best_lexical_doc.page_content)
    if best_lexical_score >= 0.82:
        return CitedStatement(
            text=text,
            verdict=AttributionVerdict.SUPPORTED,
            citations=[_citation_from_doc(text, best_lexical_doc)],
            reason="Near-verbatim support found in the retrieved passage.",
        )

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
    candidate_docs = []
    for candidate_id in decision.citation_ids:
        doc = docs_by_id.get(candidate_id)
        if doc is None or doc in candidate_docs:
            continue
        candidate_docs.append(doc)

    # A single strongest passage is easier to audit and avoids attaching broad,
    # merely related passages alongside the actual support.
    citations = []
    if candidate_docs:
        best_doc = max(
            candidate_docs,
            key=lambda doc: _support_score(text, doc.page_content),
        )
        citations = [_citation_from_doc(text, best_doc)]

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

"""Pydantic schemas shared by the citation pipeline."""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class StatementExtraction(BaseModel):
    """Structured result returned by the atomic-statement extractor."""

    statements: List[str] = Field(default_factory=list)


class AttributionVerdict(str, Enum):
    """Strict verdicts supported by the lean attribution prototype."""

    SUPPORTED = "SUPPORTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"


class EvidenceCitation(BaseModel):
    """A citation whose source metadata has been validated by application code."""

    citation_id: str
    source: str
    page: Optional[int] = None
    evidence_excerpt: str
    retrieval_score: Optional[float] = None
    context_type: str = "post_generation"


class CitedStatement(BaseModel):
    """One atomic statement and its evidence attribution result."""

    text: str
    verdict: AttributionVerdict
    citations: List[EvidenceCitation] = Field(default_factory=list)
    reason: Optional[str] = None


class CitationTrace(BaseModel):
    """Public result returned by the end-to-end citation pipeline."""

    question: str
    answer: str
    baseline_sources: List[str] = Field(default_factory=list)
    generation_context_ids: List[str] = Field(default_factory=list)
    statements: List[CitedStatement] = Field(default_factory=list)
    citation_coverage: float = 0.0
    needs_human_review: bool = False
    latency_ms: Optional[float] = None

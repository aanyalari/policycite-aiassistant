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

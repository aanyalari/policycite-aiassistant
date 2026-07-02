"""Extract independently verifiable factual statements from an answer."""

import json
import re
from typing import Any, Iterable, List

from .schemas import StatementExtraction


_CITATION_RE = re.compile(
    r"\s*\[S\d+(?:\s*,\s*S?\d+)*\]"
    r"(?:\s+[A-Za-z0-9][A-Za-z0-9_ .()&'’\-]*?\.pdf"
    r"\s*\(page\s+(?:\d+|\?)\))?",
    re.IGNORECASE,
)
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_REFERENCE_FRAGMENT_RE = re.compile(
    r"^(?:section\s+[\w.§()\-]+|chapter\s+[\w.§()\-]+|"
    r"medicare regulations at\b|42\s+c\.?f\.?r\.?\b|"
    r"publication\s+\d|pub\.?\s+\d)",
    re.IGNORECASE,
)
_PREDICATE_RE = re.compile(
    r"\b(?:is|are|was|were|be|been|being|has|have|had|must|may|can|"
    r"will|shall|requires?|required|states?|stated|defines?|defined|"
    r"applies?|applied|allows?|allowed|reduces?|reduced|amends?|amended|"
    r"provides?|provided|specifies?|specified|mandates?|mandated|uses?|"
    r"used|files?|filed|submits?|submitted|denies?|denied|includes?|"
    r"included|excludes?|excluded|found|associated|reports?|reported)\b",
    re.IGNORECASE,
)

_NON_FACTUAL_EXACT = {
    "not found in documents",
    "not found in the documents",
    "no information was found in the documents",
}

_NON_FACTUAL_PREFIXES = (
    "hello",
    "hi ",
    "hey ",
    "thank you",
    "thanks",
    "i can only answer",
    "i can help",
    "please ask",
    "based on the provided",
    "note:",
)

_DISCLAIMER_MARKERS = (
    "not medical advice",
    "not a substitute for",
    "consult a healthcare",
    "consult your healthcare",
    "consult a medical",
    "seek professional",
    "research only",
    "for informational purposes",
)

_COVERAGE_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "in", "is", "it", "of", "on", "or", "that", "the",
    "their", "this", "to", "was", "were", "with",
}


def _clean_statement(value: Any) -> str:
    if not isinstance(value, str):
        return ""

    text = value.strip()
    text = re.sub(r"^\s*(?:[-*•]+|\d+[.)])\s*", "", text)
    text = re.sub(r"^#{1,6}\s*", "", text)
    text = re.sub(r"^therefore,?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"^the correct answer is:\s*", "", text, flags=re.IGNORECASE
    )
    text = _CITATION_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip(" -*\t\r\n")
    return text


def _is_factual_candidate(text: str) -> bool:
    if not text:
        return False

    normalized = text.lower().strip(" .!?:")
    if any(message in normalized for message in _NON_FACTUAL_EXACT):
        return False
    if normalized.startswith(_NON_FACTUAL_PREFIXES):
        return False
    if any(marker in normalized for marker in _DISCLAIMER_MARKERS):
        return False
    if text.endswith(":"):
        return False
    if len(text.split()) < 4:
        return False
    if not _PREDICATE_RE.search(text):
        return False
    if _REFERENCE_FRAGMENT_RE.match(text) and not _PREDICATE_RE.search(text):
        return False
    return True


def _deduplicate(values: Iterable[Any]) -> List[str]:
    statements: List[str] = []
    seen = set()

    for value in values:
        statement = _clean_statement(value)
        if not _is_factual_candidate(statement):
            continue
        key = re.sub(r"[^a-z0-9]+", " ", statement.lower()).strip()
        if key and key not in seen:
            seen.add(key)
            statements.append(statement)

    return statements


def _fallback_extract(answer: str) -> List[str]:
    """Deterministically split prose and list items into factual sentences."""
    candidates: List[str] = []

    for raw_line in answer.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^\s*(?:[-*•]+|\d+[.)])\s*", "", line)
        line = re.sub(r"^#{1,6}\s*", "", line).strip()
        if not line or line.endswith(":"):
            continue
        candidates.extend(_SENTENCE_BOUNDARY_RE.split(line))

    return _deduplicate(candidates)


def _coverage_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in _COVERAGE_STOPWORDS
    }


def _merge_for_coverage(
    structured_statements: List[str],
    fallback_statements: List[str],
) -> List[str]:
    """Keep atomic model output while restoring factual sentences it omitted."""
    merged = list(structured_statements)
    structured_token_sets = [
        _coverage_tokens(statement) for statement in structured_statements
    ]

    for fallback_statement in fallback_statements:
        fallback_tokens = _coverage_tokens(fallback_statement)
        if not fallback_tokens:
            continue
        matches_one_statement = any(
            len(fallback_tokens & candidate_tokens)
            / min(len(fallback_tokens), len(candidate_tokens))
            >= 0.8
            for candidate_tokens in structured_token_sets
            if candidate_tokens
        )
        related_token_sets = [
            candidate_tokens
            for candidate_tokens in structured_token_sets
            if candidate_tokens
            and len(fallback_tokens & candidate_tokens)
            / min(len(fallback_tokens), len(candidate_tokens))
            >= 0.45
        ]
        related_union = (
            set().union(*related_token_sets) if related_token_sets else set()
        )
        covered_by_atomic_parts = (
            len(fallback_tokens & related_union) / len(fallback_tokens) >= 0.8
        )
        if not matches_one_statement and not covered_by_atomic_parts:
            merged.append(fallback_statement)
            structured_token_sets.append(fallback_tokens)

    return _deduplicate(merged)


def _coerce_structured_result(result: Any) -> List[str]:
    if isinstance(result, StatementExtraction):
        return result.statements
    if isinstance(result, dict):
        statements = result.get("statements")
        if isinstance(statements, list):
            return statements
        raise ValueError("Structured output is missing a statements list")
    if isinstance(result, str):
        parsed = json.loads(result)
        if isinstance(parsed, dict) and isinstance(parsed.get("statements"), list):
            return parsed["statements"]
    raise ValueError("Malformed statement extraction output")


def extract_statements(answer: str, llm=None) -> List[str]:
    """Return atomic, factual, independently verifiable statements.

    When an LLM is supplied, the function makes one temperature-zero structured
    extraction call. Any unavailable or malformed model response falls back to
    deterministic sentence splitting.
    """
    answer = (answer or "").strip()
    if not answer:
        return []
    if answer.lower().strip(" .") in _NON_FACTUAL_EXACT:
        return []

    if llm is not None:
        prompt = f"""Extract atomic factual statements from the answer below.

Rules:
- Return only independently verifiable factual claims.
- Every item must be a complete standalone sentence with an explicit subject and predicate.
- Never return a heading, document title, statute name, section reference, or noun phrase by itself.
- Split compound claims when each part can be verified independently.
- Preserve all dates, quantities, scope, exclusions, conditions, and modal words such as must, may, and generally.
- Remove citation IDs and source/page labels from the extracted text.
- Omit greetings, headings, disclaimers, and statements saying information was not found.
- Do not add, correct, or infer information.
- Deduplicate equivalent claims.

Answer:
{answer}
"""
        try:
            structured_llm = llm.with_structured_output(StatementExtraction)
            result = structured_llm.invoke(prompt)
            statements = _deduplicate(_coerce_structured_result(result))
            if statements:
                return _merge_for_coverage(
                    statements,
                    _fallback_extract(answer),
                )
        except Exception:
            pass

    return _fallback_extract(answer)

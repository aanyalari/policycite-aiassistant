"""Small, testable formatting helpers for the Streamlit evidence UI."""

from typing import Any, Mapping

import requests


def format_coverage(value: Any) -> str:
    """Return citation coverage as a bounded, human-readable percentage."""
    try:
        coverage = float(value)
    except (TypeError, ValueError):
        coverage = 0.0
    return f"{min(max(coverage, 0.0), 1.0):.0%}"


def format_latency(value: Any) -> str:
    """Return API latency in milliseconds or a clear unavailable label."""
    try:
        latency = float(value)
    except (TypeError, ValueError):
        return "Not available"
    return f"{max(latency, 0.0):,.0f} ms"


def statement_support_summary(statements: Any) -> str:
    """Return an explicit supported-statement count for a PolicyCite audit."""
    if not isinstance(statements, list):
        return "0 of 0"
    supported = sum(
        item.get("verdict") == "SUPPORTED"
        for item in statements
        if isinstance(item, Mapping)
    )
    return f"{supported} of {len(statements)}"


def evidence_location(citation: Mapping[str, Any]) -> str:
    """Build a compact source label from public citation fields."""
    source = str(citation.get("source") or "Unknown document")
    page = citation.get("page")
    return f"{source} — page {page}" if page is not None else source


def api_error_message(error: Exception) -> str:
    """Translate request failures into useful, non-technical UI guidance."""
    if isinstance(error, requests.exceptions.Timeout):
        return "The evidence check took too long. Please try again."
    if isinstance(error, requests.exceptions.ConnectionError):
        return (
            "The evidence service is unavailable. Start the FastAPI backend "
            "and then try again."
        )
    if isinstance(error, requests.exceptions.HTTPError):
        response = error.response
        if response is not None and response.status_code >= 500:
            return "The evidence service could not complete this request. Please try again."
        return "The request was rejected. Check the question and try again."
    if isinstance(error, (ValueError, requests.exceptions.JSONDecodeError)):
        return "The evidence service returned an unreadable response. Please try again."
    return "Something went wrong while checking the evidence. Please try again."

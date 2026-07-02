"""Streamlit interface for statement-level policy citation assurance."""

import os
import re

import requests
import streamlit as st

from ui_helpers import (
    api_error_message,
    evidence_location,
    statement_support_summary,
)


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
LIVE_DEMO_URL = f"{API_BASE_URL}/demo/live"
HEALTH_URL = f"{API_BASE_URL}/health"
INLINE_CITATION_RE = re.compile(r"\s*\[S\d+\]")
SOURCE_PAGE_RE = re.compile(r"\s*\(page\s+\d+\)\s*$", re.IGNORECASE)


def display_baseline_answer(answer):
    """Hide inline response-level citation markers for a cleaner demo display."""
    text = answer or ""
    text = INLINE_CITATION_RE.sub("", text)
    return " ".join(text.split())


def render_baseline_sources(sources):
    """Render answer-level generation sources separately from claim evidence."""
    st.markdown("#### Baseline sources")
    st.caption("Documents used to generate the original answer.")
    if sources:
        for source in dict.fromkeys(sources):
            st.markdown(f"- {SOURCE_PAGE_RE.sub('', source)}")
    else:
        st.caption("No baseline sources were returned.")


def render_statement_evidence(statements):
    """Render one expandable evidence card per extracted factual statement."""
    st.markdown("#### Statement-level evidence")
    if not statements:
        st.info("No factual statements were identified for evidence checking.")
        return

    for index, statement in enumerate(statements, start=1):
        verdict = statement.get("verdict", "NOT_SUPPORTED")
        icon = "✅" if verdict == "SUPPORTED" else "⚠️"
        with st.expander(f"{icon} Statement {index} — {verdict}", expanded=True):
            st.markdown(statement.get("text") or "Statement text unavailable.")

            if verdict == "SUPPORTED":
                citations = statement.get("citations") or []
                for citation_index, citation in enumerate(citations, start=1):
                    st.markdown(f"**Evidence {citation_index}: {evidence_location(citation)}**")
                    st.caption(citation.get("evidence_excerpt") or "Evidence excerpt unavailable.")
            else:
                reason = statement.get("reason") or "No sufficient supporting evidence was found."
                st.warning(f"{reason}\n\n**Review recommended.**")


def render_live_comparison(result):
    """Show one original baseline answer beside its current PolicyCite audit."""
    baseline = result.get("baseline") or {}
    policycite = result.get("policycite") or {}

    st.info(
        "One live answer was generated with the frozen original RAG behavior. "
        "PolicyCite is auditing that exact answer; it did not generate a second response."
    )
    baseline_column, policycite_column = st.columns(2, gap="large")
    with baseline_column:
        st.markdown("### Baseline answer")
        st.caption("Complete generated answer and response-level sources")
        st.markdown(display_baseline_answer(baseline.get("answer")))
        render_baseline_sources(baseline.get("sources") or [])

    with policycite_column:
        st.markdown("### Current PolicyCite audit")
        st.caption("Statement-level verification of the exact answer shown on the left")
        statements = policycite.get("statements") or []
        st.metric(
            "Statements supported",
            statement_support_summary(statements),
        )
        render_statement_evidence(statements)


st.set_page_config(page_title="PolicyCite RAG", page_icon="📚", layout="wide")
st.title("PolicyCite RAG")
st.caption("See which statements in a generated policy answer are supported by source evidence.")
st.info(
    "Research prototype only. Results may be incomplete or incorrect and should not be "
    "used as medical, legal, billing, or coverage advice."
)

try:
    health_response = requests.get(HEALTH_URL, timeout=3)
    backend_ok = health_response.ok
except requests.exceptions.RequestException:
    backend_ok = False

if not backend_ok:
    st.warning("The evidence service is offline. Start the FastAPI backend before submitting a question.")

st.markdown("### Live question")
st.caption(
    "Generate one baseline answer and audit the exact same answer with PolicyCite."
)

if "evidence_results" not in st.session_state:
    st.session_state.evidence_results = []

for saved_result in st.session_state.evidence_results:
    with st.chat_message("user"):
        st.write(saved_result["question"])
    with st.chat_message("assistant"):
        render_live_comparison(saved_result["result"])

question = st.chat_input(
    "Ask a healthcare policy question...",
    disabled=not backend_ok,
)
if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Generating the answer and checking each statement..."):
            try:
                response = requests.post(
                    LIVE_DEMO_URL,
                    json={"question": question},
                    timeout=120,
                )
                response.raise_for_status()
                result = response.json()
            except (requests.exceptions.RequestException, ValueError) as error:
                st.error(api_error_message(error))
            else:
                render_live_comparison(result)
                st.session_state.evidence_results.append(
                    {"question": question, "result": result}
                )

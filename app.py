"""Streamlit interface for statement-level policy citation assurance."""

import os

import requests
import streamlit as st

from ui_helpers import (
    api_error_message,
    evidence_location,
    format_coverage,
    format_latency,
)


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
ASK_CITED_URL = f"{API_BASE_URL}/ask_cited"
HEALTH_URL = f"{API_BASE_URL}/health"


def render_baseline_sources(sources):
    """Render answer-level generation sources separately from claim evidence."""
    st.markdown("#### Baseline answer sources")
    st.caption("Documents used to generate the original answer; these do not prove every statement.")
    if sources:
        for source in dict.fromkeys(sources):
            st.markdown(f"- {source}")
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


def render_result(result):
    """Render a complete `/ask_cited` response."""
    st.markdown("### Generated answer")
    st.markdown(result.get("answer") or "No answer was returned.")

    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("Citation coverage", format_coverage(result.get("citation_coverage")))
    metric_2.metric("Total latency", format_latency(result.get("latency_ms")))
    metric_3.metric(
        "Review status",
        "Review recommended" if result.get("needs_human_review") else "No flags",
    )

    st.divider()
    source_column, evidence_column = st.columns([1, 2])
    with source_column:
        render_baseline_sources(result.get("baseline_sources") or [])
    with evidence_column:
        render_statement_evidence(result.get("statements") or [])


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

if "evidence_results" not in st.session_state:
    st.session_state.evidence_results = []

for saved_result in st.session_state.evidence_results:
    with st.chat_message("user"):
        st.write(saved_result["question"])
    with st.chat_message("assistant"):
        render_result(saved_result["result"])

question = st.chat_input("Ask a healthcare policy question...", disabled=not backend_ok)
if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Generating the answer and checking each statement..."):
            try:
                response = requests.post(
                    ASK_CITED_URL,
                    json={"question": question},
                    timeout=120,
                )
                response.raise_for_status()
                result = response.json()
            except (requests.exceptions.RequestException, ValueError) as error:
                st.error(api_error_message(error))
            else:
                render_result(result)
                st.session_state.evidence_results.append(
                    {"question": question, "result": result}
                )

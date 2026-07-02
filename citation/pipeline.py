"""End-to-end statement-level citation assurance pipeline."""

from time import perf_counter

from rag_core import MedicalRAG, doc_key

from .attribution import judge_attribution
from .evidence_retriever import retrieve_statement_evidence
from .schemas import AttributionVerdict, CitationTrace
from .statement_extractor import extract_statements


def answer_with_citations(
    question: str,
    rag: MedicalRAG,
    llm=None,
    answer_trace=None,
) -> CitationTrace:
    """Generate an answer and attach statement-level citation assurance."""
    started_at = perf_counter()
    answer_trace = answer_trace or rag.answer_with_trace(question)
    generation_context_ids = [
        doc_key(doc) for doc in answer_trace.retrieved_docs
    ]

    factual_statements = extract_statements(answer_trace.answer, llm=llm)
    cited_statements = []
    for statement in factual_statements:
        evidence_docs = retrieve_statement_evidence(
            statement,
            rag,
            top_k=5,
            generation_context_ids=generation_context_ids,
        )
        cited_statements.append(
            judge_attribution(statement, evidence_docs, llm=llm)
        )

    supported_count = sum(
        item.verdict == AttributionVerdict.SUPPORTED
        for item in cited_statements
    )
    citation_coverage = (
        supported_count / len(cited_statements) if cited_statements else 0.0
    )
    needs_human_review = any(
        item.verdict == AttributionVerdict.NOT_SUPPORTED
        for item in cited_statements
    )

    return CitationTrace(
        question=question,
        answer=answer_trace.answer,
        baseline_sources=answer_trace.sources,
        generation_context_ids=generation_context_ids,
        statements=cited_statements,
        citation_coverage=citation_coverage,
        needs_human_review=needs_human_review,
        latency_ms=(perf_counter() - started_at) * 1000,
    )

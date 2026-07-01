"""Retrieve post-generation evidence for one factual statement."""

from typing import Collection, List, Optional

from langchain_core.documents import Document

from rag_core import MedicalRAG, doc_key, rrf_fuse


def _usable_document(doc: Document) -> bool:
    """Return whether a passage has enough information to cite safely."""
    metadata = doc.metadata
    source = metadata.get("source") or metadata.get("source_file")
    return bool(
        (doc.page_content or "").strip()
        and source
        and isinstance(metadata.get("page"), int)
    )


def retrieve_statement_evidence(
    statement: str,
    rag: MedicalRAG,
    *,
    top_k: int = 3,
    generation_context_ids: Optional[Collection[str]] = None,
) -> List[Document]:
    """Return RRF-ranked corpus passages relevant to one statement.

    Retrieval is deliberately run against the completed statement, independently
    of the question that produced it. Returned documents are defensive copies and
    include ``chunk_id``, ``retrieval_score``, and ``in_generation_context`` in
    their metadata. Pages remain zero-based, matching the ingested corpus.
    """
    query = (statement or "").strip()
    if not query or top_k <= 0:
        return []

    try:
        vector_docs = rag._vector_mmr_search(query)
        bm25_docs = rag._bm25_search(query)
    except (FileNotFoundError, RuntimeError, ValueError):
        return []

    # Fetch every unique candidate before filtering malformed passages; this
    # prevents one unusable high-ranked chunk from reducing a valid top-k list.
    candidate_count = len(vector_docs) + len(bm25_docs)
    if candidate_count == 0:
        return []

    fused_docs = rrf_fuse(
        vector_docs,
        bm25_docs,
        k=candidate_count,
        include_scores=True,
    )
    context_ids = {str(value) for value in (generation_context_ids or ())}

    evidence: List[Document] = []
    for doc in fused_docs:
        if not _usable_document(doc):
            continue
        metadata = dict(doc.metadata)
        stable_id = doc_key(doc)
        metadata["chunk_id"] = stable_id
        metadata["in_generation_context"] = stable_id in context_ids
        evidence.append(Document(page_content=doc.page_content, metadata=metadata))
        if len(evidence) == top_k:
            break

    return evidence

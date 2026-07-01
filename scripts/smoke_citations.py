"""Run statement extraction and evidence retrieval for one question."""

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from citation.evidence_retriever import retrieve_statement_evidence
from citation.statement_extractor import extract_statements
from rag_core import MedicalRAG, doc_key


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Print a baseline answer, generation context, atomic statements, "
            "and statement-specific evidence."
        )
    )
    parser.add_argument("question", help="Question to send to the RAG application")
    args = parser.parse_args()

    rag = MedicalRAG()
    trace = rag.answer_with_trace(args.question)
    statements = extract_statements(trace.answer, llm=rag.llm)
    generation_context_ids = {
        doc.metadata.get("chunk_id") or doc_key(doc)
        for doc in trace.retrieved_docs
    }
    statement_evidence = []
    for statement in statements:
        candidates = retrieve_statement_evidence(
            statement,
            rag,
            generation_context_ids=generation_context_ids,
        )
        statement_evidence.append(
            {
                "statement": statement,
                "candidates": [
                    {
                        "chunk_id": doc.metadata["chunk_id"],
                        "source": doc.metadata.get("source")
                        or doc.metadata.get("source_file"),
                        "page": doc.metadata["page"],
                        "retrieval_score": doc.metadata["retrieval_score"],
                        "in_generation_context": doc.metadata[
                            "in_generation_context"
                        ],
                        "text": doc.page_content,
                    }
                    for doc in candidates
                ],
            }
        )

    print(
        json.dumps(
            {
                "question": args.question,
                "answer": trace.answer,
                "generation_context_ids": sorted(generation_context_ids),
                "statements": statements,
                "statement_evidence": statement_evidence,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

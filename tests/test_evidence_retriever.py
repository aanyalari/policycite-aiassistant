import unittest

from langchain_core.documents import Document

from citation.evidence_retriever import retrieve_statement_evidence


def _doc(chunk_id, text=None, *, page=0, source="policy.pdf"):
    return Document(
        page_content=text or f"Evidence in {chunk_id}",
        metadata={
            "chunk_id": chunk_id,
            "source": f"policy_pdfs/{source}",
            "source_file": source,
            "page": page,
        },
    )


class _RAG:
    def __init__(self, vector_docs, bm25_docs):
        self.vector_docs = vector_docs
        self.bm25_docs = bm25_docs
        self.queries = []

    def _vector_mmr_search(self, query):
        self.queries.append(("vector", query))
        return self.vector_docs

    def _bm25_search(self, query):
        self.queries.append(("bm25", query))
        return self.bm25_docs


class EvidenceRetrieverTests(unittest.TestCase):
    def test_searches_with_statement_and_returns_rrf_ranked_top_three(self):
        shared = _doc("shared", page=4)
        rag = _RAG(
            [shared, _doc("vector-only"), _doc("fourth")],
            [shared, _doc("bm25-only"), _doc("fifth")],
        )

        result = retrieve_statement_evidence(
            "Claims use the CMS-1500 form.",
            rag,
            generation_context_ids={"shared"},
        )

        self.assertEqual(
            rag.queries,
            [
                ("vector", "Claims use the CMS-1500 form."),
                ("bm25", "Claims use the CMS-1500 form."),
            ],
        )
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].metadata["chunk_id"], "shared")
        self.assertTrue(result[0].metadata["in_generation_context"])
        self.assertFalse(result[1].metadata["in_generation_context"])
        self.assertGreater(
            result[0].metadata["retrieval_score"],
            result[1].metadata["retrieval_score"],
        )

    def test_deduplicates_chunks_and_preserves_citation_metadata(self):
        vector_copy = _doc("same", text="Dense copy", page=2)
        bm25_copy = _doc("same", text="Sparse copy", page=2)
        rag = _RAG([vector_copy], [bm25_copy])

        result = retrieve_statement_evidence("A policy statement.", rag)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].metadata["chunk_id"], "same")
        self.assertEqual(result[0].metadata["source"], "policy_pdfs/policy.pdf")
        self.assertEqual(result[0].metadata["source_file"], "policy.pdf")
        self.assertEqual(result[0].metadata["page"], 2)
        self.assertIn("retrieval_score", result[0].metadata)

    def test_skips_unusable_passages_without_mutating_retrieved_documents(self):
        unusable = Document(page_content="", metadata={"chunk_id": "bad"})
        valid = _doc("valid")
        rag = _RAG([unusable, valid], [])

        result = retrieve_statement_evidence("A policy statement.", rag)
        result[0].metadata["page"] = 99

        self.assertEqual([doc.metadata["chunk_id"] for doc in result], ["valid"])
        self.assertEqual(valid.metadata["page"], 0)

    def test_empty_statement_candidates_and_nonpositive_limit_are_safe(self):
        rag = _RAG([], [])

        self.assertEqual(retrieve_statement_evidence("", rag), [])
        self.assertEqual(retrieve_statement_evidence("Statement", rag), [])
        self.assertEqual(retrieve_statement_evidence("Statement", rag, top_k=0), [])


if __name__ == "__main__":
    unittest.main()

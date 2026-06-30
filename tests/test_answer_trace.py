import unittest

from langchain_core.documents import Document

from rag_core import AnswerResult, MedicalRAG, doc_key


class _Response:
    content = "Claims must be filed within one calendar year. [S1]"


class _LLM:
    def invoke(self, _prompt):
        return _Response()


class AnswerTraceTests(unittest.TestCase):
    def setUp(self):
        self.source_doc = Document(
            page_content="Claims must be filed within one calendar year.",
            metadata={
                "chunk_id": "policy-chunk-1",
                "source_file": "medicare_timely_filing_mln.pdf",
                "page": 0,
            },
        )
        self.rag = MedicalRAG.__new__(MedicalRAG)
        self.rag.llm = _LLM()
        self.retrieval_calls = 0

        def retrieve(_query, target_disease=None):
            self.retrieval_calls += 1
            return [self.source_doc], 0.77

        self.rag.hybrid_retrieve = retrieve

    def test_answer_preserves_result_shape_and_retrieves_once(self):
        result = self.rag.answer("What is Medicare's timely filing requirement?")

        self.assertIsInstance(result, AnswerResult)
        self.assertEqual(self.retrieval_calls, 1)
        self.assertEqual(
            result.answer,
            "Claims must be filed within one calendar year. [S1]",
        )
        self.assertEqual(
            result.sources,
            ["medicare_timely_filing_mln.pdf (page 1)"],
        )
        self.assertEqual(result.confidence, 0.77)
        self.assertEqual(
            result.normalized_query,
            "what is medicare's timely filing requirement?",
        )
        self.assertFalse(hasattr(result, "retrieved_docs"))

    def test_trace_contains_documents_passed_to_generation(self):
        seen_docs = []
        original_build_context = self.rag._build_context

        def capture_context(docs):
            seen_docs.extend(docs)
            return original_build_context(docs)

        self.rag._build_context = capture_context

        trace = self.rag.answer_with_trace(
            "What is Medicare's timely filing requirement?"
        )

        self.assertEqual(self.retrieval_calls, 1)
        self.assertEqual(
            [doc_key(doc) for doc in trace.retrieved_docs],
            [doc_key(doc) for doc in seen_docs],
        )
        self.assertEqual(
            trace.retrieved_docs[0].metadata["chunk_id"],
            "policy-chunk-1",
        )
        self.assertEqual(
            trace.sources,
            ["medicare_timely_filing_mln.pdf (page 1)"],
        )

    def test_trace_documents_are_defensive_copies(self):
        first = self.rag.answer_with_trace(
            "What is Medicare's timely filing requirement?"
        )
        first.retrieved_docs[0].page_content = "changed"
        first.retrieved_docs[0].metadata["page"] = 99

        second = self.rag.answer_with_trace(
            "What is Medicare's timely filing requirement?"
        )

        self.assertEqual(
            second.retrieved_docs[0].page_content,
            "Claims must be filed within one calendar year.",
        )
        self.assertEqual(second.retrieved_docs[0].metadata["page"], 0)
        self.assertEqual(self.source_doc.metadata["page"], 0)

    def test_fallback_document_key_is_stable_and_content_sensitive(self):
        first = Document(
            page_content="same content",
            metadata={"source_file": "policy.pdf", "page": 2},
        )
        same = Document(
            page_content="same content",
            metadata={"source_file": "policy.pdf", "page": 2},
        )
        changed = Document(
            page_content="different content",
            metadata={"source_file": "policy.pdf", "page": 2},
        )

        self.assertEqual(doc_key(first), doc_key(same))
        self.assertNotEqual(doc_key(first), doc_key(changed))


if __name__ == "__main__":
    unittest.main()

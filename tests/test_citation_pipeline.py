import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.documents import Document

from citation.pipeline import answer_with_citations
from citation.schemas import AttributionVerdict, CitedStatement, EvidenceCitation


class _RAG:
    def __init__(self, answer="First fact. Second fact."):
        self.trace = SimpleNamespace(
            answer=answer,
            sources=["policy.pdf (page 1)"],
            retrieved_docs=[
                Document(
                    page_content="Generation evidence.",
                    metadata={
                        "chunk_id": "generation-1",
                        "source_file": "policy.pdf",
                        "page": 0,
                    },
                )
            ],
        )

    def answer_with_trace(self, question):
        self.question = question
        return self.trace


class CitationPipelineTests(unittest.TestCase):
    @patch("citation.pipeline.judge_attribution")
    @patch("citation.pipeline.retrieve_statement_evidence")
    @patch("citation.pipeline.extract_statements")
    def test_connects_each_statement_to_retrieval_and_attribution(
        self, extract, retrieve, judge
    ):
        extract.return_value = ["First fact.", "Second fact."]
        first_doc = Document(page_content="First evidence.", metadata={})
        second_doc = Document(page_content="Second evidence.", metadata={})
        retrieve.side_effect = [[first_doc], [second_doc]]
        judge.side_effect = [
            CitedStatement(
                text="First fact.",
                verdict=AttributionVerdict.SUPPORTED,
                citations=[
                    EvidenceCitation(
                        citation_id="evidence-1",
                        source="policy.pdf",
                        page=1,
                        evidence_excerpt="First evidence.",
                    )
                ],
            ),
            CitedStatement(
                text="Second fact.",
                verdict=AttributionVerdict.NOT_SUPPORTED,
                reason="The evidence does not cover the qualifier.",
            ),
        ]
        rag = _RAG()
        llm = object()

        result = answer_with_citations("Question?", rag, llm=llm)

        self.assertEqual(rag.question, "Question?")
        extract.assert_called_once_with(rag.trace.answer, llm=llm)
        self.assertEqual(retrieve.call_count, 2)
        for call in retrieve.call_args_list:
            self.assertEqual(call.kwargs["top_k"], 3)
            self.assertEqual(
                call.kwargs["generation_context_ids"], ["generation-1"]
            )
        self.assertEqual(
            [call.args for call in judge.call_args_list],
            [("First fact.", [first_doc]), ("Second fact.", [second_doc])],
        )
        self.assertTrue(all(call.kwargs["llm"] is llm for call in judge.call_args_list))
        self.assertEqual(result.generation_context_ids, ["generation-1"])
        self.assertEqual(result.baseline_sources, ["policy.pdf (page 1)"])
        self.assertEqual(result.citation_coverage, 0.5)
        self.assertTrue(result.needs_human_review)
        self.assertGreaterEqual(result.latency_ms, 0.0)

    @patch("citation.pipeline.extract_statements", return_value=[])
    def test_no_factual_statements_has_zero_coverage_without_review(self, _extract):
        result = answer_with_citations("Hello", _RAG(answer="Hello!"))

        self.assertEqual(result.statements, [])
        self.assertEqual(result.citation_coverage, 0.0)
        self.assertFalse(result.needs_human_review)


if __name__ == "__main__":
    unittest.main()

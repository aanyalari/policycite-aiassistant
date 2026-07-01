import unittest
from types import SimpleNamespace
from unittest.mock import patch

import api
from citation.schemas import CitationTrace


class _BaselineRAG:
    def answer(self, question):
        self.question = question
        return SimpleNamespace(
            answer="Baseline answer.",
            sources=["policy.pdf (page 1)"],
            confidence=0.8,
            normalized_query="question?",
        )


class APITests(unittest.TestCase):
    def test_ask_response_shape_remains_unchanged(self):
        rag = _BaselineRAG()

        with patch.object(api, "rag", rag):
            result = api.ask(api.Query(question="Question?"))

        self.assertEqual(rag.question, "Question?")
        self.assertEqual(
            result,
            {
                "answer": "Baseline answer.",
                "sources": ["policy.pdf (page 1)"],
                "confidence": 0.8,
                "normalized_query": "question?",
            },
        )

    def test_ask_cited_returns_citation_trace(self):
        expected = CitationTrace(
            question="Question?",
            answer="Cited answer.",
            latency_ms=1.5,
        )

        with patch.object(
            api, "answer_with_citations", return_value=expected
        ) as pipeline:
            result = api.ask_cited(api.Query(question="Question?"))

        pipeline.assert_called_once_with("Question?", api.rag)
        self.assertIs(result, expected)


if __name__ == "__main__":
    unittest.main()

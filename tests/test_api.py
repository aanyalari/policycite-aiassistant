import unittest
from types import SimpleNamespace
from unittest.mock import patch

import api
from langchain_core.documents import Document
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


class _OriginalBaselineRAG:
    def answer_with_trace_original(self, question):
        self.question = question
        return SimpleNamespace(
            answer="Original baseline answer.",
            sources=["policy.pdf (page 3)"],
            confidence=0.7,
            normalized_query="question?",
            retrieved_docs=[],
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

    def test_live_demo_audits_exact_original_baseline_trace(self):
        replay_rag = _OriginalBaselineRAG()
        expected = CitationTrace(
            question="Question?",
            answer="Original baseline answer.",
            latency_ms=10.0,
        )

        with (
            patch.object(api, "rag", replay_rag),
            patch.object(
                api,
                "answer_with_citations",
                return_value=expected,
            ) as pipeline,
        ):
            result = api.audit_live_original_baseline(
                api.Query(question="Question?")
            )

        self.assertEqual(replay_rag.question, "Question?")
        pipeline.assert_called_once()
        self.assertIs(
            pipeline.call_args.kwargs["answer_trace"].answer,
            result.baseline.answer,
        )
        self.assertEqual(result.baseline_mode, "original_pre_correction")
        self.assertEqual(result.policycite.answer, "Original baseline answer.")

    def test_replay_audits_saved_rag_answer_with_current_pipeline(self):
        generation_doc = Document(
            page_content="Expedited requests take 72 hours.",
            metadata={"chunk_id": "saved-context"},
        )
        replay_rag = SimpleNamespace(corpus_docs=[generation_doc])
        preserved = {
            "created_at": "2026-07-01T21:06:13+00:00",
        }
        case = {
            "id": "policy-003",
            "question": "How quickly?",
            "baseline": {
                "output": {
                    "answer": "Saved RAG answer.",
                    "sources": ["policy.pdf (page 3)"],
                    "confidence": 0.95,
                    "normalized_query": "how quickly?",
                    "latency_ms": 1200.0,
                }
            },
            "policycite_rag": {
                "output": {"generation_context_ids": ["saved-context"]}
            },
        }
        expected = CitationTrace(
            question="How quickly?",
            answer="Saved RAG answer.",
            latency_ms=20.0,
        )

        with (
            patch.object(api, "rag", replay_rag),
            patch.object(
                api,
                "_load_preserved_demo_case",
                return_value=(preserved, case),
            ),
            patch.object(
                api,
                "answer_with_citations",
                return_value=expected,
            ) as pipeline,
        ):
            result = api.replay_preserved_rag_failure()

        pipeline.assert_called_once()
        question, used_rag = pipeline.call_args.args
        trace = pipeline.call_args.kwargs["answer_trace"]
        self.assertEqual(question, "How quickly?")
        self.assertIs(used_rag, replay_rag)
        self.assertEqual(trace.answer, "Saved RAG answer.")
        self.assertEqual(trace.retrieved_docs, [generation_doc])
        self.assertEqual(result.provenance.case_id, "policy-003")
        self.assertEqual(result.baseline.answer, "Saved RAG answer.")
        self.assertIs(result.policycite, expected)


if __name__ == "__main__":
    unittest.main()

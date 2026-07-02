import unittest
from types import SimpleNamespace

from citation.schemas import AttributionVerdict, CitationTrace, CitedStatement, EvidenceCitation
from evaluation.comparative import run_comparison, score_artifact


def _questions():
    return [
        {
            "id": f"q-{index}", "question": f"Question {index}?", "category": "directly_answerable",
            "expected_answerable": True, "reference_answer": "A fact.",
            "required_claims": ["A fact."], "gold_evidence": [], "notes": "",
        }
        for index in range(10)
    ]


class _RAG:
    def __init__(self):
        self.calls = 0

    def answer_with_trace(self, question):
        self.calls += 1
        return SimpleNamespace(
            answer="A fact.", sources=["policy.pdf (page 1)"], confidence=0.8,
            normalized_query=question.lower(), retrieved_docs=[],
        )


def _cited(question, rag, answer_trace=None):
    assert answer_trace is not None
    return CitationTrace(
        question=question, answer="A fact.", baseline_sources=["policy.pdf (page 1)"],
        statements=[CitedStatement(
            text="A fact.", verdict=AttributionVerdict.SUPPORTED,
            citations=[EvidenceCitation(
                citation_id="chunk-1", source="policy.pdf", page=0,
                evidence_excerpt="A fact.",
            )],
        )], citation_coverage=1.0, latency_ms=20.0,
    )


class ComparativeEvaluationTests(unittest.TestCase):
    def test_run_keeps_human_and_automated_labels_separate(self):
        rag = _RAG()
        artifact = run_comparison(
            _questions(), rag, extractor=lambda answer: [answer], cited_answerer=_cited
        )

        self.assertEqual(artifact["status"], "awaiting_human_labels")
        self.assertIsNone(artifact["aggregate_metrics"])
        cited = artifact["results"][0]["policycite_rag"]
        self.assertEqual(cited["output"]["statements"][0]["verdict"], "SUPPORTED")
        self.assertIsNone(cited["human_labels"]["statements"][0]["fully_supported"])
        self.assertEqual(rag.calls, 10)
        self.assertEqual(
            artifact["results"][0]["baseline"]["output"]["answer"],
            cited["output"]["answer"],
        )

    def test_score_requires_complete_human_labels_and_reports_counts(self):
        artifact = run_comparison(
            _questions(), _RAG(), extractor=lambda answer: [answer], cited_answerer=_cited
        )
        with self.assertRaisesRegex(ValueError, "Human labels are incomplete"):
            score_artifact(artifact)

        for result in artifact["results"]:
            for condition in ("baseline", "policycite_rag"):
                labels = result[condition]["human_labels"]
                labels["required_claims"][0]["retained"] = True
                labels["statements"][0]["fully_supported"] = True
                labels["citations"][0]["contributes_support"] = True

        scored = score_artifact(artifact)
        self.assertEqual(scored["status"], "complete")
        metrics = scored["aggregate_metrics"]["policycite_rag"]
        self.assertEqual(metrics["citation_coverage"]["numerator"], 10)
        self.assertEqual(metrics["citation_coverage"]["denominator"], 10)
        self.assertEqual(metrics["citation_f1"], 1.0)
        self.assertEqual(
            scored["aggregate_metrics"]["median_latency_overhead_ms"], 20.0
        )

    def test_checkpoints_and_resumes_without_repeating_saved_questions(self):
        checkpoints = []
        progress = []
        artifact = run_comparison(
            _questions(), _RAG(), extractor=lambda answer: [answer],
            cited_answerer=_cited, limit=1,
            checkpoint=lambda data: checkpoints.append(len(data["results"])),
            progress=progress.append,
        )
        self.assertEqual(checkpoints, [1])
        self.assertEqual(artifact["status"], "running")
        self.assertTrue(any("[1/10] q-0: baseline started" in item for item in progress))

        run_comparison(
            _questions(), _RAG(), extractor=lambda answer: [answer],
            cited_answerer=_cited, artifact=artifact, limit=1,
        )
        self.assertEqual([item["id"] for item in artifact["results"]], ["q-0", "q-1"])

    def test_condition_errors_are_preserved_and_block_scoring(self):
        def fail(_question, _rag, answer_trace=None):
            raise RuntimeError("model unavailable")

        artifact = run_comparison(
            _questions(), _RAG(), extractor=lambda answer: [answer], cited_answerer=fail
        )
        failed = artifact["results"][0]["policycite_rag"]
        self.assertEqual(failed["status"], "error")
        self.assertEqual(failed["error"]["type"], "RuntimeError")
        self.assertEqual(failed["error"]["message"], "model unavailable")
        with self.assertRaisesRegex(ValueError, "execution errors"):
            score_artifact(artifact)

    def test_score_rejects_supported_statement_without_supporting_citation(self):
        artifact = run_comparison(
            _questions(), _RAG(), extractor=lambda answer: [answer], cited_answerer=_cited
        )
        for result in artifact["results"]:
            for condition in ("baseline", "policycite_rag"):
                labels = result[condition]["human_labels"]
                labels["required_claims"][0]["retained"] = True
                labels["statements"][0]["fully_supported"] = True
                labels["citations"][0]["contributes_support"] = True
        artifact["results"][0]["policycite_rag"]["human_labels"]["citations"][0][
            "contributes_support"
        ] = False

        with self.assertRaisesRegex(
            ValueError, "must have at least one attached citation"
        ):
            score_artifact(artifact)


if __name__ == "__main__":
    unittest.main()

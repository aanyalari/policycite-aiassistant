import unittest
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest


class StreamlitAppTests(unittest.TestCase):
    def test_live_mode_renders_original_baseline_beside_policycite_audit(self):
        health_response = Mock(ok=True)
        live_response = Mock()
        live_response.raise_for_status.return_value = None
        live_response.json.return_value = {
            "question": "How quickly?",
            "baseline": {
                "answer": "Original baseline answer with a wrong conclusion.",
                "sources": ["policy.pdf (page 3)"],
                "confidence": 0.95,
                "normalized_query": "how quickly?",
                "latency_ms": 1200.0,
            },
            "policycite": {
                "question": "How quickly?",
                "answer": "Original baseline answer with a wrong conclusion.",
                "baseline_sources": ["policy.pdf (page 3)"],
                "generation_context_ids": ["saved-context"],
                "statements": [
                    {
                        "text": "Correct statement.",
                        "verdict": "SUPPORTED",
                        "citations": [],
                    },
                    {
                        "text": "Wrong conclusion.",
                        "verdict": "NOT_SUPPORTED",
                        "citations": [],
                        "reason": "The timeframes differ.",
                    },
                ],
                "citation_coverage": 0.5,
                "needs_human_review": True,
                "latency_ms": 250.0,
            },
            "baseline_mode": "original_pre_correction",
            "description": "Original baseline audited live.",
        }

        with (
            patch("requests.get", return_value=health_response),
            patch("requests.post", return_value=live_response) as post,
        ):
            app = AppTest.from_file("app.py").run()
            app.chat_input[0].set_value("How quickly?").run()

        self.assertEqual(list(app.exception), [])
        post.assert_called_once()
        self.assertTrue(post.call_args.args[0].endswith("/demo/live"))
        markdown = "\n".join(item.value for item in app.markdown)
        self.assertIn("Baseline answer", markdown)
        self.assertIn("Current PolicyCite audit", markdown)
        self.assertIn("Original baseline answer with a wrong conclusion", markdown)
        metrics = {item.label: item.value for item in app.metric}
        self.assertEqual(metrics["Statements supported"], "1 of 2")
        self.assertNotIn("Review status", metrics)


if __name__ == "__main__":
    unittest.main()

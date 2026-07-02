import unittest

import requests

from ui_helpers import (
    api_error_message,
    evidence_location,
    format_coverage,
    format_latency,
    statement_support_summary,
)


class UIHelperTests(unittest.TestCase):
    def test_formats_metrics(self):
        self.assertEqual(format_coverage(0.5), "50%")
        self.assertEqual(format_coverage(2), "100%")
        self.assertEqual(format_coverage(None), "0%")
        self.assertEqual(format_latency(1234.6), "1,235 ms")
        self.assertEqual(format_latency(None), "Not available")

    def test_builds_human_readable_evidence_location(self):
        self.assertEqual(
            evidence_location({"source": "policy.pdf", "page": 7}),
            "policy.pdf — page 7",
        )
        self.assertEqual(evidence_location({}), "Unknown document")

    def test_summarizes_supported_statements(self):
        self.assertEqual(
            statement_support_summary(
                [
                    {"verdict": "SUPPORTED"},
                    {"verdict": "NOT_SUPPORTED"},
                    {"verdict": "SUPPORTED"},
                ]
            ),
            "2 of 3",
        )
        self.assertEqual(statement_support_summary(None), "0 of 0")

    def test_translates_request_errors(self):
        self.assertIn("too long", api_error_message(requests.exceptions.Timeout()))
        self.assertIn(
            "Start the FastAPI backend",
            api_error_message(requests.exceptions.ConnectionError()),
        )


if __name__ == "__main__":
    unittest.main()

import unittest

from langchain_core.documents import Document

from citation.attribution import judge_attribution
from citation.schemas import AttributionVerdict


def _doc(chunk_id="candidate-1", *, page=0):
    return Document(
        page_content=(
            "Medicare claims must be filed no later than 12 months after "
            "the date of service. Certain exceptions may apply."
        ),
        metadata={
            "chunk_id": chunk_id,
            "source": "policy_pdfs/filing.pdf",
            "source_file": "filing.pdf",
            "page": page,
            "retrieval_score": 0.75,
        },
    )


class _StructuredJudge:
    def __init__(self, result):
        self.result = result

    def invoke(self, messages):
        return self.result


class _LLM:
    def __init__(self, result):
        self.result = result

    def with_structured_output(self, schema):
        return _StructuredJudge(self.result)


class AttributionTests(unittest.TestCase):
    def test_fully_supported_statement_receives_valid_precise_citation(self):
        result = judge_attribution(
            "Medicare claims must be filed no later than 12 months after service.",
            [_doc(page=2)],
            llm=_LLM(
                {
                    "verdict": "SUPPORTED",
                    "citation_ids": ["candidate-1"],
                    "reason": "The passage states the filing limit.",
                }
            ),
        )

        self.assertEqual(result.verdict, AttributionVerdict.SUPPORTED)
        self.assertEqual(result.citations[0].source, "filing.pdf")
        self.assertEqual(result.citations[0].page, 3)
        self.assertEqual(result.citations[0].retrieval_score, 0.75)
        self.assertEqual(
            result.citations[0].evidence_excerpt,
            "Medicare claims must be filed no later than 12 months after "
            "the date of service.",
        )

    def test_unsupported_qualifier_is_not_supported(self):
        result = judge_attribution(
            "Medicare claims must be filed within 12 months without exception.",
            [_doc()],
            llm=_LLM(
                {
                    "verdict": "NOT_SUPPORTED",
                    "citation_ids": [],
                    "reason": "The passage says exceptions may apply.",
                }
            ),
        )

        self.assertEqual(result.verdict, AttributionVerdict.NOT_SUPPORTED)
        self.assertEqual(result.citations, [])

    def test_empty_evidence_does_not_call_a_model(self):
        result = judge_attribution("A statement.", [], llm=None)

        self.assertEqual(result.verdict, AttributionVerdict.NOT_SUPPORTED)
        self.assertIn("No usable evidence", result.reason)

    def test_invented_candidate_id_is_rejected(self):
        result = judge_attribution(
            "The filing deadline is one year.",
            [_doc()],
            llm=_LLM(
                {
                    "verdict": "SUPPORTED",
                    "citation_ids": ["invented-id"],
                    "reason": "Supported.",
                }
            ),
        )

        self.assertEqual(result.verdict, AttributionVerdict.NOT_SUPPORTED)
        self.assertEqual(result.citations, [])

    def test_near_verbatim_support_recovers_model_false_negative(self):
        result = judge_attribution(
            "Medicare claims must be filed no later than 12 months after the date of service.",
            [_doc()],
            llm=_LLM(
                {
                    "verdict": "NOT_SUPPORTED",
                    "citation_ids": [],
                    "reason": "Model missed the passage.",
                }
            ),
        )

        self.assertEqual(result.verdict, AttributionVerdict.SUPPORTED)
        self.assertEqual(len(result.citations), 1)

    def test_supported_result_keeps_only_strongest_citation(self):
        weaker = _doc("candidate-2", page=5)
        weaker.page_content = "Medicare provides health coverage information."
        result = judge_attribution(
            "Medicare claims have a 12 month requirement.",
            [_doc(), weaker],
            llm=_LLM(
                {
                    "verdict": "SUPPORTED",
                    "citation_ids": ["candidate-2", "candidate-1"],
                    "reason": "Supported.",
                }
            ),
        )

        self.assertEqual(len(result.citations), 1)
        self.assertEqual(result.citations[0].citation_id, "candidate-1")

    def test_excerpt_is_copied_from_passage_not_generated_by_model(self):
        result = judge_attribution(
            "Medicare claims must be filed within 12 months.",
            [_doc()],
            llm=_LLM(
                {
                    "verdict": "SUPPORTED",
                    "citation_ids": ["candidate-1"],
                    "reason": "Supported.",
                }
            ),
        )

        self.assertEqual(result.verdict, AttributionVerdict.SUPPORTED)
        self.assertIn(result.citations[0].evidence_excerpt, _doc().page_content)


if __name__ == "__main__":
    unittest.main()

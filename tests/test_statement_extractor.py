import unittest

from citation.statement_extractor import extract_statements


class _StructuredInvoker:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def invoke(self, _prompt):
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class _LLM:
    def __init__(self, result):
        self.invoker = _StructuredInvoker(result)

    def with_structured_output(self, _schema):
        return self.invoker


class StatementExtractorTests(unittest.TestCase):
    def test_two_fact_answer_yields_two_statements(self):
        answer = (
            "Medicare claims generally must be filed within one calendar year. "
            "The CMS-1500 form is used to bill Medicare for professional services."
        )

        statements = extract_statements(answer)

        self.assertEqual(len(statements), 2)
        self.assertIn("one calendar year", statements[0])
        self.assertIn("CMS-1500", statements[1])

    def test_structured_output_splits_compound_claim_and_preserves_qualifiers(self):
        llm = _LLM(
            {
                "statements": [
                    "Medicare claims generally must be filed within one calendar year after the date of service.",
                    "The one-year requirement may have specified exceptions.",
                ]
            }
        )

        statements = extract_statements(
            "Medicare claims generally must be filed within one calendar year after the date of service, but specified exceptions may apply.",
            llm=llm,
        )

        self.assertEqual(llm.invoker.calls, 1)
        self.assertEqual(len(statements), 2)
        self.assertIn("generally must", statements[0])
        self.assertIn("may", statements[1])

    def test_disclaimer_heading_and_citations_are_excluded(self):
        answer = """### Answer
Claims must be filed within one calendar year. [S1]

### Safety disclaimer
This information is not a substitute for professional medical advice.
"""

        statements = extract_statements(answer)

        self.assertEqual(
            statements,
            ["Claims must be filed within one calendar year."],
        )

    def test_malformed_llm_output_uses_fallback_without_crashing(self):
        llm = _LLM("not valid json")

        statements = extract_statements(
            "Claims must be filed within one year. Electronic claims use the 837P format.",
            llm=llm,
        )

        self.assertEqual(llm.invoker.calls, 1)
        self.assertEqual(len(statements), 2)

    def test_structured_output_cannot_silently_drop_a_factual_sentence(self):
        answer = """Based on the provided policy sources, I found the following:

1. Medicare requires timely filing of claims within 60 days of service. [S1]
2. Claims must be filed promptly to minimize delays. [S1]
3. A study found that delayed claims increased healthcare costs. [S2]

Therefore, I have written: Not found in documents.
Note: The provided sources do not mention a specific requirement.
"""
        llm = _LLM(
            {
                "statements": [
                    "Medicare requires timely filing of claims within 60 days of service.",
                    "Claims must be filed promptly to minimize delays.",
                ]
            }
        )

        statements = extract_statements(answer, llm=llm)

        self.assertEqual(llm.invoker.calls, 1)
        self.assertEqual(len(statements), 3)
        self.assertIn("A study found", statements[2])
        self.assertFalse(any("Not found" in item for item in statements))
        self.assertFalse(any(item.startswith("Note:") for item in statements))

    def test_coverage_reconciliation_ignores_attribution_prefixes(self):
        answer = """1. The Centers for Medicare & Medicaid Services (CMS) states that "Medicare requires timely filing of claims within 60 days of service" [S1].
2. The CMS also notes that "claims must be filed in a timely manner to ensure prompt payment and minimize delays" [S1].
3. A study published in the Journal of General Internal Medicine found that delayed claim submission increased healthcare costs. [S2]
"""
        llm = _LLM(
            {
                "statements": [
                    "Medicare requires timely filing of claims within 60 days of service.",
                    "Claims must be filed in a timely manner to ensure prompt payment and minimize delays.",
                ]
            }
        )

        statements = extract_statements(answer, llm=llm)

        self.assertEqual(len(statements), 3)
        self.assertEqual(
            sum("60 days" in item for item in statements),
            1,
        )
        self.assertIn("Journal of General Internal Medicine", statements[2])

    def test_complete_citation_suffix_is_removed(self):
        answer = (
            "Claims must be filed within one calendar year. [S2] "
            "medicare_timely_filing_mln.pdf (page 23)"
        )

        statements = extract_statements(answer)

        self.assertEqual(
            statements,
            ["Claims must be filed within one calendar year."],
        )

    def test_reference_fragments_are_rejected_and_full_sentence_is_recovered(self):
        answer = (
            "Medicare regulations at 42 CFR 424.44 define the timely-filing "
            "period as 12 months. [S1] medicare_timely_filing_mln.pdf (page 20)"
        )
        llm = _LLM(
            {
                "statements": [
                    "Medicare regulations at 42 CFR 424.44",
                    "Section 6404 of the Affordable Care Act (ACA)",
                ]
            }
        )

        statements = extract_statements(answer, llm=llm)

        self.assertEqual(llm.invoker.calls, 1)
        self.assertEqual(
            statements,
            [
                "Medicare regulations at 42 CFR 424.44 define the "
                "timely-filing period as 12 months."
            ],
        )

    def test_global_token_union_does_not_hide_an_unrelated_omission(self):
        answer = (
            "Medicare requires claims within one year. "
            "A separate study reported longer hospital stays."
        )
        llm = _LLM(
            {
                "statements": [
                    "Medicare requires claims within one year.",
                    "Hospitals submit separate Medicare claims.",
                ]
            }
        )

        statements = extract_statements(answer, llm=llm)

        self.assertTrue(
            any("longer hospital stays" in statement for statement in statements)
        )

    def test_duplicate_and_non_factual_responses_are_handled(self):
        duplicate = (
            "Claims must be filed within one year. "
            "Claims must be filed within one year."
        )

        self.assertEqual(len(extract_statements(duplicate)), 1)
        self.assertEqual(extract_statements("Not found in documents."), [])
        self.assertEqual(extract_statements("Hello! Please ask a question."), [])
        self.assertEqual(extract_statements(""), [])

    def test_therefore_conclusion_is_cleaned_and_kept_for_attribution(self):
        answer = (
            "Therefore, the correct answer is: The Medicare timely filing "
            "requirement is no longer as strict as it was before January 1, 2010."
        )

        self.assertEqual(
            extract_statements(answer),
            [
                "The Medicare timely filing requirement is no longer as strict "
                "as it was before January 1, 2010."
            ],
        )


if __name__ == "__main__":
    unittest.main()

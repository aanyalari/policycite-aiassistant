import unittest

from langchain_core.documents import Document

from rag_core import (
    AnswerResult,
    MedicalRAG,
    _policy_query_has_evidence,
    _policy_answer_is_clean,
    _rerank_policy_docs,
    doc_key,
)


class _Response:
    content = "Claims must be filed within one calendar year. [S1]"


class _LLM:
    def invoke(self, _prompt):
        return _Response()


class _RecordingLLM:
    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
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

    def test_policy_trace_excludes_retrieved_documents_not_sent_to_generation(self):
        retrieved = [
            Document(
                page_content=(
                    "Medicare claims must be filed within one calendar year. "
                    f"Passage {index}."
                ),
                metadata={
                    "chunk_id": f"policy-chunk-{index}",
                    "source_file": "medicare_timely_filing_mln.pdf",
                    "page": index,
                },
            )
            for index in range(8)
        ]
        self.rag.hybrid_retrieve = lambda _query, target_disease=None: (
            retrieved,
            0.77,
        )
        seen_docs = []
        original_build_context = self.rag._build_context

        def capture_context(docs):
            seen_docs.extend(docs)
            return original_build_context(docs)

        self.rag._build_context = capture_context
        trace = self.rag.answer_with_trace(
            "What is Medicare's timely filing requirement?"
        )

        self.assertEqual(len(seen_docs), 4)
        self.assertEqual(len(trace.retrieved_docs), 4)
        self.assertEqual(
            [doc_key(doc) for doc in trace.retrieved_docs],
            [doc_key(doc) for doc in seen_docs],
        )

    def test_original_policy_baseline_uses_pre_correction_prompt_and_full_context(self):
        retrieved = [
            Document(
                page_content=(
                    "Medicare claims must be filed within one calendar year. "
                    f"Historical passage {index}."
                ),
                metadata={
                    "chunk_id": f"legacy-policy-{index}",
                    "source_file": "medicare_timely_filing_mln.pdf",
                    "page": index,
                },
            )
            for index in range(8)
        ]
        self.rag.hybrid_retrieve = lambda _query, target_disease=None: (
            retrieved,
            0.77,
        )
        recording_llm = _RecordingLLM()
        self.rag.llm = recording_llm

        trace = self.rag.answer_with_trace_original(
            "What is Medicare's timely filing requirement?"
        )

        self.assertEqual(len(trace.retrieved_docs), 8)
        self.assertEqual(len(recording_llm.prompts), 1)
        self.assertNotIn("at most four short sentences", recording_llm.prompts[0])
        self.assertNotIn("Do not reproduce headings", recording_llm.prompts[0])

    def test_policy_abstention_has_no_generation_context(self):
        retrieved = [
            Document(
                page_content="Prior authorization decisions are due within seven days.",
                metadata={"source_file": "prior-auth.pdf", "page": 0},
            ),
            Document(
                page_content="Emergency hospital claims follow timely filing rules.",
                metadata={"source_file": "filing.pdf", "page": 0},
            ),
        ]
        self.rag.hybrid_retrieve = lambda _query, target_disease=None: (
            retrieved,
            0.77,
        )

        trace = self.rag.answer_with_trace(
            "Are emergency services exempt from prior authorization?"
        )

        self.assertEqual(trace.answer, "Not found in documents.")
        self.assertEqual(trace.retrieved_docs, [])

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

    def test_policy_reranking_prioritizes_specific_query_concepts(self):
        generic = Document(
            page_content="Medicare claims have filing requirements.",
            metadata={"source_file": "generic.pdf", "page": 0},
        )
        specific = Document(
            page_content="DMEPOS suppliers submit to the DME MAC; Medicare Advantage claims go to the MA plan.",
            metadata={"source_file": "billing.pdf", "page": 1},
        )

        ranked = _rerank_policy_docs(
            [generic, specific],
            "Where should DMEPOS and Medicare Advantage claims be submitted?",
        )

        self.assertIs(ranked[0], specific)

    def test_unsupported_policy_requires_concepts_in_same_source(self):
        prior_auth = Document(
            page_content="Prior authorization decisions are due within seven days.",
            metadata={"source_file": "prior-auth.pdf", "page": 0},
        )
        emergency = Document(
            page_content="Emergency hospital claims follow the usual timely filing rule.",
            metadata={"source_file": "filing.pdf", "page": 0},
        )

        self.assertFalse(
            _policy_query_has_evidence(
                "Are emergency services exempt from prior authorization?",
                [prior_auth, emergency],
            )
        )

    def test_policy_fallback_covers_every_requested_exception_category(self):
        denial = Document(
            page_content=(
                "Contractors shall deny claims received after 12 months as "
                "untimely filed claims unless an exception applies."
            ),
            metadata={"source_file": "filing.pdf", "page": 4},
        )
        exceptions = Document(
            page_content=(
                "Exceptions include administrative error or misrepresentation, "
                "retroactive Medicare entitlement to or before the service date, "
                "qualifying State Medicaid "
                "recoupment, and retroactive disenrollment from a Medicare "
                "Advantage plan or PACE organization."
            ),
            metadata={"source_file": "filing.pdf", "page": 3},
        )

        answer = self.rag._fallback_policy_answer(
            "What happens after 12 months, and what exceptions can extend the deadline?",
            [denial, exceptions],
        )

        self.assertIn("deny claims", answer)
        self.assertIn("administrative error", answer)
        self.assertIn("State Medicaid", answer)
        self.assertIn("Medicare Advantage", answer)

    def test_policy_answer_cleanliness_rejects_pdf_headers_and_runaway_output(self):
        query = "What are the CMS-1500 and 837P used for, and how do they differ?"
        self.assertFalse(
            _policy_answer_is_clean(
                query,
                "MLN Booklet December 2025 Page 3 of 11 CMS-1500 is a paper claim form and 837P is an electronic format. [S1]",
            )
        )
        self.assertTrue(
            _policy_answer_is_clean(
                query,
                "CMS-1500 is the standard paper claim form. [S1] 837P is the standard electronic format. [S1]",
            )
        )

    def test_multi_part_prior_authorization_answer_requires_every_element(self):
        query = "What prior authorization process changes begin in 2026, including decision timing, denial notices, and public reporting?"
        incomplete = (
            "Beginning in 2026, payers must provide a specific denial reason. [S1]"
        )
        complete = (
            "Decisions are due within 72 hours for expedited requests and seven calendar days for standard requests. [S1] "
            "Beginning in 2026, payers must provide a specific reason; this does not apply to prior authorization decisions for drugs. [S1] "
            "Payers must publicly report prior authorization metrics, with the initial report due March 31, 2026. [S1]"
        )
        self.assertFalse(_policy_answer_is_clean(query, incomplete))
        self.assertTrue(_policy_answer_is_clean(query, complete))

    def test_claim_destination_requirements_include_ffs(self):
        query = "Where should Medicare FFS, DMEPOS, and Medicare Advantage claims be submitted?"
        incomplete = (
            "DMEPOS goes to the DME MAC and Medicare Advantage goes to the MA plan. [S1]"
        )
        complete = (
            "Medicare FFS goes to the MAC for the state where services were provided. [S1] "
            "DMEPOS goes to the DME MAC and Medicare Advantage goes to the MA plan. [S1]"
        )
        self.assertFalse(_policy_answer_is_clean(query, incomplete))
        self.assertTrue(_policy_answer_is_clean(query, complete))

    def test_targeted_fallback_returns_clean_claim_form_definition(self):
        billing = Document(
            page_content=(
                "MLN Booklet Medicare Billing Page 3 of 11. "
                "CMS-1500 is the standard paper claim form that non-institutional "
                "providers or suppliers use to bill Medicare Administrative Contractors. "
                "837P is the standard electronic format that health care professionals "
                "and suppliers use to submit health care claims."
            ),
            metadata={"source_file": "billing.pdf", "page": 2},
        )

        answer = self.rag._fallback_policy_answer(
            "What are the CMS-1500 and 837P used for, and how do they differ?",
            [billing],
        )

        self.assertNotIn("MLN Booklet", answer)
        self.assertIn("standard paper claim form", answer)
        self.assertIn("standard electronic format", answer)

    def test_targeted_fallback_returns_all_three_claim_destinations(self):
        billing = Document(
            page_content=(
                "Medicare FFS: For patients enrolled in Medicare FFS, submit claims "
                "to the MAC for the state where you provided the services. "
                "DMEPOS suppliers submit claims to the DME MAC for the state where "
                "the patient lives. Medicare Advantage (MA): For patients enrolled "
                "in an MA plan, submit claims to the patient's MA plan."
            ),
            metadata={"source_file": "billing.pdf", "page": 9},
        )

        answer = self.rag._fallback_policy_answer(
            "Where should Medicare FFS, DMEPOS, and Medicare Advantage claims be submitted?",
            [billing],
        )

        self.assertIn("submit claims to the MAC", answer)
        self.assertIn("DME MAC", answer)
        self.assertIn("patient's MA plan", answer)


if __name__ == "__main__":
    unittest.main()

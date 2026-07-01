# Iteration 0 Baseline Audit

| ID | Question | Answer | Sources | Confidence | Retrieval reached? | Problem observed |
|---|---|---|---|---:|---|---|
| B01 | What is Medicare's timely-filing requirement? | Paste response | Paste sources | 0.00 | Yes/No | Notes |
| B02 | What is the CMS-1500 form used for? | | | | | |
| B03 | How are professional claims submitted electronically? | | | | | |
| B04 | Does prior authorization apply to emergency services? | | | | | |
| B05 | What appeal deadline applies after a denial? | | | | | |

b01: Summary from retrieved guidelines
Guideline excerpt (page 4) [S1]
Business Requirements Pub. 100
04 Transmittal: 2140 Date: January 21, 2011 Change Request: 7270 SUBJECT: Changes to the Time Limits for Filing Medicare Fee
System Maintainers OTH ER
Guideline excerpt (page 13) [S2]
Determining End Date of Timely Filing Period—Receipt Date (Rev. 2140, Issued: 01
11, Effective: 01
10, Implementation: 02
Determination of Untimely Filing and Resulting Actions (Rev. 2140, Issued: 01
Guideline excerpt (page 9) [S3]
Definition of a Claim for Payment (Rev. 2140, Issued: 01
Appropriate Medicare Contractor (Rev. 2140, Issued: 01
Guideline excerpt (page 6) [S4]
Implementation Contact(s): David Walczak (410) 786
4475 and Fred Grabau (410) 786
0206, for questions related to the timely filing regulations. Eric Coulson (410) 786
3352, for questions related to supplier claims processing. Tom Dorsey (410) 786

| B01 | What is Medicare's timely-filing requirement? | Returned unrelated raw guideline excerpts rather than a direct requirement | S1: page 4; S2: page 13; S3: page 9; S4: page 6 | [copy confidence] | Yes | Retrieval reached, but citation-generation failure triggered a raw-chunk fallback. The response did not answer the question and included OCR-damaged text. |


**| B01 |** attempt 2: Policy: The time limit for filing all Medicare fee-for-service claims (Part A and Part B claims) is 12 months, or 1 calendar year from the date services were furnished. [S1]

**| B02 |** The CMS-1500 form is used for submitting Medicare professional and supplier claims using the electronic 837P (Professional) format. [S1]

According to the provided policy sources, the CMS-1500 form is also referred to as the "standard paper claim form" that non-institutional providers or suppliers use to bill Medicare Administrative Contractors (MACs). [S2]
medicare_timely_filing_mln.pdf (page 9)
mln006976_medicare_billing_cms_1500_837p.pdf (page 3)

Sources (file + page)
medicare_timely_filing_mln.pdf (page 4)

**| B03 |** Professional claims in Medicare are submitted electronically using the CMS-1500 form (Medicare Billing: CMS-1500 & 837P) or the electronic Health Information Exchange (EHIx) format, which is defined by ANSI ASC X12N 837P Version 5010A1.

According to the sources:

The National Uniform Claim Committee (NUCC) is responsible for designing and maintaining the CMS-1500 form. [S3]
The ANSI ASC X12N 837P Version 5010A1 is the current electronic claim version. [S3]
The Electronic Health Records Interoperability Act of 2014 requires that all Medicare claims be submitted electronically, with a few exceptions for paper claims under specific circumstances. [S2]

However, even prior to this mandate, the overwhelming majority of Medicare claims were submitted in electronic formats. [S2]

Sources (file + page)
medicare_timely_filing_mln.pdf (page 10)
mln006976_medicare_billing_cms_1500_837p.pdf (page 5)


**| B04 |** Based on the provided policy sources, prior authorization does not apply to emergency services.

According to [S1], "Prior Authorization Decision Timeframes: We are requiring impacted payers (excluding QHP issuers on the FFEs) to send prior authorization decisions within 72 hours for expedited (i.e., urgent) requests and seven calendar days for standard (i.e., non-urgent) requests."

Additionally, [S2] states that "Emergency Hospital Services and Services Outside the United States - The time limit for claims for payment for emergency hospital services and hospital services outside the United States, whether or not the hospital has elected to bill the program, is the same as for participating hospitals. (See §70.1 above.)"

Furthermore, [S3] mentions that "Emergency Hospital Services" are excluded from the prior authorization requirements.

Therefore, based on these factual statements, prior authorization does not apply to emergency services.

Sources (file + page)
cms_prior_authorization_fact_sheet.pdf (page 1)
cms_prior_authorization_fact_sheet.pdf (page 3)
medicare_timely_filing_mln.pdf (page 14)


**| B05 |** After a prior authorization denial, the appeal deadline is seven calendar days. [S2] medicare_timely_filing_mln.pdf (page 14)

According to the policy, impacted payers must provide a specific reason for denied prior authorization decisions, regardless of the method used to send the prior authorization request. This requirement does not apply to prior authorization decisions for drugs. [S1] cms_prior_authorization_fact_sheet.pdf (page 3)

The appeal deadline after a prior authorization denial is seven calendar days from the date of the denial. [S2] medicare_timely_filing_mln.pdf (page 14)

Sources (file + page)
cms_prior_authorization_fact_sheet.pdf (page 3)
medicare_timely_filing_mln.pdf (page 14)
## Findings

- Policy PDFs indexed: 3
- API health check: Passed — `{"status":"ok"}`
- Intent-routing limitation: The inherited disease-only router initially blocked policy queries.
- Citation limitation: Citations frequently pointed to related passages that did not support the associated claim.
- Best baseline-gap example: B05 — confused a seven-day decision timeframe with an appeal deadline.

B02: Conflates the paper CMS-1500 form with electronic 837P submission.
B03: Introduces unsupported terms and legislation, including “EHIx” and the alleged 2014 act.
B04: Infers that emergency services are excluded from prior authorization without supporting evidence; the timely-filing source is unrelated.

| ID | Verdict | Main issue |
|---|---|---|
| B01 | Supported | Correctly states the 12-month/one-calendar-year requirement. |
| B02 | Partially supported | Conflates the CMS-1500 paper form with the separate electronic 837P format; includes an irrelevant timely-filing source. |
| B03 | Mixed/unsupported | NUCC and 837P-version claims are supported, but “EHIx” and the alleged 2014 act are hallucinated. The actual document references ASCA and October 16, 2003. |
| B04 | Unsupported | Confuses urgent prior-authorization decisions with emergency services. The cited timely-filing passage says nothing about prior-authorization exclusions. |
| B05 | Unsupported | Seven calendar days is the standard prior-authorization **decision timeframe**, not an appeal deadline. Page 14 concerns timely-filing appeals and provides no seven-day deadline. |


## Baseline Conclusion

The policy-adapted baseline successfully retrieved relevant documents but did
not reliably connect individual claims to supporting evidence.

Of five evaluated answers:

- B01 was supported.
- B02 was partially supported.
- B03 contained both supported and fabricated claims.
- B04 drew an unsupported conclusion from unrelated passages.
- B05 confused a prior-authorization decision timeframe with an appeal deadline.

The presence of citations did not guarantee claim support. Related passages were
sometimes cited for claims they did not entail. These failures motivate
statement-level evidence retrieval and attribution checking in PolicyCite-RAG.

(.venv) aanyalari@Mac medical-rag-ai-assistant % curl -sS -w '\nLatency: %{time_total}s\n' \
  -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is Medicare’s timely-filing requirement?"}'
{"answer":"Policy: The time limit for filing all Medicare fee-for-service claims (Part A and Part B claims) is 12 months, or 1 calendar year from the date services were furnished. [S1]","sources":["medicare_timely_filing_mln.pdf (page 4)"],"confidence":0.95,"normalized_query":"what is medicare’s timely-filing requirement?"}
Latency: 19.806943s

- Answer: Policy: The time limit for filing all Medicare fee-for-service claims (Part A and Part B claims) is 12 months, or 1 calendar year from the date services were furnished. [S1]
- Source: medicare_timely_filing_mln.pdf (page 4)
- Confidence: 0.95
- Latency: 19.81 seconds
- Retrieval reached: Yes
- Human verdict: Supported
The 0.95 confidence is a retrieval-overlap heuristic, not claim-level factual confidence. Other baseline answers also received citations despite containing unsupported claims.

## Iteration 2 Smoke-Test Failure Case

Question: **What is Medicare's timely-filing requirement?**

The live smoke test produced an answer claiming that Medicare claims must be
filed within **60 days of service** and attributed an additional cost and
patient-satisfaction claim to the *Journal of General Internal Medicine*.

Neither claim states the policy contained in the loaded corpus. The policy
documents state that Medicare fee-for-service claims generally must be filed
within **12 months, or one calendar year, after the date of service**. The
60-day language elsewhere in the document concerns adjustment requests and
long-stay billing, not the general timely-filing deadline. The journal-study
claim is not present in the three-document policy corpus.

The response also cited retrieved sources while ending with “Not found in
documents,” illustrating that response-level citations and retrieval confidence
do not establish support for each generated statement.

This example is retained as a baseline failure rather than manually correcting
the generated answer. The statement extractor must preserve all three factual
claims so later statement-specific attribution can mark unsupported claims.

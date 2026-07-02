# PolicyCite-RAG PowerPoint content

This eight-slide content plan is ready to place into an editable PowerPoint.
Use a 16:9 layout, Aptos typography, dark navy background or headings, cool
blue supporting tones, and one amber accent for review/risk states.

## Slide 1 - PolicyCite-RAG

**Subtitle:** Verifiable Generative AI for Healthcare Payment Policy

**Footer:** Aanya Lari | Cotiviti Intern Assessment | Topic 3

**Speaker point:** A statement-level citation-assurance prototype inspired by
MedCite and adapted to public CMS payment-policy documents.

## Slide 2 - A source list is not an audit trail

**Left:** Example answer with three factual statements.

**Right:** Two response-level source pages with no statement mapping.

**Takeaway:** Relevance to a question does not prove support for every date,
exclusion, scope condition, or requirement in the answer.

## Slide 3 - Research foundation and deliberate scope

**MedCite ideas retained**

- RAG-grounded generation
- statement-level evaluation
- post-generation re-retrieval
- citation coverage and precision
- attribution checked independently from generation

**Assessment-scale simplifications**

- three CMS documents, not PubMed
- binary support, not partial support
- no separate evidence-reranker model
- ten frozen questions
- local Ollama models

**Source footer:** Wang et al. (2025), *MedCite*.

## Slide 4 - Architecture and provenance

```text
Question -> FAISS + BM25 -> RRF -> 4 generation passages
                                  |
                                  +-> exact generation trace
                                           |
                                           v
                                      baseline answer
                                           |
                          factual statement extraction
                                           |
                        statement-specific re-retrieval
                                           |
                    strict attribution + validated excerpt
                                           |
                        support verdict + review flag
```

**Callout:** Post-generation evidence can substantiate an answer, but it is not
represented as evidence that grounded generation.

## Slide 5 - Demonstration: supported statement

**Question:** What is Medicare's timely-filing requirement for fee-for-service
claims?

**Statement:** Medicare Part A and Part B fee-for-service claims generally must
be filed within 12 months, or one calendar year, from the date services were
furnished.

**Evidence card:** Medicare timely-filing guidance, display page 4, exact policy
excerpt, `SUPPORTED`.

**Visual:** Current Streamlit PolicyCite evidence card screenshot.

## Slide 6 - Demonstration: safe abstention

**Question:** What is the deadline for appealing a denied prior authorization
request?

**Result:** `Not found in documents.`

**Why it matters:** The corpus contains a seven-calendar-day standard-decision
timeframe, but that is not an appeal deadline. Similar terminology must not be
converted into a confident unsupported answer.

**Visual:** Current Streamlit abstention screenshot.

## Slide 7 - Measured tradeoff, not a victory lap

| Metric | Baseline | PolicyCite |
|---|---:|---:|
| Coverage | 19/21 (90.5%) | 16/21 (76.2%) |
| Precision | 32/43 (74.4%) | 18/18 (100.0%) |
| Citation F1 | 81.7% | 86.5% |
| Complete answers | 7/10 | 7/10 |
| Median latency | 6.55 s | 10.24 s |

**Headline:** PolicyCite removed non-supporting citations but declined to cite
five statements. Median overhead was 3.69 seconds.

**Caveat:** Ten questions, three documents, author signoff pending; later answer
cleanup must be rebenchmarked.

## Slide 8 - Recommendation to Cotiviti

**Recommendation:** Run a bounded, reviewer-in-the-loop policy-assurance pilot.

1. Start with one versioned payment-policy domain.
2. Require evidence for every material statement.
3. Export statement, evidence, verdict, reviewer decision, model, corpus
   version, and timestamp.
4. Measure precision, coverage, reviewer agreement, latency, and time saved.
5. Next experiment: compare RRF with lexical-first, semantic-second citation
   retrieval.

**Closing line:** Treat provenance and evaluation discipline as the product -
not decorative citations.

**Sources:** Wang et al. (2025); CMS timely-filing guidance; CMS-1500/837P
billing guidance; CMS-0057-F fact sheet; Cotiviti assessment directions.

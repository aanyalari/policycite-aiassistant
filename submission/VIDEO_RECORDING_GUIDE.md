# PolicyCite-RAG video recording guide

The Cotiviti assessment requires an MP4 containing the PowerPoint presentation,
a working POC screen share, and the candidate on camera. Record this only after
the project-author evaluation signoff and final slide review.

## Before recording

1. Start Ollama and confirm the three local models are installed.
2. From the repository root, run:

   ```bash
   .venv/bin/uvicorn api:app --host 127.0.0.1 --port 8000
   .venv/bin/streamlit run app.py --server.port 8501
   ```

3. Open `http://127.0.0.1:8501` and test both demo questions.
4. Close unrelated tabs, notifications, terminals, and files containing secrets.
5. Put the PowerPoint in presentation mode and enable the camera overlay.

## Suggested seven-minute run of show

### 0:00-0:30 - Introduction

"I am Aanya Lari. My project is PolicyCite-RAG, a healthcare payment-policy
assistant that tests whether statement-specific evidence can make generated
answers easier to verify. It addresses Cotiviti's Content Management in Health
Care topic."

### 0:30-1:20 - Problem

Explain that a response-level source list can contain related pages without
showing which page supports a particular date, exclusion, or requirement.
Emphasize that the goal is reviewer assistance, not autonomous policy advice.

### 1:20-2:10 - Research foundation

Explain that MedCite combines RAG with statement-level citation retrieval and
evaluation. State clearly that PolicyCite is a small payment-policy adaptation,
not a full reproduction of PubMed-scale MedCite.

### 2:10-3:00 - Architecture

Walk through generation retrieval, the exact generation-context trace,
statement extraction, post-generation re-retrieval, binary attribution, and the
review flag. Point out that later evidence is not represented as original
grounding.

### 3:00-4:10 - Supported-answer demo

Ask:

> What is Medicare's timely-filing requirement for fee-for-service claims?

Show the generated answer, baseline sources, statement-level evidence, document
name, page, excerpt, coverage, and review status.

### 4:10-5:00 - Unsupported-answer demo

Ask:

> What is the deadline for appealing a denied prior authorization request?

Show the `Not found in documents.` abstention. Explain that seven calendar days
is a standard decision timeframe, not an appeal deadline.

### 5:00-6:05 - Evaluation

Report counts as well as percentages. The provenance-fixed run found:

- baseline coverage: 19/21 (90.5%);
- PolicyCite coverage: 16/21 (76.2%);
- baseline precision: 32/43 (74.4%);
- PolicyCite precision: 18/18 (100.0%);
- F1: 81.7% versus 86.5%; and
- median overhead: 3.69 seconds.

Explain the result honestly: the stricter system removed non-supporting
citations but declined to cite five statements. This is a precision-coverage
tradeoff, not proof of production performance.

### 6:05-7:00 - Recommendation and close

Recommend a bounded, reviewer-in-the-loop policy-assurance pilot with versioned
sources, domain-expert review, and agreement measurement. Name the next
experiment: compare RRF with lexical-first, semantic-second citation retrieval.

## Recording quality checklist

- Candidate remains visible on camera during the presentation.
- Voice is clear and the cursor moves deliberately.
- No API keys, `.env` values, email, or personal notifications are visible.
- The demo uses the current PolicyCite UI, not inherited disease-chat screenshots.
- The MP4 opens successfully and audio remains synchronized through the end.
- The final repository contains the report, PPTX, MP4, code, and README.

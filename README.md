# PolicyCite-RAG

PolicyCite-RAG is an assessment-scale healthcare payment-policy assistant that
tests whether statement-specific evidence retrieval can make RAG answers easier
to verify. It extends an inherited medical RAG baseline with a citation-assurance
layer inspired by MedCite.

> Research prototype only. It does not provide medical, legal, coding,
> coverage, or reimbursement advice. Consequential use requires qualified
> human review.

## Why this project exists

A response-level source list does not show which page supports each factual
statement. PolicyCite-RAG preserves the baseline answer and its generation
sources, then separately:

1. extracts factual statements;
2. re-retrieves policy passages for each statement;
3. judges complete support using a separately configured attribution model;
4. attaches validated document, page, and excerpt evidence; and
5. flags unsupported statements for review.

The project addresses Cotiviti's assessment topic **Content Management in
Health Care**, particularly billing and payment policy, policy summarization,
traceability, and governance controls for generated content.

## What is implemented

- Local CMS policy corpus with FAISS dense retrieval, BM25 lexical retrieval,
  and Reciprocal Rank Fusion.
- Unchanged `/ask` endpoint for the inherited response-level baseline.
- `/ask_cited` endpoint for statement-level citations and review flags.
- Exact generation-context trace, kept separate from post-generation evidence.
- Binary `SUPPORTED` / `NOT_SUPPORTED` attribution.
- Exact evidence excerpts copied from retrieved passages, never generated.
- Streamlit evidence interface.
- Ten-question frozen evaluation set with checkpoints and resumable runs.
- Condition-specific review labels, scorer validation, counts, ratios, and
  per-question traces.
- Unit coverage for routing, extraction, retrieval, attribution, API behavior,
  evaluation, and provenance.

Deliberate simplifications include a three-document corpus, binary attribution,
no separate evidence-reranker model, no answer-rewriting loop, and no claim of
production readiness.

## Architecture

```text
User question
    |
    v
FAISS + BM25 -> RRF -> focused generation passages
    |                       |
    |                       +-> exact generation-context trace
    v
Baseline answer + response-level source list             /ask
    |
    v
Factual statement extraction
    |
    +-> statement-specific FAISS + BM25 retrieval
    +-> strict attribution against retrieved passages
    +-> document + display page + exact excerpt
    v
SUPPORTED / NOT_SUPPORTED + review recommendation        /ask_cited
```

Post-generation citations can substantiate a completed statement, but they did
not necessarily ground the original generation. The API exposes these two
provenance stages separately.

## Research foundation

The primary inspiration is Wang et al. (2025), *MedCite: Can Language Models
Generate Verifiable Text for Medicine?* PolicyCite-RAG adapts the paper's
statement-level evaluation and post-generation retrieval ideas to public CMS
payment-policy documents.

This is not a full MedCite reproduction. It does not use PubMed-scale retrieval,
BioASQ/PubMedQA, hierarchical BM25-to-MedCPT ranking, partial-support scoring,
medical-expert annotation, multi-model ablations, or MedCite's complete
double-pass citation-merging procedure.

## Current evaluation

The current reportable artifact is
`evaluation/results/20260701T231854Z-schema-v2-provenance-fixed.json`.
Both conditions use the exact same generated answer; therefore answer
correctness is a shared answer-quality check rather than an uplift claim.

| Metric | Baseline sources | PolicyCite-RAG |
| --- | ---: | ---: |
| Citation coverage | 19/21 (90.5%) | 16/21 (76.2%) |
| Citation precision | 32/43 (74.4%) | 18/18 (100.0%) |
| Citation F1 | 81.7% | 86.5% |
| Complete-answer retention | 7/10 (70.0%) | 7/10 (70.0%) |
| Median latency | 6.55 s | 10.24 s |

Median PolicyCite overhead was 3.69 seconds. On this small run, the stricter
pipeline removed non-supporting attachments and achieved perfect audited
precision, but it declined to cite five of 21 statements fully. The result is a
precision-coverage tradeoff, not a universal performance win.

The labels are a condition-specific evidence audit independent of the automated
verdicts. The project author must complete the signoff recorded in
`evaluation/LABELING_GUIDE.md` before using the results in a final submission.
Answer-cleanup guardrails were added after this measured run; they are covered
by regression tests but must be rebenchmarked when the local Ollama service is
available. The table above does not claim to measure those later guardrails.

Audit history is preserved:

- `20260701T210613Z.json`: original schema-v1 failure run;
- `20260701T223254Z-environment-failure.json`: blocked local-model trial; and
- `20260701T223314Z-invalid-labeling.json`: invalidated labels preserved for
  transparency; and
- `20260701T232956Z-pre-cleanup-unscored.json`: completed execution superseded
  by later answer-cleanup changes before labeling.

## Quick start

### Requirements

- Python 3.11+
- Ollama
- `llama3.2:1b`, `qwen2.5:3b`, and `nomic-embed-text`

```bash
git clone https://github.com/aanyalari/policycite-aiassistant.git
cd policycite-aiassistant
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
ollama pull llama3.2:1b
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

The repository includes the assessment corpus and a prebuilt policy index. To
rebuild them:

```bash
.venv/bin/python ingest.py
```

Start the API and UI in separate terminals:

```bash
.venv/bin/uvicorn api:app --reload
.venv/bin/streamlit run app.py
```

Open `http://127.0.0.1:8501`.

## API examples

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is Medicare timely filing?"}'

curl -X POST http://127.0.0.1:8000/ask_cited \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is Medicare timely filing?"}'
```

### Replay a genuine RAG failure

The Streamlit UI includes a `Replay genuine RAG failure` mode for the preserved
`policy-003` answer in `evaluation/results/20260701T210613Z.json`. The answer was
generated by this project's RAG model during that evaluation run and is not
edited or regenerated for the replay. The current PolicyCite pipeline audits it
live, supporting the correct 72-hour/seven-day statements and flagging the
incorrect conclusion that both request types have seven days.

The fixed replay endpoint is also available directly:

```bash
curl -X POST http://127.0.0.1:8000/demo/replay/policy-003
```

This is a reproducible regression demonstration, not a claim that the current
answer generator will reproduce the same historical error on demand.

### Live original-baseline comparison

The UI's `Live question` mode intentionally uses the frozen pre-correction RAG
behavior: the original retrieval order, broader generation context, earlier
policy prompt, and earlier fallback. It generates one answer and passes the exact
same answer trace to the current PolicyCite pipeline. The left panel therefore
shows the complete original baseline output, while the right panel shows only
PolicyCite's statement-level audit. Current correction code remains available
for evaluation and regression work; it is not silently presented as the
baseline.

```bash
curl -X POST http://127.0.0.1:8000/demo/live \
  -H 'Content-Type: application/json' \
  -d '{"question":"How quickly must impacted payers decide expedited and standard prior authorization requests?"}'
```

## Reproduce the evaluation

```bash
.venv/bin/python evaluation/run_comparative.py run --limit 1
.venv/bin/python evaluation/run_comparative.py run \
  --resume evaluation/results/<timestamp>.json
```

Review every field using `evaluation/LABELING_GUIDE.md`, then score:

```bash
.venv/bin/python evaluation/run_comparative.py score \
  evaluation/results/<timestamp>.json
```

The scorer rejects incomplete labels, execution errors, and any statement
marked fully supported without an attached citation labeled as contributing
support.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## Repository map

```text
citation/                 statement extraction, retrieval, attribution, schemas
data/                     policy corpus
evaluation/               frozen questions, labeling guide, runner, results
policy_pdfs/              public CMS source documents
policy_vector_db/         prebuilt local FAISS index
submission/               report, slides, and video preparation material
tests/                    unit and regression tests
api.py                    FastAPI endpoints
app.py                    Streamlit evidence interface
rag_core.py               inherited RAG plus focused policy behavior
```

## Limitations and next experiments

- A citation establishes support in the loaded corpus, not universal truth or
  policy currency.
- The local 1B answer model can emit awkward, incomplete, or irrelevant text.
- The attribution model can still make false-positive or false-negative calls.
- PDF extraction can lose table and layout semantics.
- Ten questions and three documents do not establish generalization.
- The next research experiment should compare RRF with lexical-first,
  semantic-second citation ranking, motivated by MedCite's retrieval ablation.
- A second independent reviewer should label a subset and report agreement.
- Production work would require effective-date metadata, access control,
  monitoring, and a real reviewer workflow.

## Baseline attribution and license

This project builds on
[mshoaib2006/medical-rag-ai-assistant](https://github.com/mshoaib2006/medical-rag-ai-assistant),
distributed under the MIT License. The inherited baseline includes ingestion,
hybrid retrieval, answer generation, FastAPI, Streamlit, and response-level
sources. PolicyCite's new contribution is the citation-assurance and evaluation
layer described above.

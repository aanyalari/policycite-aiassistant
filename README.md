# PolicyCite-RAG

PolicyCite-RAG is a healthcare payment-policy assistant that adds
statement-level evidence checks to a retrieval-augmented generation baseline.
The project is built as a small, inspectable prototype for Cotiviti's
assessment topic on content management in health care.

> Research prototype only. It does not provide medical, legal, coding,
> coverage, or reimbursement advice. Consequential use requires qualified
> human review.

## What it does

A response-level source list does not show which page supports each factual
statement. PolicyCite-RAG preserves the baseline answer and its generation
sources, then separately:

1. extracts factual statements;
2. re-retrieves policy passages for each statement;
3. judges complete support using a separately configured attribution model;
4. attaches validated document, page, and excerpt evidence; and
5. flags unsupported statements for review.

The project addresses Cotiviti's assessment topic **Content Management in
Health Care**, especially billing and payment policy, traceability, and review
controls for generated content.

## Features

- Local CMS policy corpus with FAISS dense retrieval, BM25 lexical retrieval,
  and Reciprocal Rank Fusion.
- Single live Streamlit demo that shows one baseline answer beside the current
  PolicyCite audit.
- `/ask` for the inherited response-level baseline.
- `/ask_cited` for statement-level citations and review flags.
- Exact generation-context trace, kept separate from post-generation evidence.
- Binary `SUPPORTED` / `NOT_SUPPORTED` attribution.
- Exact evidence excerpts copied from retrieved passages, never generated.
- Ten-question frozen evaluation set with checkpoints and resumable runs.
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

## Baseline attribution and license

This project builds on
[mshoaib2006/medical-rag-ai-assistant](https://github.com/mshoaib2006/medical-rag-ai-assistant),
distributed under the MIT License. The inherited baseline includes ingestion,
hybrid retrieval, answer generation, FastAPI, Streamlit, and response-level
sources. PolicyCite's new contribution is the citation-assurance and evaluation
layer described above.

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
.venv/bin/uvicorn api:app --host 127.0.0.1 --port 8000
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

### Live demo

The Streamlit app opens directly into one live comparison flow. It generates a
baseline answer once, then sends that exact answer trace through the current
PolicyCite audit. The screen stays intentionally simple:

- left panel: baseline answer and source list
- right panel: statement-level evidence and supported / unsupported verdicts
- no replay mode in the main UI

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

The same test suite runs automatically in GitHub Actions on every push and pull
request.

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


"""Reproducible comparative evaluation for baseline and cited RAG answers."""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Optional

from citation.pipeline import answer_with_citations
from citation.statement_extractor import extract_statements


def _dump_model(value: Any) -> Dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return json.loads(value.json())


def load_questions(path: Path) -> List[Dict[str, Any]]:
    questions = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                questions.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
    if len(questions) != 10:
        raise ValueError(f"Expected exactly 10 questions, found {len(questions)}")
    ids = [item.get("id") for item in questions]
    if len(set(ids)) != len(ids) or any(not item for item in ids):
        raise ValueError("Every question must have a unique non-empty id")
    return questions


def _human_labels(question: Dict[str, Any], statements: Iterable[str], citations: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "required_claims": [
            {"claim": claim, "retained": None}
            for claim in question["required_claims"]
        ],
        "statements": [
            {"text": statement, "fully_supported": None}
            for statement in statements
        ],
        "citations": [
            {**citation, "contributes_support": None}
            for citation in citations
        ],
        "notes": "",
    }


def new_artifact(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "configuration": config or {},
        "metric_definitions": {
            "citation_coverage": "fully supported factual statements / all factual statements",
            "citation_precision": "attached citations that contribute support / all attached citations",
            "citation_f1": "2 * precision * coverage / (precision + coverage)",
            "answer_correctness_retention": "answers retaining all required claims / evaluated answers",
            "median_latency_overhead_ms": "median cited latency - median baseline latency",
        },
        "aggregate_metrics": None,
        "results": [],
    }


def _reference(question: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "expected_answerable": question["expected_answerable"],
        "reference_answer": question["reference_answer"],
        "required_claims": question["required_claims"],
        "gold_evidence": question["gold_evidence"],
        "notes": question.get("notes", ""),
    }


def _failed_condition(question: Dict[str, Any], exc: Exception, latency_ms: float) -> Dict[str, Any]:
    return {
        "status": "error",
        "output": None,
        "error": {"type": type(exc).__name__, "message": str(exc), "latency_ms": latency_ms},
        "human_labels": _human_labels(question, [], []),
    }


def run_comparison(
    questions: List[Dict[str, Any]],
    rag: Any,
    *,
    extractor: Callable[[str], List[str]] = extract_statements,
    cited_answerer: Callable[..., Any] = answer_with_citations,
    config: Optional[Dict[str, Any]] = None,
    artifact: Optional[Dict[str, Any]] = None,
    checkpoint: Optional[Callable[[Dict[str, Any]], None]] = None,
    progress: Optional[Callable[[str], None]] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Run both conditions, checkpointing each question for safe resumption."""
    artifact = artifact or new_artifact(config)
    if config and not artifact.get("configuration"):
        artifact["configuration"] = config
    completed_ids = {item["id"] for item in artifact["results"]}
    remaining = [item for item in questions if item["id"] not in completed_ids]
    if limit is not None:
        remaining = remaining[:limit]

    total = len(questions)
    for question in remaining:
        position = next(index for index, item in enumerate(questions, 1) if item["id"] == question["id"])
        prefix = f"[{position}/{total}] {question['id']}"
        if progress:
            progress(f"{prefix}: baseline started")
        started = perf_counter()
        baseline = None
        try:
            baseline = rag.answer_with_trace(question["question"])
            baseline_latency = (perf_counter() - started) * 1000
            baseline_statements = extractor(baseline.answer)
            # Response-level baseline sources are treated as attached to every
            # extracted statement, producing the same statement-citation unit
            # used to audit PolicyCite-RAG.
            baseline_citations = [
                {
                    "statement_index": statement_index,
                    "citation_index": source_index,
                    "source": source,
                }
                for statement_index, _statement in enumerate(baseline_statements)
                for source_index, source in enumerate(baseline.sources)
            ]
            baseline_result = {
                    "status": "complete",
                    "output": {
                        "answer": baseline.answer,
                        "sources": baseline.sources,
                        "confidence": baseline.confidence,
                        "normalized_query": baseline.normalized_query,
                        "latency_ms": baseline_latency,
                    },
                    "human_labels": _human_labels(
                        question, baseline_statements, baseline_citations
                    ),
                }
        except Exception as exc:
            baseline_result = _failed_condition(
                question, exc, (perf_counter() - started) * 1000
            )
        if progress:
            progress(f"{prefix}: baseline {baseline_result['status']}")

        if progress:
            progress(f"{prefix}: PolicyCite-RAG started")
        started = perf_counter()
        try:
            cited = cited_answerer(
                question["question"], rag, answer_trace=baseline
            )
            cited_data = _dump_model(cited)
            if cited_data.get("latency_ms") is not None:
                cited_data["latency_ms"] += baseline_latency
            cited_citations = []
            for statement_index, statement in enumerate(cited_data["statements"]):
                for citation_index, citation in enumerate(statement["citations"]):
                    cited_citations.append(
                        {
                            "statement_index": statement_index,
                            "citation_index": citation_index,
                            "citation_id": citation["citation_id"],
                            "source": citation["source"],
                            "page": citation.get("page"),
                            "evidence_excerpt": citation["evidence_excerpt"],
                        }
                    )
            cited_result = {
                    "status": "complete",
                    "output": cited_data,
                    "human_labels": _human_labels(
                        question,
                        [item["text"] for item in cited_data["statements"]],
                        cited_citations,
                    ),
                }
        except Exception as exc:
            cited_result = _failed_condition(
                question, exc, (perf_counter() - started) * 1000
            )
        if progress:
            progress(f"{prefix}: PolicyCite-RAG {cited_result['status']}")

        artifact["results"].append(
            {
                "id": question["id"],
                "question": question["question"],
                "category": question["category"],
                "reference": _reference(question),
                "baseline": baseline_result,
                "policycite_rag": cited_result,
            }
        )
        artifact["updated_at"] = datetime.now(timezone.utc).isoformat()
        artifact["status"] = (
            "awaiting_human_labels" if len(artifact["results"]) == total else "running"
        )
        if checkpoint:
            checkpoint(artifact)
        if progress:
            progress(f"{prefix}: checkpoint saved")
    return artifact


def _ratio(numerator: int, denominator: int) -> Dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else 0.0,
    }


def _condition_metrics(results: List[Dict[str, Any]], condition: str) -> Dict[str, Any]:
    labels = [item[condition]["human_labels"] for item in results]
    statement_values = [
        label["fully_supported"]
        for group in labels
        for label in group["statements"]
    ]
    citation_values = [
        label["contributes_support"]
        for group in labels
        for label in group["citations"]
    ]
    correctness = [
        all(label["retained"] for label in group["required_claims"])
        for group in labels
    ]
    coverage = _ratio(sum(statement_values), len(statement_values))
    precision = _ratio(sum(citation_values), len(citation_values))
    denominator = coverage["value"] + precision["value"]
    f1 = 2 * coverage["value"] * precision["value"] / denominator if denominator else 0.0
    latency_key = "latency_ms"
    latencies = [item[condition]["output"][latency_key] for item in results]
    return {
        "citation_coverage": coverage,
        "citation_precision": precision,
        "citation_f1": f1,
        "answer_correctness_retention": _ratio(sum(correctness), len(correctness)),
        "median_latency_ms": statistics.median(latencies),
    }


def _missing_labels(artifact: Dict[str, Any]) -> List[str]:
    missing = []
    for result in artifact["results"]:
        for condition in ("baseline", "policycite_rag"):
            labels = result[condition]["human_labels"]
            for group, field in (
                ("required_claims", "retained"),
                ("statements", "fully_supported"),
                ("citations", "contributes_support"),
            ):
                for index, label in enumerate(labels[group]):
                    if label.get(field) is not True and label.get(field) is not False:
                        missing.append(f"{result['id']}.{condition}.{group}[{index}].{field}")
    return missing


def _invalid_supported_statements(artifact: Dict[str, Any]) -> List[str]:
    """Reject coverage labels that are unsupported by attached citations."""
    invalid = []
    for result in artifact["results"]:
        for condition in ("baseline", "policycite_rag"):
            labels = result[condition]["human_labels"]
            contributing_by_statement = {
                citation.get("statement_index")
                for citation in labels["citations"]
                if citation.get("contributes_support") is True
            }
            for index, statement in enumerate(labels["statements"]):
                if (
                    statement.get("fully_supported") is True
                    and index not in contributing_by_statement
                ):
                    invalid.append(
                        f"{result['id']}.{condition}.statements[{index}].fully_supported"
                    )
    return invalid


def score_artifact(artifact: Dict[str, Any]) -> Dict[str, Any]:
    """Compute metrics only after every human label is explicitly boolean."""
    if len(artifact.get("results", [])) != artifact.get("configuration", {}).get("question_count", 10):
        raise ValueError("Evaluation is incomplete; resume the run before scoring")
    errors = [
        f"{result['id']}.{condition}"
        for result in artifact["results"]
        for condition in ("baseline", "policycite_rag")
        if result[condition].get("status") == "error"
    ]
    if errors:
        raise ValueError("Evaluation contains execution errors: " + ", ".join(errors))
    missing = _missing_labels(artifact)
    if missing:
        preview = ", ".join(missing[:5])
        remainder = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        raise ValueError(f"Human labels are incomplete: {preview}{remainder}")
    invalid = _invalid_supported_statements(artifact)
    if invalid:
        preview = ", ".join(invalid[:5])
        remainder = f" (+{len(invalid) - 5} more)" if len(invalid) > 5 else ""
        raise ValueError(
            "A fully supported statement must have at least one attached "
            f"citation labeled as contributing support: {preview}{remainder}"
        )

    baseline = _condition_metrics(artifact["results"], "baseline")
    cited = _condition_metrics(artifact["results"], "policycite_rag")
    artifact["aggregate_metrics"] = {
        "baseline": baseline,
        "policycite_rag": cited,
        "median_latency_overhead_ms": (
            cited["median_latency_ms"] - baseline["median_latency_ms"]
        ),
    }
    artifact["status"] = "complete"
    artifact["scored_at"] = datetime.now(timezone.utc).isoformat()
    return artifact

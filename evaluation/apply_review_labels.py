"""Apply the documented evidence-review labels to the fixed schema-v2 run.

These labels were prepared without copying the automated attribution verdicts.
The project author must still review and sign off before submission.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


LABELS = {
    "policy-001": {"claims": [1, 1], "baseline_statements": [1, 1], "policycite_statements": [1, 1], "baseline_citations": [1, 1, 1, 1], "policycite_citations": [1, 1]},
    "policy-002": {"claims": [1, 1], "baseline_statements": [1, 1], "policycite_statements": [1, 1], "baseline_citations": [1, 1], "policycite_citations": [1, 1]},
    "policy-003": {"claims": [1, 1, 0], "baseline_statements": [1], "policycite_statements": [1], "baseline_citations": [1], "policycite_citations": [1]},
    "policy-004": {"claims": [0, 0, 0], "baseline_statements": [0, 1, 1, 1, 0], "policycite_statements": [0, 1, 0, 1, 0], "baseline_citations": [0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1], "policycite_citations": [1, 1, 1, 1]},
    "policy-005": {"claims": [1, 1], "baseline_statements": [1, 1], "policycite_statements": [1, 1], "baseline_citations": [1, 1, 1, 1], "policycite_citations": [1, 1]},
    "policy-006": {"claims": [1, 1, 1, 1, 1], "baseline_statements": [1, 1, 1, 1], "policycite_statements": [1, 1, 1, 1], "baseline_citations": [1, 1, 1, 1, 1, 1, 1, 1], "policycite_citations": [1, 1, 1, 1]},
    "policy-007": {"claims": [0, 1, 0, 0, 0], "baseline_statements": [1], "policycite_statements": [1], "baseline_citations": [1], "policycite_citations": [1]},
    "policy-008": {"claims": [1, 1, 1], "baseline_statements": [1, 1, 1, 1], "policycite_statements": [1, 1, 0, 0], "baseline_citations": [0, 1, 0, 1, 0, 1, 0, 1], "policycite_citations": [1, 1]},
    "policy-009": {"claims": [1], "baseline_statements": [], "policycite_statements": [], "baseline_citations": [], "policycite_citations": []},
    "policy-010": {"claims": [1], "baseline_statements": [], "policycite_statements": [], "baseline_citations": [], "policycite_citations": []},
}
EXPECTED_CREATED_AT = "2026-07-01T23:19:39.736842+00:00"

NOTES = {
    "policy-003": "The answer retained both timeframes but omitted the QHP-on-FFE exclusion.",
    "policy-004": "Strict review: the answer contradicts itself on drugs, changes request method to decision method, adds unrelated content, and ends with a truncated statement.",
    "policy-007": "The answer retained only the 2026 specific-denial-reason requirement; the other four required claims were omitted.",
    "policy-008": "The answer retained all required destinations, but PolicyCite attached evidence only to the first two of four extracted statements.",
    "policy-009": "Negative control correctly abstained with 'Not found in documents.'",
    "policy-010": "Negative control correctly abstained with 'Not found in documents.'",
}


def assign(items: list[dict], field: str, values: list[int], location: str) -> None:
    if len(items) != len(values):
        raise ValueError(f"{location}: expected {len(values)} labels, found {len(items)} items")
    for item, value in zip(items, values):
        item[field] = bool(value)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: apply_review_labels.py ARTIFACT")
    path = Path(sys.argv[1])
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("schema_version") != 2:
        raise ValueError("Review labels are defined only for schema v2")
    if artifact.get("created_at") != EXPECTED_CREATED_AT:
        raise ValueError(
            "These review decisions apply only to the provenance-fixed "
            f"artifact created at {EXPECTED_CREATED_AT}"
        )
    if {result["id"] for result in artifact["results"]} != set(LABELS):
        raise ValueError("Artifact question IDs do not match the reviewed run")

    for result in artifact["results"]:
        question_id = result["id"]
        decisions = LABELS[question_id]
        for condition in ("baseline", "policycite_rag"):
            labels = result[condition]["human_labels"]
            condition_key = "baseline" if condition == "baseline" else "policycite"
            assign(labels["required_claims"], "retained", decisions["claims"], f"{question_id}.{condition}.claims")
            assign(labels["statements"], "fully_supported", decisions[f"{condition_key}_statements"], f"{question_id}.{condition}.statements")
            assign(labels["citations"], "contributes_support", decisions[f"{condition_key}_citations"], f"{question_id}.{condition}.citations")
            labels["notes"] = NOTES.get(question_id, "Checked against the frozen reference and corpus evidence.")

    artifact["labeling"] = {
        "method": "condition-specific evidence review against frozen references and corpus passages",
        "automated_verdicts_used_as_labels": False,
        "reviewer": "Codex evidence audit",
        "project_author_signoff": "required_before_submission",
    }
    path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

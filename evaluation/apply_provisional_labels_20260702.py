"""Apply Codex's provisional evidence review to the 20260702 fresh run.

The project author must personally verify these decisions before human signoff.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


EXPECTED_CREATED_AT = "2026-07-02T04:10:34.108090+00:00"

LABELS = {
    "policy-001": {"claims": [1, 1], "statements": [1, 1]},
    "policy-002": {"claims": [1, 1], "statements": [1, 1]},
    "policy-003": {"claims": [1, 1, 1], "statements": [1]},
    "policy-004": {"claims": [1, 1, 1], "statements": [1]},
    "policy-005": {"claims": [1, 1], "statements": [1]},
    "policy-006": {"claims": [1, 1, 1, 1, 0], "statements": [1, 1, 1, 1]},
    "policy-007": {"claims": [1, 1, 1, 1, 1], "statements": [1, 1, 1, 1]},
    "policy-008": {"claims": [1, 1, 1], "statements": [1, 1, 1]},
    "policy-009": {"claims": [1], "statements": []},
    "policy-010": {"claims": [1], "statements": []},
}

NOTES = {
    "policy-004": (
        "The answer retained the drug-decision exclusion, but the statement "
        "extractor omitted that sentence from statement-level evaluation."
    ),
    "policy-005": (
        "Provisional judgment: 'electronically unless ... waiver or exception "
        "under ASCA' retains the requested paper-submission exception by implication."
    ),
    "policy-006": (
        "The answer retained four exception categories, but did not state the "
        "retroactive-disrollment qualifier for the MA/PACE claim; that required "
        "claim is false. The standalone retroactive-entitlement sentence was also "
        "omitted by statement extraction."
    ),
    "policy-007": (
        "The answer retained the drug-decision exclusion, but the statement "
        "extractor omitted that sentence from statement-level evaluation."
    ),
    "policy-009": "Negative control correctly abstained with 'Not found in documents.'",
    "policy-010": "Negative control correctly abstained with 'Not found in documents.'",
}


def assign(items: list[dict], field: str, values: list[int], location: str) -> None:
    if len(items) != len(values):
        raise ValueError(
            f"{location}: expected {len(values)} decisions, found {len(items)} items"
        )
    for item, value in zip(items, values):
        item[field] = bool(value)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: apply_provisional_labels_20260702.py ARTIFACT")
    path = Path(sys.argv[1])
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("created_at") != EXPECTED_CREATED_AT:
        raise ValueError("This review applies only to the 20260702T041034Z artifact")
    if {result["id"] for result in artifact.get("results", [])} != set(LABELS):
        raise ValueError("Artifact question IDs do not match the reviewed run")

    for result in artifact["results"]:
        question_id = result["id"]
        decisions = LABELS[question_id]
        for condition in ("baseline", "policycite_rag"):
            labels = result[condition]["human_labels"]
            assign(
                labels["required_claims"],
                "retained",
                decisions["claims"],
                f"{question_id}.{condition}.required_claims",
            )
            assign(
                labels["statements"],
                "fully_supported",
                decisions["statements"],
                f"{question_id}.{condition}.statements",
            )
            # Every attached citation in this run contributes at least partial
            # support to its linked statement. This is judged independently of
            # the automated verdict.
            assign(
                labels["citations"],
                "contributes_support",
                [1] * len(labels["citations"]),
                f"{question_id}.{condition}.citations",
            )
            labels["notes"] = NOTES.get(
                question_id,
                "Provisionally checked against the frozen reference and evidence.",
            )

    artifact["labeling"] = {
        "method": "provisional condition-specific evidence review against frozen references and corpus evidence",
        "automated_verdicts_used_as_labels": False,
        "provisional_reviewer": "Codex",
        "project_author_signoff": "pending",
    }
    path.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

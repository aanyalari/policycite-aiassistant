"""Run a tiny live smoke test against the configured attribution model."""

import argparse
import sys
from pathlib import Path

from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from citation.attribution import judge_attribution


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Judge whether an evidence passage fully supports a statement."
    )
    parser.add_argument("--statement", help="Statement to verify")
    parser.add_argument("--evidence", help="Evidence passage used for verification")
    parser.add_argument("--source", default="example.pdf", help="Displayed source")
    parser.add_argument(
        "--page",
        type=int,
        default=1,
        help="Displayed one-based page number (default: 1)",
    )
    args = parser.parse_args()

    if bool(args.statement) != bool(args.evidence):
        parser.error("--statement and --evidence must be supplied together")

    if args.statement:
        evidence_text = args.evidence
        source = args.source
        internal_page = max(args.page - 1, 0)
        examples = [("Custom statement", args.statement)]
    else:
        evidence_text = (
            "Medicare claims must be filed no later than 12 months after "
            "the date of service. Certain exceptions may apply."
        )
        source = "filing.pdf"
        internal_page = 2
        examples = [
            (
                "Fully supported statement",
                "Medicare claims must be filed no later than 12 months after "
                "the date of service.",
            ),
            (
                "Statement with unsupported qualifier",
                "Medicare claims must be filed within 12 months without exception.",
            ),
        ]

    evidence = Document(
        page_content=evidence_text,
        metadata={
            "chunk_id": "custom-evidence-1",
            "source_file": source,
            "page": internal_page,
            "retrieval_score": 0.9,
        },
    )

    print("Evidence passage:")
    print(evidence.page_content)
    print()

    for label, statement in examples:
        result = judge_attribution(statement, [evidence])
        print(label)
        print(f"Statement: {statement}")
        print(f"Verdict: {result.verdict.value}")
        print(f"Reason: {result.reason or '(model returned no reason)'}")
        if result.citations:
            citation = result.citations[0]
            print(
                f"Citation: {citation.source}, page {citation.page}, "
                f"ID {citation.citation_id}"
            )
            print(f'Excerpt: "{citation.evidence_excerpt}"')
        else:
            print("Citation: none")
        print()


if __name__ == "__main__":
    main()

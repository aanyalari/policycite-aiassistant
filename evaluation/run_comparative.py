"""CLI for running and scoring the Iteration 8 comparison."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.comparative import load_questions, new_artifact, run_comparison, score_artifact


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or score the Iteration 8 evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run both conditions and create a labeling artifact")
    run_parser.add_argument("--questions", type=Path, default=ROOT / "evaluation/questions.jsonl")
    run_parser.add_argument("--output", type=Path)
    run_parser.add_argument("--resume", type=Path, help="continue an existing checkpoint artifact")
    run_parser.add_argument("--limit", type=int, help="process only this many remaining questions")
    score_parser = subparsers.add_parser("score", help="score a manually labeled artifact in place")
    score_parser.add_argument("artifact", type=Path)
    args = parser.parse_args()

    if args.command == "run":
        from rag_core import MedicalRAG

        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be at least 1")
        if args.resume and args.output:
            parser.error("use --resume or --output, not both")
        output = args.resume or args.output or ROOT / "evaluation/results" / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json"
        )
        config = {
            "questions_file": str(args.questions.resolve()),
            "question_count": 10,
            "conditions": {
                "baseline": "one MedicalRAG.answer_with_trace generation with response-level sources",
                "policycite_rag": "statement citations applied to the exact same generated answer",
            },
            "environment": {
                key: os.getenv(key)
                for key in (
                    "VECTOR_DB", "CORPUS_PATH", "OLLAMA_CHAT_MODEL",
                    "ATTRIBUTION_PROVIDER", "ATTRIBUTION_MODEL", "OLLAMA_REQUEST_TIMEOUT",
                    "OLLAMA_NUM_PREDICT",
                )
            },
        }
        config["environment"]["OLLAMA_REQUEST_TIMEOUT"] = os.getenv(
            "OLLAMA_REQUEST_TIMEOUT", "180"
        )
        config["environment"]["OLLAMA_NUM_PREDICT"] = os.getenv(
            "OLLAMA_NUM_PREDICT", "256"
        )
        artifact = None
        if args.resume:
            if not args.resume.is_file():
                parser.error(f"resume artifact does not exist: {args.resume}")
            artifact = json.loads(args.resume.read_text(encoding="utf-8"))
            print(f"Resuming {args.resume} with {len(artifact.get('results', []))}/10 saved", flush=True)
        else:
            artifact = new_artifact(config)
            _write_json(output, artifact)
            print(f"Created checkpoint: {output}", flush=True)

        artifact = run_comparison(
            load_questions(args.questions),
            MedicalRAG(),
            config=config,
            artifact=artifact,
            checkpoint=lambda data: _write_json(output, data),
            progress=lambda message: print(message, flush=True),
            limit=args.limit,
        )
        _write_json(output, artifact)
        print(f"Saved evaluation artifact: {output}", flush=True)
        if artifact["status"] == "awaiting_human_labels":
            print("Complete every human_labels boolean, then run the score command.", flush=True)
        else:
            print("Run again with --resume to continue remaining questions.", flush=True)
    else:
        artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
        score_artifact(artifact)
        _write_json(args.artifact, artifact)
        print(f"Scored evaluation artifact: {args.artifact}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run a resumable execution-accuracy evaluation on a frozen BIRD split."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from semantic_text2sql.agent import TextToSQLAgent
from semantic_text2sql.benchmark import compare_sql
from semantic_text2sql.database import DatabaseRegistry
from semantic_text2sql.llm import (
    AgentRouterClaudeModel,
    GroqSQLModel,
    OllamaSQLModel,
    RoutingSQLModel,
)
from semantic_text2sql.models import DEFAULT_OLLAMA_MODEL, GenerateRequest
from semantic_text2sql.profiling import ProfileStore

PIPELINE_VERSION = "semantic-text2sql-local-v2.2-table-column-profiles"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--database-root", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--provider", choices=("ollama", "agentrouter", "groq"), default="ollama")
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--agentrouter-base-url", default="https://agentrouter.org")
    parser.add_argument("--groq-base-url", default="https://api.groq.com/openai/v1")
    parser.add_argument("--max-attempts", type=int, default=3, choices=range(1, 5))
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--limit", type=int, default=100, choices=range(1, 101))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    indices = split.get("test_ids")
    if not isinstance(dataset, list) or len(dataset) != 500:
        raise ValueError("Expected the 500-case BIRD Mini-Dev JSON array.")
    if not isinstance(indices, list) or not all(isinstance(item, int) for item in indices):
        raise ValueError("Split file must contain an integer test_ids list.")
    if len(indices) != 100 or len(set(indices)) != 100:
        raise ValueError("This evaluator requires exactly 100 unique frozen test IDs.")
    selected_indices = indices[: args.limit]
    selected = [(index, dataset[index]) for index in selected_indices]
    signature = {
        "pipeline_version": PIPELINE_VERSION,
        "dataset": str(args.dataset.resolve()),
        "database_root": str(args.database_root.resolve()),
        "split_file": str(args.split_file.resolve()),
        "profile_root": str(args.profile_root.resolve()),
        "indices": selected_indices,
        "provider": args.provider,
        "model": args.model,
        "max_attempts": args.max_attempts,
        "timeout_seconds": args.timeout_seconds,
    }
    metadata = {
        **signature,
        "created_at": datetime.now(UTC).isoformat(),
        "primary_metric": "BIRD-style set equality of predicted and gold execution results",
        "run_signature": signature,
    }
    results: list[dict[str, Any]] = []
    if args.resume:
        if not args.output.is_file():
            raise ValueError("Resume output does not exist.")
        checkpoint = json.loads(args.output.read_text(encoding="utf-8"))
        if checkpoint.get("metadata", {}).get("run_signature") != signature:
            raise ValueError("Checkpoint configuration does not match this run.")
        results = checkpoint["results"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    registry = DatabaseRegistry(args.database_root)
    model_router = RoutingSQLModel(
        OllamaSQLModel(args.ollama_base_url),
        AgentRouterClaudeModel(
            _agentrouter_key(),
            args.agentrouter_base_url,
        ),
        GroqSQLModel(os.environ.get("GROQ_API_KEY"), args.groq_base_url),
    )
    agent = TextToSQLAgent(registry, model_router, profiles=ProfileStore(args.profile_root))
    completed = {int(item["dataset_index"]) for item in results}
    total = len(selected)
    print(f"Evaluating {total} BIRD cases; resuming after {len(results)}.", flush=True)

    for position, (index, case) in enumerate(selected, start=1):
        if index in completed:
            continue
        started = perf_counter()
        response = await agent.generate(
            GenerateRequest(
                db_id=str(case["db_id"]),
                question=str(case["question"]),
                evidence=str(case.get("evidence") or "") or None,
                provider=args.provider,
                model=args.model,
                max_attempts=args.max_attempts,
                execute=False,
            )
        )
        comparison = (
            compare_sql(
                registry,
                str(case["db_id"]),
                response.sql,
                str(case["SQL"]),
                timeout_seconds=args.timeout_seconds,
            )
            if response.sql
            else None
        )
        result = {
            "dataset_index": index,
            "question_id": case.get("question_id"),
            "db_id": case["db_id"],
            "difficulty": case.get("difficulty", "unknown"),
            "question": case["question"],
            "evidence": case.get("evidence"),
            "gold_sql": case["SQL"],
            "predicted_sql": response.sql,
            "accepted": response.accepted,
            "attempts_used": len(response.attempts),
            "validation_codes": [item.validation.code for item in response.attempts],
            "termination_reason": response.termination_reason,
            "executable": bool(comparison and comparison.executable),
            "equivalent": bool(comparison and comparison.equivalent),
            "comparison": comparison.__dict__ if comparison else None,
            "latency_ms": round((perf_counter() - started) * 1_000),
        }
        results.append(result)
        _write(args.output, metadata, results, complete=False)
        status = "PASS" if result["equivalent"] else "FAIL"
        print(
            f"[{position:03d}/{total}] {status} db={case['db_id']} index={index} "
            f"attempts={result['attempts_used']} latency={result['latency_ms']}ms",
            flush=True,
        )

    report = {
        "complete": True,
        "metadata": metadata,
        "summary": summarize(results),
        "results": results,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)
    return 0


def summarize(results: list[dict[str, Any]]) -> dict[str, object]:
    count = len(results)
    passed = sum(bool(item["equivalent"]) for item in results)
    latencies = sorted(int(item["latency_ms"]) for item in results)
    codes = Counter(code for item in results for code in item["validation_codes"])
    return {
        "completed": count,
        "passed": passed,
        "failed": count - passed,
        "execution_accuracy": round(passed / count, 4) if count else 0.0,
        "accepted_sql_rate": round(sum(bool(item["accepted"]) for item in results) / count, 4)
        if count
        else 0.0,
        "executable_rate": round(sum(bool(item["executable"]) for item in results) / count, 4)
        if count
        else 0.0,
        "average_attempts": round(sum(int(item["attempts_used"]) for item in results) / count, 2)
        if count
        else 0.0,
        "median_latency_ms": round(statistics.median(latencies)) if latencies else 0,
        "validation_codes": dict(codes),
    }


def _agentrouter_key() -> str | None:
    return os.environ.get("AGENTROUTER_API_KEY")


def _write(
    path: Path,
    metadata: dict[str, object],
    results: list[dict[str, Any]],
    *,
    complete: bool,
) -> None:
    path.write_text(
        json.dumps({"complete": complete, "metadata": metadata, "results": results}, indent=2)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

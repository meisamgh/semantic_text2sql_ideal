#!/usr/bin/env python3
"""Run a resumable v3 ensemble evaluation on the frozen BIRD split."""

from __future__ import annotations

import argparse
import asyncio
import json
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
from semantic_text2sql.ensemble import EnsembleTextToSQLAgent
from semantic_text2sql.historical import HistoricalQueryStore
from semantic_text2sql.llm import OllamaSQLModel
from semantic_text2sql.models import DEFAULT_OLLAMA_MODEL, EnsembleRequest
from semantic_text2sql.profiling import ProfileStore

PIPELINE_VERSION = "semantic-text2sql-v3-ensemble-1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--database-root", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--candidate-count", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--max-attempts", type=int, default=2, choices=(1, 2, 3))
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--limit", type=int, default=50, choices=range(1, 101))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    indices = split.get("test_ids")
    if not isinstance(dataset, list) or len(dataset) != 500:
        raise ValueError("Expected the 500-case BIRD Mini-Dev dataset.")
    if not isinstance(indices, list) or len(indices) != 100 or len(set(indices)) != 100:
        raise ValueError("Expected exactly 100 unique protected test IDs.")
    selected_indices = indices[: args.limit]
    history = HistoricalQueryStore(args.history)
    if len(history.records) != 399:
        raise ValueError("Expected exactly 399 leakage-screened historical examples.")
    if {record.get("dataset_index") for record in _raw_history(args.history)} & set(indices):
        raise ValueError("Historical corpus overlaps protected evaluation IDs.")

    signature = {
        "pipeline_version": PIPELINE_VERSION,
        "dataset": str(args.dataset.resolve()),
        "database_root": str(args.database_root.resolve()),
        "split_file": str(args.split_file.resolve()),
        "profile_root": str(args.profile_root.resolve()),
        "history": str(args.history.resolve()),
        "indices": selected_indices,
        "provider": "ollama",
        "model": args.model,
        "candidate_count": args.candidate_count,
        "max_attempts": args.max_attempts,
        "timeout_seconds": args.timeout_seconds,
    }
    metadata = {
        **signature,
        "created_at": datetime.now(UTC).isoformat(),
        "primary_metric": "BIRD-style set equality of selected SQL and gold execution",
        "run_signature": signature,
    }
    results: list[dict[str, Any]] = []
    if args.resume:
        checkpoint = json.loads(args.output.read_text(encoding="utf-8"))
        if checkpoint.get("metadata", {}).get("run_signature") != signature:
            raise ValueError("Checkpoint configuration does not match this run.")
        results = checkpoint["results"]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    registry = DatabaseRegistry(args.database_root)
    base = TextToSQLAgent(
        registry,
        OllamaSQLModel(args.ollama_base_url, timeout=180),
        profiles=ProfileStore(args.profile_root),
    )
    ensemble = EnsembleTextToSQLAgent(base, history)
    completed = {int(item["dataset_index"]) for item in results}
    print(f"Evaluating {len(selected_indices)} cases; resuming after {len(results)}.", flush=True)

    for position, index in enumerate(selected_indices, 1):
        if index in completed:
            continue
        case = dataset[index]
        started = perf_counter()
        response = await ensemble.generate(
            EnsembleRequest(
                db_id=str(case["db_id"]),
                question=str(case["question"]),
                evidence=str(case.get("evidence") or "") or None,
                provider="ollama",
                model=args.model,
                candidate_count=args.candidate_count,
                max_attempts=args.max_attempts,
                historical_top_k=3,
                max_rows=100,
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
            "gold_sql": case["SQL"],
            "selected_sql": response.sql,
            "accepted": response.accepted,
            "selected_candidate": response.selected_candidate,
            "cluster_sizes": [cluster.size for cluster in response.clusters],
            "candidate_sql": [candidate.sql for candidate in response.candidates],
            "candidate_validation_codes": [
                [attempt.validation.code for attempt in candidate.attempts]
                for candidate in response.candidates
            ],
            "inspection_findings": response.inspection_findings,
            "historical_example_scores": [
                example.score for example in response.historical_examples
            ],
            "executable": bool(comparison and comparison.executable),
            "equivalent": bool(comparison and comparison.equivalent),
            "comparison": comparison.__dict__ if comparison else None,
            "latency_ms": round((perf_counter() - started) * 1_000),
        }
        results.append(result)
        _write(args.output, metadata, results, complete=False)
        status = "PASS" if result["equivalent"] else "FAIL"
        print(
            f"[{position:03d}/{len(selected_indices)}] {status} db={case['db_id']} "
            f"index={index} clusters={result['cluster_sizes']} "
            f"latency={result['latency_ms']}ms",
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
    latencies = [int(item["latency_ms"]) for item in results]
    cluster_shapes = Counter(
        tuple(item["cluster_sizes"]) for item in results
    )
    return {
        "completed": count,
        "passed": passed,
        "failed": count - passed,
        "execution_accuracy": round(passed / count, 4) if count else 0.0,
        "accepted_sql_rate": round(
            sum(bool(item["accepted"]) for item in results) / count, 4
        ) if count else 0.0,
        "median_latency_ms": round(statistics.median(latencies)) if latencies else 0,
        "cluster_shapes": {str(key): value for key, value in cluster_shapes.items()},
    }


def _raw_history(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, list) else raw.get("records", [])


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

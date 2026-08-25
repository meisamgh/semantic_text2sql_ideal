#!/usr/bin/env python3
"""Paired Spider execution comparison with Model 1 enabled versus bypassed."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import sqlite3
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx
from dotenv import load_dotenv
from sqlglot import exp, parse_one


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spider-root", type=Path, required=True)
    parser.add_argument("--profile-root", type=Path, default=Path("profiles/spider"))
    parser.add_argument("--provider", default="agentrouter")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=500)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/results/spider_model1_ab.json")
    )
    args = parser.parse_args()
    load_dotenv()
    raise SystemExit(asyncio.run(run(args)))


async def run(args: argparse.Namespace) -> int:
    data_root = args.spider_root.resolve()
    os.environ["TEXT2SQL_DATABASE_ROOT"] = str(data_root / "database")
    os.environ["TEXT2SQL_PROFILE_ROOT"] = str(args.profile_root.resolve())
    os.environ["TEXT2SQL_SCHEMA_RERANKER_ENABLED"] = "true"
    os.environ["TEXT2SQL_SCHEMA_RERANKER_MODEL"] = str(
        Path("models/schema_reranker/v1/model.txt").resolve()
    )
    os.environ["TEXT2SQL_HISTORY_ENABLED"] = "false"
    os.environ["TEXT2SQL_CONTEXT_PLANNER_ENABLED"] = "true"

    from semantic_text2sql.api import create_app

    rows = json.loads((data_root / "dev.json").read_text(encoding="utf-8"))
    eligible = eligible_indices(rows, data_root / "database", args.max_rows)
    random.Random(args.seed).shuffle(eligible)
    chosen = sorted(eligible[: args.count])
    if len(chosen) < args.count:
        raise ValueError(f"Only {len(chosen)} Spider questions were eligible.")

    app = create_app()
    transport = httpx.ASGITransport(app=app)
    results: list[dict[str, Any]] = []
    async with httpx.AsyncClient(transport=transport, base_url="http://benchmark") as client:
        for position, index in enumerate(chosen, 1):
            item = rows[index]
            gold_rows, ordered = execute_gold(item, data_root / "database", args.max_rows)
            paired: dict[str, Any] = {
                "dataset_index": index,
                "db_id": item["db_id"],
                "question": item["question"],
                "gold_sql": item["query"],
            }
            for mode in ("retrieval", "model1"):
                started = perf_counter()
                response = await client.post(
                    "/api/chat",
                    json={
                        "session_id": f"spider-{index}-{mode}",
                        "db_id": item["db_id"],
                        "message": item["question"],
                        "provider": args.provider,
                        "model": args.model,
                        "context_mode": mode,
                        "execute": True,
                        "max_rows": args.max_rows,
                    },
                    timeout=180,
                )
                payload = response.json()
                generation = payload.get("generation") or {}
                predicted_rows = generation.get("rows") or []
                correct = bool(generation.get("accepted")) and equivalent(
                    predicted_rows, gold_rows, ordered=ordered
                )
                telemetry = generation.get("telemetry") or {}
                paired[mode] = {
                    "correct": correct,
                    "accepted": bool(generation.get("accepted")),
                    "sql": generation.get("sql"),
                    "termination_reason": generation.get("termination_reason"),
                    "attempts": len(generation.get("attempts") or []),
                    "tokens": (generation.get("token_usage") or {}).get("total_tokens", 0),
                    "latency_ms": round((perf_counter() - started) * 1_000),
                    "model1_called": bool(telemetry.get("planner_call_used")),
                    "selected_tables": telemetry.get("selected_table_count", 0),
                    "selected_columns": telemetry.get("selected_column_count", 0),
                }
            results.append(paired)
            print(
                f"{position}/{len(chosen)} index={index} "
                f"retrieval={paired['retrieval']['correct']} "
                f"model1={paired['model1']['correct']}",
                flush=True,
            )
            write_report(args, results, chosen)
    write_report(args, results, chosen)
    return 0


def eligible_indices(rows: list[dict[str, Any]], database_root: Path, max_rows: int) -> list[int]:
    eligible: list[int] = []
    for index, item in enumerate(rows):
        try:
            result, _ = execute_gold(item, database_root, max_rows)
            if len(result) <= max_rows:
                eligible.append(index)
        except sqlite3.Error:
            continue
    return eligible


def execute_gold(
    item: dict[str, Any], database_root: Path, max_rows: int
) -> tuple[list[list[Any]], bool]:
    db_id = str(item["db_id"])
    path = database_root / db_id / f"{db_id}.sqlite"
    with sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True) as connection:
        connection.execute("PRAGMA query_only = ON")
        result = [
            list(row)
            for row in connection.execute(str(item["query"])).fetchmany(max_rows + 1)
        ]
    ordered = any(True for _ in parse_one(str(item["query"]), read="sqlite").find_all(exp.Order))
    return result, ordered


def equivalent(predicted: list[list[Any]], gold: list[list[Any]], *, ordered: bool) -> bool:
    if len(predicted) != len(gold):
        return False
    normalized_predicted = [tuple(normalize(value) for value in row) for row in predicted]
    normalized_gold = [tuple(normalize(value) for value in row) for row in gold]
    if ordered:
        return normalized_predicted == normalized_gold
    return Counter(normalized_predicted) == Counter(normalized_gold)


def normalize(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        return round(value, 6)
    return value


def write_report(
    args: argparse.Namespace, results: list[dict[str, Any]], indices: list[int]
) -> None:
    def arm(mode: str) -> dict[str, Any]:
        completed = [item[mode] for item in results]
        count = len(completed)
        token_values = [int(item["tokens"]) for item in completed]
        token_accounting = any(token_values)
        return {
            "correct": sum(bool(item["correct"]) for item in completed),
            "execution_accuracy": (
                round(sum(bool(item["correct"]) for item in completed) / count, 4)
                if count
                else 0.0
            ),
            "accepted": sum(bool(item["accepted"]) for item in completed),
            "model1_call_rate": (
                round(sum(bool(item["model1_called"]) for item in completed) / count, 4)
                if count
                else 0.0
            ),
            "token_accounting_available": token_accounting,
            "average_tokens": (
                round(sum(token_values) / count, 1) if count and token_accounting else None
            ),
            "average_latency_ms": (
                round(sum(int(item["latency_ms"]) for item in completed) / count)
                if count
                else 0
            ),
            "average_selected_tables": (
                round(sum(int(item["selected_tables"]) for item in completed) / count, 2)
                if count
                else 0.0
            ),
            "average_selected_columns": (
                round(sum(int(item["selected_columns"]) for item in completed) / count, 2)
                if count
                else 0.0
            ),
        }

    summary = {
        "benchmark": "Spider 1.0 dev paired A/B",
        "provider": args.provider,
        "model": args.model,
        "seed": args.seed,
        "requested_count": args.count,
        "completed_count": len(results),
        "indices": indices,
        "retrieval": arm("retrieval"),
        "model1": arm("model1"),
        "paired": {
            "both_correct": sum(
                item["retrieval"]["correct"] and item["model1"]["correct"]
                for item in results
            ),
            "retrieval_only": sum(
                item["retrieval"]["correct"] and not item["model1"]["correct"]
                for item in results
            ),
            "model1_only": sum(
                item["model1"]["correct"] and not item["retrieval"]["correct"]
                for item in results
            ),
            "neither": sum(
                not item["retrieval"]["correct"] and not item["model1"]["correct"]
                for item in results
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"summary": summary, "results": results}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

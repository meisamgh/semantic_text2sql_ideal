#!/usr/bin/env python3
"""Paired Model-1 A/B run on 20 hard debit-card questions with labelled SQL."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sqlite3
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx
from dotenv import load_dotenv
from sqlglot import exp, parse_one

QUESTION_IDS = [
    1481,
    1482,
    1526,
    1476,
    1531,
    1533,
    1472,
    1473,
    1479,
    1480,
    1490,
    1501,
    1506,
    1509,
    1521,
    1529,
    1471,
    1498,
    1500,
    1525,
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("../bird-bench/bird_retrieval_lab/data/bird_mini_dev.jsonl"),
    )
    parser.add_argument(
        "--database-root",
        type=Path,
        default=Path("../bird-bench/llm/mini_dev_data/minidev/MINIDEV/dev_databases"),
    )
    parser.add_argument(
        "--profile-root",
        type=Path,
        default=Path("../semantic_text2sql_v5/profiles"),
    )
    parser.add_argument("--provider", default="agentrouter")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--max-rows", type=int, default=500)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/debit_card_model1_ab_20.json"),
    )
    args = parser.parse_args()
    load_dotenv()
    raise SystemExit(asyncio.run(run(args)))


async def run(args: argparse.Namespace) -> int:
    database_root = args.database_root.resolve()
    os.environ["TEXT2SQL_DATABASE_ROOT"] = str(database_root)
    os.environ["TEXT2SQL_PROFILE_ROOT"] = str(args.profile_root.resolve())
    os.environ["TEXT2SQL_SCHEMA_RERANKER_ENABLED"] = "true"
    os.environ["TEXT2SQL_SCHEMA_RERANKER_MODEL"] = str(
        Path("models/schema_reranker/v1/model.txt").resolve()
    )
    os.environ["TEXT2SQL_CONTEXT_PLANNER_ENABLED"] = "true"

    all_rows = [json.loads(line) for line in args.dataset.read_text().splitlines() if line]
    by_id = {int(item["question_id"]): item for item in all_rows}
    rows = [by_id[question_id] for question_id in QUESTION_IDS]
    app = _create_app()
    transport = httpx.ASGITransport(app=app)
    results: list[dict[str, Any]] = []
    async with httpx.AsyncClient(transport=transport, base_url="http://benchmark") as client:
        for position, item in enumerate(rows, 1):
            gold_rows, ordered = execute_sql(
                database_root, str(item["db_id"]), str(item["sql"]), args.max_rows
            )
            paired: dict[str, Any] = {
                "question_id": item["question_id"],
                "difficulty": item["difficulty"],
                "question": item["question"],
                "evidence": item.get("evidence", ""),
                "gold_sql": item["sql"],
            }
            for mode in ("retrieval", "model1"):
                started = perf_counter()
                response = await client.post(
                    "/api/chat",
                    json={
                        "session_id": f"debit20-{item['question_id']}-{mode}",
                        "db_id": "debit_card_specializing",
                        "message": item["question"],
                        "provider": args.provider,
                        "model": args.model,
                        "context_mode": mode,
                        "execute": True,
                        "max_rows": args.max_rows,
                    },
                    timeout=240,
                )
                if response.status_code != 200:
                    raise RuntimeError(
                        f"API error for {item['question_id']} {mode}: "
                        f"HTTP {response.status_code} {response.text}"
                    )
                payload = response.json()
                generation = payload.get("generation") or {}
                predicted = generation.get("rows") or []
                telemetry = generation.get("telemetry") or {}
                paired[mode] = {
                    "correct": bool(generation.get("accepted"))
                    and equivalent(predicted, gold_rows, ordered=ordered),
                    "accepted": bool(generation.get("accepted")),
                    "sql": generation.get("sql"),
                    "termination_reason": generation.get("termination_reason"),
                    "attempts": len(generation.get("attempts") or []),
                    "latency_ms": round((perf_counter() - started) * 1000),
                    "model1_called": bool(telemetry.get("planner_call_used")),
                    "selected_tables": telemetry.get("selected_table_count", 0),
                    "selected_columns": telemetry.get("selected_column_count", 0),
                }
            results.append(paired)
            print(
                f"{position}/20 qid={item['question_id']} "
                f"retrieval={paired['retrieval']['correct']} "
                f"model1={paired['model1']['correct']}",
                flush=True,
            )
            write_report(args, results)
    return 0


def _create_app() -> Any:
    from semantic_text2sql.api import create_app

    return create_app()


def execute_sql(root: Path, db_id: str, sql: str, max_rows: int) -> tuple[list[list[Any]], bool]:
    db_path = root / db_id / f"{db_id}.sqlite"
    with sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True) as connection:
        connection.execute("PRAGMA query_only = ON")
        rows = [list(row) for row in connection.execute(sql).fetchmany(max_rows + 1)]
    ordered = any(True for _ in parse_one(sql, read="sqlite").find_all(exp.Order))
    return rows, ordered


def equivalent(predicted: list[list[Any]], gold: list[list[Any]], *, ordered: bool) -> bool:
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


def write_report(args: argparse.Namespace, results: list[dict[str, Any]]) -> None:
    def summarize(mode: str) -> dict[str, Any]:
        rows = [item[mode] for item in results]
        total = len(rows)
        return {
            "correct": sum(bool(item["correct"]) for item in rows),
            "execution_accuracy": round(sum(bool(item["correct"]) for item in rows) / total, 4),
            "accepted": sum(bool(item["accepted"]) for item in rows),
            "average_latency_ms": round(sum(int(item["latency_ms"]) for item in rows) / total),
            "average_selected_tables": round(
                sum(int(item["selected_tables"]) for item in rows) / total, 2
            ),
            "average_selected_columns": round(
                sum(int(item["selected_columns"]) for item in rows) / total, 2
            ),
        }

    summary = {
        "benchmark": "debit_card_specializing labelled hard-20 paired A/B",
        "provider": args.provider,
        "model": args.model,
        "completed_count": len(results),
        "retrieval": summarize("retrieval"),
        "model1": summarize("model1"),
        "paired": {
            "both_correct": sum(
                item["retrieval"]["correct"] and item["model1"]["correct"] for item in results
            ),
            "retrieval_only": sum(
                item["retrieval"]["correct"] and not item["model1"]["correct"] for item in results
            ),
            "model1_only": sum(
                item["model1"]["correct"] and not item["retrieval"]["correct"] for item in results
            ),
            "neither": sum(
                not item["retrieval"]["correct"] and not item["model1"]["correct"]
                for item in results
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()

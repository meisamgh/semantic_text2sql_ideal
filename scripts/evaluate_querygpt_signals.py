#!/usr/bin/env python3
"""Evaluate QueryGPT component signals after predictions have been frozen."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import TypedDict

from semantic_text2sql.benchmark import compare_sql
from semantic_text2sql.database import DatabaseError, DatabaseRegistry
from semantic_text2sql.querygpt_metrics import qualitative_similarity, table_overlap


class EvaluationRow(TypedDict):
    dataset_index: int
    successful_run: bool
    run_has_output: bool
    execution_equivalent: bool
    table_overlap: float
    qualitative_structural_similarity: float


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--database-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = json.loads(args.predictions.read_text(encoding="utf-8"))
    predictions = artifact.get("predictions", artifact.get("results", []))
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    registry = DatabaseRegistry(args.database_root)
    results: list[EvaluationRow] = []
    for prediction in predictions:
        index = int(prediction["dataset_index"])
        predicted_sql = prediction.get("sql") or prediction.get("selected_sql")
        gold_sql = str(dataset[index]["SQL"])
        comparison = (
            compare_sql(
                registry,
                str(prediction["db_id"]),
                predicted_sql,
                gold_sql,
                timeout_seconds=30,
            )
            if predicted_sql
            else None
        )
        has_output = False
        if comparison is not None and comparison.executable:
            try:
                _, rows, _ = registry.execute(
                    str(prediction["db_id"]), predicted_sql, max_rows=1
                )
                has_output = bool(rows)
            except DatabaseError:
                pass
        results.append(
            {
                "dataset_index": index,
                "successful_run": bool(comparison and comparison.executable),
                "run_has_output": has_output,
                "execution_equivalent": bool(comparison and comparison.equivalent),
                "table_overlap": table_overlap(predicted_sql or "", gold_sql),
                "qualitative_structural_similarity": qualitative_similarity(
                    predicted_sql or "", gold_sql
                ),
            }
        )
    count = len(results)
    report = {
        "complete": bool(artifact.get("complete")),
        "intent_accuracy": "not_measured_no_bird_intent_labels",
        "summary": {
            "completed": count,
            "execution_accuracy": _mean(results, "execution_equivalent"),
            "successful_run_rate": _mean(results, "successful_run"),
            "run_has_output_rate": _mean(results, "run_has_output"),
            "mean_table_overlap": _mean(results, "table_overlap"),
            "mean_qualitative_structural_similarity": _mean(
                results, "qualitative_structural_similarity"
            ),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


def _mean(records: list[EvaluationRow], field: str) -> float:
    values = [float(record[field]) for record in records]  # type: ignore[literal-required]
    return statistics.mean(values) if values else 0.0


if __name__ == "__main__":
    main()

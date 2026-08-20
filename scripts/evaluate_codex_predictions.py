#!/usr/bin/env python3
"""Evaluate frozen predictions; this is the only stage that reads target gold SQL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from semantic_text2sql.benchmark import compare_sql
from semantic_text2sql.database import DatabaseRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--database-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = json.loads(args.predictions.read_text(encoding="utf-8"))
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    registry = DatabaseRegistry(args.database_root)
    results = []
    for prediction in artifact["predictions"]:
        index = int(prediction["dataset_index"])
        comparison = compare_sql(
            registry,
            str(prediction["db_id"]),
            prediction.get("sql"),
            str(dataset[index]["SQL"]),
            timeout_seconds=30,
        )
        results.append(
            {
                "dataset_index": index,
                "db_id": prediction["db_id"],
                "sql": prediction.get("sql"),
                "executable": comparison.executable,
                "equivalent": comparison.equivalent,
                "error": comparison.error,
            }
        )
    passed = sum(item["equivalent"] for item in results)
    report = {
        "complete": bool(artifact.get("complete")),
        "summary": {
            "completed": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "execution_accuracy": passed / len(results) if results else 0,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()

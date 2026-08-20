#!/usr/bin/env python3
"""Create a gold-free queue containing every question for one database."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--db-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    queue = [
        {
            "dataset_index": index,
            "question_id": item.get("question_id"),
            "db_id": item["db_id"],
            "question": item["question"],
            "evidence": item.get("evidence"),
            "difficulty": item.get("difficulty", "unknown"),
        }
        for index, item in enumerate(dataset)
        if item.get("db_id") == args.db_id
    ]
    if not queue:
        raise ValueError(f"No questions found for database {args.db_id}.")
    if any("SQL" in item or "sql" in item for item in queue):
        raise ValueError("Generated queue contains a forbidden SQL field.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(queue, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(queue)} protected {args.db_id} questions without gold SQL.")


if __name__ == "__main__":
    main()

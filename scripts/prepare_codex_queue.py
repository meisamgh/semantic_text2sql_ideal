#!/usr/bin/env python3
"""Create a protected 50-question work queue containing no gold SQL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    indices = split["test_ids"][:50]
    queue = [
        {
            "dataset_index": index,
            "question_id": dataset[index].get("question_id"),
            "db_id": dataset[index]["db_id"],
            "question": dataset[index]["question"],
            "evidence": dataset[index].get("evidence"),
            "difficulty": dataset[index].get("difficulty", "unknown"),
        }
        for index in indices
    ]
    forbidden = {"SQL", "sql", "gold_sql", "answer"}
    if any(forbidden & set(item) for item in queue):
        raise ValueError("Protected queue contains a forbidden answer field.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(queue, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(queue)} protected questions without gold SQL.")


if __name__ == "__main__":
    main()

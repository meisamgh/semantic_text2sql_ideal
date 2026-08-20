#!/usr/bin/env python3
"""Build a leakage-safe ICL corpus from the frozen development IDs only."""

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
    train_ids = split.get("rag_ids")
    test_ids = set(split.get("test_ids", []))
    if not isinstance(train_ids, list) or len(train_ids) != 399:
        raise ValueError("Expected exactly 399 leakage-screened RAG IDs.")
    excluded_duplicates = split.get("excluded_sql_duplicate_ids")
    if not isinstance(excluded_duplicates, list) or len(excluded_duplicates) != 1:
        raise ValueError("Expected one SQL-duplicate record to be excluded from retrieval.")
    if set(train_ids) & test_ids:
        raise ValueError("Training and protected evaluation IDs overlap.")
    records = [
        {
            "dataset_index": index,
            "db_id": dataset[index]["db_id"],
            "question": dataset[index]["question"],
            "sql": dataset[index]["SQL"],
            "success": True,
        }
        for index in train_ids
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(records)} historical examples; 100 protected IDs and one SQL duplicate "
        "excluded."
    )


if __name__ == "__main__":
    main()

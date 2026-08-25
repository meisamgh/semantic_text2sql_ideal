#!/usr/bin/env python3
"""Train and evaluate the optional LightGBM schema reranker on labelled gold SQL."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
from sqlglot import exp, parse_one

from semantic_text2sql.database import DatabaseRegistry
from semantic_text2sql.historical import HistoricalQueryStore
from semantic_text2sql.hybrid_retrieval import (
    RERANKER_FEATURES,
    FastEmbedEncoder,
    HybridSchemaRetriever,
    _reranker_features,
)
from semantic_text2sql.models import SchemaInfo


@dataclass
class QuestionGroup:
    question: str
    identifiers: list[str]
    features: list[list[float]]
    labels: list[int]
    gold_tables: set[str]
    gold_columns: set[str]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--history", type=Path, default=Path("benchmarks/data/bird_history_seed42_400.json")
    )
    parser.add_argument("--database-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/schema_reranker/v1/model.txt"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    args = parser.parse_args()

    records = json.loads(args.history.read_text(encoding="utf-8"))
    if isinstance(records, dict):
        records = records.get("records", [])
    if not isinstance(records, list):
        raise ValueError("History must contain a JSON list of labelled records.")

    rng = random.Random(args.seed)
    valid_records = [
        record for record in records if isinstance(record, dict) and record.get("success") is True
    ]
    rng.shuffle(valid_records)
    split_at = max(1, round(len(valid_records) * (1 - args.holdout_fraction)))
    train_records, holdout_records = valid_records[:split_at], valid_records[split_at:]
    training_history = HistoricalQueryStore.from_records(train_records)
    registry = DatabaseRegistry(args.database_root)
    retriever = HybridSchemaRetriever(FastEmbedEncoder(), max_tables=5, max_columns_per_table=5)
    train_groups, train_skipped = build_groups(
        train_records, registry, retriever, training_history, exclude_self=True
    )
    holdout_groups, holdout_skipped = build_groups(
        holdout_records, registry, retriever, training_history, exclude_self=False
    )
    skipped = train_skipped + holdout_skipped
    if not train_groups or not holdout_groups:
        raise ValueError("Training and holdout sets must both contain questions.")

    train_x, train_y, train_sizes = flatten(train_groups)
    train_data = lgb.Dataset(
        np.asarray(train_x, dtype=np.float32),
        label=np.asarray(train_y, dtype=np.int32),
        group=train_sizes,
        feature_name=list(RERANKER_FEATURES),
    )
    model = lgb.train(
        {
            "objective": "lambdarank",
            "learning_rate": 0.04,
            "num_leaves": 15,
            "min_data_in_leaf": 10,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "seed": args.seed,
            "verbosity": -1,
        },
        train_data,
        num_boost_round=250,
    )

    baseline = evaluate(holdout_groups, None)
    learned = evaluate(holdout_groups, model, preserve_rrf_tables=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(args.output))
    metrics = {
        "training_questions": len(train_groups),
        "holdout_questions": len(holdout_groups),
        "skipped_questions": skipped,
        "seed": args.seed,
        "features": list(RERANKER_FEATURES),
        "baseline_rrf": baseline,
        "ml_reranker": learned,
    }
    metrics_path = args.output.with_name("metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"model={args.output}")
    print(f"metrics={metrics_path}")


def build_group(
    question: str,
    sql: str,
    schema: SchemaInfo,
    retriever: HybridSchemaRetriever,
    historical_evidence: dict[str, dict[str, float]],
) -> QuestionGroup:
    _, _, trace = retriever.retrieve(question, None, schema, None, None, historical_evidence)
    gold_tables, gold_columns = gold_objects(sql, schema)
    identifiers = [
        *[f"table:{table.name}" for table in schema.tables],
        *[
            f"column:{table.name}.{column.name}"
            for table in schema.tables
            for column in table.columns
        ],
    ]
    features = [
        _reranker_features(
            identifier,
            schema,
            trace.bm25_ranks,
            trace.embedding_ranks,
            trace.value_match_ranks,
            trace.rrf_scores,
            historical_evidence,
        )
        for identifier in identifiers
    ]
    labels = [
        2
        if identifier.removeprefix("table:") in gold_tables
        or identifier.removeprefix("column:") in gold_columns
        else 0
        for identifier in identifiers
    ]
    return QuestionGroup(question, identifiers, features, labels, gold_tables, gold_columns)


def build_groups(
    records: list[dict[str, object]],
    registry: DatabaseRegistry,
    retriever: HybridSchemaRetriever,
    history: HistoricalQueryStore,
    *,
    exclude_self: bool,
) -> tuple[list[QuestionGroup], int]:
    groups: list[QuestionGroup] = []
    skipped = 0
    for index, record in enumerate(records, 1):
        try:
            db_id = str(record["db_id"])
            question = str(record["question"])
            schema = registry.inspect(db_id)
            historical_evidence = history.schema_evidence(
                question,
                db_id,
                schema,
                top_k=3,
                min_score=0.65,
                exclude_record_id=(str(record.get("dataset_index", "")) if exclude_self else None),
            )
            groups.append(
                build_group(
                    question,
                    str(record["sql"]),
                    schema,
                    retriever,
                    historical_evidence,
                )
            )
        except (KeyError, ValueError):
            skipped += 1
        if index % 25 == 0 or index == len(records):
            print(f"feature_groups={index}/{len(records)}", flush=True)
    return groups, skipped


def gold_objects(sql: str, schema: SchemaInfo) -> tuple[set[str], set[str]]:
    tree = parse_one(sql, read="sqlite")
    aliases = {
        table.alias_or_name.casefold(): table.name
        for table in tree.find_all(exp.Table)
        if table.name
    }
    gold_tables = {table.name for table in tree.find_all(exp.Table) if table.name}
    schema_columns = {
        table.name: {column.name.casefold(): column.name for column in table.columns}
        for table in schema.tables
    }
    gold_columns: set[str] = set()
    for column in tree.find_all(exp.Column):
        if column.name == "*":
            continue
        if column.table:
            table_name = aliases.get(column.table.casefold(), column.table)
            actual = schema_columns.get(table_name, {}).get(column.name.casefold())
            if actual:
                gold_columns.add(f"{table_name}.{actual}")
            continue
        matches = [
            f"{table}.{columns[column.name.casefold()]}"
            for table, columns in schema_columns.items()
            if table in gold_tables and column.name.casefold() in columns
        ]
        gold_columns.update(matches)
    return gold_tables, gold_columns


def flatten(groups: list[QuestionGroup]) -> tuple[list[list[float]], list[int], list[int]]:
    return (
        [features for group in groups for features in group.features],
        [label for group in groups for label in group.labels],
        [len(group.identifiers) for group in groups],
    )


def evaluate(
    groups: list[QuestionGroup], model: Any | None, *, preserve_rrf_tables: bool = False
) -> dict[str, float]:
    table_hits = 0
    column_recall = 0.0
    for group in groups:
        scores = (
            [row[3] for row in group.features]
            if model is None
            else [float(value) for value in model.predict(group.features)]
        )
        ranked = sorted(
            zip(group.identifiers, scores, strict=True), key=lambda item: (-item[1], item[0])
        )
        table_ranking = (
            sorted(
                zip(group.identifiers, group.features, strict=True),
                key=lambda item: (-item[1][3], item[0]),
            )
            if preserve_rrf_tables
            else ranked
        )
        tables = [
            item.removeprefix("table:") for item, _ in table_ranking if item.startswith("table:")
        ]
        selected_tables = set(tables[:5])
        table_hits += int(group.gold_tables <= selected_tables)

        selected_columns: set[str] = set()
        for table in selected_tables:
            table_columns = [
                item.removeprefix("column:")
                for item, _ in ranked
                if item.startswith(f"column:{table}.")
            ]
            selected_columns.update(table_columns[:5])
        if group.gold_columns:
            column_recall += len(group.gold_columns & selected_columns) / len(group.gold_columns)
        else:
            column_recall += 1.0
    count = len(groups)
    return {
        "table_exact_recall_at_5": round(table_hits / count, 4),
        "mean_gold_column_recall_at_5_per_table": round(column_recall / count, 4),
    }


if __name__ == "__main__":
    main()

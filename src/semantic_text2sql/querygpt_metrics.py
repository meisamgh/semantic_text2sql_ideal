"""QueryGPT-style component signals for post-generation evaluation only."""

from __future__ import annotations

import sqlglot
from sqlglot import expressions as exp


def sql_features(sql: str) -> dict[str, set[str]]:
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except sqlglot.errors.ParseError:
        return {"tables": set(), "columns": set(), "operations": set()}
    operations = {
        type(node).__name__
        for node in tree.walk()
        if isinstance(
            node,
            exp.Join | exp.Group | exp.Having | exp.Order | exp.Limit | exp.Where | exp.AggFunc,
        )
    }
    return {
        "tables": {table.name.casefold() for table in tree.find_all(exp.Table)},
        "columns": {column.name.casefold() for column in tree.find_all(exp.Column)},
        "operations": operations,
    }


def table_overlap(predicted_sql: str, gold_sql: str) -> float:
    predicted = sql_features(predicted_sql)["tables"]
    gold = sql_features(gold_sql)["tables"]
    if not gold:
        return 1.0 if not predicted else 0.0
    return len(predicted & gold) / len(gold)


def qualitative_similarity(predicted_sql: str, gold_sql: str) -> float:
    predicted = sql_features(predicted_sql)
    gold = sql_features(gold_sql)
    scores = []
    for key in ("tables", "columns", "operations"):
        union = predicted[key] | gold[key]
        scores.append(len(predicted[key] & gold[key]) / len(union) if union else 1.0)
    return sum(scores) / len(scores)

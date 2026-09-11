"""Offline review probes. Run from repo root: .venv/bin/python docs/review_v5_reproductions.py.

These assert observed defects, not desired behavior. No external DB or model is used.
"""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

from fastapi.testclient import TestClient

from semantic_text2sql.agent import TextToSQLAgent
from semantic_text2sql.api import create_app
from semantic_text2sql.benchmark import compare_sql
from semantic_text2sql.context import build_context_plan, model_context_payload
from semantic_text2sql.database import DatabaseRegistry
from semantic_text2sql.hybrid_retrieval import HybridSchemaRetriever
from semantic_text2sql.models import ColumnInfo, GenerateRequest, SemanticContract, TokenUsage
from semantic_text2sql.profiling import _profile_column, profile_database
from semantic_text2sql.recovery import RecoveryCoordinator, RecoveryTools
from semantic_text2sql.validator import validate_sql


class BrokenEncoder:
    def encode(self, texts):
        raise RuntimeError("offline")


class FixedModel:
    def __init__(self, sql):
        self.sql = sql

    async def generate(self, **kwargs):
        return self.sql, 1, TokenUsage(input_tokens=10, output_tokens=5)


class FinalWithoutEvidence:
    async def complete_detailed(self, *args):
        return json.dumps({"action": "INFORM", "diagnosis": "The value does not exist.",
                           "confidence": 0.99}), TokenUsage(input_tokens=20, output_tokens=10)


class SlowFinal(FinalWithoutEvidence):
    async def complete_detailed(self, *args):
        await asyncio.sleep(0.15)
        return await super().complete_detailed(*args)


class InvalidFinal(FinalWithoutEvidence):
    async def complete_detailed(self, *args):
        return "invalid json", TokenUsage(input_tokens=20, output_tokens=10)


class ReviewCompleter(FinalWithoutEvidence):
    async def complete_detailed(self, *args):
        if "Classify one message" in args[-1]:
            return json.dumps({"operation": "CHECK_CORRECTNESS", "depends_on_previous": True,
                               "resolved_instruction": "check correctness", "confidence": 1.0}), TokenUsage()
        return await super().complete_detailed(*args)


def main():
    observed = {}
    with tempfile.TemporaryDirectory(prefix="text2sql-review-") as directory:
        root = Path(directory)
        (root / "shop").mkdir()
        dbpath = root / "shop" / "shop.sqlite"
        con = sqlite3.connect(dbpath)
        con.executescript("CREATE TABLE orders(id INTEGER PRIMARY KEY, amount REAL);"
                          "INSERT INTO orders VALUES(1,10),(2,10);"
                          "CREATE TABLE private_data(secret TEXT);"
                          "INSERT INTO private_data VALUES('synthetic-only');"
                          "CREATE TABLE parent(a INTEGER,b INTEGER,PRIMARY KEY(a,b));"
                          "CREATE TABLE child(x INTEGER,y INTEGER,FOREIGN KEY(x,y) REFERENCES parent(a,b));"
                          "INSERT INTO parent VALUES(1,1),(2,2); INSERT INTO child VALUES(1,1);")
        con.close()
        registry = DatabaseRegistry(root)
        schema = registry.inspect("shop")

        result = asyncio.run(TextToSQLAgent(registry, FixedModel("SELECT missing FROM orders")).generate(
            GenerateRequest(db_id="shop", question="Show orders", execute=False)))
        assert result.accepted and result.execution_status == "EXECUTABLE"
        observed["invalid_column_without_execution"] = result.execution_status

        result = asyncio.run(TextToSQLAgent(registry, FixedModel("SELECT secret FROM private_data")).generate(
            GenerateRequest(db_id="shop", question="Show orders", approved_tables=["orders"], execute=True)))
        assert result.accepted and result.rows == [["synthetic-only"]]
        observed["approved_tables_not_enforced"] = result.rows

        comparison = compare_sql(registry, "shop", "SELECT DISTINCT amount FROM orders", "SELECT amount FROM orders")
        assert comparison.equivalent and comparison.predicted_rows != comparison.gold_rows
        observed["set_comparison_loses_duplicates"] = comparison.__dict__

        plan = build_context_plan(schema, None, SemanticContract(), "Show orders", None)
        payload = model_context_payload(plan, None, SemanticContract(), None, "Show orders", "sqlite")
        assert payload["tables"]["orders"]["columns"]["amount"]["type"] == "UNKNOWN"
        observed["live_type_lost_without_profile"] = "REAL -> UNKNOWN"

        _, _, trace = HybridSchemaRetriever(BrokenEncoder()).retrieve("orders", None, schema, None, None)
        assert trace.embedding_ranks
        observed["failed_dense_still_votes"] = trace.embedding_ranks

        prof = _profile_column("orders", ColumnInfo(name="id", data_type="INTEGER", primary_key=True),
                               [123456, 123457], 2, None, 10, exact=True)
        assert prof.semantic_type == "date" and prof.observed_format == "YYYYMM"
        observed["six_digit_id_profile"] = {"type": prof.semantic_type, "format": prof.observed_format}

        profile = profile_database(registry, "shop", "sqlite")
        composite_plan = build_context_plan(schema, profile, SemanticContract(), "Show parents", None)
        composite = [r for r in composite_plan.relationships if r.left_table == "parent"]
        assert len(composite) == 1 and composite[0].left_column == "a"
        observed["composite_relationship_drops_second_key"] = composite[0].model_dump()
        probe_tools = RecoveryTools(registry, "shop", schema, None)
        probe_tools.begin_recovery()
        counts = probe_tools.probe_filter_counts(
            "SELECT p.a FROM parent p JOIN child c ON p.a=c.x AND p.b=c.y WHERE p.a=2",
            allowed_tables=["parent", "child"])
        assert counts[0]["match_count"] == 0
        assert registry.execute("shop", "SELECT COUNT(*) FROM parent WHERE a=2", max_rows=1)[1] == [[1]]
        observed["independent_filter_probe_retains_join"] = counts[0]

        def investigate(completer):
            return RecoveryCoordinator(RecoveryTools(registry, "shop", schema, None,
                                       max_recovery_seconds=0.1)).ainvestigate(
                question="Show orders", failed_sql="SELECT amount FROM orders", failure_code="DATABASE_ERROR",
                failure_message="synthetic failure", allowed_tables=["orders"],
                completer=completer, model="offline")

        trace = asyncio.run(investigate(FinalWithoutEvidence()))
        assert trace.agent_action == "INFORM" and not trace.tool_calls
        observed["ungrounded_final_accepted"] = {"action": trace.agent_action, "tools": len(trace.tool_calls)}
        assert trace.usage.token_usage.total_tokens == 30 and trace.usage.estimated_llm_cost_usd == 0
        observed["positive_usage_zero_cost"] = trace.usage.model_dump(mode="json")

        trace = asyncio.run(investigate(InvalidFinal()))
        assert trace.usage.llm_calls == 0 and trace.usage.token_usage.total_tokens == 0
        observed["failed_recovery_call_usage_lost"] = trace.usage.llm_calls

        started = perf_counter()
        trace = asyncio.run(investigate(SlowFinal()))
        elapsed = perf_counter() - started
        assert elapsed > 0.1 and not trace.usage.budget_exhausted
        observed["recovery_deadline_not_enforced"] = {"limit_ms": 100, "elapsed_ms": round(elapsed * 1000)}

        assert validate_sql("SELECT pg_sleep(1)", schema, dialect="postgres").valid
        observed["function_policy_missing"] = "pg_sleep accepted by AST; not executed"

        env = {"TEXT2SQL_DATABASE_ROOT": str(root), "TEXT2SQL_PROFILE_ROOT": str(root / "profiles"),
               "TEXT2SQL_GLOSSARY_ROOT": str(root / "glossaries"), "TEXT2SQL_SCHEMA_RERANKER_ENABLED": "false",
               "TEXT2SQL_HISTORY_ENABLED": "false", "TEXT2SQL_SQL_MODEL": ""}
        model = FixedModel("SELECT SUM(amount) AS total FROM orders")
        with patch.dict(os.environ, env), patch("semantic_text2sql.hybrid_retrieval.FastEmbedEncoder.encode", BrokenEncoder.encode):
            client = TestClient(create_app(TextToSQLAgent(registry, model), {"ollama": ReviewCompleter()}))
            request = {"session_id": "review", "db_id": "shop", "message": "Total amount", "model": "offline"}
            first = client.post("/api/chat", json=request).json()
            assert first["generation"]["rows"] == [[20.0]]
            model.sql = "SELECT MAX(amount) AS total FROM orders"
            second = client.post("/api/chat", json={**request, "message": "check correctness"}).json()
            assert second["human_review"] and second["state"]["last_sql"] == model.sql
            observed["disputed_candidate_replaces_state"] = second["state"]["last_sql"]
            model.sql = "SELECT amount FROM orders WHERE id = 999"
            third = client.post("/api/chat", json={**request, "session_id": "zero", "message": "Show order 999"}).json()
            total = third["token_usage"]["input_tokens"] + third["token_usage"]["output_tokens"]
            assert total == 15
            observed["anomaly_recovery_usage_not_in_chat_total"] = total
            observed["serialized_usage_has_no_total_tokens"] = "total_tokens" not in third["token_usage"]
            assert third["generation"]["recovery"] is None
            observed["anomaly_trace_not_returned"] = True

        class TrackingRegistry(DatabaseRegistry):
            def connect(self, db_id):
                connection = super().connect(db_id)
                self.last_connection = connection
                return connection

        tracking = TrackingRegistry(root)
        tracking.execute("shop", "SELECT amount FROM orders", max_rows=5)
        assert tracking.last_connection.execute("SELECT 1").fetchone() == (1,)
        observed["sqlite_connection_not_closed_by_context"] = True
        tracking.last_connection.close()
    print(json.dumps(observed, indent=2))


if __name__ == "__main__":
    main()

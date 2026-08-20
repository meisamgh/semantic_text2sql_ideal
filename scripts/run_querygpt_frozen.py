#!/usr/bin/env python3
"""Generate resumable QueryGPT-v4 predictions from a gold-free frozen queue."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from semantic_text2sql.agent import TextToSQLAgent
from semantic_text2sql.database import DatabaseRegistry
from semantic_text2sql.glossary import GlossaryStore
from semantic_text2sql.historical import HistoricalQueryStore, format_examples
from semantic_text2sql.interpreter import interpret_question, reconcile_contract
from semantic_text2sql.linker import compact_interpreter_profile_context
from semantic_text2sql.llm import AgentRouterClaudeModel, ModelError
from semantic_text2sql.models import (
    GenerateRequest,
    IntentRequest,
    PreliminarySemanticIR,
    SchemaInfo,
    TableProposalRequest,
)
from semantic_text2sql.profiling import ProfileStore
from semantic_text2sql.querygpt import QueryGPTFlow, WorkspaceRegistry
from semantic_text2sql.semantic import plan_semantics

PIPELINE_VERSION = "querygpt-inspired-v4.4-business-glossary"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--database-root", type=Path, required=True)
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--workspaces", type=Path)
    parser.add_argument("--glossary-root", type=Path)
    parser.add_argument("--historical-top-k", type=int, default=3, choices=range(0, 6))
    parser.add_argument("--model", default="claude-opus-5")
    parser.add_argument("--limit", type=int, default=50, choices=range(1, 51))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    queue = json.loads(args.queue.read_text(encoding="utf-8"))[: args.limit]
    if any("SQL" in item or "sql" in item for item in queue):
        raise ValueError("Generation queue contains a forbidden SQL field.")
    history = HistoricalQueryStore(args.history)
    if len(history.records) != 399:
        raise ValueError("Expected 399 leakage-screened historical examples.")
    signature = {
        "pipeline_version": PIPELINE_VERSION,
        "queue": str(args.queue.resolve()),
        "database_root": str(args.database_root.resolve()),
        "profile_root": str(args.profile_root.resolve()),
        "history": str(args.history.resolve()),
        "workspaces": str(args.workspaces.resolve()) if args.workspaces else None,
        "glossary_root": str(args.glossary_root.resolve()) if args.glossary_root else None,
        "historical_top_k": args.historical_top_k,
        "model": args.model,
        "indices": [int(item["dataset_index"]) for item in queue],
        "max_tables": 5,
        "max_columns_per_table": 8,
        "max_attempts": 3,
    }
    metadata = {
        **signature,
        "created_at": datetime.now(UTC).isoformat(),
        "gold_access": False,
        "table_ack_policy": "automatic acceptance of the table-agent proposal for benchmark",
        "run_signature": signature,
    }
    predictions: list[dict[str, Any]] = []
    if args.resume:
        checkpoint = json.loads(args.output.read_text(encoding="utf-8"))
        if checkpoint.get("metadata", {}).get("run_signature") != signature:
            raise ValueError("Checkpoint configuration does not match this run.")
        predictions = checkpoint["predictions"]
    completed = {int(item["dataset_index"]) for item in predictions}

    sqlite = DatabaseRegistry(args.database_root)
    profiles = ProfileStore(args.profile_root)
    workspaces = WorkspaceRegistry(args.workspaces) if args.workspaces else WorkspaceRegistry()
    glossaries = GlossaryStore(args.glossary_root)
    flow = QueryGPTFlow(sqlite, None, profiles, workspaces)
    model = AgentRouterClaudeModel(
        os.environ.get("AGENTROUTER_API_KEY"),
        os.environ.get("AGENTROUTER_BASE_URL", "https://agentrouter.org"),
        timeout=180,
    )
    agent = TextToSQLAgent(sqlite, model, profiles=profiles)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Generating {len(queue)} QueryGPT-v4 predictions after {len(predictions)}.", flush=True)

    for position, item in enumerate(queue, 1):
        index = int(item["dataset_index"])
        if index in completed:
            continue
        started = perf_counter()
        intent = flow.classify_intent(
            IntentRequest(
                db_id=str(item["db_id"]),
                question=str(item["question"]),
                evidence=str(item.get("evidence") or "") or None,
            )
        )
        schema = sqlite.inspect(str(item["db_id"]))
        business_context = glossaries.retrieve(
            str(item["db_id"]), str(item["question"]), top_k=8
        )
        business_context += (
            "\n\nQuestion-relevant column metadata:\n"
            + compact_interpreter_profile_context(
                str(item["question"]),
                str(item.get("evidence") or "") or None,
                profiles.load("sqlite", str(item["db_id"])),
            )
        )
        proposal_request = TableProposalRequest(
            db_id=str(item["db_id"]),
            question=str(item["question"]),
            evidence=str(item.get("evidence") or "") or None,
            workspace_id=intent.selected_workspace,
            max_tables=5,
        )
        interpreted, proposal = await asyncio.gather(
            _interpret_or_fallback(
                model,
                args.model,
                str(item["question"]),
                str(item.get("evidence") or "") or None,
                schema,
                business_context,
            ),
            asyncio.to_thread(flow.propose_tables, proposal_request),
        )
        contract = reconcile_contract(
            plan_semantics(str(item["question"]), str(item.get("evidence") or "") or None),
            interpreted,
            schema,
        )
        live_tables = {table.name for table in schema.tables}
        approved_tables = list(
            dict.fromkeys(
                [
                    *proposal.proposed_tables,
                    *(table for table in contract.proposed_tables if table in live_tables),
                ]
            )
        )
        examples = history.search(
            str(item["question"]),
            str(item["db_id"]),
            top_k=args.historical_top_k,
        )
        enriched_evidence = "\n\n".join(
            value
            for value in (
                str(item.get("evidence") or "").strip(),
                format_examples(examples),
            )
            if value
        )
        response = await agent.generate(
            GenerateRequest(
                db_id=str(item["db_id"]),
                question=intent.enhanced_question,
                evidence=enriched_evidence or None,
                dialect="sqlite",
                provider="agentrouter",
                model=args.model,
                max_attempts=3,
                execute=False,
                approved_tables=approved_tables,
                semantic_contract=contract,
                business_context=business_context,
            )
        )
        prediction = {
            "dataset_index": index,
            "question_id": item.get("question_id"),
            "db_id": item["db_id"],
            "question": item["question"],
            "evidence": item.get("evidence"),
            "workspace_id": intent.selected_workspace,
            "workspace_matches": [match.model_dump() for match in intent.matches],
            "enhanced_question": intent.enhanced_question,
            "proposed_tables": proposal.proposed_tables,
            "proposed_columns": proposal.proposed_columns,
            "preliminary_semantic_ir": interpreted.model_dump(),
            "business_glossary_context": business_context,
            "approved_tables": approved_tables,
            "semantic_contract": (
                response.semantic_contract.model_dump()
                if response.semantic_contract is not None
                else None
            ),
            "historical_example_scores": [example.score for example in examples],
            "sql": response.sql,
            "accepted": response.accepted,
            "attempts": len(response.attempts),
            "validation_codes": [attempt.validation.code for attempt in response.attempts],
            "termination_reason": response.termination_reason,
            "latency_ms": round((perf_counter() - started) * 1_000),
        }
        predictions.append(prediction)
        _write(args.output, metadata, predictions, complete=False)
        status = "ACCEPTED" if response.accepted else "REJECTED"
        print(
            f"[{position:03d}/{len(queue)}] {status} db={item['db_id']} index={index} "
            f"tables={approved_tables} latency={prediction['latency_ms']}ms",
            flush=True,
        )
    _write(args.output, metadata, predictions, complete=True)


async def _interpret_or_fallback(
    model: AgentRouterClaudeModel,
    model_name: str,
    question: str,
    evidence: str | None,
    schema: SchemaInfo,
    business_context: str,
) -> PreliminarySemanticIR:
    try:
        return await interpret_question(
            model, model_name, question, evidence, schema, business_context
        )
    except (ModelError, ValueError) as exc:
        return PreliminarySemanticIR(
            ambiguities=[f"Interpreter unavailable; deterministic fallback: {type(exc).__name__}"],
            confidence=0,
        )


def _write(
    path: Path,
    metadata: dict[str, Any],
    predictions: list[dict[str, Any]],
    *,
    complete: bool,
) -> None:
    path.write_text(
        json.dumps(
            {"complete": complete, "metadata": metadata, "predictions": predictions},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    asyncio.run(main())

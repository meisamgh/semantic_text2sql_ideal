#!/usr/bin/env python3
"""Run and checkpoint the bounded Claude agent over a gold-free work queue."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Any

from semantic_text2sql.codex_agent import CodexClaudeAgent
from semantic_text2sql.database import DatabaseRegistry
from semantic_text2sql.historical import HistoricalQueryStore
from semantic_text2sql.llm import AgentRouterClaudeModel
from semantic_text2sql.profiling import ProfileStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--database-root", type=Path, required=True)
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--model", default="claude-opus-5")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    queue = json.loads(args.queue.read_text(encoding="utf-8"))[: args.limit]
    if any("SQL" in item or "sql" in item for item in queue):
        raise ValueError("Gold SQL is forbidden in the generation queue.")
    predictions: list[dict[str, Any]] = []
    if args.resume and args.predictions.is_file():
        predictions = json.loads(args.predictions.read_text(encoding="utf-8"))["predictions"]
    completed = {int(item["dataset_index"]) for item in predictions}
    run_id = str(uuid.uuid4())
    agent = CodexClaudeAgent(
        DatabaseRegistry(args.database_root),
        ProfileStore(args.profile_root),
        HistoricalQueryStore(args.history),
        AgentRouterClaudeModel(
            os.environ.get("AGENTROUTER_API_KEY"),
            os.environ.get("AGENTROUTER_BASE_URL", "https://agentrouter.org"),
            timeout=180,
        ),
    )
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    for position, item in enumerate(queue, 1):
        if int(item["dataset_index"]) in completed:
            continue
        prediction, trace = await agent.solve(
            run_id=run_id,
            dataset_index=int(item["dataset_index"]),
            db_id=str(item["db_id"]),
            question=str(item["question"]),
            evidence=str(item.get("evidence") or "") or None,
            model=args.model,
        )
        predictions.append(prediction)
        with args.trace.open("a", encoding="utf-8") as handle:
            for event in trace:
                handle.write(json.dumps(event) + "\n")
        args.predictions.write_text(
            json.dumps(
                {"complete": False, "model": args.model, "predictions": predictions},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        status = "ACCEPTED" if prediction["accepted"] else "REJECTED"
        print(f"[{position:03d}/{len(queue)}] {status} index={item['dataset_index']}", flush=True)
    args.predictions.write_text(
        json.dumps(
            {"complete": True, "model": args.model, "predictions": predictions}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""Run a gold-free question queue through the real conversational API."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from fastapi.testclient import TestClient

from semantic_text2sql.api import create_app

PIPELINE_VERSION = "querygpt-inspired-v5-conversation"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--model", default="claude-opus-5")
    parser.add_argument("--session-id", default="bird-debit-card-all30")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    queue = json.loads(args.queue.read_text(encoding="utf-8"))
    if any("SQL" in item or "sql" in item for item in queue):
        raise ValueError("Conversation queue contains a forbidden SQL field.")
    signature = {
        "pipeline_version": PIPELINE_VERSION,
        "queue": str(args.queue.resolve()),
        "model": args.model,
        "session_id": args.session_id,
        "indices": [int(item["dataset_index"]) for item in queue],
    }
    metadata = {
        **signature,
        "created_at": datetime.now(UTC).isoformat(),
        "gold_access": False,
        "historical_retrieval": False,
        "run_signature": signature,
    }
    predictions: list[dict[str, Any]] = []
    if args.resume:
        checkpoint = json.loads(args.output.read_text(encoding="utf-8"))
        if checkpoint.get("metadata", {}).get("run_signature") != signature:
            raise ValueError("Checkpoint configuration does not match this chat run.")
        predictions = checkpoint["predictions"]
    completed = {int(item["dataset_index"]) for item in predictions}
    client = TestClient(create_app())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Running {len(queue)} chat turns after {len(predictions)}.", flush=True)
    for position, item in enumerate(queue, 1):
        index = int(item["dataset_index"])
        if index in completed:
            continue
        started = perf_counter()
        response = client.post(
            "/api/chat",
            json={
                "session_id": args.session_id,
                "db_id": item["db_id"],
                "message": item["question"],
                "evidence": item.get("evidence"),
                "provider": "agentrouter",
                "model": args.model,
                "execute": False,
            },
        )
        response.raise_for_status()
        body = response.json()
        generation = body.get("generation") or {}
        prediction = {
            "dataset_index": index,
            "question_id": item.get("question_id"),
            "db_id": item["db_id"],
            "question": item["question"],
            "evidence": item.get("evidence"),
            "operation": body["operation"],
            "resolved_question": body["resolved_question"],
            "turn_count": (body.get("state") or {}).get("turn_count"),
            "sql": generation.get("sql"),
            "accepted": bool(generation.get("accepted")),
            "attempts": len(generation.get("attempts") or []),
            "validation_codes": [
                attempt["validation"]["code"] for attempt in generation.get("attempts") or []
            ],
            "semantic_contract": generation.get("semantic_contract"),
            "termination_reason": generation.get("termination_reason"),
            "latency_ms": round((perf_counter() - started) * 1_000),
        }
        predictions.append(prediction)
        _write(args.output, metadata, predictions, complete=False)
        print(
            f"[{position:03d}/{len(queue)}] operation={body['operation']} "
            f"accepted={prediction['accepted']} index={index} latency={prediction['latency_ms']}ms",
            flush=True,
        )
    _write(args.output, metadata, predictions, complete=True)


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
    main()

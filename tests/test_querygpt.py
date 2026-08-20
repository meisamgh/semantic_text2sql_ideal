from __future__ import annotations

import json

from semantic_text2sql.models import IntentRequest, TableProposalRequest
from semantic_text2sql.profiling import ProfileStore
from semantic_text2sql.querygpt import QueryGPTFlow, WorkspaceRegistry


def test_intent_selects_curated_workspace(registry, tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "workspaces.json"
    path.write_text(
        json.dumps(
            [
                {
                    "workspace_id": "commerce",
                    "name": "Commerce",
                    "description": "Customers, orders, sales, status, and countries.",
                    "db_ids": ["shop"],
                    "tables": ["customers", "orders"],
                    "terms": ["customer", "order", "amount"],
                }
            ]
        )
    )
    flow = QueryGPTFlow(
        registry,
        None,
        ProfileStore(tmp_path / "profiles"),
        WorkspaceRegistry(path),
    )

    result = flow.classify_intent(
        IntentRequest(db_id="shop", question="Total order amount by customer country")
    )

    assert result.selected_workspace == "commerce"
    assert result.matches[0].score > 0
    assert "Business workspace: Commerce" in result.enhanced_question


def test_table_agent_returns_acknowledgeable_pruned_schema(registry, tmp_path) -> None:  # type: ignore[no-untyped-def]
    flow = QueryGPTFlow(
        registry,
        None,
        ProfileStore(tmp_path / "profiles"),
        WorkspaceRegistry(),
    )

    proposal = flow.propose_tables(
        TableProposalRequest(
            db_id="shop",
            question="Total order amount by customer country",
            max_tables=2,
        )
    )

    assert proposal.requires_ack is True
    assert proposal.proposed_tables == ["customers", "orders"]
    assert set(proposal.proposed_columns["customers"]) >= {"customer_id", "country"}
    assert set(proposal.proposed_columns["orders"]) >= {"customer_id", "amount"}
    assert len(proposal.relationships) == 1

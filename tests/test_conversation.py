from __future__ import annotations

import json
from typing import Literal

from fastapi.testclient import TestClient

from semantic_text2sql.agent import TextToSQLAgent
from semantic_text2sql.api import create_app
from semantic_text2sql.conversation import classify_operation, requires_model_interpretation
from semantic_text2sql.models import (
    DEFAULT_OLLAMA_MODEL,
    SchemaInfo,
    StrategyHints,
    TokenUsage,
)


class ConversationModel:
    def __init__(self) -> None:
        self.responses = [
            "SELECT customer_id, COUNT(*) FROM orders GROUP BY customer_id",
            "SELECT o.customer_id, COUNT(*) FROM orders o JOIN customers c "
            "ON o.customer_id = c.customer_id WHERE c.country = 'Germany' "
            "GROUP BY o.customer_id",
            "SELECT customer_id, COUNT(*) FROM orders GROUP BY customer_id",
            "SELECT o.customer_id, COUNT(*) FROM orders o JOIN customers c "
            "ON o.customer_id = c.customer_id WHERE c.country = 'Germany' "
            "GROUP BY o.customer_id",
            "SELECT o.customer_id, COUNT(*) + 1 FROM orders o JOIN customers c "
            "ON o.customer_id = c.customer_id WHERE c.country = 'Germany' "
            "GROUP BY o.customer_id",
            "SELECT o.customer_id, COUNT(*) FROM orders o JOIN customers c "
            "ON o.customer_id = c.customer_id WHERE c.country = 'Germany' "
            "GROUP BY o.customer_id",
        ]
        self.previous_sql: list[str | None] = []

    async def generate(
        self,
        *,
        provider: Literal["ollama", "agentrouter"],
        model: str,
        question: str,
        evidence: str | None,
        schema: SchemaInfo,
        strategy: StrategyHints,
        dialect: Literal["sqlite", "postgres"],
        profile_context: str,
        previous_sql: str | None,
        feedback: str | None,
        rejected_shapes: list[str],
        generation_style: Literal["reasoning", "icl", "alternative"],
    ) -> tuple[str, int]:
        self.previous_sql.append(previous_sql)
        return self.responses.pop(0), 1


class TurnModel:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def complete_detailed(self, model: str, prompt: str):  # type: ignore[no-untyped-def]
        self.calls.append((model, prompt))
        return (
            json.dumps(
                {
                    "operation": "CORRECTION",
                    "depends_on_previous": True,
                    "resolved_instruction": (
                        "Calculate the average per customer before selecting the minimum."
                    ),
                    "correction_type": "wrong_aggregation",
                    "confidence": 0.96,
                }
            ),
            TokenUsage(),
        )


class UncertainTurnModel:
    async def complete_detailed(self, model: str, prompt: str):  # type: ignore[no-untyped-def]
        return (
            json.dumps(
                {
                    "operation": "REFINE",
                    "depends_on_previous": True,
                    "resolved_instruction": "Use that solution.",
                    "correction_type": None,
                    "confidence": 0.4,
                }
            ),
            TokenUsage(input_tokens=10, output_tokens=5),
        )


def test_classifier_detects_correction_and_explanation() -> None:
    assert classify_operation("That's wrong. Only paying customers", True) == "CORRECTION"
    assert classify_operation("Use only paying customers", True, "missing_filter") == "CORRECTION"
    assert classify_operation("Why did you use the orders table?", True) == "EXPLAIN"
    assert classify_operation("Optimize it", True) == "OPTIMIZE"
    assert requires_model_interpretation(
        "You need to calculate average per customer before selecting the minimum", True
    )
    assert not requires_model_interpretation("Optimize it", True)


def test_ambiguous_turn_uses_client_selected_model(registry, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("TEXT2SQL_DATABASE_ROOT", str(registry.root))
    monkeypatch.setenv("TEXT2SQL_GLOSSARY_ROOT", str(registry.root / "glossaries"))
    sql_model = ConversationModel()
    turn_model = TurnModel()
    client = TestClient(
        create_app(
            TextToSQLAgent(registry, sql_model),
            conversation_completers={"ollama": turn_model, "agentrouter": turn_model},
        )
    )
    base = {
        "session_id": "model-resolver",
        "db_id": "shop",
        "provider": "ollama",
        "model": DEFAULT_OLLAMA_MODEL,
        "execute": True,
    }

    first = client.post(
        "/api/chat", json={**base, "message": "How many orders for each customer?"}
    ).json()
    corrected = client.post(
        "/api/chat",
        json={
            **base,
            "message": "You need to calculate average per customer before selecting the minimum",
        },
    ).json()

    assert first["conversation_interpretation"]["source"] == "rules"
    assert corrected["operation"] == "CORRECTION"
    assert corrected["conversation_interpretation"] == {
        "operation": "CORRECTION",
        "depends_on_previous": True,
        "resolved_instruction": (
            "Calculate the average per customer before selecting the minimum."
        ),
        "correction_type": "wrong_aggregation",
        "confidence": 0.96,
        "source": "model",
        "provider": "ollama",
        "model": DEFAULT_OLLAMA_MODEL,
    }
    assert turn_model.calls[0][0] == DEFAULT_OLLAMA_MODEL
    assert "Previous accepted SQL" in turn_model.calls[0][1]
    assert "Correction category=wrong_aggregation" in corrected["resolved_question"]
    assert "Calculate the average per customer" in corrected["resolved_question"]


def test_low_confidence_turn_asks_for_clarification(registry, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("TEXT2SQL_DATABASE_ROOT", str(registry.root))
    sql_model = ConversationModel()
    uncertain = UncertainTurnModel()
    client = TestClient(
        create_app(
            TextToSQLAgent(registry, sql_model),
            conversation_completers={"ollama": uncertain, "agentrouter": uncertain},
        )
    )
    base = {
        "session_id": "uncertain-resolver",
        "db_id": "shop",
        "provider": "ollama",
        "model": DEFAULT_OLLAMA_MODEL,
        "execute": True,
    }
    client.post("/api/chat", json={**base, "message": "How many orders per customer?"})
    response = client.post("/api/chat", json={**base, "message": "Use that solution"}).json()

    assert response["clarification_required"] is True
    assert response["generation"] is None
    assert response["state"]["turn_count"] == 1
    assert response["token_usage"]["input_tokens"] == 10


def test_chat_preserves_adds_removes_and_resets_context(registry, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("TEXT2SQL_DATABASE_ROOT", str(registry.root))
    monkeypatch.setenv("TEXT2SQL_GLOSSARY_ROOT", str(registry.root / "glossaries"))
    client = TestClient(create_app(TextToSQLAgent(registry, ConversationModel())))
    base = {
        "session_id": "conversation-1",
        "db_id": "shop",
        "provider": "ollama",
        "model": "test-model",
        "execute": True,
    }

    first = client.post(
        "/api/chat", json={**base, "message": "How many orders for each customer?"}
    ).json()
    second = client.post(
        "/api/chat", json={**base, "message": "Now only customers in Germany"}
    ).json()
    third = client.post(
        "/api/chat", json={**base, "message": "Now remove the Germany filter"}
    ).json()
    correction = client.post(
        "/api/chat",
        json={
            **base,
            "message": "That is wrong; only Germany should be included",
            "feedback_category": "missing_filter",
        },
    ).json()
    optimized = client.post(
        "/api/chat", json={**base, "message": "Optimize it"}
    ).json()
    explanation = client.post(
        "/api/chat", json={**base, "message": "Why did you use these tables?"}
    ).json()
    reset = client.post("/api/chat", json={**base, "message": "Reset context"}).json()

    assert first["operation"] == "NEW_QUERY"
    assert first["generation"]["rows"] == [[1, 2], [2, 1]]
    assert second["operation"] == "ADD_FILTER"
    assert second["generation"]["rows"] == [[1, 2]]
    assert second["state"]["modifications"] == ["Now only customers in Germany"]
    assert second["state"]["contract_deltas"] == [
        {
            "operation": "ADD_FILTER",
            "instruction": "Now only customers in Germany",
            "feedback_category": None,
        }
    ]
    assert third["operation"] == "REMOVE_FILTER"
    assert third["generation"]["rows"] == [[1, 2], [2, 1]]
    assert "later instructions override" in third["resolved_question"]
    assert correction["operation"] == "CORRECTION"
    assert correction["generation"]["rows"] == [[1, 2]]
    assert correction["state"]["corrections"] == [
        "Correction category=missing_filter: That is wrong; only Germany should be included"
    ]
    assert optimized["operation"] == "OPTIMIZE"
    assert optimized["generation"]["accepted"] is True
    assert optimized["generation"]["attempts"][0]["validation"]["code"] == (
        "OPTIMIZATION_CHANGED_RESULT"
    )
    assert optimized["generation"]["attempts"][1]["validation"]["code"] == (
        "OPTIMIZATION_NOT_FASTER"
    )
    assert optimized["generation"]["optimization"]["status"] == "equivalent_not_faster"
    assert optimized["generation"]["optimization"]["selected_sql"] == "baseline"
    assert explanation["operation"] == "EXPLAIN"
    assert explanation["generation"] is None
    assert "aggregation=None" in explanation["explanation"]
    assert reset["operation"] == "RESET_CONTEXT"
    assert reset["state"] is None

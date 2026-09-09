from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from semantic_text2sql.agent import TextToSQLAgent
from semantic_text2sql.api import create_app


class ZeroRowsModel:
    async def generate(self, **kwargs: Any) -> tuple[str, int]:
        return "SELECT customer_id FROM customers WHERE country = 'Spain'", 7


class NullRowsModel:
    async def generate(self, **kwargs: Any) -> tuple[str, int]:
        return (
            "SELECT customer_id, NULL AS note FROM customers "
            "WHERE country = 'Germany' AND customer_id > 0",
            9,
        )


def test_api_reports_executable_zero_row_result_once(
    registry, monkeypatch, tmp_path  # type: ignore[no-untyped-def]
) -> None:
    monkeypatch.setenv("TEXT2SQL_DATABASE_ROOT", str(registry.root))
    monkeypatch.setenv("TEXT2SQL_PROFILE_ROOT", str(tmp_path / "profiles"))
    monkeypatch.setenv("TEXT2SQL_GLOSSARY_ROOT", str(tmp_path / "glossaries"))
    client = TestClient(create_app(TextToSQLAgent(registry, ZeroRowsModel())))

    response = client.post(
        "/api/chat",
        json={
            "session_id": "zero-result-review",
            "db_id": "shop",
            "message": "List customers in Spain",
            "provider": "ollama",
            "model": "unused",
            "context_mode": "retrieval",
        },
    )

    body = response.json()
    assert response.status_code == 200
    assert body["generation"]["accepted"] is True
    assert body["generation"]["row_count"] == 0
    assert body["human_review"] is None
    assert "Spain" in body["message"]


def test_api_reports_null_result_once(
    registry, monkeypatch, tmp_path  # type: ignore[no-untyped-def]
) -> None:
    monkeypatch.setenv("TEXT2SQL_DATABASE_ROOT", str(registry.root))
    monkeypatch.setenv("TEXT2SQL_PROFILE_ROOT", str(tmp_path / "profiles"))
    monkeypatch.setenv("TEXT2SQL_GLOSSARY_ROOT", str(tmp_path / "glossaries"))
    client = TestClient(create_app(TextToSQLAgent(registry, NullRowsModel())))

    response = client.post(
        "/api/chat",
        json={
            "session_id": "null-result-review",
            "db_id": "shop",
            "message": "List German customers and their optional note",
            "provider": "ollama",
            "model": "unused",
            "context_mode": "retrieval",
        },
    )

    body = response.json()
    assert response.status_code == 200
    assert body["generation"]["accepted"] is True
    assert body["human_review"] is None
    assert "NULL" in body["message"] or "missing value" in body["message"]

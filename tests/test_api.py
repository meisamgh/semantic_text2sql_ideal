from __future__ import annotations

import time

from fastapi.testclient import TestClient

from semantic_text2sql.api import create_app
from semantic_text2sql.models import ChatRequest


def test_database_catalog_reports_both_dialects(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("POSTGRES_BOOKS_DSN", raising=False)
    monkeypatch.delenv("TEXT2SQL_POSTGRES_DATABASES", raising=False)

    response = TestClient(create_app()).get("/api/databases")

    assert response.status_code == 200
    assert response.json() == [
        {"db_id": "books", "dialect": "sqlite", "configured": True},
        {"db_id": "books_postgres", "dialect": "postgres", "configured": False},
    ]


def test_analytics_capabilities_are_compact_and_do_not_expose_credentials() -> None:
    response = TestClient(create_app()).get("/api/databases/books/analytics-capabilities")

    assert response.status_code == 200
    payload = response.json()
    assert payload["db_id"] == "books"
    assert payload["dialect"] == "sqlite"
    assert payload["tables"]
    assert set(payload) == {
        "db_id",
        "dialect",
        "tables",
        "business_entities",
        "measures",
        "dimensions",
        "time_columns",
        "available_kpis",
    }
    assert "dsn" not in response.text.casefold()
    assert "password" not in response.text.casefold()


def test_database_catalog_uses_postgres_allowlist_without_exposing_dsns(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("POSTGRES_BOOKS_DSN", raising=False)
    monkeypatch.setenv(
        "TEXT2SQL_POSTGRES_DATABASES",
        '{"business":"postgresql://reader:secret@database/business"}',
    )

    response = TestClient(create_app()).get("/api/databases")

    assert response.status_code == 200
    assert {"db_id": "business", "dialect": "postgres", "configured": True} in response.json()
    assert "secret" not in response.text


def test_health_endpoint() -> None:
    response = TestClient(create_app()).get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "online"}


def test_model_catalog_contains_only_agentrouter_and_groq(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("AGENTROUTER_API_KEY", "configured-key")

    options = TestClient(create_app()).get("/api/models").json()
    assert {item["provider"] for item in options} == {"agentrouter", "groq"}
    agentrouter = [item for item in options if item["provider"] == "agentrouter"]
    assert [item["model"] for item in agentrouter] == [
        "gpt-5.6-sol",
        "glm-5.3",
        "deepseek-v4-flash",
        "claude-opus-5",
        "claude-opus-4-8",
    ]
    assert all(item["configured"] is True for item in agentrouter)


def test_production_api_rejects_models_outside_catalog() -> None:
    response = TestClient(create_app()).post(
        "/api/chat",
        json={
            "session_id": "unsupported-model",
            "db_id": "books",
            "message": "Count books",
            "provider": "agentrouter",
            "model": "old-model",
        },
    )

    assert response.status_code == 422
    assert "not allowed" in response.json()["detail"]


def test_public_post_surface_contains_only_current_workflow() -> None:
    app = create_app()
    post_paths = {route.path for route in app.routes if "POST" in getattr(route, "methods", set())}

    assert post_paths == {"/api/check", "/api/chat", "/api/chat/jobs"}


def test_public_check_endpoint_does_not_execute_sql_by_default() -> None:
    response = TestClient(create_app()).post(
        "/api/check",
        json={"db_id": "books", "sql": "SELECT 1", "execute": True},
    )

    assert response.status_code == 403


def test_web_chat_application_is_served() -> None:
    client = TestClient(create_app())

    page = client.get("/")
    script = client.get("/static/app.js")

    assert page.status_code == 200
    assert "Query Room" in page.text
    assert 'id="chatForm"' in page.text
    assert 'id="contextModelSelect"' in page.text
    assert 'id="sqlModelSelect"' in page.text
    assert 'id="contextModeSelect"' not in page.text
    assert 'id="suggestions"' not in page.text
    assert "How many records are in each category?" not in page.text
    assert 'class="technical-panel"' in page.text
    assert 'data-tab="context"' in page.text
    assert 'data-tab="issues"' in page.text
    assert 'data-tab="tokens"' in page.text
    assert "Validation attempts" not in page.text
    assert script.status_code == 200
    assert 'api("/api/chat"' in script.text
    assert "renderTokenAccounting" in script.text
    assert "const label = item.model;" in script.text
    assert "`${item.provider} · ${item.model}`" not in script.text
    assert '"OPTIMIZATION_NOT_FASTER"' in script.text
    assert 'class="semantic-status"' not in page.text
    assert "renderHighlightedSql" in script.text
    assert "renderSqlTokens" in script.text
    assert "sql-line-number" in script.text
    assert '"NOT SELECTED"' in script.text
    assert 'class="human-review-panel"' in page.text
    assert 'class="human-review-editor"' not in page.text
    assert '"Enter another ID"' in script.text
    assert "Enter the replacement ID in the chat" in script.text
    assert 'item.dialect === "sqlite"' not in script.text
    assert "`${item.db_id} (${item.dialect})`" in script.text

    sql_position = page.text.index('class="sql-panel"')
    result_position = page.text.index('class="result-panel"')
    details_position = page.text.index('class="technical-panel"')
    assert sql_position < result_position < details_position


def test_chat_request_accepts_independent_context_and_sql_models() -> None:
    request = ChatRequest(
        session_id="models-test",
        db_id="books",
        message="Count the books",
        provider="agentrouter",
        model="gpt-5.6-sol",
        context_provider="groq",
        context_model="qwen/qwen3.6-27b",
    )

    assert (request.context_provider, request.context_model) == (
        "groq",
        "qwen/qwen3.6-27b",
    )
    assert (request.provider, request.model) == ("agentrouter", "gpt-5.6-sol")


def test_retrieval_and_model1_are_distinct_request_paths() -> None:
    retrieval = ChatRequest(
        session_id="retrieval-test",
        db_id="books",
        message="Count the books",
        context_mode="retrieval",
    )
    model1 = ChatRequest(
        session_id="model1-test",
        db_id="books",
        message="Count the books",
        context_mode="model1",
        context_provider="agentrouter",
        context_model="gpt-5.6-sol",
    )

    assert retrieval.context_mode == "retrieval"
    assert retrieval.context_provider is None
    assert retrieval.context_model is None
    assert model1.context_mode == "model1"
    assert model1.context_provider == "agentrouter"
    assert model1.context_model == "gpt-5.6-sol"


def test_retrieval_is_the_api_default() -> None:
    request = ChatRequest(
        session_id="default-path-test",
        db_id="books",
        message="Count the books",
    )

    assert request.context_mode == "retrieval"


def test_chat_job_reports_progress_and_completion() -> None:
    client = TestClient(create_app())
    started = client.post(
        "/api/chat/jobs",
        json={
            "session_id": "job-test",
            "db_id": "books",
            "message": "Reset context",
        },
    )

    assert started.status_code == 202
    job_id = started.json()["job_id"]
    job = client.get(f"/api/chat/jobs/{job_id}").json()
    for _ in range(20):
        if job["status"] == "completed":
            break
        time.sleep(0.01)
        job = client.get(f"/api/chat/jobs/{job_id}").json()

    assert job["status"] == "completed"
    assert job["response"]["operation"] == "RESET_CONTEXT"
    assert job["elapsed_ms"] >= 0

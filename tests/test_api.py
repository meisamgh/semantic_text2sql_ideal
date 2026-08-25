from __future__ import annotations

import time

from fastapi.testclient import TestClient

from semantic_text2sql.api import create_app
from semantic_text2sql.models import ChatRequest


def test_database_catalog_reports_both_dialects(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("POSTGRES_BOOKS_DSN", raising=False)

    response = TestClient(create_app()).get("/api/databases")

    assert response.status_code == 200
    assert response.json() == [
        {"db_id": "books", "dialect": "sqlite", "configured": True},
        {"db_id": "books_postgres", "dialect": "postgres", "configured": False},
    ]


def test_health_endpoint() -> None:
    response = TestClient(create_app()).get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "online"}


def test_public_post_surface_contains_only_current_workflow() -> None:
    app = create_app()
    post_paths = {route.path for route in app.routes if "POST" in getattr(route, "methods", set())}

    assert post_paths == {"/api/check", "/api/chat", "/api/chat/jobs"}


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
    assert 'class="token-panel"' in page.text
    assert script.status_code == 200
    assert 'api("/api/chat"' in script.text
    assert "renderTokenAccounting" in script.text
    assert "const label = item.model;" in script.text
    assert "`${item.provider} · ${item.model}`" not in script.text

    sql_position = page.text.index('class="sql-panel"')
    result_position = page.text.index('class="result-panel"')
    validation_position = page.text.index('class="semantic-status"')
    attempts_position = page.text.index('class="attempts-panel"')
    tokens_position = page.text.index('class="token-panel"')
    assert sql_position < result_position < validation_position
    assert validation_position < attempts_position < tokens_position


def test_chat_request_accepts_independent_context_and_sql_models() -> None:
    request = ChatRequest(
        session_id="models-test",
        db_id="books",
        message="Count the books",
        provider="agentrouter",
        model="gpt-5.6-sol",
        context_provider="ollama",
        context_model="qwen3.5:9b",
    )

    assert (request.context_provider, request.context_model) == ("ollama", "qwen3.5:9b")
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

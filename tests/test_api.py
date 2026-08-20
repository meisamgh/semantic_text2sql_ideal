from __future__ import annotations

import time

from fastapi.testclient import TestClient

from semantic_text2sql.api import create_app


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
    post_paths = {
        route.path
        for route in app.routes
        if "POST" in getattr(route, "methods", set())
    }

    assert post_paths == {"/api/check", "/api/chat", "/api/chat/jobs"}


def test_web_chat_application_is_served() -> None:
    client = TestClient(create_app())

    page = client.get("/")
    script = client.get("/static/app.js")

    assert page.status_code == 200
    assert "Query Room" in page.text
    assert 'id="chatForm"' in page.text
    assert script.status_code == 200
    assert 'api("/api/chat"' in script.text


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

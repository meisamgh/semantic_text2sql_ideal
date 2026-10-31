from __future__ import annotations

import sqlite3
import time
from typing import Any

from fastapi.testclient import TestClient

from semantic_text2sql.agent import TextToSQLAgent
from semantic_text2sql.api import create_app
from semantic_text2sql.conversation import ConversationStore
from semantic_text2sql.models import ChatRequest
from semantic_text2sql.run_history import RunHistoryStore


class FixedSQLModel:
    def __init__(self, sql: str) -> None:
        self.sql = sql
        self.calls = 0

    async def generate(self, **kwargs: Any) -> tuple[str, int]:
        self.calls += 1
        return self.sql, 1


def test_unresolved_history_branch_preserves_accepted_state_and_result(
    registry: Any, monkeypatch: Any, tmp_path: Any,
) -> None:
    monkeypatch.setenv("TEXT2SQL_DATABASE_ROOT", str(registry.root))
    state_path = tmp_path / "conversation.sqlite"
    monkeypatch.setenv("TEXT2SQL_CONVERSATION_STORE", str(state_path))
    model = FixedSQLModel("SELECT COUNT(*) AS customer_count FROM customers")
    history = RunHistoryStore(None)
    client = TestClient(create_app(TextToSQLAgent(registry, model), run_history=history))
    base = {"session_id": "branch-isolation", "db_id": "shop",
            "provider": "agentrouter", "model": "test-model"}
    first = client.post("/api/chat", json={**base, "message": "Count customers"}).json()
    assert first["generation"]["accepted"]
    store = ConversationStore(state_path)
    original = store.get("branch-isolation")
    assert original is not None
    accepted = store.put(original.model_copy(update={
        "resolved_question": "Count orders", "last_sql": "SELECT COUNT(*) FROM orders",
    }), expected_version=original.version)
    model.sql = "SELECT missing_column FROM missing_table"
    failed = client.post("/api/chat", json={
        **base, "message": "New question: count customers again", "parent_run_id": first["run_id"],
    })
    assert failed.status_code == 200
    assert failed.json()["generation"] is None or failed.json()["generation"]["accepted"] is False
    after = ConversationStore(state_path).get("branch-isolation")
    assert after is not None and after.version == accepted.version
    assert after.last_sql == accepted.last_sql


def test_new_conversation_reset_does_not_call_model_or_delete_history(
    registry: Any, monkeypatch: Any
) -> None:
    monkeypatch.setenv("TEXT2SQL_DATABASE_ROOT", str(registry.root))
    model = FixedSQLModel("SELECT COUNT(*) AS customer_count FROM customers")
    history = RunHistoryStore(None)
    client = TestClient(create_app(TextToSQLAgent(registry, model), run_history=history))
    request = {
        "session_id": "reset-without-model", "db_id": "shop",
        "provider": "agentrouter", "model": "test-model",
        "message": "How many customers?",
    }
    first = client.post("/api/chat", json=request)
    assert first.status_code == 200
    assert first.json()["generation"]["accepted"] is True
    assert model.calls == 1
    assert client.post("/api/conversations/reset-without-model/reset").status_code == 204
    assert model.calls == 1
    assert history.get(first.json()["run_id"], "reset-without-model") is not None
    second = client.post("/api/chat", json=request)
    assert second.status_code == 200
    assert second.json()["operation"] == "NEW_QUERY"


def test_visualization_followups_reuse_complete_results_without_new_sql(
    registry,
    monkeypatch,
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("TEXT2SQL_DATABASE_ROOT", str(registry.root))
    monkeypatch.setenv("TEXT2SQL_PROFILE_ROOT", str(tmp_path / "profiles"))
    monkeypatch.setenv("TEXT2SQL_GLOSSARY_ROOT", str(tmp_path / "glossaries"))
    model = FixedSQLModel(
        "SELECT country, COUNT(*) AS customer_count FROM customers GROUP BY country"
    )
    store = RunHistoryStore(None)
    client = TestClient(create_app(TextToSQLAgent(registry, model), run_history=store))
    base = {
        "session_id": "viz-session",
        "db_id": "shop",
        "provider": "agentrouter",
        "model": "test-model",
    }

    ordinary = client.post(
        "/api/chat", json={**base, "message": "Show customer counts by country."}
    )
    assert ordinary.status_code == 200
    first = ordinary.json()
    assert first["generation"]["accepted"] is True
    assert first["presentation"]["chart_spec"] is None
    initial_calls = model.calls

    plotted = client.post("/api/chat", json={**base, "message": "Plot it."})
    assert plotted.status_code == 200
    chart = plotted.json()
    assert chart["operation"] == "VISUALIZE"
    assert chart["presentation"]["chart_spec"]["chart_type"] == "bar"
    assert chart["generation"]["sql"] == first["generation"]["sql"]
    assert chart["token_usage"]["input_tokens"] is None
    assert chart["call_ledger"] == []
    assert model.calls == initial_calls

    incompatible = client.post("/api/chat", json={**base, "message": "Make it a line chart."})
    assert incompatible.status_code == 200
    assert incompatible.json()["presentation"]["chart_spec"] is None
    assert "ordered date or numeric" in incompatible.json()["message"]
    assert model.calls == initial_calls
    assert [item.operation for item in store.list_sessions("viz-session")[:3]] == [
        "VISUALIZE",
        "VISUALIZE",
        "NEW_QUERY",
    ]

    explicit = client.post("/api/chat", json={
        **base, "session_id": "fresh-viz",
        "message": "Plot customer counts by country as a bar chart.",
    })
    assert explicit.status_code == 200
    assert explicit.json()["presentation"]["chart_spec"]["chart_type"] == "bar"
    assert not any(item["component"] == "chart_selection"
                   for item in explicit.json()["call_ledger"])

    incomplete = client.post("/api/chat", json={
        **base, "session_id": "small-viz",
        "message": "Plot customer counts by country.", "max_rows": 1,
    })
    assert incomplete.status_code == 200
    assert incomplete.json()["generation"]["truncated"] is True
    assert incomplete.json()["presentation"]["chart_spec"] is None
    assert "truncated" in incomplete.json()["presentation"]["chart_reason"]

    limited_followup = client.post("/api/chat", json={
        **base, "session_id": "small-viz", "message": "Plot it.",
    }).json()
    assert limited_followup["clarification_required"] is True
    assert "exceeded its row limit" in limited_followup["message"]


def test_chart_type_change_reuses_previous_axes_without_model_or_sql(
    registry, monkeypatch, tmp_path
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("TEXT2SQL_DATABASE_ROOT", str(registry.root))
    monkeypatch.setenv("TEXT2SQL_PROFILE_ROOT", str(tmp_path / "profiles"))
    monkeypatch.setenv("TEXT2SQL_GLOSSARY_ROOT", str(tmp_path / "glossaries"))
    model = FixedSQLModel(
        "SELECT order_id, customer_id, amount FROM orders ORDER BY order_id"
    )
    client = TestClient(create_app(TextToSQLAgent(registry, model)))
    base = {"session_id": "change-chart-type", "db_id": "shop",
            "provider": "agentrouter", "model": "test-model"}
    first = client.post("/api/chat", json={
        **base, "message": "Show order amounts."
    }).json()
    plotted = client.post("/api/chat", json={
        **base, "message": "Make it a scatter plot."
    }).json()
    original = plotted["presentation"]["chart_spec"]
    assert original is not None, first["message"]
    assert original["chart_type"] == "scatter"
    calls = model.calls

    changed = client.post("/api/chat", json={
        **base, "message": "Change this plot to a bar chart."
    }).json()
    chart = changed["presentation"]["chart_spec"]
    assert chart is not None, changed["message"]
    assert chart["chart_type"] == "bar"
    assert (chart["x"], chart["y"]) == (original["x"], original["y"])
    assert changed["generation"]["sql"] == first["generation"]["sql"]
    assert changed["call_ledger"] == []
    assert model.calls == calls

def test_five_column_plot_followup_uses_full_ram_result_not_history_preview_or_model(
    registry, monkeypatch, tmp_path
) -> None:  # type: ignore[no-untyped-def]
    with sqlite3.connect(registry.resolve("shop")) as connection:
        connection.executemany(
            "INSERT INTO orders VALUES (?, 1, ?, 'complete')",
            [(number, float(number * number)) for number in range(13, 35)],
        )
    monkeypatch.setenv("TEXT2SQL_DATABASE_ROOT", str(registry.root))
    monkeypatch.setenv("TEXT2SQL_PROFILE_ROOT", str(tmp_path / "profiles"))
    monkeypatch.setenv("TEXT2SQL_GLOSSARY_ROOT", str(tmp_path / "glossaries"))
    model = FixedSQLModel(
        "SELECT o.order_id, c.name, c.country, o.amount, o.status "
        "FROM orders AS o JOIN customers AS c ON c.customer_id = o.customer_id "
        "ORDER BY o.order_id"
    )
    history = RunHistoryStore(None)
    client = TestClient(create_app(TextToSQLAgent(registry, model), run_history=history))
    base = {"session_id": "five-column-viz", "db_id": "shop",
            "provider": "agentrouter", "model": "test-model"}
    first = client.post("/api/chat", json={**base, "message": "Show the orders."}).json()
    assert first["generation"]["accepted"] is True
    assert len(first["generation"]["columns"]) == 5
    assert len(first["generation"]["rows"]) > 20
    assert len(history.list_sessions(base["session_id"])[0].rows_preview) == 20
    model_calls = model.calls

    chart = client.post("/api/chat", json={**base, "message": "Plot it."}).json()
    assert chart["operation"] == "VISUALIZE"
    assert chart["presentation"]["chart_spec"] is not None, chart["message"]
    assert len(chart["presentation"]["chart_spec"]["data"]) == len(first["generation"]["rows"])
    assert chart["generation"]["rows"] == first["generation"]["rows"]
    assert chart["call_ledger"] == []
    assert model.calls == model_calls


def test_multi_column_plot_followup_uses_ram_without_model_or_sql(
    registry, monkeypatch, tmp_path
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("TEXT2SQL_DATABASE_ROOT", str(registry.root))
    monkeypatch.setenv("TEXT2SQL_PROFILE_ROOT", str(tmp_path / "profiles"))
    monkeypatch.setenv("TEXT2SQL_GLOSSARY_ROOT", str(tmp_path / "glossaries"))
    # Grouped rows keep one measure per category so the deterministic
    # recommender can chart them without inventing client-side aggregation
    # over this row-grain result.
    model = FixedSQLModel(
        "SELECT c.name AS name, c.country AS country, "
        "SUM(o.amount) AS total_amount, COUNT(*) AS order_count "
        "FROM orders AS o JOIN customers AS c ON c.customer_id = o.customer_id "
        "GROUP BY c.name, c.country"
    )
    client = TestClient(
        create_app(TextToSQLAgent(registry, model), run_history=RunHistoryStore(None))
    )
    base = {"session_id": "multi-column-viz", "db_id": "shop",
            "provider": "agentrouter", "model": "test-model"}
    first = client.post("/api/chat", json={**base, "message": "Show the orders."}).json()
    assert first["generation"]["accepted"] is True
    assert len(first["generation"]["columns"]) == 4
    model_calls = model.calls

    chart = client.post("/api/chat", json={**base, "message": "Plot it."}).json()
    assert chart["operation"] == "VISUALIZE"
    assert chart["presentation"]["chart_spec"] is not None, chart["message"]
    assert chart["generation"]["rows"] == first["generation"]["rows"]
    assert chart["call_ledger"] == []
    assert model.calls == model_calls


def test_database_catalog_reports_both_dialects(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("POSTGRES_BOOKS_DSN", raising=False)
    monkeypatch.delenv("TEXT2SQL_POSTGRES_DATABASES", raising=False)

    response = TestClient(create_app()).get("/api/databases")

    assert response.status_code == 200
    assert response.json() == [
        {"db_id": "books", "dialect": "sqlite", "configured": True},
        {"db_id": "books_postgres", "dialect": "postgres", "configured": False},
    ]


def test_ui_config_reads_safe_model_defaults_from_environment(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("TEXT2SQL_SQL_PROVIDER", "agentrouter")
    monkeypatch.setenv("TEXT2SQL_SQL_MODEL", "gpt-5.6-sol")
    monkeypatch.setenv("TEXT2SQL_CONTEXT_MODE", "model1")
    monkeypatch.setenv("TEXT2SQL_CONTEXT_PROVIDER", "groq")
    monkeypatch.setenv("TEXT2SQL_CONTEXT_MODEL", "qwen/qwen3.8-27b")
    monkeypatch.setenv("TEXT2SQL_DEFAULT_DB_ID", "books")
    response = TestClient(create_app()).get("/api/ui-config")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json() == {
        "sql_provider": "agentrouter",
        "sql_model": "gpt-5.6-sol",
        "context_mode": "model1",
        "context_provider": "groq",
        "context_model": "qwen/qwen3.8-27b",
        "assistant_provider": "agentrouter",
        "assistant_model": "gpt-5.6-sol",
        "plotting_provider": "agentrouter",
        "plotting_model": "gpt-5.6-sol",
        "default_db_id": "books",
    }


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


def test_history_records_runs_and_serves_reopen(registry, monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("TEXT2SQL_DATABASE_ROOT", str(registry.root))
    monkeypatch.setenv("TEXT2SQL_PROFILE_ROOT", str(tmp_path / "profiles"))
    monkeypatch.setenv("TEXT2SQL_GLOSSARY_ROOT", str(tmp_path / "glossaries"))
    store = RunHistoryStore(None)
    client = TestClient(
        create_app(
            TextToSQLAgent(
                registry,
                FixedSQLModel("SELECT customer_id, COUNT(*) FROM orders GROUP BY customer_id"),
            ),
            run_history=store,
        )
    )
    base = {
        "session_id": "history-test",
        "db_id": "shop",
        "provider": "agentrouter",
        "model": "test-model",
        "execute": True,
    }
    client.post("/api/chat", json={**base, "message": "How many orders for each customer?"})
    client.post("/api/chat", json={**base, "message": "Now only customers in Germany"})

    listing = client.get("/api/history", params={"session_id": "history-test"})
    assert listing.status_code == 200
    assert listing.headers["cache-control"] == "no-store"
    runs = listing.json()
    assert [run["version"] for run in runs] == [2, 1]
    assert [run["original_message"] for run in runs] == [
        "Now only customers in Germany",
        "How many orders for each customer?",
    ]
    assert runs[0]["parent_run_id"] == runs[1]["run_id"]
    assert runs[1]["parent_run_id"] is None
    assert all(run["accepted"] is True for run in runs)
    assert all(len(run["rows_preview"]) <= 25 for run in runs)
    assert all(run["db_id"] == "shop" for run in runs)

    reopened = client.get(
        f"/api/history/{runs[0]['run_id']}", params={"session_id": "history-test"}
    )
    assert reopened.status_code == 200
    assert reopened.json()["sql"].startswith("SELECT")
    assert reopened.json()["columns"] == ["customer_id", "COUNT(*)"]
    assert (
        client.get("/api/history/missing-run", params={"session_id": "history-test"}).status_code
        == 404
    )

    cleared = client.delete("/api/history", params={"session_id": "history-test"})
    assert cleared.status_code == 204
    assert client.get("/api/history", params={"session_id": "history-test"}).json() == []


def test_sql_diff_endpoint_is_sqlglot_aware_and_falls_back_gracefully() -> None:
    client = TestClient(create_app())

    changed = client.post(
        "/api/sql/diff",
        json={
            "previous_sql": "SELECT customer_id, COUNT(*) FROM orders GROUP BY customer_id",
            "updated_sql": "SELECT o.customer_id, COUNT(*) + 1 FROM orders o "
            "JOIN customers c ON o.customer_id = c.customer_id GROUP BY o.customer_id",
        },
    )
    assert changed.status_code == 200
    body = changed.json()
    assert body["parse_error"] is False
    assert {change["kind"] for change in body["changes"]} == {"remove", "insert"}

    formatting = client.post(
        "/api/sql/diff",
        json={
            "previous_sql": "SELECT * FROM orders",
            "updated_sql": "select * from orders;",
        },
    )
    assert formatting.json()["parse_error"] is False
    assert formatting.json()["changes"] == []
    assert set(formatting.json()["invariants"].values()) == {"unchanged"}

    broken = client.post(
        "/api/sql/diff",
        json={
            "previous_sql": "not sql at all ((",
            "updated_sql": "still broken",
        },
    )
    assert broken.status_code == 200
    assert broken.json()["parse_error"] is True
    assert broken.json()["changes"]


def test_health_endpoint() -> None:
    response = TestClient(create_app()).get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "online"}


def test_schema_browser_returns_tables_and_types_without_rows() -> None:
    client = TestClient(create_app())
    response = client.get("/api/databases/books/schema")
    assert response.status_code == 200
    schema = response.json()
    assert schema["tables"]
    for table in schema["tables"]:
        assert set(table) == {"name", "columns"}
        assert table["columns"]
        for column in table["columns"]:
            assert set(column) == {"name", "type", "primary_key"}
    assert client.get("/api/databases/not_configured/schema").status_code == 404


def test_schema_browser_exposes_declared_relationships(registry, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("TEXT2SQL_DATABASE_ROOT", str(registry.root))
    response = TestClient(create_app()).get("/api/databases/shop/schema")
    assert response.status_code == 200
    assert {
        "from_table": "orders",
        "from_column": "customer_id",
        "to_table": "customers",
        "to_column": "customer_id",
        "source": "declared_foreign_key",
        "cardinality": None,
    } in response.json()["relationships"]


def test_model_catalog_contains_configured_remote_providers(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("AGENTROUTER_API_KEY", "configured-key")
    monkeypatch.setenv("JUSTDOWORK_API_KEY", "configured-key")
    monkeypatch.setenv("JUSTDOWORK_BASE_URL", "https://justdowork.test/v1")
    monkeypatch.setenv("CODECRAFT_API_KEY", "configured-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "configured-key")

    options = TestClient(create_app()).get("/api/models").json()
    assert {item["provider"] for item in options} == {
        "agentrouter",
        "groq",
        "justdowork",
        "codecraft",
        "true_sota",
        "openrouter",
    }
    agentrouter = [item for item in options if item["provider"] == "agentrouter"]
    assert [item["model"] for item in agentrouter] == [
        "gpt-5.6-sol",
        "glm-5.3",
        "deepseek-v4-flash",
        "claude-opus-5",
        "claude-opus-4-8",
    ]
    assert all(item["configured"] is True for item in agentrouter)
    assert [item["model"] for item in options if item["provider"] == "justdowork"] == [
        "gpt-5.6-sol",
        "claude-opus-4-8",
    ]
    assert [item["model"] for item in options if item["provider"] == "codecraft"] == [
        "claude-opus-5"
    ]
    assert [item["model"] for item in options if item["provider"] == "openrouter"] == [
        "nvidia/nemotron-3-ultra-550b-a55b:free"
    ]


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
    # Included routers are lazy route groups in current FastAPI; inspect the
    # complete public schema rather than omitting grouped endpoints.
    post_paths = {path for path, methods in app.openapi()["paths"].items() if "post" in methods}

    assert post_paths == {
        "/api/conversations/{session_id}/reset",
        "/api/check",
        "/api/chat",
        "/api/chat/jobs",
        "/api/results/page",
        "/api/results/inspect",
        "/api/dashboards",
        "/api/dashboards/undo",
        "/api/studio",
        "/api/studio/{identity}/edit",
        "/api/studio/{identity}/save",
        "/api/studio/{identity}/rename",
        "/api/studio/{identity}/publish",
        "/api/studio/{identity}/undo",
        "/api/sql/diff",
        "/api/voice/transcribe",
        "/api/history/{run_id}/continue",
        "/api/history/{run_id}/rerun",
    }


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
    assert "What would you like to know?" in page.text
    assert 'id="chatForm"' in page.text
    assert 'id="contextModelSelect"' not in page.text
    assert 'id="sqlModelSelect"' not in page.text
    assert 'id="contextModeSelect"' not in page.text
    assert 'id="suggestions"' not in page.text
    assert "How many records are in each category?" not in page.text
    assert 'class="response-card"' in page.text
    assert 'class="sql-result-split"' in page.text
    assert 'class="versions-panel"' in page.text
    assert 'class="recovery-panel"' in page.text
    assert 'class="human-review"' in page.text
    assert 'id="inspectorPane"' in page.text
    assert "Download CSV" in page.text
    assert "Validation attempts" not in page.text
    assert script.status_code == 200
    assert 'api("/api/chat/jobs"' in script.text
    assert "renderTokenAccounting" in script.text
    assert 'api("/api/ui-config", { cache: "no-store" })' in script.text
    assert page.text.index('id="databaseSelect"') < page.text.index('id="chatForm"')
    assert page.text.index('id="connectionCard"') < page.text.index('id="chatForm"')
    assert "`${item.provider} · ${item.model}`" not in script.text
    assert '"OPTIMIZATION_NOT_FASTER"' in script.text
    assert 'class="semantic-status"' not in page.text
    assert "renderHighlightedSql" in script.text
    assert "renderSqlTokens" in script.text
    assert "sql-line-number" in script.text
    assert '"NOT SELECTED"' in script.text
    assert 'class="human-review-panel"' not in page.text
    assert 'class="human-review-editor"' not in page.text
    assert '"Enter another ID"' in script.text
    assert "Enter the replacement ID in the chat" in script.text
    assert 'item.dialect === "sqlite"' not in script.text
    assert "`${item.db_id} (${item.dialect})`" in script.text

    sql_position = page.text.index('class="panel-card sql-panel-pane"')
    result_position = page.text.index('class="panel-card result-panel-pane"')
    versions_position = page.text.index('class="versions-panel"')
    recovery_position = page.text.index('class="recovery-panel"')
    # Run details remain in the inspector; the answer template stays clean.
    assert 'class="technical-details"' not in page.text
    tokens_position = page.text.index('id="tokensContent"')
    assert 'id="pipelineContent"' in page.text
    assert 'class="attempts-panel' not in page.text
    assert 'class="attempts-panel' not in script.text
    # The inspector (with token and pipeline details) is defined before the
    # answer template in the document, and the answer keeps only its own
    # closable panels inline.
    assert tokens_position < sql_position
    assert (
        sql_position
        < result_position
        < versions_position
        < recovery_position
    )


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


def test_chat_request_uses_environment_models_unless_explicitly_overridden(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("TEXT2SQL_SQL_PROVIDER", "groq")
    monkeypatch.setenv("TEXT2SQL_SQL_MODEL", "qwen/qwen3.8-27b")
    monkeypatch.setenv("TEXT2SQL_CONTEXT_MODE", "model1")
    monkeypatch.setenv("TEXT2SQL_CONTEXT_PROVIDER", "agentrouter")
    monkeypatch.setenv("TEXT2SQL_CONTEXT_MODEL", "gpt-5.6-sol")

    default = ChatRequest(session_id="env-models", db_id="books", message="Count the books")
    assert (default.provider, default.model) == ("groq", "qwen/qwen3.8-27b")
    assert (default.context_mode, default.context_provider, default.context_model) == (
        "model1", "agentrouter", "gpt-5.6-sol"
    )

    explicit = ChatRequest(
        session_id="explicit-models", db_id="books", message="Count the books",
        provider="agentrouter", model="gpt-5.6-sol", context_mode="retrieval",
    )
    assert (explicit.provider, explicit.model, explicit.context_mode) == (
        "agentrouter", "gpt-5.6-sol", "retrieval"
    )


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
    assert "events" in job
    assert job["events"][0] == {"stage": "conversation", "elapsed_ms": 0}
    time.sleep(0.02)
    assert client.get(f"/api/chat/jobs/{job_id}").json()["elapsed_ms"] == job["elapsed_ms"]


def test_history_restore_rebinds_plot_to_selected_result(
    registry: Any, monkeypatch: Any, tmp_path: Any
) -> None:
    monkeypatch.setenv("TEXT2SQL_DATABASE_ROOT", str(registry.root))
    monkeypatch.setenv("TEXT2SQL_PROFILE_ROOT", str(tmp_path / "profiles"))
    monkeypatch.setenv("TEXT2SQL_GLOSSARY_ROOT", str(tmp_path / "glossaries"))
    model = FixedSQLModel(
        "SELECT country, COUNT(*) AS customer_count FROM customers GROUP BY country"
    )
    store = RunHistoryStore(None)
    client = TestClient(create_app(TextToSQLAgent(registry, model), run_history=store))
    base = {
        "session_id": "restore-result", "db_id": "shop",
        "provider": "agentrouter", "model": "test-model",
    }
    first = client.post("/api/chat", json={**base, "message": "Count by country"}).json()
    second = client.post("/api/chat", json={**base, "message": "Count by country again"}).json()
    assert first["result_id"] != second["result_id"]
    calls_before_restore = model.calls
    restored = client.post(
        f"/api/history/{first['run_id']}/continue", params={"session_id": "restore-result"}
    )
    assert restored.status_code == 200
    chart = client.post("/api/chat", json={**base, "message": "Plot it"})
    assert chart.status_code == 200
    body = chart.json()
    assert body["presentation"]["chart_spec"] is not None
    assert body["result_id"] == first["result_id"]
    assert body["presentation_state"]["source_run_id"] == first["run_id"]
    assert body["presentation_state"]["source_result_id"] == first["result_id"]
    assert model.calls == calls_before_restore

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.authentication import AuthCredentials, SimpleUser
from starlette.middleware.authentication import AuthenticationMiddleware
from test_dashboard import snapshot
from test_dashboard_store import document

from semantic_text2sql.agent import TextToSQLAgent
from semantic_text2sql.ai_dashboard_designer import DashboardRevisions
from semantic_text2sql.api import create_app
from semantic_text2sql.dashboard_store import DashboardPrincipal, DashboardStore
from semantic_text2sql.dashboard_studio import studio_router
from semantic_text2sql.models import ChatResponse, TokenUsage
from semantic_text2sql.run_history import RunHistoryStore


def local_client(app):
    return TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 55000))


def studio_app(path, *, with_source=True, edit=None):
    store = DashboardStore(path)
    revisions = DashboardRevisions()
    spec = revisions.commit("owner", document().spec, None)
    source = snapshot()

    async def unavailable(*args):
        return ChatResponse(
            session_id="owner",
            operation="VISUALIZE",
            resolved_question="",
            message="Design unavailable",
        )

    app = FastAPI()
    app.include_router(
        studio_router(
            store,
            revisions,
            lambda session, db, result: (
                source
                if with_source and (session, db, result) == ("owner", "sales", "result")
                else None
            ),
            edit or unavailable,
        )
    )
    return app, {
        "session_id": "owner",
        "db_id": "sales",
        "result_id": "result",
        "design_revision_id": spec.revision_id,
        "name": "Revenue",
    }


def test_library_create_restart_open_publish_versions_delete(tmp_path):
    path = tmp_path / "history.sqlite"
    app, payload = studio_app(path)
    client = local_client(app)
    saved = client.post("/api/studio", json=payload)
    assert saved.status_code == 200, saved.text
    saved = saved.json()
    identity = saved["dashboard_id"]
    assert saved["owner"] == "local-workspace" and saved["status"] == "Draft"
    assert client.get("/api/studio").json()[0]["id"] == identity
    # New app instance, no original RAM source and no model/executor capability.
    restarted, _ = studio_app(path, with_source=False)
    client = local_client(restarted)
    reopened = client.get(f"/api/studio/{identity}")
    assert reopened.json() == saved
    assert "no-store" in reopened.headers["cache-control"]
    assert (
        client.post(f"/api/studio/{identity}/publish", json={"expected_version": 1}).status_code
        == 422
    )
    published = client.post(
        f"/api/studio/{identity}/publish", json={"expected_version": 1, "approve": True}
    )
    assert published.json()["status"] == "Published"
    assert (
        client.post(
            f"/api/studio/{identity}/rename", json={"expected_version": 1, "name": "Stale"}
        ).status_code
        == 409
    )
    renamed = client.post(
        f"/api/studio/{identity}/rename", json={"expected_version": 2, "name": "Renamed"}
    ).json()
    assert renamed["version"] == 3
    assert len(client.get(f"/api/studio/{identity}/versions").json()) == 3
    historical = client.get(f"/api/studio/{identity}?version=1").json()
    assert historical["document"] == saved["document"]
    copied = client.post(
        f"/api/studio/{identity}/save",
        json={"expected_version": 3, "save_as_new": True, "source_version": 1, "name": "Copy"},
    ).json()
    assert copied["dashboard_id"] != identity and copied["status"] == "Draft"
    assert client.delete(f"/api/studio/{identity}?expected_version=2").status_code == 409
    assert client.delete(f"/api/studio/{identity}?expected_version=3").status_code == 204
    assert client.get(f"/api/studio/{identity}").status_code == 404


def test_local_security_remote_host_origin_owner_and_forged_specs(tmp_path):
    app, payload = studio_app(tmp_path / "history.sqlite")
    assert TestClient(app).get("/api/studio").status_code == 401
    client = local_client(app)
    assert client.get("/api/studio", headers={"Host": "evil.example"}).status_code == 401
    assert client.get("/api/studio", headers={"Origin": "https://evil.example"}).status_code == 403
    assert client.get("/api/studio", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
    assert client.get("/api/studio", headers={"X-Forwarded-For": "127.0.0.1"}).status_code == 401
    assert client.post("/api/studio", json={**payload, "owner": "somebody-else"}).status_code == 422
    assert (
        client.post(
            "/api/studio", json={**payload, "spec": document().spec.model_dump()}
        ).status_code
        == 422
    )
    assert (
        client.post("/api/studio", json={**payload, "design_revision_id": "old"}).status_code == 409
    )
    assert client.post("/api/studio", json={**payload, "session_id": "another"}).status_code == 404
    assert client.post("/api/studio/share", json={}).status_code in {404, 405}


def test_authenticated_ownership_requires_trusted_session_binding(tmp_path):
    from starlette.authentication import AuthenticationBackend

    class TrustedIdentity(AuthenticationBackend):
        async def authenticate(self, connection):
            # Synthetic ASGI identity fixture; not a client-controlled login.
            class Identity(SimpleUser):
                @property
                def identity(self):
                    return self.username

            return AuthCredentials(["authenticated"]), Identity("alice")

    app, payload = studio_app(tmp_path / "history.sqlite")
    app.add_middleware(AuthenticationMiddleware, backend=TrustedIdentity())
    client = TestClient(app)
    assert client.post("/api/studio", json=payload).status_code == 403
    store = DashboardStore(tmp_path / "history.sqlite")
    mine = store.create(DashboardPrincipal("auth:alice", True), "Mine", document())
    theirs = store.create(DashboardPrincipal("auth:bob", True), "Other", document())
    assert client.get(f"/api/studio/{mine.dashboard_id}").status_code == 200
    assert client.get(f"/api/studio/{theirs.dashboard_id}").status_code == 404
    assert [row["name"] for row in client.get("/api/studio").json()] == ["Mine"]


def test_server_validated_preview_save_and_optimistic_conflict(tmp_path):
    calls = []
    themes = []

    async def edit(current, objective):
        calls.append(objective)
        themes.append(current.document.spec.theme)
        spec = current.document.spec.model_copy(update={"theme": "contrast"})
        return ChatResponse(
            session_id="studio",
            operation="VISUALIZE",
            resolved_question=objective,
            message="Preview ready.",
            dashboard=spec.model_dump(),
        )

    app, payload = studio_app(tmp_path / "history.sqlite", edit=edit)
    client = local_client(app)
    saved = client.post("/api/studio", json=payload).json()
    identity = saved["dashboard_id"]
    preview = client.post(
        f"/api/studio/{identity}/edit",
        json={"expected_version": 1, "message": "Make it higher contrast"},
    ).json()
    assert calls == ["Make it higher contrast"]
    second = client.post(
        f"/api/studio/{identity}/edit",
        json={
            "expected_version": 1,
            "message": "Keep this draft and make another change",
            "preview_id": preview["preview_id"],
        },
    )
    assert second.status_code == 200
    assert themes == ["green", "contrast"]
    assert client.get(f"/api/studio/{identity}").json()["version"] == 1
    assert (
        client.post(
            f"/api/studio/{identity}/save", json={"expected_version": 1, "preview_id": "forged"}
        ).status_code
        == 409
    )
    saved = client.post(
        f"/api/studio/{identity}/save",
        json={"expected_version": 1, "preview_id": preview["preview_id"]},
    ).json()
    assert saved["version"] == 2 and saved["document"]["spec"]["theme"] == "contrast"
    assert (
        client.post(
            f"/api/studio/{identity}/save",
            json={"expected_version": 1, "preview_id": preview["preview_id"]},
        ).status_code
        == 409
    )
    undone = client.post(f"/api/studio/{identity}/undo", json={"expected_version": 2}).json()
    assert undone["version"] == 3 and undone["document"]["spec"]["theme"] == "green"


@pytest.mark.parametrize("route", ["design", "refresh"])
def test_real_top_layer_edit_after_restart_never_executes_sql(
    registry, tmp_path, monkeypatch, route
):
    path = tmp_path / "history.sqlite"
    owner = DashboardPrincipal("local-workspace")
    saved = DashboardStore(path).create(owner, "Revenue", document())
    monkeypatch.setenv("TEXT2SQL_DATABASE_ROOT", str(registry.root))
    monkeypatch.setenv("TEXT2SQL_ASSISTANT_PROVIDER", "agentrouter")
    monkeypatch.setenv("TEXT2SQL_ASSISTANT_MODEL", "script")
    monkeypatch.setenv("TEXT2SQL_DASHBOARD_PROVIDER", "agentrouter")
    monkeypatch.setenv("TEXT2SQL_DASHBOARD_MODEL", "script")

    class SQL:
        async def generate(self, **kwargs):
            raise AssertionError("Saved dashboard cannot generate SQL")

    class Model:
        calls = []

        async def complete_detailed(self, model, prompt):
            self.calls.append("top")
            if route == "refresh":
                decision = (
                    {
                        "action": "CALL_TOOL",
                        "intent": "MODIFY_QUERY",
                        "tool": "ask_data",
                        "tool_arguments": {"objective": "Refresh the data", "operation": "REFINE"},
                    }
                    if len(self.calls) == 1
                    else {
                        "action": "ANSWER",
                        "intent": "DIRECT_ANSWER",
                        "answer": (
                            "Run a new governed analysis in Conversations for different data."
                        ),
                    }
                )
                return json.dumps(decision), TokenUsage()
            return json.dumps(
                {
                    "action": "CALL_TOOL",
                    "intent": "VISUALIZE",
                    "tool": "ai_dashboard_designer",
                    "tool_arguments": {
                        "objective": "Use higher contrast without changing the data"
                    },
                }
            ), TokenUsage()

        async def complete_task(self, model, prompt, task, **kwargs):
            if task != "dashboard":
                return await self.complete_detailed(model, prompt)
            self.calls.append(task)
            return json.dumps(
                {"result_id": "result", "mode": "PATCH", "theme": "contrast"}
            ), TokenUsage()

    def forbidden(*args, **kwargs):
        raise AssertionError("Saved dashboard cannot inspect or execute business SQL")

    monkeypatch.setattr(registry, "inspect", forbidden)
    monkeypatch.setattr(registry, "execute", forbidden)
    model = Model()
    app = create_app(
        TextToSQLAgent(registry, SQL()),
        conversation_completers={"agentrouter": model},
        run_history=RunHistoryStore(path),
    )
    client = local_client(app)
    assert client.get(f"/api/studio/{saved.dashboard_id}").status_code == 200
    assert model.calls == []
    assert client.post("/api/conversations/owner/reset").status_code == 204
    assert client.get(f"/api/studio/{saved.dashboard_id}").status_code == 200
    response = client.post(
        f"/api/studio/{saved.dashboard_id}/edit",
        json={"expected_version": 1, "message": "Use higher contrast"},
    )
    assert response.status_code == 200, response.text
    if route == "design":
        assert response.json()["preview_id"]
        assert model.calls == ["top", "dashboard"]
        assert response.json()["spec"]["theme"] == "contrast"
    else:
        assert response.json()["preview_id"] is None
        assert model.calls == ["top", "top"]
        assert "new governed analysis" in response.json()["message"]
    assert (
        client.get(f"/api/studio/{saved.dashboard_id}").json()["document"]["spec"]["theme"]
        == "green"
    )


def test_edit_version_changed_while_model_was_running(tmp_path):
    path = tmp_path / "history.sqlite"

    async def concurrent_edit(current, objective):
        DashboardStore(path).revise(
            DashboardPrincipal("local-workspace"),
            current.dashboard_id,
            current.version,
            name="Concurrent edit",
        )
        return ChatResponse(
            session_id="studio",
            operation="VISUALIZE",
            resolved_question=objective,
            dashboard=current.document.spec.model_dump(),
            message="Ready",
        )

    app, payload = studio_app(path, edit=concurrent_edit)
    client = local_client(app)
    saved = client.post("/api/studio", json=payload).json()
    response = client.post(
        f"/api/studio/{saved['dashboard_id']}/edit",
        json={"expected_version": 1, "message": "Make it wider"},
    )
    assert response.status_code == 409
    current = client.get(f"/api/studio/{saved['dashboard_id']}").json()
    assert current["version"] == 2 and current["name"] == "Concurrent edit"


def test_private_history_file_cannot_be_exposed_as_a_business_database(registry, monkeypatch):
    monkeypatch.setenv("TEXT2SQL_DATABASE_ROOT", str(registry.root))
    with pytest.raises(ValueError, match="must not be an exposed business database"):
        create_app(run_history=RunHistoryStore(registry.resolve("shop")))


def test_memory_only_history_cannot_claim_a_permanent_save():
    response = local_client(create_app(run_history=RunHistoryStore(None))).get("/api/studio")
    assert response.status_code == 503
    assert "file-backed history" in response.json()["detail"]

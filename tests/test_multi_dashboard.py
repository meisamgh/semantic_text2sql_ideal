from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from test_ai_dashboard_designer import Designer
from test_dashboard import snapshot

from semantic_text2sql.agent import TextToSQLAgent
from semantic_text2sql.ai_dashboard_designer import (
    design_context,
    design_dashboard,
    eligible_catalog,
)
from semantic_text2sql.api import create_app
from semantic_text2sql.dashboard_service import DashboardError, DashboardRequest, DashboardSpec
from semantic_text2sql.dashboard_store import (
    DashboardDocument,
    DashboardPrincipal,
    DashboardSnapshot,
    DashboardStore,
)
from semantic_text2sql.models import SemanticContract, TokenUsage
from semantic_text2sql.multi_dashboard import composition_catalog
from semantic_text2sql.run_history import RunHistoryStore
from semantic_text2sql.runtime import RequestBudget


def sources():
    revenue = snapshot()
    customers = replace(
        revenue,
        result_id="customers",
        columns=["customers"],
        rows=[[12]],
        row_count=1,
        chart_spec=None,
        question="Customer count",
        sql="SELECT COUNT(DISTINCT customer_id) AS customers FROM sales",
        semantic_contract=SemanticContract(output_grain=["all customers"]),
    )
    return {"result": revenue, "customers": customers}


def request():
    return DashboardRequest(session_id="owner", db_id="sales", result_id="customers")


def test_independent_grains_and_widget_lineage_never_merge_rows():
    evidence = sources()
    spec, metrics, charts = composition_catalog(request(), evidence)
    assert spec.schema_version == 2 and set(spec.sources) == set(evidence)
    assert {k.result_id for k in spec.kpis} == set(evidence)
    for widget in charts.values():
        assert widget.chart_spec.source_row_count == evidence[widget.result_id].row_count
    assert spec.sources["customers"].semantic_contract.output_grain == ["all customers"]
    assert len(metrics) >= 2
    payload = spec.model_dump()
    payload["widgets"][0]["result_id"] = "unavailable"
    with pytest.raises(ValidationError):
        DashboardSpec.model_validate(payload)
    with pytest.raises(DashboardError):
        composition_catalog(request(), {**evidence, "bad": replace(snapshot(), truncated=True)})


def test_multi_result_persistence_restart_and_legacy_json(tmp_path):
    evidence = sources()
    spec, _, _ = composition_catalog(request(), evidence)
    doc = DashboardDocument(
        spec=spec,
        source=DashboardSnapshot.capture(evidence["customers"]),
        sources={key: DashboardSnapshot.capture(value) for key, value in evidence.items()},
    )
    path, principal = tmp_path / "history.sqlite", DashboardPrincipal("owner")
    saved = DashboardStore(path).create(principal, "Executive view", doc)
    reopened = DashboardStore(path).open(principal, saved.dashboard_id)
    assert reopened.document == doc
    assert reopened.document.results()["result"].rows == snapshot().rows
    legacy, _, _ = eligible_catalog(
        snapshot(), request().model_copy(update={"result_id": "result"})
    )
    old = DashboardDocument(spec=legacy, source=DashboardSnapshot.capture(snapshot())).model_dump()
    old.pop("sources")
    old["spec"].pop("sources")
    old["spec"].pop("schema_version")
    DashboardDocument.model_validate(old).verify()


def test_designer_profiles_omit_raw_rows_and_identifier_values():
    source = replace(
        snapshot(),
        columns=["customer_id", "email", "revenue"],
        rows=[[987654321, "private@example.test", 100]],
        row_count=1,
        chart_spec=None,
    )
    base, metrics, charts = eligible_catalog(
        source, request().model_copy(update={"result_id": "result"})
    )
    context = json.dumps(design_context(source, base, metrics, charts, None))
    assert "987654321" not in context and "private@example.test" not in context
    assert '"sample": []' in context


def test_one_targeted_correction_is_metered_and_no_third_call():
    evidence = sources()
    _, metrics, _ = composition_catalog(request(), evidence)
    valid = {
        "result_id": "customers",
        "kpis": [
            {"kpi_id": next(k.kpi_id for k in metrics.values() if k.result_id == identity)}
            for identity in evidence
        ],
    }

    class Correcting:
        calls = 0

        async def complete_task(self, model, prompt, task, **kwargs):
            self.calls += 1
            if self.calls == 2:
                assert "Unknown eligible metric" in prompt
            proposal = (
                valid
                if self.calls == 2
                else {"result_id": "customers", "kpis": [{"kpi_id": "invented"}]}
            )
            return json.dumps(proposal), TokenUsage(input_tokens=100, output_tokens=20)

    budget, model = RequestBudget(5, max_database_calls=0), Correcting()
    result = asyncio.run(
        design_dashboard(
            source=evidence["customers"],
            request=request(),
            completer=model,
            provider="test",
            model="script",
            budget=budget,
            sources=evidence,
            repair=True,
        )
    )
    assert result.status == "READY" and model.calls == budget.model_calls == 2
    assert [entry.status for entry in result.ledger] == ["FAILED", "SUCCEEDED"]
    assert result.usage.input_tokens == 200 and budget.database_calls == 0
    invalid = Designer({"result_id": "customers", "kpis": [{"kpi_id": "invented"}]})
    failed = asyncio.run(
        design_dashboard(
            source=evidence["customers"],
            request=request(),
            completer=invalid,
            provider="test",
            model="script",
            budget=RequestBudget(5),
            sources=evidence,
            repair=True,
        )
    )
    assert failed.status == "FALLBACK" and invalid.calls == 2


def test_selected_widget_edit_preserves_every_other_widget():
    evidence = sources()
    previous, _, _ = composition_catalog(request(), evidence)
    previous = previous.model_copy(update={"design_mode": "ai"})
    target = previous.widgets[0].widget_id
    model = Designer(
        {
            "result_id": "customers",
            "mode": "PATCH",
            "charts": [{"candidate_id": target, "column_span": 12}],
        }
    )
    result = asyncio.run(
        design_dashboard(
            source=evidence["customers"],
            request=request(),
            completer=model,
            provider="test",
            model="script",
            budget=RequestBudget(5),
            sources=evidence,
            previous=previous,
            target_widget_id=target,
        )
    )
    assert result.status == "READY" and result.spec
    assert result.spec.kpis == previous.kpis
    assert result.spec.widgets[0].column_span == 12
    assert result.spec.widgets[0].chart_spec.data == previous.widgets[0].chart_spec.data


def test_conversation_three_governed_queries_then_design_save_and_reopen(
    registry, tmp_path, monkeypatch
):
    monkeypatch.setenv("TEXT2SQL_DATABASE_ROOT", str(registry.root))
    monkeypatch.setenv("TEXT2SQL_CONTEXT_MODE", "retrieval")
    monkeypatch.setenv("TEXT2SQL_CONVERSATIONAL_AGENT_ENABLED", "true")
    monkeypatch.setenv("TEXT2SQL_ASSISTANT_PROVIDER", "agentrouter")
    monkeypatch.setenv("TEXT2SQL_ASSISTANT_MODEL", "script")
    monkeypatch.setenv("TEXT2SQL_DASHBOARD_PROVIDER", "agentrouter")
    monkeypatch.setenv("TEXT2SQL_DASHBOARD_MODEL", "script")
    monkeypatch.setenv("TEXT2SQL_ASSISTANT_MAX_MODEL_CALLS", "12")
    goal = "Build a dashboard of revenue, customer count and order status"
    objectives = [
        "Total amount by status",
        "Count distinct customer_id in orders",
        "Count orders by status",
    ]

    class SQL:
        calls = 0

        async def generate(self, **kwargs):
            queries = [
                "SELECT status, SUM(amount) AS revenue FROM orders GROUP BY status",
                "SELECT COUNT(DISTINCT customer_id) AS customers FROM orders",
                "SELECT status, COUNT(*) AS order_count FROM orders GROUP BY status",
            ]
            query = queries[self.calls]
            self.calls += 1
            return query, 1

    class Model:
        top_calls = 0
        design_calls = 0

        async def complete_detailed(self, model, prompt):
            self.top_calls += 1
            context = json.loads(prompt.split("\nContext: ")[1])
            if self.top_calls <= 3:
                decision = {
                    "action": "CALL_TOOL",
                    "intent": "NEW_QUERY",
                    "tool": "ask_data",
                    "finish_after_tool": True,
                    "tool_arguments": {
                        "objective": objectives[self.top_calls - 1],
                        "operation": "NEW_QUERY",
                        "dashboard_goal": goal,
                    },
                }
            else:
                assert len(context["dashboard_evidence"]) == 3
                decision = {
                    "action": "CALL_TOOL",
                    "intent": "VISUALIZE",
                    "tool": "ai_dashboard_designer",
                    "tool_arguments": {
                        "objective": goal,
                        "source_result_ids": [
                            r["result_id"] for r in context["dashboard_evidence"]
                        ],
                    },
                }
            return json.dumps(decision), TokenUsage(input_tokens=10, output_tokens=5)

        async def complete_task(self, model, prompt, task, **kwargs):
            if task != "dashboard":
                return await self.complete_detailed(model, prompt)
            self.design_calls += 1
            context = json.loads(
                prompt.split(
                    "Verified context (profiles/sample are bounded, not the entire result): "
                )[1]
            )
            return json.dumps(
                {
                    "result_id": context["result_id"],
                    "title": "Business performance",
                    "kpis": [
                        {
                            "kpi_id": next(
                                k["kpi_id"]
                                for k in context["kpi_catalog"]
                                if k["result_id"] == source["result_id"]
                            )
                        }
                        for source in context["source_profiles"]
                    ],
                    "charts": [
                        {"candidate_id": c["candidate_id"]} for c in context["chart_catalog"][:2]
                    ],
                }
            ), TokenUsage(input_tokens=100, output_tokens=30)

    history = RunHistoryStore(tmp_path / "history.sqlite")
    model, sql = Model(), SQL()
    app = create_app(
        TextToSQLAgent(registry, sql),
        conversation_completers={"agentrouter": model},
        run_history=history,
    )
    client = TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000))
    response = client.post(
        "/api/chat",
        json={
            "session_id": "owner",
            "db_id": "shop",
            "provider": "agentrouter",
            "model": "script",
            "message": goal,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["dashboard"], body
    assert sql.calls == 3 and model.top_calls == 4 and model.design_calls == 1
    assert body["dashboard"]["schema_version"] == 2 and len(body["dashboard"]["sources"]) == 3
    saved = client.post(
        "/api/studio",
        json={
            "session_id": "owner",
            "db_id": "shop",
            "result_id": body["result_id"],
            "design_revision_id": body["dashboard"]["revision_id"],
            "name": "Business performance",
        },
    )
    assert saved.status_code == 200, saved.text
    assert len(saved.json()["document"]["sources"]) == 3
    identity = saved.json()["dashboard_id"]
    reopened = DashboardStore(history.path).open(DashboardPrincipal("local-workspace"), identity)
    assert len(reopened.document.sources) == 3
    assert sql.calls == 3 and model.design_calls == 1


def test_saved_dashboard_adds_missing_evidence_without_refreshing_existing_results(
    registry, tmp_path, monkeypatch
):
    monkeypatch.setenv("TEXT2SQL_DATABASE_ROOT", str(registry.root))
    monkeypatch.setenv("TEXT2SQL_CONTEXT_MODE", "retrieval")
    for setting in ("ASSISTANT", "DASHBOARD"):
        monkeypatch.setenv(f"TEXT2SQL_{setting}_PROVIDER", "agentrouter")
        monkeypatch.setenv(f"TEXT2SQL_{setting}_MODEL", "script")
    evidence = {key: replace(value, db_id="shop") for key, value in sources().items()}
    req = request().model_copy(update={"db_id": "shop"})
    spec, _, _ = composition_catalog(req, evidence)
    doc = DashboardDocument(
        spec=spec,
        source=DashboardSnapshot.capture(evidence["customers"]),
        sources={key: DashboardSnapshot.capture(value) for key, value in evidence.items()},
    )
    history = RunHistoryStore(tmp_path / "history.sqlite")
    principal = DashboardPrincipal("local-workspace")
    saved = DashboardStore(history.path).create(principal, "Performance", doc)

    class SQL:
        calls = 0

        async def generate(self, **kwargs):
            self.calls += 1
            return "SELECT status, MAX(amount) AS maximum_amount FROM orders GROUP BY status", 1

    class Model:
        calls = []
        tasks = []

        async def complete_detailed(self, model, prompt):
            context = json.loads(prompt.split("\nContext: ")[1])
            self.calls.append("top")
            if not context["dashboard_evidence"]:
                decision = {
                    "action": "CALL_TOOL",
                    "intent": "NEW_QUERY",
                    "tool": "ask_data",
                    "tool_arguments": {
                        "objective": "Maximum amount by status",
                        "operation": "NEW_QUERY",
                        "dashboard_goal": "Add maximum amount to the existing dashboard",
                        "changes": [
                            {
                                "operation": "CHANGE_METRIC",
                                "target": "metric",
                                "new_value": "maximum amount",
                            }
                        ],
                    },
                }
            else:
                decision = {
                    "action": "CALL_TOOL",
                    "intent": "VISUALIZE",
                    "tool": "ai_dashboard_designer",
                    "tool_arguments": {
                        "objective": "Add maximum amount without changing existing widgets",
                        "source_result_ids": [
                            *evidence,
                            context["dashboard_evidence"][0]["result_id"],
                        ],
                    },
                }
            return json.dumps(decision), TokenUsage()

        async def complete_task(self, model, prompt, task, **kwargs):
            self.tasks.append((task, prompt[:120]))
            if task != "dashboard":
                return await self.complete_detailed(model, prompt)
            self.calls.append("design")
            context = json.loads(
                prompt.split(
                    "Verified context (profiles/sample are bounded, not the entire result): "
                )[1]
            )
            candidate = next(c for c in context["chart_catalog"] if c["result_id"] not in evidence)
            return json.dumps(
                {
                    "result_id": "customers",
                    "mode": "PATCH",
                    "charts": [{"candidate_id": candidate["candidate_id"]}],
                }
            ), TokenUsage()

    sql, model = SQL(), Model()
    client = TestClient(
        create_app(
            TextToSQLAgent(registry, sql),
            conversation_completers={"agentrouter": model},
            run_history=history,
        ),
        base_url="http://127.0.0.1",
        client=("127.0.0.1", 50000),
    )
    response = client.post(
        f"/api/studio/{saved.dashboard_id}/edit",
        json={"expected_version": 1, "message": "Add maximum amount by status"},
    )
    assert response.status_code == 200, response.text
    preview = response.json()
    assert preview["preview_id"], (preview, model.calls, model.tasks, sql.calls)
    assert len(preview["spec"]["sources"]) == 3
    assert preview["spec"]["widgets"][: len(spec.widgets)] == [
        w.model_dump(mode="json") for w in spec.widgets
    ]
    assert sql.calls == 1 and model.calls == ["top", "top", "design"]
    committed = client.post(
        f"/api/studio/{saved.dashboard_id}/save",
        json={"expected_version": 1, "preview_id": preview["preview_id"]},
    )
    assert committed.status_code == 200, committed.text
    restored = DashboardStore(history.path).open(principal, saved.dashboard_id)
    assert restored.document.source == doc.source
    for identity, retained_snapshot in doc.results().items():
        assert restored.document.results()[identity] == retained_snapshot
    assert len(restored.document.results()) == 3 and sql.calls == 1


def test_ordinary_query_does_not_trigger_dashboard(registry, tmp_path):
    from test_conversational_agent import Completer, _agent

    from semantic_text2sql.models import ChatRequest

    calls = []
    completer = Completer(
        [
            {
                "action": "CALL_TOOL",
                "intent": "NEW_QUERY",
                "tool": "ask_data",
                "tool_arguments": {"objective": "Count orders", "operation": "NEW_QUERY"},
                "finish_after_tool": True,
            }
        ]
    )
    agent = _agent(registry, tmp_path, completer, calls)
    response = asyncio.run(
        agent.run(ChatRequest(session_id="ordinary", db_id="shop", message="Count orders"))
    )
    assert response.dashboard is None and len(calls) == 1 and completer.calls == 1


def test_widget_replacement_preserves_identity_and_unaffected_metrics():
    evidence = sources()
    previous, metrics, _ = composition_catalog(request(), evidence)
    previous = previous.model_copy(update={"design_mode": "ai"})
    old = next(k for k in previous.kpis if k.result_id == "result")
    new = next(k for k in metrics.values() if k.result_id == "customers")
    proposal = {
        "result_id": "customers",
        "mode": "PATCH",
        "kpis": [{"kpi_id": new.kpi_id, "replaces_widget_id": old.kpi_id}],
    }
    result = asyncio.run(
        design_dashboard(
            source=evidence["customers"],
            request=request(),
            completer=Designer(proposal),
            provider="test",
            model="script",
            budget=RequestBudget(5),
            sources=evidence,
            previous=previous,
            target_widget_id=old.kpi_id,
        )
    )
    assert result.status == "READY" and result.spec
    replaced = next(k for k in result.spec.kpis if k.kpi_id == old.kpi_id)
    assert replaced.result_id == "customers" and replaced.value == 12
    assert result.spec.widgets == previous.widgets
    assert [k for k in result.spec.kpis if k.kpi_id != old.kpi_id] == [
        k for k in previous.kpis if k.kpi_id != old.kpi_id
    ]
    invalid = {"result_id": "customers", "mode": "PATCH", "theme": "neutral"}
    bad = asyncio.run(
        design_dashboard(
            source=evidence["customers"],
            request=request(),
            completer=Designer(invalid),
            provider="test",
            model="script",
            budget=RequestBudget(5),
            sources=evidence,
            previous=previous,
            target_widget_id=old.kpi_id,
        )
    )
    assert bad.status == "FALLBACK" and bad.spec.kpis == previous.kpis


def test_duplicate_replacement_targets_are_rejected_without_overwriting_widgets():
    evidence = sources()
    previous, metrics, _ = composition_catalog(request(), evidence)
    old = previous.kpis[0]
    candidates = list(metrics)[:2]
    assert len(candidates) == 2
    proposal = {
        "result_id": "result",
        "mode": "PATCH",
        "kpis": [{"kpi_id": key, "replaces_widget_id": old.kpi_id} for key in candidates],
    }
    result = asyncio.run(
        design_dashboard(
            source=evidence["result"],
            request=request(),
            completer=Designer(proposal),
            provider="test",
            model="script",
            budget=RequestBudget(5),
            sources=evidence,
            previous=previous,
        )
    )
    assert result.status == "FALLBACK"
    assert result.spec.kpis == previous.kpis


def test_source_limits_and_database_isolation_reject_before_design():
    evidence = sources()
    mixed = {**evidence, "foreign": replace(snapshot(), result_id="foreign", db_id="another-db")}
    with pytest.raises(DashboardError):
        composition_catalog(request(), mixed)
    excessive = {**evidence, **{str(i): replace(snapshot(), result_id=str(i)) for i in range(3)}}
    with pytest.raises(DashboardError):
        composition_catalog(request(), excessive)
    empty = {**evidence, "result": replace(snapshot(), rows=[], row_count=0, chart_spec=None)}
    with pytest.raises(DashboardError, match="no usable dashboard data"):
        composition_catalog(request(), empty)


def test_failed_required_analysis_cannot_be_hidden_by_a_later_dashboard(registry, tmp_path):
    from test_conversational_agent import Completer, _agent

    from semantic_text2sql.models import ChatRequest, ChatResponse

    decisions = Completer(
        [
            {
                "action": "CALL_TOOL",
                "intent": "NEW_QUERY",
                "tool": "ask_data",
                "tool_arguments": {
                    "objective": "Retention",
                    "operation": "NEW_QUERY",
                    "dashboard_goal": "Revenue and retention",
                },
            },
            {
                "action": "CALL_TOOL",
                "intent": "VISUALIZE",
                "tool": "ai_dashboard_designer",
                "tool_arguments": {"objective": "Revenue and retention"},
            },
        ]
    )
    agent = _agent(registry, tmp_path, decisions, [])
    retained = replace(snapshot(), db_id="shop")
    agent.get_last_response = lambda *_: retained

    async def failed(request):
        return ChatResponse(
            session_id=request.session_id,
            operation="NEW_QUERY",
            resolved_question="Retention",
            message="Retention definition is missing.",
        )

    async def incomplete(request, objective, current, undo):
        base, _, _ = eligible_catalog(
            retained, DashboardRequest(session_id="owner", db_id="shop", result_id="result")
        )
        return ChatResponse(
            session_id=request.session_id,
            operation="VISUALIZE",
            result_id="result",
            resolved_question=objective,
            message="Ready",
            generation=retained.generation,
            dashboard=base.model_dump(),
        )

    agent.sql_handler, agent.dashboard_handler = failed, incomplete
    response = asyncio.run(
        agent.run(
            ChatRequest(
                session_id="owner", db_id="shop", message="Build revenue and retention dashboard"
            )
        )
    )
    assert response.dashboard is None and "incomplete" in response.message

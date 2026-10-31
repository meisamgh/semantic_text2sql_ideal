"""Bounded dashboard specialist: verified choices, optional targeted repair, never SQL."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from statistics import median
from time import perf_counter
from typing import Any, Literal

from pydantic import Field, ValidationError

from semantic_text2sql.dashboard_service import (
    DashboardError,
    DashboardRequest,
    DashboardSpec,
    DashboardWidget,
    KPISpec,
    build_dashboard,
    eligible_kpis,
)
from semantic_text2sql.models import CallLedgerEntry, StrictModel, TokenUsage
from semantic_text2sql.result_cache import RecentResult
from semantic_text2sql.runtime import RequestBudget, model_call_cap, use_request_budget
from semantic_text2sql.visualization_recommender import profile_chart_result


class KPIChoice(StrictModel):
    kpi_id: str = Field(max_length=100)
    order: int | None = Field(default=None, ge=0, le=3)
    column_span: Literal[3, 4, 6, 12] | None = None
    replaces_widget_id: str | None = Field(default=None, max_length=100)


class ChartChoice(StrictModel):
    candidate_id: str = Field(max_length=100)
    order: int | None = Field(default=None, ge=0, le=3)
    column_span: Literal[4, 6, 8, 12] | None = None
    title: str | None = Field(default=None, max_length=160)
    x_label: str | None = Field(default=None, max_length=100)
    y_label: str | None = Field(default=None, max_length=100)
    x: str | None = None
    y: str | None = None
    series: str | None = None
    replaces_widget_id: str | None = Field(default=None, max_length=100)


class DashboardProposal(StrictModel):
    result_id: str
    mode: Literal["REPLACE", "PATCH", "REQUERY_REQUIRED"] = "REPLACE"
    title: str = Field(default="", max_length=160)
    description: str = Field(default="", max_length=600)
    kpis: list[KPIChoice] = Field(default_factory=list, max_length=4)
    charts: list[ChartChoice] = Field(default_factory=list, max_length=4)
    remove_ids: list[str] = Field(default_factory=list, max_length=8)
    filter_refs: list[str] | None = Field(default=None, max_length=24)
    theme: Literal["green", "neutral", "contrast"] | None = None
    rationale: str = Field(default="", max_length=600)
    missing_requirements: str = Field(default="", max_length=1000)


@dataclass(frozen=True)
class DashboardDesignOutcome:
    status: Literal["READY", "FALLBACK", "REQUERY_REQUIRED"]
    spec: DashboardSpec | None
    reason: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    ledger: tuple[CallLedgerEntry, ...] = ()


def _id(prefix: str, value: str) -> str:
    return prefix + hashlib.sha256(value.encode()).hexdigest()[:16]


def eligible_catalog(
    source: RecentResult, request: DashboardRequest
) -> tuple[DashboardSpec, dict[str, KPISpec], dict[str, DashboardWidget]]:
    charts: list[DashboardWidget] = []
    base = build_dashboard(request, lambda *_: source, chart_catalog=charts)
    temporal = next(
        (w.chart_spec for w in charts if w.chart_spec.time_format in {"year", "month", "day"}), None
    )
    kpis = {}
    for kpi in eligible_kpis(source, temporal, limit=24):
        key = _id("kpi-", json.dumps([source.result_id, kpi.metric, kpi.calculation, kpi.periods]))
        kpis[key] = kpi.model_copy(update={"kpi_id": key})
    catalog = {}
    for widget in charts:
        key = _id(
            "chart-", json.dumps(widget.chart_spec.model_dump(exclude={"data"}), sort_keys=True)
        )
        catalog[key] = widget.model_copy(update={"widget_id": key})
    return base, kpis, catalog


def design_context(
    source: RecentResult,
    base: DashboardSpec,
    kpis: dict[str, KPISpec],
    charts: dict[str, DashboardWidget],
    previous: DashboardSpec | None,
) -> dict[str, Any]:
    profiles = []
    for index, column in enumerate(source.columns[:24]):
        values = [row[index] for row in source.rows]
        numeric = [v for v in values if type(v) in {int, float}]
        nonnull = [v for v in values if v is not None]
        item: dict[str, Any] = {
            "column": column,
            "nulls": len(values) - len(nonnull),
            "cardinality": len({repr(v) for v in nonnull}),
        }
        identifier = bool(re.search(r"(?:id$|_id$|email|phone|name|token|password)", column, re.I))
        if numeric and len(numeric) == len(nonnull) and not identifier:
            item.update(
                type="numeric", minimum=min(numeric), maximum=max(numeric), median=median(numeric)
            )
        else:
            item.update(
                type="numeric" if numeric and len(numeric) == len(nonnull) else "text",
                values_redacted=True,
                distribution_counts=sorted(
                    Counter(repr(v) for v in nonnull).values(), reverse=True
                )[:5],
            )
        profiles.append(item)
    contract = source.semantic_contract
    return {
        "result_id": source.result_id,
        "run_id": source.run_id,
        "execution_timestamp": source.execution_timestamp,
        "complete": True,
        "rows_count": source.row_count,
        "column_profiles": profiles,
        "roles": profile_chart_result(
            source.columns[:24], [row[:24] for row in source.rows], sample_rows=0
        )[0],
        "omitted_profile_columns": max(0, len(source.columns) - 24),
        "grain": contract.output_grain or contract.grain if contract else [],
        "constraints": "Retained independently per source; no literal values disclosed.",
        "approved_metrics": [m.model_dump() for m in contract.approved_metrics[:8]]
        if contract
        else [],
        "kpi_catalog": [
            k.model_dump(
                include={
                    "kpi_id",
                    "metric",
                    "result_id",
                    "title",
                    "calculation",
                    "unit",
                    "source_grain",
                }
            )
            for k in kpis.values()
        ],
        "chart_catalog": [
            {
                "candidate_id": key,
                "result_id": w.result_id,
                **w.chart_spec.model_dump(
                    include={
                        "chart_type",
                        "x",
                        "y",
                        "series",
                        "x_components",
                        "y_measures",
                        "title",
                        "measure_units",
                        "time_format",
                    }
                ),
            }
            for key, w in charts.items()
        ],
        "filter_catalog": [
            {"ref": f"filter-{i}", **f.model_dump()}
            for i, f in enumerate(base.verified_filter_metadata)
        ],
        "provenance": source.provenance[:4],
        "sample": [],
        "previous": (
            {
                "revision_id": previous.revision_id,
                "title": previous.title,
                "description": previous.description,
                "theme": previous.theme,
                "kpis": [{"kpi_id": k.kpi_id, "column_span": k.column_span} for k in previous.kpis],
                "charts": [
                    {
                        "candidate_id": w.widget_id,
                        "column_span": w.column_span,
                        **w.chart_spec.model_dump(exclude={"data", "source_columns"}),
                    }
                    for w in previous.widgets
                ],
            }
            if previous
            else None
        ),
    }


_PROMPT = """You are the AI dashboard design specialist, not a conversational router or SQL agent.
Design for the client's objective using ONLY verified KPI IDs and validated chart candidate IDs.
Independently select the most useful eligible KPIs, primary chart and complementary supporting
charts; do not repeat a fixed template. Choose order, widths, readable titles/labels and theme.
Sections are always read-only global filters, KPIs, charts, optional source data in that order.
Use REPLACE for initial design; PATCH for revisions, listing only changed/added entries and
explicit remove_ids. Preserve all unmentioned widgets, encodings, filters, definitions and data.
To replace a selected widget with another eligible metric/chart, use replaces_widget_id;
the existing widget identity/order is retained while its validated source may change.
The first chart is the primary visualization. Select at most four KPIs and four charts.
Axes must exactly match the selected candidate; never invent an alias, metric, formula or data.
Do not add HTML, JavaScript, Python, executable options, numerical findings or causal claims.
Descriptions/rationale explain design decisions only. Titles/labels must preserve metric meaning.
Applied SQL constraints always remain visible. filter_refs select optional verified metadata only;
global filter editing/recalculation is NOT enabled. Approved themes: green, neutral, contrast.
If the requested data/metric/breakdown is not in the catalogs, return REQUERY_REQUIRED and precise
missing_requirements. Never substitute an unrelated KPI or silently request different data.
Only the existing Top-layer assistant can decide to invoke governed Text-to-SQL.
If selected_widget_id is present, return PATCH affecting only that exact widget;
preserve other widgets and all global settings. repair_feedback is a deterministic
validation error: correct that issue without inventing new data or changing scope.
Return ONLY JSON matching this schema: {schema}
Client objective: {objective}
Verified context (profiles/sample are bounded, not the entire result): {context}
"""


def _materialize(
    proposal: DashboardProposal,
    base: DashboardSpec,
    kpis: dict[str, KPISpec],
    charts: dict[str, DashboardWidget],
    previous: DashboardSpec | None,
) -> DashboardSpec:
    if proposal.result_id != base.result_id:
        raise ValueError("Dashboard result identity mismatch")
    if proposal.mode == "PATCH" and previous is None:
        raise ValueError("A patch requires a previous validated dashboard")
    if previous and previous.design_mode != "deterministic" and proposal.mode != "PATCH":
        raise ValueError("Dashboard revisions must explicitly preserve unaffected widgets")
    if proposal.mode == "PATCH" and (
        previous is None
        or previous.result_id != base.result_id
        and previous.result_id not in base.sources
    ):
        raise ValueError("Revision requires the same accepted source result")
    annotations = [
        proposal.title,
        proposal.description,
        proposal.rationale,
        *(choice.title or "" for choice in proposal.charts),
        *(choice.x_label or "" for choice in proposal.charts),
        *(choice.y_label or "" for choice in proposal.charts),
    ]
    if any(
        re.search(r"\d|<[^>]+>|\b(?:increased|decreased|grew|fell)\b", text, re.I)
        for text in annotations
    ):
        raise ValueError("Design annotations must not assert numerical facts or contain markup")
    patch = proposal.mode == "PATCH" and previous is not None
    selected_kpis = {k.kpi_id: k for k in previous.kpis} if patch and previous else {}
    selected_charts = {w.widget_id: w for w in previous.widgets} if patch and previous else {}
    if len(set(proposal.remove_ids)) != len(proposal.remove_ids):
        raise ValueError("Duplicate removal")
    for key in proposal.remove_ids:
        if not patch or key not in {*selected_kpis, *selected_charts}:
            raise ValueError("Unknown widget removal")
        selected_kpis.pop(key, None)
        selected_charts.pop(key, None)
    for ids, catalog_keys in [
        ([c.kpi_id for c in proposal.kpis], set(kpis)),
        ([c.candidate_id for c in proposal.charts], set(charts)),
    ]:
        if len(ids) != len(set(ids)) or set(ids).intersection(proposal.remove_ids):
            raise ValueError("Duplicate or contradictory selection")
        if any(key not in catalog_keys for key in ids):
            raise ValueError("Unknown eligible metric or chart candidate")
    for choices, key_name in ((proposal.kpis, "kpi_id"), (proposal.charts, "candidate_id")):
        targets = [c.replaces_widget_id or getattr(c, key_name) for c in choices]
        if len(targets) != len(set(targets)) or set(targets).intersection(proposal.remove_ids):
            raise ValueError("Duplicate or contradictory widget target")
    for choice in proposal.kpis:
        target = choice.replaces_widget_id or choice.kpi_id
        if choice.replaces_widget_id and (not patch or target not in selected_kpis):
            raise ValueError("Only an existing KPI widget can be replaced")
        old = selected_kpis.get(target)
        selected_kpis[target] = kpis[choice.kpi_id].model_copy(
            update={
                "kpi_id": target,
                "column_span": choice.column_span or (old.column_span if old else 3),
            }
        )
    for chart_choice in proposal.charts:
        candidate = charts[chart_choice.candidate_id]
        for field_name in ("x", "y", "series"):
            if field_name in chart_choice.model_fields_set and getattr(
                chart_choice, field_name
            ) != getattr(candidate.chart_spec, field_name):
                raise ValueError("Axes must match the validated candidate")
        target = chart_choice.replaces_widget_id or chart_choice.candidate_id
        if chart_choice.replaces_widget_id and (not patch or target not in selected_charts):
            raise ValueError("Only an existing chart widget can be replaced")
        old_chart = selected_charts.get(target)
        spec = (
            old_chart.chart_spec
            if old_chart and not chart_choice.replaces_widget_id
            else candidate.chart_spec
        )
        labels = {
            name: getattr(chart_choice, name)
            for name in ("title", "x_label", "y_label")
            if getattr(chart_choice, name) is not None
        }
        for label_name, column in (("x_label", spec.x), ("y_label", spec.y)):
            if label_name in labels and column:

                def words(text: str) -> set[str]:
                    return set(re.findall(r"[a-z]+", text.lower()))

                if not words(labels[label_name]).issubset(words(column) | {"axis", "value"}):
                    raise ValueError("An axis label cannot rename the verified measure")
        selected_charts[target] = candidate.model_copy(
            update={
                "widget_id": target,
                "column_span": chart_choice.column_span
                or (old_chart.column_span if old_chart else 6),
                "chart_spec": spec.model_copy(update=labels),
            }
        )

    def ordered(items: dict[str, Any], choices: Any, key_name: str) -> list[Any]:
        order = {
            c.replaces_widget_id or getattr(c, key_name): c.order
            for c in choices
            if c.order is not None
        }
        original = {key: index for index, key in enumerate(items)}
        return [
            items[key]
            for key in sorted(
                items,
                key=lambda key: (
                    order.get(key, original[key]),
                    0 if key in order else 1,
                    original[key],
                ),
            )
        ]

    kpi_list = ordered(selected_kpis, proposal.kpis, "kpi_id")
    chart_list = ordered(selected_charts, proposal.charts, "candidate_id")
    if len(kpi_list) > 4 or len(chart_list) > 4:
        raise ValueError("Dashboard exceeds widget limits")
    if not patch and not (kpi_list or chart_list) and (kpis or charts):
        raise ValueError("Empty selection despite eligible evidence")
    existing_sources = (
        (set(previous.sources) or {previous.result_id}) if patch and previous else set()
    )
    required_sources = set(base.sources) - existing_sources
    represented = {item.result_id for item in [*kpi_list, *chart_list]}
    if required_sources - represented:
        raise ValueError("Each newly requested analysis needs a corresponding KPI or chart")
    old_rows = {w.widget_id: w.row for w in previous.widgets} if patch and previous else {}
    reordered = any(choice.order is not None for choice in proposal.charts)
    for i, widget in enumerate(chart_list):
        chart_list[i] = widget.model_copy(
            update={"row": old_rows.get(widget.widget_id, i) if not reordered else i}
        )
    filters = base.verified_filter_metadata
    if proposal.filter_refs is not None:
        allowed = {f"filter-{i}": item for i, item in enumerate(filters)}
        if len(set(proposal.filter_refs)) != len(proposal.filter_refs) or any(
            key not in allowed for key in proposal.filter_refs
        ):
            raise ValueError("Unknown verified filter field")
        filters = [allowed[key] for key in proposal.filter_refs]
    elif patch and previous:
        filters = previous.verified_filter_metadata
    materialized = base.model_copy(
        update={
            "title": proposal.title or (previous.title if patch and previous else base.title),
            "description": proposal.description
            or (previous.description if patch and previous else ""),
            "rationale": proposal.rationale,
            "theme": proposal.theme or (previous.theme if patch and previous else "green"),
            "kpis": kpi_list,
            "widgets": chart_list,
            "verified_filter_metadata": filters,
            "design_mode": "ai",
            "data": (
                previous.data
                if patch and previous and previous.result_id == base.result_id
                else base.data
            ),
            "parent_revision_id": previous.revision_id if previous else None,
        }
    )
    return DashboardSpec.model_validate(materialized.model_dump())


async def design_dashboard(
    *,
    source: RecentResult,
    request: DashboardRequest,
    completer: Any,
    provider: str,
    model: str,
    budget: RequestBudget,
    previous: DashboardSpec | None = None,
    patch_only: bool = False,
    sources: dict[str, RecentResult] | None = None,
    repair: bool = False,
    repair_feedback: str = "",
    target_widget_id: str | None = None,
) -> DashboardDesignOutcome:
    if sources and len(sources) > 1:
        from semantic_text2sql.multi_dashboard import composition_catalog

        base, kpis, charts = composition_catalog(request, sources)
    else:
        base, kpis, charts = eligible_catalog(source, request)
    if (
        previous
        and previous.result_id != source.result_id
        and (not sources or previous.result_id not in sources)
    ):
        raise DashboardError("A revision cannot change the source result.")
    if previous and previous.design_mode == "deterministic":
        # Phase 1 IDs predate the designer catalog. Resolve by verified identity,
        # never by a model's suggested metric name.
        previous = previous.model_copy(
            update={
                "kpis": [
                    k.model_copy(update={"kpi_id": key})
                    for k in previous.kpis
                    for key, candidate in kpis.items()
                    if k.result_id == candidate.result_id
                    and (k.metric, k.calculation, k.periods)
                    == (candidate.metric, candidate.calculation, candidate.periods)
                ],
                "widgets": previous.widgets,
            }
        )
    if previous:
        for metric in previous.kpis:
            if metric.kpi_id and any(
                metric.model_dump(exclude={"kpi_id", "column_span"})
                == candidate.model_dump(exclude={"kpi_id", "column_span"})
                for candidate in kpis.values()
            ):
                kpis[metric.kpi_id] = metric
        for widget in previous.widgets:
            if any(
                widget.result_id == candidate.result_id
                and widget.chart_spec.model_dump(exclude={"title", "x_label", "y_label"})
                == candidate.chart_spec.model_dump(exclude={"title", "x_label", "y_label"})
                for candidate in charts.values()
            ):
                charts[widget.widget_id] = widget
    context = design_context(source, base, kpis, charts, previous)
    context["selected_widget_id"] = target_widget_id
    context["repair_feedback"] = repair_feedback
    if sources:
        context["source_profiles"] = [
            {
                "result_id": identity,
                "db_id": item.db_id,
                "row_count": item.row_count,
                "roles": profile_chart_result(
                    item.columns[:24], [row[:24] for row in item.rows], sample_rows=0
                )[0],
                "grain": item.semantic_contract.output_grain if item.semantic_contract else [],
                "approved_metrics": [
                    metric.model_dump(exclude_none=True)
                    for metric in item.semantic_contract.approved_metrics[:4]
                ]
                if item.semantic_contract
                else [],
                "active_period": item.semantic_contract.active_period
                if item.semantic_contract
                else None,
                "execution_timestamp": item.execution_timestamp,
            }
            for identity, item in sources.items()
        ]
    prompt = _PROMPT.format(
        schema=json.dumps(DashboardProposal.model_json_schema()),
        objective=request.objective,
        context=json.dumps(context, default=str),
    )
    usage = TokenUsage()
    started = perf_counter()
    before = budget.model_calls
    failed = False
    repairable_error = ""
    try:
        if len(prompt) > 32000:
            raise ValueError("Design context exceeds the bounded prompt limit")
        if completer is None:
            raise ValueError("Dashboard model provider is unavailable")
        with use_request_budget(budget), model_call_cap(budget, 1):
            if callable(getattr(completer, "complete_task", None)):
                operation = completer.complete_task(model, prompt, "dashboard", json_mode=True)
            else:
                operation = completer.complete_detailed(model=model, prompt=prompt)
            raw, usage = await asyncio.wait_for(
                budget.wait(operation, kind="model"), timeout=min(12.0, budget.remaining())
            )
        proposal = DashboardProposal.model_validate_json(raw)
        if proposal.result_id != source.result_id:
            raise ValueError("Wrong result ID")
        if proposal.mode == "REQUERY_REQUIRED":
            if not proposal.missing_requirements.strip():
                raise ValueError("Requery requires explicit missing requirements")
            status, spec, reason = "REQUERY_REQUIRED", None, proposal.missing_requirements
        else:
            if patch_only and previous is not None and proposal.mode != "PATCH":
                raise ValueError("Saved dashboard edits must preserve unaffected widgets")
            if target_widget_id:
                known = (
                    {k.kpi_id for k in previous.kpis} | {w.widget_id for w in previous.widgets}
                    if previous
                    else set()
                )
                changed = (
                    {k.replaces_widget_id or k.kpi_id for k in proposal.kpis}
                    | {w.replaces_widget_id or w.candidate_id for w in proposal.charts}
                    | set(proposal.remove_ids)
                )
                if (
                    target_widget_id not in known
                    or changed - {target_widget_id}
                    or proposal.mode != "PATCH"
                    or proposal.title
                    or proposal.description
                    or proposal.filter_refs is not None
                    or proposal.theme is not None
                ):
                    raise ValueError("Edit only the selected widget; preserve every other setting")
            spec = _materialize(proposal, base, kpis, charts, previous)
            status, reason = "READY", "Validated AI dashboard design."
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        failed = True
        if isinstance(exc, ValidationError):
            repairable_error = json.dumps(exc.errors(include_input=False, include_url=False))[:600]
        elif isinstance(exc, ValueError):
            repairable_error = str(exc)[:600]
        usage = getattr(exc, "usage", usage)
        # Invalid revisions must preserve the exact previous design, not silently
        # replace its metrics. Initial failures use the existing Phase 1 builder.
        spec = (previous or base).model_copy(
            update={
                "design_mode": "fallback",
                "notices": [
                    *(previous or base).notices,
                    "AI design unavailable or invalid; retained the last validated design.",
                ],
            }
        )
        status, reason = "FALLBACK", "A validated fallback was used; no analytical data changed."
    ledger = (
        (
            CallLedgerEntry(
                component="dashboard_designer",
                kind="MODEL",
                status="FAILED" if failed else "SUCCEEDED",
                provider=provider,
                requested_model=model,
                effective_model=model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                latency_ms=int((perf_counter() - started) * 1000),
            ),
        )
        if budget.model_calls > before
        else ()
    )
    if failed and repair and repairable_error and ledger and budget.remaining() > 0:
        corrected = await design_dashboard(
            source=source,
            request=request,
            completer=completer,
            provider=provider,
            model=model,
            budget=budget,
            previous=previous,
            patch_only=patch_only,
            sources=sources,
            repair=False,
            repair_feedback=repairable_error,
            target_widget_id=target_widget_id,
        )
        combined = {
            key: (
                None
                if getattr(usage, key) is None and getattr(corrected.usage, key) is None
                else (getattr(usage, key) or 0) + (getattr(corrected.usage, key) or 0)
            )
            for key in TokenUsage.model_fields
        }
        return DashboardDesignOutcome(
            corrected.status,
            corrected.spec,
            corrected.reason,
            TokenUsage.model_validate(combined),
            (*ledger, *corrected.ledger),
        )
    return DashboardDesignOutcome(status, spec, reason, usage, ledger)  # type: ignore[arg-type]


class DashboardRevisions:
    """Bounded temporary revisions. Not another result store or conversation agent."""

    def __init__(self) -> None:
        self._items: dict[tuple[str, str, str], list[DashboardSpec]] = {}

    def get(self, session: str, db: str, result: str) -> DashboardSpec | None:
        revisions = self._items.get((session, db, result), [])
        return revisions[-1] if revisions else None

    def commit(self, session: str, spec: DashboardSpec, expected: str | None) -> DashboardSpec:
        key = (session, spec.db_id, spec.result_id)
        current = self.get(*key)
        if (current.revision_id if current else None) != expected:
            raise DashboardError("Dashboard changed concurrently; reopen before editing.", 409)
        payload = spec.model_dump(exclude={"revision_id"})
        revision = spec.model_copy(
            update={
                "revision_id": _id("revision-", json.dumps(payload, sort_keys=True, default=str)),
                "parent_revision_id": expected,
            }
        )
        self._items.setdefault(key, []).append(revision)
        self._items[key] = self._items[key][-10:]
        if len(self._items) > 100:
            self._items.pop(next(iter(self._items)))
        return revision

    def undo(self, session: str, db: str, result: str) -> DashboardSpec:
        history = self._items.get((session, db, result), [])
        if len(history) < 2:
            raise DashboardError("No earlier dashboard revision is available.")
        history.pop()
        return history[-1]

    def clear(self, session: str) -> None:
        for key in list(self._items):
            if key[0] == session:
                self._items.pop(key)

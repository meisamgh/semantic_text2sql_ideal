"""Compose independently verified result catalogs; never join or aggregate sources."""

from __future__ import annotations

from semantic_text2sql.ai_dashboard_designer import _id, eligible_catalog
from semantic_text2sql.dashboard_service import (
    DashboardError,
    DashboardRequest,
    DashboardSourceRef,
    DashboardSpec,
    DashboardWidget,
    KPISpec,
)
from semantic_text2sql.result_cache import RecentResult


def composition_catalog(
    request: DashboardRequest, sources: dict[str, RecentResult]
) -> tuple[DashboardSpec, dict[str, KPISpec], dict[str, DashboardWidget]]:
    if not 1 <= len(sources) <= 4 or request.result_id not in sources:
        raise DashboardError("A dashboard supports one to four complete analytical results.")
    refs, metrics, charts, parts = {}, {}, {}, {}
    for identity, source in sources.items():
        if identity != source.result_id or source.db_id != request.db_id:
            raise DashboardError("Dashboard results must belong to the authorized database.")
        part, eligible_metrics, eligible_charts = eligible_catalog(
            source, request.model_copy(update={"result_id": identity})
        )
        if not eligible_metrics and not eligible_charts:
            raise DashboardError(
                "One analysis has no usable dashboard data. Adjust its scope or metric."
            )
        parts[identity] = part
        refs[identity] = DashboardSourceRef(
            result_id=identity,
            db_id=source.db_id,
            run_id=source.run_id,
            question=source.question,
            row_count=source.row_count,
            execution_timestamp=source.execution_timestamp,
            semantic_contract=source.semantic_contract,
            applied_constraints=part.applied_constraints,
            source_sql=source.sql,
        )
        metrics.update(eligible_metrics)
        for key, widget in eligible_charts.items():
            widget_id = _id("chart-", identity + ":" + key)
            charts[widget_id] = widget.model_copy(update={"widget_id": widget_id})
    # Safe fallback: one eligible KPI/chart per independent source, never a
    # sum/average across incompatible scopes. The model may select any catalog IDs.
    selected_metrics: list[KPISpec] = []
    selected_charts: list[DashboardWidget] = []
    for identity in sources:
        metric = next((m for m in metrics.values() if m.result_id == identity), None)
        chart = next((w for w in charts.values() if w.result_id == identity), None)
        if metric:
            selected_metrics.append(metric)
        if chart:
            selected_charts.append(chart.model_copy(update={"row": len(selected_charts)}))
    base = parts[request.result_id].model_copy(
        update={
            "schema_version": 2,
            "sources": refs,
            "title": "Business dashboard",
            "kpis": selected_metrics,
            "widgets": selected_charts,
            "verified_filter_metadata": [
                field.model_copy(update={"result_id": identity})
                for identity, part in parts.items()
                for field in part.verified_filter_metadata
            ],
        }
    )
    return DashboardSpec.model_validate(base.model_dump()), metrics, charts

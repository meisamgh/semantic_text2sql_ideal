"""Single-result dashboards. No database connection or model capability."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from typing import Any, Literal

from pydantic import Field, model_validator
from sqlglot import exp, parse_one

from semantic_text2sql.models import ChartSpec, StrictModel
from semantic_text2sql.result_cache import RecentResult
from semantic_text2sql.visualization_recommender import (
    LLMChartProposal,
    materialize_recommendation,
    recommend_charts,
    validate_chart_proposal,
)


class DashboardRequest(StrictModel):
    session_id: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_-]+$")
    db_id: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_-]+$")
    result_id: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_-]+$")
    include_data: bool = False
    design_mode: Literal["deterministic", "ai"] = "deterministic"
    objective: str = Field(
        default="Design a professional dashboard from this result.", max_length=4000
    )


class KPISpec(StrictModel):
    kpi_id: str = ""
    column_span: Literal[3, 4, 6, 12] = 3
    result_id: str
    metric: str
    title: str
    calculation: Literal["DIRECT", "TOTAL", "COUNT", "AVERAGE", "PERIOD_GROWTH"]
    value: int | float = Field(allow_inf_nan=False)
    unit: str | None = None
    definition: str
    source_grain: list[str] = Field(default_factory=list)
    baseline: float | None = None
    current: float | None = None
    periods: list[str] = Field(default_factory=list, max_length=2)


class DashboardWidget(StrictModel):
    widget_id: str
    result_id: str
    column_span: Literal[4, 6, 8, 12] = 6
    row: int = Field(ge=0, le=3)
    chart_spec: ChartSpec


class DashboardData(StrictModel):
    columns: list[str]
    rows: list[list[Any]] = Field(max_length=500)


class DashboardFilterMetadata(StrictModel):
    field: str
    table: str | None = None
    column: str | None = None
    source: Literal["stored_result", "verified_schema"]


class DashboardSpec(StrictModel):
    dashboard_id: str
    result_id: str
    run_id: str | None
    db_id: str
    title: str
    row_count: int = Field(ge=0, le=500)
    complete: Literal[True] = True
    execution_timestamp: float | None = None
    section_order: tuple[
        Literal["filters"], Literal["kpis"], Literal["charts"], Literal["data"]
    ] = (
        "filters",
        "kpis",
        "charts",
        "data",
    )
    applied_constraints: list[str] = Field(default_factory=list)
    verified_filter_metadata: list[DashboardFilterMetadata] = Field(default_factory=list)
    kpis: list[KPISpec] = Field(default_factory=list, max_length=4)
    widgets: list[DashboardWidget] = Field(default_factory=list, max_length=4)
    data: DashboardData | None = None
    notices: list[str] = Field(default_factory=list)
    description: str = Field(default="", max_length=600)
    rationale: str = Field(default="", max_length=600)
    theme: Literal["green", "neutral", "contrast"] = "green"
    design_mode: Literal["deterministic", "ai", "fallback"] = "deterministic"
    revision_id: str = ""
    parent_revision_id: str | None = None

    @model_validator(mode="after")
    def source_identity(self) -> DashboardSpec:
        if any(k.result_id != self.result_id for k in self.kpis):
            raise ValueError("KPI source does not match dashboard result")
        if any(
            w.result_id != self.result_id or w.chart_spec.source_row_count != self.row_count
            for w in self.widgets
        ):
            raise ValueError("Chart source does not match dashboard result")
        if self.data and (
            len(self.data.rows) != self.row_count
            or any(len(row) != len(self.data.columns) for row in self.data.rows)
        ):
            raise ValueError("Dashboard table must contain the exact result")
        return self


class DashboardError(ValueError):
    def __init__(self, message: str, status_code: int = 409) -> None:
        super().__init__(message)
        self.status_code = status_code


def _label(value: str) -> str:
    return value.replace("_", " ").strip().capitalize()


def _numbers(source: RecentResult, column: str) -> list[int | float] | None:
    values = [row[source.columns.index(column)] for row in source.rows]
    if not values or not all(type(v) in {int, float} and math.isfinite(v) for v in values):
        return None
    return values


def _proof(source: RecentResult) -> tuple[dict[str, str], list[str]]:
    """Conservative proof: one relation, no join/window/CTE, all group keys selected.

    Only SUM(physical_column) over disjoint SQL groups is additive here. This
    never interprets an AVG, ratio, DISTINCT count or alias as additive.
    """
    try:
        root = parse_one(source.sql, read=source.dialect)
    except Exception:
        return {}, []
    if (
        not isinstance(root, exp.Select)
        or any(
            root.find(kind)
            for kind in (
                exp.Join,
                exp.With,
                exp.Window,
                exp.Distinct,
                exp.Having,
                exp.Limit,
                exp.Offset,
                exp.Subquery,
            )
        )
        or len(list(root.find_all(exp.Table))) != 1
    ):
        return {}, []
    groups = root.args.get("group")
    if groups is None or not groups.expressions:
        return {}, []
    projections = {item.alias_or_name: item.unalias() for item in root.expressions}
    grain = []
    for expression in groups.expressions:
        if not isinstance(expression, exp.Column):
            return {}, []
        name = next(
            (
                alias
                for alias, item in projections.items()
                if isinstance(item, exp.Column) and item.sql() == expression.sql()
            ),
            None,
        )
        if name is None or name not in source.columns:
            return {}, []
        grain.append(name)
    keys = [tuple(row[source.columns.index(name)] for name in grain) for row in source.rows]
    if len({repr(key) for key in keys}) != len(keys):
        return {}, []
    table = next(root.find_all(exp.Table))
    tables = source.verified_context.get("tables", {})
    metadata = tables.get(table.name, {}) if isinstance(tables, dict) else {}
    if not isinstance(metadata, dict):
        return {}, grain
    physical_grain = metadata.get("grain")
    if not physical_grain or "unknown" in str(physical_grain).casefold():
        return {}, grain
    sums = {
        alias: item.sql()
        for alias, item in projections.items()
        if isinstance(item, exp.Sum)
        and isinstance(item.this, exp.Column)
        and alias in source.columns
    }
    return sums, grain


def eligible_kpis(source: RecentResult, temporal_chart: ChartSpec | None = None,
                  *, limit: int = 4) -> list[KPISpec]:
    if not source.rows:
        return []
    identity = source.result_id or ""
    if source.row_count == 1:
        return [
            KPISpec(
                result_id=identity,
                metric=column,
                title=_label(column),
                calculation="DIRECT",
                value=values[0],
                definition="Exact value returned by the accepted SQL; no recalculation.",
            )
            for column in source.columns
            if (values := _numbers(source, column)) is not None
        ][:limit]
    additive, grain = _proof(source)
    if not grain:
        return []
    result = [
        KPISpec(
            result_id=identity,
            metric="returned_groups",
            title="Returned groups",
            calculation="COUNT",
            value=source.row_count,
            source_grain=grain,
            definition="Count of complete returned SQL groups; not distinct customers.",
        )
    ]
    for column, definition in additive.items():
        values = _numbers(source, column)
        if values is None:
            continue
        try:
            total = math.fsum(values)
        except OverflowError:
            continue
        if not math.isfinite(total):
            continue
        result.append(
            KPISpec(
                result_id=identity,
                metric=column,
                title=f"Total · {_label(column)}",
                calculation="TOTAL",
                value=total,
                source_grain=grain,
                definition=f"Sum of disjoint returned {definition} groups at grain {grain}.",
            )
        )
        result.append(
            KPISpec(
                result_id=identity,
                metric=column,
                title=f"Average group total · {_label(column)}",
                calculation="AVERAGE",
                value=total / len(values),
                source_grain=grain,
                definition=f"Mean of returned {definition} group totals, not a source-row average.",
            )
        )
        contract = source.semantic_contract
        periods = contract.comparison_periods if contract else []
        chart = source.chart_spec
        if chart is None or chart.source_row_count != source.row_count:
            chart = temporal_chart
        if (
            len(periods) == 2
            and len(set(periods)) == 2
            and chart
            and chart.time_format in {"month", "year", "day"}
        ):
            temporal = chart.x
            if temporal in grain:
                position = source.columns.index(temporal)
                rest = [source.columns.index(name) for name in grain if name != temporal]
                slices = [
                    [row for row in source.rows if str(row[position]) == period]
                    for period in periods
                ]
                domains = [{tuple(row[i] for i in rest) for row in rows} for rows in slices]
                if all(slices) and domains[0] == domains[1]:
                    index = source.columns.index(column)
                    baseline, current = [math.fsum(row[index] for row in rows) for rows in slices]
                    if baseline > 0 and current >= 0:
                        change = (current - baseline) / abs(baseline) * 100
                        if math.isfinite(change):
                            result.insert(
                                1,
                                KPISpec(
                                    result_id=identity,
                                    metric=column,
                                    title=f"{_label(column)} change",
                                    calculation="PERIOD_GROWTH",
                                    value=change,
                                    unit="percent",
                                    source_grain=grain,
                                    baseline=baseline,
                                    current=current,
                                    periods=periods,
                                    definition="(Current − baseline) / abs(baseline) × 100; "
                                    "explicit periods with matching group populations.",
                                ),
                            )
    return result[:limit]


def build_dashboard(
    request: DashboardRequest, resolve: Callable[[str, str, str], RecentResult | None],
    *, chart_catalog: list[DashboardWidget] | None = None,
) -> DashboardSpec:
    source = resolve(request.session_id, request.db_id, request.result_id)
    if source is None or source.result_id != request.result_id or source.db_id != request.db_id:
        raise DashboardError("This result is unavailable in the current session and database.", 404)
    if source.execution_status != "ACCEPTED":
        raise DashboardError("The result must have completed validated read-only execution.")
    if source.truncated or source.row_count != len(source.rows) or len(source.rows) > 500:
        raise DashboardError("A dashboard requires the complete retained result, not a preview.")
    if len(set(source.columns)) != len(source.columns) or any(
        len(row) != len(source.columns) for row in source.rows
    ):
        raise DashboardError("The result has ambiguous columns or inconsistent rows.")
    if any(
        type(value) not in {str, int, float, bool, type(None)}
        or isinstance(value, float)
        and not math.isfinite(value)
        for row in source.rows
        for value in row
    ):
        raise DashboardError("Phase 1 dashboards require finite, flat scalar result cells.")
    candidates = recommend_charts(source.question, source.columns, source.rows, truncated=False)
    widgets: list[DashboardWidget] = []
    families: set[str] = set()
    additive, _ = _proof(source)
    for candidate in (
        [candidates.primary_chart, *candidates.alternative_charts] if candidates else []
    ):
        kind = candidate.chart_type
        family = (
            "composition"
            if kind
            in {"pie", "donut", "treemap", "stacked_bar", "stacked_area", "percent_stacked_bar"}
            else "trend"
            if kind in {"line", "multi_line", "step_line", "area"}
            else "comparison"
            if "bar" in kind
            else "relationship"
            if kind in {"scatter", "bubble"}
            else "distribution"
            if kind in {"histogram", "box"}
            else "intensity"
        )
        if family in families and chart_catalog is None:
            continue
        if family == "composition" and candidate.y not in additive:
            continue
        valid = validate_chart_proposal(
            LLMChartProposal.model_validate(
                candidate.model_dump(
                    include={
                        "chart_type",
                        "x",
                        "y",
                        "x_components",
                        "y_measures",
                        "series",
                        "reason",
                    }
                )
            ),
            columns=source.columns,
            rows=source.rows,
            question="",
            truncated=False,
            score=1,
        )
        if valid is None:
            continue
        spec = materialize_recommendation(
            valid, question="", columns=source.columns, rows=source.rows, truncated=False
        )
        if chart_catalog is not None:
            chart_catalog.append(DashboardWidget(widget_id=f"candidate-{len(chart_catalog)}",
                result_id=request.result_id, row=min(len(chart_catalog)//2, 3), chart_spec=spec))
        if family in families or len(widgets) >= 4:
            continue
        widgets.append(
            DashboardWidget(
                widget_id=f"chart-{len(widgets)}",
                result_id=request.result_id,
                row=len(widgets) // 2,
                chart_spec=spec,
            )
        )
        families.add(family)
        if len(widgets) == 4 and chart_catalog is None:
            break
    if len(widgets) % 2:
        widgets[-1] = widgets[-1].model_copy(update={"column_span": 12})
    constraints = []
    try:
        parsed = parse_one(source.sql, read=source.dialect)
        if parsed is not None:
            constraints = [
                where.this.sql(dialect=source.dialect) for where in parsed.find_all(exp.Where)
            ]
            constraints.extend(
                f"Aggregate filter: {having.this.sql(dialect=source.dialect)}"
                for having in parsed.find_all(exp.Having)
            )
            if parsed.args.get("limit") is not None:
                constraints.append(parsed.args["limit"].sql(dialect=source.dialect))
    except Exception:
        pass
    digest = hashlib.sha256(
        f"{request.db_id}:{request.result_id}:{request.include_data}".encode()
    ).hexdigest()[:16]
    return DashboardSpec(
        dashboard_id=f"dashboard-{digest}",
        result_id=request.result_id,
        run_id=source.run_id,
        db_id=source.db_id,
        title=source.question,
        row_count=source.row_count,
        execution_timestamp=source.execution_timestamp,
        applied_constraints=constraints,
        verified_filter_metadata=[
            DashboardFilterMetadata(
                field=f.field,
                table=f.table,
                column=f.column,
                source="stored_result" if f.source == "stored_result" else "verified_schema",
            )
            for f in (
                [*source.filter_catalog.client_filters, *source.filter_catalog.sql_filters]
                if source.filter_catalog
                else []
            )
            if f.source in {"stored_result", "verified_schema"}
        ],
        kpis=eligible_kpis(
            source,
            next(
                (
                    widget.chart_spec
                    for widget in widgets
                    if widget.chart_spec.time_format in {"month", "year", "day"}
                ),
                None,
            ),
        ),
        widgets=widgets,
        data=DashboardData(columns=source.columns, rows=source.rows)
        if request.include_data
        else None,
        notices=["Temporary view of one retained result. Filters are read-only."]
        + (["The accepted result contains no rows."] if not source.rows else [])
        + (["No complementary chart can be validated for this result."] if not widgets else []),
    )

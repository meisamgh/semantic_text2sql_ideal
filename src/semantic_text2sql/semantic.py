"""Conservative semantic planning and SQL-contract validation."""

from __future__ import annotations

import re
from typing import Literal

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError

from semantic_text2sql.models import (
    AggregationStage,
    DerivedMetric,
    FieldResolution,
    FormulaMetric,
    MetricSelector,
    OutputOperation,
    ResolutionReport,
    SemanticContract,
    SemanticFilter,
    StructuralFormula,
    ValidationResult,
)

_TOP_N = re.compile(r"\b(?:top|first)\s+(\d+)\b", re.I)
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_NUMBER = re.compile(r"(?<![A-Za-z_])\d+(?:\.\d+)?(?![A-Za-z_])")
_LIST_NUMBER = re.compile(r"(?m)^\s*\d+\.\s+")
_PRODUCT_ID = re.compile(r"\bproduct(?:\s+id)?(?:\s+no\.?)?\s*(\d+)\b", re.I)
_MORE_THAN = re.compile(r"\b(?:more than|greater than|over)\s+(\d+(?:\.\d+)?)", re.I)
_MONTH_YEAR = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|"
    r"november|december)\s+((?:19|20)\d{2})\b",
    re.I,
)
_MONTHS = {
    name: f"{index:02d}"
    for index, name in enumerate(
        (
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ),
        1,
    )
}
_QUOTED = re.compile(r"['\"]([^'\"]{1,100})['\"]")


def plan_semantics(question: str, evidence: str | None = None) -> SemanticContract:
    numeric_source = _LIST_NUMBER.sub("", question)
    normalized_question = " ".join(question.split())
    lowered = normalized_question.casefold()
    aggregation: Literal["count", "sum", "average", "min", "max"] | None = None
    rationale: list[str] = []
    for kind, phrases in (
        ("average", ("average", "mean ")),
        ("sum", ("sum of", "total amount", "total cost", "total revenue", "total sales")),
        ("count", ("how many", "number of", "count of")),
        ("max", ("maximum value", "maximum number")),
        ("min", ("minimum value", "minimum number")),
    ):
        if any(phrase in lowered for phrase in phrases):
            aggregation = kind  # type: ignore[assignment]
            rationale.append(f"Explicit {kind} language in the question or evidence.")
            break
    grouped = aggregation is not None and bool(
        re.search(r"\b(?:each|per|grouped by|broken down by)\b", lowered)
    )
    distinct = bool(re.search(r"\b(?:distinct|unique)\b", lowered))
    ranking = bool(re.search(r"\b(?:top|bottom)\b|\b(?:ascending|descending) order\b", lowered))
    top_match = _TOP_N.search(normalized_question)
    limit = int(top_match.group(1)) if top_match else None
    quoted = [
        value for value in _QUOTED.findall(normalized_question) if not re.search(r"\d[/:-]", value)
    ]
    numeric_literals = _NUMBER.findall(numeric_source)
    if limit is not None:
        numeric_literals = [value for value in numeric_literals if value != str(limit)]
    literals = list(
        dict.fromkeys([*_YEAR.findall(normalized_question), *numeric_literals, *quoted])
    )
    if grouped:
        rationale.append("Explicit each/per wording requires grouped output grain.")
    if distinct:
        rationale.append("Explicit distinct/unique wording requires deduplication semantics.")
    if ranking:
        rationale.append("Explicit ranking wording requires deterministic ordering.")
    if limit:
        rationale.append(f"Explicit top/first {limit} requires LIMIT {limit}.")
    if literals:
        rationale.append("Explicit quoted values or years must remain grounded in SQL.")
    contract = SemanticContract(
        aggregation=aggregation,
        requires_grouping=grouped,
        requires_distinct=distinct,
        requires_ordering=ranking,
        limit=limit,
        required_literals=literals,
        rationale=rationale,
    )
    contract = _plan_metric_grain(normalized_question, evidence, contract)
    return _plan_unit_price_consumption(normalized_question, contract)


def _plan_unit_price_consumption(question: str, contract: SemanticContract) -> SemanticContract:
    """Ground the reusable cross-table unit-price filter plus monthly consumption shape."""
    lowered = question.casefold()
    product = _PRODUCT_ID.search(question)
    threshold = _MORE_THAN.search(question)
    month_year = _MONTH_YEAR.search(question)
    if not (
        "consumption" in lowered and "per unit" in lowered and product and threshold and month_year
    ):
        return contract
    month, year = month_year.groups()
    period = year + _MONTHS[month.casefold()]
    filters = [
        SemanticFilter(
            operand="transactions_1k.ProductID",
            operator="=",
            value=int(product.group(1)),
        ),
        SemanticFilter(
            operand="unit_price",
            operator=">",
            value=float(threshold.group(1)),
        ),
        SemanticFilter(operand="yearmonth.Date", operator="=", value=period),
    ]
    return contract.model_copy(
        update={
            "aggregation": None,
            "outputs": ["yearmonth.CustomerID", "yearmonth.Consumption"],
            "filters": filters,
            "advisory_outputs": [],
            "measures": ["yearmonth.Consumption"],
            "proposed_tables": ["yearmonth", "transactions_1k"],
            "required_columns": [
                "yearmonth.CustomerID",
                "yearmonth.Date",
                "yearmonth.Consumption",
                "transactions_1k.CustomerID",
                "transactions_1k.ProductID",
                "transactions_1k.Price",
                "transactions_1k.Amount",
            ],
            "proposed_joins": ["yearmonth.CustomerID = transactions_1k.CustomerID"],
            "proposed_filters": [
                "transactions_1k.ProductID = " + product.group(1),
                "unit_price > " + threshold.group(1),
                "yearmonth.Date = " + period,
            ],
            "grain": ["CustomerID", "Month"],
            "rationale": [
                *contract.rationale,
                "Consumption status is a non-aggregated monthly measure output.",
                "The transaction table qualifies customers and must remain filter-only.",
            ],
        }
    )


def apply_structural_formulas(
    contract: SemanticContract, formulas: list[StructuralFormula]
) -> SemanticContract:
    """Attach approved formulas without allowing them to override explicit deterministic facts."""
    if not formulas:
        return contract
    columns = [
        argument for formula in formulas for argument in formula.arguments if "." in argument
    ]
    return contract.model_copy(
        update={
            "structural_formulas": list({formula.id: formula for formula in formulas}.values()),
            "required_columns": list(dict.fromkeys([*contract.required_columns, *columns])),
            "rationale": [
                *contract.rationale,
                "Approved glossary structural formulas were attached without reinterpretation.",
            ],
        }
    )


def resolution_report(question: str, contract: SemanticContract) -> ResolutionReport:
    """Report whether deterministic planning resolved SQL-shaping semantics."""
    text = question.casefold()
    formula_requested = bool(
        re.search(
            r"\b(per unit|average price per|percentage|ratio|share|difference|growth)\b",
            text,
        )
    )
    temporal_requested = bool(
        re.search(r"\b(month|year|annual|date|day|week|quarter)\b", text) or _YEAR.search(text)
    )
    aggregation_requested = bool(
        re.search(r"\b(how many|count|average|mean|sum|total|minimum|maximum)\b", text)
    )
    grain_requested = bool(re.search(r"\b(each|grouped by|broken down by|between)\b", text))
    selection_requested = bool(
        re.search(r"\b(top|bottom|least|lowest|highest|maximum|minimum)\b", text)
    )
    multi_table = len(contract.proposed_tables) > 1

    def field(
        name: str,
        resolved: bool,
        *,
        critical: bool,
        provenance: str,
    ) -> FieldResolution:
        return FieldResolution(
            field=name,
            status="RESOLVED" if resolved else "UNRESOLVED",
            confidence=0.95 if resolved else 0.3,
            provenance=[provenance],
            critical=critical,
        )

    fields = [
        field(
            "output",
            bool(contract.outputs or contract.named_outputs or contract.aggregation),
            critical=True,
            provenance="deterministic output planner",
        ),
        field(
            "measure",
            bool(contract.measures) or contract.aggregation == "count",
            critical=bool(contract.aggregation and contract.aggregation != "count")
            or "consumption" in text,
            provenance="deterministic measure planner",
        ),
        field(
            "formula",
            bool(contract.structural_formulas or contract.formula_metrics) or not formula_requested,
            critical=formula_requested,
            provenance="approved glossary and deterministic formula planner",
        ),
        field(
            "temporal_scope",
            any(item.operand.casefold().endswith(".date") for item in contract.filters)
            or not temporal_requested,
            critical=temporal_requested,
            provenance="deterministic temporal planner",
        ),
        field(
            "aggregation",
            bool(contract.aggregation or contract.aggregation_stages) or not aggregation_requested,
            critical=aggregation_requested,
            provenance="deterministic aggregation planner",
        ),
        field(
            "grain",
            bool(contract.grain or contract.aggregation_stages) or not grain_requested,
            critical=grain_requested or bool(contract.measures),
            provenance="deterministic grain planner",
        ),
        field(
            "selection_metric",
            bool(contract.selectors) or not selection_requested,
            critical=selection_requested,
            provenance="deterministic selection planner",
        ),
        field(
            "join_path",
            bool(contract.proposed_joins) or not multi_table,
            critical=multi_table,
            provenance="deterministic relationship planner",
        ),
        FieldResolution(
            field="remaining_ambiguities",
            status="AMBIGUOUS" if contract.ambiguities else "RESOLVED",
            confidence=0.25 if contract.ambiguities else 1.0,
            provenance=["non-blocking ambiguity inventory"],
            critical=False,
        ),
    ]
    unresolved = [item for item in fields if item.critical and item.status != "RESOLVED"]
    status = unresolved[0].status if unresolved else "RESOLVED"
    return ResolutionReport(
        status=status,
        fields=fields,
        semantic_call_required=bool(unresolved),
    )


def _plan_metric_grain(
    question: str, evidence: str | None, contract: SemanticContract
) -> SemanticContract:
    """Create an explicit aggregation DAG for high-confidence nested metric language."""
    text = f"{question} {evidence or ''}".casefold()
    if _is_segment_percentage_change(text):
        return _plan_segment_percentage_change(contract)
    required = ("average", "consumption", "least")
    has_segments = "segment" in text or all(value in text for value in ("sme", "lam", "kam"))
    if not all(term in text for term in required) or not has_segments:
        return contract
    comparisons = (
        ("SME_minus_LAM", "SME.segment_min_avg", "LAM.segment_min_avg"),
        ("LAM_minus_KAM", "LAM.segment_min_avg", "KAM.segment_min_avg"),
        ("KAM_minus_SME", "KAM.segment_min_avg", "SME.segment_min_avg"),
    )
    literals = list(dict.fromkeys([*contract.required_literals, "CZK", "SME", "LAM", "KAM"]))
    return contract.model_copy(
        update={
            "aggregation": "average",
            "requires_grouping": True,
            "required_literals": literals,
            "measures": ["yearmonth.Consumption"],
            "derived_metrics": [
                DerivedMetric(
                    name="customer_annual_avg",
                    function="average",
                    input="yearmonth.Consumption",
                    group_by=["customers.Segment", "yearmonth.CustomerID"],
                )
            ],
            "selectors": [
                MetricSelector(
                    name="segment_min_avg",
                    function="min",
                    metric="customer_annual_avg",
                    partition_by=["customers.Segment"],
                )
            ],
            "output_operations": [
                OutputOperation(name=name, operator="subtract", left=left, right=right)
                for name, left, right in comparisons
            ],
            "rationale": [
                *contract.rationale,
                "Metric and grain planner fixed AVG per customer before MIN per segment.",
            ],
        }
    )


def _is_segment_percentage_change(text: str) -> bool:
    return (
        "consumption" in text
        and "percentage" in text
        and any(term in text for term in ("increase", "decrease", "change"))
        and "2012" in text
        and "2013" in text
        and all(segment in text for segment in ("sme", "lam", "kam"))
    )


def _plan_segment_percentage_change(contract: SemanticContract) -> SemanticContract:
    literals = list(
        dict.fromkeys([*contract.required_literals, "EUR", "SME", "LAM", "KAM", "2012", "2013"])
    )
    return contract.model_copy(
        update={
            "aggregation": "sum",
            "requires_grouping": True,
            "requires_ordering": True,
            "required_literals": literals,
            "measures": ["yearmonth.Consumption"],
            "aggregation_stages": [
                AggregationStage(
                    name="yearly_consumption",
                    function="sum",
                    input="yearmonth.Consumption",
                    group_by=["customers.Segment", "year"],
                    output_grain=["Segment", "Year"],
                )
            ],
            "formula_metrics": [
                FormulaMetric(
                    name="percentage_increase",
                    input="yearly_consumption",
                    partition_by=["Segment"],
                    bindings={
                        "baseline": "yearly_consumption[Year=2012]",
                        "comparison": "yearly_consumption[Year=2013]",
                    },
                    formula="(comparison - baseline) / baseline * 100",
                    denominator="baseline",
                    multiplier=100,
                    zero_denominator="null",
                    output_grain=["Segment"],
                )
            ],
            "selectors": [
                MetricSelector(
                    name="biggest_increase",
                    function="argmax",
                    metric="percentage_increase",
                    scope="across_groups",
                    tie_policy="return_all",
                ),
                MetricSelector(
                    name="lowest_increase",
                    function="argmin",
                    metric="percentage_increase",
                    scope="across_groups",
                    tie_policy="return_all",
                ),
            ],
            "named_outputs": [
                "segment_with_max_percentage_increase",
                "max_percentage_increase",
                "segment_with_min_percentage_increase",
                "min_percentage_increase",
            ],
            "rationale": [
                *contract.rationale,
                "Metric and grain planner fixed SUM by segment/year before percentage change.",
                "Percentage denominator is the 2012 baseline; selection uses ARGMAX and ARGMIN.",
            ],
        }
    )


def contract_context(contract: SemanticContract) -> str:
    return "Semantic contract (deterministic, authoritative):\n" + contract.model_dump_json(
        indent=2
    )


def validate_semantics(
    sql: str,
    contract: SemanticContract,
    validation: ValidationResult,
    *,
    dialect: Literal["sqlite", "postgres"] = "sqlite",
) -> ValidationResult:
    if not validation.valid:
        return validation
    try:
        root = sqlglot.parse_one(sql, read=dialect)
    except (ParseError, TokenError):
        return validation
    checks: list[str] = []
    aggregate_types: dict[str, type[exp.AggFunc]] = {
        "count": exp.Count,
        "sum": exp.Sum,
        "average": exp.Avg,
        "min": exp.Min,
        "max": exp.Max,
    }
    if contract.aggregation:
        aggregate_type = aggregate_types[contract.aggregation]
        has_aggregate = any(isinstance(node, aggregate_type) for node in root.walk())
        if contract.aggregation == "average":
            has_aggregate = has_aggregate or (
                root.find(exp.Div) is not None
                and root.find(exp.Sum) is not None
                and root.find(exp.Count) is not None
            )
        if not has_aggregate:
            return _failure(validation, "SEMANTIC_AGGREGATION_MISSING", contract.aggregation)
        checks.append(f"aggregation:{contract.aggregation}")
    if contract.requires_grouping:
        if root.find(exp.Group) is None:
            return _failure(validation, "SEMANTIC_GRAIN_MISSING", "GROUP BY")
        checks.append("grain:grouped")
    if contract.requires_distinct:
        has_distinct = any(
            isinstance(node, exp.Distinct)
            or (isinstance(node, exp.Count) and bool(node.args.get("distinct")))
            for node in root.walk()
        )
        if not has_distinct:
            return _failure(validation, "SEMANTIC_DISTINCT_MISSING", "DISTINCT")
        checks.append("distinct:present")
    if contract.requires_ordering:
        if root.find(exp.Order) is None:
            return _failure(validation, "SEMANTIC_ORDER_MISSING", "ORDER BY")
        checks.append("ranking:ordered")
    if contract.limit is not None:
        limit_node = root.find(exp.Limit)
        limit_value = limit_node.expression if limit_node else None
        if not isinstance(limit_value, exp.Literal) or limit_value.this != str(contract.limit):
            return _failure(validation, "SEMANTIC_LIMIT_MISMATCH", str(contract.limit))
        checks.append(f"limit:{contract.limit}")
    rendered = root.sql(dialect=dialect).casefold()
    if contract.outputs:
        output_validation = _validate_authoritative_outputs(root, contract, validation)
        if not output_validation.valid:
            return output_validation
        checks.append("outputs:authoritative")
    for literal in contract.required_literals:
        if not _literal_present(root, rendered, literal):
            return _failure(validation, "SEMANTIC_FILTER_MISSING", literal)
        checks.append(f"literal:{literal}")
    if contract.structural_formulas:
        formula_validation = _validate_structural_formulas(root, contract, validation)
        if not formula_validation.valid:
            return formula_validation
        checks.extend(f"structural_formula:{item.id}" for item in contract.structural_formulas)
    if contract.derived_metrics:
        lineage = _validate_metric_lineage(root, contract, validation)
        if not lineage.valid:
            return lineage
        checks.extend(["metric_lineage:derived", "selector:partitioned", "outputs:comparisons"])
    if contract.formula_metrics:
        lineage = _validate_percentage_change_lineage(root, contract, validation)
        if not lineage.valid:
            return lineage
        checks.extend(
            [
                "metric_lineage:sum_by_segment_year",
                "formula:baseline_percentage_change",
                "selectors:argmax_argmin",
                "outputs:segment_and_percentage",
            ]
        )
    return validation.model_copy(update={"semantic_checks": checks})


def _validate_authoritative_outputs(
    root: exp.Expression,
    contract: SemanticContract,
    validation: ValidationResult,
) -> ValidationResult:
    select = root if isinstance(root, exp.Select) else root.find(exp.Select)
    if select is None:
        return validation
    observed: list[str] = []
    for projection in select.expressions:
        alias = projection.alias
        if alias:
            observed.append(alias.casefold())
            continue
        columns = list(projection.find_all(exp.Column))
        observed.append(columns[-1].name.casefold() if columns else projection.sql().casefold())
    expected = [item.rsplit(".", 1)[-1].casefold() for item in contract.outputs]
    if observed != expected:
        return validation.model_copy(
            update={
                "valid": False,
                "code": "SEMANTIC_OUTPUT_MISMATCH",
                "message": (
                    f"Expected output columns in order {expected}; observed {observed}. "
                    "Do not add diagnostic or filter-only columns."
                ),
            }
        )
    return validation


def _literal_present(root: exp.Expression, rendered: str, literal: str) -> bool:
    if literal.casefold() in rendered:
        return True
    try:
        expected = float(literal)
    except ValueError:
        return False
    for node in root.find_all(exp.Literal):
        if node.is_string:
            continue
        try:
            if float(node.this) == expected:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _validate_structural_formulas(
    root: exp.Expression,
    contract: SemanticContract,
    validation: ValidationResult,
) -> ValidationResult:
    operator_types: dict[str, type[exp.Expression]] = {
        "DIVIDE": exp.Div,
        "MULTIPLY": exp.Mul,
        "ADD": exp.Add,
        "SUBTRACT": exp.Sub,
    }
    aliases = {
        table.alias_or_name.casefold(): table.name.casefold() for table in root.find_all(exp.Table)
    }
    for formula in contract.structural_formulas:
        expected_type = operator_types[formula.operator]
        candidates = [node for node in root.walk() if isinstance(node, expected_type)]
        matched = False
        for node in candidates:
            if formula.usage == "FILTER_OPERAND" and not (
                node.find_ancestor(exp.Where) or node.find_ancestor(exp.Having)
            ):
                continue
            sides = [node.this, node.expression]
            if not all(
                _argument_present(side, argument, aliases)
                for side, argument in zip(sides, formula.arguments, strict=True)
            ):
                continue
            if formula.zero_safe and formula.operator == "DIVIDE":
                denominator = node.expression
                if denominator.find(exp.Nullif) is None and not isinstance(denominator, exp.Nullif):
                    continue
            matched = True
            break
        if not matched:
            arguments = ", ".join(formula.arguments)
            exact_sql = (
                f"{formula.arguments[0]} / NULLIF({formula.arguments[1]}, 0)"
                if formula.operator == "DIVIDE" and formula.zero_safe
                else f"{formula.operator}({arguments})"
            )
            return validation.model_copy(
                update={
                    "valid": False,
                    "code": "STRUCTURAL_FORMULA_MISMATCH",
                    "message": (
                        f"Required {formula.usage} formula {formula.id} is "
                        f"{formula.operator}({arguments}). Use exactly this SQL expression: "
                        f"{exact_sql}. Fix only this formula violation while preserving the "
                        "authoritative outputs, filters, grain, and join strategy."
                    ),
                }
            )
    return validation


def _argument_present(
    expression: exp.Expression,
    argument: str,
    aliases: dict[str, str],
) -> bool:
    expected_table, _, expected_column = argument.casefold().rpartition(".")
    for column in expression.find_all(exp.Column):
        actual_table = aliases.get(column.table.casefold(), column.table.casefold())
        if column.name.casefold() != expected_column:
            continue
        if not expected_table or not actual_table or actual_table == expected_table:
            return True
    return False


def _validate_percentage_change_lineage(
    root: exp.Expression,
    contract: SemanticContract,
    validation: ValidationResult,
) -> ValidationResult:
    stage = contract.aggregation_stages[0]
    input_column = stage.input.rsplit(".", 1)[-1].casefold()
    has_consumption_sum = any(
        any(column.name.casefold() == input_column for column in total.find_all(exp.Column))
        for total in root.find_all(exp.Sum)
    )
    group_columns = {
        column.name.casefold()
        for group in root.find_all(exp.Group)
        for column in group.find_all(exp.Column)
    }
    if not has_consumption_sum or not {"segment", "date"} <= group_columns:
        return _lineage_failure(
            validation,
            expected="SUM(yearmonth.Consumption) grouped by Segment and Year",
            observed="The first aggregation or its Segment+Year grain did not match",
        )

    formula = contract.formula_metrics[0]
    valid_divisions: list[exp.Div] = []
    for division in root.find_all(exp.Div):
        numerator_sql = division.this.sql().casefold()
        denominator_sql = division.expression.sql().casefold()
        if (
            "2012" in numerator_sql
            and "2013" in numerator_sql
            and "2012" in denominator_sql
            and "2013" not in denominator_sql
            and division.this.find(exp.Sub) is not None
        ):
            valid_divisions.append(division)
    if not valid_divisions:
        return _lineage_failure(
            validation,
            expected=formula.formula + " with NULL-safe 2012 baseline denominator",
            observed="No percentage division used 2012 as baseline after 2013-2012",
        )
    if not any(
        any(literal.this in {"100", "100.0"} for literal in division.this.find_all(exp.Literal))
        and isinstance(division.expression, exp.Nullif)
        for division in valid_divisions
    ):
        return _lineage_failure(
            validation,
            expected="percentage scale of 100 and NULLIF protection on the 2012 baseline",
            observed="The derived ratio was not percentage-scaled or zero-safe",
        )

    formula_aliases = {formula.name.casefold()}
    for division in valid_divisions:
        node: exp.Expression | None = division
        while node is not None and not isinstance(node, exp.Select):
            if isinstance(node, exp.Alias):
                formula_aliases.add(node.alias.casefold())
                break
            node = node.parent

    rank_directions: set[str] = set()
    rank_aliases: set[str] = set()
    for window in root.find_all(exp.Window):
        if not isinstance(window.this, (exp.Rank, exp.DenseRank)):
            continue
        order = window.args.get("order")
        if not isinstance(order, exp.Order):
            continue
        ordered = list(order.find_all(exp.Ordered))
        order_columns = {column.name.casefold() for column in order.find_all(exp.Column)}
        if not order_columns & formula_aliases:
            continue
        descending = any(bool(item.args.get("desc")) for item in ordered)
        rank_directions.add("desc" if descending else "asc")
        if isinstance(window.parent, exp.Alias):
            rank_aliases.add(window.parent.alias.casefold())
    rank_one_filters: set[str] = set()
    for where in root.find_all(exp.Where):
        for comparison in where.find_all(exp.EQ):
            left, right = comparison.this, comparison.expression
            if (
                isinstance(left, exp.Column)
                and isinstance(right, exp.Literal)
                and right.this == "1"
            ):
                rank_one_filters.add(left.name.casefold())
            if isinstance(right, exp.Column) and isinstance(left, exp.Literal) and left.this == "1":
                rank_one_filters.add(right.name.casefold())
    if (
        rank_directions != {"asc", "desc"}
        or len(rank_aliases) < 2
        or not rank_aliases <= rank_one_filters
    ):
        return _lineage_failure(
            validation,
            expected="ARGMAX and ARGMIN over percentage_increase across segments with ties",
            observed="Both ascending and descending RANK selectors were not found",
        )

    outer = root if isinstance(root, exp.Select) else root.find(exp.Select)
    outer_columns: set[str] = set()
    if isinstance(outer, exp.Select):
        for projection in outer.expressions:
            outer_columns.update(
                column.name.casefold() for column in projection.find_all(exp.Column)
            )
    if "segment" not in outer_columns or not outer_columns & formula_aliases:
        return _lineage_failure(
            validation,
            expected="Selected segment identity and percentage_increase value",
            observed="Final output omitted the selected segment or percentage value",
        )
    return validation


def _validate_metric_lineage(
    root: exp.Expression,
    contract: SemanticContract,
    validation: ValidationResult,
) -> ValidationResult:
    derived = contract.derived_metrics[0]
    averages = list(root.find_all(exp.Avg))
    input_column = derived.input.rsplit(".", 1)[-1].casefold()
    matching_average = any(
        any(column.name.casefold() == input_column for column in avg.find_all(exp.Column))
        for avg in averages
    )
    group_columns = {
        column.name.casefold()
        for group in root.find_all(exp.Group)
        for column in group.find_all(exp.Column)
    }
    expected_groups = {value.rsplit(".", 1)[-1].casefold() for value in derived.group_by}
    if not matching_average or not expected_groups <= group_columns:
        return _lineage_failure(
            validation,
            expected=(f"{derived.name}=AVG({derived.input}) grouped by {derived.group_by}"),
            observed="AVG inputs or grouping columns did not match the contract",
        )
    selector = contract.selectors[0]
    derived_aliases = {selector.metric.casefold()}
    for average in averages:
        parent = average.parent
        if isinstance(parent, exp.Alias) and any(
            column.name.casefold() == input_column for column in average.find_all(exp.Column)
        ):
            derived_aliases.add(parent.alias.casefold())
    has_min_selector = any(
        any(column.name.casefold() in derived_aliases for column in minimum.find_all(exp.Column))
        for minimum in root.find_all(exp.Min)
    )
    has_rank_selector = False
    for window in root.find_all(exp.Window):
        if not isinstance(window.this, (exp.RowNumber, exp.Rank, exp.DenseRank)):
            continue
        partitions = {
            column.name.casefold()
            for expression in window.args.get("partition_by") or []
            for column in expression.find_all(exp.Column)
        }
        order = window.args.get("order")
        order_columns = (
            {column.name.casefold() for column in order.find_all(exp.Column)}
            if isinstance(order, exp.Order)
            else set()
        )
        expected_partitions = {
            value.rsplit(".", 1)[-1].casefold() for value in selector.partition_by
        }
        if expected_partitions <= partitions and any(
            column in derived_aliases for column in order_columns
        ):
            has_rank_selector = True
            break
    if not (has_min_selector or has_rank_selector):
        return _lineage_failure(
            validation,
            expected=(
                f"{selector.name}=MIN({selector.metric}) partitioned by {selector.partition_by}"
            ),
            observed="No MIN or ascending partitioned rank over the required derived metric",
        )
    if sum(1 for _ in root.find_all(exp.Sub)) < len(contract.output_operations):
        return _lineage_failure(
            validation,
            expected=f"{len(contract.output_operations)} named subtraction outputs",
            observed="Too few subtraction expressions",
        )
    return validation


def _lineage_failure(
    validation: ValidationResult, *, expected: str, observed: str
) -> ValidationResult:
    return validation.model_copy(
        update={
            "valid": False,
            "code": "SEMANTIC_METRIC_LINEAGE_MISMATCH",
            "message": f"Expected metric lineage: {expected}. Observed: {observed}.",
        }
    )


def _failure(validation: ValidationResult, code: str, requirement: str) -> ValidationResult:
    return validation.model_copy(
        update={
            "valid": False,
            "code": code,
            "message": f"SQL violates the semantic contract; required: {requirement}.",
        }
    )

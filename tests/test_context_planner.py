from __future__ import annotations

import pytest
from pydantic import ValidationError

from semantic_text2sql.context import build_context_plan, model_context_payload
from semantic_text2sql.context_planner import (
    reconcile_context_contract,
    verify_context_request,
)
from semantic_text2sql.models import (
    ColumnInfo,
    ContextRequest,
    DatabaseProfile,
    PlannerAggregation,
    PlannerMetadataRequirement,
    PlannerRanking,
    RelationshipProfile,
    SchemaInfo,
    SemanticFilter,
    StructuralFormula,
    TableInfo,
    TableProfile,
)
from semantic_text2sql.semantic import apply_structural_formulas, plan_semantics


def _schema() -> SchemaInfo:
    def table(name: str, columns: list[str]) -> TableInfo:
        return TableInfo(
            name=name,
            create_sql="",
            columns=[ColumnInfo(name=item, data_type="TEXT") for item in columns],
        )

    return SchemaInfo(
        db_id="debit_card_specializing",
        tables=[
            table("customers", ["CustomerID", "Segment", "Currency"]),
            table(
                "transactions_1k",
                ["TransactionID", "Date", "CustomerID", "ProductID", "Amount", "Price"],
            ),
            table("yearmonth", ["CustomerID", "Date", "Consumption"]),
        ],
    )


def _profile() -> DatabaseProfile:
    return DatabaseProfile(
        db_id="debit_card_specializing",
        dialect="sqlite",
        profiled_at="2026-01-01",
        columns=[],
        tables=[
            TableProfile(table="customers", summary="", grain="CustomerID"),
            TableProfile(table="transactions_1k", summary="", grain="TransactionID"),
            TableProfile(table="yearmonth", summary="", grain="CustomerID + Date"),
        ],
        relationships=[
            RelationshipProfile(
                parent_table="transactions_1k",
                parent_column="CustomerID",
                child_table="yearmonth",
                child_column="CustomerID",
                type="MANY_TO_MANY",
                parent_key_unique=False,
                child_key_unique=False,
                inferred=True,
            ),
            RelationshipProfile(
                parent_table="customers",
                parent_column="CustomerID",
                child_table="transactions_1k",
                child_column="CustomerID",
                type="ONE_TO_MANY",
                parent_key_unique=True,
                child_key_unique=False,
                inferred=True,
            ),
        ],
    )


def _unit_price_contract(question: str):  # type: ignore[no-untyped-def]
    return apply_structural_formulas(
        plan_semantics(question),
        [
            StructuralFormula(
                id="unit_price",
                operator="DIVIDE",
                arguments=["transactions_1k.Price", "transactions_1k.Amount"],
                zero_safe=True,
                source="APPROVED_GLOSSARY",
                usage="FILTER_OPERAND",
            )
        ],
    )


def test_context_request_forbids_sql_and_arbitrary_metadata() -> None:
    with pytest.raises(ValidationError):
        ContextRequest.model_validate({"outputs": [], "sql": "SELECT 1"})
    with pytest.raises(ValidationError):
        PlannerMetadataRequirement(kind="ANYTHING", targets=["customers"])  # type: ignore[arg-type]


def test_product5_verifier_force_adds_formula_and_relationship_dependencies() -> None:
    question = (
        "For all the people who paid more than 29.00 per unit of product ID No. 5, "
        "give their consumption status in August 2012."
    )
    contract = _unit_price_contract(question)
    requested = ContextRequest(
        outputs=["yearmonth.CustomerID", "yearmonth.Consumption"],
        tables=["transactions_1k", "yearmonth"],
        columns={
            "transactions_1k": ["ProductID", "Price"],
            "yearmonth": ["Date", "Consumption"],
        },
        filters=[
            SemanticFilter(operand="transactions_1k.ProductID", operator="=", value=5),
            SemanticFilter(operand="unit_price", operator=">", value=29),
            SemanticFilter(operand="yearmonth.Date", operator="=", value="August 2012"),
        ],
        measures=["yearmonth.Consumption"],
        business_concepts=["unit_price"],
        metadata_requirements=[
            PlannerMetadataRequirement(kind="DATE_FORMAT", targets=["yearmonth.Date"]),
        ],
    )

    verified = verify_context_request(requested, _schema(), contract, _profile())

    assert set(verified.columns["transactions_1k"]) == {
        "ProductID",
        "Price",
        "CustomerID",
        "Amount",
    }
    assert set(verified.columns["yearmonth"]) == {"Date", "Consumption", "CustomerID"}
    assert {item.kind for item in verified.metadata_requirements} >= {
        "RELATIONSHIP",
        "JOIN_CARDINALITY",
        "TABLE_GRAIN",
        "DATE_FORMAT",
        "FORMULA_DEFINITION",
    }

    reconciled = reconcile_context_contract(contract, verified)
    selected_schema = _schema().model_copy(
        update={
            "tables": [
                table.model_copy(
                    update={
                        "columns": [
                            column
                            for column in table.columns
                            if column.name in verified.columns.get(table.name, [])
                        ]
                    }
                )
                for table in _schema().tables
                if table.name in verified.tables
            ]
        }
    )
    plan = build_context_plan(selected_schema, _profile(), reconciled, question, None, verified)
    context = model_context_payload(plan, _profile(), reconciled, None, question, "sqlite")
    relationship = context["relationships"][0]
    assert "recommended_strategy" not in relationship
    assert "filter_side" not in relationship
    assert relationship["cardinality"] == "MANY_TO_MANY"


def test_top_spending_plan_keeps_ranking_and_requested_metric_separate() -> None:
    requested = ContextRequest(
        outputs=["customers.CustomerID", "average_unit_price", "customers.Currency"],
        tables=["customers", "transactions_1k"],
        columns={
            "customers": ["CustomerID", "Currency"],
            "transactions_1k": ["CustomerID", "Price", "Amount"],
        },
        measures=["transactions_1k.Price"],
        aggregations=[
            PlannerAggregation(
                function="sum", input="transactions_1k.Price", output="total_spending"
            ),
            PlannerAggregation(function="average", input="unit_price", output="average_unit_price"),
        ],
        ranking=[PlannerRanking(metric="total_spending", direction="DESC", limit=1)],
        business_concepts=["total_spending", "unit_price", "average_unit_price"],
    )
    contract = _unit_price_contract(
        "Who is the top spending customer and what is their average price per single item?"
    )

    verified = verify_context_request(requested, _schema(), contract, _profile())

    assert verified.aggregations[0].input == "transactions_1k.Price"
    assert verified.aggregations[1].input == "unit_price"
    assert verified.ranking[0].metric == "total_spending"
    assert verified.columns["customers"] == ["CustomerID", "Currency"]
    assert set(verified.columns["transactions_1k"]) == {"CustomerID", "Price", "Amount"}


def test_customer_spending_date_request_does_not_force_unneeded_tables() -> None:
    requested = ContextRequest(
        outputs=["total_spending"],
        tables=["transactions_1k"],
        columns={"transactions_1k": ["CustomerID", "Price", "Date"]},
        filters=[SemanticFilter(operand="transactions_1k.CustomerID", operator="=", value=38508)],
        measures=["transactions_1k.Price"],
        aggregations=[
            PlannerAggregation(
                function="sum", input="transactions_1k.Price", output="total_spending"
            )
        ],
        metadata_requirements=[
            PlannerMetadataRequirement(kind="DATE_FORMAT", targets=["transactions_1k.Date"]),
            PlannerMetadataRequirement(kind="FORMULA_DEFINITION", targets=["total_spending"]),
        ],
    )
    verified = verify_context_request(
        requested,
        _schema(),
        plan_semantics("What did customer 38508 spend in January 2012?"),
        _profile(),
    )

    assert verified.tables == ["transactions_1k"]
    assert set(verified.columns) == {"transactions_1k"}
    assert not any(item.kind == "RELATIONSHIP" for item in verified.metadata_requirements)


def test_single_table_global_ratio_has_no_join_or_date_metadata() -> None:
    requested = ContextRequest(
        outputs=["currency_ratio"],
        tables=["customers"],
        columns={"customers": ["Currency"]},
        filters=[SemanticFilter(operand="customers.Currency", operator="IN", value=["EUR", "CZK"])],
        business_concepts=["currency_ratio"],
    )
    verified = verify_context_request(
        requested,
        _schema(),
        plan_semantics("What is the ratio of EUR to CZK customers?"),
        _profile(),
    )

    assert verified.tables == ["customers"]
    assert verified.group_by == []
    assert not any(
        item.kind in {"RELATIONSHIP", "JOIN_CARDINALITY", "DATE_FORMAT"}
        for item in verified.metadata_requirements
    )


def test_controller_restores_primary_key_omitted_by_model_1() -> None:
    schema = _schema()
    customers = schema.tables[0].model_copy(
        update={
            "columns": [
                column.model_copy(update={"primary_key": column.name == "CustomerID"})
                for column in schema.tables[0].columns
            ]
        }
    )
    schema = schema.model_copy(update={"tables": [customers, *schema.tables[1:]]})
    requested = ContextRequest(
        tables=["customers"],
        columns={"customers": ["Currency"]},
    )

    verified = verify_context_request(
        requested,
        schema,
        plan_semantics("What is the ratio of EUR to CZK customers?"),
        _profile(),
    )

    assert verified.columns["customers"] == ["Currency", "CustomerID"]

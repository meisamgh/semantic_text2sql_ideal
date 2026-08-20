from __future__ import annotations

from semantic_text2sql.models import StructuralFormula, ValidationResult
from semantic_text2sql.semantic import (
    apply_structural_formulas,
    plan_semantics,
    resolution_report,
    validate_semantics,
)


def _valid() -> ValidationResult:
    return ValidationResult(valid=True, code="SQL_VALID", message="ok")


def test_planner_extracts_grouped_count_and_top_n() -> None:
    contract = plan_semantics("What are the top 3 categories by number of books in 2024?")

    assert contract.aggregation == "count"
    assert contract.requires_ordering is True
    assert contract.limit == 3
    assert contract.required_literals == ["2024"]


def test_semantic_validator_rejects_missing_grouping() -> None:
    contract = plan_semantics("How many books are there in each category?")

    result = validate_semantics("SELECT COUNT(*) FROM books", contract, _valid())

    assert result.valid is False
    assert result.code == "SEMANTIC_GRAIN_MISSING"


def test_semantic_validator_accepts_matching_contract() -> None:
    contract = plan_semantics("Show the top 3 categories by number of books in 2024")
    sql = (
        "SELECT category, COUNT(*) FROM books WHERE year = 2024 "
        "GROUP BY category ORDER BY COUNT(*) DESC LIMIT 3"
    )

    result = validate_semantics(sql, contract, _valid())

    assert result.valid is True
    assert result.semantic_checks == [
        "aggregation:count",
        "ranking:ordered",
        "limit:3",
        "literal:2024",
    ]


def test_structural_formula_must_be_the_filter_operand() -> None:
    contract = apply_structural_formulas(
        plan_semantics("People who paid more than 29 per unit for product 5"),
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
    wrong = "SELECT CustomerID FROM transactions_1k WHERE Price > 29 AND ProductID = 5"
    correct = (
        "SELECT CustomerID FROM transactions_1k "
        "WHERE Price / NULLIF(Amount, 0) > 29 AND ProductID = 5"
    )

    rejected = validate_semantics(wrong, contract, _valid())
    assert rejected.valid is False
    assert rejected.code == "STRUCTURAL_FORMULA_MISMATCH"
    assert validate_semantics(correct, contract, _valid()).valid is True


def test_product_unit_price_consumption_ir_is_fully_resolved() -> None:
    question = (
        "For all the people who paid more than 29.00 per unit of product ID No. 5, "
        "give their consumption status in August 2012."
    )
    contract = apply_structural_formulas(
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

    assert contract.outputs == ["yearmonth.CustomerID", "yearmonth.Consumption"]
    assert contract.measures == ["yearmonth.Consumption"]
    assert contract.aggregation is None
    assert contract.grain == ["CustomerID", "Month"]
    assert [item.operand for item in contract.filters] == [
        "transactions_1k.ProductID",
        "unit_price",
        "yearmonth.Date",
    ]
    assert resolution_report(question, contract).semantic_call_required is False


def test_metric_grain_planner_and_lineage_validator() -> None:
    question = (
        "What is the difference in the annual average consumption of the customers with the "
        "least amount of consumption paid in CZK for 2013 between SME and LAM, LAM and KAM, "
        "and KAM and SME?"
    )
    contract = plan_semantics(question)
    correct = """WITH customer_annual AS (
      SELECT c.Segment, y.CustomerID, AVG(y.Consumption) AS customer_annual_avg
      FROM yearmonth y JOIN customers c ON c.CustomerID=y.CustomerID
      WHERE c.Currency='CZK' AND SUBSTR(y.Date,1,4)='2013'
        AND c.Segment IN ('SME','LAM','KAM')
      GROUP BY c.Segment,y.CustomerID
    ), segment_low AS (
      SELECT Segment, MIN(customer_annual_avg) AS segment_min_avg
      FROM customer_annual GROUP BY Segment
    )
    SELECT MAX(CASE WHEN Segment='SME' THEN segment_min_avg END)-
           MAX(CASE WHEN Segment='LAM' THEN segment_min_avg END),
           MAX(CASE WHEN Segment='LAM' THEN segment_min_avg END)-
           MAX(CASE WHEN Segment='KAM' THEN segment_min_avg END),
           MAX(CASE WHEN Segment='KAM' THEN segment_min_avg END)-
           MAX(CASE WHEN Segment='SME' THEN segment_min_avg END)
    FROM segment_low"""
    wrong = correct.replace("AVG(y.Consumption)", "AVG(t.Amount * t.Price)").replace(
        "yearmonth y JOIN customers", "transactions_1k t JOIN customers"
    )

    assert contract.measures == ["yearmonth.Consumption"]
    assert contract.derived_metrics[0].name == "customer_annual_avg"
    assert contract.selectors[0].metric == "customer_annual_avg"
    assert validate_semantics(correct, contract, _valid()).valid is True
    rejected = validate_semantics(wrong, contract, _valid())
    assert rejected.valid is False
    assert rejected.code == "SEMANTIC_METRIC_LINEAGE_MISMATCH"


def test_percentage_change_dag_and_lineage_validator() -> None:
    question = (
        "Which EUR segment among SME, LAM, and KAM had the maximum and minimum "
        "percentage increase in consumption from 2012 to 2013?"
    )
    contract = plan_semantics(question)
    sql = """WITH yearly AS (
      SELECT c.Segment, SUBSTR(y.Date,1,4) AS year,
             SUM(y.Consumption) AS yearly_consumption
      FROM yearmonth y JOIN customers c ON c.CustomerID=y.CustomerID
      WHERE c.Currency='EUR' AND c.Segment IN ('SME','LAM','KAM')
        AND SUBSTR(y.Date,1,4) IN ('2012','2013')
      GROUP BY c.Segment, SUBSTR(y.Date,1,4)
    ), percentages AS (
      SELECT Segment,
        (SUM(CASE WHEN year='2013' THEN yearly_consumption ELSE 0 END) -
         SUM(CASE WHEN year='2012' THEN yearly_consumption ELSE 0 END)) * 100.0 /
         NULLIF(SUM(CASE WHEN year='2012' THEN yearly_consumption ELSE 0 END), 0)
         AS percentage_increase
      FROM yearly GROUP BY Segment
    ), ranked AS (
      SELECT Segment, percentage_increase,
        RANK() OVER (ORDER BY percentage_increase DESC) AS max_rank,
        RANK() OVER (ORDER BY percentage_increase ASC) AS min_rank
      FROM percentages
    )
    SELECT Segment, percentage_increase FROM ranked
    WHERE max_rank=1 OR min_rank=1
    ORDER BY percentage_increase DESC"""

    assert contract.aggregation_stages[0].group_by == ["customers.Segment", "year"]
    assert contract.formula_metrics[0].bindings["baseline"].endswith("[Year=2012]")
    assert [selector.function for selector in contract.selectors] == ["argmax", "argmin"]
    result = validate_semantics(sql, contract, _valid())
    assert result.valid is True
    assert "selectors:argmax_argmin" in result.semantic_checks


def test_percentage_change_rejects_unselected_extrema() -> None:
    question = (
        "Which EUR segment among SME, LAM, and KAM had the maximum and minimum "
        "percentage increase in consumption from 2012 to 2013?"
    )
    contract = plan_semantics(question)
    sql = """WITH yearly AS (
      SELECT c.Segment, SUBSTR(y.Date,1,4) AS year,
             SUM(y.Consumption) AS yearly_consumption
      FROM yearmonth y JOIN customers c ON c.CustomerID=y.CustomerID
      WHERE c.Currency='EUR' AND c.Segment IN ('SME','LAM','KAM')
        AND SUBSTR(y.Date,1,4) IN ('2012','2013')
      GROUP BY c.Segment, SUBSTR(y.Date,1,4)
    ), percentages AS (
      SELECT Segment,
        (SUM(CASE WHEN year='2013' THEN yearly_consumption ELSE 0 END) -
         SUM(CASE WHEN year='2012' THEN yearly_consumption ELSE 0 END)) * 100.0 /
         NULLIF(SUM(CASE WHEN year='2012' THEN yearly_consumption ELSE 0 END), 0)
         AS percentage_increase
      FROM yearly GROUP BY Segment
    ) SELECT Segment, percentage_increase FROM percentages
      ORDER BY percentage_increase DESC"""

    result = validate_semantics(sql, contract, _valid())
    assert result.valid is False
    assert result.code == "SEMANTIC_METRIC_LINEAGE_MISMATCH"

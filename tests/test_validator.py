from semantic_text2sql.models import SchemaInfo
from semantic_text2sql.validator import clean_model_sql, validate_sql


def test_validator_accepts_cte_output_alias(registry) -> None:  # type: ignore[no-untyped-def]
    schema = registry.inspect("shop")
    result = validate_sql(
        "WITH totals AS (SELECT customer_id, SUM(amount) AS total_amount "
        "FROM orders GROUP BY customer_id) "
        "SELECT customer_id, total_amount FROM totals WHERE total_amount > 50",
        schema,
    )

    assert result.valid is True


def test_validator_rejects_unknown_column(registry) -> None:  # type: ignore[no-untyped-def]
    result = validate_sql("SELECT missing FROM orders", registry.inspect("shop"))

    assert result.valid is False
    assert result.code == "SQL_UNAUTHORIZED_COLUMN"


def test_validator_rejects_writes_and_multiple_statements(registry) -> None:  # type: ignore[no-untyped-def]
    schema = registry.inspect("shop")

    assert validate_sql("DELETE FROM orders", schema).code == "SQL_NOT_READ_ONLY"
    assert validate_sql("SELECT 1; SELECT 2", schema).code == "SQL_MULTIPLE_STATEMENTS"


def test_clean_model_sql_removes_one_fence() -> None:
    assert clean_model_sql("```sql\nSELECT 1;\n```") == "SELECT 1"


def test_clean_model_sql_extracts_sql_fence_from_explanation() -> None:
    response = (
        "Here is the corrected query:\n\n"
        "```sql\nSELECT SUBSTR(y.Date, 1, 4) FROM yearmonth AS y;\n```\n\n"
        "This uses the stored YYYYMM format."
    )

    assert clean_model_sql(response) == "SELECT SUBSTR(y.Date, 1, 4) FROM yearmonth AS y"


def test_empty_schema_does_not_authorize_invented_table() -> None:
    result = validate_sql("SELECT * FROM invented", SchemaInfo(db_id="empty", tables=[]))

    assert result.code == "SQL_UNAUTHORIZED_TABLE"

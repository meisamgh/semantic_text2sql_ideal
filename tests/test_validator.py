import pytest

from semantic_text2sql.models import SchemaInfo
from semantic_text2sql.validator import clean_model_sql, validate_sql


@pytest.mark.parametrize("sql", [
    "WITH secret AS (SELECT customer_id FROM customers) SELECT * FROM analytics.secret",
    "WITH secret AS (SELECT * FROM secret) SELECT * FROM secret",
    "SELECT * FROM secret WHERE EXISTS "
    "(WITH secret AS (SELECT * FROM customers) SELECT * FROM secret)",
    "WITH a AS (SELECT * FROM secret), secret AS (SELECT * FROM customers) SELECT * FROM a",
    'WITH "SECRET" AS (SELECT * FROM customers) SELECT * FROM secret',
    "WITH secret AS (SELECT * FROM customers) SELECT * FROM customers "
    "WHERE EXISTS (SELECT 1 FROM private.secret)",
])
def test_cte_names_do_not_hide_physical_tables(registry, sql) -> None:  # type: ignore[no-untyped-def]
    result = validate_sql(sql, registry.inspect("shop"), dialect="postgres",
                          allowed_tables={"customers"})
    assert not result.valid
    assert result.code == "SQL_TABLE_NOT_AUTHORIZED"


@pytest.mark.parametrize("sql", [
    "WITH orders AS (SELECT * FROM orders) SELECT * FROM orders",
    "WITH a AS (SELECT * FROM orders), b AS (SELECT * FROM a) SELECT * FROM b",
    "WITH Orders AS (SELECT * FROM orders) SELECT * FROM ORDERS",
    "WITH orders AS (SELECT * FROM orders) SELECT * FROM analytics.orders",
    "WITH RECURSIVE n(value) AS (SELECT 1 UNION ALL SELECT value+1 FROM n WHERE value<3) "
    "SELECT value FROM n",
])
def test_visible_ctes_and_authorized_physical_tables_remain_valid(registry, sql) -> None:  # type: ignore[no-untyped-def]
    result = validate_sql(sql, registry.inspect("shop"), dialect="postgres",
                          allowed_tables={"orders"})
    assert result.valid, result
    assert set(result.tables) <= {"orders"}


def test_qualified_allowlist_does_not_authorize_another_schema(registry) -> None:  # type: ignore[no-untyped-def]
    schema = registry.inspect("shop")
    assert validate_sql("SELECT * FROM analytics.orders", schema,
                        allowed_tables={"analytics.orders"}).valid
    rejected = validate_sql("SELECT * FROM private.orders", schema,
                            allowed_tables={"analytics.orders"})
    assert rejected.code == "SQL_TABLE_NOT_AUTHORIZED"


def test_validator_accepts_cte_output_alias(registry) -> None:  # type: ignore[no-untyped-def]
    schema = registry.inspect("shop")
    result = validate_sql(
        "WITH totals AS (SELECT customer_id, SUM(amount) AS total_amount "
        "FROM orders GROUP BY customer_id) "
        "SELECT customer_id, total_amount FROM totals WHERE total_amount > 50",
        schema,
    )

    assert result.valid is True


def test_validator_leaves_unknown_column_to_execution(registry) -> None:  # type: ignore[no-untyped-def]
    result = validate_sql("SELECT missing FROM orders", registry.inspect("shop"))

    assert result.valid is True
    assert result.code == "SQL_SAFETY_VALID"


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


def test_validator_does_not_check_table_existence() -> None:
    result = validate_sql("SELECT * FROM invented", SchemaInfo(db_id="empty", tables=[]))

    assert result.valid is True
    assert result.code == "SQL_SAFETY_VALID"


def test_validator_enforces_explicit_table_authorization(registry) -> None:  # type: ignore[no-untyped-def]
    result = validate_sql(
        "SELECT name FROM customers",
        registry.inspect("shop"),
        allowed_tables={"orders"},
    )

    assert result.valid is False
    assert result.code == "SQL_TABLE_NOT_AUTHORIZED"

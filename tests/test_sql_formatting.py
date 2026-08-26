from semantic_text2sql.models import GenerateResponse, StrategyHints
from semantic_text2sql.sql_formatting import format_sql_for_display


def test_format_sql_for_display_pretty_prints_nested_query() -> None:
    sql = "SELECT CustomerID,COUNT(*) AS n FROM customers GROUP BY CustomerID ORDER BY n DESC"

    formatted = format_sql_for_display(sql, "sqlite")

    assert formatted is not None
    assert "SELECT\n  CustomerID," in formatted
    assert "GROUP BY\n  CustomerID" in formatted
    assert "ORDER BY\n  n DESC" in formatted


def test_generate_response_serializes_original_and_formatted_sql() -> None:
    sql = "SELECT CustomerID FROM customers WHERE Currency = 'EUR'"
    response = GenerateResponse(
        db_id="db",
        question="question",
        provider="ollama",
        model="model",
        strategy=StrategyHints(mode="exact"),
        sql=sql,
        accepted=True,
        execution_status="EXECUTABLE",
        termination_reason="accepted",
    )

    payload = response.model_dump()
    assert payload["sql"] == sql
    assert payload["formatted_sql"] == (
        "SELECT\n  CustomerID\nFROM customers\nWHERE\n  Currency = 'EUR'"
    )

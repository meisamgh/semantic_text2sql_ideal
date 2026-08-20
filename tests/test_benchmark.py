from semantic_text2sql.benchmark import compare_sql


def test_execution_comparison_accepts_equivalent_sql(registry) -> None:  # type: ignore[no-untyped-def]
    result = compare_sql(
        registry,
        "shop",
        "SELECT name FROM customers WHERE country = 'Germany'",
        "SELECT name FROM customers WHERE customer_id = 1",
    )

    assert result.executable is True
    assert result.equivalent is True


def test_execution_comparison_rejects_wrong_result(registry) -> None:  # type: ignore[no-untyped-def]
    result = compare_sql(
        registry,
        "shop",
        "SELECT name FROM customers WHERE country = 'Italy'",
        "SELECT name FROM customers WHERE country = 'Germany'",
    )

    assert result.executable is True
    assert result.equivalent is False

from semantic_text2sql.agent import _validate_profile_aware_sql
from semantic_text2sql.models import DatabaseProfile, RelationshipProfile, ValidationResult
from semantic_text2sql.profiling import profile_database


def _valid() -> ValidationResult:
    return ValidationResult(valid=True, code="SQL_VALID", message="valid")


def test_rejects_parent_measure_aggregated_after_one_to_many_join(registry) -> None:  # type: ignore[no-untyped-def]
    profile = profile_database(registry, "shop", "sqlite")
    result = _validate_profile_aware_sql(
        "SELECT SUM(c.customer_id) FROM customers c "
        "JOIN orders o ON o.customer_id = c.customer_id",
        profile,
        _valid(),
        dialect="sqlite",
        measures=["customers.customer_id"],
    )

    assert result.valid is False
    assert result.code == "JOIN_CARDINALITY_RISK"


def test_allows_child_measure_and_distinct_parent_count(registry) -> None:  # type: ignore[no-untyped-def]
    profile = profile_database(registry, "shop", "sqlite")
    child_measure = _validate_profile_aware_sql(
        "SELECT SUM(o.amount) FROM customers c JOIN orders o "
        "ON o.customer_id = c.customer_id",
        profile,
        _valid(),
        dialect="sqlite",
        measures=["orders.amount"],
    )
    parent_count = _validate_profile_aware_sql(
        "SELECT COUNT(DISTINCT c.customer_id) FROM customers c JOIN orders o "
        "ON o.customer_id = c.customer_id",
        profile,
        _valid(),
        dialect="sqlite",
        measures=["customers.customer_id"],
    )

    assert child_measure.valid is True
    assert parent_count.valid is True


def test_rejects_sibling_child_fanout() -> None:
    profile = DatabaseProfile(
        db_id="sales",
        dialect="sqlite",
        profiled_at="2026-01-01T00:00:00+00:00",
        columns=[],
        relationships=[
            RelationshipProfile(
                parent_table="customers",
                parent_column="id",
                child_table="orders",
                child_column="customer_id",
                type="ONE_TO_MANY",
                parent_key_unique=True,
                child_key_unique=False,
            ),
            RelationshipProfile(
                parent_table="customers",
                parent_column="id",
                child_table="payments",
                child_column="customer_id",
                type="ONE_TO_MANY",
                parent_key_unique=True,
                child_key_unique=False,
            ),
        ],
    )
    result = _validate_profile_aware_sql(
        "SELECT SUM(o.amount) FROM customers c "
        "JOIN orders o ON o.customer_id = c.id "
        "JOIN payments p ON p.customer_id = c.id",
        profile,
        _valid(),
        dialect="sqlite",
        measures=["orders.amount"],
    )

    assert result.code == "JOIN_CARDINALITY_RISK"

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from semantic_text2sql.agent import TextToSQLAgent
from semantic_text2sql.api import create_app
from semantic_text2sql.glossary import BusinessGlossary, GlossaryTerm
from semantic_text2sql.models import (
    ColumnProfile,
    ContextRequest,
    DatabaseProfile,
    ValueFrequency,
)
from semantic_text2sql.value_grounding import ground_question_values


def _profile() -> DatabaseProfile:
    return DatabaseProfile(
        db_id="billing",
        dialect="sqlite",
        profiled_at="2026-09-09T00:00:00Z",
        columns=[
            ColumnProfile(
                table="customers",
                column="Currency",
                database_type="TEXT",
                semantic_type="text",
                row_count=100,
                null_count=0,
                null_ratio=0,
                distinct_count=2,
                allowed_values=["CZK", "EUR"],
                top_values=[
                    ValueFrequency(value="CZK", count=80),
                    ValueFrequency(value="EUR", count=20),
                ],
            )
        ],
    )


def _segment_profile() -> DatabaseProfile:
    return DatabaseProfile(
        db_id="billing",
        dialect="sqlite",
        profiled_at="2026-09-09T00:00:00Z",
        columns=[
            ColumnProfile(
                table="customers",
                column="Segment",
                database_type="TEXT",
                semantic_type="text",
                row_count=100,
                null_count=0,
                null_ratio=0,
                distinct_count=3,
                allowed_values=["KAM", "LAM", "SME"],
                top_values=[
                    ValueFrequency(value="SME", count=70),
                    ValueFrequency(value="LAM", count=20),
                    ValueFrequency(value="KAM", count=10),
                ],
            )
        ],
    )


def _glossary() -> BusinessGlossary:
    return BusinessGlossary(
        db_id="billing",
        version="1",
        source="test",
        precedence=[],
        terms=[
            GlossaryTerm(
                term="customer currency",
                definition="Configured account currency.",
                synonyms=["paid in", "pay in", "euro", "CZK", "EUR"],
                value_aliases={"euro": "EUR", "koruna": "CZK"},
                columns=["customers.Currency"],
            )
        ],
    )


def _context() -> ContextRequest:
    return ContextRequest(
        tables=["customers"],
        columns={"customers": ["CustomerID", "Currency"]},
        business_concepts=["customer_currency"],
    )


def test_unavailable_explicit_category_requires_clarification() -> None:
    result = ground_question_values(
        "Which year had the most gas consumption paid in Dollar?",
        _context(),
        _profile(),
        _glossary(),
    )

    assert result.issue is not None
    assert result.issue.user_value == "Dollar"
    assert result.issue.column == "customers.Currency"
    assert result.issue.available_values == ["CZK", "EUR"]


def test_canonical_category_value_is_accepted() -> None:
    result = ground_question_values(
        "Which year had the most gas consumption paid in EUR?",
        _context(),
        _profile(),
        _glossary(),
    )

    assert result.issue is None
    assert result.evidence == []


def test_approved_alias_becomes_grounding_evidence() -> None:
    result = ground_question_values(
        "Which year had the most gas consumption paid in euro?",
        _context(),
        _profile(),
        _glossary(),
    )

    assert result.issue is None
    assert "means 'EUR'" in result.evidence[0]


def test_unknown_value_in_comparison_list_requires_clarification() -> None:
    glossary = BusinessGlossary(
        db_id="billing",
        version="1",
        source="test",
        precedence=[],
        terms=[
            GlossaryTerm(
                term="customer segment",
                definition="Customer classification.",
                synonyms=["SME", "LAM", "KAM"],
                columns=["customers.Segment"],
            )
        ],
    )
    context = ContextRequest(
        tables=["customers"],
        columns={"customers": ["CustomerID", "Segment"]},
    )

    result = ground_question_values(
        "Compare the annual change between OOS and LAM, LAM and KAM, and KAM and SME.",
        context,
        _segment_profile(),
        glossary,
    )

    assert result.issue is not None
    assert result.issue.user_value == "OOS"
    assert result.issue.column == "customers.Segment"
    assert result.issue.available_values == ["KAM", "LAM", "SME"]


class FailIfCalledModel:
    async def generate(self, **kwargs: Any) -> tuple[str, int]:
        raise AssertionError("Model 2 must not be called before value clarification.")


class ZeroRowsModel:
    async def generate(self, **kwargs: Any) -> tuple[str, int]:
        return "SELECT customer_id FROM customers WHERE country = 'Spain'", 7


class NullRowsModel:
    async def generate(self, **kwargs: Any) -> tuple[str, int]:
        return (
            "SELECT customer_id, NULL AS note FROM customers "
            "WHERE country = 'Germany' AND customer_id > 0",
            9,
        )


def test_api_clarifies_unknown_value_before_model_call(
    registry, monkeypatch, tmp_path  # type: ignore[no-untyped-def]
) -> None:
    profile_root = tmp_path / "profiles"
    glossary_root = tmp_path / "glossaries"
    profile_root.mkdir()
    glossary_root.mkdir()
    profile = _profile().model_copy(update={"db_id": "shop"})
    profile = profile.model_copy(
        update={
            "columns": [
                profile.columns[0].model_copy(
                    update={"table": "customers", "column": "country"}
                )
            ]
        }
    )
    glossary = _glossary().model_copy(update={"db_id": "shop"})
    glossary = glossary.model_copy(
        update={
            "terms": [
                glossary.terms[0].model_copy(
                    update={"columns": ["customers.country"]}
                )
            ]
        }
    )
    (profile_root / "sqlite__shop.json").write_text(profile.model_dump_json())
    (glossary_root / "shop.json").write_text(glossary.model_dump_json())
    monkeypatch.setenv("TEXT2SQL_DATABASE_ROOT", str(registry.root))
    monkeypatch.setenv("TEXT2SQL_PROFILE_ROOT", str(profile_root))
    monkeypatch.setenv("TEXT2SQL_GLOSSARY_ROOT", str(glossary_root))

    client = TestClient(create_app(TextToSQLAgent(registry, FailIfCalledModel())))
    response = client.post(
        "/api/chat",
        json={
            "session_id": "grounding-before-model",
            "db_id": "shop",
            "message": "Which year had the most consumption paid in Dollar?",
            "provider": "ollama",
            "model": "unused",
            "context_mode": "retrieval",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["clarification_required"] is True
    assert body["generation"]["attempts"] == []
    assert body["generation"]["grounding_issue"]["user_value"] == "Dollar"
    assert body["human_review"]["options"] == ["CZK", "EUR"]


def test_api_escalates_executable_zero_row_result_to_human_review(
    registry, monkeypatch, tmp_path  # type: ignore[no-untyped-def]
) -> None:
    monkeypatch.setenv("TEXT2SQL_DATABASE_ROOT", str(registry.root))
    monkeypatch.setenv("TEXT2SQL_PROFILE_ROOT", str(tmp_path / "profiles"))
    monkeypatch.setenv("TEXT2SQL_GLOSSARY_ROOT", str(tmp_path / "glossaries"))

    client = TestClient(create_app(TextToSQLAgent(registry, ZeroRowsModel())))
    response = client.post(
        "/api/chat",
        json={
            "session_id": "zero-result-review",
            "db_id": "shop",
            "message": "List customers in Spain",
            "provider": "ollama",
            "model": "unused",
            "context_mode": "retrieval",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["generation"]["accepted"] is True
    assert body["generation"]["row_count"] == 0
    assert body["human_review"]["question"] == (
        "Would you like to accept this result or change the request?"
    )
    assert "Accept no matching data" in body["human_review"]["options"]
    assert body["human_review"]["filter_checks"] == [
        {
            "filter": "country = 'Spain'",
            "plain_language": "country must equal Spain",
            "match_count": 0,
            "status": "NO_MATCH",
        }
    ]


def test_api_escalates_null_result_and_checks_all_filters(
    registry, monkeypatch, tmp_path  # type: ignore[no-untyped-def]
) -> None:
    monkeypatch.setenv("TEXT2SQL_DATABASE_ROOT", str(registry.root))
    monkeypatch.setenv("TEXT2SQL_PROFILE_ROOT", str(tmp_path / "profiles"))
    monkeypatch.setenv("TEXT2SQL_GLOSSARY_ROOT", str(tmp_path / "glossaries"))

    client = TestClient(create_app(TextToSQLAgent(registry, NullRowsModel())))
    response = client.post(
        "/api/chat",
        json={
            "session_id": "null-result-review",
            "db_id": "shop",
            "message": "List German customers and their optional note",
            "provider": "ollama",
            "model": "unused",
            "context_mode": "retrieval",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["generation"]["accepted"] is True
    assert body["human_review"]["question"] == (
        "Would you like to accept the missing value or review the request?"
    )
    evidence = " ".join(body["human_review"]["evidence"])
    assert "country = 'Germany'" in evidence
    assert "customer_id > 0" in evidence
    assert len(body["human_review"]["filter_checks"]) == 2

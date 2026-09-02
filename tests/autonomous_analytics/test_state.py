from __future__ import annotations

from datetime import UTC, datetime

import pytest

from autonomous_analytics.models.evidence import Evidence
from autonomous_analytics.models.investigation import (
    Hypothesis,
    InvestigationBudget,
    InvestigationState,
)


def state(**updates: object) -> InvestigationState:
    values: dict[str, object] = {
        "case_id": "case-1",
        "observation": "Signups increased while purchases remained flat.",
        "primary_metrics": ["signups", "purchases"],
    }
    values.update(updates)
    return InvestigationState.model_validate(values)


def test_investigation_budget_is_explicit_and_immutable_by_record_call() -> None:
    investigation = state(
        budget=InvestigationBudget(
            max_sql_queries=1,
            max_python_calls=1,
            max_resource_searches=0,
            max_total_tool_calls=2,
        )
    )

    after_sql = investigation.record_call("sql")

    assert investigation.sql_calls == 0
    assert after_sql.sql_calls == 1
    assert after_sql.can_call("sql") is False
    assert after_sql.can_call("python") is True
    assert after_sql.can_call("resource") is False
    with pytest.raises(ValueError, match="budget is exhausted"):
        after_sql.record_call("sql")


def test_state_separates_rejected_hypotheses_and_preserves_evidence() -> None:
    evidence = Evidence(
        evidence_id="ev-1",
        source_type="metric",
        question="How did purchases change?",
        result_summary={"change_7d": 0.02},
        created_at=datetime(2026, 1, 8, tzinfo=UTC),
    )
    rejected = Hypothesis(
        hypothesis_id="h-1",
        statement="The signup increase already translated to purchases.",
        status="rejected",
        contradicting_evidence_ids=["ev-1"],
    )

    investigation = state(evidence=[evidence], rejected_hypotheses=[rejected])

    assert investigation.evidence[0].result_summary["change_7d"] == 0.02
    assert investigation.rejected_hypotheses[0].status == "rejected"


def test_state_rejects_hypothesis_in_both_partitions() -> None:
    active = Hypothesis(hypothesis_id="h-1", statement="Campaign mix changed.")
    rejected = active.model_copy(update={"status": "rejected"})

    with pytest.raises(ValueError, match="active and rejected"):
        state(hypotheses=[active], rejected_hypotheses=[rejected])

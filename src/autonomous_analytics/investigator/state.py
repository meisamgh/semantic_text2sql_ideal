"""Small state transition helpers shared by future investigator orchestration."""

from autonomous_analytics.models.investigation import InvestigationState
from autonomous_analytics.models.tools import ToolResult


def record_text_to_sql_result(
    state: InvestigationState, result: ToolResult
) -> InvestigationState:
    """Charge one SQL call and retain both successful and failed-call evidence."""

    updated = state.record_call("sql")
    if result.evidence is None:
        return updated
    return updated.model_copy(update={"evidence": [*updated.evidence, result.evidence]})

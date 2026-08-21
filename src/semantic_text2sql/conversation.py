"""Explicit session state and deterministic conversational-query resolution."""

from __future__ import annotations

import json
import re
from threading import Lock
from typing import Any, Literal, Protocol, cast

from pydantic import ValidationError

from semantic_text2sql.models import (
    ContractDelta,
    ConversationState,
    ModelProvider,
    TokenUsage,
    TurnInterpretation,
)

Operation = Literal[
    "NEW_QUERY",
    "REFINE",
    "OPTIMIZE",
    "ADD_FILTER",
    "REMOVE_FILTER",
    "CHANGE_METRIC",
    "CHANGE_GRAIN",
    "COMPARE",
    "CORRECTION",
    "EXPLAIN",
    "RESET_CONTEXT",
]
_FOLLOWUP = re.compile(
    r"\b(now|also|only|instead|same|those|them|it|previous|again|what about|how about)\b",
    re.I,
)
_EXPLICIT_FOLLOWUP_START = re.compile(
    r"^(?:and\s+)?(?:now|also|instead|only|same|remove|without|exclude|drop|compare|"
    r"what about|how about)\b",
    re.I,
)
_STANDALONE_START = re.compile(
    r"^(?:what|which|who|how|why|when|where|is|are|do|does|did|can|could|would|"
    r"please|list|state|give|among|for all|in\s+(?:19|20)\d{2})\b",
    re.I,
)
_CORRECTION = re.compile(
    r"\b(that(?:'s| is) wrong|wrong result|i meant|should be|incorrect|"
    r"not correct|correct solution|the solution|use this sql|instead of)\b",
    re.I,
)
_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.I | re.S)


class ConversationCompleter(Protocol):
    async def complete(self, model: str, prompt: str) -> str: ...


class ConversationStore:
    """Process-local state; replace with durable storage for multi-worker production."""

    def __init__(self) -> None:
        self._states: dict[str, ConversationState] = {}
        self._lock = Lock()

    def get(self, session_id: str) -> ConversationState | None:
        with self._lock:
            return self._states.get(session_id)

    def put(self, state: ConversationState) -> None:
        with self._lock:
            self._states[state.session_id] = state

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._states.pop(session_id, None)


def classify_operation(
    message: str, has_state: bool, feedback_category: str | None = None
) -> Operation:
    value = " ".join(message.casefold().split())
    if re.search(r"\b(reset|start over|start again|clear context|new conversation)\b", value):
        return "RESET_CONTEXT"
    if not has_state:
        return "NEW_QUERY"
    if feedback_category or _CORRECTION.search(value):
        return "CORRECTION"
    if re.search(r"^(?:why|explain|how did you|show provenance|which tables)\b", value):
        return "EXPLAIN"
    if re.search(r"\b(optimi[sz]e|more efficient|improve performance|faster query)\b", value):
        return "OPTIMIZE"
    if _EXPLICIT_FOLLOWUP_START.search(value):
        pass
    elif _STANDALONE_START.search(value) or not _FOLLOWUP.search(value):
        return "NEW_QUERY"
    if re.search(r"\b(remove|without|exclude|drop)\b", value):
        return "REMOVE_FILTER"
    if re.search(r"\b(compare|versus|vs\.?|difference)\b", value):
        return "COMPARE"
    if re.search(r"\b(group by|per |each |grain|break down)\b", value):
        return "CHANGE_GRAIN"
    if re.search(r"\b(metric|instead calculate|instead show|change to)\b", value):
        return "CHANGE_METRIC"
    if re.search(r"\b(only|where|for |after|before|in 20\d\d)\b", value):
        return "ADD_FILTER"
    return "REFINE"


def requires_model_interpretation(
    message: str, has_state: bool, feedback_category: str | None = None
) -> bool:
    """Use a model only when rules cannot confidently identify a stateful turn."""
    if not has_state or feedback_category:
        return False
    value = " ".join(message.casefold().split())
    if re.search(r"\b(reset|start over|start again|clear context|new conversation)\b", value):
        return False
    if _CORRECTION.search(value):
        return False
    if re.search(r"^(?:why|explain|how did you|show provenance|which tables)\b", value):
        return False
    if re.search(r"\b(optimi[sz]e|more efficient|improve performance|faster query)\b", value):
        return False
    return not (_EXPLICIT_FOLLOWUP_START.search(value) or _STANDALONE_START.search(value))


async def interpret_turn_detailed(
    completer: ConversationCompleter,
    *,
    provider: ModelProvider,
    model: str,
    message: str,
    previous: ConversationState,
) -> tuple[TurnInterpretation, TokenUsage]:
    """Resolve an ambiguous conversational turn without asking the model for SQL."""
    prompt = f"""Classify one message in a conversational text-to-SQL application.
Return exactly one JSON object:
{{"operation":"CORRECTION","depends_on_previous":true,
"resolved_instruction":"...","correction_type":null,"confidence":0.0}}

Allowed operation values: NEW_QUERY, REFINE, OPTIMIZE, ADD_FILTER, REMOVE_FILTER,
CHANGE_METRIC, CHANGE_GRAIN, COMPARE, CORRECTION, EXPLAIN, RESET_CONTEXT.

Rules:
- This call must not generate SQL.
- NEW_QUERY means the message is an independent analytical question.
- CORRECTION means the user says the previous interpretation/result/SQL is wrong or supplies
  a solution that should repair it.
- REFINE and the ADD/REMOVE/CHANGE operations modify the previous accepted request.
- If the message contains SQL as a proposed fix, preserve it verbatim in resolved_instruction
  and classify it as CORRECTION.
- resolved_instruction must state the user's request clearly without inventing requirements.
- depends_on_previous must be false only for NEW_QUERY or RESET_CONTEXT.

Previous root question: {previous.root_question}
Previous resolved request: {previous.resolved_question}
Previous accepted SQL: {previous.last_sql or "None"}
Current message: {message}
"""
    detailed = getattr(completer, "complete_detailed", None)
    if callable(detailed):
        raw, usage = await cast(Any, detailed)(model, prompt)
    else:
        raw = await completer.complete(model, prompt)
        usage = TokenUsage()
    raw = raw.strip()
    match = _FENCE.fullmatch(raw)
    value = match.group(1) if match else raw
    try:
        parsed = TurnInterpretation.model_validate(json.loads(value))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError("Conversation model returned an invalid turn interpretation.") from exc
    return parsed.model_copy(
        update={"source": "model", "provider": provider, "model": model}
    ), usage


def resolve_turn(
    session_id: str,
    db_id: str,
    message: str,
    previous: ConversationState | None,
    feedback_category: str | None = None,
    interpreted: TurnInterpretation | None = None,
) -> tuple[Operation, ConversationState | None]:
    operation: Operation = (
        interpreted.operation
        if interpreted is not None
        else classify_operation(
            message,
            previous is not None and previous.db_id == db_id,
            feedback_category,
        )
    )
    effective_message = interpreted.resolved_instruction if interpreted else message
    if operation == "RESET_CONTEXT":
        return operation, None
    if operation == "NEW_QUERY" or previous is None or previous.db_id != db_id:
        state = ConversationState(
            session_id=session_id,
            db_id=db_id,
            root_question=effective_message,
            resolved_question=effective_message,
            turn_count=1,
        )
        return "NEW_QUERY", state
    if operation == "EXPLAIN":
        return operation, previous
    correction_type = feedback_category or (interpreted.correction_type if interpreted else None)
    correction = (
        f"Correction category={correction_type or 'other'}: {effective_message}"
        if operation == "CORRECTION"
        else effective_message
    )
    modifications = [*previous.modifications, correction]
    delta = ContractDelta(
        operation=operation,
        instruction=effective_message,
        feedback_category=feedback_category,
    )
    instructions = "\n".join(f"{index}. {item}" for index, item in enumerate(modifications, 1))
    resolved = (
        f"Base request: {previous.root_question}\n"
        "Conversation modifications in chronological order; later instructions override earlier "
        f"ones:\n{instructions}"
    )
    return operation, previous.model_copy(
        update={
            "modifications": modifications,
            "resolved_question": resolved,
            "turn_count": previous.turn_count + 1,
            "corrections": [
                *previous.corrections,
                *([correction] if operation == "CORRECTION" else []),
            ],
            "contract_deltas": [*previous.contract_deltas, delta],
        }
    )

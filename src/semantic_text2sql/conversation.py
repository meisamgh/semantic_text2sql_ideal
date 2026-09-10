"""Explicit session state and deterministic conversational-query resolution."""

from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path
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
    "EXPLAIN_SQL",
    "EXPLAIN_RESULT",
    "EXPLAIN_INTERPRETATION",
    "EXPLAIN_CONTEXT",
    "EXPLAIN_FAILURE",
    "CHECK_CORRECTNESS",
    "RESET_CONTEXT",
]
EXPLANATION_OPERATIONS = frozenset(
    {
        "EXPLAIN",
        "EXPLAIN_SQL",
        "EXPLAIN_RESULT",
        "EXPLAIN_INTERPRETATION",
        "EXPLAIN_CONTEXT",
        "EXPLAIN_FAILURE",
    }
)
STATE_REQUIRED_OPERATIONS = frozenset(
    {
        "REFINE",
        "OPTIMIZE",
        "ADD_FILTER",
        "REMOVE_FILTER",
        "CHANGE_METRIC",
        "CHANGE_GRAIN",
        "COMPARE",
        "CORRECTION",
        "CHECK_CORRECTNESS",
        *EXPLANATION_OPERATIONS,
    }
)
_FOLLOWUP = re.compile(
    r"\b(now|also|only|instead|same|this|that|these|those|them|it|previous|again|"
    r"what about|how about)\b",
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
_EXPLICIT_VALUE_REPLACEMENT = re.compile(
    r'^\s*Replace filter value "(?P<old>[^"]+)" with "(?P<new>[^"]+)"\.\s*$',
    re.I,
)


class ConversationCompleter(Protocol):
    async def complete(self, model: str, prompt: str) -> str: ...


class ConversationStore:
    """Versioned conversation state with optional durable SQLite persistence."""

    def __init__(self, path: Path | None = None, *, ttl_seconds: int = 86_400) -> None:
        self._states: dict[str, ConversationState] = {}
        self._lock = Lock()
        self.path = path
        self.ttl_seconds = max(60, ttl_seconds)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(path) as connection:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS conversations ("
                    "session_id TEXT PRIMARY KEY, version INTEGER NOT NULL, "
                    "state_json TEXT NOT NULL, updated_at REAL NOT NULL)"
                )

    def get(self, session_id: str) -> ConversationState | None:
        with self._lock:
            if self.path is not None:
                with sqlite3.connect(self.path) as connection:
                    connection.execute(
                        "DELETE FROM conversations WHERE updated_at < ?",
                        (time.time() - self.ttl_seconds,),
                    )
                    row = connection.execute(
                        "SELECT state_json FROM conversations WHERE session_id = ?",
                        (session_id,),
                    ).fetchone()
                return ConversationState.model_validate_json(row[0]) if row else None
            return self._states.get(session_id)

    def put(
        self, state: ConversationState, *, expected_version: int | None = None
    ) -> ConversationState:
        with self._lock:
            current = self._read_unlocked(state.session_id)
            current_version = current.version if current else None
            if expected_version != current_version:
                raise ConversationConflict(
                    f"Conversation changed from version {expected_version} to {current_version}."
                )
            saved = state.model_copy(update={"version": (current_version or 0) + 1})
            if self.path is None:
                self._states[state.session_id] = saved
            else:
                with sqlite3.connect(self.path) as connection:
                    connection.execute(
                        "INSERT INTO conversations(session_id, version, state_json, updated_at) "
                        "VALUES (?, ?, ?, ?) ON CONFLICT(session_id) DO UPDATE SET "
                        "version=excluded.version, state_json=excluded.state_json, "
                        "updated_at=excluded.updated_at",
                        (
                            saved.session_id,
                            saved.version,
                            saved.model_dump_json(),
                            time.time(),
                        ),
                    )
            return saved

    def reset(self, session_id: str) -> None:
        with self._lock:
            if self.path is None:
                self._states.pop(session_id, None)
            else:
                with sqlite3.connect(self.path) as connection:
                    connection.execute(
                        "DELETE FROM conversations WHERE session_id = ?", (session_id,)
                    )

    def _read_unlocked(self, session_id: str) -> ConversationState | None:
        if self.path is None:
            return self._states.get(session_id)
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                "SELECT state_json FROM conversations WHERE session_id = ?", (session_id,)
            ).fetchone()
        return ConversationState.model_validate_json(row[0]) if row else None


class ConversationConflict(RuntimeError):
    """Raised when a stale asynchronous turn attempts to replace newer state."""


def classify_operation(
    message: str, has_state: bool, feedback_category: str | None = None
) -> Operation:
    value = " ".join(message.casefold().split())
    if re.search(r"\b(reset|start over|start again|clear context|new conversation)\b", value):
        return "RESET_CONTEXT"
    if feedback_category or _CORRECTION.search(value):
        return "CORRECTION"
    if has_state and re.search(
        r"\b(check|verify|validate|review)\b.*\b(correct|correctness|accurate|accuracy)\b|"
        r"\bis (?:this|the) (?:sql|query|answer|result) correct\b",
        value,
    ):
        return "CHECK_CORRECTNESS"
    explanation = _classify_explanation(value)
    if explanation is not None:
        return explanation
    if re.search(r"\b(optimi[sz]e|more efficient|improve performance|faster query)\b", value):
        return "OPTIMIZE"
    if not has_state:
        return "NEW_QUERY"
    if _EXPLICIT_FOLLOWUP_START.search(value):
        pass
    elif not _FOLLOWUP.search(value):
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
    if _classify_explanation(value) is not None:
        return False
    if re.search(r"\b(optimi[sz]e|more efficient|improve performance|faster query)\b", value):
        return False
    if _EXPLICIT_FOLLOWUP_START.search(value):
        return False
    return not (_STANDALONE_START.search(value) and not _FOLLOWUP.search(value))


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
"resolved_instruction":"...","correction_type":null,"target":null,"confidence":0.0}}

Allowed operation values: NEW_QUERY, REFINE, OPTIMIZE, ADD_FILTER, REMOVE_FILTER,
CHANGE_METRIC, CHANGE_GRAIN, COMPARE, CORRECTION, EXPLAIN_SQL, EXPLAIN_RESULT,
EXPLAIN_INTERPRETATION, EXPLAIN_CONTEXT, EXPLAIN_FAILURE, CHECK_CORRECTNESS, RESET_CONTEXT.

Rules:
- This call must not generate SQL.
- NEW_QUERY means the message is an independent analytical question.
- CORRECTION means the user says the previous interpretation/result/SQL is wrong or supplies
  a solution that should repair it.
- REFINE and the ADD/REMOVE/CHANGE operations modify the previous accepted request.
- EXPLAIN_SQL asks how or why the accepted SQL uses a join, CTE, filter, aggregation, ordering,
  DISTINCT, window function, or other SQL construct.
- EXPLAIN_RESULT asks why the executed result has particular values, rows, NULLs, duplicates,
  missing rows, or an empty result.
- EXPLAIN_INTERPRETATION asks how the analytical request was understood.
- EXPLAIN_CONTEXT asks which tables, columns, relationships, glossary facts, or metadata were used.
- EXPLAIN_FAILURE asks why generation, validation, model access, or execution failed.
- CHECK_CORRECTNESS asks for an evidence-based review of the previously accepted SQL/result.
- If the message contains SQL as a proposed fix, preserve it verbatim in resolved_instruction
  and classify it as CORRECTION.
- resolved_instruction must state the user's request clearly without inventing requirements.
- target should identify the requested aspect when clear, for example JOIN_STRATEGY, RESULT_ROWS,
  INTERPRETATION, CONTEXT, FAILURE, or PERFORMANCE; otherwise use null.
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
    if operation in EXPLANATION_OPERATIONS:
        return operation, previous
    if operation == "CHECK_CORRECTNESS":
        return operation, previous
    replacement = _EXPLICIT_VALUE_REPLACEMENT.fullmatch(effective_message)
    if operation == "CORRECTION" and replacement:
        old_value = replacement.group("old")
        new_value = replacement.group("new")
        updated_question, count = re.subn(
            rf"(?<!\w){re.escape(old_value)}(?!\w)",
            new_value,
            previous.resolved_question,
            flags=re.I,
        )
        if count:
            correction = f"Replaced filter value {old_value!r} with {new_value!r}."
            delta = ContractDelta(
                operation=operation,
                instruction=correction,
                feedback_category=feedback_category,
            )
            return operation, previous.model_copy(
                update={
                    "root_question": re.sub(
                        rf"(?<!\w){re.escape(old_value)}(?!\w)",
                        new_value,
                        previous.root_question,
                        flags=re.I,
                    ),
                    "resolved_question": updated_question,
                    "turn_count": previous.turn_count + 1,
                    "modifications": [*previous.modifications, correction],
                    "corrections": [*previous.corrections, correction],
                    "contract_deltas": [*previous.contract_deltas, delta],
                }
            )
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


def _classify_explanation(value: str) -> Operation | None:
    if not re.search(
        r"^(?:(?:can|could|would) you (?:please )?|please )?(?:tell me )?"
        r"(?:why|explain|how did you|how was|show provenance|which tables|which columns|"
        r"what does (?:this|the) (?:query|sql) do)\b",
        value,
    ):
        return None
    if re.search(r"\b(fail(?:ed|ure)?|error|rejected|unavailable|timeout|did not run)\b", value):
        return "EXPLAIN_FAILURE"
    if re.search(
        r"\b(result|rows?|values?|null|empty|duplicate|missing|returned|calculated)\b",
        value,
    ):
        return "EXPLAIN_RESULT"
    if re.search(r"\b(interpret|understand|meaning|assume|assumption|request)\b", value):
        return "EXPLAIN_INTERPRETATION"
    if re.search(
        r"\b(tables?|columns?|schema|context|metadata|glossary|relationship|provenance)\b",
        value,
    ):
        return "EXPLAIN_CONTEXT"
    return "EXPLAIN_SQL"


def intent_target(message: str, operation: Operation) -> str | None:
    """Attach a compact, non-semantic focus label for downstream explanation prompts."""
    value = message.casefold()
    if operation == "OPTIMIZE":
        return "PERFORMANCE"
    if operation == "EXPLAIN_FAILURE":
        return "FAILURE"
    if operation == "EXPLAIN_RESULT":
        return "RESULT_ROWS"
    if operation == "EXPLAIN_INTERPRETATION":
        return "INTERPRETATION"
    if operation == "EXPLAIN_CONTEXT":
        return "CONTEXT"
    if operation == "EXPLAIN_SQL":
        if "join" in value:
            return "JOIN_STRATEGY"
        if "distinct" in value:
            return "DISTINCT"
        if "group" in value or "aggregation" in value:
            return "AGGREGATION"
        if "filter" in value or "where" in value:
            return "FILTER"
        if "order" in value or "limit" in value or "top" in value:
            return "ORDERING"
        return "SQL_STRUCTURE"
    return None

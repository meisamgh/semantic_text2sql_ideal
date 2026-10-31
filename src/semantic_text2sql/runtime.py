"""Shared bounded-runtime primitives for one Text-to-SQL request."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Lock
from time import monotonic
from typing import TypeVar

T = TypeVar("T")

class RequestBudgetExceeded(TimeoutError):
    pass


@dataclass
class RequestBudget:
    timeout_seconds: float
    max_model_calls: int = 6
    max_database_calls: int = 12
    started_at: float = field(default_factory=monotonic)
    model_calls: int = 0
    database_calls: int = 0
    _lock: Lock = field(default_factory=Lock, repr=False)

    def remaining(self) -> float:
        return max(0.0, self.started_at + self.timeout_seconds - monotonic())

    async def wait(self, operation: Awaitable[T], *, kind: str) -> T:
        try:
            self.claim(kind)
        except RequestBudgetExceeded:
            _close_unstarted(operation)
            raise
        remaining = self.remaining()
        if remaining <= 0:
            _close_unstarted(operation)
            raise RequestBudgetExceeded("The request deadline was exhausted.")
        try:
            return await asyncio.wait_for(operation, timeout=remaining)
        except RequestBudgetExceeded:
            raise
        except TimeoutError as exc:
            raise RequestBudgetExceeded("The request deadline was exhausted.") from exc

    def claim(self, kind: str) -> None:
        """Atomically reserve one real provider or database operation."""
        with self._lock:
            if kind == "model":
                cap = _MODEL_CALL_CAP.get()
                if cap is not None and cap[0] == id(self) and self.model_calls >= cap[1]:
                    raise RequestBudgetExceeded("The specialist model-call cap was exhausted.")
                if self.model_calls >= self.max_model_calls:
                    raise RequestBudgetExceeded("The request model-call budget was exhausted.")
                self.model_calls += 1
            elif kind == "database":
                if self.database_calls >= self.max_database_calls:
                    raise RequestBudgetExceeded("The request database-call budget was exhausted.")
                self.database_calls += 1
            if self.remaining() <= 0:
                raise RequestBudgetExceeded("The request deadline was exhausted.")


def _close_unstarted(operation: Awaitable[object]) -> None:
    close = getattr(operation, "close", None)
    if callable(close):
        close()


_ACTIVE_BUDGET: ContextVar[RequestBudget | None] = ContextVar(
    "text2sql_active_request_budget", default=None
)
_MODEL_CALL_CAP: ContextVar[tuple[int, int] | None] = ContextVar("model_call_cap", default=None)


@contextmanager
def model_call_cap(budget: RequestBudget, maximum: int) -> Iterator[None]:
    """Limit a specialist's calls, including provider retries, on the same budget."""
    token = _MODEL_CALL_CAP.set((id(budget), budget.model_calls + maximum))
    try:
        yield
    finally:
        _MODEL_CALL_CAP.reset(token)


def active_request_budget() -> RequestBudget | None:
    return _ACTIVE_BUDGET.get()


@contextmanager
def use_request_budget(budget: RequestBudget) -> Iterator[None]:
    token = _ACTIVE_BUDGET.set(budget)
    try:
        yield
    finally:
        _ACTIVE_BUDGET.reset(token)

"""Shared bounded-runtime primitives for one Text-to-SQL request."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from dataclasses import dataclass, field
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

    def remaining(self) -> float:
        return max(0.0, self.started_at + self.timeout_seconds - monotonic())

    async def wait(self, operation: Awaitable[T], *, kind: str) -> T:
        if kind == "model":
            self.model_calls += 1
            if self.model_calls > self.max_model_calls:
                _close_unstarted(operation)
                raise RequestBudgetExceeded("The request model-call budget was exhausted.")
        elif kind == "database":
            self.database_calls += 1
            if self.database_calls > self.max_database_calls:
                _close_unstarted(operation)
                raise RequestBudgetExceeded("The request database-call budget was exhausted.")
        remaining = self.remaining()
        if remaining <= 0:
            _close_unstarted(operation)
            raise RequestBudgetExceeded("The request deadline was exhausted.")
        try:
            return await asyncio.wait_for(operation, timeout=remaining)
        except TimeoutError as exc:
            raise RequestBudgetExceeded("The request deadline was exhausted.") from exc


def _close_unstarted(operation: Awaitable[object]) -> None:
    close = getattr(operation, "close", None)
    if callable(close):
        close()

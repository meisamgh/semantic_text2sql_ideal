"""In-memory governed KPI registry with deterministic integrity checks."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from pydantic import TypeAdapter

from autonomous_analytics.models.kpi import KPIDefinition


class KPIRegistryError(ValueError):
    """Raised when registry definitions are ambiguous or internally inconsistent."""


class KPIRegistry:
    def __init__(self, definitions: Iterable[KPIDefinition] = ()) -> None:
        self._definitions: dict[str, KPIDefinition] = {}
        for definition in definitions:
            self.register(definition)

    def register(self, definition: KPIDefinition, *, replace: bool = False) -> None:
        if definition.name in self._definitions and not replace:
            raise KPIRegistryError(f"KPI {definition.name!r} is already registered")
        self._definitions[definition.name] = definition

    def get(self, name: str) -> KPIDefinition:
        try:
            return self._definitions[name]
        except KeyError as exc:
            raise KPIRegistryError(f"KPI {name!r} is not registered") from exc

    def list_definitions(self) -> list[KPIDefinition]:
        return [self._definitions[name] for name in sorted(self._definitions)]

    def related(self, name: str) -> list[KPIDefinition]:
        definition = self.get(name)
        names = {
            *definition.upstream_metrics,
            *definition.downstream_metrics,
            *(item.target for item in definition.expected_relationships),
        }
        return [self._definitions[item] for item in sorted(names) if item in self._definitions]

    def validate_relationships(self) -> None:
        missing: list[str] = []
        for definition in self._definitions.values():
            references = {
                *definition.upstream_metrics,
                *definition.downstream_metrics,
                *(item.target for item in definition.expected_relationships),
            }
            missing.extend(
                f"{definition.name}->{target}"
                for target in sorted(references)
                if target not in self._definitions
            )
        if missing:
            raise KPIRegistryError("Unknown KPI relationships: " + ", ".join(missing))

    @classmethod
    def from_json(cls, path: Path) -> KPIRegistry:
        raw = json.loads(path.read_text(encoding="utf-8"))
        definitions = TypeAdapter(list[KPIDefinition]).validate_python(raw)
        registry = cls(definitions)
        registry.validate_relationships()
        return registry

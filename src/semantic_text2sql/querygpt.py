"""QueryGPT-inspired workspace, intent, table proposal, and column-pruning flow."""

from __future__ import annotations

import json
import re
from pathlib import Path

from semantic_text2sql.database import DatabaseRegistry
from semantic_text2sql.linker import select_schema
from semantic_text2sql.models import (
    IntentRequest,
    IntentResponse,
    SchemaInfo,
    TableProposalRequest,
    TableProposalResponse,
    WorkspaceInfo,
    WorkspaceMatch,
)
from semantic_text2sql.postgres import PostgresRegistry
from semantic_text2sql.profiling import ProfileStore

_TOKEN = re.compile(r"[A-Za-z0-9]+")


class WorkspaceRegistry:
    def __init__(self, path: Path | None = None) -> None:
        self.workspaces: dict[str, WorkspaceInfo] = {}
        if path and path.is_file():
            raw = json.loads(path.read_text(encoding="utf-8"))
            for item in raw:
                workspace = WorkspaceInfo.model_validate(item)
                self.workspaces[workspace.workspace_id] = workspace

    def for_database(self, db_id: str, schema: SchemaInfo) -> list[WorkspaceInfo]:
        matches = [item for item in self.workspaces.values() if db_id in item.db_ids]
        if matches:
            return matches
        return [
            WorkspaceInfo(
                workspace_id=f"database:{db_id}",
                name=db_id.replace("_", " ").title(),
                description=f"All documented tables in the {db_id} database.",
                db_ids=[db_id],
                tables=[table.name for table in schema.tables],
                terms=[],
                system=False,
            )
        ]

    def get(self, workspace_id: str, db_id: str, schema: SchemaInfo) -> WorkspaceInfo:
        for workspace in self.for_database(db_id, schema):
            if workspace.workspace_id == workspace_id:
                return workspace
        raise ValueError(f"Workspace {workspace_id} is not available for database {db_id}.")


class QueryGPTFlow:
    def __init__(
        self,
        sqlite: DatabaseRegistry,
        postgres: PostgresRegistry | None,
        profiles: ProfileStore,
        workspaces: WorkspaceRegistry,
    ) -> None:
        self.sqlite = sqlite
        self.postgres = postgres
        self.profiles = profiles
        self.workspaces = workspaces

    def _schema(self, db_id: str, dialect: str = "sqlite") -> SchemaInfo:
        if dialect == "postgres" and self.postgres:
            return self.postgres.inspect(db_id)
        return self.sqlite.inspect(db_id)

    def classify_intent(self, request: IntentRequest) -> IntentResponse:
        schema = self._schema(request.db_id)
        query = _tokens(f"{request.question} {request.evidence or ''}")
        ranked: list[WorkspaceMatch] = []
        for workspace in self.workspaces.for_database(request.db_id, schema):
            description_tokens = _tokens(
                f"{workspace.name} {workspace.description} {' '.join(workspace.terms)}"
            )
            table_tokens = _tokens(" ".join(workspace.tables))
            semantic_hits = sorted(query & description_tokens)
            table_hits = sorted(query & table_tokens)
            score = 2.0 * len(semantic_hits) + len(table_hits)
            reasons = []
            if semantic_hits:
                reasons.append("Matched workspace terms: " + ", ".join(semantic_hits))
            if table_hits:
                reasons.append("Matched table terms: " + ", ".join(table_hits))
            if not reasons:
                reasons.append("Database fallback workspace.")
            ranked.append(
                WorkspaceMatch(
                    workspace_id=workspace.workspace_id,
                    name=workspace.name,
                    score=score,
                    reasons=reasons,
                )
            )
        ranked.sort(key=lambda item: (-item.score, item.workspace_id))
        enhanced = _enhance(request.question, request.evidence, ranked[0].name)
        return IntentResponse(
            original_question=request.question,
            enhanced_question=enhanced,
            matches=ranked,
            selected_workspace=ranked[0].workspace_id,
        )

    def propose_tables(self, request: TableProposalRequest) -> TableProposalResponse:
        schema = self._schema(request.db_id)
        intent = self.classify_intent(
            IntentRequest(
                db_id=request.db_id,
                question=request.question,
                evidence=request.evidence,
            )
        )
        workspace_id = request.workspace_id or intent.selected_workspace
        workspace = self.workspaces.get(workspace_id, request.db_id, schema)
        allowed = (
            set(workspace.tables) if workspace.tables else {table.name for table in schema.tables}
        )
        scoped = schema.model_copy(
            update={
                "tables": [table for table in schema.tables if table.name in allowed],
                "relationships": [
                    item
                    for item in schema.relationships
                    if item.from_table in allowed and item.to_table in allowed
                ],
            }
        )
        profile = self.profiles.load(schema.dialect, request.db_id)
        _, selection = select_schema(
            intent.enhanced_question,
            request.evidence,
            scoped,
            profile,
            max_tables=request.max_tables,
            max_columns_per_table=8,
        )
        selected = set(selection.tables)
        return TableProposalResponse(
            workspace_id=workspace_id,
            proposed_tables=selection.tables,
            proposed_columns=selection.columns,
            relationships=[
                item
                for item in scoped.relationships
                if item.from_table in selected and item.to_table in selected
            ],
        )


def _enhance(question: str, evidence: str | None, workspace: str) -> str:
    value = " ".join(question.split())
    suffix = f" Business workspace: {workspace}."
    if evidence:
        suffix += " Apply the supplied evidence exactly; do not invent additional filters."
    return value + suffix


def _tokens(value: str) -> set[str]:
    return {token.casefold() for token in _TOKEN.findall(value)}

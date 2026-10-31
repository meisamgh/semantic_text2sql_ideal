"""Permanent, versioned dashboards in the existing history SQLite database.

Only server-validated source snapshots enter this store. It has no SQL executor,
provider, credentials or refresh capability.
"""

from __future__ import annotations

import os
import re
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field

from semantic_text2sql.ai_dashboard_designer import eligible_catalog
from semantic_text2sql.dashboard_service import DashboardError, DashboardRequest, DashboardSpec
from semantic_text2sql.dialects import DatabaseDialect
from semantic_text2sql.models import (
    ChartSpec,
    FilterCatalog,
    ModelProvider,
    SemanticContract,
    StrategyHints,
    StrictModel,
)
from semantic_text2sql.result_cache import RecentResult


@dataclass(frozen=True)
class DashboardPrincipal:
    subject: str
    authenticated: bool = False


class DashboardSnapshot(StrictModel):
    result_id: str
    run_id: str | None = None
    db_id: str
    question: str
    sql: str = Field(max_length=20000)
    dialect: DatabaseDialect
    provider: ModelProvider = "agentrouter"
    model: str = "unknown"
    columns: list[str]
    rows: list[list[Any]] = Field(max_length=500)
    row_count: int = Field(ge=0, le=500)
    execution_timestamp: float | None = None
    semantic_contract: SemanticContract | None = None
    provenance: list[str] = Field(default_factory=list)
    # Deliberately excludes provider configuration, arbitrary model context,
    # historical examples and database connection strings.
    table_grains: dict[str, str] = Field(default_factory=dict)
    chart_spec: ChartSpec | None = None
    filter_catalog: FilterCatalog | None = None

    @classmethod
    def capture(cls, source: RecentResult) -> DashboardSnapshot:
        if any(
            re.search(
                r"password|api[_-]?key|access[_-]?token|refresh[_-]?token|"
                r"client[_-]?secret|private[_-]?key|authorization",
                name,
                re.I,
            )
            for name in source.columns
        ):
            raise DashboardError("Credential-like result fields cannot be saved.")
        if (
            not source.result_id
            or source.execution_status != "ACCEPTED"
            or source.truncated
            or len(source.rows) != source.row_count
        ):
            raise DashboardError("Only complete accepted results can be saved.")
        return cls(
            result_id=source.result_id,
            run_id=source.run_id,
            db_id=source.db_id,
            question=source.question,
            sql=source.sql,
            dialect=source.dialect,
            provider=source.provider,
            model=source.model,
            columns=source.columns,
            rows=source.rows,
            row_count=source.row_count,
            execution_timestamp=source.execution_timestamp,
            semantic_contract=source.semantic_contract,
            provenance=source.provenance,
            table_grains={
                name: str(table.get("grain", "unknown"))
                for name, table in source.verified_context.get("tables", {}).items()
                if isinstance(table, dict)
            },
            chart_spec=source.chart_spec,
            filter_catalog=source.filter_catalog,
        )

    def result(self) -> RecentResult:
        return RecentResult(
            run_id=self.run_id,
            result_id=self.result_id,
            chart_id=None,
            db_id=self.db_id,
            question=self.question,
            sql=self.sql,
            columns=self.columns,
            rows=self.rows,
            row_count=self.row_count,
            truncated=False,
            provider=self.provider,
            model=self.model,
            dialect=self.dialect,
            strategy=StrategyHints(mode="exact"),
            chart_spec=self.chart_spec,
            summary="Saved analytical snapshot; not refreshed.",
            provenance=self.provenance,
            execution_timestamp=self.execution_timestamp,
            semantic_contract=self.semantic_contract,
            filter_catalog=self.filter_catalog,
            verified_context={
                "tables": {name: {"grain": grain} for name, grain in self.table_grains.items()}
            },
        )


class DashboardDocument(StrictModel):
    spec: DashboardSpec
    source: DashboardSnapshot

    def verify(self) -> None:
        """Recheck retained evidence, not merely JSON shape, without a DB call."""
        source = self.source.result()
        request = DashboardRequest(
            session_id="saved", db_id=source.db_id, result_id=self.source.result_id
        )
        base, metrics, charts = eligible_catalog(source, request)
        if (
            self.spec.result_id != self.source.result_id
            or self.spec.db_id != source.db_id
            or self.spec.row_count != source.row_count
            or self.spec.run_id != source.run_id
            or self.spec.execution_timestamp != source.execution_timestamp
            or self.spec.applied_constraints != base.applied_constraints
        ):
            raise DashboardError("Dashboard and snapshot evidence do not agree.")
        if any(
            field not in base.verified_filter_metadata
            for field in self.spec.verified_filter_metadata
        ):
            raise DashboardError("Saved filter metadata is not supported by its source.")
        if self.spec.data and (
            self.spec.data.columns != source.columns or self.spec.data.rows != source.rows
        ):
            raise DashboardError("Saved table differs from its source snapshot.")
        for kpi in self.spec.kpis:
            actual = kpi.model_dump(exclude={"kpi_id", "column_span"})
            if not any(
                actual == m.model_dump(exclude={"kpi_id", "column_span"}) for m in metrics.values()
            ):
                raise DashboardError("A saved KPI is not supported by its source.")
        for widget in self.spec.widgets:
            actual_chart = widget.chart_spec.model_dump(exclude={"title", "x_label", "y_label"})
            if not any(
                actual_chart == w.chart_spec.model_dump(exclude={"title", "x_label", "y_label"})
                for w in charts.values()
            ):
                raise DashboardError("A saved chart is not supported by its source.")


class SavedDashboard(StrictModel):
    dashboard_id: str
    owner: str
    name: str
    version: int
    status: Literal["Draft", "Published"]
    published_version: int | None = None
    created_at: float
    updated_at: float
    permission: Literal["owner", "editor", "viewer"] = "owner"
    document: DashboardDocument


# Migration is additive and runs atomically alongside query_runs; no history rows
# are modified. Explicit versioning allows later additive migrations.
_MIGRATION = (
    "CREATE TABLE IF NOT EXISTS dashboard_studio_migrations (version INTEGER PRIMARY KEY)",
    "CREATE TABLE IF NOT EXISTS saved_dashboards ("
    "id TEXT PRIMARY KEY, owner TEXT NOT NULL, name TEXT NOT NULL, version INTEGER NOT NULL, "
    "status TEXT NOT NULL, published_version INTEGER, created_at REAL NOT NULL, "
    "updated_at REAL NOT NULL)",
    "CREATE INDEX IF NOT EXISTS saved_dashboards_owner ON saved_dashboards(owner, updated_at)",
    "CREATE TABLE IF NOT EXISTS saved_dashboard_revisions ("
    "dashboard_id TEXT NOT NULL REFERENCES saved_dashboards(id) ON DELETE CASCADE, "
    "version INTEGER NOT NULL, payload TEXT NOT NULL, action TEXT NOT NULL, "
    "actor TEXT NOT NULL, created_at REAL NOT NULL, parent_version INTEGER, "
    "PRIMARY KEY(dashboard_id,version))",
    "CREATE TABLE IF NOT EXISTS saved_dashboard_access ("
    "dashboard_id TEXT NOT NULL REFERENCES saved_dashboards(id) ON DELETE CASCADE, "
    "subject TEXT NOT NULL, role TEXT NOT NULL CHECK(role IN ('viewer','editor')), "
    "PRIMARY KEY(dashboard_id,subject))",
    "INSERT OR IGNORE INTO dashboard_studio_migrations(version) VALUES(1)",
)


class DashboardStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        # New DBs are private from creation; existing history files are tightened.
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(descriptor)
        path.chmod(0o600)
        with self._transaction() as db:
            for statement in _MIGRATION:
                db.execute(statement)
            columns = {row[1] for row in db.execute("PRAGMA table_info(saved_dashboard_revisions)")}
            if "name" not in columns:
                db.execute(
                    "ALTER TABLE saved_dashboard_revisions ADD COLUMN name TEXT NOT NULL DEFAULT ''"
                )
                db.execute(
                    "UPDATE saved_dashboard_revisions SET name=(SELECT name FROM "
                    "saved_dashboards WHERE id=dashboard_id)"
                )
            db.execute("INSERT OR IGNORE INTO dashboard_studio_migrations(version) VALUES(2)")

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with closing(sqlite3.connect(self.path, timeout=5)) as db, db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys = ON")
            db.execute("BEGIN IMMEDIATE")
            yield db

    @staticmethod
    def _access(
        db: sqlite3.Connection,
        identity: str,
        principal: DashboardPrincipal,
        *,
        edit: bool = False,
        owner: bool = False,
    ) -> sqlite3.Row:
        row = db.execute(
            "SELECT d.*, CASE WHEN d.owner=? THEN 'owner' ELSE a.role END AS permission "
            "FROM saved_dashboards d LEFT JOIN saved_dashboard_access a "
            "ON a.dashboard_id=d.id AND a.subject=? WHERE d.id=?",
            (principal.subject, principal.subject, identity),
        ).fetchone()
        if row is None or row["permission"] is None:
            raise DashboardError("Dashboard is unavailable.", 404)
        if (
            owner
            and row["permission"] != "owner"
            or edit
            and row["permission"] not in {"owner", "editor"}
        ):
            raise DashboardError("This dashboard is read-only for your account.", 403)
        # ACLs are inert in local workspace mode; only authenticated identities
        # may receive delegated access when a deployment integrates identity.
        if row["permission"] != "owner" and not principal.authenticated:
            raise DashboardError("Authenticated access is required.", 403)
        assert isinstance(row, sqlite3.Row)
        return row

    @staticmethod
    def _name(value: str) -> str:
        if not value.strip() or len(value) > 160:
            raise DashboardError("Use a dashboard name between one and 160 characters.", 422)
        return value.strip()

    @staticmethod
    def _payload(document: DashboardDocument) -> str:
        document.verify()
        payload = document.model_dump_json()
        if len(payload.encode()) > 4_000_000:
            raise DashboardError("This dashboard exceeds the saved snapshot size limit.", 413)
        return payload

    def create(
        self, principal: DashboardPrincipal, name: str, document: DashboardDocument
    ) -> SavedDashboard:
        payload, name = self._payload(document), self._name(name)
        with self._transaction() as db:
            identity = self._insert(db, principal, name, payload)
        return self.open(principal, identity)

    @staticmethod
    def _insert(
        db: sqlite3.Connection, principal: DashboardPrincipal, name: str, payload: str
    ) -> str:
        identity, now = uuid4().hex, time()
        db.execute(
            "INSERT INTO saved_dashboards VALUES(?,?,?,?,?,?,?,?)",
            (identity, principal.subject, name, 1, "Draft", None, now, now),
        )
        db.execute(
            "INSERT INTO saved_dashboard_revisions VALUES(?,?,?,?,?,?,?,?)",
            (identity, 1, payload, "create", principal.subject, now, None, name),
        )
        return identity

    def copy(
        self,
        principal: DashboardPrincipal,
        identity: str,
        expected: int,
        *,
        name: str,
        document: DashboardDocument,
        source_version: int | None = None,
    ) -> SavedDashboard:
        payload = self._payload(document)
        with self._transaction() as db:
            head = self._access(db, identity, principal, edit=True)
            if head["version"] != expected:
                raise DashboardError("Dashboard changed; reopen before Save as New.", 409)
            row = db.execute(
                "SELECT payload FROM saved_dashboard_revisions WHERE dashboard_id=? AND version=?",
                (identity, source_version or expected),
            ).fetchone()
            if row is None:
                raise DashboardError("Source version is unavailable.", 404)
            original = DashboardDocument.model_validate_json(row["payload"])
            if (
                document.source != original.source
                or source_version is not None
                and document != original
            ):
                raise DashboardError("The copy does not match its saved source version.")
            copied = self._insert(db, principal, self._name(name), payload)
        return self.open(principal, copied)

    def list_dashboards(self, principal: DashboardPrincipal) -> list[dict[str, Any]]:
        with self._transaction() as db:
            rows = db.execute(
                "SELECT d.*,a.role,p.name AS published_name,p.created_at AS published_at "
                "FROM saved_dashboards d LEFT JOIN saved_dashboard_access a "
                "ON a.dashboard_id=d.id AND a.subject=? LEFT JOIN saved_dashboard_revisions p "
                "ON p.dashboard_id=d.id AND p.version=d.published_version "
                "WHERE d.owner=? OR (? AND (a.role='editor' OR "
                "(a.role='viewer' AND d.published_version IS NOT NULL))) "
                "ORDER BY d.updated_at DESC LIMIT 100",
                (principal.subject, principal.subject, principal.authenticated),
            ).fetchall()
            items = []
            for row in rows:
                item = {
                    k: value
                    for k, value in dict(row).items()
                    if k not in {"role", "published_name", "published_at"}
                }
                if row["owner"] != principal.subject and row["role"] == "viewer":
                    item.update(
                        name=row["published_name"],
                        version=row["published_version"],
                        status="Published",
                        updated_at=row["published_at"],
                    )
                items.append(item)
            return items

    def open(
        self, principal: DashboardPrincipal, identity: str, version: int | None = None
    ) -> SavedDashboard:
        with self._transaction() as db:
            header = self._access(db, identity, principal)
            target = header["version"] if version is None else version
            if header["permission"] == "viewer" and version is None:
                target = header["published_version"]
            row = db.execute(
                "SELECT payload,name,action,created_at FROM saved_dashboard_revisions "
                "WHERE dashboard_id=? AND version=?",
                (identity, target),
            ).fetchone()
            if row is None or header["permission"] == "viewer" and row["action"] != "publish":
                raise DashboardError("Dashboard version is unavailable.", 404)
            document = DashboardDocument.model_validate_json(row["payload"])
            document.verify()
            return SavedDashboard(
                dashboard_id=identity,
                owner=header["owner"],
                name=row["name"],
                version=target,
                status="Published" if row["action"] == "publish" else "Draft",
                published_version=header["published_version"],
                created_at=header["created_at"],
                updated_at=row["created_at"],
                permission=header["permission"],
                document=document,
            )

    def revise(
        self,
        principal: DashboardPrincipal,
        identity: str,
        expected: int,
        *,
        document: DashboardDocument | None = None,
        name: str | None = None,
        action: Literal["save", "rename", "publish", "undo"] = "save",
        approved: bool = False,
    ) -> SavedDashboard:
        payload = self._payload(document) if document is not None else None
        with self._transaction() as db:
            head = self._access(db, identity, principal, edit=True, owner=action == "publish")
            if head["version"] != expected:
                raise DashboardError("Dashboard changed; reopen it before saving.", 409)
            if action == "publish" and not approved:
                raise DashboardError("Publishing requires explicit approval.", 422)
            current = db.execute(
                "SELECT * FROM saved_dashboard_revisions WHERE dashboard_id=? AND version=?",
                (identity, expected),
            ).fetchone()
            parent = expected
            next_name = self._name(name) if name is not None else head["name"]
            if action == "undo":
                if current["parent_version"] is None:
                    raise DashboardError("No earlier revision is available.")
                target = db.execute(
                    "SELECT * FROM saved_dashboard_revisions WHERE dashboard_id=? AND version=?",
                    (identity, current["parent_version"]),
                ).fetchone()
                payload, parent = target["payload"], target["parent_version"]
                next_name = target["name"]
            payload = payload or current["payload"]
            if document is None:
                # Validate before writing, including approval/undo of an older
                # revision under current validators. A failed validation rolls back.
                DashboardDocument.model_validate_json(payload).verify()
            # Same source required for Save; changed analyses need Save as New.
            if (
                document
                and document.source
                != DashboardDocument.model_validate_json(current["payload"]).source
            ):
                raise DashboardError("A different source analysis requires Save as New.")
            version, now = expected + 1, time()
            db.execute(
                "INSERT INTO saved_dashboard_revisions VALUES(?,?,?,?,?,?,?,?)",
                (identity, version, payload, action, principal.subject, now, parent, next_name),
            )
            db.execute(
                "UPDATE saved_dashboards SET version=?,name=?,status=?,published_version=?,"
                "updated_at=? WHERE id=? AND version=?",
                (
                    version,
                    next_name,
                    "Published" if action == "publish" else "Draft",
                    version if action == "publish" else head["published_version"],
                    now,
                    identity,
                    expected,
                ),
            )
        return self.open(principal, identity, version)

    def history(self, principal: DashboardPrincipal, identity: str) -> list[dict[str, Any]]:
        with self._transaction() as db:
            header = self._access(db, identity, principal)
            return [
                dict(row)
                for row in db.execute(
                    "SELECT version,action,actor,created_at,parent_version "
                    "FROM saved_dashboard_revisions WHERE dashboard_id=? "
                    "AND (? != 'viewer' OR action='publish') "
                    "ORDER BY version DESC LIMIT 100",
                    (identity, header["permission"]),
                )
            ]

    def delete(self, principal: DashboardPrincipal, identity: str, expected: int) -> None:
        with self._transaction() as db:
            head = self._access(db, identity, principal, owner=True)
            if head["version"] != expected:
                raise DashboardError("Dashboard changed; reopen it before deleting.", 409)
            db.execute("DELETE FROM saved_dashboards WHERE id=?", (identity,))

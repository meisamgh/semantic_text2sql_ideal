from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from test_dashboard import snapshot

from semantic_text2sql.dashboard_service import DashboardError, DashboardRequest, build_dashboard
from semantic_text2sql.dashboard_store import (
    DashboardDocument,
    DashboardPrincipal,
    DashboardSnapshot,
    DashboardStore,
)


def document():
    source = snapshot()
    spec = build_dashboard(
        DashboardRequest(session_id="owner", db_id="sales", result_id="result", include_data=True),
        lambda *_: source,
    )
    return DashboardDocument(spec=spec, source=DashboardSnapshot.capture(source))


def test_restart_preserves_snapshot_sql_filters_and_spec(tmp_path):
    path = tmp_path / "history.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE query_runs(id TEXT)")
        db.execute("INSERT INTO query_runs VALUES('untouched')")
    owner = DashboardPrincipal("alice", True)
    saved = DashboardStore(path).create(owner, "Revenue", document())
    reopened = DashboardStore(path).open(owner, saved.dashboard_id)
    assert reopened == saved
    assert reopened.document.source.sql == snapshot().sql
    assert reopened.document.source.rows == snapshot().rows
    assert reopened.document.spec.section_order == ("filters", "kpis", "charts", "data")
    assert path.stat().st_mode & 0o777 == 0o600
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT * FROM query_runs").fetchone() == ("untouched",)
        assert db.execute(
            "SELECT version FROM dashboard_studio_migrations ORDER BY version"
        ).fetchall() == [(1,), (2,)]


def test_owner_isolation_and_explicit_publication(tmp_path):
    store = DashboardStore(tmp_path / "history.sqlite")
    owner, other = DashboardPrincipal("alice", True), DashboardPrincipal("bob", True)
    saved = store.create(owner, "Revenue", document())
    assert store.list_dashboards(other) == []
    for operation in [
        lambda: store.open(other, saved.dashboard_id),
        lambda: store.history(other, saved.dashboard_id),
        lambda: store.revise(other, saved.dashboard_id, 1, name="Hacked"),
        lambda: store.delete(other, saved.dashboard_id, 1),
    ]:
        with pytest.raises(DashboardError) as error:
            operation()
        assert error.value.status_code == 404
    with pytest.raises(DashboardError):
        store.revise(owner, saved.dashboard_id, 1, action="publish")
    published = store.revise(owner, saved.dashboard_id, 1, action="publish", approved=True)
    assert published.status == "Published" and published.published_version == 2
    renamed = store.revise(owner, saved.dashboard_id, 2, action="rename", name="Executive view")
    assert renamed.status == "Draft" and renamed.published_version == 2
    assert store.open(owner, saved.dashboard_id, 2).status == "Published"
    undone = store.revise(owner, saved.dashboard_id, 3, action="undo")
    assert undone.name == "Revenue" and undone.version == 4 and undone.status == "Draft"
    assert store.history(owner, saved.dashboard_id)[0]["action"] == "undo"


def test_optimistic_concurrency_across_store_instances(tmp_path):
    path = tmp_path / "history.sqlite"
    store = DashboardStore(path)
    owner = DashboardPrincipal("owner", True)
    saved = store.create(owner, "Original", document())

    def update(name):
        try:
            return DashboardStore(path).revise(owner, saved.dashboard_id, 1, name=name).version
        except DashboardError as error:
            return error.status_code

    with ThreadPoolExecutor(2) as executor:
        assert sorted(executor.map(update, ["First", "Second"])) == [2, 409]
    assert len(store.history(owner, saved.dashboard_id)) == 2
    with pytest.raises(DashboardError):
        store.delete(owner, saved.dashboard_id, 1)


def test_acl_viewer_editor_and_owner_permissions(tmp_path):
    path = tmp_path / "history.sqlite"
    store, owner = DashboardStore(path), DashboardPrincipal("owner", True)
    saved = store.create(owner, "Revenue", document())
    # Trusted administrative fixture only. The app exposes NO sharing/grant API.
    with sqlite3.connect(path) as db:
        db.executemany(
            "INSERT INTO saved_dashboard_access VALUES(?,?,?)",
            [(saved.dashboard_id, "viewer", "viewer"), (saved.dashboard_id, "editor", "editor")],
        )
    viewer, editor = DashboardPrincipal("viewer", True), DashboardPrincipal("editor", True)
    assert store.list_dashboards(viewer) == []
    with pytest.raises(DashboardError):
        store.open(viewer, saved.dashboard_id)
    store.revise(owner, saved.dashboard_id, 1, action="publish", approved=True)
    assert store.open(viewer, saved.dashboard_id).permission == "viewer"
    with pytest.raises(DashboardError):
        store.revise(viewer, saved.dashboard_id, 1)
    with pytest.raises(DashboardError):
        store.revise(editor, saved.dashboard_id, 1, action="publish", approved=True)
    assert store.revise(editor, saved.dashboard_id, 2, name="New title").version == 3
    assert store.open(viewer, saved.dashboard_id).version == 2
    assert store.list_dashboards(viewer)[0]["name"] == "Revenue"
    assert len(store.history(viewer, saved.dashboard_id)) == 1
    with pytest.raises(DashboardError):
        store.open(viewer, saved.dashboard_id, 3)
    with pytest.raises(DashboardError):
        store.open(DashboardPrincipal("viewer", False), saved.dashboard_id)
    store.delete(owner, saved.dashboard_id, 3)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM saved_dashboard_revisions").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM saved_dashboard_access").fetchone()[0] == 0


def test_snapshot_excludes_configuration_and_rejects_incomplete_or_credentials():
    source = replace(
        snapshot(),
        verified_context={
            "api_key": "MUST_NOT_PERSIST",
            "tables": {"sales": {"grain": "one row per sale_id", "password": "MUST_NOT_PERSIST"}},
        },
    )
    assert "MUST_NOT_PERSIST" not in DashboardSnapshot.capture(source).model_dump_json()
    with pytest.raises(DashboardError):
        DashboardSnapshot.capture(replace(source, truncated=True))
    with pytest.raises(DashboardError):
        DashboardSnapshot.capture(replace(source, columns=["api_key"]))


def test_tampered_kpi_and_source_references_are_rejected(tmp_path):
    store, owner = DashboardStore(tmp_path / "history.sqlite"), DashboardPrincipal("owner")
    bad = document().model_copy(deep=True)
    bad.spec.kpis[0] = bad.spec.kpis[0].model_copy(update={"value": 999999})
    with pytest.raises(DashboardError):
        store.create(owner, "Fake KPI", bad)
    bad = document().model_copy(deep=True)
    bad.source.rows[0][-1] = 999999
    with pytest.raises(DashboardError):
        store.create(owner, "Fake chart", bad)


def test_source_changes_require_save_as_new_and_delete_is_permanent(tmp_path):
    store, owner = DashboardStore(tmp_path / "history.sqlite"), DashboardPrincipal("owner")
    saved = store.create(owner, "Revenue", document())
    changed = document()
    changed.source.provenance = ["a different extraction"]
    with pytest.raises(DashboardError):
        store.revise(owner, saved.dashboard_id, 1, document=changed)
    copied = store.create(owner, "A separate snapshot", changed)
    assert copied.dashboard_id != saved.dashboard_id and copied.version == 1
    store.delete(owner, saved.dashboard_id, 1)
    with pytest.raises(DashboardError):
        store.open(owner, saved.dashboard_id)


def test_multiple_undo_steps_keep_monotonic_versions(tmp_path):
    store, owner = DashboardStore(tmp_path / "history.sqlite"), DashboardPrincipal("owner")
    saved = store.create(owner, "Original", document())
    store.revise(owner, saved.dashboard_id, 1, name="Second")
    store.revise(owner, saved.dashboard_id, 2, name="Third")
    assert store.revise(owner, saved.dashboard_id, 3, action="undo").name == "Second"
    assert store.revise(owner, saved.dashboard_id, 4, action="undo").name == "Original"
    with pytest.raises(DashboardError):
        store.revise(owner, saved.dashboard_id, 5, action="undo")
    with sqlite3.connect(store.path) as db:
        assert (
            json.loads(
                db.execute("SELECT payload FROM saved_dashboard_revisions LIMIT 1").fetchone()[0]
            )["source"]["result_id"]
            == "result"
        )


@pytest.mark.parametrize("row_count", [0, 1, 21, 100, 500])
def test_complete_snapshots_are_not_history_previews(tmp_path, row_count):
    source = replace(
        snapshot(),
        rows=[[f"{2020 + i // 12}{i % 12 + 1:02d}", "DE", float(i)] for i in range(row_count)],
        row_count=row_count,
        chart_spec=None,
    )
    spec = build_dashboard(
        DashboardRequest(session_id="owner", db_id="sales", result_id="result", include_data=True),
        lambda *_: source,
    )
    doc = DashboardDocument(spec=spec, source=DashboardSnapshot.capture(source))
    path, principal = tmp_path / "history.sqlite", DashboardPrincipal("owner")
    saved = DashboardStore(path).create(principal, "A full snapshot", doc)
    restored = DashboardStore(path).open(principal, saved.dashboard_id)
    assert restored.document.source.rows == source.rows
    assert restored.document.spec.data.rows == source.rows


def test_invalid_retained_evidence_cannot_be_published_or_partially_committed(tmp_path):
    path = tmp_path / "history.sqlite"
    store, owner = DashboardStore(path), DashboardPrincipal("owner")
    saved = store.create(owner, "Revenue", document())
    broken = document().model_dump(mode="json")
    broken["spec"]["kpis"][0]["value"] = 999999
    with sqlite3.connect(path) as db:
        db.execute(
            "UPDATE saved_dashboard_revisions SET payload=? WHERE dashboard_id=?",
            (json.dumps(broken), saved.dashboard_id),
        )
    with pytest.raises(DashboardError):
        store.revise(owner, saved.dashboard_id, 1, action="publish", approved=True)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT version,status FROM saved_dashboards").fetchone() == (1, "Draft")
        assert db.execute("SELECT COUNT(*) FROM saved_dashboard_revisions").fetchone()[0] == 1

"""Scoped permanent dashboard HTTP API; never accepts SQL or authoritative specs."""

from __future__ import annotations

import asyncio
import os
import sqlite3
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from ipaddress import ip_address
from typing import Any, TypeVar
from urllib.parse import urlsplit
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import Field, ValidationError
from starlette.authentication import BaseUser

from semantic_text2sql.ai_dashboard_designer import DashboardRevisions
from semantic_text2sql.dashboard_service import DashboardError
from semantic_text2sql.dashboard_store import (
    DashboardDocument,
    DashboardPrincipal,
    DashboardSnapshot,
    DashboardStore,
    SavedDashboard,
)
from semantic_text2sql.models import ChatResponse, StrictModel
from semantic_text2sql.result_cache import RecentResult

T = TypeVar("T")


def dashboard_principal(request: Request) -> DashboardPrincipal:
    """Only trusted ASGI authentication may assert a remote identity.

    No bearer/header/owner/session ID is interpreted as a login. Local mode is
    explicitly a single OS-user workspace, not simulated production auth.
    """
    if request.headers.get("sec-fetch-site") == "cross-site":
        raise HTTPException(403, "Cross-site dashboard access is not allowed.")
    origin = request.headers.get("origin")
    if origin and (
        urlsplit(origin).netloc != request.url.netloc
        or urlsplit(origin).scheme != request.url.scheme
    ):
        raise HTTPException(403, "Cross-origin dashboard access is not allowed.")
    user = request.scope.get("user")
    if isinstance(user, BaseUser) and user.is_authenticated:
        try:
            subject = user.identity
        except NotImplementedError as exc:
            raise HTTPException(401, "A stable authenticated subject is required.") from exc
        if not subject:
            raise HTTPException(401, "Authenticated subject is unavailable.")
        return DashboardPrincipal("auth:" + subject, authenticated=True)
    try:
        local = bool(request.client and ip_address(request.client.host).is_loopback)
    except ValueError:
        local = False
    if (
        os.environ.get("TEXT2SQL_DASHBOARD_LOCAL_WORKSPACE", "true").lower() == "true"
        and local
        and request.url.hostname in {"localhost", "127.0.0.1", "::1"}
        and not any(
            header in request.headers
            for header in {"forwarded", "x-forwarded-for", "x-forwarded-host", "x-real-ip"}
        )
    ):
        return DashboardPrincipal("local-workspace")
    raise HTTPException(401, "Dashboard Studio requires authenticated access or local loopback.")


class StudioCreate(StrictModel):
    session_id: str = Field(min_length=1, max_length=100)
    db_id: str = Field(min_length=1, max_length=100)
    result_id: str = Field(min_length=1, max_length=100)
    design_revision_id: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=160)


class StudioChange(StrictModel):
    expected_version: int = Field(ge=1)
    name: str | None = Field(default=None, min_length=1, max_length=160)
    preview_id: str | None = Field(default=None, max_length=100)
    save_as_new: bool = False
    approve: bool = False
    source_version: int | None = Field(default=None, ge=1)


class StudioEdit(StrictModel):
    expected_version: int = Field(ge=1)
    message: str = Field(min_length=1, max_length=4000)
    preview_id: str | None = Field(default=None, max_length=100)
    widget_id: str | None = Field(default=None, max_length=100)


@dataclass(frozen=True)
class _Preview:
    owner: str
    dashboard_id: str
    version: int
    document: DashboardDocument


def studio_router(
    store: DashboardStore | None,
    revisions: DashboardRevisions,
    source_resolver: Callable[[str, str, str], RecentResult | None],
    edit: Callable[[SavedDashboard, str], Awaitable[ChatResponse]],
    source_bundle: Callable[[str, str, str], dict[str, RecentResult]] | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/api/studio")
    previews: dict[str, _Preview] = {}

    def access(request: Request, response: Response) -> tuple[DashboardStore, DashboardPrincipal]:
        response.headers["Cache-Control"] = "no-store, private"
        response.headers["Vary"] = "Cookie, Authorization"
        principal = dashboard_principal(request)
        if store is None:
            raise HTTPException(503, "Permanent dashboards require a file-backed history database.")
        return store, principal

    def checked(operation: Callable[[], T]) -> T:
        try:
            return operation()
        except DashboardError as exc:
            raise HTTPException(exc.status_code, str(exc)) from exc
        except ValidationError as exc:
            raise HTTPException(409, "Saved dashboard evidence could not be validated.") from exc
        except sqlite3.Error as exc:
            raise HTTPException(503, "Dashboard storage is temporarily unavailable.") from exc

    @router.get("")
    def library(request: Request, response: Response) -> list[dict[str, Any]]:
        db, principal = access(request, response)
        return checked(lambda: db.list_dashboards(principal))

    @router.post("", response_model=SavedDashboard)
    async def create(body: StudioCreate, request: Request, response: Response) -> SavedDashboard:
        db, principal = access(request, response)
        if principal.authenticated and body.session_id not in request.scope.get(
            "queryflow.authorized_sessions", ()
        ):
            raise HTTPException(403, "Authenticated session ownership is not configured.")
        spec = revisions.get(body.session_id, body.db_id, body.result_id)
        source = source_resolver(body.session_id, body.db_id, body.result_id)
        if spec is None or source is None:
            raise HTTPException(404, "The validated dashboard source is unavailable.")
        if spec.revision_id != body.design_revision_id:
            raise HTTPException(409, "The design changed; reopen it before saving.")
        return await asyncio.to_thread(
            checked,
            lambda: db.create(
                principal,
                body.name,
                DashboardDocument(
                    spec=spec,
                    source=DashboardSnapshot.capture(source),
                    sources=(
                        {
                            key: DashboardSnapshot.capture(item)
                            for key, item in source_bundle(
                                body.session_id, body.db_id, body.result_id
                            ).items()
                        }
                        if source_bundle and spec.schema_version == 2
                        else {}
                    ),
                ),
            ),
        )

    @router.get("/{identity}", response_model=SavedDashboard)
    def open_dashboard(
        identity: str, request: Request, response: Response, version: int | None = None
    ) -> SavedDashboard:
        db, principal = access(request, response)
        return checked(lambda: db.open(principal, identity, version))

    @router.get("/{identity}/versions")
    def history(identity: str, request: Request, response: Response) -> list[dict[str, Any]]:
        db, principal = access(request, response)
        return checked(lambda: db.history(principal, identity))

    @router.post("/{identity}/edit")
    async def edit_dashboard(
        identity: str, body: StudioEdit, request: Request, response: Response
    ) -> dict[str, Any]:
        db, principal = access(request, response)
        current = await asyncio.to_thread(checked, lambda: db.open(principal, identity))
        if current.permission == "viewer":
            raise HTTPException(403, "This dashboard is read-only for your account.")
        if current.version != body.expected_version:
            raise HTTPException(409, "Dashboard changed; reopen it before editing.")
        if body.preview_id:
            prior = previews.get(body.preview_id)
            if (
                prior is None
                or prior.owner != principal.subject
                or prior.dashboard_id != identity
                or prior.version != current.version
            ):
                raise HTTPException(409, "Design preview is unavailable or stale; reopen it.")
            current = current.model_copy(update={"document": prior.document})
        if body.widget_id:
            known = {w.widget_id for w in current.document.spec.widgets}
            known.update(k.kpi_id for k in current.document.spec.kpis)
            if body.widget_id not in known:
                raise HTTPException(409, "That dashboard component is unavailable; reopen it.")
            current._selected_widget = body.widget_id
        result = await edit(current, body.message)
        if result.dashboard is None:
            return {
                "preview_id": None,
                "message": result.message,
                "call_ledger": result.call_ledger,
            }
        from semantic_text2sql.dashboard_service import DashboardSpec

        document = checked(
            lambda: DashboardDocument(
                spec=DashboardSpec.model_validate(result.dashboard),
                source=current.document.source,
                sources=(
                    {
                        key: DashboardSnapshot.capture(value)
                        for key, value in result._dashboard_evidence.items()
                    }
                    if result._dashboard_evidence
                    and (result.dashboard or {}).get("schema_version") == 2
                    else current.document.sources
                ),
            )
        )
        await asyncio.to_thread(checked, document.verify)
        # Re-check authorization/version after the model awaited; no stale previews.
        latest = await asyncio.to_thread(checked, lambda: db.open(principal, identity))
        if latest.permission == "viewer":
            raise HTTPException(403, "You no longer have permission to edit this dashboard.")
        if latest.version != current.version:
            raise HTTPException(409, "Dashboard changed during design; reopen it.")
        preview_id = uuid4().hex
        previews[preview_id] = _Preview(principal.subject, identity, current.version, document)
        if len(previews) > 100:
            previews.pop(next(iter(previews)))
        return {
            "preview_id": preview_id,
            "spec": document.spec,
            "message": result.message,
            "call_ledger": result.call_ledger,
            "token_usage": result.token_usage,
        }

    @router.post("/{identity}/save", response_model=SavedDashboard)
    def save(
        identity: str, body: StudioChange, request: Request, response: Response
    ) -> SavedDashboard:
        db, principal = access(request, response)
        current = checked(lambda: db.open(principal, identity))
        if current.permission == "viewer":
            raise HTTPException(403, "This dashboard is read-only for your account.")
        if current.version != body.expected_version:
            raise HTTPException(409, "Dashboard changed; reopen it before saving.")
        document = current.document
        if body.source_version is not None:
            if not body.save_as_new or body.preview_id:
                raise HTTPException(422, "Historical versions require Save as New.")
            document = checked(lambda: db.open(principal, identity, body.source_version)).document
        if body.preview_id:
            preview = previews.get(body.preview_id)
            if (
                preview is None
                or preview.owner != principal.subject
                or preview.dashboard_id != identity
                or preview.version != current.version
            ):
                raise HTTPException(409, "Design preview is unavailable or stale; edit again.")
            document = preview.document
        if body.save_as_new:
            return checked(
                lambda: db.copy(
                    principal,
                    identity,
                    body.expected_version,
                    name=body.name or current.name,
                    document=document,
                    source_version=body.source_version,
                )
            )
        return checked(
            lambda: db.revise(
                principal, identity, body.expected_version, document=document, name=body.name
            )
        )

    @router.post("/{identity}/rename", response_model=SavedDashboard)
    def rename(
        identity: str, body: StudioChange, request: Request, response: Response
    ) -> SavedDashboard:
        db, principal = access(request, response)
        if not body.name:
            raise HTTPException(422, "A dashboard name is required.")
        return checked(
            lambda: db.revise(
                principal, identity, body.expected_version, name=body.name, action="rename"
            )
        )

    @router.post("/{identity}/publish", response_model=SavedDashboard)
    def publish(
        identity: str, body: StudioChange, request: Request, response: Response
    ) -> SavedDashboard:
        db, principal = access(request, response)
        if body.preview_id or body.save_as_new or body.source_version is not None:
            raise HTTPException(422, "Save the design before approving its saved version.")
        return checked(
            lambda: db.revise(
                principal, identity, body.expected_version, action="publish", approved=body.approve
            )
        )

    @router.post("/{identity}/undo", response_model=SavedDashboard)
    def undo(
        identity: str, body: StudioChange, request: Request, response: Response
    ) -> SavedDashboard:
        db, principal = access(request, response)
        return checked(lambda: db.revise(principal, identity, body.expected_version, action="undo"))

    @router.delete("/{identity}", status_code=204)
    def delete(
        identity: str, expected_version: int, request: Request, response: Response
    ) -> Response:
        db, principal = access(request, response)
        checked(lambda: db.delete(principal, identity, expected_version))
        return Response(status_code=204, headers={"Cache-Control": "no-store, private"})

    return router

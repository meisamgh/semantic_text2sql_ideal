"""Local multi-candidate generation, execution consensus, and bounded selection."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict

from semantic_text2sql.agent import TextToSQLAgent
from semantic_text2sql.historical import HistoricalQueryStore, format_examples
from semantic_text2sql.models import (
    EnsembleRequest,
    EnsembleResponse,
    GenerateRequest,
    GenerateResponse,
    ResultCluster,
)

_STYLES = ("reasoning", "icl", "alternative")


class EnsembleTextToSQLAgent:
    def __init__(self, agent: TextToSQLAgent, history: HistoricalQueryStore) -> None:
        self.agent = agent
        self.history = history

    async def generate(self, request: EnsembleRequest) -> EnsembleResponse:
        examples = self.history.search(
            request.question, request.db_id, top_k=request.historical_top_k
        )
        evidence = (request.evidence or "") + format_examples(examples)
        candidates = []
        for style in _STYLES[: request.candidate_count]:
            candidate = await self.agent.generate(
                GenerateRequest(
                    db_id=request.db_id,
                    question=request.question,
                    evidence=evidence or None,
                    dialect=request.dialect,
                    provider=request.provider,
                    model=request.model,
                    max_attempts=request.max_attempts,
                    execute=True,
                    max_rows=request.max_rows,
                    generation_style=style,  # type: ignore[arg-type]
                )
            )
            candidates.append(candidate)
        clusters = _cluster(candidates)
        findings = {
            index: _inspect_result(request.question, candidate)
            for index, candidate in enumerate(candidates)
        }
        selected = _select(candidates, clusters, findings)
        if selected is None:
            return EnsembleResponse(
                db_id=request.db_id,
                question=request.question,
                dialect=request.dialect,
                provider=request.provider,
                model=request.model,
                historical_examples=examples,
                candidates=candidates,
                clusters=clusters,
                inspection_findings=findings,
                selected_candidate=None,
                sql=None,
                accepted=False,
                selection_reason="No candidate passed validation and execution.",
            )
        winner = candidates[selected]
        cluster_size = next(
            cluster.size for cluster in clusters if selected in cluster.candidate_indices
        )
        reason = (
            f"Selected candidate {selected} from the largest execution-result cluster "
            f"of size {cluster_size}; ties prefer fewer attempts and reasoning style."
        )
        return EnsembleResponse(
            db_id=request.db_id,
            question=request.question,
            dialect=request.dialect,
            provider=request.provider,
            model=request.model,
            historical_examples=examples,
            candidates=candidates,
            clusters=clusters,
            inspection_findings=findings,
            selected_candidate=selected,
            sql=winner.sql,
            columns=winner.columns,
            rows=winner.rows,
            accepted=True,
            selection_reason=reason,
        )


def _cluster(candidates: list[GenerateResponse]) -> list[ResultCluster]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        if not candidate.accepted or candidate.sql is None:
            continue
        rows = sorted(json.dumps(row, sort_keys=True, default=str) for row in candidate.rows)
        payload = json.dumps({"column_count": len(candidate.columns), "rows": rows}, sort_keys=True)
        fingerprint = hashlib.sha256(payload.encode()).hexdigest()
        groups[fingerprint].append(index)
    return sorted(
        (
            ResultCluster(
                fingerprint=fingerprint,
                candidate_indices=indices,
                size=len(indices),
            )
            for fingerprint, indices in groups.items()
        ),
        key=lambda item: (-item.size, item.candidate_indices[0]),
    )


def _select(
    candidates: list[GenerateResponse],
    clusters: list[ResultCluster],
    findings: dict[int, list[str]],
) -> int | None:
    if not clusters:
        return None
    ranked_clusters = sorted(
        clusters,
        key=lambda cluster: (
            -cluster.size,
            sum(len(findings[index]) for index in cluster.candidate_indices),
            cluster.candidate_indices[0],
        ),
    )
    eligible = ranked_clusters[0].candidate_indices
    return min(
        eligible,
        key=lambda index: (
            len(findings[index]),
            len(candidates[index].attempts),
            index,
        ),
    )


def _inspect_result(question: str, candidate: GenerateResponse) -> list[str]:
    if not candidate.accepted:
        return ["candidate_not_executable"]
    findings: list[str] = []
    if len(candidate.columns) != len(set(candidate.columns)):
        findings.append("duplicate_output_columns")
    if candidate.rows and any(
        all(row[index] is None for row in candidate.rows) for index in range(len(candidate.columns))
    ):
        findings.append("all_null_output_column")
    lowered = question.casefold()
    if ("how many" in lowered or "number of" in lowered) and len(candidate.columns) > 2:
        findings.append("unexpected_count_output_shape")
    return findings

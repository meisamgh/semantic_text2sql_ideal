from __future__ import annotations

import asyncio

from semantic_text2sql.ensemble import EnsembleTextToSQLAgent
from semantic_text2sql.historical import HistoricalQueryStore
from semantic_text2sql.models import EnsembleRequest, GenerateResponse, StrategyHints


class FakeCandidateAgent:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, request) -> GenerateResponse:  # type: ignore[no-untyped-def]
        index = self.calls
        self.calls += 1
        rows = [["Germany", 2]] if index < 2 else [["Italy", 1]]
        return GenerateResponse(
            db_id=request.db_id,
            question=request.question,
            provider=request.provider,
            model=request.model,
            dialect=request.dialect,
            strategy=StrategyHints(mode="exact"),
            sql=f"SELECT country, {index} AS total FROM customers",
            accepted=True,
            rows=rows,
            columns=["country", "total"],
            row_count=1,
            termination_reason="accepted",
        )


def test_ensemble_selects_largest_result_cluster() -> None:
    ensemble = EnsembleTextToSQLAgent(  # type: ignore[arg-type]
        FakeCandidateAgent(), HistoricalQueryStore()
    )

    result = asyncio.run(
        ensemble.generate(EnsembleRequest(db_id="shop", question="Count customers by country"))
    )

    assert result.accepted is True
    assert result.selected_candidate == 0
    assert result.rows == [["Germany", 2]]
    assert sorted(cluster.size for cluster in result.clusters) == [1, 2]


def test_history_retrieval_filters_database_and_failed_records(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "history.json"
    path.write_text(
        """[
          {"db_id":"shop","question":"count customers by country","SQL":"SELECT 1","success":true},
          {"db_id":"other","question":"count customers by country","SQL":"SELECT 2"},
          {"db_id":"shop","question":"count orders","SQL":"SELECT 3","success":false}
        ]"""
    )

    results = HistoricalQueryStore(path).search("customers grouped by country", "shop", top_k=3)

    assert len(results) == 1
    assert results[0].sql == "SELECT 1"

from semantic_text2sql.strategy import route_question


def test_exact_strategy_for_relational_filter() -> None:
    assert route_question("Books published after 2020 under 20 euros").mode == "exact"


def test_semantic_strategy_for_concept() -> None:
    result = route_question("Find books about freedom similar to '1984'")

    assert result.mode == "semantic"
    assert "1984" in result.semantic_terms


def test_hybrid_strategy_is_explicit() -> None:
    result = route_question("Recommend books similar to '1984'; the author may be spelled wrong")

    assert result.mode == "hybrid"

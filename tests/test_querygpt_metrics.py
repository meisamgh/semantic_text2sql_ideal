from semantic_text2sql.querygpt_metrics import qualitative_similarity, table_overlap


def test_table_overlap_uses_gold_table_recall() -> None:
    predicted = "SELECT * FROM trips JOIN cities USING(city_id)"
    gold = "SELECT * FROM trips JOIN drivers USING(driver_id)"

    assert table_overlap(predicted, gold) == 0.5


def test_structural_similarity_rewards_equivalent_shape() -> None:
    left = "SELECT country, COUNT(*) FROM customers GROUP BY country"
    right = "SELECT country, COUNT(customer_id) FROM customers GROUP BY country"

    assert qualitative_similarity(left, right) > 0.7

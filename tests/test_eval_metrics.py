from batterygemma.eval.metrics import aggregate, wilson_interval


def test_wilson_interval_stays_within_bounds_for_small_n():
    lo, hi = wilson_interval(5, 5)  # 100% on tiny n - naive normal approx would give (1.0, 1.0)
    assert 0.0 <= lo < hi <= 1.0
    assert lo < 0.8  # Wilson correctly reflects meaningful uncertainty at n=5 even though 100% were observed


def test_wilson_interval_zero_n():
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_aggregate_computes_accuracy_and_ci_for_correct_categories():
    results = [
        {"gold_id": "1", "category": "closed_qa", "correct": True},
        {"gold_id": "2", "category": "closed_qa", "correct": True},
        {"gold_id": "3", "category": "closed_qa", "correct": False},
    ]
    summary = aggregate(results)
    assert summary["closed_qa"]["n"] == 3
    assert summary["closed_qa"]["accuracy"] == round(2 / 3, 4)
    lo, hi = summary["closed_qa"]["ci95"]
    assert lo < summary["closed_qa"]["accuracy"] < hi


def test_aggregate_computes_mean_score_for_open_qa():
    results = [
        {"gold_id": "1", "category": "open_qa", "correct": True, "score": 5},
        {"gold_id": "2", "category": "open_qa", "correct": False, "score": 3},
    ]
    summary = aggregate(results)
    assert summary["open_qa"]["mean_score"] == 4.0
    assert summary["open_qa"]["accuracy"] == 0.5


def test_aggregate_computes_mean_scores_per_dimension_for_ideation():
    results = [
        {"gold_id": "1", "category": "ideation", "scores": {"groundedness": 4, "correctness": 5}},
        {"gold_id": "2", "category": "ideation", "scores": {"groundedness": 2, "correctness": 3}},
    ]
    summary = aggregate(results)
    assert summary["ideation"]["mean_scores"] == {"groundedness": 3.0, "correctness": 4.0}
    assert "accuracy" not in summary["ideation"]

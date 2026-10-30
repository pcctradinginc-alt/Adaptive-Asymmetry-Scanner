from modules.research_evidence_coverage import audit


def test_coverage_counts_missing_and_genuine_labels():
    rows = [
        {"date": "2026-08-01", "ticker": "AAA", "signal_timestamp": "2026-08-01T15:00:00Z",
         "signal_id": "one", "direction": "BULLISH", "features": {"mom": 1.0},
         "outcomes": {"ret_20d": 0.5, "real_strat_ret_20d": 0.1}},
        {"date": "2026-08-01", "ticker": "BBB", "features": {"mom": None},
         "outcomes": {"ret_20d": -0.1}},
    ]
    result = audit(rows, "2026-10-30")
    assert result["counts"]["candidate_rows"] == 2
    assert result["counts"]["mature_20d_option_label"] == 1
    assert result["counts"]["not_eligible_for_20d_option_learning"] == 1
    assert result["feature_presence"]["mom"] == 1
    assert result["closed_loop_training_verified"] is False

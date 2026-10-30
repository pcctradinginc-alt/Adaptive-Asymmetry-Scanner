from modules.research_lifecycle_integrity import inspect


def test_duplicate_and_premature_marks_fail_closed():
    row = {"date": "2026-10-25", "ticker": "AAPL", "signal_id": "a",
           "signal_timestamp": "2026-10-25T15:00:00Z", "features": {"vix": 20},
           "outcomes": {"real_strat_ret_20d": 0.5}}
    result = inspect([row, row], "2026-10-30")
    assert result["issues"]["duplicate_signal"] == 1
    assert result["issues"]["premature_mark_20d"] == 2
    assert result["structural_integrity"] == "FAIL"
    assert result["production_learning_closed_loop"] == "NOT_VERIFIED"


def test_missing_mature_marks_not_assumed_zero():
    row = {"date": "2026-08-01", "ticker": "AAPL", "signal_id": "a",
           "signal_timestamp": "2026-08-01T15:00:00Z", "features": {}}
    result = inspect([row], "2026-10-30")
    assert result["lifecycle"]["mature_missing_20d"] == 1
    assert result["structural_integrity"] == "PASS"

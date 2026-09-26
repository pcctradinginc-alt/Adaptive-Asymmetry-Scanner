"""
tests/test_hypothesis_generator.py – Tests for the Hypothesen-Generator

Tests the challenger_snippet() function and related hypothesis-generation logic.
"""

import sys
from datetime import date
from pathlib import Path

# Add parent directory to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from monthly_report import challenger_snippet, LEDGER_FIELD_MAP


def test_ledger_field_map():
    """Verify the field mapping covers expected gates."""
    expected_keys = {"mismatch", "impact", "surprise", "score", "dte"}
    assert set(LEDGER_FIELD_MAP.keys()) == expected_keys, \
        f"Field map should have keys {expected_keys}, got {set(LEDGER_FIELD_MAP.keys())}"


def test_challenger_snippet_score_gate():
    """Test snippet generation for a trade_score suggestion."""
    suggestion = {
        "gate": "Trade-Score-Floor",
        "field": "score",
        "mode": "min",
        "current": 55,
        "suggested": 61,
        "gain_pp": 5.2,
        "current_stats": {"n": 50, "win_rate": 0.52, "mean": 0.05},
        "suggested_stats": {"n": 45, "win_rate": 0.57, "mean": 0.06},
    }
    today = date(2026, 9, 26)
    snippet = challenger_snippet(suggestion, today)

    # Check that snippet contains expected content
    assert "trade_score_floor_61" in snippet, "Snippet ID should include gate and new value"
    assert "features.trade_score" in snippet, "Should reference the correct ledger field"
    assert 'value: 61' in snippet, "Should have the new threshold value"
    assert 'op: ">="' in snippet, "Score gate should use >= operator"
    assert "2026-09-27" in snippet, "start_date should be tomorrow"
    assert "2026-09-26" in snippet, "registered_on should be today"
    assert "outcomes.real_opt_ret_45d" in snippet, "Should use the correct metric"
    assert "status: active" in snippet, "Should start as active"


def test_challenger_snippet_mismatch_gate():
    """Test snippet generation for a mismatch suggestion (max mode)."""
    suggestion = {
        "gate": "Mismatch-Cap",
        "field": "mismatch",
        "mode": "max",
        "current": 7.0,
        "suggested": 6.0,
        "gain_pp": 3.5,
        "current_stats": {"n": 60, "win_rate": 0.48, "mean": 0.02},
        "suggested_stats": {"n": 55, "win_rate": 0.51, "mean": 0.03},
    }
    today = date(2026, 9, 26)
    snippet = challenger_snippet(suggestion, today)

    # Check mismatch-specific content
    assert "mismatch_cap_6p0" in snippet, "Snippet ID should encode mismatch value"
    assert "features.mismatch" in snippet, "Should reference mismatch ledger field"
    assert 'op: "<="' in snippet, "Mismatch gate (max) should use <= operator"


def test_challenger_snippet_no_ledger_field():
    """Test snippet generation for a suggestion with no ledger field (DTE)."""
    suggestion = {
        "gate": "DTE-Floor",
        "field": "dte",
        "mode": "min",
        "current": 45,
        "suggested": 60,
        "gain_pp": 4.0,
        "current_stats": {"n": 40, "win_rate": 0.50, "mean": 0.04},
        "suggested_stats": {"n": 38, "win_rate": 0.54, "mean": 0.05},
    }
    today = date(2026, 9, 26)
    snippet = challenger_snippet(suggestion, today)

    # Should return a comment, not a full YAML block
    assert "kein Ledger-Feld" in snippet, "Should indicate no ledger field"
    assert "Challenger nicht direkt auswertbar" in snippet, "Should note it's not directly evaluable"


def test_challenger_snippet_start_date_after_registered_on():
    """Verify that start_date is strictly after registered_on."""
    suggestion = {
        "gate": "Impact-Floor",
        "field": "impact",
        "mode": "min",
        "current": 4,
        "suggested": 5,
        "gain_pp": 2.0,
        "current_stats": {"n": 30, "win_rate": 0.53, "mean": 0.07},
        "suggested_stats": {"n": 28, "win_rate": 0.55, "mean": 0.08},
    }
    today = date(2026, 9, 26)
    snippet = challenger_snippet(suggestion, today)

    # Parse the YAML to check dates
    import yaml
    # Extract the YAML portion (skip the leading spaces for parsing)
    lines = snippet.strip().split('\n')
    yaml_str = '\n'.join(lines)

    # Manually check date ordering (simple string comparison is sufficient for ISO dates)
    assert "2026-09-26" in yaml_str, "registered_on should be today"
    assert "2026-09-27" in yaml_str, "start_date should be tomorrow"

    # Verify start_date comes after registered_on in the YAML
    reg_idx = yaml_str.find("2026-09-26")
    start_idx = yaml_str.find("2026-09-27")
    assert reg_idx < start_idx, "registered_on should appear before start_date in output"


def test_challenger_snippet_metric_and_guardrails():
    """Verify metric and guardrail values in snippet."""
    suggestion = {
        "gate": "Surprise-Floor",
        "field": "surprise",
        "mode": "min",
        "current": 3,
        "suggested": 4,
        "gain_pp": 1.5,
        "current_stats": {"n": 25, "win_rate": 0.56, "mean": 0.09},
        "suggested_stats": {"n": 22, "win_rate": 0.57, "mean": 0.10},
    }
    today = date(2026, 9, 26)
    snippet = challenger_snippet(suggestion, today)

    assert "outcomes.real_opt_ret_45d" in snippet, "Should use opt_ret_45d metric"
    assert "min_n: 30" in snippet, "Should enforce min_n guardrail"
    assert "horizon_days: 45" in snippet, "Should set horizon_days to 45"
    assert "max_duration_days: 180" in snippet, "Should set max_duration_days to 180"


if __name__ == "__main__":
    # Run basic sanity checks
    test_ledger_field_map()
    test_challenger_snippet_score_gate()
    test_challenger_snippet_mismatch_gate()
    test_challenger_snippet_no_ledger_field()
    test_challenger_snippet_start_date_after_registered_on()
    test_challenger_snippet_metric_and_guardrails()
    print("All hypothesis_generator tests passed!")

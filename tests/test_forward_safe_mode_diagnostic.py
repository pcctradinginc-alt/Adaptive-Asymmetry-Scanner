"""Correlated memory rows must never be presented as independent alpha proof."""
from __future__ import annotations

import math

from modules import meta_cognition as mc


def test_negative_dependent_rows_keep_protective_gate_without_significance_claim():
    # Twelve tickers can share a signal date, so the ordinary t is invalid
    # as an independent 12-period significance test.
    result = mc.safe_mode({}, {}, {}, [-0.30 + i * 0.005 for i in range(12)], {})
    assert result["active"]
    assert any("PERFORMANCE BREAKDOWN" in x for x in result["reasons"])
    assert any("PROVISIONAL" in x and "NOT tested" in x for x in result["reasons"])
    diagnostic = result["forward_diagnostic"]
    assert diagnostic["raw_rows"] == 12
    assert diagnostic["independent_significance"] is False
    assert diagnostic["production_promotion_eligible"] is False
    assert diagnostic["status"] == "PROVISIONAL_CORRELATED_SCREEN"
    assert math.isfinite(diagnostic["raw_t_12"])


def test_insufficient_and_flat_forward_are_not_claimed_as_alpha():
    short = mc.safe_mode({}, {}, {}, [-0.2] * 11, {})
    assert not short["active"]
    assert short["forward_diagnostic"]["status"] == "INSUFFICIENT_OR_NOT_EVALUATED"
    flat = mc.safe_mode({}, {}, {}, [0.01] * 12, {})
    assert flat["forward_diagnostic"]["raw_t_12"] == 0
    assert not flat["forward_diagnostic"]["independent_significance"]


def test_invalid_forward_values_do_not_produce_false_statistic():
    forward = [-0.1] * 11 + [float("nan")]
    result = mc.safe_mode({}, {}, {}, forward, {})
    assert result["active"]
    assert result["forward_diagnostic"]["status"] == "INVALID_VALUES"
    assert result["forward_diagnostic"]["raw_t_12"] is None
    assert any("CORRUPT FORWARD OUTCOMES" in x for x in result["reasons"])

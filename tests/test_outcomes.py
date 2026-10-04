"""Lernen nur aus verlässlichen Outcomes (echte Quotes) – Audit 2026-10-03."""
from __future__ import annotations

import feedback
from modules.outcomes import RELIABLE_OUTCOME_METHODS, is_reliable_outcome


def test_reliability_definition():
    assert is_reliable_outcome({"outcome_method": "option_quote"})
    assert is_reliable_outcome({"outcome_method": "option_quote_bid_zero"})
    assert not is_reliable_outcome({"outcome_method": "delta_approx_clipped"})
    assert not is_reliable_outcome({"outcome_method": "stock_fallback"})
    assert not is_reliable_outcome({"outcome_reliable": False, "outcome_method": "option_quote"})
    assert not is_reliable_outcome({"outcome_method_reconstructed": True})
    assert not is_reliable_outcome({})        # Altbestand ohne Methode = UNKNOWN (Owner-Entscheidung 2026-10-04)
    assert not is_reliable_outcome({"outcome_method": "option_quote", "outcome_method_reconstructed": "delta_approx"})
    assert "delta_approx" not in RELIABLE_OUTCOME_METHODS


def test_pearson_weights_ignore_approximated_outcomes():
    good = [{"outcome": 0.1 * (i % 3), "entry_date": f"2026-0{1 + i % 9}-1{i % 9}",
             "features": {"impact": i % 7, "mismatch": i % 5, "eps_drift": 0.01 * i}, "outcome_reliable": True}
            for i in range(12)]
    junk = [{"outcome": 5.0, "entry_date": "2026-05-01", "features": {"impact": 9, "mismatch": 9, "eps_drift": 0.2},
             "outcome_method": "delta_approx_clipped"} for _ in range(30)]
    w_good = feedback.compute_pearson_weights({"closed_trades": good, "model_weights": {}})
    w_mix = feedback.compute_pearson_weights({"closed_trades": good + junk, "model_weights": {}})
    assert w_good == w_mix                                           # Näherungen ändern nichts

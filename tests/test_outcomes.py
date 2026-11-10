"""Lernen nur aus verlässlichen Outcomes (echte Quotes) – Audit 2026-10-03."""
from __future__ import annotations

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


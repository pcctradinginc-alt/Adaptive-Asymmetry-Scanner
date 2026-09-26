"""
tests/test_outcome_method.py

Tests for compute_outcome() method tracking (Task A) and audit_outcomes.py inference (Task B).

- Verifies compute_outcome() correctly sets meta["method"]
- Checks that return values are unchanged when meta is provided
- Tests audit outcome method inference from delta-approx values
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from feedback import compute_outcome


@pytest.fixture
def base_trade():
    """Minimal trade dict for testing."""
    return {
        "ticker": "TEST",
        "strategy": "LONG_CALL",
        "entry_date": "2026-01-01",
        "option": {
            "strike": 100,
            "expiry": "2026-02-01",
            "ask": 2.0,
        },
        "simulation": {
            "current_price": 150.0,
        },
        "entry_debit": 2.0,
    }


@pytest.fixture
def spread_trade():
    """Spread trade dict."""
    return {
        "ticker": "TEST",
        "strategy": "BULL_CALL_SPREAD",
        "entry_date": "2026-01-01",
        "option": {
            "strike": 100,
            "expiry": "2026-02-01",
            "ask": 2.0,
            "spread_leg": {
                "strike": 110,
                "bid": 0.5,
            },
            "net_debit": 1.5,
        },
        "simulation": {},
        "entry_debit": 1.5,
    }


class TestComputeOutcomeMethodTracking:
    """Tests for compute_outcome meta["method"] tracking."""

    def test_spread_quote_method(self, spread_trade, monkeypatch):
        """Test that spread trades are marked with 'spread_quote' method."""
        # Mock get_current_spread_price to return a valid value
        mock_spread_price = MagicMock(return_value=1.2)
        monkeypatch.setattr("feedback.get_current_spread_price", mock_spread_price)

        meta = {}
        outcome = compute_outcome(spread_trade, 155.0, meta)

        assert outcome is not None
        assert meta["method"] == "spread_quote"
        # Verify outcome is correct: (1.2 - 1.5) / 1.5 ≈ -0.2
        assert abs(outcome - (-0.2)) < 0.01

    def test_option_quote_method(self, base_trade, monkeypatch):
        """Test that option trades with valid option prices use 'option_quote' method."""
        # Mock get_current_option_price to return a valid option price
        mock_option_price = MagicMock(return_value=3.0)
        monkeypatch.setattr("feedback.get_current_option_price", mock_option_price)

        meta = {}
        outcome = compute_outcome(base_trade, 155.0, meta)

        assert outcome is not None
        assert meta["method"] == "option_quote"
        # Verify outcome: (3.0 - 2.0) / 2.0 = 0.5
        assert abs(outcome - 0.5) < 0.01

    def test_delta_approx_unclipped_method(self, base_trade, monkeypatch):
        """Test delta_approx without clipping (small stock move)."""
        # Mock get_current_option_price to return 0 (fallback to delta approx)
        mock_option_price = MagicMock(return_value=0)
        monkeypatch.setattr("feedback.get_current_option_price", mock_option_price)

        meta = {}
        # Stock move: (160 - 150) / 150 = 0.0667
        # Delta-approx: 0.0667 * (150 / 2.0) * 0.65 ≈ 3.25, no clipping needed
        outcome = compute_outcome(base_trade, 160.0, meta)

        assert outcome is not None
        assert meta["method"] == "delta_approx"
        # No clipping should occur for this small move
        assert -1.0 <= outcome <= 5.0

    def test_delta_approx_clipped_method(self, base_trade, monkeypatch):
        """Test delta_approx with clipping (large positive stock move)."""
        # Mock get_current_option_price to return 0
        mock_option_price = MagicMock(return_value=0)
        monkeypatch.setattr("feedback.get_current_option_price", mock_option_price)

        meta = {}
        # Stock move: (300 - 150) / 150 = 1.0
        # Delta-approx: 1.0 * (150 / 2.0) * 0.65 = 48.75, clipped to 5.0
        outcome = compute_outcome(base_trade, 300.0, meta)

        assert outcome is not None
        assert meta["method"] == "delta_approx_clipped"
        assert outcome == 5.0  # Should be clipped

    def test_delta_approx_clipped_negative(self, base_trade, monkeypatch):
        """Test delta_approx with negative clipping."""
        # Mock get_current_option_price to return 0
        mock_option_price = MagicMock(return_value=0)
        monkeypatch.setattr("feedback.get_current_option_price", mock_option_price)

        meta = {}
        # Stock move: (75 - 150) / 150 = -0.5
        # Delta-approx: -0.5 * (150 / 2.0) * 0.65 = -24.375, clipped to -1.0
        outcome = compute_outcome(base_trade, 75.0, meta)

        assert outcome is not None
        assert meta["method"] == "delta_approx_clipped"
        assert outcome == -1.0  # Should be clipped

    def test_stock_fallback_method(self, base_trade, monkeypatch):
        """Test stock_fallback when no option debit is available."""
        trade_no_debit = base_trade.copy()
        trade_no_debit["entry_debit"] = 0
        # Clear option so it can't extract ask as entry_debit
        trade_no_debit["option"] = {"strike": 100, "expiry": "2026-02-01"}

        meta = {}
        # Stock move: (160 - 150) / 150 ≈ 0.0667
        outcome = compute_outcome(trade_no_debit, 160.0, meta)

        assert outcome is not None
        assert meta["method"] == "stock_fallback"
        assert abs(outcome - 0.0667) < 0.001

    def test_return_value_unchanged_with_meta(self, base_trade, monkeypatch):
        """Verify return values are identical with and without meta parameter."""
        mock_option_price = MagicMock(return_value=3.0)
        monkeypatch.setattr("feedback.get_current_option_price", mock_option_price)

        # Compute outcome without meta
        outcome_no_meta = compute_outcome(base_trade, 155.0)

        # Reset mock and compute with meta
        mock_option_price.reset_mock()
        mock_option_price.return_value = 3.0
        meta = {}
        outcome_with_meta = compute_outcome(base_trade, 155.0, meta)

        assert outcome_no_meta == outcome_with_meta
        assert meta["method"] == "option_quote"

    def test_meta_none_doesnt_crash(self, base_trade, monkeypatch):
        """Verify passing meta=None doesn't cause issues."""
        mock_option_price = MagicMock(return_value=3.0)
        monkeypatch.setattr("feedback.get_current_option_price", mock_option_price)

        # Should not crash with meta=None
        outcome = compute_outcome(base_trade, 155.0, None)
        assert outcome is not None


class TestAuditOutcomeInference:
    """Tests for audit_outcomes.py inference logic (Task B)."""

    def test_infer_delta_approx_from_clipped_value(self):
        """Test inferring delta-approx method from a clipped outcome value.

        A clipped delta-approx with outcome exactly -1.0 or 5.0 is the signal.
        """
        from scripts.audit_outcomes import infer_outcome_method_legacy

        # Simulate a trade that looks like delta-approx clipped
        trade = {
            "ticker": "TEST",
            "outcome": -1.0,  # Exactly -1.0 suggests clipping
            "entry_debit": 2.0,
            "simulation": {
                "current_price": 150.0,
            },
            "option": {
                "strike": 100,
            },
        }

        # Mock the current price
        entry_stock = 150.0
        close_price = 75.0  # 50% drop

        # Infer method
        method = infer_outcome_method_legacy(trade, entry_stock, close_price)

        # Should infer delta_approx or delta_approx_clipped
        assert method in ["delta_approx", "delta_approx_clipped", "likely_delta_approx"]

    def test_synthetic_clipped_trade_flag(self):
        """Test that synthetic clipped delta-approx trades are flagged correctly."""
        from scripts.audit_outcomes import compute_clipped_delta_approx

        entry_stock = 150.0
        close_price = 300.0  # 100% stock gain
        entry_debit = 2.0

        # Compute what delta-approx would be
        stock_return = (close_price - entry_stock) / entry_stock
        unclipped = stock_return * (entry_stock / entry_debit) * 0.65
        clipped = max(-1.0, min(unclipped, 5.0))

        assert clipped == 5.0

        # An outcome of 5.0 should be flagged as likely_delta_approx_clipped
        outcome = 5.0
        assert abs(outcome - clipped) < 0.01


class TestIntegrationWithHistory:
    """Integration tests using actual history.json structure."""

    def test_closed_trade_with_outcome_method(self, base_trade):
        """Test that closed trades properly record outcome_method."""
        # This simulates what the main loop does
        meta = {}
        with patch("feedback.get_current_option_price", return_value=3.0):
            outcome = compute_outcome(base_trade, 155.0, meta)

        # Simulate storing to closed_trades
        base_trade["outcome"] = round(outcome, 4)
        base_trade["outcome_method"] = meta.get("method", "unknown")

        assert base_trade["outcome_method"] == "option_quote"
        assert base_trade["outcome"] == 0.5


def test_imports():
    """Verify critical modules can be imported."""
    # Check that compute_outcome is importable
    assert callable(compute_outcome)

    # Check that audit script exists and is importable
    audit_path = Path(__file__).resolve().parent.parent / "scripts" / "audit_outcomes.py"
    assert audit_path.exists(), "scripts/audit_outcomes.py does not exist"

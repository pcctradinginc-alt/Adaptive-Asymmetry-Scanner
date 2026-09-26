"""
Tests for monthly_report.py statistical helpers and output.
"""

import pytest
from datetime import date
from monthly_report import (
    dedup_closed_trades,
    rolling_stats,
    bootstrap_ci,
    build_rolling_stats_html,
    month_stats,
)


def test_dedup_closed_trades_basic():
    """Test dedup keeps one trade per (ticker, entry_date)."""
    trades = [
        {"ticker": "AAPL", "entry_date": "2024-01-01", "outcome": 0.05},
        {"ticker": "AAPL", "entry_date": "2024-01-01", "outcome": 0.10},  # dup, should be skipped
        {"ticker": "MSFT", "entry_date": "2024-01-01", "outcome": -0.02},
        {"ticker": "AAPL", "entry_date": "2024-01-02", "outcome": 0.08},
    ]
    result = dedup_closed_trades(trades)
    assert len(result) == 3
    assert result[0]["outcome"] == 0.05  # First AAPL/2024-01-01
    assert result[1]["outcome"] == -0.02  # MSFT/2024-01-01
    assert result[2]["outcome"] == 0.08  # AAPL/2024-01-02


def test_dedup_closed_trades_empty():
    """Test dedup with empty list."""
    result = dedup_closed_trades([])
    assert result == []


def test_rolling_stats_basic():
    """Test rolling_stats returns correct metrics."""
    trades = [
        {"outcome": 0.05, "close_date": "2024-01-01"},
        {"outcome": 0.10, "close_date": "2024-01-02"},
        {"outcome": -0.02, "close_date": "2024-01-03"},
        {"outcome": -0.98, "close_date": "2024-01-04"},  # total loss
    ]
    stats = rolling_stats(trades, window=None)

    assert stats["n"] == 4
    assert stats["win_rate"] == 0.5  # 2 wins out of 4
    assert stats["mean"] == pytest.approx((0.05 + 0.10 - 0.02 - 0.98) / 4)
    assert stats["total_loss_rate"] == 0.25  # 1 out of 4 <= -0.95


def test_rolling_stats_window():
    """Test rolling_stats with a window parameter."""
    trades = [
        {"outcome": 0.05, "close_date": "2024-01-01"},
        {"outcome": 0.10, "close_date": "2024-01-02"},
        {"outcome": -0.02, "close_date": "2024-01-03"},
        {"outcome": 0.08, "close_date": "2024-01-04"},
    ]
    stats = rolling_stats(trades, window=2)

    # Last 2 trades: -0.02, 0.08 -> 1 win out of 2
    assert stats["n"] == 2
    assert stats["win_rate"] == 0.5


def test_rolling_stats_mean_ex_top3():
    """Test mean_ex_top3 calculation."""
    trades = [
        {"outcome": 0.50, "close_date": "2024-01-01"},
        {"outcome": 0.40, "close_date": "2024-01-02"},
        {"outcome": 0.30, "close_date": "2024-01-03"},
        {"outcome": 0.05, "close_date": "2024-01-04"},
        {"outcome": -0.10, "close_date": "2024-01-05"},
    ]
    stats = rolling_stats(trades, window=None)

    # Top 3 are 0.50, 0.40, 0.30; excluding them leaves 0.05, -0.10
    assert stats["mean_ex_top3"] == pytest.approx((0.05 - 0.10) / 2)


def test_rolling_stats_mean_ex_top3_too_few():
    """Test mean_ex_top3 returns None if <= 3 trades."""
    trades = [
        {"outcome": 0.05, "close_date": "2024-01-01"},
        {"outcome": 0.10, "close_date": "2024-01-02"},
    ]
    stats = rolling_stats(trades, window=None)
    assert stats["mean_ex_top3"] is None


def test_rolling_stats_profit_factor():
    """Test profit_factor calculation (sum_wins / abs(sum_losses))."""
    trades = [
        {"outcome": 0.10, "close_date": "2024-01-01"},
        {"outcome": 0.20, "close_date": "2024-01-02"},
        {"outcome": -0.05, "close_date": "2024-01-03"},
        {"outcome": -0.10, "close_date": "2024-01-04"},
    ]
    stats = rolling_stats(trades, window=None)

    # sum_wins = 0.10 + 0.20 = 0.30
    # sum_losses = -0.05 + -0.10 = -0.15
    # profit_factor = 0.30 / 0.15 = 2.0
    assert stats["profit_factor"] == pytest.approx(2.0)


def test_rolling_stats_profit_factor_no_losses():
    """Test profit_factor returns None if no losses."""
    trades = [
        {"outcome": 0.10, "close_date": "2024-01-01"},
        {"outcome": 0.20, "close_date": "2024-01-02"},
    ]
    stats = rolling_stats(trades, window=None)
    assert stats["profit_factor"] is None


def test_rolling_stats_empty():
    """Test rolling_stats with empty list."""
    stats = rolling_stats([], window=None)
    assert stats["n"] == 0
    assert stats["win_rate"] is None
    assert stats["mean"] is None


def test_bootstrap_ci_deterministic():
    """Test bootstrap_ci is deterministic with seed."""
    values = list(range(1, 51))  # 50 values
    ci1 = bootstrap_ci(values, n_boot=2000, seed=42)
    ci2 = bootstrap_ci(values, n_boot=2000, seed=42)

    assert ci1 == ci2


def test_bootstrap_ci_too_few():
    """Test bootstrap_ci returns None for n < 10."""
    values = list(range(1, 9))  # 8 values
    ci = bootstrap_ci(values, seed=42)
    assert ci is None


def test_bootstrap_ci_valid():
    """Test bootstrap_ci returns valid CI for sufficient data."""
    values = list(range(1, 101))  # 100 values
    ci = bootstrap_ci(values, n_boot=2000, seed=42)

    assert ci is not None
    lower, upper = ci
    assert lower < upper
    # CI should roughly contain the mean
    mean = sum(values) / len(values)
    assert lower < mean < upper


def test_rolling_stats_total_loss_rate_boundary():
    """Test total_loss_rate uses <= -0.95 boundary."""
    trades = [
        {"outcome": -0.94, "close_date": "2024-01-01"},  # Just above boundary
        {"outcome": -0.95, "close_date": "2024-01-02"},  # On boundary
        {"outcome": -0.96, "close_date": "2024-01-03"},  # Below boundary
    ]
    stats = rolling_stats(trades, window=None)

    # Should count -0.95 and -0.96
    assert stats["total_loss_rate"] == pytest.approx(2 / 3)


def test_build_rolling_stats_html_basic():
    """Test that build_rolling_stats_html doesn't contain misleading language."""
    trades = [
        {"ticker": "AAPL", "entry_date": "2024-01-01", "outcome": 0.05, "close_date": "2024-01-10"},
        {"ticker": "MSFT", "entry_date": "2024-01-02", "outcome": 0.10, "close_date": "2024-01-11"},
        {"ticker": "GOOG", "entry_date": "2024-01-03", "outcome": -0.02, "close_date": "2024-01-12"},
    ]
    prev_stats = {"win_rate": 0.5, "n": 5}

    html = build_rolling_stats_html(trades, prev_stats, {"win_rate": 0.6, "n": 5})

    # Should NOT contain "das System wird besser/schlechter"
    assert "das System wird besser" not in html
    assert "das System wird schlechter" not in html
    # Should contain the table
    assert "Beobachtete Performance" in html
    assert "Win-Rate" in html
    assert "Fenster" in html


def test_build_rolling_stats_html_honest_language():
    """Test that the output uses honest statistical language."""
    trades = [
        {"ticker": "AAPL", "entry_date": "2024-01-01", "outcome": 0.05, "close_date": "2024-01-10"},
        {"ticker": "MSFT", "entry_date": "2024-01-02", "outcome": 0.10, "close_date": "2024-01-11"},
        {"ticker": "GOOG", "entry_date": "2024-01-03", "outcome": -0.02, "close_date": "2024-01-12"},
    ]
    prev_stats = {"win_rate": 0.5, "n": 5}

    html = build_rolling_stats_html(trades, prev_stats, {"win_rate": 0.6, "n": 5})

    # Should contain honest language about uncertainty
    assert "Stichprobe" in html or "Δ Win-Rate" in html
    # Zu wenig Daten für einen KI-Vergleich → neutrale Aussage, nie "besser"
    assert "kein Nachweis einer Veränderung" in html
    assert "wird besser" not in html


def test_month_stats_includes_total_losses():
    """Test that month_stats correctly counts total losses."""
    trades = [
        {"outcome": 0.05, "close_date": "2024-01-01"},
        {"outcome": -0.99, "close_date": "2024-01-02"},
        {"outcome": -1.0, "close_date": "2024-01-02"},
        {"outcome": 0.10, "close_date": "2024-01-03"},
    ]

    stats = month_stats(trades, "2024-01")
    # -0.99 and -1.0 are <= -0.99
    assert stats["total_losses"] == 2

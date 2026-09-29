"""Regression 2026-09-29: Outcomes zu realisierbaren Preisen (kein
scheinbares Alpha durch Mid-/Ask-Exit, kein falsches Instrument)."""
from __future__ import annotations

import feedback as fb


def _patch_quotes(monkeypatch, quotes):
    """quotes: {(strike, type): (bid, ask)}"""
    monkeypatch.setattr(fb, "_use_tradier", lambda: True)

    def tq(ticker, strike, expiry, t):
        q = quotes.get((float(strike), t))
        return None if q is None else {"bid": q[0], "ask": q[1], "source": "tradier"}
    monkeypatch.setattr(fb, "_tradier_option_quote", tq)
    monkeypatch.setattr(fb, "_yfinance_option_quote", lambda *a: None)


def test_long_option_exit_at_bid_not_mid(monkeypatch):
    _patch_quotes(monkeypatch, {(100.0, "call"): (4.0, 4.4)})
    trade = {"ticker": "A", "strategy": "LONG_CALL", "entry_debit": 4.0,
             "option": {"strike": 100, "expiry": "2099-01-01"}, "simulation": {"current_price": 100}}
    meta = {}
    assert fb.compute_outcome(trade, 100.0, meta) == 0.0          # Bid 4.0 vs Entry 4.0 (Mid wäre +5 %)
    assert meta["method"] == "option_quote"


def test_bid_zero_is_minus_100_not_delta_approx(monkeypatch):
    _patch_quotes(monkeypatch, {(100.0, "call"): (0.0, 0.05)})
    trade = {"ticker": "A", "strategy": "LONG_CALL", "entry_debit": 2.0,
             "option": {"strike": 100, "expiry": "2099-01-01"}, "simulation": {"current_price": 100}}
    meta = {}
    assert fb.compute_outcome(trade, 130.0, meta) == -1.0
    assert meta["method"] == "option_quote_bid_zero"


def test_no_put_fallback_when_strategy_known(monkeypatch):
    _patch_quotes(monkeypatch, {(100.0, "put"): (9.0, 9.2)})       # nur der PUT existiert
    assert fb.get_option_quote("A", {"strike": 100, "expiry": "2099-01-01"}, "LONG_CALL") is None
    assert fb.get_option_quote("A", {"strike": 100, "expiry": "2099-01-01"}, "")["bid"] == 9.0


def test_spread_closed_at_long_bid_minus_short_ask(monkeypatch):
    _patch_quotes(monkeypatch, {(100.0, "call"): (5.0, 5.4), (110.0, "call"): (1.8, 2.0)})
    opt = {"strike": 100, "expiry": "2099-01-01", "spread_leg": {"strike": 110}}
    assert fb.get_current_spread_price("A", opt, "BULL_CALL_SPREAD") == 3.0   # 5.0 - 2.0 (Mid wäre 3.3)


def test_expired_bear_put_spread_intrinsic(monkeypatch):
    import pandas as pd

    class _T:
        def history(self, **kw):
            return pd.DataFrame({"Close": [92.0]})
    monkeypatch.setattr(fb.yf, "Ticker", lambda t: _T())
    opt = {"strike": 100, "expiry": "2026-01-16", "spread_leg": {"strike": 90}}
    assert fb._expired_spread_intrinsic("A", opt) == 8.0                     # Long-Put 100 / Short-Put 90

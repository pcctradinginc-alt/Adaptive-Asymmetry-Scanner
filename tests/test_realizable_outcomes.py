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


def _hist(monkeypatch, closes: dict, seen: list | None = None):
    import pandas as pd

    class _T:
        def history(self, **kw):
            if seen is not None:
                seen.append(kw)
            return pd.DataFrame({"Close": list(closes.values())}, index=pd.DatetimeIndex(list(closes)))
    monkeypatch.setattr(fb.yf, "Ticker", lambda t: _T())


def test_expired_bear_put_spread_intrinsic(monkeypatch):
    seen: list = []
    _hist(monkeypatch, {"2026-01-16": 92.0, "2026-01-20": 95.0}, seen)
    opt = {"strike": 100, "expiry": "2026-01-16", "spread_leg": {"strike": 90}}
    assert fb._expired_spread_intrinsic("A", opt) == 8.0                     # Long-Put 100 / Short-Put 90
    assert seen and seen[0]["auto_adjust"] is False                          # Strikes sind unbereinigt


def test_expiry_settlement_needs_exact_expiry_bar(monkeypatch):
    """Kein Balken exakt am Verfallstag (Feiertag/Lücke) -> kein Settlement (nie der Folgetag)."""
    _hist(monkeypatch, {"2026-01-20": 95.0})
    assert fb._expired_spread_intrinsic("A", {"strike": 100, "expiry": "2026-01-16", "spread_leg": {"strike": 90}}) is None
    assert fb._expired_option_intrinsic("A", {"strike": 100, "expiry": "2026-01-16"}, "LONG_PUT") is None


def test_expired_long_option_is_settlement_not_delta_approx(monkeypatch):
    _hist(monkeypatch, {"2026-01-16": 92.0})
    base = {"ticker": "A", "entry_debit": 5.0, "option": {"strike": 100, "expiry": "2026-01-16"},
            "simulation": {"current_price": 100.0}}
    meta: dict = {}
    r = fb.compute_outcome({**base, "strategy": "LONG_PUT"}, 92.0, meta)
    assert r == (8.0 - 5.0) / 5.0 and meta["method"] == "expiry_intrinsic"
    assert meta["price_type"] == "SETTLEMENT_INTRINSIC"
    meta2: dict = {}
    assert fb.compute_outcome({**base, "strategy": "LONG_CALL"}, 92.0, meta2) == -1.0     # wertlos verfallen
    from modules.outcomes import RELIABLE_OUTCOME_METHODS
    assert "expiry_intrinsic" in RELIABLE_OUTCOME_METHODS and "spread_quote" in RELIABLE_OUTCOME_METHODS


def test_expired_spread_labelled_settlement_not_quote(monkeypatch):
    _hist(monkeypatch, {"2026-01-16": 106.0})
    t = {"ticker": "A", "strategy": "BULL_CALL_SPREAD", "option": {"strike": 100, "expiry": "2026-01-16",
         "net_debit": 4.0, "spread_leg": {"strike": 110}}, "simulation": {}}
    meta: dict = {}
    r = fb.compute_outcome(t, 106.0, meta)
    assert r == (6.0 - 4.0) / 4.0
    assert meta["method"] == "expiry_intrinsic" and meta["price_type"] == "SETTLEMENT_INTRINSIC"


def test_delta_approx_sign_for_puts(monkeypatch):
    monkeypatch.setattr(fb, "get_option_quote", lambda *a: None)
    monkeypatch.setattr(fb, "get_current_option_price", lambda *a, **k: 0.0)
    base = {"ticker": "A", "entry_debit": 5.0, "option": {"strike": 100, "expiry": "2099-01-01"},
            "simulation": {"current_price": 100.0}}
    put_up = fb.compute_outcome({**base, "strategy": "LONG_PUT"}, 110.0, {})      # Aktie +10 %
    call_up = fb.compute_outcome({**base, "strategy": "LONG_CALL"}, 110.0, {})
    assert put_up < 0 < call_up
    assert fb.compute_outcome({**base, "strategy": "LONG_PUT"}, 90.0, {}) > 0     # Aktie -10 % -> Put +

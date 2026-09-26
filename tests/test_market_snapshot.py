"""
Tests für modules/market_snapshot.py – Tradier-Marktdaten-Snapshots.

Kein Netzwerkzugriff: alle Tradier-Calls laufen über die kleinen
"_fetch_*"-Funktionen, die hier gemockt werden.
"""

from datetime import datetime, timezone

import pytest

from modules import market_snapshot as ms


# ── us_market_session ────────────────────────────────────────────────────────

def test_session_pre_market_est():
    # 2026-01-15 (Do) 08:00 America/New_York (EST, UTC-5) = 13:00 UTC
    ts = datetime(2026, 1, 15, 13, 0, tzinfo=timezone.utc)
    assert ms.us_market_session(ts) == "pre"


def test_session_regular_est():
    # 10:00 EST = 15:00 UTC
    ts = datetime(2026, 1, 15, 15, 0, tzinfo=timezone.utc)
    assert ms.us_market_session(ts) == "regular"


def test_session_post_est():
    # 17:00 EST = 22:00 UTC
    ts = datetime(2026, 1, 15, 22, 0, tzinfo=timezone.utc)
    assert ms.us_market_session(ts) == "post"


def test_session_pre_market_edt_dst():
    # 2026-07-15 (Mi) 08:00 America/New_York (EDT, UTC-4) = 12:00 UTC
    ts = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)
    assert ms.us_market_session(ts) == "pre"


def test_session_regular_edt_dst():
    # 10:00 EDT = 14:00 UTC
    ts = datetime(2026, 7, 15, 14, 0, tzinfo=timezone.utc)
    assert ms.us_market_session(ts) == "regular"


def test_session_post_edt_dst():
    # 17:00 EDT = 21:00 UTC
    ts = datetime(2026, 7, 15, 21, 0, tzinfo=timezone.utc)
    assert ms.us_market_session(ts) == "post"


def test_session_weekend_closed_even_during_market_hours():
    # 2026-01-17 ist ein Samstag; 15:00 UTC wäre unter der Woche "regular"
    ts = datetime(2026, 1, 17, 15, 0, tzinfo=timezone.utc)
    assert ms.us_market_session(ts) == "closed"


def test_session_sunday_closed():
    ts = datetime(2026, 1, 18, 15, 0, tzinfo=timezone.utc)
    assert ms.us_market_session(ts) == "closed"


def test_session_naive_datetime_treated_as_utc():
    ts = datetime(2026, 1, 15, 15, 0)  # kein tzinfo
    assert ms.us_market_session(ts) == "regular"


def test_session_never_raises_on_garbage():
    assert ms.us_market_session(None) == "closed"


# ── fetch_underlying_quotes ───────────────────────────────────────────────────

def test_fetch_underlying_quotes_no_api_key_returns_empty(monkeypatch):
    monkeypatch.delenv("TRADIER_API_KEY", raising=False)
    called = {"n": 0}
    def spy(symbols):
        called["n"] += 1
        return []
    monkeypatch.setattr(ms, "_fetch_quotes_raw", spy)
    out = ms.fetch_underlying_quotes(["AAPL", "MSFT"])
    assert out == {}
    assert called["n"] == 0  # kein Netzwerk-Call ohne Key


def test_fetch_underlying_quotes_batches_by_chunk(monkeypatch):
    monkeypatch.setenv("TRADIER_API_KEY", "test-key")
    calls = []
    def fake_fetch(symbols):
        calls.append(list(symbols))
        return [{"symbol": s, "bid": 10.0, "ask": 10.2, "last": 10.1,
                  "prevclose": 9.9, "open": 10.0} for s in symbols]
    monkeypatch.setattr(ms, "_fetch_quotes_raw", fake_fetch)

    tickers = [f"T{i}" for i in range(250)]
    out = ms.fetch_underlying_quotes(tickers)

    assert len(calls) == 3  # 250 / 100 = 3 Chunks
    assert all(len(c) <= ms.QUOTE_CHUNK for c in calls)
    assert len(out) == 250
    assert out["T0"]["mid"] == pytest.approx(10.1)
    assert out["T0"]["source"] == "tradier"


def test_fetch_underlying_quotes_dedupes_tickers(monkeypatch):
    monkeypatch.setenv("TRADIER_API_KEY", "test-key")
    calls = []
    def fake_fetch(symbols):
        calls.append(list(symbols))
        return [{"symbol": "AAPL", "bid": 100.0, "ask": 100.2}]
    monkeypatch.setattr(ms, "_fetch_quotes_raw", fake_fetch)
    ms.fetch_underlying_quotes(["AAPL", "AAPL", "AAPL"])
    assert calls == [["AAPL"]]


def test_fetch_underlying_quotes_never_raises(monkeypatch):
    monkeypatch.setenv("TRADIER_API_KEY", "test-key")
    def boom(symbols):
        raise RuntimeError("network down")
    monkeypatch.setattr(ms, "_fetch_quotes_raw", boom)
    out = ms.fetch_underlying_quotes(["AAPL"])
    assert out == {}


# ── select_contract ───────────────────────────────────────────────────────────

FAKE_CHAIN = [
    {"option_type": "call", "strike": 95.0, "symbol": "AAPL261120C00095000",
     "bid": 8.0, "ask": 8.4, "open_interest": 100,
     "greeks": {"mid_iv": 0.30, "delta": 0.65}},
    {"option_type": "call", "strike": 100.0, "symbol": "AAPL261120C00100000",
     "bid": 5.0, "ask": 5.4, "open_interest": 500,
     "greeks": {"mid_iv": 0.32, "delta": 0.52}},
    {"option_type": "call", "strike": 105.0, "symbol": "AAPL261120C00105000",
     "bid": 3.0, "ask": 3.4, "open_interest": 300,
     "greeks": {"mid_iv": 0.33, "delta": 0.40}},
    {"option_type": "put", "strike": 100.0, "symbol": "AAPL261120P00100000",
     "bid": 4.0, "ask": 4.4, "open_interest": 200,
     "greeks": {"mid_iv": 0.31, "delta": -0.48}},
]


def test_select_contract_picks_first_expiry_meeting_dte_floor(monkeypatch):
    monkeypatch.setenv("TRADIER_API_KEY", "test-key")
    today = datetime.now(timezone.utc).date()
    near = (today + __import__("datetime").timedelta(days=10)).strftime("%Y-%m-%d")
    far  = (today + __import__("datetime").timedelta(days=130)).strftime("%Y-%m-%d")
    monkeypatch.setattr(ms, "_fetch_expirations", lambda t: [near, far])

    seen_expirations = []
    def fake_chain(t, expiration):
        seen_expirations.append(expiration)
        return FAKE_CHAIN
    monkeypatch.setattr(ms, "_fetch_chain", fake_chain)

    result = ms.select_contract("AAPL", "BULLISH", dte_floor=120, spot=101.0)
    assert result is not None
    assert result["expiry"] == far  # near (10 DTE) liegt unter dem Floor
    assert seen_expirations == [far]  # near-Expiry wurde nie gefetcht


def test_select_contract_nearest_strike_call_bullish(monkeypatch):
    monkeypatch.setenv("TRADIER_API_KEY", "test-key")
    exp = (datetime.now(timezone.utc).date() + __import__("datetime").timedelta(days=130)).strftime("%Y-%m-%d")
    monkeypatch.setattr(ms, "_fetch_expirations", lambda t: [exp])
    monkeypatch.setattr(ms, "_fetch_chain", lambda t, e: FAKE_CHAIN)

    result = ms.select_contract("AAPL", "BULLISH", dte_floor=120, spot=101.0)
    assert result["strike"] == 100.0  # näher an 101 als 95/105
    assert result["symbol"] == "AAPL261120C00100000"
    assert result["bid"] == 5.0 and result["ask"] == 5.4
    assert result["mid"] == pytest.approx(5.2)
    assert result["iv"] == pytest.approx(0.32)
    assert result["delta"] == pytest.approx(0.52)
    assert result["open_interest"] == 500


def test_select_contract_put_for_bearish(monkeypatch):
    monkeypatch.setenv("TRADIER_API_KEY", "test-key")
    exp = (datetime.now(timezone.utc).date() + __import__("datetime").timedelta(days=130)).strftime("%Y-%m-%d")
    monkeypatch.setattr(ms, "_fetch_expirations", lambda t: [exp])
    monkeypatch.setattr(ms, "_fetch_chain", lambda t, e: FAKE_CHAIN)

    result = ms.select_contract("XOM", "BEARISH", dte_floor=120, spot=101.0)
    assert result["symbol"] == "AAPL261120P00100000"
    assert result["strike"] == 100.0


def test_select_contract_no_api_key_returns_none(monkeypatch):
    monkeypatch.delenv("TRADIER_API_KEY", raising=False)
    called = {"n": 0}
    monkeypatch.setattr(ms, "_fetch_expirations", lambda t: (called.__setitem__("n", called["n"] + 1), [])[1])
    result = ms.select_contract("AAPL", "BULLISH", dte_floor=120, spot=100.0)
    assert result is None
    assert called["n"] == 0


def test_select_contract_unknown_direction_returns_none(monkeypatch):
    monkeypatch.setenv("TRADIER_API_KEY", "test-key")
    result = ms.select_contract("AAPL", None, dte_floor=120, spot=100.0)
    assert result is None


def test_select_contract_no_matching_expiry_returns_none(monkeypatch):
    monkeypatch.setenv("TRADIER_API_KEY", "test-key")
    near = (datetime.now(timezone.utc).date() + __import__("datetime").timedelta(days=5)).strftime("%Y-%m-%d")
    monkeypatch.setattr(ms, "_fetch_expirations", lambda t: [near])
    result = ms.select_contract("AAPL", "BULLISH", dte_floor=120, spot=100.0)
    assert result is None


def test_select_contract_never_raises(monkeypatch):
    monkeypatch.setenv("TRADIER_API_KEY", "test-key")
    def boom(t):
        raise RuntimeError("network down")
    monkeypatch.setattr(ms, "_fetch_expirations", boom)
    result = ms.select_contract("AAPL", "BULLISH", dte_floor=120, spot=100.0)
    assert result is None


# ── fetch_option_quotes ───────────────────────────────────────────────────────

def test_fetch_option_quotes_batched(monkeypatch):
    monkeypatch.setenv("TRADIER_API_KEY", "test-key")
    def fake_fetch(symbols):
        return [{"symbol": s, "bid": 1.0, "ask": 1.2} for s in symbols]
    monkeypatch.setattr(ms, "_fetch_quotes_raw", fake_fetch)
    out = ms.fetch_option_quotes(["AAPL261120C00100000"])
    assert out["AAPL261120C00100000"]["bid"] == 1.0
    assert out["AAPL261120C00100000"]["mid"] == pytest.approx(1.1)


def test_fetch_option_quotes_no_api_key_returns_empty(monkeypatch):
    monkeypatch.delenv("TRADIER_API_KEY", raising=False)
    out = ms.fetch_option_quotes(["AAPL261120C00100000"])
    assert out == {}

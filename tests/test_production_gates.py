"""Audit P1-8: Verhaltenstests für Produktionsmodule der täglichen Mail
(risk_gates, prescreener) – mit Negativfällen, ohne Netz (Fakes)."""
from __future__ import annotations

import json
import types
from datetime import date, timedelta

import pandas as pd
import pytest

from modules import prescreener as ps
from modules import risk_gates as rg


# ── risk_gates ──────────────────────────────────────────────────────────────

class _FakeTicker:
    def __init__(self, hist=None, info=None, calendar=None, raise_hist=False):
        self._hist, self.info, self.calendar, self._raise = hist, info or {}, calendar, raise_hist

    def history(self, **k):
        if self._raise:
            raise TimeoutError("curl 28")
        return self._hist if self._hist is not None else pd.DataFrame()


def _vix(monkeypatch, value=None, fred=None, raise_hist=False):
    hist = pd.DataFrame({"Close": [value - 1, value]}) if value is not None else pd.DataFrame()
    monkeypatch.setattr(rg.yf, "Ticker", lambda s: _FakeTicker(hist=hist, raise_hist=raise_hist))

    def fake_get(url, **k):
        if fred is None:
            raise ConnectionError("offline")
        return types.SimpleNamespace(status_code=200, text=f"DATE,VIXCLS\n2026-10-01,{fred}\n")
    monkeypatch.setattr(rg.requests, "get", fake_get)


def test_vix_below_gate_allows_trading(monkeypatch):
    _vix(monkeypatch, value=18.0)
    g = rg.RiskGates()
    assert g.global_ok() is True and g.last_vix == 18.0


def test_vix_at_or_above_hard_gate_blocks(monkeypatch):
    _vix(monkeypatch, value=rg.VIX_HARD_GATE)
    g = rg.RiskGates()
    assert g.global_ok() is False and g.last_vix == rg.VIX_HARD_GATE


def test_vix_falls_back_to_fred_then_blocks_if_extreme(monkeypatch):
    _vix(monkeypatch, value=None, fred="41.5", raise_hist=True)
    g = rg.RiskGates()
    assert g.global_ok() is False and g.last_vix == 41.5


def test_vix_unavailable_is_fail_open_with_fallback_value(monkeypatch):
    """Dokumentiertes (bewusstes) Verhalten: ohne VIX läuft die Pipeline mit
    Fallback 20 weiter. Wird das geändert, muss dieser Test bewusst angepasst werden."""
    _vix(monkeypatch, value=None, fred=None, raise_hist=True)
    g = rg.RiskGates()
    assert g.global_ok() is True and g.last_vix == rg.VIX_FALLBACK


def test_earnings_gate_window(monkeypatch):
    soon = pd.DataFrame({0: [pd.Timestamp(date.today() + timedelta(days=5))]}, index=["Earnings Date"])
    late = pd.DataFrame({0: [pd.Timestamp(date.today() + timedelta(days=40))]}, index=["Earnings Date"])
    past = pd.DataFrame({0: [pd.Timestamp(date.today() - timedelta(days=2))]}, index=["Earnings Date"])
    for cal, expect in ((soon, True), (late, False), (past, False), (pd.DataFrame(), False), (None, False)):
        monkeypatch.setattr(rg.yf, "Ticker", lambda s, c=cal: _FakeTicker(calendar=c))
        assert rg.RiskGates().has_upcoming_earnings("X") is expect


def test_earnings_gate_error_does_not_block(monkeypatch):
    def boom(s):
        raise RuntimeError("yahoo down")
    monkeypatch.setattr(rg.yf, "Ticker", boom)
    assert rg.RiskGates().has_upcoming_earnings("X") is False


# ── prescreener ─────────────────────────────────────────────────────────────

class _Resp:
    def __init__(self, text):
        self.content = [types.SimpleNamespace(text=text)]


class _Client:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        self.messages = self

    def create(self, **k):
        self.calls += 1
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return _Resp(r)


def _screener(monkeypatch, replies, liquid=lambda t: True):
    monkeypatch.setattr(ps.anthropic, "Anthropic", lambda **k: _Client(replies))
    monkeypatch.setattr(ps.time, "sleep", lambda s: None)
    s = ps.Prescreener()
    monkeypatch.setattr(s, "_has_options_liquidity", liquid)
    return s


def _cands(n=3):
    return [{"ticker": f"T{i}", "news": [f"news {i}"]} for i in range(n)]


def _json(results):
    return json.dumps({"results": results})


def test_prescreener_keeps_only_valid_yes(monkeypatch):
    out = _json([{"ticker": "T0", "decision": "[YES]", "category": "structural_change", "reason": "a"},
                 {"ticker": "T1", "decision": "[YES]", "category": "routine_news", "reason": "b"},     # Override
                 {"ticker": "T2", "decision": "[NO]", "category": "earnings", "reason": "c"}])
    s = _screener(monkeypatch, [out])
    res = s.run(_cands())
    assert [c["ticker"] for c in res] == ["T0"] and res[0]["prescreen_category"] == "structural_change"
    assert s.failed_tickers == []


def test_prescreener_illiquid_options_rejected(monkeypatch):
    out = _json([{"ticker": "T0", "decision": "[YES]", "category": "catalyst", "reason": "a"}])
    s = _screener(monkeypatch, [out], liquid=lambda t: False)
    assert s.run(_cands(1)) == []


def test_prescreener_parses_fenced_json_and_retries_parse_errors(monkeypatch):
    good = "```json\n" + _json([{"ticker": "T0", "decision": "[YES]", "category": "catalyst"}]) + "\n```"
    s = _screener(monkeypatch, ["kein json", good])
    assert [c["ticker"] for c in s.run(_cands(1))] == ["T0"]
    assert s.client.calls == 2


def test_prescreener_api_outage_is_not_a_no_signal(monkeypatch):
    s = _screener(monkeypatch, [RuntimeError("529")] * ps.MAX_RETRIES)
    assert s.run(_cands(2)) == []
    assert s.failed_tickers == ["T0", "T1"]                     # als API-Fehler gezählt, nicht als NO


def test_prescreener_empty_input(monkeypatch):
    assert _screener(monkeypatch, []).run([]) == []


def test_prescreener_batches(monkeypatch):
    n = ps.BATCH_SIZE + 5
    replies = [_json([]), _json([{"ticker": f"T{n - 1}", "decision": "[YES]", "category": "catalyst"}])]
    s = _screener(monkeypatch, replies)
    assert [c["ticker"] for c in s.run(_cands(n))] == [f"T{n - 1}"] and s.client.calls == 2


def test_options_liquidity_check(monkeypatch):
    s = ps.Prescreener.__new__(ps.Prescreener)
    import yfinance as yf
    monkeypatch.setattr(yf, "Ticker", lambda t: types.SimpleNamespace(options=("2026-10-16", "2026-11-20")))
    assert s._has_options_liquidity("X") is True
    monkeypatch.setattr(yf, "Ticker", lambda t: types.SimpleNamespace(options=("2026-10-16",)))
    assert s._has_options_liquidity("X") is False

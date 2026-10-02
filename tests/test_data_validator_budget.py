"""Regression: Lauf 2026-10-01 verbrachte 26 min still in der EPS-Gegenprüfung
(company_tickers.json je Kandidat, EDGAR-Fehler -> Alpha-Vantage-Fallback mit
12,5 s Zwangspause je Kandidat) -> Deep Analysis 0 Kandidaten."""
import importlib

import modules.data_validator as dv


class _Resp:
    def __init__(self, status, payload=None):
        self.status_code = status
        self._p = payload or {}

    def json(self):
        return self._p


def _fresh(monkeypatch):
    importlib.reload(dv)
    monkeypatch.setattr(dv.time, "sleep", lambda s: None)
    return dv


def test_ticker_map_loaded_once(monkeypatch):
    m = _fresh(monkeypatch)
    calls = []

    def fake_get(url, **kw):
        calls.append(url)
        if url.endswith("company_tickers.json"):
            return _Resp(200, {"0": {"ticker": "AAA", "cik_str": 1},
                               "1": {"ticker": "BBB", "cik_str": 2}})
        return _Resp(200, {"facts": {}})

    monkeypatch.setattr(m.requests, "get", fake_get)
    for t in ["AAA", "BBB", "ZZZ", "YYY"]:
        m.fetch_eps_sec_edgar(t)
    assert sum(u.endswith("company_tickers.json") for u in calls) == 1


def test_edgar_circuit_breaker_and_av_cap(monkeypatch):
    m = _fresh(monkeypatch)
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "x")
    calls = {"sec": 0, "av": 0}

    def fake_get(url, **kw):
        if "sec.gov" in url:
            calls["sec"] += 1
            return _Resp(403)
        calls["av"] += 1
        return _Resp(200, {"EPS": "1.0"})

    monkeypatch.setattr(m.requests, "get", fake_get)
    for i in range(50):
        r = m.validate_candidate_data({"ticker": f"T{i}", "info": {"trailingEps": 1.0}})
        assert "confidence" in r["data_validation"]["eps_cross_check"]
    assert calls["sec"] <= m._SEC_MAX_FAILS
    assert calls["av"] <= m._AV_MAX_CALLS

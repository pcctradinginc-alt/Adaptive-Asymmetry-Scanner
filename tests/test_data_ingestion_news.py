"""Regression 2026-09-28: Finnhub-429 ohne Drossel + kaputter yfinance-News-
Fallback ließen den Hard-Filter fast nur Ticker A–E durch (Log 2026-09-25:
116 von 126 Kandidaten). Keine Netzwerkzugriffe."""
from __future__ import annotations

from datetime import datetime, timedelta

from modules import data_ingestion as di


class _Clock:
    def __init__(self):
        self.t = 0.0
        self.sleeps = []

    def now(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


def test_rate_limiter_never_exceeds_budget_per_window():
    c = _Clock()
    rl = di._RateLimiter(3, 60.0, clock=c.now, sleep=c.sleep)
    stamps = []
    for _ in range(7):
        rl.acquire()
        stamps.append(c.t)
    for s in stamps:
        assert sum(1 for x in stamps if s <= x < s + 60.0) <= 3
    assert c.sleeps, "Limiter hat nie gewartet"


def test_yfinance_news_new_and_old_format():
    now = datetime(2026, 9, 28, 18, 0)
    since = now - timedelta(hours=48)
    items = [
        {"id": "a", "content": {"title": "New format recent", "pubDate": "2026-09-28T12:00:00Z"}},
        {"id": "b", "content": {"title": "New format old", "pubDate": "2026-09-20T12:00:00Z"}},
        {"title": "Old format recent", "providerPublishTime": int((now - timedelta(hours=1)).timestamp())},
        {"content": {"title": "No date"}},
        "garbage",
    ]
    assert di.parse_yfinance_news(items, since) == ["New format recent", "Old format recent"]


class _Resp:
    def __init__(self, status, payload=None, headers=None):
        self.status_code, self._payload, self.headers = status, payload, headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def json(self):
        return self._payload


def test_finnhub_429_is_retried_and_counted(monkeypatch):
    calls = []
    seq = [_Resp(429, headers={"Retry-After": "1"}), _Resp(200, [{"headline": "H1"}, {"headline": "H2"}])]

    def fake_get(*a, **k):
        calls.append(1)
        return seq[len(calls) - 1]

    monkeypatch.setattr(di.requests, "get", fake_get)
    monkeypatch.setattr(di.time, "sleep", lambda s: None)
    monkeypatch.setattr(di, "_FINNHUB_LIMITER", di._RateLimiter(1000, 60.0))
    ing = di.DataIngestion.__new__(di.DataIngestion)
    ing.news_source_stats = {}
    assert ing._fetch_finnhub_news("ZTS", "key") == ["H1", "H2"]
    assert len(calls) == 2
    assert ing.news_source_stats == {"finnhub_429": 1, "finnhub_ok": 1}


def test_low_rv_rejected_without_news_call(monkeypatch):
    """RV unter Schwelle UND unter dem Override-Minimum -> kein News-Abruf."""
    info = {"marketCap": 5e10, "averageVolume": 5e6, "currentPrice": 100.0, "volume": 5e5}  # RV 0.1

    class _T:
        def __init__(self, t):
            self.info = info

    monkeypatch.setattr(di.yf, "Ticker", _T)
    ing = di.DataIngestion.__new__(di.DataIngestion)
    called = []
    monkeypatch.setattr(ing, "_fetch_news", lambda *a: called.append(1) or ["x"], raising=False)
    res, st = ing._evaluate_ticker("ZZZ", {}, 20.0)
    assert res is None and st["rel_volume"] == 1 and not called


def test_news_override_still_possible_between_override_min_and_threshold(monkeypatch):
    info = {"marketCap": 5e10, "averageVolume": 5e6, "currentPrice": 100.0, "volume": 1.5e6}  # RV 0.3

    class _T:
        def __init__(self, t):
            self.info = info

    monkeypatch.setattr(di.yf, "Ticker", _T)
    ing = di.DataIngestion.__new__(di.DataIngestion)
    monkeypatch.setattr(ing, "_fetch_news", lambda *a: ["a", "b", "c"], raising=False)
    res, st = ing._evaluate_ticker("ZZZ", {}, 20.0)
    assert res is not None and res["ticker"] == "ZZZ"


def test_newsapi_company_name_uses_full_name():
    assert di.newsapi_company_name("Bank of America Corporation") == "Bank of America"
    assert di.newsapi_company_name("The Home Depot, Inc.") == "Home Depot"
    assert di.newsapi_company_name("3M Company") is None           # zu kurz -> Ticker-Suche
    assert di.newsapi_company_name("Apple Inc.") == "Apple"

"""fred_liquidity (B4): RRPONTSYD / WTREGEN / T10Y2Y als ALFRED-Vintages, RESEARCH_SHADOW. http.fetch gemockt."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import patch

from modules.external import http
from modules.external.pit import utc_now
from modules.external.registry import SourceRegistry, discover_connectors, gate_source
from modules.external.sources import real_economy as re_
from modules.external.sources.base import SourceStatus

UTC = timezone.utc
EXPECTED = {"RRPONTSYD": ("us_on_rrp", "usd_billions"), "WTREGEN": ("us_treasury_general_account", "usd_millions"),
            "T10Y2Y": ("us_curve_10y_2y", "percentage_points")}


def _alfred(rows, params):
    a, b = params["realtime_start"], params["realtime_end"]
    content = json.dumps({"observations": [{"date": d, "realtime_start": max(rs, a), "realtime_end": min(re, b),
                                            "value": v} for d, rs, re, v in rows if rs <= b and re >= a]}).encode()
    return http.FetchResult(url="https://fixture/alfred", status=200, content=content,
                            content_type="application/json", retrieved_at=utc_now(), content_hash="h",
                            fingerprint="fp", bytes=len(content))


def test_registered_research_only_and_auth_gated(monkeypatch):
    reg = SourceRegistry()
    cfg = reg.sources["fred_liquidity"]
    assert cfg["role"] == "RESEARCH_SHADOW" and cfg["research_only"] is True and cfg["criticality"] == "low"
    assert discover_connectors()["fred_liquidity"] is re_.FredLiquidityConnector
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    assert gate_source(cfg)[0] is False
    with patch.object(re_.http, "fetch") as m:
        assert re_.FredLiquidityConnector(cfg).fetch(utc_now()).status == SourceStatus.AUTH_MISSING
    m.assert_not_called()


def test_vintages_windows_and_missing_values(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "k")
    calls = []

    def fake(url, params=None, **kw):
        calls.append(dict(params))
        return _alfred([("2026-10-01", "2026-10-02", "9999-12-31", "512.3"),
                        ("2026-10-02", "2026-10-05", "9999-12-31", ".")], params)

    with patch.object(re_.http, "fetch", side_effect=fake):
        res = re_.FredLiquidityConnector(SourceRegistry().sources["fred_liquidity"]).fetch(
            datetime(2026, 10, 15, tzinfo=UTC))
    assert res.status == SourceStatus.PASS
    assert {c["series_id"] for c in calls} == set(EXPECTED)
    assert all(c["observation_start"] == "2021-01-01" for c in calls)
    assert {(c["realtime_start"], c["realtime_end"]) for c in calls} == {("2021-01-01", "2025-12-31"),
                                                                         ("2026-01-01", "9999-12-31")}
    for key, (metric, unit) in EXPECTED.items():
        obs = sorted((o for o in res.observations if o.metric == metric), key=lambda o: o.observation_time)
        assert [o.value for o in obs] == [512.3, None], key                     # fehlend bleibt None, nie 0
        assert obs[0].unit == unit and obs[0].series_id == key and obs[0].entity_id == "US"
        assert obs[0].available_at == datetime(2026, 10, 2, tzinfo=UTC)          # realtime_start, kein Look-ahead

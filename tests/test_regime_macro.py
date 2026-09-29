"""fred_regime_macro + modules/external/regime.py: Veröffentlichungszeitpunkte
(ALFRED-Vintages), Revisionen ohne Look-ahead, Publication Lag, Frische,
Regime-Labels. Keine Netzwerkzugriffe."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from modules.external import http, regime
from modules.external.pit import AvailabilityPrecision, Observation, utc_now
from modules.external.registry import SourceRegistry
from modules.external.sources import real_economy as re_
from modules.external.sources.base import SourceStatus

UTC = timezone.utc


def _alfred(rows):
    content = json.dumps({"observations": [
        {"date": d, "realtime_start": rs, "realtime_end": re, "value": v} for d, rs, re, v in rows]}).encode()
    return http.FetchResult(url="https://fixture/alfred", status=200, content=content,
                            content_type="application/json", retrieved_at=utc_now(),
                            content_hash="h", fingerprint="fp", bytes=len(content))


def test_connector_fetches_all_six_series_with_vintages_and_units(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "k")
    calls = []

    def fake(url, params=None, **kw):
        calls.append(params)
        return _alfred([("2026-07-01", "2026-08-12", "2026-09-10", "321.5"),
                        ("2026-07-01", "2026-09-11", "9999-12-31", "321.9"),     # Revision
                        ("2026-08-01", "2026-09-11", "9999-12-31", ".")])        # fehlend
    with patch.object(re_.http, "fetch", side_effect=fake):
        res = re_.FredRegimeMacroConnector(SourceRegistry().sources["fred_regime_macro"]).fetch(
            datetime(2026, 9, 29, tzinfo=UTC))
    assert res.status == SourceStatus.PASS
    assert [c["series_id"] for c in calls] == ["CPIAUCSL", "NFCICREDIT", "NFCI", "WALCL", "DTWEXBGS", "DCOILWTICO"]
    by = {c["series_id"]: c for c in calls}
    assert by["CPIAUCSL"]["observation_start"] == "2015-01-01" and by["CPIAUCSL"]["realtime_start"] == "1776-07-04"
    assert by["NFCI"]["realtime_start"] == "2025-01-01" and by["NFCICREDIT"]["observation_start"] == "2025-01-01"
    cpi = [o for o in res.observations if o.metric == "us_cpi"]
    assert len(cpi) == 3 and cpi[0].unit == "index_1982_84"
    assert cpi[0].available_at == datetime(2026, 8, 12, tzinfo=UTC)          # Veröffentlichungstag
    assert cpi[1].value == 321.9 and cpi[1].vintage_time == datetime(2026, 9, 11, tzinfo=UTC)
    assert cpi[2].value is None                                             # "." nie als 0
    assert {o.unit for o in res.observations if o.metric == "wti"} == {"usd_per_barrel"}


def test_without_key_no_http_call(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with patch.object(re_.http, "fetch") as m:
        assert re_.FredRegimeMacroConnector({}).fetch(utc_now()).status == SourceStatus.AUTH_MISSING
    m.assert_not_called()


def _o(metric, period, value, available):
    return Observation(source_id="fred_regime_macro", dataset="d", series_id=metric, entity_id="US",
                       metric=metric, value=value, unit="u", observation_time=period,
                       available_at=available, retrieved_at=available,
                       availability_precision=AvailabilityPrecision.EXACT_DATE, parser_version="1",
                       vintage_time=available)


def _cpi_history(yoy_now=0.04, revision=None):
    """CPI monatlich 2024-01..2026-07, veröffentlicht ~12 Tage nach Monatsende."""
    obs = []
    for i in range(31):
        y, m = 2024 + i // 12, 1 + i % 12
        period = datetime(y, m, 1, tzinfo=UTC)
        val = 300.0 * (1 + yoy_now) ** (i / 12)
        pub = (period + timedelta(days=42))
        obs.append(_o("us_cpi", period, val, pub))
    if revision is not None:
        last = obs[-1]
        obs.append(_o("us_cpi", last.observation_time, revision[0], revision[1]))
    return obs


def test_publication_lag_unreleased_period_is_invisible():
    obs = _cpi_history()
    july = datetime(2026, 7, 1, tzinfo=UTC)
    before = regime.pit_series(obs, "us_cpi", july + timedelta(days=41))
    after = regime.pit_series(obs, "us_cpi", july + timedelta(days=42))
    assert before[-1][0] < july and after[-1][0] == july


def test_revision_not_visible_before_its_vintage():
    july = datetime(2026, 7, 1, tzinfo=UTC)
    obs = _cpi_history(revision=(999.0, datetime(2026, 9, 20, tzinfo=UTC)))
    assert regime.pit_series(obs, "us_cpi", datetime(2026, 9, 1, tzinfo=UTC))[-1][1] != 999.0
    assert regime.pit_series(obs, "us_cpi", datetime(2026, 9, 21, tzinfo=UTC))[-1] == (july, 999.0)


def test_inflation_labels_and_staleness():
    obs = _cpi_history(yoy_now=0.04)
    st = regime.regime_state(obs, datetime(2026, 8, 20, tzinfo=UTC))
    assert abs(st["cpi_yoy"] - 0.04) < 1e-6 and st["labels"]["inflation"] == "high_inflation"
    assert regime.regime_state(obs, datetime(2027, 3, 1, tzinfo=UTC))["labels"] == {}   # veraltet -> kein Regime


def test_daily_series_labels():
    obs, base = [], datetime(2026, 1, 1, tzinfo=UTC)
    for i in range(100):
        t = base + timedelta(days=i)
        obs.append(_o("wti", t, 60.0 + i * 0.1, t + timedelta(days=1)))
        obs.append(_o("usd_broad", t, 120.0 - i * 0.05, t + timedelta(days=1)))
    for i in range(20):
        t = base + timedelta(weeks=i)
        obs.append(_o("fed_total_assets", t, 7_000_000 - i * 10_000, t + timedelta(days=1)))
        obs.append(_o("nfci_credit", t, 0.3, t + timedelta(days=5)))
        obs.append(_o("nfci", t, -0.4, t + timedelta(days=5)))
    lab = regime.regime_state(obs, base + timedelta(days=101))["labels"]
    assert lab == {"oil": "oil_up", "dollar": "dollar_down", "liquidity": "contracting",
                   "credit": "credit_tight", "fin_conditions": "loose"}


def test_regimes_by_date_uses_only_prior_publications():
    obs = _cpi_history()
    july_pub = datetime(2026, 7, 1, tzinfo=UTC) + timedelta(days=42)
    d = july_pub.date().isoformat()                 # am Veröffentlichungstag selbst noch unbekannt
    st_prev = regime.regime_state(obs, july_pub - timedelta(seconds=1))
    assert regime.regimes_by_date(obs, [d])[d] == st_prev["labels"]


def test_registry_entry_is_licensed_and_frequency_checked():
    c = SourceRegistry().sources["fred_regime_macro"]
    assert c["license_status"] == "OK" and c["enabled"] and c["requires_auth"]
    assert set(c["expected_series_ids"]) == {"CPIAUCSL", "NFCICREDIT", "NFCI", "WALCL", "DTWEXBGS", "DCOILWTICO"}
    assert "BAMLH0A0HYM2" not in json.dumps(c)       # ICE-Lizenz: bewusst nicht

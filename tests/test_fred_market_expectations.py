"""fred_market_expectations (modules/external/sources/real_economy.py):
markt-implizite Erwartungsreihen (Treasury 2J/10J, Fed Funds, Breakevens) als
ALFRED-Vintages. Keine Netzwerkzugriffe: http.fetch wird gemockt."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from modules.external import http
from modules.external.pit import AvailabilityPrecision, utc_now
from modules.external.registry import (
    REQUIRED_FIELDS, SourceRegistry, discover_connectors, gate_source, load_source_configs,
)
from modules.external.sources import real_economy as re_
from modules.external.sources.base import SourceStatus

UTC = timezone.utc
ROOT = Path(__file__).resolve().parent.parent
SOURCE_ID = "fred_market_expectations"

# series key -> (metric, dataset, unit) laut Spezifikation
EXPECTED = {
    "DGS2":   ("us_treasury_2y", "rates", "percent"),
    "DGS10":  ("us_treasury_10y", "rates", "percent"),
    "DFF":    ("us_fed_funds_effective", "policy", "percent"),
    "T5YIE":  ("us_breakeven_5y", "inflation_expectations", "percent"),
    "T10YIE": ("us_breakeven_10y", "inflation_expectations", "percent"),
    "T5YIFR": ("us_inflation_5y5y_forward", "inflation_expectations", "percent"),
}


def _alfred(rows):
    """rows: (date, realtime_start, realtime_end, value) -> ALFRED-JSON-Antwort."""
    content = json.dumps({"observations": [
        {"date": d, "realtime_start": rs, "realtime_end": re, "value": v} for d, rs, re, v in rows]}).encode()
    return http.FetchResult(url="https://fixture/alfred", status=200, content=content,
                            content_type="application/json", retrieved_at=utc_now(),
                            content_hash="h", fingerprint="fp", bytes=len(content))


def _connector():
    return re_.FredMarketExpectationsConnector(SourceRegistry().sources[SOURCE_ID])


def test_without_key_auth_missing_and_no_http_call(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with patch.object(re_.http, "fetch") as m:
        res = re_.FredMarketExpectationsConnector({}).fetch(utc_now())
    assert res.status == SourceStatus.AUTH_MISSING
    assert res.observations == []
    m.assert_not_called()


def test_vintages_of_revised_observation_and_missing_value(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "k")

    def fake(url, params=None, **kw):
        return _alfred([
            ("2026-09-01", "2026-09-02", "2026-09-09", "4.10"),     # Erstveröffentlichung
            ("2026-09-01", "2026-09-10", "9999-12-31", "4.12"),     # Revision (eigener Vintage)
            ("2026-09-07", "2026-09-08", "9999-12-31", "."),        # Feiertag/fehlend
        ])
    with patch.object(re_.http, "fetch", side_effect=fake):
        res = _connector().fetch(datetime(2026, 10, 1, tzinfo=UTC))

    assert res.status == SourceStatus.PASS
    for key, (metric, dataset, unit) in EXPECTED.items():
        obs = [o for o in res.observations if o.metric == metric]
        # zwei Vintages derselben Beobachtung + ein fehlender Wert je Serie
        assert len(obs) == 3, key
        first, revised, missing = obs
        assert {o.dataset for o in obs} == {dataset} and {o.unit for o in obs} == {unit}
        assert {o.series_id for o in obs} == {key}
        assert first.source_id == SOURCE_ID and first.entity_id == "US"
        assert first.value == 4.10 and revised.value == 4.12
        assert first.observation_time == revised.observation_time == datetime(2026, 9, 1, tzinfo=UTC)
        # PIT: available_at = vintage_time = realtime_start (nie früher)
        assert first.available_at == first.vintage_time == datetime(2026, 9, 2, tzinfo=UTC)
        assert revised.available_at == revised.vintage_time == datetime(2026, 9, 10, tzinfo=UTC)
        assert first.availability_precision == AvailabilityPrecision.EXACT_DATE
        assert first.attrs["realtime_end"] == "2026-09-09"
        # "." ist fehlend -> None, niemals 0
        assert missing.value is None
        assert missing.available_at == datetime(2026, 9, 8, tzinfo=UTC)


def test_requests_all_six_series_with_observation_start_and_full_realtime_range(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "k")
    calls = []

    def fake(url, params=None, **kw):
        calls.append((url, dict(params)))
        return _alfred([("2026-09-01", "2026-09-02", "9999-12-31", "3.5")])
    with patch.object(re_.http, "fetch", side_effect=fake):
        res = _connector().fetch(datetime(2026, 10, 1, tzinfo=UTC))

    assert res.status == SourceStatus.PASS
    assert [p["series_id"] for _, p in calls] == list(EXPECTED)
    for url, p in calls:
        assert url.endswith("/series/observations")
        assert p["observation_start"] == "2015-01-01"
        assert p["realtime_start"] == "1776-07-04" and p["realtime_end"] == "9999-12-31"
        assert p["api_key"] == "k" and p["file_type"] == "json"
    assert res.latest_observation_time == datetime(2026, 9, 1, tzinfo=UTC)
    assert len(res.observations) == 6


def test_connector_class_constants():
    cls = re_.FredMarketExpectationsConnector
    assert issubclass(cls, re_.FredUsMacroConnector)
    assert cls.source_id == SOURCE_ID
    assert cls.OBSERVATION_START == "2015-01-01"
    assert cls.LABEL == "DGS2/DGS10/DFF/T5YIE/T10YIE/T5YIFR"
    assert list(cls.SERIES) == list(EXPECTED)
    for key, (metric, dataset, unit) in EXPECTED.items():
        spec = cls.SERIES[key]
        assert (spec["metric"], spec["dataset"], spec["unit"]) == (metric, dataset, unit)
    assert cls.SERIES["DFF"]["search_text"] == "Federal Funds Effective Rate"
    assert cls.SERIES["T5YIFR"]["search_text"] == "5-Year, 5-Year Forward Inflation Expectation Rate"
    # bestehende Konnektoren unverändert (Registrierung additiv)
    assert re_.CONNECTORS["fred_world_macro"] is re_.FredWorldMacroConnector
    assert re_.CONNECTORS["fred_regime_macro"] is re_.FredRegimeMacroConnector


def test_connector_registered_and_yaml_entry_validates():
    assert re_.CONNECTORS[SOURCE_ID] is re_.FredMarketExpectationsConnector
    assert discover_connectors()[SOURCE_ID] is re_.FredMarketExpectationsConnector

    # lädt + validiert ALLE config/external_sources/*.yaml (Pflichtfelder, license_status, criticality)
    configs = load_source_configs(ROOT / "config" / "external_sources")
    cfg = configs[SOURCE_ID]
    assert all(f in cfg for f in REQUIRED_FIELDS)
    assert cfg["family"] == "real_economy" and cfg["access_method"] == "fred_alfred_api"
    assert cfg["requires_auth"] is True and cfg["auth_env_variable"] == "FRED_API_KEY"
    assert cfg["auth_optional"] is True
    assert cfg["license_status"] == "OK" and cfg["enabled"] is True
    assert cfg["frequency"] == "daily" and cfg["expected_update_cadence"] == "P1D"
    assert cfg["pit_quality"] == "EXACT_DATE"
    assert cfg["supports_history"] and cfg["supports_vintages"] and cfg["supports_release_time"]
    assert cfg["max_staleness_days"] == 7 and cfg["role"] == "REGIME"
    assert cfg["point_in_time_capable"] is True and cfg["revision_risk"] == "low"
    # erwartete Serien-IDs decken sich mit dem Konnektor (sonst würde die Discovery greifen)
    assert cfg["expected_series_ids"] == {k: k for k in re_.FredMarketExpectationsConnector.SERIES}

    reg = SourceRegistry()
    assert reg.sources[SOURCE_ID]["family"] == "real_economy"
    assert reg.connector_for(SOURCE_ID) is re_.FredMarketExpectationsConnector
    assert isinstance(reg.build_connector(SOURCE_ID), re_.FredMarketExpectationsConnector)
    assert SOURCE_ID in {s["source_id"] for s in reg.iter_sources("real_economy")}


def test_gate_requires_api_key(monkeypatch):
    cfg = SourceRegistry().sources[SOURCE_ID]
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    assert gate_source(cfg) == (False, "AUTH_MISSING")
    monkeypatch.setenv("FRED_API_KEY", "k")
    assert gate_source(cfg) == (True, "ok")


def test_ice_bofa_spreads_not_archived():
    """ICE-Lizenz: HY-OAS & Co. dürfen NICHT in dieser Quelle landen."""
    assert not any(k.startswith("BAML") for k in re_.FredMarketExpectationsConnector.SERIES)

"""
tests/test_real_economy_connectors.py

Recorded-style Fixture-Tests für modules/external/sources/real_economy.py.
Es findet KEIN echter Netzwerkzugriff statt: `http.fetch` wird pro Test
gemockt und liefert vorbereitete FetchResult-Objekte aus
tests/fixtures/external/real_economy/.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from modules.external import http
from modules.external.pit import AvailabilityPrecision, utc_now
from modules.external.sources import real_economy as re_
from modules.external.sources.base import SourceStatus

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "external" / "real_economy"

NOW = datetime(2026, 8, 30, tzinfo=timezone.utc)


def _fr(name: str, content_type: str = "application/json") -> http.FetchResult:
    content = (FIXTURES / name).read_bytes()
    now = utc_now()
    return http.FetchResult(
        url=f"https://fixture/{name}", status=200, content=content,
        content_type=content_type, retrieved_at=now,
        content_hash="deadbeef", fingerprint="fp", bytes=len(content),
    )


# ---------------------------------------------------------------------------
# _parse_monthly_period: "2026M08" / "2026-08" / "2026-M08"
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,expected", [
    ("2026-08", datetime(2026, 8, 1, tzinfo=timezone.utc)),
    ("2026M08", datetime(2026, 8, 1, tzinfo=timezone.utc)),
    ("2026-M08", datetime(2026, 8, 1, tzinfo=timezone.utc)),
    (" 2026-01 ", datetime(2026, 1, 1, tzinfo=timezone.utc)),
])
def test_parse_monthly_period_accepts_known_formats(label, expected):
    assert re_._parse_monthly_period(label) == expected


@pytest.mark.parametrize("label", ["2026Q1", "2026", "not-a-date", "2026-13", "2026M13"])
def test_parse_monthly_period_rejects_unknown_formats(label):
    with pytest.raises(ValueError):
        re_._parse_monthly_period(label)


# ---------------------------------------------------------------------------
# eurostat_sentiment
# ---------------------------------------------------------------------------

def test_eurostat_sentiment_discovers_dataset_and_selects_metrics_by_label():
    conn = re_.EurostatSentimentConnector({})
    with patch.object(re_.http, "fetch", side_effect=[
        _fr("eurostat_sentiment_toc.txt", content_type="text/plain"),
        _fr("eurostat_sentiment_jsonstat.json"),
    ]):
        result = conn.fetch(NOW)

    assert result.status == SourceStatus.PASS
    assert result.discovered_ids["eurostat_sentiment_chosen"] == "ei_bssi_m_r2"
    metrics = {o.metric for o in result.observations}
    assert metrics == {"esi", "industrial_confidence"}
    # NSA-Zeile (s_adj != SA) und Consumer-confidence-Zeile (falsches indic-
    # Label) dürfen NIE übernommen werden -- nur die 6 gültigen SA/ESI/ICI-
    # Kombinationen.
    assert len(result.observations) == 6
    assert {o.value for o in result.observations} == {101.5, 102.0, 5.0, 6.0, 100.0, 3.0}

    de_esi = [o for o in result.observations
              if o.metric == "esi" and o.entity_id == "DE"
              and o.observation_time == datetime(2026, 8, 1, tzinfo=timezone.utc)]
    assert len(de_esi) == 1
    assert de_esi[0].value == 102.0
    assert de_esi[0].availability_precision == AvailabilityPrecision.EXACT_TIMESTAMP
    assert de_esi[0].available_at == datetime(2026, 8, 30, 8, 0, tzinfo=timezone.utc)


def test_eurostat_sentiment_schema_changed_when_no_metric_label_matches():
    conn = re_.EurostatSentimentConnector({})
    broken = {
        "version": "2.0", "class": "dataset", "id": ["geo", "time"], "size": [1, 1],
        "dimension": {
            "geo": {"category": {"index": {"DE": 0}, "label": {"DE": "Germany"}}},
            "time": {"category": {"index": {"2026-08": 0}, "label": {"2026-08": "2026-08"}}},
        },
        "value": {"0": 42.0},
    }
    import json as jsonmod
    fake = http.FetchResult(
        url="https://fixture/broken", status=200, content=jsonmod.dumps(broken).encode(),
        content_type="application/json", retrieved_at=utc_now(),
        content_hash="x", fingerprint="fp", bytes=10,
    )
    with patch.object(re_.http, "fetch", side_effect=[
        _fr("eurostat_sentiment_toc.txt", content_type="text/plain"),
        fake,
    ]):
        result = conn.fetch(NOW)
    # kein "indic"-Dimension vorhanden -> _metric_fn liefert für jede Zeile
    # None -> keine Beobachtungen, klar SCHEMA_CHANGED statt Stille.
    assert result.status == SourceStatus.SCHEMA_CHANGED
    assert result.observations == []


def test_eurostat_sentiment_toc_unreachable_falls_back_to_expected_code():
    conn = re_.EurostatSentimentConnector({})
    with patch.object(re_.http, "fetch", side_effect=[
        http.FetchError("boom"),
        _fr("eurostat_sentiment_jsonstat.json"),
    ]):
        result = conn.fetch(NOW)
    assert result.status == SourceStatus.PASS
    assert result.discovered_ids["fallback_dataset_code_source"] == (
        "config.expected_dataset_code (verify in preflight)")


# ---------------------------------------------------------------------------
# eurostat_industrial_production
# ---------------------------------------------------------------------------

def test_eurostat_industrial_production_selects_total_industry_excl_construction():
    conn = re_.EurostatIndustrialProductionConnector({})
    with patch.object(re_.http, "fetch", side_effect=[
        _fr("eurostat_industrial_production_toc.txt", content_type="text/plain"),
        _fr("eurostat_industrial_production_jsonstat.json"),
    ]):
        result = conn.fetch(NOW)

    assert result.status == SourceStatus.PASS
    assert result.discovered_ids["eurostat_industrial_production_chosen"] == "sts_inpr_m"
    # Manufacturing-only (nace_r2=C) und Turnover (indic_bt=TOVV) und SA
    # (statt SCA) dürfen NIE übernommen werden.
    assert len(result.observations) == 5
    metrics = {o.metric for o in result.observations}
    assert metrics == {"production_volume_index"}
    de_series = sorted(
        (o.observation_time, o.value) for o in result.observations if o.entity_id == "DE"
    )
    assert de_series == [
        (datetime(2026, 6, 1, tzinfo=timezone.utc), 108.0),
        (datetime(2026, 7, 1, tzinfo=timezone.utc), 109.0),
        (datetime(2026, 8, 1, tzinfo=timezone.utc), 110.0),
    ]
    eu_series = sorted(
        (o.observation_time, o.value) for o in result.observations if o.entity_id == "EU27_2020"
    )
    assert eu_series == [
        (datetime(2026, 7, 1, tzinfo=timezone.utc), 105.5),
        (datetime(2026, 8, 1, tzinfo=timezone.utc), 106.0),
    ]


def test_eurostat_industrial_production_duplicate_identity_raises_schema_changed():
    conn = re_.EurostatIndustrialProductionConnector({})
    dup = {
        "version": "2.0", "class": "dataset", "updated": "2026-08-30T09:00:00+02:00",
        "id": ["s_adj", "nace_r2", "indic_bt", "geo", "time"], "size": [1, 2, 1, 1, 1],
        "dimension": {
            "s_adj": {"category": {"index": {"SCA": 0}, "label": {"SCA": "Calendar and seasonally adjusted data"}}},
            # category.index als LISTE mit zwei identischen Codes -- beide
            # Indexpositionen bilden auf denselben Code "B-D" ab (wie im
            # road_freight-Regressionstest für dieselbe Kollisionsklasse).
            "nace_r2": {"category": {"index": ["B-D", "B-D"],
                                      "label": ["Industry (except construction)",
                                                "Industry (except construction)"]}},
            "indic_bt": {"category": {"index": {"PROD": 0}, "label": {"PROD": "Production (volume)"}}},
            "geo": {"category": {"index": {"DE": 0}, "label": {"DE": "Germany"}}},
            "time": {"category": {"index": {"2026-08": 0}, "label": {"2026-08": "2026-08"}}},
        },
        "value": {"0": 100.0, "1": 101.0},
    }
    import json as jsonmod
    fake = http.FetchResult(
        url="https://fixture/dup", status=200, content=jsonmod.dumps(dup).encode(),
        content_type="application/json", retrieved_at=utc_now(),
        content_hash="x", fingerprint="fp", bytes=10,
    )
    with patch.object(re_.http, "fetch", side_effect=[
        _fr("eurostat_industrial_production_toc.txt", content_type="text/plain"),
        fake,
    ]):
        result = conn.fetch(NOW)
    assert result.status == SourceStatus.SCHEMA_CHANGED
    assert result.observations == []
    assert "identity" in result.message.lower()


# ---------------------------------------------------------------------------
# fred_us_macro
# ---------------------------------------------------------------------------

def test_fred_us_macro_auth_missing_without_api_key_no_http_call(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    conn = re_.FredUsMacroConnector({})
    with patch.object(re_.http, "fetch") as mocked:
        result = conn.fetch(NOW)
    mocked.assert_not_called()
    assert result.status == SourceStatus.AUTH_MISSING


def test_fred_us_macro_fetches_both_series_with_api_key(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "test-key")
    conn = re_.FredUsMacroConnector({})
    with patch.object(re_.http, "fetch", side_effect=[
        _fr("fred_umcsent_observations.json"),
        _fr("fred_indpro_observations.json"),
    ]):
        result = conn.fetch(NOW)

    assert result.status == SourceStatus.PASS
    metrics = {o.metric for o in result.observations}
    assert metrics == {"us_umcsent", "us_indpro"}
    assert len(result.observations) == 4
    for o in result.observations:
        assert o.entity_id == "US"
        assert o.availability_precision == AvailabilityPrecision.EXACT_DATE


def test_fred_us_macro_schema_changed_without_observations_key(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "test-key")
    conn = re_.FredUsMacroConnector({})
    with patch.object(re_.http, "fetch", side_effect=[
        _fr("fred_schema_broken.json"),
    ]):
        result = conn.fetch(NOW)
    assert result.status == SourceStatus.SCHEMA_CHANGED

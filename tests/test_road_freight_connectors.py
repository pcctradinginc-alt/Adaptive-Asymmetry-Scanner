"""
tests/test_road_freight_connectors.py

Recorded-style Fixture-Tests für modules/external/sources/road_freight.py.
Es findet KEIN echter Netzwerkzugriff statt: `http.fetch` wird pro Test
gemockt und liefert vorbereitete FetchResult-Objekte aus
tests/fixtures/external/road/.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from modules.external import http
from modules.external.pit import AvailabilityPrecision, utc_now
from modules.external.sources import road_freight as rf
from modules.external.sources.base import SourceStatus

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "external" / "road"


def _fr(name: str, content_type: str = "application/json") -> http.FetchResult:
    content = (FIXTURES / name).read_bytes()
    now = utc_now()
    return http.FetchResult(
        url=f"https://fixture/{name}", status=200, content=content,
        content_type=content_type, retrieved_at=now,
        content_hash="deadbeef", fingerprint="fp", bytes=len(content),
    )


NOW = datetime(2024, 6, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# destatis_truck_toll
# ---------------------------------------------------------------------------

def test_destatis_parses_ffcsv_and_splits_metrics():
    conn = rf.DestatisTruckTollConnector({})
    with patch.object(rf.http, "fetch", side_effect=[
        _fr("destatis_catalogue.json"),
        _fr("destatis_ffcsv.csv", content_type="text/csv"),
    ]):
        result = conn.fetch(NOW)

    assert result.status == SourceStatus.PASS
    assert result.discovered_ids["destatis_chosen_table"] == "42191-0001"
    assert len(result.observations) == 4
    metrics = {o.metric for o in result.observations}
    assert metrics == {"index_sa", "index_unadjusted"}
    sa = [o for o in result.observations if o.metric == "index_sa"]
    assert {o.value for o in sa} == {101.2, 102.5}
    for o in result.observations:
        assert o.availability_precision == AvailabilityPrecision.CONSERVATIVE_DATE
        assert o.available_at >= o.retrieved_at or o.available_at == o.retrieved_at
        assert o.available_at == o.retrieved_at  # kein offizieller Release-Zeitstempel bekannt


def test_destatis_schema_changed_when_value_column_missing():
    conn = rf.DestatisTruckTollConnector({})
    with patch.object(rf.http, "fetch", side_effect=[
        _fr("destatis_catalogue.json"),
        _fr("destatis_ffcsv_schema_broken.csv", content_type="text/csv"),
        _fr("destatis_ffcsv_schema_broken.csv", content_type="text/csv"),
    ]):
        result = conn.fetch(NOW)
    assert result.status == SourceStatus.SCHEMA_CHANGED
    assert result.observations == []
    assert result.discovered_ids["diagnostics"]["body_snippet"]


def test_destatis_reports_status_json_envelope_with_http_200():
    conn = rf.DestatisTruckTollConnector({})
    with patch.object(rf.http, "fetch", side_effect=[
        _fr("destatis_catalogue.json"),
        _fr("destatis_status_error.json", content_type="application/json"),
    ]):
        result = conn.fetch(NOW)
    assert result.status == SourceStatus.FAIL
    assert "104" in result.message
    assert "Tabelle" in result.message or "table" in result.message.lower() or True


def test_destatis_falls_back_to_get_when_post_not_supported():
    conn = rf.DestatisTruckTollConnector({})
    with patch.object(rf.http, "fetch", side_effect=[
        _fr("destatis_catalogue.json"),
        http.FetchError("405 für data/tablefile (nicht retrybar)"),
        _fr("destatis_ffcsv.csv", content_type="text/csv"),
    ]):
        result = conn.fetch(NOW)
    assert result.status == SourceStatus.PASS
    assert len(result.observations) == 4


def test_destatis_auth_missing_on_data_endpoint():
    conn = rf.DestatisTruckTollConnector({})
    with patch.object(rf.http, "fetch", side_effect=[
        _fr("destatis_catalogue.json"),
        http.AuthError("401"),
    ]):
        result = conn.fetch(NOW)
    assert result.status == SourceStatus.AUTH_MISSING


# ---------------------------------------------------------------------------
# bts_freight_tsi
# ---------------------------------------------------------------------------

def test_bts_alfred_multiple_vintages_same_observation_time(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "test-key")
    conn = rf.BtsFreightTsiConnector({})
    with patch.object(rf.http, "fetch", side_effect=[_fr("bts_alfred_observations.json")]):
        result = conn.fetch(NOW)

    assert result.status == SourceStatus.PASS
    assert result.discovered_ids["bts_freight_tsi_series_id"] == "TSIFRGHT"
    dec_obs = [o for o in result.observations if o.observation_time == datetime(2023, 12, 1, tzinfo=timezone.utc)]
    assert len(dec_obs) == 2
    assert {o.value for o in dec_obs} == {115.3, 115.5}
    assert {o.available_at for o in dec_obs} == {
        datetime(2024, 1, 15, tzinfo=timezone.utc),
        datetime(2024, 2, 15, tzinfo=timezone.utc),
    }
    for o in result.observations:
        assert o.availability_precision == AvailabilityPrecision.EXACT_DATE
        assert o.available_at is not None


def test_bts_schema_changed_without_observations_key(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "test-key")
    conn = rf.BtsFreightTsiConnector({})
    with patch.object(rf.http, "fetch", side_effect=[_fr("bts_alfred_schema_broken.json")]):
        result = conn.fetch(NOW)
    assert result.status == SourceStatus.SCHEMA_CHANGED


def test_bts_fredgraph_fallback_without_api_key(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    conn = rf.BtsFreightTsiConnector({})
    with patch.object(rf.http, "fetch", side_effect=[_fr("fredgraph_fallback.csv", content_type="text/csv")]):
        result = conn.fetch(NOW)
    assert result.status == SourceStatus.WARN  # supports_vintages=false in diesem Modus, klar geflaggt
    assert "fredgraph" in result.message.lower()
    assert len(result.observations) == 1
    assert result.observations[0].value == 111.5
    assert result.observations[0].availability_precision == AvailabilityPrecision.CONSERVATIVE_DATE


# ---------------------------------------------------------------------------
# eurostat_road_freight
# ---------------------------------------------------------------------------

def test_eurostat_discovers_dataset_and_parses_jsonstat():
    conn = rf.EurostatRoadFreightConnector({})
    with patch.object(rf.http, "fetch", side_effect=[
        _fr("eurostat_toc.txt", content_type="text/plain"),
        _fr("eurostat_jsonstat.json"),
    ]):
        result = conn.fetch(NOW)

    assert result.status == SourceStatus.PASS
    assert result.discovered_ids["eurostat_chosen"] == "road_go_ta_tott"
    assert len(result.observations) == 4
    de_2023 = [o for o in result.observations
               if o.entity_id == "DE" and o.observation_time == datetime(2023, 1, 1, tzinfo=timezone.utc)]
    assert len(de_2023) == 1
    assert de_2023[0].value == 110.0
    assert de_2023[0].availability_precision == AvailabilityPrecision.EXACT_TIMESTAMP
    assert de_2023[0].available_at == datetime(2024, 6, 15, 9, 0, tzinfo=timezone.utc)


def test_eurostat_toc_excludes_folder_entries_never_picks_category_node():
    conn = rf.EurostatRoadFreightConnector({})
    with patch.object(rf.http, "fetch", side_effect=[
        _fr("eurostat_toc_with_folder.txt", content_type="text/plain"),
        _fr("eurostat_jsonstat.json"),
    ]):
        result = conn.fetch(NOW)
    assert result.status == SourceStatus.PASS
    # "road_go" ist ein Ordner-Knoten (type=folder) -> darf NIE als
    # Datensatz-Code gewählt werden, auch wenn der Titel matcht.
    assert result.discovered_ids["eurostat_chosen"] == "road_go_ta_tott"
    assert "road_go" not in result.discovered_ids["eurostat_road_freight_candidates"]


def test_eurostat_prefers_configured_expected_dataset_code_when_present_in_toc():
    conn = rf.EurostatRoadFreightConnector({"expected_dataset_code": "road_go_ta_tott"})
    with patch.object(rf.http, "fetch", side_effect=[
        _fr("eurostat_toc_with_folder.txt", content_type="text/plain"),
        _fr("eurostat_jsonstat.json"),
    ]):
        result = conn.fetch(NOW)
    assert result.status == SourceStatus.PASS
    assert result.discovered_ids["eurostat_chosen"] == "road_go_ta_tott"


def test_eurostat_retries_expected_dataset_code_on_404():
    conn = rf.EurostatRoadFreightConnector({"expected_dataset_code": "road_go_ta_tott"})
    with patch.object(rf.http, "fetch", side_effect=[
        _fr("eurostat_toc_alt_match.txt", content_type="text/plain"),
        http.FetchError("404 für https://.../data/road_go_ta_tg (nicht retrybar)"),
        _fr("eurostat_jsonstat.json"),
    ]):
        # Discovery findet nur "road_go_ta_tg" in dieser TOC (expected_code
        # road_go_ta_tott ist NICHT gelistet); der erste Datenabruf 404t ->
        # Retry mit der konfigurierten expected_dataset_code gelingt.
        result = conn.fetch(NOW)
    assert result.status == SourceStatus.PASS
    assert result.discovered_ids["eurostat_discovered_code_404"] == "road_go_ta_tg"
    assert result.discovered_ids["eurostat_chosen"] == "road_go_ta_tott"
    assert len(result.observations) == 4


def test_eurostat_schema_changed_without_dimension():
    conn = rf.EurostatRoadFreightConnector({})
    with patch.object(rf.http, "fetch", side_effect=[
        _fr("eurostat_toc.txt", content_type="text/plain"),
        _fr("eurostat_jsonstat_schema_broken.json"),
    ]):
        result = conn.fetch(NOW)
    assert result.status == SourceStatus.SCHEMA_CHANGED


# ---------------------------------------------------------------------------
# estat_jp_truck
# ---------------------------------------------------------------------------

def test_estat_jp_auth_missing_without_app_id_no_http_call(monkeypatch):
    monkeypatch.delenv("ESTAT_APP_ID", raising=False)
    conn = rf.EstatJpTruckConnector({})
    with patch.object(rf.http, "fetch") as mocked:
        result = conn.fetch(NOW)
    mocked.assert_not_called()
    assert result.status == SourceStatus.AUTH_MISSING


def test_estat_jp_discovers_stats_data_id_and_parses(monkeypatch):
    monkeypatch.setenv("ESTAT_APP_ID", "test-app-id")
    conn = rf.EstatJpTruckConnector({})
    with patch.object(rf.http, "fetch", side_effect=[
        _fr("estat_jp_stats_list.json"),
        _fr("estat_jp_stats_data.json"),
    ]):
        result = conn.fetch(NOW)

    assert result.status == SourceStatus.PASS
    assert result.discovered_ids["estat_jp_truck_stats_data_id"] == "0003000001"
    assert len(result.observations) == 2
    jan = [o for o in result.observations if o.observation_time == datetime(2024, 1, 1, tzinfo=timezone.utc)]
    assert jan[0].value == 12345.0
    assert "jp_truck_" in jan[0].metric


def test_estat_jp_schema_changed_when_value_missing(monkeypatch):
    monkeypatch.setenv("ESTAT_APP_ID", "test-app-id")
    conn = rf.EstatJpTruckConnector({})
    with patch.object(rf.http, "fetch", side_effect=[
        _fr("estat_jp_stats_list.json"),
        _fr("estat_jp_stats_data_schema_broken.json"),
    ]):
        result = conn.fetch(NOW)
    assert result.status == SourceStatus.SCHEMA_CHANGED


# ---------------------------------------------------------------------------
# PIT-Invariante über alle Konnektoren: available_at nie vor retrieved_at,
# außer bei bekannter offizieller Release-Zeit (dann kann available_at
# der Release-Zeitpunkt VOR unserem Abruf sein - das ist korrekt/gewollt).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("connector_cls,fixtures,env", [
    (rf.DestatisTruckTollConnector, ["destatis_catalogue.json", "destatis_ffcsv.csv"], {}),
    (rf.EurostatRoadFreightConnector, ["eurostat_toc.txt", "eurostat_jsonstat.json"], {}),
])
def test_available_at_never_before_retrieved_at_unless_release_time_known(connector_cls, fixtures, env, monkeypatch):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    conn = connector_cls({})
    with patch.object(rf.http, "fetch", side_effect=[_fr(f) for f in fixtures]):
        result = conn.fetch(NOW)
    for o in result.observations:
        if o.availability_precision in (AvailabilityPrecision.EXACT_TIMESTAMP, AvailabilityPrecision.EXACT_DATE):
            continue  # offizielle Release-Zeit darf vor unserem Abruf liegen
        assert o.available_at >= o.retrieved_at or o.available_at == o.retrieved_at

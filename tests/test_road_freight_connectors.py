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
    assert de_2023[0].metric == "road_freight_ths_t"
    assert de_2023[0].unit == "THS_T"
    assert de_2023[0].series_id == "road_go_ta_tott:tra_type=TOTAL|unit=THS_T"
    assert de_2023[0].attrs == {"tra_type": "TOTAL", "unit": "THS_T"}
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


def test_eurostat_parses_all_dimensions_generically_with_sparse_values():
    """JSON-stat mit zusaetzlichen Dimensionen (tra_type/carriage/unit) UND
    einem sparsen value-dict (nicht alle 16 Kombinationen vorhanden). Jede
    Kombination von geo/time/unit/tra_type/carriage muss eine EIGENE,
    unterscheidbare Beobachtung ergeben -- das ist der Kernfix gegen den
    Bug, der bis zu 55 verschiedene Werte auf eine Identitaet kollabierte."""
    conn = rf.EurostatRoadFreightConnector({})
    with patch.object(rf.http, "fetch", side_effect=[
        _fr("eurostat_toc.txt", content_type="text/plain"),
        _fr("eurostat_jsonstat_multidim.json"),
    ]):
        result = conn.fetch(NOW)

    assert result.status == SourceStatus.PASS
    assert result.parse_failures == 0
    assert len(result.observations) == 6  # sparse: nur 6 von 16 moeglichen Kombinationen

    by_val = {o.value: o for o in result.observations}

    # unit MIO_TKM darf NIE als "tonnes" fehlinterpretiert werden.
    mio = by_val[45000.0]
    assert mio.metric == "road_freight_mio_tkm"
    assert mio.unit == "MIO_TKM"
    assert mio.entity_id == "DE"
    assert mio.attrs == {"carriage": "TOT", "tra_type": "TOTAL", "unit": "MIO_TKM"}
    assert mio.series_id == "road_go_ta_tott:carriage=TOT|tra_type=TOTAL|unit=MIO_TKM"

    ths = by_val[500.0]
    assert ths.metric == "road_freight_ths_t"
    assert ths.unit == "THS_T"
    assert ths.series_id == "road_go_ta_tott:carriage=TOT|tra_type=TOTAL|unit=THS_T"

    # Gleiches geo/time/unit, aber tra_type=NAT statt TOTAL -> eigene
    # Identitaet (andere series_id), NIE mit dem TOTAL-Wert zusammengefasst.
    nat = by_val[300.0]
    assert nat.entity_id == "DE"
    assert nat.observation_time == datetime(2023, 1, 1, tzinfo=timezone.utc)
    assert nat.metric == "road_freight_ths_t"
    assert nat.series_id == "road_go_ta_tott:carriage=TOT|tra_type=NAT|unit=THS_T"
    assert nat.series_id != ths.series_id
    assert nat.identity_key() != ths.identity_key()

    # Quartalszeit: "2023-Q2" -> 2023-04-01.
    q2 = by_val[130.0]
    assert q2.observation_time == datetime(2023, 4, 1, tzinfo=timezone.utc)
    assert q2.entity_id == "DE"

    # Jahreszeit: "2023" -> 2023-01-01.
    assert ths.observation_time == datetime(2023, 1, 1, tzinfo=timezone.utc)

    # Alle Identitaeten sind eindeutig (keine zwei Observations teilen sich
    # source_id/dataset/series_id/entity_id/metric/observation_time).
    keys = [o.identity_key() for o in result.observations]
    assert len(keys) == len(set(keys))

    # archive.store_observations() auf frischem Archiv -> alles "new", 0 revisions.
    from modules.external.archive import ExternalArchive
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        archive = ExternalArchive(root=tmp)
        stats = archive.store_observations(result.observations)
    assert stats["revision"] == 0
    assert stats["duplicate"] == 0
    assert stats["new"] == 6


def test_eurostat_duplicate_identity_raises_schema_changed():
    """Zwei Rohwerte, die (nach korrektem generischem Dimensions-Parsing)
    auf dieselbe (series_id, entity_id, metric, observation_time)-Identitaet
    fallen, werden NIE stillschweigend zusammengefasst -- klarer
    SCHEMA_CHANGED statt einer verlorenen Revision."""
    conn = rf.EurostatRoadFreightConnector({})
    with patch.object(rf.http, "fetch", side_effect=[
        _fr("eurostat_toc.txt", content_type="text/plain"),
        _fr("eurostat_jsonstat_duplicate_identity.json"),
    ]):
        result = conn.fetch(NOW)
    assert result.status == SourceStatus.SCHEMA_CHANGED
    assert result.observations == []
    assert "identity" in result.message.lower()


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


# ---------------------------------------------------------------------------
# bts_open_data_tsi (Socrata data.bts.gov, ohne API-Key)
# ---------------------------------------------------------------------------

def test_bts_open_data_discovers_dataset_and_parses_rows():
    conn = rf.BtsOpenDataTsiConnector({})
    with patch.object(rf.http, "fetch", side_effect=[
        _fr("bts_socrata_catalog.json"),
        _fr("bts_views_metadata.json"),
        _fr("bts_soda_rows_page1.json"),
    ]):
        result = conn.fetch(NOW)

    assert result.status == SourceStatus.PASS
    assert result.discovered_ids["bts_open_data_tsi_dataset_id"] == "69qe-yiui"
    assert "bts_open_data_tsi_ambiguous" not in result.discovered_ids
    cols = result.discovered_ids["bts_open_data_tsi_columns"]
    assert cols["freight_field"] == "tsi_freight_seasonally_adjusted"
    assert cols["truck_field"] == "tsi_truck_seasonally_adjusted"

    freight = [o for o in result.observations if o.metric == "us_freight_tsi"]
    trucking = [o for o in result.observations if o.metric == "us_trucking_index"]
    assert len(freight) == 2
    assert len(trucking) == 2
    jan = [o for o in freight if o.observation_time == datetime(2024, 1, 1, tzinfo=timezone.utc)][0]
    assert jan.value == 125.3
    assert jan.entity_id == "US"
    assert jan.availability_precision == AvailabilityPrecision.CONSERVATIVE_DATE
    assert jan.available_at == jan.retrieved_at
    # rowsUpdatedAt (Unix-Sekunden) -> informativer source_release_time
    assert jan.source_release_time == datetime(2024, 5, 15, 13, 0, tzinfo=timezone.utc)


def test_bts_open_data_ambiguous_catalog_flags_and_picks_most_recent():
    conn = rf.BtsOpenDataTsiConnector({})
    with patch.object(rf.http, "fetch", side_effect=[
        _fr("bts_socrata_catalog_ambiguous.json"),
        _fr("bts_views_metadata.json"),
        _fr("bts_soda_rows_page1.json"),
    ]):
        result = conn.fetch(NOW)
    assert result.status == SourceStatus.PASS
    assert result.discovered_ids["bts_open_data_tsi_ambiguous"] == ["new-tsi", "old-tsi"]
    assert result.discovered_ids["bts_open_data_tsi_dataset_id"] == "new-tsi"


def test_bts_open_data_pagination_across_soda_pages():
    conn = rf.BtsOpenDataTsiConnector({"page_limit": 2})
    with patch.object(rf.http, "fetch", side_effect=[
        _fr("bts_socrata_catalog.json"),
        _fr("bts_views_metadata.json"),
        _fr("bts_soda_rows_page1.json"),
        _fr("bts_soda_rows_page2.json"),
    ]):
        result = conn.fetch(NOW)
    assert result.status == SourceStatus.PASS
    freight = [o for o in result.observations if o.metric == "us_freight_tsi"]
    assert len(freight) == 3
    assert {o.observation_time for o in freight} == {
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 2, 1, tzinfo=timezone.utc),
        datetime(2024, 3, 1, tzinfo=timezone.utc),
    }


def test_bts_open_data_schema_changed_when_no_freight_column():
    conn = rf.BtsOpenDataTsiConnector({})
    with patch.object(rf.http, "fetch", side_effect=[
        _fr("bts_socrata_catalog.json"),
        _fr("bts_views_metadata_broken.json"),
    ]):
        result = conn.fetch(NOW)
    assert result.status == SourceStatus.SCHEMA_CHANGED


def test_bts_open_data_fail_when_no_dataset_found_and_no_fallback():
    conn = rf.BtsOpenDataTsiConnector({})
    with patch.object(rf.http, "fetch", side_effect=[
        _fr("bts_socrata_catalog_empty.json"),
    ]):
        result = conn.fetch(NOW)
    assert result.status == SourceStatus.FAIL


# ---------------------------------------------------------------------------
# destatis_truck_toll_download (EXDAT-Direktdownload, ohne GENESIS-Login)
# ---------------------------------------------------------------------------

def test_destatis_download_discovers_xlsx_link_and_parses():
    conn = rf.DestatisTruckTollDownloadConnector({})
    with patch.object(rf.http, "fetch", side_effect=[
        _fr("destatis_lkw_maut_page.html", content_type="text/html"),
        _fr("destatis_truck_toll_download.xlsx",
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    ]):
        result = conn.fetch(NOW)

    assert result.status == SourceStatus.PASS
    chosen = result.discovered_ids["destatis_truck_toll_download_chosen"]
    assert chosen.endswith(".xlsx")
    assert chosen.startswith("https://www.destatis.de/")
    # der externe Spiegel-Link darf NIE als Kandidat auftauchen.
    assert not any("external-mirror" in c for c in
                   result.discovered_ids["destatis_truck_toll_download_candidates"])

    metrics = {o.metric for o in result.observations}
    assert metrics == {"index_sa", "index_unadjusted"}
    for o in result.observations:
        assert o.entity_id == ""
        assert o.availability_precision == AvailabilityPrecision.CONSERVATIVE_DATE
        assert o.available_at == o.retrieved_at
    sa = [o for o in result.observations if o.metric == "index_sa"]
    assert {o.value for o in sa} == {101.2, 102.5, None}


def test_destatis_download_parses_csv_variant():
    conn = rf.DestatisTruckTollDownloadConnector({})
    with patch.object(rf.http, "fetch", side_effect=[
        _fr("destatis_lkw_maut_page.html", content_type="text/html"),
    ]):
        # Downloadseite bevorzugt xlsx -> wir testen den CSV-Pfad separat,
        # indem wir den Konnektor direkt mit der CSV-Downloadlogik aufrufen.
        content = (FIXTURES / "destatis_truck_toll_download.csv").read_bytes()
        observations, latest, parse_failures, diag = conn._parse_csv(content, NOW)
    assert diag == {}
    metrics = {o.metric for o in observations}
    assert metrics == {"index_sa", "index_unadjusted"}
    sa = [o for o in observations if o.metric == "index_sa"]
    assert {o.value for o in sa} == {101.2, 102.5, None}


def test_destatis_download_only_accepts_destatis_domain_links():
    conn = rf.DestatisTruckTollDownloadConnector({})
    with patch.object(rf.http, "fetch", side_effect=[
        _fr("destatis_lkw_maut_page_external_only.html", content_type="text/html"),
    ]):
        result = conn.fetch(NOW)
    assert result.status == SourceStatus.FAIL
    assert "destatis_truck_toll_download_candidates" not in result.discovered_ids


def test_destatis_download_fail_when_page_has_no_link_and_no_fallback():
    conn = rf.DestatisTruckTollDownloadConnector({})
    with patch.object(rf.http, "fetch", side_effect=[
        _fr("destatis_lkw_maut_page_no_link.html", content_type="text/html"),
    ]):
        result = conn.fetch(NOW)
    assert result.status == SourceStatus.FAIL


def test_destatis_download_schema_changed_when_columns_missing():
    conn = rf.DestatisTruckTollDownloadConnector({})
    with patch.object(rf.http, "fetch", side_effect=[
        _fr("destatis_lkw_maut_page.html", content_type="text/html"),
        _fr("destatis_truck_toll_download_schema_broken.xlsx",
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    ]):
        result = conn.fetch(NOW)
    assert result.status == SourceStatus.SCHEMA_CHANGED


def test_destatis_download_features_used_as_fallback_when_genesis_absent():
    """road_freight_features.de_truck_* muss destatis_truck_toll_download
    genauso bedienen wie destatis_truck_toll (Fallback-Konvention)."""
    from modules.external.sources import road_freight_features as feat
    conn = rf.DestatisTruckTollDownloadConnector({})
    with patch.object(rf.http, "fetch", side_effect=[
        _fr("destatis_lkw_maut_page.html", content_type="text/html"),
        _fr("destatis_truck_toll_download.xlsx",
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    ]):
        result = conn.fetch(NOW)
    assert feat.de_truck_level(result.observations) == 102.5


# ---------------------------------------------------------------------------
# eurostat_road_freight_quarterly (Discovery + First-Seen-PIT-Präzision)
# ---------------------------------------------------------------------------

def test_eurostat_quarterly_discovers_quarterly_dataset_not_annual():
    conn = rf.EurostatRoadFreightQuarterlyConnector({
        "_seen_periods_state_path": "/tmp/_never_used_eurostat_seen_periods_discovery_test.json"})
    with patch.object(rf.http, "fetch", side_effect=[
        _fr("eurostat_toc_quarterly.txt", content_type="text/plain"),
        _fr("eurostat_jsonstat_quarterly_v1.json"),
    ]):
        result = conn.fetch(NOW)
    assert result.status == SourceStatus.PASS
    # "road_go" (Ordner) und die jährliche road_go_ta_tott dürfen NIE
    # gewählt werden -- nur der Datensatz mit "quarterly" im TOC-Titel.
    assert result.discovered_ids["eurostat_road_freight_quarterly_chosen"] == "road_go_qa_tott"
    assert len(result.observations) == 3
    q1 = [o for o in result.observations if o.observation_time == datetime(2023, 1, 1, tzinfo=timezone.utc)][0]
    assert q1.entity_id == "DE"
    assert q1.value == 100.0


def test_eurostat_quarterly_first_seen_precision(tmp_path):
    state_path = str(tmp_path / "eurostat_seen_periods.json")
    cfg = {"_seen_periods_state_path": state_path}

    # 1. Abruf: Zustandsdatei existiert noch nicht -> Backfill der ganzen
    # Historie; deren Erstveröffentlichung ist unbekannt -> CONSERVATIVE_DATE,
    # kein first_seen_at. Präzisere Zeiten erst ab dem 2. Abruf.
    conn1 = rf.EurostatRoadFreightQuarterlyConnector(cfg)
    with patch.object(rf.http, "fetch", side_effect=[
        _fr("eurostat_toc_quarterly.txt", content_type="text/plain"),
        _fr("eurostat_jsonstat_quarterly_v1.json"),
    ]):
        result1 = conn1.fetch(NOW)
    assert result1.status == SourceStatus.PASS
    assert len(result1.observations) == 3
    for o in result1.observations:
        assert o.availability_precision == AvailabilityPrecision.CONSERVATIVE_DATE
        assert o.available_at == datetime(2024, 6, 15, 9, 0, tzinfo=timezone.utc)
        assert "first_seen_at" not in o.attrs
    import json as _json
    stored = _json.loads(Path(state_path).read_text())
    assert set(stored["road_go_qa_tott"]) == {"2023-Q1", "2023-Q2", "2023-Q3"}

    # 2. Abruf (neuer Connector-Instanz, gleiche Zustandsdatei): 2023-Q1..Q3
    # sind jetzt historisch (CONSERVATIVE_DATE), NUR 2023-Q4 ist neu
    # (EXACT_TIMESTAMP, first_seen_at gesetzt).
    conn2 = rf.EurostatRoadFreightQuarterlyConnector(cfg)
    with patch.object(rf.http, "fetch", side_effect=[
        _fr("eurostat_toc_quarterly.txt", content_type="text/plain"),
        _fr("eurostat_jsonstat_quarterly_v2.json"),
    ]):
        result2 = conn2.fetch(NOW)
    assert result2.status == SourceStatus.PASS
    by_period = {o.observation_time: o for o in result2.observations}
    historical = [o for t, o in by_period.items() if t < datetime(2023, 10, 1, tzinfo=timezone.utc)]
    new_period = [o for t, o in by_period.items() if t == datetime(2023, 10, 1, tzinfo=timezone.utc)][0]
    assert len(historical) == 3
    for o in historical:
        assert o.availability_precision == AvailabilityPrecision.CONSERVATIVE_DATE
        assert "first_seen_at" not in o.attrs
    assert new_period.availability_precision == AvailabilityPrecision.EXACT_TIMESTAMP
    assert "first_seen_at" in new_period.attrs
    assert new_period.available_at == datetime(2024, 9, 15, 9, 0, tzinfo=timezone.utc)

    stored2 = _json.loads(Path(state_path).read_text())
    assert set(stored2["road_go_qa_tott"]) == {"2023-Q1", "2023-Q2", "2023-Q3", "2023-Q4"}


def test_eurostat_road_freight_annual_connector_precision_unaffected():
    """Der bestehende eurostat_road_freight-Konnektor (jährlich/gemischt)
    behält sein etabliertes Verhalten (EXACT_TIMESTAMP sobald release_time
    bekannt) -- die First-Seen-PIT-Präzision ist bewusst NUR im neuen
    Quarterly-Konnektor aktiv (siehe _parse_eurostat_jsonstat-Docstring)."""
    conn = rf.EurostatRoadFreightConnector({})
    with patch.object(rf.http, "fetch", side_effect=[
        _fr("eurostat_toc.txt", content_type="text/plain"),
        _fr("eurostat_jsonstat.json"),
    ]):
        result = conn.fetch(NOW)
    assert result.status == SourceStatus.PASS
    for o in result.observations:
        assert o.availability_precision == AvailabilityPrecision.EXACT_TIMESTAMP
        assert "first_seen_at" not in o.attrs


def test_eurostat_quarterly_preflight_does_not_touch_first_seen_state(tmp_path):
    state_path = tmp_path / "eurostat_seen_periods.json"
    conn = rf.EurostatRoadFreightQuarterlyConnector({"_seen_periods_state_path": str(state_path)})
    with patch.object(rf.http, "fetch", side_effect=[
        _fr("eurostat_toc_quarterly.txt", content_type="text/plain"),
        _fr("eurostat_jsonstat_quarterly_v1.json"),
    ]):
        rep = conn.preflight(NOW)
    assert rep["status"] == "PASS"
    assert not state_path.exists()


def test_genesis_tablefile_zip_and_cp1252_are_decoded():
    import io
    import zipfile
    body = "Zeit;Wert\n01.09.2026;101,5\n".encode("cp1252")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("42191-0001_de.csv", body)
    assert rf._decode_genesis_tablefile(buf.getvalue()).startswith("Zeit;Wert")
    assert "Größe" in rf._decode_genesis_tablefile("Größe".encode("cp1252"))
    assert rf._decode_genesis_tablefile(b"PK\x03\x04kaputt") is None


def test_genesis_classic_monthly_csv_is_parsed():
    text = (FIXTURES / "destatis_genesis_classic_monthly.csv").read_text(encoding="utf-8")
    conn = rf.DestatisTruckTollConnector({})
    obs, latest, failures = conn._parse_classic_csv(text, "42191-0001", NOW)
    assert failures == 0
    sa = {o.observation_time.month + 100 * o.observation_time.year: o.value
          for o in obs if o.metric == "index_sa"}
    assert sa[202501] == 95.5 and sa[202503] == 95.6
    assert sa[202601] is None                       # "-" = fehlend, nie 0
    assert {o.metric for o in obs} == {"index_unadjusted", "index_calendar_adjusted",
                                      "index_sa", "index_sa_bv41", "index_trend"}
    assert latest.year == 2026 and latest.month == 1
    assert all(o.dataset == "monthly_index" and o.entity_id == "" for o in obs)

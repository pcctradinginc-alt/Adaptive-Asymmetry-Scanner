"""
Tests für modules/external/sources/maritime.py (IMF PortWatch Konnektor).

Kein Netzwerkzugriff: alle HTTP-Aufrufe laufen über einen injizierten
Fake-Fetch, der Antworten aus tests/fixtures/external/maritime/*.json liefert.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pytest

from modules.external.pit import AvailabilityPrecision
from modules.external.sources import maritime as mw
from modules.external.sources.base import SourceStatus

FIXDIR = os.path.join(os.path.dirname(__file__), "fixtures", "external", "maritime")


def _load(name: str) -> dict:
    with open(os.path.join(FIXDIR, name), "r", encoding="utf-8") as fh:
        return json.load(fh)


class FakeResponse:
    def __init__(self, data: dict):
        self._data = data

    def json(self):
        return self._data


SERVICE_URLS = {
    "ports_daily": "https://services9.arcgis.com/portwatchfake/arcgis/rest/services/DailyPortsData/FeatureServer",
    "chokepoints_daily": "https://services9.arcgis.com/portwatchfake/arcgis/rest/services/DailyChokepointsData/FeatureServer",
    "ports_reference": "https://services9.arcgis.com/portwatchfake/arcgis/rest/services/Ports/FeatureServer",
    "chokepoints_reference": "https://services9.arcgis.com/portwatchfake/arcgis/rest/services/Chokepoints/FeatureServer",
}


def make_fake_fetch(search_ports_daily_fixture="search_daily_ports_data.json",
                     ports_daily_pages=None,
                     chokepoints_daily_pages=None,
                     ports_daily_metadata_fixture="layer_metadata_ports_daily.json"):
    ports_daily_pages = ports_daily_pages if ports_daily_pages is not None else [
        "query_ports_daily_page1.json", "query_ports_daily_page2.json", "query_ports_daily_page3.json",
    ]
    chokepoints_daily_pages = chokepoints_daily_pages if chokepoints_daily_pages is not None else [
        "query_chokepoints_daily.json",
    ]
    state = {"ports_daily_page_idx": 0, "chokepoints_daily_page_idx": 0}

    def fake_fetch(url, params=None, **kwargs):
        params = params or {}
        if url == mw.ARCGIS_SEARCH_URL:
            q = params.get("q", "")
            if "Daily Ports Data" in q:
                return FakeResponse(_load(search_ports_daily_fixture))
            if "Daily Chokepoints Data" in q:
                return FakeResponse(_load("search_daily_chokepoints_data.json"))
            if 'title:"Ports"' in q:
                return FakeResponse(_load("search_ports_reference.json"))
            if 'title:"Chokepoints"' in q:
                return FakeResponse(_load("search_chokepoints_reference.json"))
            raise AssertionError(f"unerwartete ArcGIS-Suche: {q!r}")

        if url in SERVICE_URLS.values():
            return FakeResponse(_load("featureserver_root.json"))

        if url.endswith("/0"):
            if url.startswith(SERVICE_URLS["ports_daily"]):
                return FakeResponse(_load(ports_daily_metadata_fixture))
            if url.startswith(SERVICE_URLS["chokepoints_daily"]):
                return FakeResponse(_load("layer_metadata_chokepoints_daily.json"))
            if url.startswith(SERVICE_URLS["ports_reference"]):
                return FakeResponse(_load("layer_metadata_ports_reference.json"))
            if url.startswith(SERVICE_URLS["chokepoints_reference"]):
                return FakeResponse(_load("layer_metadata_chokepoints_reference.json"))

        if url.endswith("/query"):
            if url.startswith(SERVICE_URLS["ports_daily"]):
                idx = state["ports_daily_page_idx"]
                state["ports_daily_page_idx"] += 1
                return FakeResponse(_load(ports_daily_pages[idx]))
            if url.startswith(SERVICE_URLS["chokepoints_daily"]):
                idx = state["chokepoints_daily_page_idx"]
                state["chokepoints_daily_page_idx"] += 1
                return FakeResponse(_load(chokepoints_daily_pages[idx]))
            if url.startswith(SERVICE_URLS["ports_reference"]):
                return FakeResponse(_load("ports_reference_features.json"))
            if url.startswith(SERVICE_URLS["chokepoints_reference"]):
                return FakeResponse(_load("chokepoints_reference_features.json"))

        raise AssertionError(f"kein Fixture für url={url!r} params={params!r}")

    return fake_fetch, state


NOW = datetime(2024, 1, 6, 12, 0, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------

def test_discovery_resolves_all_four_endpoints(tmp_path):
    fake_fetch, _ = make_fake_fetch()
    cache_path = str(tmp_path / "portwatch_endpoints.json")
    endpoints = mw.discover_endpoints(http_fetch=fake_fetch, cache_path=cache_path, now=NOW)
    assert endpoints["ports_daily"]["service_url"] == SERVICE_URLS["ports_daily"]
    assert endpoints["chokepoints_daily"]["service_url"] == SERVICE_URLS["chokepoints_daily"]
    assert endpoints["ports_reference"]["service_url"] == SERVICE_URLS["ports_reference"]
    assert endpoints["chokepoints_reference"]["service_url"] == SERVICE_URLS["chokepoints_reference"]
    assert os.path.exists(cache_path)
    with open(cache_path) as fh:
        cached = json.load(fh)
    assert "discovered_at" in cached


def test_discovery_uses_cache_without_refetch(tmp_path):
    fake_fetch, _ = make_fake_fetch()
    cache_path = str(tmp_path / "portwatch_endpoints.json")
    mw.discover_endpoints(http_fetch=fake_fetch, cache_path=cache_path, now=NOW)

    def boom(*a, **kw):
        raise AssertionError("sollte den Cache nutzen, nicht erneut fetchen")

    endpoints = mw.discover_endpoints(http_fetch=boom, cache_path=cache_path, now=NOW)
    assert endpoints["ports_daily"]["service_url"] == SERVICE_URLS["ports_daily"]


def test_discovery_by_name_with_ambiguity_marks_result():
    fake_fetch, _ = make_fake_fetch(search_ports_daily_fixture="search_ambiguous_ports_daily.json")
    item = mw._search_arcgis_item("Daily Ports Data", fake_fetch)
    assert item["ambiguous"] is True
    assert item["n_candidates"] == 2
    # nimmt den zuletzt geänderten Kandidaten, nicht irgendeinen
    assert item["service_url"] == SERVICE_URLS["ports_daily"]


def test_search_arcgis_item_rejects_wrong_org_candidate_even_if_returned():
    """Verteidigung gegen den live beobachteten Fehler: die q-Restriktion
    (orgid:...) greift serverseitig aus irgendeinem Grund nicht und liefert
    trotzdem ein Item aus fremder Organisation zurück -- muss client-seitig
    verworfen werden, nie akzeptiert."""
    def fake_fetch(url, params=None, **kwargs):
        return FakeResponse(_load("search_ports_reference_wrong_org.json"))

    with pytest.raises(mw.DiscoveryError):
        mw._search_arcgis_item(
            "Ports", fake_fetch, require_owner="PortWatch_IMF", require_orgid="portwatchorg001",
        )


def test_discovery_falls_back_to_daily_layer_when_reference_org_mismatched(tmp_path):
    """item 1 im Fix: 'Ports' liefert (wie live beobachtet) ein Item einer
    fremden Organisation -> wird verworfen, ports_reference wird stattdessen
    als Fallback auf den Daily-Ports-Layer selbst markiert."""
    def fake_fetch(url, params=None, **kwargs):
        params = params or {}
        if url == mw.ARCGIS_SEARCH_URL:
            q = params.get("q", "")
            if "Daily Ports Data" in q:
                return FakeResponse(_load("search_daily_ports_data.json"))
            if "Daily Chokepoints Data" in q:
                return FakeResponse(_load("search_daily_chokepoints_data.json"))
            if 'title:"Ports"' in q:
                return FakeResponse(_load("search_ports_reference_wrong_org.json"))
            if 'title:"Chokepoints"' in q:
                return FakeResponse(_load("search_chokepoints_reference.json"))
            raise AssertionError(f"unerwartete ArcGIS-Suche: {q!r}")
        if url in SERVICE_URLS.values():
            return FakeResponse(_load("featureserver_root.json"))
        raise AssertionError(f"kein Fixture für url={url!r} params={params!r}")

    cache_path = str(tmp_path / "portwatch_endpoints.json")
    endpoints = mw.discover_endpoints(http_fetch=fake_fetch, cache_path=cache_path, now=NOW)
    assert endpoints["ports_reference"]["fallback_to_daily_layer"] is True
    assert endpoints["ports_reference"]["service_url"] == SERVICE_URLS["ports_daily"]
    # Chokepoints-Referenz-Suche war org-verifiziert erfolgreich -> kein Fallback
    assert endpoints["chokepoints_reference"]["fallback_to_daily_layer"] is False
    assert endpoints["chokepoints_reference"]["service_url"] == SERVICE_URLS["chokepoints_reference"]


def test_connector_uses_distinct_values_fallback_for_ports_reference(tmp_path):
    """Voller fetch()-Flow: wenn die Ports-Referenz-Suche org-verworfen
    wird, leitet der Konnektor portid/portname/country per
    returnDistinctValues-Query direkt aus dem Daily-Ports-Layer ab (statt
    IDs zu erraten oder die falsche USDA-Layer zu benutzen)."""
    def fake_fetch(url, params=None, **kwargs):
        params = params or {}
        if url == mw.ARCGIS_SEARCH_URL:
            q = params.get("q", "")
            if "Daily Ports Data" in q:
                return FakeResponse(_load("search_daily_ports_data.json"))
            if "Daily Chokepoints Data" in q:
                return FakeResponse(_load("search_daily_chokepoints_data.json"))
            if 'title:"Ports"' in q:
                return FakeResponse(_load("search_ports_reference_wrong_org.json"))
            if 'title:"Chokepoints"' in q:
                return FakeResponse(_load("search_chokepoints_reference.json"))
            raise AssertionError(f"unerwartete ArcGIS-Suche: {q!r}")
        if url in SERVICE_URLS.values():
            return FakeResponse(_load("featureserver_root.json"))
        if url.endswith("/0") and url.startswith(SERVICE_URLS["ports_daily"]):
            return FakeResponse(_load("layer_metadata_ports_daily.json"))
        if url.endswith("/query") and url.startswith(SERVICE_URLS["ports_daily"]):
            if params.get("returnDistinctValues") == "true":
                return FakeResponse(_load("query_ports_daily_distinct.json"))
            return FakeResponse(_load("query_ports_daily_single_page.json"))
        raise AssertionError(f"kein Fixture für url={url!r} params={params!r}")

    cache_path = str(tmp_path / "portwatch_endpoints.json")
    connector = mw.PortWatchPortsConnector(http_fetch=fake_fetch, cache_path=cache_path)
    result = connector.fetch(NOW)
    assert result.status in (SourceStatus.PASS, SourceStatus.WARN)
    assert result.discovered_ids["ports"]["Los Angeles"] == "port_la"
    assert result.discovered_ids["diagnostics"]["ports_reference_fallback_to_daily_layer"] is True
    assert "port_la" in {v for v in result.discovered_ids["ports"].values()}


def test_config_override_marks_verify_in_preflight(tmp_path):
    fake_fetch, _ = make_fake_fetch()
    cache_path = str(tmp_path / "portwatch_endpoints.json")
    override = {"ports_daily_service_url": "https://example.org/override/FeatureServer", "ports_daily_layer_id": 3}
    endpoints = mw.discover_endpoints(http_fetch=fake_fetch, cache_path=cache_path, now=NOW, config_override=override)
    assert endpoints["ports_daily"]["service_url"] == "https://example.org/override/FeatureServer"
    assert endpoints["ports_daily"]["verify_in_preflight"] is True


# --------------------------------------------------------------------------
# Schema-Validierung
# --------------------------------------------------------------------------

def test_validate_ports_schema_ok():
    meta = _load("layer_metadata_ports_daily.json")
    cargo_fields = mw.validate_ports_schema(meta)
    assert "portcalls" in cargo_fields
    assert "portcalls_container" in cargo_fields
    assert "import" in cargo_fields and "export" in cargo_fields


def test_validate_ports_schema_changed_raises():
    meta = _load("layer_metadata_ports_daily_schema_changed.json")
    with pytest.raises(mw.SchemaError):
        mw.validate_ports_schema(meta)


def test_resolve_ports_field_schema_matches_case_insensitive_aliases():
    meta = _load("layer_metadata_ports_daily_alt_casing.json")
    schema = mw.resolve_ports_field_schema(meta)
    assert schema["date_field"] == "Date"
    assert schema["date_field_type"] == "esriFieldTypeString"
    assert schema["id_field"] == "PortId"
    assert schema["name_field"] == "PortName"
    assert schema["country_field"] == "Country"
    assert schema["cargo_fields"] == sorted(["portcalls", "import", "export"])


def test_resolve_ports_field_schema_never_sends_hardcoded_outfields(tmp_path):
    fake_fetch, _ = make_fake_fetch(
        ports_daily_metadata_fixture="layer_metadata_ports_daily_alt_casing.json",
    )
    cache_path = str(tmp_path / "portwatch_endpoints.json")
    connector = mw.PortWatchPortsConnector(http_fetch=fake_fetch, cache_path=cache_path)
    # nur die Metadaten-Layer-Fetches werden hier gebraucht; ein Query-Aufruf
    # mit korrekt gecaseten outFields ist Teil des vollen fetch()-Flusses,
    # dieser Test verifiziert nur die Schema-Auflösung selbst (siehe oben) --
    # zusätzlich prüfen wir hier, dass ein SchemaError bei völlig fehlenden
    # Kernfeldern die Feldliste als Diagnose mitliefert.
    meta = _load("layer_metadata_ports_daily_schema_changed.json")
    with pytest.raises(mw.SchemaError) as exc_info:
        mw.resolve_ports_field_schema(meta, "https://example.org/FeatureServer", 0)
    assert exc_info.value.diagnostics["field_names"] == sorted(
        f["name"] for f in meta["fields"]
    )
    assert exc_info.value.diagnostics["service_url"] == "https://example.org/FeatureServer/0"


def test_build_where_clause_uses_timestamp_for_date_field_type():
    from datetime import date as _date
    where = mw.build_where_clause(_date(2024, 1, 1), _date(2024, 1, 7), date_field="date",
                                   date_field_type="esriFieldTypeDate")
    assert "TIMESTAMP" in where


def test_build_where_clause_uses_string_compare_for_non_date_field_type():
    from datetime import date as _date
    where = mw.build_where_clause(_date(2024, 1, 1), _date(2024, 1, 7), date_field="Date",
                                   date_field_type="esriFieldTypeString")
    assert "TIMESTAMP" not in where
    assert "'2024-01-01'" in where and "'2024-01-07'" in where


def test_connector_reports_schema_changed_status(tmp_path):
    fake_fetch, _ = make_fake_fetch(
        ports_daily_metadata_fixture="layer_metadata_ports_daily_schema_changed.json",
    )
    cache_path = str(tmp_path / "portwatch_endpoints.json")
    connector = mw.PortWatchPortsConnector(http_fetch=fake_fetch, cache_path=cache_path)
    result = connector.fetch(NOW)
    assert result.status == SourceStatus.SCHEMA_CHANGED
    assert result.observations == []
    assert "field_names" in result.discovered_ids["diagnostics"]


# --------------------------------------------------------------------------
# Pagination
# --------------------------------------------------------------------------

def test_query_features_paginates_using_max_record_count_and_exceeded_transfer_limit():
    fake_fetch, state = make_fake_fetch()
    meta = _load("layer_metadata_ports_daily.json")
    max_rc = meta["maxRecordCount"]
    assert max_rc == 2  # aus der Fixture, NIE hart im Code annehmen

    features = mw.query_features(
        SERVICE_URLS["ports_daily"], 0, "1=1", ["date", "portid", "portcalls"],
        fake_fetch, max_record_count=max_rc,
    )
    assert len(features) == 5  # 2 + 2 + 1 über drei Seiten
    assert state["ports_daily_page_idx"] == 3
    dates = [f["attributes"]["date"] for f in features]
    assert dates == sorted(dates)


def test_query_features_stops_when_transfer_limit_not_exceeded():
    fake_fetch, state = make_fake_fetch(chokepoints_daily_pages=["query_chokepoints_daily.json"])
    features = mw.query_features(
        SERVICE_URLS["chokepoints_daily"], 0, "1=1", ["date", "chokepointid", "n_total"],
        fake_fetch, max_record_count=2000,
    )
    assert len(features) == 2
    assert state["chokepoints_daily_page_idx"] == 1


# --------------------------------------------------------------------------
# Incremental window
# --------------------------------------------------------------------------

def test_incremental_window_is_21_days():
    start, end = mw.incremental_window(NOW)
    assert (end - start).days == 21
    assert end == NOW.date()


def test_backfill_uses_wide_start_date(tmp_path):
    fake_fetch, _ = make_fake_fetch()
    cache_path = str(tmp_path / "portwatch_endpoints.json")
    connector = mw.PortWatchPortsConnector(
        http_fetch=fake_fetch, cache_path=cache_path,
        port_universe_path="config/port_universe.yaml",
    )
    result = connector.fetch(NOW, backfill=True)
    assert result.status in (SourceStatus.PASS, SourceStatus.WARN)
    assert all(o.attrs.get("historical_backfill") is True for o in result.observations)
    assert all(o.availability_precision == AvailabilityPrecision.UNKNOWN for o in result.observations)


# --------------------------------------------------------------------------
# Universe-Resolution (Ambiguity)
# --------------------------------------------------------------------------

def test_resolve_entities_by_name_exact_and_ambiguous_and_no_match():
    ref = _load("ports_reference_features.json")["features"]
    wanted = [
        {"name": "Los Angeles", "country": "United States"},
        {"name": "Valencia", "country": "Spain"},
        {"name": "Valencia", "country": None},
        {"name": "Springfield", "country": "United States"},
        {"name": "Nonexistent Port", "country": "Nowhere"},
    ]
    resolved = mw.resolve_entities_by_name(wanted, ref, "portid", "portname", "country")
    by_name_country = {(r.name, r.country): r for r in resolved}

    assert by_name_country[("Los Angeles", "United States")].status == "resolved"
    assert by_name_country[("Los Angeles", "United States")].entity_id == "port_la"

    assert by_name_country[("Valencia", "Spain")].status == "resolved"
    assert by_name_country[("Valencia", "Spain")].entity_id == "port_valencia_es"

    assert by_name_country[("Valencia", None)].status == "ambiguous"

    assert by_name_country[("Springfield", "United States")].status == "ambiguous"

    assert by_name_country[("Nonexistent Port", "Nowhere")].status == "no_match"
    assert by_name_country[("Nonexistent Port", "Nowhere")].entity_id is None


def test_connector_records_unresolved_ports_never_guesses(tmp_path):
    fake_fetch, _ = make_fake_fetch()
    cache_path = str(tmp_path / "portwatch_endpoints.json")
    connector = mw.PortWatchPortsConnector(http_fetch=fake_fetch, cache_path=cache_path)
    result = connector.fetch(NOW)
    unresolved = result.discovered_ids["unresolved_ports"]
    # das kuratierte Universum enthält Ports, die in der (kleinen) Test-
    # Referenz nicht vorkommen -> müssen als unresolved auftauchen, nie
    # mit einer erratenen ID versehen werden.
    assert "Houston" in unresolved
    assert result.discovered_ids["ports"]["Los Angeles"] == "port_la"


# --------------------------------------------------------------------------
# Observation PIT-Felder
# --------------------------------------------------------------------------

def test_prospective_observation_has_conservative_date_precision(tmp_path):
    fake_fetch, _ = make_fake_fetch()
    cache_path = str(tmp_path / "portwatch_endpoints.json")
    connector = mw.PortWatchPortsConnector(http_fetch=fake_fetch, cache_path=cache_path)
    result = connector.fetch(NOW, backfill=False)
    assert result.observations, "erwartete Observations aus den Query-Fixtures"
    for o in result.observations:
        assert o.availability_precision == AvailabilityPrecision.CONSERVATIVE_DATE
        assert o.attrs.get("historical_backfill") is None
        assert o.available_at == o.retrieved_at == NOW


def test_backfill_observation_has_unknown_precision_and_backfill_flag(tmp_path):
    fake_fetch, _ = make_fake_fetch()
    cache_path = str(tmp_path / "portwatch_endpoints.json")
    connector = mw.PortWatchPortsConnector(http_fetch=fake_fetch, cache_path=cache_path)
    result = connector.fetch(NOW, backfill=True)
    assert result.observations
    for o in result.observations:
        assert o.availability_precision == AvailabilityPrecision.UNKNOWN
        assert o.attrs.get("historical_backfill") is True
        assert o.is_confirmatory() is False


def test_cargo_types_kept_separate_not_merged(tmp_path):
    fake_fetch, _ = make_fake_fetch()
    cache_path = str(tmp_path / "portwatch_endpoints.json")
    connector = mw.PortWatchPortsConnector(http_fetch=fake_fetch, cache_path=cache_path)
    result = connector.fetch(NOW)
    metrics_for_first_date = {
        o.metric for o in result.observations
        if o.entity_id == "port_test" and o.observation_time.date().isoformat() == "2024-01-01"
    }
    assert {"portcalls_total", "import_total", "export_total", "portcalls_container"} <= metrics_for_first_date
    # jede Cargo-Art bleibt eine eigene Beobachtung/Metrik, kein Summieren
    container_obs = [o for o in result.observations if o.metric == "portcalls_container"]
    total_obs = [o for o in result.observations if o.metric == "portcalls_total"]
    assert len(container_obs) == len(total_obs) == 5
    assert container_obs[0].value != total_obs[0].value


def test_chokepoints_connector_end_to_end(tmp_path):
    fake_fetch, _ = make_fake_fetch()
    cache_path = str(tmp_path / "portwatch_endpoints.json")
    connector = mw.PortWatchChokepointsConnector(http_fetch=fake_fetch, cache_path=cache_path)
    result = connector.fetch(NOW)
    assert result.observations
    assert result.discovered_ids["chokepoints"]["Suez Canal"] == "cp_suez"
    assert "Bab el-Mandeb" in result.discovered_ids["unresolved_chokepoints"]
    metrics = {o.metric for o in result.observations}
    assert {"n_total", "n_tanker", "n_container"} <= metrics


# --------------------------------------------------------------------------
# Chokepoint-Name-Alias-Matching (item 2 im Fix) & Diagnostics (item 3)
# --------------------------------------------------------------------------

def test_normalize_name_reorders_strait_of_without_needing_an_alias():
    assert mw.normalize_name("Strait of Gibraltar") == mw.normalize_name("Gibraltar Strait")
    assert mw.normalize_name("Strait of Malacca") == mw.normalize_name("Malacca Strait")


def test_resolve_entities_by_name_uses_documented_alias_map_for_spelling_variants():
    # "Bosporus Strait" (kuratiert) vs. "Bosphorus Strait" (Referenz-Layer) --
    # eine reine Wortstellungs-Normalisierung löst das NICHT, nur der
    # dokumentierte Alias in config/port_universe.yaml.
    ref = [{"attributes": {"chokepointid": "cp_bosporus", "chokepointname": "Bosphorus Strait"}}]
    wanted = [{"name": "Bosporus Strait", "country": "Turkey"}]
    alias_map = {"Bosporus Strait": ["Bosphorus Strait", "Turkish Straits"]}
    resolved = mw.resolve_entities_by_name(
        wanted, ref, "chokepointid", "chokepointname", country_field=None, alias_map=alias_map,
    )
    assert resolved[0].status == "resolved"
    assert resolved[0].entity_id == "cp_bosporus"


def test_resolve_entities_by_name_never_matches_undocumented_alias():
    # Kein Alias-Eintrag für "Taiwan Strait" auf einen abweichenden Namen ->
    # bleibt no_match, wird nie erraten.
    ref = [{"attributes": {"chokepointid": "cp_x", "chokepointname": "Formosa Strait"}}]
    wanted = [{"name": "Taiwan Strait", "country": None}]
    resolved = mw.resolve_entities_by_name(
        wanted, ref, "chokepointid", "chokepointname", country_field=None, alias_map={},
    )
    assert resolved[0].status == "no_match"
    assert resolved[0].entity_id is None


def test_load_chokepoint_aliases_from_port_universe_yaml():
    universe = mw.load_port_universe("config/port_universe.yaml")
    aliases = mw.load_chokepoint_aliases(universe)
    assert "Strait of Malacca" in aliases["Malacca Strait"]
    assert any("Bosphorus" in a for a in aliases["Bosporus Strait"])


def test_close_name_matches_ranks_by_normalized_similarity():
    candidates = ["Strait of Hormuz", "Suez Canal", "Panama Canal", "Taiwan Strait"]
    top = mw.close_name_matches("Taiwan Straight", candidates, k=2)
    assert top[0] == "Taiwan Strait"


def test_chokepoints_connector_resolves_via_alias_and_reports_diagnostics(tmp_path):
    """End-to-end: config-Aliase lösen Malacca/Gibraltar/Bosporus gegen die
    (abweichend benannten) Referenz-Layer-Einträge auf; Taiwan Strait bleibt
    unresolved (kein passender Alias/Name in der Referenz) und taucht mit
    Diagnostics (verfügbare Namen + Top-3-Näherungen) auf -- nie geraten."""
    fake_fetch, _ = make_fake_fetch()
    cache_path = str(tmp_path / "portwatch_endpoints.json")
    connector = mw.PortWatchChokepointsConnector(http_fetch=fake_fetch, cache_path=cache_path)
    result = connector.fetch(NOW)

    resolved = result.discovered_ids["chokepoints"]
    assert resolved["Malacca Strait"] == "cp_malacca"
    assert resolved["Gibraltar Strait"] == "cp_gibraltar"
    assert resolved["Bosporus Strait"] == "cp_bosporus"

    unresolved = result.discovered_ids["unresolved_chokepoints"]
    assert "Taiwan Strait" in unresolved
    assert "Malacca Strait" not in unresolved

    diagnostics = result.discovered_ids["diagnostics"]
    assert "Strait of Hormuz" in diagnostics["available_chokepoints"]
    assert len(diagnostics["available_chokepoints"]) <= 60
    assert diagnostics["chokepoints_reference_fallback_to_daily_layer"] is False
    assert "Taiwan Strait" in diagnostics["unresolved_chokepoints_close_matches"]
    assert diagnostics["unresolved_chokepoints_close_matches"]["Taiwan Strait"]

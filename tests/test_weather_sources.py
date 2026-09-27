"""
tests/test_weather_sources.py

Tests für modules/external/sources/weather.py – NWS-Grid-Parsing (inkl.
ISO8601-Dauer-Expansion + Einheitenkonvertierung), issue vs. valid time,
Alerts-Klassifikation, NCEI-Normals-Parsing, NHC-Parsing (inkl. "keine
aktiven Stürme"), Exposure-Hierarchie und "fehlend -> None". Alle Tests
laufen ohne Netzwerk gegen Fixtures unter tests/fixtures/external/weather/.
"""

import json
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from modules.external.pit import AvailabilityPrecision, Observation
from modules.external.sources import weather as w

FIXTURES = Path(__file__).parent / "fixtures" / "external" / "weather"


def _load(name: str) -> dict:
    with open(FIXTURES / name, "r", encoding="utf-8") as f:
        return json.load(f)


JFK_LOCATION = {"code": "JFK", "name": "JFK", "lat": 40.6413, "lon": -73.7781, "category": "airline_hubs"}


# --------------------------------------------------------------------------- #
# ISO8601-Dauer / validTime
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("duration,expected_seconds", [
    ("PT1H", 3600),
    ("P1D", 86400),
    ("P7DT18H", 7 * 86400 + 18 * 3600),
    ("PT30M", 1800),
    ("P1W", 7 * 86400),
])
def test_parse_iso8601_duration(duration, expected_seconds):
    assert w.parse_iso8601_duration(duration).total_seconds() == expected_seconds


def test_parse_iso8601_duration_rejects_garbage():
    with pytest.raises(ValueError):
        w.parse_iso8601_duration("not-a-duration")


def test_parse_valid_time_splits_start_and_duration():
    start, duration = w.parse_valid_time("2024-01-01T12:00:00+00:00/PT6H")
    assert start == datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert duration.total_seconds() == 6 * 3600


# --------------------------------------------------------------------------- #
# Einheiten-Konvertierung
# --------------------------------------------------------------------------- #

def test_convert_grid_value_degc_to_degf():
    value, unit = w.convert_grid_value(0.0, "wmoUnit:degC")
    assert value == pytest.approx(32.0)
    assert unit == "degF"


def test_convert_grid_value_mm_to_inches():
    value, unit = w.convert_grid_value(25.4, "wmoUnit:mm")
    assert value == pytest.approx(1.0)
    assert unit == "in"


def test_convert_grid_value_kmh_to_mph():
    value, unit = w.convert_grid_value(100.0, "wmoUnit:km_h-1")
    assert value == pytest.approx(62.137119, rel=1e-4)
    assert unit == "mph"


def test_convert_grid_value_none_stays_none():
    value, unit = w.convert_grid_value(None, "wmoUnit:degC")
    assert value is None
    assert unit == "degF"


# --------------------------------------------------------------------------- #
# nws_forecast Grid-Parsing
# --------------------------------------------------------------------------- #

def test_parse_grid_response_builds_observations_with_converted_units():
    points = _load("nws_points_jfk.json")
    grid = _load("nws_grid_jfk_issue1.json")
    retrieved_at = datetime(2024, 1, 1, 9, 5, 0, tzinfo=timezone.utc)

    obs = w.parse_grid_response(points, grid, JFK_LOCATION, w.DEFAULT_GRID_ELEMENTS, retrieved_at)

    temp_obs = [o for o in obs if o.metric == "temperature"]
    assert len(temp_obs) == 3
    first = sorted(temp_obs, key=lambda o: o.forecast_valid_time)[0]
    assert first.unit == "degF"
    assert first.value == pytest.approx(32.0)  # 0 degC -> 32 degF
    assert first.entity_id == "JFK"
    assert first.series_id == "OKX/32,34"


def test_grid_forecast_issue_time_equals_grid_update_time():
    points = _load("nws_points_jfk.json")
    grid = _load("nws_grid_jfk_issue1.json")
    retrieved_at = datetime(2024, 1, 1, 9, 5, 0, tzinfo=timezone.utc)
    obs = w.parse_grid_response(points, grid, JFK_LOCATION, w.DEFAULT_GRID_ELEMENTS, retrieved_at)
    expected_issue = datetime(2024, 1, 1, 9, 0, 0, tzinfo=timezone.utc)
    assert all(o.forecast_issue_time == expected_issue for o in obs)
    assert all(o.source_release_time == expected_issue for o in obs)


def test_grid_forecast_valid_time_is_interval_start_not_issue_time():
    points = _load("nws_points_jfk.json")
    grid = _load("nws_grid_jfk_issue1.json")
    retrieved_at = datetime(2024, 1, 1, 9, 5, 0, tzinfo=timezone.utc)
    obs = w.parse_grid_response(points, grid, JFK_LOCATION, w.DEFAULT_GRID_ELEMENTS, retrieved_at)
    max_temp = [o for o in obs if o.metric == "maxTemperature"][0]
    assert max_temp.forecast_valid_time == datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    assert max_temp.forecast_valid_time != max_temp.forecast_issue_time


def test_grid_available_at_equals_retrieved_at_not_update_time():
    """available_at = retrieved_at, auch wenn updateTime (source_release_time)
    zeitlich davor liegt – wir können keine frühere Verfügbarkeit für unseren
    eigenen Abruf belegen."""
    points = _load("nws_points_jfk.json")
    grid = _load("nws_grid_jfk_issue1.json")
    retrieved_at = datetime(2024, 1, 1, 9, 5, 0, tzinfo=timezone.utc)
    obs = w.parse_grid_response(points, grid, JFK_LOCATION, w.DEFAULT_GRID_ELEMENTS, retrieved_at)
    for o in obs:
        assert o.available_at == retrieved_at
        assert o.source_release_time < o.available_at


def test_grid_missing_value_becomes_none_not_zero():
    points = _load("nws_points_jfk.json")
    grid = _load("nws_grid_jfk_issue1.json")
    retrieved_at = datetime(2024, 1, 1, 9, 5, 0, tzinfo=timezone.utc)
    obs = w.parse_grid_response(points, grid, JFK_LOCATION, w.DEFAULT_GRID_ELEMENTS, retrieved_at)
    snow_obs = [o for o in obs if o.metric == "snowfallAmount"]
    null_entry = [o for o in snow_obs if o.value is None]
    assert len(null_entry) == 1


# --------------------------------------------------------------------------- #
# Forecast-Revisionen: gleiche valid_time, unterschiedliche issue_time
# --------------------------------------------------------------------------- #

def test_two_issues_produce_distinct_observations_for_same_valid_time():
    points = _load("nws_points_jfk.json")
    grid1 = _load("nws_grid_jfk_issue1.json")
    grid2 = _load("nws_grid_jfk_issue2.json")
    r1 = datetime(2024, 1, 1, 9, 5, 0, tzinfo=timezone.utc)
    r2 = datetime(2024, 1, 1, 15, 5, 0, tzinfo=timezone.utc)
    obs1 = w.parse_grid_response(points, grid1, JFK_LOCATION, ["temperature"], r1)
    obs2 = w.parse_grid_response(points, grid2, JFK_LOCATION, ["temperature"], r2)

    same_valid_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    o1 = [o for o in obs1 if o.forecast_valid_time == same_valid_time][0]
    o2 = [o for o in obs2 if o.forecast_valid_time == same_valid_time][0]

    assert o1.forecast_issue_time != o2.forecast_issue_time
    assert o1.value == pytest.approx(32.0)   # 0 degC
    assert o2.value == pytest.approx(37.4)   # 3 degC
    # verschiedene valid_time zwischen den Issues darf NIE verglichen werden
    different_valid_time = datetime(2024, 1, 1, 15, 0, 0, tzinfo=timezone.utc)
    o2_only = [o for o in obs2 if o.forecast_valid_time == different_valid_time]
    o1_only = [o for o in obs1 if o.forecast_valid_time == different_valid_time]
    assert o2_only and not o1_only  # issue1 hatte diesen validTime gar nicht


# --------------------------------------------------------------------------- #
# nws_alerts – Klassifikation
# --------------------------------------------------------------------------- #

def test_alerts_response_filters_to_configured_event_types():
    alerts = _load("nws_alerts_active.json")
    retrieved_at = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    obs = w.parse_alerts_response(alerts, "JFK", w.DEFAULT_ALERT_EVENT_TYPES, retrieved_at)
    events = {o.attrs["event"] for o in obs if o.metric == "alert_active"}
    assert events == {"Winter Storm Warning"}   # "Flood Watch" ist nicht in DEFAULT_ALERT_EVENT_TYPES


def test_alerts_response_produces_count_by_class():
    alerts = _load("nws_alerts_active.json")
    retrieved_at = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    obs = w.parse_alerts_response(alerts, "JFK", w.DEFAULT_ALERT_EVENT_TYPES, retrieved_at)
    counts = [o for o in obs if o.metric == "alert_count"]
    assert len(counts) == 1
    assert counts[0].value == 1.0
    assert counts[0].attrs["event"] == "Winter Storm Warning"


def test_alerts_release_time_is_sent_field_available_at_is_retrieved():
    alerts = _load("nws_alerts_active.json")
    retrieved_at = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    obs = w.parse_alerts_response(alerts, "JFK", w.DEFAULT_ALERT_EVENT_TYPES, retrieved_at)
    active = [o for o in obs if o.metric == "alert_active"][0]
    assert active.available_at == retrieved_at
    assert active.source_release_time == datetime(2024, 1, 1, 5, 0, 0, tzinfo=timezone.utc)  # sent 00:00-05:00


# --------------------------------------------------------------------------- #
# ncei_normals
# --------------------------------------------------------------------------- #

def test_parse_station_search_returns_candidates_with_coordinates():
    search = _load("ncei_station_search_jfk.json")
    candidates = w.parse_station_search_response(search)
    assert len(candidates) == 2
    assert candidates[0]["id"] == "GHCND:USW00094789"


def test_nearest_station_picks_closest_by_distance():
    search = _load("ncei_station_search_jfk.json")
    candidates = w.parse_station_search_response(search)
    nearest = w.nearest_station(candidates, 40.6413, -73.7781)  # JFK coordinates
    assert nearest["id"] == "GHCND:USW00094789"
    assert nearest["distance_km"] < 5.0


def test_nearest_station_never_hardcodes_when_no_candidates():
    assert w.nearest_station([], 40.0, -73.0) is None


def test_parse_normals_response_missing_value_is_none_not_zero():
    records = _load("ncei_normals_jfk.json")
    retrieved_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    obs = w.parse_normals_response(records, "GHCND:USW00094789", "JFK", retrieved_at)
    prcp_day2 = [o for o in obs if o.metric == "DLY-PRCP-NORMAL"
                 and o.observation_time == datetime(2010, 1, 2, tzinfo=timezone.utc)][0]
    assert prcp_day2.value is None   # "-9999" Sentinel -> None, nie 0


def test_parse_normals_response_values_and_units():
    records = _load("ncei_normals_jfk.json")
    retrieved_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    obs = w.parse_normals_response(records, "GHCND:USW00094789", "JFK", retrieved_at)
    tavg = [o for o in obs if o.metric == "DLY-TAVG-NORMAL"
            and o.observation_time == datetime(2010, 1, 1, tzinfo=timezone.utc)][0]
    assert tavg.value == pytest.approx(34.5)
    assert tavg.unit == "degF"


# --------------------------------------------------------------------------- #
# nhc_storms
# --------------------------------------------------------------------------- #

def test_nhc_no_active_storms_returns_empty_observations():
    storms = _load("nhc_current_storms_none.json")
    retrieved_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    obs = w.parse_current_storms(storms, retrieved_at, exposure_regions=[])
    assert obs == []


def test_nhc_skips_non_dict_storm_entries_instead_of_crashing():
    storms = {"activeStorms": ["AL052024", 42, None]}
    retrieved_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    obs = w.parse_current_storms(storms, retrieved_at, exposure_regions=[])
    assert obs == []


def test_nhc_raises_schema_error_when_top_level_is_not_a_dict():
    retrieved_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(w.NhcSchemaError) as exc_info:
        w.parse_current_storms(["not", "a", "dict"], retrieved_at, exposure_regions=[])
    assert "body_snippet" in exc_info.value.diagnostics


def test_nhc_raises_schema_error_when_active_storms_is_not_a_list():
    retrieved_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(w.NhcSchemaError):
        w.parse_current_storms({"activeStorms": "AL052024"}, retrieved_at, exposure_regions=[])


def test_nhc_connector_reports_schema_changed_for_unexpected_active_storms_type():
    connector = w.NhcStormsConnector(source_cfg={})
    fake_res = SimpleNamespace(
        json=lambda: {"activeStorms": "not-a-list"},
        content=b'{"activeStorms": "not-a-list"}',
        content_type="application/json",
        retrieved_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        url="https://www.nhc.noaa.gov/CurrentStorms.json", status=200,
        fingerprint="fp", content_hash="hash", bytes=10,
    )
    with patch.object(w.http, "fetch", return_value=fake_res):
        result = connector.fetch(datetime(2024, 1, 1, tzinfo=timezone.utc))
    assert result.status.value == "SCHEMA_CHANGED"
    assert "body_snippet" in result.discovered_ids["diagnostics"]


def test_nhc_active_storm_parses_core_fields_and_signed_latlon():
    storms = _load("nhc_current_storms_active.json")
    retrieved_at = datetime(2024, 9, 27, tzinfo=timezone.utc)
    obs = w.parse_current_storms(storms, retrieved_at, exposure_regions=[])
    by_metric = {o.metric: o for o in obs if o.entity_id == "AL052024"}
    assert by_metric["lat"].value == pytest.approx(25.5)
    assert by_metric["lon"].value == pytest.approx(-84.0)   # 'W' -> negative
    assert by_metric["intensity_kt"].value == pytest.approx(90.0)
    assert by_metric["pressure_mb"].value == pytest.approx(965.0)


def test_nhc_forecast_issue_time_is_advisory_issuance():
    storms = _load("nhc_current_storms_active.json")
    retrieved_at = datetime(2024, 9, 27, tzinfo=timezone.utc)
    obs = w.parse_current_storms(storms, retrieved_at, exposure_regions=[])
    expected = datetime(2024, 9, 26, 21, 0, 0, tzinfo=timezone.utc)
    assert all(o.forecast_issue_time == expected for o in obs)


def test_nhc_distance_to_exposure_uses_forecast_track_when_present():
    storms = _load("nhc_current_storms_active.json")
    retrieved_at = datetime(2024, 9, 27, tzinfo=timezone.utc)
    regions = [{"code": "FL_COAST", "lat": 27.9944, "lon": -81.7603}]
    obs = w.parse_current_storms(storms, retrieved_at, exposure_regions=regions)
    dist = [o for o in obs if o.metric == "min_distance_to_exposure_km"][0]
    assert dist.value is not None
    assert dist.value > 0
    assert dist.attrs["nearest_region"] == "FL_COAST"


def test_nhc_distance_is_none_without_forecast_track():
    storms = json.loads(json.dumps(_load("nhc_current_storms_active.json")))
    del storms["activeStorms"][0]["forecastTrack"]
    retrieved_at = datetime(2024, 9, 27, tzinfo=timezone.utc)
    regions = [{"code": "FL_COAST", "lat": 27.9944, "lon": -81.7603}]
    obs = w.parse_current_storms(storms, retrieved_at, exposure_regions=regions)
    dist = [o for o in obs if o.metric == "min_distance_to_exposure_km"][0]
    assert dist.value is None
    assert dist.attrs["limitation"] is not None


# --------------------------------------------------------------------------- #
# noaa_storm_events – reine Parse-Logik + Backfill-Gate
# --------------------------------------------------------------------------- #

def test_parse_storm_events_counts_by_state_and_type():
    csv_text = (FIXTURES / "storm_events_details_sample.csv").read_text()
    counts = w.parse_storm_events_counts(csv_text)
    assert counts[("TEXAS", "Tornado")] == 2
    assert counts[("TEXAS", "Flood")] == 1
    assert counts[("FLORIDA", "Hurricane")] == 1


def test_storm_events_disabled_by_default():
    connector = w.NoaaStormEventsConnector(source_cfg={})
    result = connector.fetch(datetime(2024, 1, 1, tzinfo=timezone.utc))
    assert result.status.value == "DEFERRED"
    assert result.observations == []


def test_pick_details_filename_selects_requested_year():
    listing = (FIXTURES / "storm_events_listing.html").read_text()
    name = w.NoaaStormEventsConnector._pick_details_filename(listing, 2024)
    assert name == "StormEvents_details-ftp_v1.0_d2024_c20250115.csv.gz"


# --------------------------------------------------------------------------- #
# ecmwf_open_data – Registry-only / DEFERRED
# --------------------------------------------------------------------------- #

def test_ecmwf_connector_always_deferred():
    connector = w.EcmwfOpenDataConnector(source_cfg={})
    result = connector.fetch(datetime(2024, 1, 1, tzinfo=timezone.utc))
    assert result.status.value == "DEFERRED"


def _registry_sources_by_id() -> dict:
    import yaml
    reg_path = Path(__file__).resolve().parent.parent / "config" / "external_sources" / "weather.yaml"
    reg = yaml.safe_load(reg_path.read_text())
    return {entry["source_id"]: entry for entry in reg["sources"]}


def test_ecmwf_registry_has_status_override_deferred():
    entry = _registry_sources_by_id()["ecmwf_open_data"]
    assert entry["license_status"] == "OK"
    assert entry["status_override"] == "DEFERRED"


# --------------------------------------------------------------------------- #
# CONNECTORS-Registry
# --------------------------------------------------------------------------- #

def test_connectors_registry_has_all_six_sources():
    expected = {"nws_forecast", "nws_alerts", "ncei_normals", "nhc_storms",
                "noaa_storm_events", "ecmwf_open_data"}
    assert set(w.CONNECTORS) == expected


def test_all_registry_source_ids_match_yaml_config():
    sources_by_id = _registry_sources_by_id()
    assert set(sources_by_id) == set(w.CONNECTORS)
    for source_id, cls in w.CONNECTORS.items():
        assert cls.source_id == source_id


def test_registry_entries_pass_shared_registry_validation():
    """Muss gegen die tatsächliche geteilte Registry-Infrastruktur
    (modules/external/registry.py) validieren, nicht nur strukturell zu
    weather.py passen."""
    from modules.external.registry import load_source_configs
    configs = load_source_configs()
    for source_id in w.CONNECTORS:
        assert source_id in configs, f"{source_id} fehlt in der geladenen Registry"
        assert configs[source_id]["family"] == "weather"


# --------------------------------------------------------------------------- #
# Locations-Config lädt korrekt
# --------------------------------------------------------------------------- #

def test_weather_locations_config_loads_and_has_curated_categories():
    cfg = w.load_weather_locations()
    assert "airline_hubs" in cfg
    assert "population_regions" in cfg
    assert "coastal_exposure_regions" in cfg
    codes = {loc["code"] for loc in w.all_locations(cfg)}
    assert "JFK" in codes
    assert "MEM_FDX" in codes


def test_location_by_code_returns_none_for_unknown_code():
    assert w.location_by_code("NOT_A_REAL_CODE") is None

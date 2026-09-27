"""
tests/test_weather_features.py

Tests für modules/external/sources/weather_features.py: HDD/CDD-Berechnung,
Normals-Anomalie-Mathematik, Forecast-Revisionen (NUR gleiche valid_time),
fehlende Daten -> None (nie 0), Hub-/Utility-/P&C-Feature-Grundlagen.

Seit der NWS-Tagesaggregation (siehe modules/external/sources/weather.py:
aggregate_daily_forecast) liefert der Konnektor bereits tagesaggregierte
Metriken (tmean/tmax/tmin/hdd/cdd/precip_total/snow_total/ice_total/
pop_max/wind_max/gust_max) mit forecast_valid_time = lokaler Tagesbeginn;
weather_features.py wählt daraus nur noch je Tag die jüngste Issue-Version
(Revision), aggregiert NICHT mehr selbst über Stunden.
"""

from datetime import date, datetime, timezone

import pytest

from modules.external.pit import AvailabilityPrecision, Observation
from modules.external.sources import weather_features as wf


def _obs(entity_id, metric, value, forecast_valid_time=None, forecast_issue_time=None,
         observation_time=None, unit="degF", attrs=None, dataset="grid_forecast_daily",
         source_id="nws_forecast"):
    obs_time = observation_time or forecast_valid_time or datetime(2024, 1, 1, tzinfo=timezone.utc)
    return Observation(
        source_id=source_id, dataset=dataset, series_id="OKX/32,34", entity_id=entity_id,
        metric=metric, value=value, unit=unit, observation_time=obs_time,
        available_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        retrieved_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        availability_precision=AvailabilityPrecision.EXACT_TIMESTAMP, parser_version="1",
        forecast_issue_time=forecast_issue_time, forecast_valid_time=forecast_valid_time,
        attrs=attrs or {},
    )


def _day(entity_id, d, tmean=None, tmax=None, tmin=None, hdd=None, cdd=None,
         issue=datetime(2024, 1, 1, tzinfo=timezone.utc)):
    """Baut die (bereits tagesaggregierten) Observations EINES Kalendertags,
    wie sie der Konnektor jetzt liefert (forecast_valid_time = Tagesbeginn)."""
    vt = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    out = []
    for metric, value in (("tmean", tmean), ("tmax", tmax), ("tmin", tmin), ("hdd", hdd), ("cdd", cdd)):
        if value is not None:
            out.append(_obs(entity_id, metric, value, forecast_valid_time=vt, forecast_issue_time=issue))
    return out


# --------------------------------------------------------------------------- #
# HDD/CDD
# --------------------------------------------------------------------------- #

def test_hdd_cdd_base_65f():
    d = date(2024, 1, 1)
    obs = _day("JFK", d, tmean=65.0, hdd=0.0, cdd=0.0)
    daily = wf.daily_temperature_aggregates(obs)
    assert daily["JFK"][d]["tmean"] == pytest.approx(65.0)
    assert daily["JFK"][d]["hdd"] == pytest.approx(0.0)
    assert daily["JFK"][d]["cdd"] == pytest.approx(0.0)


def test_hdd_positive_when_cold():
    d = date(2024, 1, 1)
    obs = _day("JFK", d, tmean=40.0, hdd=25.0, cdd=0.0)
    daily = wf.daily_temperature_aggregates(obs)
    assert daily["JFK"][d]["hdd"] == pytest.approx(25.0)
    assert daily["JFK"][d]["cdd"] == pytest.approx(0.0)


def test_cdd_positive_when_hot():
    d = date(2024, 7, 1)
    obs = _day("JFK", d, tmean=90.0, hdd=0.0, cdd=25.0)
    daily = wf.daily_temperature_aggregates(obs)
    assert daily["JFK"][d]["cdd"] == pytest.approx(25.0)
    assert daily["JFK"][d]["hdd"] == pytest.approx(0.0)


def test_daily_aggregates_missing_entity_returns_no_entry_not_zero():
    daily = wf.daily_temperature_aggregates([])
    assert daily == {}


def test_daily_aggregates_missing_field_is_none_not_zero():
    """tmax fehlt für diesen Tag -> None, nie 0 (nur tmean/hdd/cdd geliefert)."""
    d = date(2024, 1, 1)
    obs = _day("JFK", d, tmean=50.0, hdd=15.0, cdd=0.0)
    daily = wf.daily_temperature_aggregates(obs)
    assert daily["JFK"][d]["tmax"] is None
    assert daily["JFK"][d]["tmin"] is None


# --------------------------------------------------------------------------- #
# Anomalien vs. Normals
# --------------------------------------------------------------------------- #

def test_anomaly_math_basic():
    assert wf.anomaly(70.0, 65.0) == pytest.approx(5.0)
    assert wf.anomaly(60.0, 65.0) == pytest.approx(-5.0)


def test_anomaly_none_when_value_missing():
    assert wf.anomaly(None, 65.0) is None


def test_anomaly_none_when_normal_missing():
    assert wf.anomaly(70.0, None) is None


def test_anomalies_for_day_uses_month_day_index_not_year():
    d = date(2024, 1, 1)
    forecast_obs = _day("JFK", d, tmean=70.0)
    daily = wf.daily_temperature_aggregates(forecast_obs)
    normals_obs = [_obs("JFK", "DLY-TAVG-NORMAL", 60.0,
                         observation_time=datetime(2010, 1, 1, tzinfo=timezone.utc),  # anderes Jahr!
                         dataset="normals_daily_1991_2020", source_id="ncei_normals")]
    normals = wf.normals_by_month_day(normals_obs)
    result = wf.anomalies_for_day(daily, "JFK", d, normals)
    assert result["tmean_anomaly"] == pytest.approx(10.0)


def test_extreme_flag_true_above_threshold():
    assert wf.extreme_flag(80.0, 65.0, threshold=10.0) is True


def test_extreme_flag_false_below_threshold():
    assert wf.extreme_flag(70.0, 65.0, threshold=10.0) is False


def test_extreme_flag_none_when_missing():
    assert wf.extreme_flag(None, 65.0, threshold=10.0) is None


# --------------------------------------------------------------------------- #
# Forecast-Revisionen: NUR gleiche valid_time
# --------------------------------------------------------------------------- #

def test_forecast_revision_same_valid_time():
    older = _obs("JFK", "tmean", 50.0, forecast_valid_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
                 forecast_issue_time=datetime(2024, 1, 1, 0, tzinfo=timezone.utc))
    newer = _obs("JFK", "tmean", 55.0, forecast_valid_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
                 forecast_issue_time=datetime(2024, 1, 1, 6, tzinfo=timezone.utc))
    assert wf.forecast_revision(newer, older) == pytest.approx(5.0)


def test_forecast_revision_none_for_different_valid_time():
    older = _obs("JFK", "tmean", 50.0, forecast_valid_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
                 forecast_issue_time=datetime(2024, 1, 1, 0, tzinfo=timezone.utc))
    newer = _obs("JFK", "tmean", 55.0, forecast_valid_time=datetime(2024, 1, 2, tzinfo=timezone.utc),
                 forecast_issue_time=datetime(2024, 1, 1, 6, tzinfo=timezone.utc))
    assert wf.forecast_revision(newer, older) is None


def test_forecast_revision_none_when_newer_is_not_actually_later():
    a = _obs("JFK", "tmean", 50.0, forecast_valid_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
             forecast_issue_time=datetime(2024, 1, 1, 6, tzinfo=timezone.utc))
    b = _obs("JFK", "tmean", 55.0, forecast_valid_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
             forecast_issue_time=datetime(2024, 1, 1, 0, tzinfo=timezone.utc))
    # b wurde VOR a ausgegeben -> als "newer" übergeben ist das ein Fehlaufruf
    assert wf.forecast_revision(b, a) is None


def test_forecast_revision_none_for_different_entity():
    a = _obs("JFK", "tmean", 50.0, forecast_valid_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
             forecast_issue_time=datetime(2024, 1, 1, 0, tzinfo=timezone.utc))
    b = _obs("EWR", "tmean", 55.0, forecast_valid_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
             forecast_issue_time=datetime(2024, 1, 1, 6, tzinfo=timezone.utc))
    assert wf.forecast_revision(b, a) is None


def test_revisions_for_metric_picks_two_latest_issues_per_valid_time():
    vt = datetime(2024, 1, 1, tzinfo=timezone.utc)
    obs = [
        _obs("JFK", "tmean", 50.0, forecast_valid_time=vt, forecast_issue_time=datetime(2024, 1, 1, 0, tzinfo=timezone.utc)),
        _obs("JFK", "tmean", 52.0, forecast_valid_time=vt, forecast_issue_time=datetime(2024, 1, 1, 6, tzinfo=timezone.utc)),
        _obs("JFK", "tmean", 55.0, forecast_valid_time=vt, forecast_issue_time=datetime(2024, 1, 1, 12, tzinfo=timezone.utc)),
    ]
    revisions = wf.revisions_for_metric(obs, "JFK", "tmean")
    assert len(revisions) == 1
    assert revisions[0]["revision"] == pytest.approx(3.0)  # 55 - 52 (die zwei jüngsten)


def test_revisions_for_metric_empty_with_single_issue():
    vt = datetime(2024, 1, 1, tzinfo=timezone.utc)
    obs = [_obs("JFK", "tmean", 50.0, forecast_valid_time=vt, forecast_issue_time=datetime(2024, 1, 1, 0, tzinfo=timezone.utc))]
    assert wf.revisions_for_metric(obs, "JFK", "tmean") == []


def test_daily_temperature_aggregates_uses_latest_issue_for_same_valid_day():
    """Zwei Issues für denselben Gültigkeitstag -> nur die jüngste Version
    fließt in daily_temperature_aggregates ein (Revision, kein Duplikat)."""
    vt = datetime(2024, 1, 1, tzinfo=timezone.utc)
    obs = [
        _obs("JFK", "tmean", 50.0, forecast_valid_time=vt, forecast_issue_time=datetime(2024, 1, 1, 0, tzinfo=timezone.utc)),
        _obs("JFK", "tmean", 58.0, forecast_valid_time=vt, forecast_issue_time=datetime(2024, 1, 1, 12, tzinfo=timezone.utc)),
    ]
    daily = wf.daily_temperature_aggregates(obs)
    assert daily["JFK"][date(2024, 1, 1)]["tmean"] == pytest.approx(58.0)


# --------------------------------------------------------------------------- #
# Rolling-Aggregate
# --------------------------------------------------------------------------- #

def test_rolling_aggregate_mean():
    daily = {date(2024, 1, 1): 10.0, date(2024, 1, 2): 20.0, date(2024, 1, 3): 30.0}
    assert wf.rolling_aggregate(daily, date(2024, 1, 3), 3, "mean") == pytest.approx(20.0)


def test_rolling_aggregate_sum():
    daily = {date(2024, 1, 1): 1.0, date(2024, 1, 2): 2.0}
    assert wf.rolling_aggregate(daily, date(2024, 1, 2), 2, "sum") == pytest.approx(3.0)


def test_rolling_aggregate_none_when_no_data_in_window():
    assert wf.rolling_aggregate({}, date(2024, 1, 1), 7, "mean") is None


# --------------------------------------------------------------------------- #
# Airline-Hub-Features: fehlende Daten -> None, nicht 0/False
# --------------------------------------------------------------------------- #

def test_hub_features_missing_hub_data_yields_none_fields():
    result = wf.airline_hub_features(["ATL"], [], [], date(2024, 1, 1))
    hub = result["hubs"]["ATL"]
    assert hub["hub_precip_probability"] is None
    assert hub["hub_snow_amount"] is None
    assert hub["hub_wind_speed"] is None
    assert hub["hub_thunderstorm_alert"] is None
    assert result["network_hubs_exposed"] == 0
    assert result["network_exposure_share"] is None  # kein Hub mit Daten -> unbekannt, nicht 0.0


def test_hub_features_detects_exposure_from_snow_and_alert():
    d = date(2024, 1, 1)
    vt = datetime(2024, 1, 1, tzinfo=timezone.utc)
    forecast_obs = [
        _obs("ATL", "snow_total", 3.0, forecast_valid_time=vt,
             forecast_issue_time=datetime(2024, 1, 1, 0, tzinfo=timezone.utc), unit="in"),
    ]
    alert_obs = [
        _obs("ATL", "alert_count", 1.0, dataset="alerts_active", source_id="nws_alerts",
             attrs={"event": "Winter Storm Warning"}),
    ]
    result = wf.airline_hub_features(["ATL"], forecast_obs, alert_obs, d)
    hub = result["hubs"]["ATL"]
    assert hub["hub_snow_amount"] == pytest.approx(3.0)
    assert hub["hub_winter_storm_alert"] is True
    assert result["network_hubs_exposed"] == 1
    assert result["network_exposure_share"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Construction-Disruption-Days
# --------------------------------------------------------------------------- #

def test_construction_disruption_days_counts_freeze_and_heavy_precip():
    obs = (
        _day("TX_CONSTR", date(2024, 1, 1), tmin=20.0)
        + [_obs("TX_CONSTR", "precip_total", 1.5,
                forecast_valid_time=datetime(2024, 1, 2, tzinfo=timezone.utc),
                forecast_issue_time=datetime(2024, 1, 1, tzinfo=timezone.utc), unit="in")]
    )
    days = wf.construction_disruption_days("TX_CONSTR", obs, date(2024, 1, 2), 3)
    assert days == 2


def test_construction_disruption_days_none_without_any_data():
    assert wf.construction_disruption_days("NOWHERE", [], date(2024, 1, 2), 3) is None


# --------------------------------------------------------------------------- #
# P&C: active_tropical_system None wenn kein Storm-Abruf, nicht False
# --------------------------------------------------------------------------- #

def test_pc_features_active_tropical_system_none_without_storm_fetch():
    result = wf.pc_insurance_features(["FL_COAST"], [], [])
    assert result["active_tropical_system"] is None
    assert result["min_distance_to_track_km"] is None
    assert result["wind_probability_34kt"] is None


def test_pc_features_detects_active_storm():
    storm_obs = [_obs("AL052024", "intensity_kt", 90.0, dataset="active_storms", source_id="nhc_storms",
                       forecast_issue_time=datetime(2024, 9, 26, 21, tzinfo=timezone.utc))]
    result = wf.pc_insurance_features(["FL_COAST"], [], storm_obs)
    assert result["active_tropical_system"] is True

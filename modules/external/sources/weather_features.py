"""
modules/external/sources/weather_features.py – reine Feature-Engineering-
Funktionen auf Basis von modules.external.pit.Observation-Listen (Ausgabe der
Weather-Konnektoren in modules/external/sources/weather.py).

Keine Netzwerkzugriffe, keine Seiteneffekte. Fehlende Eingabedaten -> None
(NIE 0 als Ersatz – 0 wäre eine Aussage über den Wert, None ist "unbekannt").

HDD/CDD-Konvention (NCEI): HDD_tag = max(0, 65°F - Tmean_tag),
CDD_tag = max(0, Tmean_tag - 65°F), mit Tmean_tag = (Tmax_tag + Tmin_tag) / 2
(NCEI-Standardformel für tägliche Normals; für Stundenwerte verwenden wir den
Mittelwert aller Stundenwerte des Tages als Tmean, was für Feature-Zwecke
hinreichend nah an der NCEI-Konvention liegt – dokumentierte Vereinfachung).
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Any, Iterable

from modules.external.pit import Observation

HDD_CDD_BASE_F = 65.0


# --------------------------------------------------------------------------- #
# Basis-Hilfsfunktionen
# --------------------------------------------------------------------------- #

def _date_of(dt: datetime | None) -> date | None:
    return dt.date() if dt is not None else None


def latest_per_valid_time(observations: Iterable[Observation]) -> dict:
    """Für (entity_id, metric, forecast_valid_time) die Observation mit dem
    jüngsten forecast_issue_time (die aktuellste Prognose für diesen
    Gültigkeitszeitpunkt). Observations ohne forecast_valid_time werden
    übersprungen (das ist für Alerts/Storms zuständig, nicht für Grid-
    Forecasts)."""
    best: dict[tuple, Observation] = {}
    for o in observations:
        if o.forecast_valid_time is None:
            continue
        key = (o.entity_id, o.metric, o.forecast_valid_time)
        cur = best.get(key)
        if cur is None or (o.forecast_issue_time or o.available_at) > (cur.forecast_issue_time or cur.available_at):
            best[key] = o
    return best


def group_by_entity_metric_date(observations: Iterable[Observation],
                                 use_latest_issue: bool = True) -> dict:
    """entity_id -> metric -> date -> list[value] (nur Werte ungleich None).
    use_latest_issue=True: pro (entity, metric, valid_time) nur die jüngste
    Prognoseversion verwenden (Standardfall für Aggregation/Anomalie)."""
    if use_latest_issue:
        obs_iter = latest_per_valid_time(observations).values()
    else:
        obs_iter = observations
    out: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for o in obs_iter:
        d = _date_of(o.forecast_valid_time or o.observation_time)
        if d is None or o.value is None:
            continue
        out[o.entity_id][o.metric][d].append(o.value)
    return out


# --------------------------------------------------------------------------- #
# Tägliche Aggregate aus stündlichen/Perioden-Prognosen
# --------------------------------------------------------------------------- #

def daily_temperature_aggregates(observations: Iterable[Observation]) -> dict:
    """entity_id -> date -> {tmean, tmax, tmin, hdd, cdd} in °F.
    tmax/tmin nutzen bevorzugt die 'maxTemperature'/'minTemperature'-Grid-
    Elemente (Tagesperioden); fallen sie für einen Tag aus, wird auf
    max()/min() der stündlichen 'temperature'-Werte dieses Tages
    zurückgefallen. tmean = Mittel der stündlichen 'temperature'-Werte."""
    grouped = group_by_entity_metric_date(observations)
    out: dict = {}
    for entity, metrics in grouped.items():
        out[entity] = {}
        hourly_by_date = metrics.get("temperature", {})
        maxt_by_date = metrics.get("maxTemperature", {})
        mint_by_date = metrics.get("minTemperature", {})
        all_dates = set(hourly_by_date) | set(maxt_by_date) | set(mint_by_date)
        for d in all_dates:
            hourly_vals = hourly_by_date.get(d, [])
            tmean = statistics.fmean(hourly_vals) if hourly_vals else None
            tmax = (max(maxt_by_date[d]) if d in maxt_by_date and maxt_by_date[d]
                    else (max(hourly_vals) if hourly_vals else None))
            tmin = (min(mint_by_date[d]) if d in mint_by_date and mint_by_date[d]
                    else (min(hourly_vals) if hourly_vals else None))
            hdd = max(0.0, HDD_CDD_BASE_F - tmean) if tmean is not None else None
            cdd = max(0.0, tmean - HDD_CDD_BASE_F) if tmean is not None else None
            out[entity][d] = {"tmean": tmean, "tmax": tmax, "tmin": tmin, "hdd": hdd, "cdd": cdd}
    return out


def _daily_sum(observations: Iterable[Observation], metric: str) -> dict:
    grouped = group_by_entity_metric_date(observations)
    out: dict = {}
    for entity, metrics in grouped.items():
        by_date = metrics.get(metric, {})
        out[entity] = {d: sum(vals) for d, vals in by_date.items() if vals}
    return out


def daily_precip_snow_ice_wind(observations: Iterable[Observation]) -> dict:
    """entity_id -> date -> {precip_in, snow_in, ice_in, wind_mph_max,
    gust_mph_max, pop_max}. precip/snow/ice sind Tagessummen (NWS liefert sie
    bereits als Periodenmengen); Wind/Gust/PoP sind Tagesmaxima."""
    grouped = group_by_entity_metric_date(observations)
    out: dict = {}
    for entity, metrics in grouped.items():
        out[entity] = {}
        precip = metrics.get("quantitativePrecipitation", {})
        snow = metrics.get("snowfallAmount", {})
        ice = metrics.get("iceAccumulation", {})
        wind = metrics.get("windSpeed", {})
        gust = metrics.get("windGust", {})
        pop = metrics.get("probabilityOfPrecipitation", {})
        all_dates = set(precip) | set(snow) | set(ice) | set(wind) | set(gust) | set(pop)
        for d in all_dates:
            out[entity][d] = {
                "precip_in": sum(precip[d]) if precip.get(d) else None,
                "snow_in": sum(snow[d]) if snow.get(d) else None,
                "ice_in": sum(ice[d]) if ice.get(d) else None,
                "wind_mph_max": max(wind[d]) if wind.get(d) else None,
                "gust_mph_max": max(gust[d]) if gust.get(d) else None,
                "pop_max": max(pop[d]) if pop.get(d) else None,
            }
    return out


# --------------------------------------------------------------------------- #
# Rolling-Aggregate (1d/3d/7d)
# --------------------------------------------------------------------------- #

def rolling_aggregate(daily: dict, as_of: date, window_days: int, how: str = "mean") -> float | None:
    """daily: date -> value (bereits None-gefiltert vom Aufrufer erwartet,
    aber wir filtern hier defensiv nochmal). how: 'mean' oder 'sum'."""
    vals = []
    for i in range(window_days):
        d = as_of - timedelta(days=i)
        v = daily.get(d)
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    return sum(vals) if how == "sum" else statistics.fmean(vals)


# --------------------------------------------------------------------------- #
# Anomalien vs. NCEI-Normals
# --------------------------------------------------------------------------- #

_NORMAL_METRIC_FOR = {
    "tmean": "DLY-TAVG-NORMAL", "tmax": "DLY-TMAX-NORMAL", "tmin": "DLY-TMIN-NORMAL",
    "hdd": "DLY-HTDD-NORMAL", "cdd": "DLY-CLDD-NORMAL", "precip_in": "DLY-PRCP-NORMAL",
}


def normals_by_month_day(normals_observations: Iterable[Observation]) -> dict:
    """entity_id -> metric -> (month, day) -> value. NCEI-Normals sind auf
    ein Platzhalterjahr datiert; wir indexieren daher über (Monat, Tag), NIE
    über das Jahr der Normal-Beobachtung."""
    out: dict = defaultdict(lambda: defaultdict(dict))
    for o in normals_observations:
        if o.value is None or o.observation_time is None:
            continue
        md = (o.observation_time.month, o.observation_time.day)
        out[o.entity_id][o.metric][md] = o.value
    return out


def anomaly(value: float | None, normal_value: float | None) -> float | None:
    if value is None or normal_value is None:
        return None
    return value - normal_value


def anomalies_for_day(daily_values: dict, entity: str, d: date, normals: dict) -> dict:
    """daily_values: Ausgabe von daily_temperature_aggregates()/-precip...
    (entity -> date -> {feature: value}). Gibt {feature}_anomaly zurück,
    None wenn Wert oder Normal fehlt."""
    out = {}
    day_vals = (daily_values.get(entity) or {}).get(d) or {}
    entity_normals = normals.get(entity) or {}
    md = (d.month, d.day)
    for feature, normal_metric in _NORMAL_METRIC_FOR.items():
        v = day_vals.get(feature)
        n = (entity_normals.get(normal_metric) or {}).get(md)
        out[f"{feature}_anomaly"] = anomaly(v, n)
    return out


# --------------------------------------------------------------------------- #
# Extreme-Perzentil-Flags vs. Normals (dokumentierte Vereinfachung: da unser
# Normals-Datensatz nur Mittelwert-Normals führt, keine echten Perzentile,
# nutzen wir einen konfigurierbaren Sigma-ähnlichen Schwellenwert auf Basis
# der Abweichung vom Mittelwert-Normal, nicht auf Basis einer geschätzten
# Standardabweichung -> das ist explizit KEINE echte Perzentilberechnung.)
# --------------------------------------------------------------------------- #

def extreme_flag(value: float | None, normal_value: float | None, threshold: float) -> bool | None:
    a = anomaly(value, normal_value)
    if a is None:
        return None
    return abs(a) >= threshold


# --------------------------------------------------------------------------- #
# Forecast-Revisionen — NUR für identische forecast_valid_time vergleichen
# --------------------------------------------------------------------------- #

def forecast_revision(newer: Observation, older: Observation) -> float | None:
    """newer.value - older.value, aber NUR wenn beide dieselbe entity_id,
    denselben metric und denselben forecast_valid_time beschreiben und newer
    tatsächlich später ausgegeben wurde (forecast_issue_time). Andernfalls
    None (nie einen Wert für verschiedene Gültigkeitszeitpunkte liefern)."""
    if newer.entity_id != older.entity_id or newer.metric != older.metric:
        return None
    if newer.forecast_valid_time is None or newer.forecast_valid_time != older.forecast_valid_time:
        return None
    if newer.forecast_issue_time is None or older.forecast_issue_time is None:
        return None
    if newer.forecast_issue_time <= older.forecast_issue_time:
        return None
    if newer.value is None or older.value is None:
        return None
    return newer.value - older.value


def revisions_for_metric(observations: Iterable[Observation], entity_id: str, metric: str) -> list[dict]:
    """Alle Revisionen für (entity_id, metric): pro forecast_valid_time wird
    die vorletzte gegen die letzte Ausgabe verglichen (zwei jüngste
    forecast_issue_time-Versionen). Gültigkeitszeitpunkte werden NIE
    gegeneinander verglichen."""
    by_valid_time: dict = defaultdict(list)
    for o in observations:
        if o.entity_id != entity_id or o.metric != metric or o.forecast_valid_time is None:
            continue
        by_valid_time[o.forecast_valid_time].append(o)

    out = []
    for valid_time, versions in by_valid_time.items():
        versions = [v for v in versions if v.forecast_issue_time is not None]
        versions.sort(key=lambda o: o.forecast_issue_time)
        if len(versions) < 2:
            continue
        newer, older = versions[-1], versions[-2]
        rev = forecast_revision(newer, older)
        out.append({"forecast_valid_time": valid_time, "revision": rev,
                    "newer_issue": newer.forecast_issue_time, "older_issue": older.forecast_issue_time})
    return out


# --------------------------------------------------------------------------- #
# Airline-Hub-Features
# --------------------------------------------------------------------------- #

SEVERE_ALERT_EVENTS = {
    "Winter Storm Warning", "Hurricane Warning", "Storm Surge Warning",
    "Flood Warning", "Tornado Warning", "Severe Thunderstorm Warning",
    "Ice Storm Warning", "Blizzard Warning", "Excessive Heat Warning",
}


def _hub_daily_values(forecast_daily_precip_snow: dict, entity: str, d: date) -> dict:
    return (forecast_daily_precip_snow.get(entity) or {}).get(d) or {}


def _alert_counts_for_hub(alert_observations: Iterable[Observation], hub_code: str) -> dict:
    """hub_code -> {event: count} aus 'alert_count'-Observations."""
    counts: dict[str, int] = {}
    for o in alert_observations:
        if o.entity_id != hub_code or o.metric != "alert_count" or o.value is None:
            continue
        event = o.attrs.get("event")
        counts[event] = counts.get(event, 0) + int(o.value)
    return counts


def airline_hub_features(hub_codes: list[str], forecast_observations: Iterable[Observation],
                          alert_observations: Iterable[Observation], as_of: date) -> dict:
    """Baut Hub-Level und aggregierte Netzwerk-Features für eine Airline mit
    gegebenen hub_codes (siehe config/weather_exposures.yaml). Fehlt ein Hub
    komplett in den Beobachtungen, werden seine Felder None (nicht 0), und er
    zählt nicht als "exposed", aber network_hubs_exposed/-_share bleiben auf
    Basis der Hubs mit Daten berechnet (dokumentierte Vereinfachung: kein
    Unterschied zwischen "kein Alarm" und "keine Daten" bei der Share-
    Berechnung – siehe Docstring von network_exposure_share unten)."""
    forecast_observations = list(forecast_observations)
    alert_observations = list(alert_observations)
    daily_ps = daily_precip_snow_ice_wind(forecast_observations)

    hubs: dict[str, dict] = {}
    exposed_count = 0
    valid_hub_count = 0
    for hub in hub_codes:
        vals = _hub_daily_values(daily_ps, hub, as_of)
        alert_counts = _alert_counts_for_hub(alert_observations, hub)
        severe_count = sum(n for ev, n in alert_counts.items() if ev in SEVERE_ALERT_EVENTS)
        has_any_data = bool(vals) or bool(alert_counts)
        hub_feat = {
            "hub_precip_probability": vals.get("pop_max"),
            "hub_snow_probability": None if vals.get("snow_in") is None else (1.0 if vals["snow_in"] > 0 else 0.0),
            "hub_snow_amount": vals.get("snow_in"),
            "hub_freezing_precip": vals.get("ice_in"),
            "hub_wind_speed": vals.get("wind_mph_max"),
            "hub_wind_gust": vals.get("gust_mph_max"),
            "hub_thunderstorm_alert": alert_counts.get("Severe Thunderstorm Warning", 0) > 0 if alert_counts else None,
            "hub_winter_storm_alert": alert_counts.get("Winter Storm Warning", 0) > 0 if alert_counts else None,
            "hub_severe_alert_count": severe_count if alert_counts else None,
        }
        hubs[hub] = hub_feat
        if has_any_data:
            valid_hub_count += 1
            if _hub_is_exposed(hub_feat):
                exposed_count += 1

    network_hubs_exposed = exposed_count if hub_codes else None
    network_exposure_share = (exposed_count / valid_hub_count) if valid_hub_count else None
    # Gleiche Basis wie network_exposure_share: nur Hubs mit tatsächlichen
    # Daten fliessen in den Index ein (kein Data-Gap als "kein Exposure").
    disruption_index = network_exposure_share
    return {
        "hubs": hubs,
        "network_hubs_exposed": network_hubs_exposed,
        "network_exposure_share": network_exposure_share,
        "weather_disruption_hub_index": disruption_index,
    }


def _hub_is_exposed(hub_feat: dict) -> bool:
    """Einfache, dokumentierte (nicht optimierte) Exposure-Regel: irgendein
    aktiver Warntyp ODER Schnee/Vereisung > 0 ODER PoP >= 0.5."""
    checks = [
        hub_feat.get("hub_thunderstorm_alert") is True,
        hub_feat.get("hub_winter_storm_alert") is True,
        (hub_feat.get("hub_severe_alert_count") or 0) > 0,
        (hub_feat.get("hub_snow_amount") or 0) > 0,
        (hub_feat.get("hub_freezing_precip") or 0) > 0,
        (hub_feat.get("hub_precip_probability") or 0) >= 50,
    ]
    return any(checks)


# --------------------------------------------------------------------------- #
# Utilities-Features
# --------------------------------------------------------------------------- #

def utility_features(entity: str, forecast_observations: Iterable[Observation],
                      normals_observations: Iterable[Observation], as_of: date,
                      electric_vs_gas: str | None = None) -> dict:
    """electric_vs_gas: config-getriebenes Feld (z.B. 'electric', 'gas',
    'both'), NICHT hier entschieden — Aufrufer reicht es aus
    config/weather_exposures.yaml o.ä. durch; None wenn unbekannt."""
    temp_daily = daily_temperature_aggregates(forecast_observations)
    normals = normals_by_month_day(normals_observations)
    day_vals = (temp_daily.get(entity) or {}).get(as_of) or {}
    anomalies = anomalies_for_day(temp_daily, entity, as_of, normals)

    hdd_by_date = {d: v["hdd"] for d, v in (temp_daily.get(entity) or {}).items() if v.get("hdd") is not None}
    cdd_by_date = {d: v["cdd"] for d, v in (temp_daily.get(entity) or {}).items() if v.get("cdd") is not None}

    return {
        "hdd": day_vals.get("hdd"),
        "cdd": day_vals.get("cdd"),
        "hdd_anomaly": anomalies.get("hdd_anomaly"),
        "cdd_anomaly": anomalies.get("cdd_anomaly"),
        "hdd_1d": rolling_aggregate(hdd_by_date, as_of, 1, "sum"),
        "hdd_7d": rolling_aggregate(hdd_by_date, as_of, 7, "sum"),
        "cdd_1d": rolling_aggregate(cdd_by_date, as_of, 1, "sum"),
        "cdd_7d": rolling_aggregate(cdd_by_date, as_of, 7, "sum"),
        "electric_vs_gas_exposure": electric_vs_gas,
    }


# --------------------------------------------------------------------------- #
# P&C-Insurance-Features
# --------------------------------------------------------------------------- #

WARNING_METRICS = {
    "hurricane_warning_exposure": "Hurricane Warning",
    "storm_surge_warning_exposure": "Storm Surge Warning",
    "flood_warning_exposure": "Flood Warning",
    "tornado_warning_exposure": "Tornado Warning",
    "severe_warning_exposure": "Severe Thunderstorm Warning",
}


def pc_insurance_features(exposure_region_codes: list[str],
                           alert_observations: Iterable[Observation],
                           storm_observations: Iterable[Observation]) -> dict:
    alert_observations = list(alert_observations)
    storm_observations = list(storm_observations)

    active_storm_ids = sorted({o.entity_id for o in storm_observations if o.dataset == "active_storms"})
    active_tropical_system = bool(active_storm_ids) if storm_observations or active_storm_ids else None
    if not storm_observations:
        active_tropical_system = None  # kein Abruf -> unbekannt, nicht "False"

    min_distance_km = None
    dist_obs = [o for o in storm_observations if o.metric == "min_distance_to_exposure_km" and o.value is not None]
    if dist_obs:
        min_distance_km = min(o.value for o in dist_obs)

    wind_probability_34kt = None  # nicht implementiert (siehe Registry-Notes) -> immer None, dokumentiert
    wind_probability_50kt = None
    wind_probability_64kt = None

    out = {
        "active_tropical_system": active_tropical_system,
        "min_distance_to_track_km": min_distance_km,
        "wind_probability_34kt": wind_probability_34kt,
        "wind_probability_50kt": wind_probability_50kt,
        "wind_probability_64kt": wind_probability_64kt,
    }

    affected_regions: set[str] = set()
    for feature_name, event in WARNING_METRICS.items():
        region_hit = False
        count = 0
        for region in exposure_region_codes:
            has = any(o.entity_id == region and o.metric == "alert_count" and o.attrs.get("event") == event
                      and (o.value or 0) > 0 for o in alert_observations)
            if has:
                count += 1
                region_hit = True
                affected_regions.add(region)
        if any(o.entity_id in exposure_region_codes for o in alert_observations):
            out[feature_name] = count
        else:
            out[feature_name] = None

    out["affected_exposure_region_count"] = len(affected_regions) if alert_observations else None
    return out


# --------------------------------------------------------------------------- #
# Homebuilder / Construction-Features
# --------------------------------------------------------------------------- #

HEAVY_PRECIP_IN = 1.0     # in/Tag
HEAVY_SNOW_IN = 2.0       # in/Tag
FREEZE_TMIN_F = 32.0
EXTREME_HEAT_TMAX_F = 100.0


def construction_disruption_days(entity: str, forecast_observations: Iterable[Observation],
                                  as_of: date, window_days: int) -> int | None:
    """Zählt Tage in [as_of-window_days+1, as_of] mit Starkregen, Starkschnee,
    Frost (Tmin<=32°F) oder Extremhitze (Tmax>=100°F). None, wenn für kein
    einziges dieser Tage überhaupt Daten vorliegen."""
    temp_daily = daily_temperature_aggregates(forecast_observations)
    precip_daily = daily_precip_snow_ice_wind(forecast_observations)
    t_by_date = temp_daily.get(entity) or {}
    p_by_date = precip_daily.get(entity) or {}

    n_days_with_data = 0
    disruption_days = 0
    for i in range(window_days):
        d = as_of - timedelta(days=i)
        t = t_by_date.get(d)
        p = p_by_date.get(d)
        if t is None and p is None:
            continue
        n_days_with_data += 1
        hit = False
        if p and p.get("precip_in") is not None and p["precip_in"] >= HEAVY_PRECIP_IN:
            hit = True
        if p and p.get("snow_in") is not None and p["snow_in"] >= HEAVY_SNOW_IN:
            hit = True
        if t and t.get("tmin") is not None and t["tmin"] <= FREEZE_TMIN_F:
            hit = True
        if t and t.get("tmax") is not None and t["tmax"] >= EXTREME_HEAT_TMAX_F:
            hit = True
        if hit:
            disruption_days += 1
    return disruption_days if n_days_with_data else None


# --------------------------------------------------------------------------- #
# Retail-Features
# --------------------------------------------------------------------------- #

def retail_features(region_weights: dict[str, float], forecast_observations: Iterable[Observation],
                     normals_observations: Iterable[Observation], alert_observations: Iterable[Observation],
                     storm_observations: Iterable[Observation], as_of: date) -> dict:
    """region_weights: {region_code: population_weight} aus
    config/weather_locations.yaml (population_regions[].population_weight)."""
    temp_daily = daily_temperature_aggregates(forecast_observations)
    normals = normals_by_month_day(normals_observations)

    region_anomalies: dict[str, float] = {}
    for region in region_weights:
        day_vals = (temp_daily.get(region) or {}).get(as_of) or {}
        tmean = day_vals.get("tmean")
        md = (as_of.month, as_of.day)
        normal = ((normals.get(region) or {}).get("DLY-TAVG-NORMAL") or {}).get(md)
        a = anomaly(tmean, normal)
        if a is not None:
            region_anomalies[region] = a

    if region_anomalies:
        total_w = sum(region_weights[r] for r in region_anomalies)
        weighted_anomaly = (sum(region_anomalies[r] * region_weights[r] for r in region_anomalies) / total_w
                             if total_w else None)
        dispersion = statistics.pstdev(region_anomalies.values()) if len(region_anomalies) > 1 else 0.0
    else:
        weighted_anomaly = None
        dispersion = None

    alert_observations = list(alert_observations)
    snow_exposure = any(
        o.entity_id in region_weights and o.metric == "alert_count" and o.attrs.get("event") == "Winter Storm Warning"
        and (o.value or 0) > 0 for o in alert_observations) if alert_observations else None
    heat_exposure = any(
        o.entity_id in region_weights and o.metric == "alert_count" and o.attrs.get("event") == "Excessive Heat Warning"
        and (o.value or 0) > 0 for o in alert_observations) if alert_observations else None
    severe_exposure = any(
        o.entity_id in region_weights and o.metric == "alert_count"
        and o.attrs.get("event") == "Severe Thunderstorm Warning" and (o.value or 0) > 0
        for o in alert_observations) if alert_observations else None

    storm_observations = list(storm_observations)
    hurricane_exposure = bool(storm_observations) if storm_observations else None

    return {
        "population_weighted_temperature_anomaly": weighted_anomaly,
        "population_weighted_temperature_anomaly_dispersion": dispersion,
        "snow_exposure": snow_exposure,
        "heat_exposure": heat_exposure,
        "severe_exposure": severe_exposure,
        "hurricane_exposure": hurricane_exposure,
    }


# --------------------------------------------------------------------------- #
# Logistics-Features
# --------------------------------------------------------------------------- #

def logistics_features(hub_codes: list[str], forecast_observations: Iterable[Observation],
                        alert_observations: Iterable[Observation], as_of: date) -> dict:
    """Wiederverwendet die Hub-Logik aus airline_hub_features (gleiche
    Signal-Quellen: PoP/Schnee/Eis/Wind/Alerts je Hub/Corridor-Location)."""
    base = airline_hub_features(hub_codes, forecast_observations, alert_observations, as_of)
    return {
        "hubs": base["hubs"],
        "hubs_exposed": base["network_hubs_exposed"],
        "hub_exposure_share": base["network_exposure_share"],
        "logistics_weather_disruption": base["weather_disruption_hub_index"],
    }

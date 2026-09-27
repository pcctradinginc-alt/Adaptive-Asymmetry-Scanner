"""
modules/external/sources/weather.py – Weather-Konnektoren (family: weather).

Quellen (registriert in config/external_sources/weather.yaml):
  nws_forecast      – api.weather.gov /points -> forecastGridData (quantitativ)
  nws_alerts        – api.weather.gov /alerts/active (Warnungen)
  ncei_normals      – NCEI 1991-2020 Daily Normals (statische Klimatologie)
  nhc_storms        – NHC CurrentStorms.json (aktive Tropenzyklone)
  noaa_storm_events – NCEI Storm Events Bulk-CSV (nur Backfill, default AUS)
  ecmwf_open_data   – nur Registry-Eintrag, DEFERRED (kein Connector-Code)

Grundregeln (siehe modules/external/pit.py, modules/external/sources/base.py):
  - Keine Grid-Koordinaten/Stations-IDs hartcodieren: NWS-Grid wird über
    /points/{lat},{lon} pro Location aufgelöst; NCEI-Stationen über die
    Stationssuche relativ zur konfigurierten Location.
  - available_at = retrieved_at, WEIL wir für unseren eigenen Abruf keine
    frühere Verfügbarkeit belegen können, selbst wenn ein offizieller
    Release-/Update-Zeitstempel (source_release_time) existiert, der zeitlich
    davor liegt.
  - Fehlende Werte -> value=None (nie 0 als Ersatz).
  - Alle Netzwerkzugriffe laufen über modules.external.http.fetch (Retries/
    Backoff/AuthError zentral); Connectoren werfen nie in die Pipeline hinein
    (Fehler -> ConnectorResult.status, siehe base.py).
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

from modules.external import http
from modules.external.pit import AvailabilityPrecision, Observation, ensure_utc, utc_now
from modules.external.sources.base import Connector, ConnectorResult, RawRecord, SourceStatus

REPO_ROOT = Path(__file__).resolve().parents[3]

PARSER_VERSION = "1"

DEFAULT_USER_AGENT_CONTACT = "research@adaptive-asymmetry-scanner.local"  # VERIFY: siehe README, kein offizieller Kontakt dort hinterlegt


def configured_user_agent_contact(source_cfg: dict | None = None) -> str:
    """Kontakt für den erforderlichen NWS-User-Agent-Header:
    config.yaml external_context.weather.user_agent_contact > per-Source
    'user_agent_contact' im Registry-Eintrag > DEFAULT_USER_AGENT_CONTACT.
    VERIFY vor Produktion: README enthält aktuell keinen offiziellen Kontakt,
    der Default hier ist ein Platzhalter."""
    # Vorrang: Umgebungsvariable NWS_USER_AGENT_CONTACT (GitHub-Secret), damit
    # keine Kontaktadresse im Repo stehen muss.
    import os
    env_contact = os.environ.get("NWS_USER_AGENT_CONTACT", "").strip()
    if env_contact:
        return env_contact
    try:
        from modules.config import cfg
        weather_cfg = getattr(getattr(cfg, "external_context", None), "weather", None)
        contact = getattr(weather_cfg, "user_agent_contact", None) if weather_cfg else None
        if contact:
            return contact
    except Exception:
        pass
    if source_cfg and source_cfg.get("user_agent_contact"):
        return source_cfg["user_agent_contact"]
    return DEFAULT_USER_AGENT_CONTACT


def user_agent_header(source_cfg: dict | None = None) -> str:
    return f"AdaptiveAsymmetryScanner/1.0 ({configured_user_agent_contact(source_cfg)})"

DEFAULT_GRID_ELEMENTS = [
    "temperature", "maxTemperature", "minTemperature",
    "probabilityOfPrecipitation", "quantitativePrecipitation",
    "snowfallAmount", "iceAccumulation", "windSpeed", "windGust",
]

DEFAULT_ALERT_EVENT_TYPES = [
    "Winter Storm Warning", "Hurricane Warning", "Storm Surge Warning",
    "Flood Warning", "Tornado Warning", "Severe Thunderstorm Warning",
    "Ice Storm Warning", "Blizzard Warning", "Excessive Heat Warning",
]

HDD_CDD_BASE_F = 65.0  # NCEI-Konvention: Base 65°F für Heating/Cooling Degree Days

DEFAULT_FORECAST_DAYS = 7  # nur Tage 0..N-1 relativ zum lokalen Issue-Tag archivieren

DAILY_METRIC_UNITS = {
    "tmax": "degF", "tmin": "degF", "tmean": "degF",
    "hdd": "degF-day", "cdd": "degF-day",
    "precip_total": "in", "snow_total": "in", "ice_total": "in",
    "pop_max": "%", "wind_max": "mph", "gust_max": "mph",
}


# --------------------------------------------------------------------------- #
# YAML-Config-Loader (gecached auf Modulebene, damit Connectoren sie nicht
# bei jedem fetch() neu von Disk lesen müssen)
# --------------------------------------------------------------------------- #

_CONFIG_CACHE: dict[str, Any] = {}


def _load_yaml(rel_path: str) -> dict:
    if rel_path not in _CONFIG_CACHE:
        p = REPO_ROOT / rel_path
        with open(p, "r", encoding="utf-8") as f:
            _CONFIG_CACHE[rel_path] = yaml.safe_load(f) or {}
    return _CONFIG_CACHE[rel_path]


def load_weather_locations(rel_path: str = "config/weather_locations.yaml") -> dict:
    return _load_yaml(rel_path)


def all_locations(locations_cfg: dict | None = None) -> list[dict]:
    """Flache Liste aller kuratierten Locations über alle Kategorien hinweg,
    jede mit ihrer Kategorie annotiert."""
    cfg = locations_cfg if locations_cfg is not None else load_weather_locations()
    out: list[dict] = []
    for category, entries in cfg.items():
        if not isinstance(entries, list):
            continue
        for e in entries:
            item = dict(e)
            item["category"] = category
            out.append(item)
    return out


def location_by_code(code: str, locations_cfg: dict | None = None) -> dict | None:
    for loc in all_locations(locations_cfg):
        if loc.get("code") == code:
            return loc
    return None


# --------------------------------------------------------------------------- #
# ISO8601-Dauer / validTime-Parsing (NWS forecastGridData)
# --------------------------------------------------------------------------- #

_ISO_DURATION_RE = re.compile(
    r"^P(?:(?P<weeks>\d+)W)?(?:(?P<days>\d+)D)?"
    r"(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+(?:\.\d+)?)S)?)?$"
)


def parse_iso8601_duration(duration: str) -> timedelta:
    """Parst eine ISO8601-Dauer (z.B. 'P1D', 'PT6H', 'P7DT18H') in ein
    timedelta. Unterstützt Wochen/Tage/Stunden/Minuten/Sekunden (kein
    Jahr/Monat, da NWS-Grids diese nicht verwenden)."""
    m = _ISO_DURATION_RE.match(duration.strip())
    if not m or duration.strip() in ("P", ""):
        raise ValueError(f"Ungültige ISO8601-Dauer: {duration!r}")
    parts = m.groupdict()
    weeks = int(parts["weeks"] or 0)
    days = int(parts["days"] or 0)
    hours = int(parts["hours"] or 0)
    minutes = int(parts["minutes"] or 0)
    seconds = float(parts["seconds"] or 0)
    return timedelta(weeks=weeks, days=days, hours=hours, minutes=minutes, seconds=seconds)


def parse_valid_time(valid_time: str) -> tuple[datetime, timedelta]:
    """NWS validTime-Format: '<ISO8601-Start>/<ISO8601-Dauer>'.
    Gibt (Intervall-Start UTC, Dauer) zurück. forecast_valid_time laut
    Aufgabenstellung = Intervall-Start."""
    start_str, _, dur_str = valid_time.partition("/")
    start = ensure_utc(start_str)
    if start is None:
        raise ValueError(f"Ungültiger validTime-Start: {valid_time!r}")
    duration = parse_iso8601_duration(dur_str) if dur_str else timedelta(0)
    return start, duration


# --------------------------------------------------------------------------- #
# Einheiten-Konvertierung (NWS liefert wmoUnit:* / SI; wir normalisieren auf
# US-übliche Einheiten, da NCEI-Normals ebenfalls in °F/HDD/CDD/in vorliegen
# und unsere Anomalie-Berechnung sonst Einheiten mischen würde)
# --------------------------------------------------------------------------- #

def convert_grid_value(value: float | None, uom: str) -> tuple[float | None, str]:
    if value is None:
        return None, _target_unit_for_uom(uom)
    uom = (uom or "").replace("wmoUnit:", "")
    if uom == "degC":
        return value * 9.0 / 5.0 + 32.0, "degF"
    if uom in ("mm", "mm/h"):
        return value / 25.4, "in"
    if uom == "m":
        return value * 39.3700787, "in"
    if uom in ("km_h-1", "km/h"):
        return value * 0.62137119, "mph"
    if uom in ("percent", "%"):
        return value, "%"
    return value, uom


def _target_unit_for_uom(uom: str) -> str:
    uom = (uom or "").replace("wmoUnit:", "")
    return {
        "degC": "degF", "mm": "in", "mm/h": "in", "m": "in",
        "km_h-1": "mph", "km/h": "mph", "percent": "%", "%": "%",
    }.get(uom, uom)


# --------------------------------------------------------------------------- #
# Geo-Hilfsfunktionen
# --------------------------------------------------------------------------- #

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return haversine_km(lat1, lon1, lat2, lon2) * 0.62137119


# --------------------------------------------------------------------------- #
# nws_forecast – Pure Parse-Logik (testbar ohne HTTP)
# --------------------------------------------------------------------------- #

def parse_grid_response(points_json: dict, grid_json: dict, location: dict,
                         elements: list[str], retrieved_at: datetime,
                         source_id: str = "nws_forecast",
                         parser_version: str = PARSER_VERSION) -> list[Observation]:
    """Baut Observations aus /points-Antwort + forecastGridData-Antwort für
    EINE Location. updateTime der Grid-Antwort = forecast_issue_time =
    source_release_time (offizieller Zeitstempel -> EXACT_TIMESTAMP)."""
    props = grid_json.get("properties", {})
    update_time = ensure_utc(props.get("updateTime"))
    if update_time is None:
        raise ValueError("forecastGridData ohne updateTime")

    point_props = points_json.get("properties", {})
    grid_id = point_props.get("gridId")
    grid_x = point_props.get("gridX")
    grid_y = point_props.get("gridY")
    series_id = f"{grid_id}/{grid_x},{grid_y}"

    retrieved_at = ensure_utc(retrieved_at) or utc_now()
    observations: list[Observation] = []
    for element in elements:
        block = props.get(element)
        if not block:
            continue
        uom = block.get("uom", "")
        for entry in block.get("values", []):
            valid_time_raw = entry.get("validTime")
            if not valid_time_raw:
                continue
            start, duration = parse_valid_time(valid_time_raw)
            raw_value = entry.get("value")
            value, unit = convert_grid_value(raw_value, uom)
            observations.append(Observation(
                source_id=source_id,
                dataset="grid_forecast",
                series_id=series_id,
                entity_id=location["code"],
                metric=element,
                value=value,
                unit=unit,
                observation_time=start,
                available_at=retrieved_at,
                retrieved_at=retrieved_at,
                availability_precision=AvailabilityPrecision.EXACT_TIMESTAMP,
                parser_version=parser_version,
                source_release_time=update_time,
                forecast_issue_time=update_time,
                forecast_valid_time=start,
                attrs={
                    "location_code": location["code"],
                    "lat": location.get("lat"), "lon": location.get("lon"),
                    "gridId": grid_id, "gridX": grid_x, "gridY": grid_y,
                    "duration_iso": valid_time_raw.partition("/")[2],
                    "duration_seconds": duration.total_seconds(),
                    "raw_uom": uom,
                },
            ))
    return observations


def _resolve_location_timezone(points_json: dict):
    """points_json.properties.timeZone (IANA-Name, z.B. 'America/New_York')
    -> zoneinfo.ZoneInfo, sonst None (Aufrufer fällt dann auf UTC zurück)."""
    tz_name = (points_json.get("properties") or {}).get("timeZone")
    if not tz_name:
        return None
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(tz_name)
    except Exception:
        return None


def aggregate_daily_forecast(hourly_observations: list[Observation], points_json: dict, location: dict,
                              retrieved_at: datetime, forecast_days: int = DEFAULT_FORECAST_DAYS,
                              source_id: str = "nws_forecast",
                              parser_version: str = PARSER_VERSION) -> list[Observation]:
    """Aggregiert die stündlichen/Perioden-Observations EINES Grid-Abrufs
    (aus parse_grid_response, NUR im Speicher, wird NIE archiviert) zu
    Tageswerten: tmax, tmin, tmean, hdd, cdd, precip_total, snow_total,
    ice_total, pop_max, wind_max, gust_max — je (Location, lokaler
    Kalendertag, Issue-Zeit). Nur Tage 0..forecast_days-1 relativ zum
    lokalen Tag von forecast_issue_time (updateTime) werden archiviert.

    Kalendertag = lokaler Tag in der Timezone aus /points (properties.
    timeZone), sonst UTC (kein Rateversuch über eine hartcodierte Zone).
    identity_key() der resultierenden Observations enthält
    forecast_issue_time -> eine neue Issue-Zeit erzeugt IMMER neue Zeilen;
    Revisionen entstehen NUR beim Vergleich derselben forecast_valid_time
    (siehe weather_features.forecast_revision)."""
    if not hourly_observations:
        return []
    tz = _resolve_location_timezone(points_json)
    retrieved_at = ensure_utc(retrieved_at) or utc_now()
    update_time = hourly_observations[0].forecast_issue_time
    if update_time is None:
        return []
    series_id = hourly_observations[0].series_id
    issue_local_day = (update_time.astimezone(tz) if tz else update_time).date()

    by_day: dict[date, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for o in hourly_observations:
        if o.value is None or o.forecast_valid_time is None:
            continue
        local_dt = o.forecast_valid_time.astimezone(tz) if tz else o.forecast_valid_time
        day = local_dt.date()
        offset = (day - issue_local_day).days
        if offset < 0 or offset >= forecast_days:
            continue
        by_day[day][o.metric].append(o.value)

    observations: list[Observation] = []
    for day in sorted(by_day):
        metrics = by_day[day]
        hourly_t = metrics.get("temperature", [])
        maxt = metrics.get("maxTemperature", [])
        mint = metrics.get("minTemperature", [])
        tmax = max(maxt) if maxt else (max(hourly_t) if hourly_t else None)
        tmin = min(mint) if mint else (min(hourly_t) if hourly_t else None)
        if hourly_t:
            tmean = statistics.fmean(hourly_t)
        elif tmax is not None and tmin is not None:
            tmean = (tmax + tmin) / 2.0
        else:
            tmean = None
        hdd = max(0.0, HDD_CDD_BASE_F - tmean) if tmean is not None else None
        cdd = max(0.0, tmean - HDD_CDD_BASE_F) if tmean is not None else None
        precip = metrics.get("quantitativePrecipitation", [])
        snow = metrics.get("snowfallAmount", [])
        ice = metrics.get("iceAccumulation", [])
        pop = metrics.get("probabilityOfPrecipitation", [])
        wind = metrics.get("windSpeed", [])
        gust = metrics.get("windGust", [])

        day_start = datetime(day.year, day.month, day.day, tzinfo=tz or timezone.utc)
        values = {
            "tmax": tmax, "tmin": tmin, "tmean": tmean, "hdd": hdd, "cdd": cdd,
            "precip_total": sum(precip) if precip else None,
            "snow_total": sum(snow) if snow else None,
            "ice_total": sum(ice) if ice else None,
            "pop_max": max(pop) if pop else None,
            "wind_max": max(wind) if wind else None,
            "gust_max": max(gust) if gust else None,
        }
        for metric, value in values.items():
            if value is None:
                continue
            observations.append(Observation(
                source_id=source_id, dataset="grid_forecast_daily", series_id=series_id,
                entity_id=location["code"], metric=metric, value=value,
                unit=DAILY_METRIC_UNITS.get(metric, ""),
                observation_time=day_start, available_at=retrieved_at, retrieved_at=retrieved_at,
                availability_precision=AvailabilityPrecision.EXACT_TIMESTAMP,
                parser_version=parser_version, source_release_time=update_time,
                forecast_issue_time=update_time, forecast_valid_time=day_start,
                attrs={"location_code": location["code"]},
            ))
    return observations


def write_nws_locations_manifest(discovered: dict, locations: list[dict],
                                  archive_root: str | Path | None = None) -> Path:
    """Schreibt/aktualisiert outputs/external_data/manifests/nws_locations.json
    – statische Location-Metadaten (lat/lon/grid ids/timezone) EINMAL zentral,
    statt sie in jeder normalisierten Zeile zu wiederholen (siehe attrs=
    {"location_code"} in aggregate_daily_forecast)."""
    from modules.external.archive import DEFAULT_ROOT
    root = Path(archive_root or DEFAULT_ROOT)
    path = root / "manifests" / "nws_locations.json"
    existing: dict = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except Exception:
            existing = {}
    by_code = {l["code"]: l for l in locations}
    now_iso = utc_now().isoformat(timespec="seconds")
    for code, grid in discovered.items():
        loc = by_code.get(code, {})
        existing[code] = {
            "lat": loc.get("lat"), "lon": loc.get("lon"),
            "gridId": grid.get("gridId"), "gridX": grid.get("gridX"), "gridY": grid.get("gridY"),
            "timezone": grid.get("timezone"),
            "updated_at": now_iso,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2, sort_keys=True))
    return path


class NwsForecastConnector(Connector):
    source_id = "nws_forecast"
    parser_version = PARSER_VERSION

    def _user_agent(self) -> str:
        return user_agent_header(self.cfg)

    def _locations(self) -> list[dict]:
        codes = self.cfg.get("location_codes")
        cfg = load_weather_locations(self.cfg.get("locations_config", "config/weather_locations.yaml"))
        locs = all_locations(cfg)
        if codes:
            locs = [l for l in locs if l.get("code") in codes]
        return [l for l in locs if l.get("lat") is not None and l.get("lon") is not None]

    def _forecast_days(self) -> int:
        try:
            from modules.config import cfg
            weather_cfg = getattr(getattr(cfg, "external_context", None), "weather", None)
            v = getattr(weather_cfg, "forecast_days", None) if weather_cfg else None
            if v:
                return int(v)
        except Exception:
            pass
        return int(self.cfg.get("forecast_days", DEFAULT_FORECAST_DAYS))

    def fetch(self, now: datetime) -> ConnectorResult:
        base_url = self.cfg.get("base_url", "https://api.weather.gov")
        elements = self.cfg.get("elements", DEFAULT_GRID_ELEMENTS)
        headers = {"User-Agent": self._user_agent(), "Accept": "application/geo+json"}
        forecast_days = self._forecast_days()
        archive_root = self.cfg.get("archive_root")

        observations: list[Observation] = []
        raw: list[RawRecord] = []
        discovered: dict = {}
        failures = 0
        latest_release: datetime | None = None

        locations = self._locations()
        if not locations:
            return ConnectorResult(source_id=self.source_id, status=SourceStatus.FAIL,
                                    message="keine Locations konfiguriert")

        for loc in locations:
            try:
                points_url = f"{base_url}/points/{loc['lat']},{loc['lon']}"
                points_res = http.fetch(points_url, headers=headers)
                raw.append(_to_raw(self.source_id, "points", points_res))
                points_json = points_res.json()

                grid_url = points_json["properties"]["forecastGridData"]
                grid_res = http.fetch(grid_url, headers=headers)
                raw.append(_to_raw(self.source_id, "grid_forecast", grid_res))
                grid_json = grid_res.json()

                hourly_obs = parse_grid_response(points_json, grid_json, loc, elements,
                                                  retrieved_at=grid_res.retrieved_at,
                                                  source_id=self.source_id,
                                                  parser_version=self.parser_version)
                obs = aggregate_daily_forecast(hourly_obs, points_json, loc,
                                                retrieved_at=grid_res.retrieved_at,
                                                forecast_days=forecast_days,
                                                source_id=self.source_id,
                                                parser_version=self.parser_version)
                observations.extend(obs)
                discovered[loc["code"]] = {
                    "gridId": points_json["properties"].get("gridId"),
                    "gridX": points_json["properties"].get("gridX"),
                    "gridY": points_json["properties"].get("gridY"),
                    "timezone": points_json["properties"].get("timeZone"),
                }
                for o in obs:
                    if latest_release is None or (o.source_release_time and o.source_release_time > latest_release):
                        latest_release = o.source_release_time
            except (http.FetchError, ValueError, KeyError) as e:
                failures += 1
                continue

        if discovered:
            try:
                write_nws_locations_manifest(discovered, locations, archive_root=archive_root)
            except Exception:
                pass  # Manifest ist ein Komfort-Cache, darf den Fetch nie brechen

        if not observations and failures:
            status = SourceStatus.FAIL
        elif failures:
            status = SourceStatus.WARN
        else:
            status = SourceStatus.PASS

        return ConnectorResult(
            source_id=self.source_id, status=status, observations=observations, raw=raw,
            message=f"{failures} von {len(locations)} Locations fehlgeschlagen" if failures else "",
            latest_release_time=latest_release, discovered_ids=discovered, parse_failures=failures,
        )


def _to_raw(source_id: str, dataset: str, res: "http.FetchResult") -> RawRecord:
    return RawRecord(source_id=source_id, dataset=dataset, url=res.url, fingerprint=res.fingerprint,
                      retrieved_at=res.retrieved_at, status_code=res.status, content_type=res.content_type,
                      content_hash=res.content_hash, bytes=res.bytes)


# --------------------------------------------------------------------------- #
# nws_alerts
# --------------------------------------------------------------------------- #

def parse_alerts_response(alerts_json: dict, location_code: str, event_types: list[str],
                           retrieved_at: datetime, source_id: str = "nws_alerts",
                           parser_version: str = PARSER_VERSION) -> list[Observation]:
    retrieved_at = ensure_utc(retrieved_at) or utc_now()
    observations: list[Observation] = []
    counts: dict[str, int] = {}
    for feature in alerts_json.get("features", []):
        props = feature.get("properties", {})
        event = props.get("event")
        if event_types and event not in event_types:
            continue
        onset = ensure_utc(props.get("onset")) or ensure_utc(props.get("effective")) or ensure_utc(props.get("sent"))
        sent = ensure_utc(props.get("sent"))
        expires = ensure_utc(props.get("expires")) or ensure_utc(props.get("ends"))
        if onset is None:
            continue
        alert_id = props.get("id", "")
        ugc = (props.get("geocode") or {}).get("UGC", [])
        observations.append(Observation(
            source_id=source_id, dataset="alerts_active", series_id=alert_id,
            entity_id=location_code, metric="alert_active", value=1.0, unit="count",
            observation_time=onset, available_at=retrieved_at, retrieved_at=retrieved_at,
            availability_precision=AvailabilityPrecision.EXACT_TIMESTAMP,
            parser_version=parser_version, source_release_time=sent,
            attrs={"event": event, "severity": props.get("severity"),
                   "onset": props.get("onset"), "expires": props.get("expires") or props.get("ends"),
                   "ugc": ugc, "alert_id": alert_id, "location_code": location_code},
        ))
        counts[event] = counts.get(event, 0) + 1

    now_for_count = retrieved_at
    for event, n in counts.items():
        observations.append(Observation(
            source_id=source_id, dataset="alerts_active", series_id=f"count:{event}",
            entity_id=location_code, metric="alert_count", value=float(n), unit="count",
            observation_time=now_for_count, available_at=retrieved_at, retrieved_at=retrieved_at,
            availability_precision=AvailabilityPrecision.EXACT_TIMESTAMP,
            parser_version=parser_version, source_release_time=None,
            attrs={"event": event, "location_code": location_code},
        ))
    return observations


class NwsAlertsConnector(Connector):
    source_id = "nws_alerts"
    parser_version = PARSER_VERSION

    def _user_agent(self) -> str:
        return user_agent_header(self.cfg)

    def _locations(self) -> list[dict]:
        cfg = load_weather_locations(self.cfg.get("locations_config", "config/weather_locations.yaml"))
        return [l for l in all_locations(cfg) if l.get("lat") is not None and l.get("lon") is not None]

    def fetch(self, now: datetime) -> ConnectorResult:
        base_url = self.cfg.get("base_url", "https://api.weather.gov")
        event_types = self.cfg.get("event_types", DEFAULT_ALERT_EVENT_TYPES)
        headers = {"User-Agent": self._user_agent(), "Accept": "application/geo+json"}

        observations: list[Observation] = []
        raw: list[RawRecord] = []
        failures = 0
        locations = self._locations()
        latest_release: datetime | None = None

        for loc in locations:
            try:
                url = f"{base_url}/alerts/active"
                res = http.fetch(url, params={"point": f"{loc['lat']},{loc['lon']}"}, headers=headers)
                raw.append(_to_raw(self.source_id, "alerts_active", res))
                obs = parse_alerts_response(res.json(), loc["code"], event_types,
                                             retrieved_at=res.retrieved_at,
                                             source_id=self.source_id, parser_version=self.parser_version)
                observations.extend(obs)
                for o in obs:
                    if o.source_release_time and (latest_release is None or o.source_release_time > latest_release):
                        latest_release = o.source_release_time
            except (http.FetchError, ValueError, KeyError):
                failures += 1
                continue

        status = SourceStatus.PASS if not failures else (SourceStatus.WARN if observations else SourceStatus.FAIL)
        return ConnectorResult(source_id=self.source_id, status=status, observations=observations, raw=raw,
                                message=f"{failures} von {len(locations)} Locations fehlgeschlagen" if failures else "",
                                latest_release_time=latest_release, parse_failures=failures)


# --------------------------------------------------------------------------- #
# ncei_normals
# --------------------------------------------------------------------------- #

NORMALS_DATATYPES = [
    "DLY-TAVG-NORMAL", "DLY-TMAX-NORMAL", "DLY-TMIN-NORMAL",
    "DLY-HTDD-NORMAL", "DLY-CLDD-NORMAL", "DLY-PRCP-NORMAL",
]

_METRIC_UNIT = {
    "DLY-TAVG-NORMAL": "degF", "DLY-TMAX-NORMAL": "degF", "DLY-TMIN-NORMAL": "degF",
    "DLY-HTDD-NORMAL": "degF-day", "DLY-CLDD-NORMAL": "degF-day", "DLY-PRCP-NORMAL": "in",
}


def parse_station_search_response(search_json: dict) -> list[dict]:
    """NCEI CDO Web Services v2 /stations Antwort -> Liste {id, name, lat, lon}."""
    out = []
    for r in search_json.get("results", []):
        if r.get("latitude") is None or r.get("longitude") is None:
            continue
        out.append({"id": r.get("id"), "name": r.get("name"),
                    "lat": float(r["latitude"]), "lon": float(r["longitude"])})
    return out


def nearest_station(candidates: list[dict], lat: float, lon: float) -> dict | None:
    """Wählt die nächstgelegene Station aus einer Kandidatenliste (nie eine
    ID hartcodieren: die Auswahl erfolgt ausschließlich über Distanz zur
    konfigurierten Location-Koordinate)."""
    if not candidates:
        return None
    best, best_dist = None, None
    for c in candidates:
        d = haversine_km(lat, lon, c["lat"], c["lon"])
        if best_dist is None or d < best_dist:
            best, best_dist = c, d
    if best is not None:
        best = dict(best)
        best["distance_km"] = best_dist
    return best


def parse_normals_response(records: list[dict], station_id: str, location_code: str,
                            retrieved_at: datetime, source_id: str = "ncei_normals",
                            parser_version: str = PARSER_VERSION) -> list[Observation]:
    """access/services/data/v1 Antwort (Liste von Records mit DATE + den
    DLY-*-NORMAL-Feldern) -> Observations. observation_time = Normal-Datum
    (Platzhalterjahr aus der API, i.d.R. 2010); frequency=static, daher
    source_release_time=None (kein amtlicher Release-Zeitpunkt pro Abruf
    bekannt) und availability_precision=INFERRED."""
    retrieved_at = ensure_utc(retrieved_at) or utc_now()
    observations: list[Observation] = []
    for rec in records:
        date_raw = rec.get("DATE")
        if not date_raw:
            continue
        obs_time = ensure_utc(date_raw)
        if obs_time is None:
            continue
        for dt_code in NORMALS_DATATYPES:
            raw_val = rec.get(dt_code)
            if raw_val in (None, "", "-9999"):
                value = None
            else:
                try:
                    value = float(raw_val)
                except (TypeError, ValueError):
                    value = None
            observations.append(Observation(
                source_id=source_id, dataset="normals_daily_1991_2020", series_id=station_id,
                entity_id=location_code, metric=dt_code, value=value,
                unit=_METRIC_UNIT.get(dt_code, ""), observation_time=obs_time,
                available_at=retrieved_at, retrieved_at=retrieved_at,
                availability_precision=AvailabilityPrecision.INFERRED,
                parser_version=parser_version, source_release_time=None,
                attrs={"station_id": station_id, "location_code": location_code},
            ))
    return observations


class NceiNormalsConnector(Connector):
    source_id = "ncei_normals"
    parser_version = PARSER_VERSION

    def _locations(self) -> list[dict]:
        cfg = load_weather_locations(self.cfg.get("locations_config", "config/weather_locations.yaml"))
        return [l for l in all_locations(cfg) if l.get("lat") is not None and l.get("lon") is not None]

    def _resolve_station(self, loc: dict, headers: dict, raw: list) -> dict | None:
        override = (self.cfg.get("station_overrides") or {}).get(loc["code"])
        if override:
            # "verify in preflight" – nur zulässig, wenn wir uns der Format-
            # Gültigkeit sicher sind; wird trotzdem gegen die Suche geprüft,
            # sobald preflight() läuft (kein blindes Vertrauen im Betrieb).
            return {"id": override, "distance_km": None, "source": "override"}
        search_url = self.cfg.get("station_search_url", "https://www.ncei.noaa.gov/cdo-web/api/v2/stations")
        import os
        env_var = self.cfg.get("auth_env_variable", "NCEI_CDO_TOKEN")
        token = self.cfg.get("cdo_token") or (os.environ.get(env_var) if env_var else None)
        if not token:
            return None
        res = http.fetch(search_url, params={
            "extent": f"{loc['lat']-1},{loc['lon']-1},{loc['lat']+1},{loc['lon']+1}",
            "datasetid": "NORMAL_DLY", "limit": 25,
        }, headers={**headers, "token": token})
        raw.append(_to_raw(self.source_id, "station_search", res))
        candidates = parse_station_search_response(res.json())
        return nearest_station(candidates, loc["lat"], loc["lon"])

    def fetch(self, now: datetime) -> ConnectorResult:
        base_url = self.cfg.get("base_url", "https://www.ncei.noaa.gov/access/services/data/v1")
        headers = {"User-Agent": user_agent_header(self.cfg)}
        observations: list[Observation] = []
        raw: list[RawRecord] = []
        discovered: dict = {}
        failures = 0
        auth_missing = False

        locations = self._locations()
        for loc in locations:
            try:
                station = self._resolve_station(loc, headers, raw)
                if station is None:
                    auth_missing = True
                    continue
                discovered[loc["code"]] = station
                res = http.fetch(base_url, params={
                    "dataset": self.cfg.get("dataset", "normals-daily-1991-2020"),
                    "stations": station["id"],
                    "dataTypes": ",".join(NORMALS_DATATYPES),
                    "format": "json",
                }, headers=headers)
                raw.append(_to_raw(self.source_id, "normals_daily", res))
                records = res.json()
                if not isinstance(records, list):
                    records = []
                obs = parse_normals_response(records, station["id"], loc["code"],
                                              retrieved_at=res.retrieved_at,
                                              source_id=self.source_id, parser_version=self.parser_version)
                observations.extend(obs)
            except (http.FetchError, ValueError, KeyError):
                failures += 1
                continue

        if not observations and auth_missing and not failures:
            status = SourceStatus.AUTH_MISSING
        elif not observations and failures:
            status = SourceStatus.FAIL
        elif failures or auth_missing:
            status = SourceStatus.WARN
        else:
            status = SourceStatus.PASS

        return ConnectorResult(source_id=self.source_id, status=status, observations=observations, raw=raw,
                                message="NCEI CDO Token fehlt (NCEI_CDO_TOKEN/cdo_token) für Stationssuche" if auth_missing else "",
                                discovered_ids=discovered, parse_failures=failures)


# --------------------------------------------------------------------------- #
# nhc_storms
# --------------------------------------------------------------------------- #

class NhcSchemaError(Exception):
    """CurrentStorms.json entspricht nicht dem erwarteten Top-Level-Schema
    ({"activeStorms": [...]}) -> SCHEMA_CHANGED, nie eine Exception in die
    Pipeline werfen lassen. `diagnostics` trägt sichere Debug-Infos."""

    def __init__(self, message: str, diagnostics: dict | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


FETCH_SUMMARY_DATASET = "fetch_summary"


class NhcAdvisoryParseError(Exception):
    """Der Text eines Forecast/Advisory-Produkts (TCM, z.B. MIATCMAT#) passt
    nicht mehr auf das erwartete Zeilenformat (FORECAST/OUTLOOK VALID ...).
    Wird pro Sturm gefangen: nur DIESER Sturm verliert seinen Forecast-Track
    (die Observation bekommt attrs['advisory_status']='SCHEMA_CHANGED'),
    andere Stürme und der Rest der Pipeline laufen unbeeinflusst weiter --
    nie eine Exception aus parse_current_storms() heraus für dieses Problem."""


# CurrentStorms.json.activeStorms[i].forecastTrack ist LIVE nur ein Verweis
# auf ein GIS-Produkt (KMZ/Shapefile-ZIP), keine Punktliste -- siehe NHC-API-
# Dokumentation. Die tatsächliche Vorhersagespur kommt daher aus dem
# offiziellen Forecast/Advisory-TEXT-Produkt (TCM), dessen URL+Issuance im
# Feld 'forecastAdvisory' steht: {"advNum", "issuance", "url"}. VERIFY live:
# Feldname/-form kann sich ändern (siehe NhcAdvisoryParseError-Handling).

# Zeilen wie "FORECAST VALID 28/0000Z 17.5N 105.2W" bzw.
# "OUTLOOK VALID 30/0000Z 20.0N 110.0W" bzw.
# "EXTENDED FORECAST VALID 29/1200Z 22.0N 108.0W".
_ADVISORY_VALID_RE = re.compile(
    r"(?P<kind>EXTENDED\s+FORECAST|FORECAST|OUTLOOK)\s+VALID\s+"
    r"(?P<day>\d{2})/(?P<hour>\d{2})(?P<minute>\d{2})Z\s+"
    r"(?P<lat>\d{1,2}\.\d)(?P<lat_hemi>[NS])\s+"
    r"(?P<lon>\d{1,3}\.\d)(?P<lon_hemi>[EW])",
    re.IGNORECASE,
)

# Aktuelle Position: "INITIAL 26/2100Z 25.5N 84.0W" (Tag/Zeit VOR Position)
_ADVISORY_INIT_RE = re.compile(
    r"INIT(?:IAL)?\s+"
    r"(?P<day>\d{2})/(?P<hour>\d{2})(?P<minute>\d{2})Z\s+"
    r"(?P<lat>\d{1,2}\.\d)(?P<lat_hemi>[NS])\s+"
    r"(?P<lon>\d{1,3}\.\d)(?P<lon_hemi>[EW])",
    re.IGNORECASE,
)

# Aktuelle Position: "CENTER LOCATED NEAR 25.5N 84.0W AT 26/2100Z" (Position
# VOR Tag/Zeit; das 'AT dd/hhmmZ' ist optional in manchen Produktvarianten).
_ADVISORY_CENTER_RE = re.compile(
    r"CENTER\s+LOCATED\s+NEAR\s+"
    r"(?P<lat>\d{1,2}\.\d)(?P<lat_hemi>[NS])\s+"
    r"(?P<lon>\d{1,3}\.\d)(?P<lon_hemi>[EW])"
    r"(?:\s+AT\s+(?P<day>\d{2})/(?P<hour>\d{2})(?P<minute>\d{2})Z)?",
    re.IGNORECASE,
)

# "MAX WIND  85 KT...GUSTS 105 KT." (GUSTS-Teil optional/variable Wortform).
_ADVISORY_MAX_WIND_RE = re.compile(
    r"MAX(?:IMUM)?\s+(?:SUSTAINED\s+)?WIND[S]?\s+(?P<wind>\d{2,3})\s*KT"
    r"(?:\s*\.{0,3}\s*GUSTS?(?:\s+TO)?\s*(?P<gust>\d{2,3})\s*KT)?",
    re.IGNORECASE,
)

# Windradien-Zeile: "64 KT... 30NE  20SE  15SW  25NW."
_ADVISORY_WIND_RADII_RE = re.compile(
    r"^\s*(?P<thresh>\d{2,3})\s*KT\.{0,3}\s*"
    r"(?P<ne>\d{1,3})NE\s+(?P<se>\d{1,3})SE\s+(?P<sw>\d{1,3})SW\s+(?P<nw>\d{1,3})NW",
    re.IGNORECASE | re.MULTILINE,
)


def _dd_hemi_to_signed(num_str: str, hemi: str) -> float:
    val = float(num_str)
    return -val if hemi.upper() in ("S", "W") else val


def _advisory_valid_time(day: int, hour: int, minute: int, issuance: datetime) -> datetime:
    """DD/HHMMZ -> UTC-datetime relativ zur Advisory-Issuance. Monats-
    (und ggf. Jahres-)Übergang: TCM-Vorhersagehorizonte sind <= 5 Tage, daher
    heißt ein Vorhersage-Tag < Issuance-Tag zuverlässig 'nächster Monat'
    (z.B. Issuance 30. Sep, VALID 01/... -> 1. Okt). VERIFY: reine
    Heuristik, kein expliziter Monats-/Jahresfeld im TCM-Text vorhanden."""
    issuance = ensure_utc(issuance)
    year, month = issuance.year, issuance.month
    if day < issuance.day:
        month += 1
        if month > 12:
            month = 1
            year += 1
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


def _extract_advisory_wind_radii(block: str) -> dict[str, dict[str, float]]:
    radii: dict[str, dict[str, float]] = {}
    for m in _ADVISORY_WIND_RADII_RE.finditer(block):
        radii[m.group("thresh")] = {
            "NE": float(m.group("ne")), "SE": float(m.group("se")),
            "SW": float(m.group("sw")), "NW": float(m.group("nw")),
        }
    return radii


def _extract_advisory_max_wind(block: str) -> tuple[float | None, float | None]:
    m = _ADVISORY_MAX_WIND_RE.search(block)
    if not m:
        return None, None
    gust = float(m.group("gust")) if m.group("gust") else None
    return float(m.group("wind")), gust


def _extract_advisory_pre_text(raw_text: str) -> str:
    """Der Forecast/Advisory-Text wird meist als HTML-Seite mit einem
    <pre>-Block ausgeliefert (z.B. nhc.noaa.gov/text/...shtml); ist bereits
    reiner Text (kein <pre>-Tag gefunden, z.B. in Tests), wird er
    unverändert zurückgegeben."""
    m = re.search(r"<pre[^>]*>(.*?)</pre>", raw_text, re.IGNORECASE | re.DOTALL)
    body = m.group(1) if m else raw_text
    import html as _html
    return _html.unescape(body)


def parse_forecast_advisory_points(text: str, issuance: datetime) -> list[dict]:
    """Reine Text-Parse-Logik (kein HTML/HTTP): zerlegt den TCM-Volltext in
    Absätze (leerzeilengetrennt) und extrahiert je Absatz höchstens einen
    Track-Punkt (FORECAST/OUTLOOK/EXTENDED FORECAST VALID, oder die
    aktuelle Position über INITIAL/CENTER LOCATED NEAR). Nie ein KeyError/
    IndexError nach außen: unlesbarer Text (kein einziger erkannter Punkt)
    -> NhcAdvisoryParseError (Aufrufer entscheidet SCHEMA_CHANGED-Handling)."""
    issuance = ensure_utc(issuance)
    if issuance is None:
        raise NhcAdvisoryParseError("issuance fehlt oder ist ungültig")

    points: list[dict] = []
    for block in re.split(r"\n\s*\n", text or ""):
        m = _ADVISORY_VALID_RE.search(block)
        if m:
            kind_raw = m.group("kind").upper()
            kind = "outlook" if kind_raw == "OUTLOOK" else "forecast"
        else:
            m = _ADVISORY_INIT_RE.search(block) or _ADVISORY_CENTER_RE.search(block)
            kind = "initial"
            if m is None or m.group("day") is None:
                continue  # kein Track-Punkt in diesem Absatz -- kein Fehler
        try:
            lat = _dd_hemi_to_signed(m.group("lat"), m.group("lat_hemi"))
            lon = _dd_hemi_to_signed(m.group("lon"), m.group("lon_hemi"))
            valid_time = _advisory_valid_time(
                int(m.group("day")), int(m.group("hour")), int(m.group("minute")), issuance)
        except (TypeError, ValueError):
            continue
        wind, gust = _extract_advisory_max_wind(block)
        points.append({
            "kind": kind, "valid_time": valid_time, "lat": lat, "lon": lon,
            "max_wind_kt": wind, "gust_kt": gust,
            "wind_radii_nm": _extract_advisory_wind_radii(block),
        })

    if not points:
        raise NhcAdvisoryParseError(
            "keine FORECAST/OUTLOOK/INITIAL-Zeilen im Advisory-Text gefunden "
            "(Textformat hat sich vermutlich geändert)"
        )
    return points


def parse_forecast_advisory_text(raw_text: str, storm_id: str, issuance: datetime,
                                  adv_num: str | None, retrieved_at: datetime,
                                  source_id: str = "nhc_storms",
                                  parser_version: str = PARSER_VERSION) -> list[Observation]:
    """HTML-oder-Text-Forecast/Advisory (TCM) -> Observations (track_lat,
    track_lon, track_max_wind_kt) je Track-Punkt (Issuance-Zeitpunkt =
    forecast_issue_time, geparste VALID-Zeit = forecast_valid_time,
    EXACT_TIMESTAMP, attrs={'kind': 'forecast'|'outlook'|'initial',
    'advisory': adv_num}). Wirft NhcAdvisoryParseError bei unlesbarem Text
    (Aufrufer fängt das pro Sturm ab, siehe parse_current_storms)."""
    body = _extract_advisory_pre_text(raw_text)
    points = parse_forecast_advisory_points(body, issuance)
    return _advisory_points_to_observations(
        points, storm_id, issuance, adv_num, retrieved_at,
        source_id=source_id, parser_version=parser_version,
    )


def _advisory_points_to_observations(points: list[dict], storm_id: str, issuance: datetime,
                                      adv_num: str | None, retrieved_at: datetime,
                                      source_id: str, parser_version: str) -> list[Observation]:
    retrieved_at = ensure_utc(retrieved_at) or utc_now()
    issuance = ensure_utc(issuance)
    observations: list[Observation] = []
    for pt in points:
        attrs = {"kind": pt["kind"], "advisory": adv_num}
        if pt.get("gust_kt") is not None:
            attrs["gust_kt"] = pt["gust_kt"]
        if pt.get("wind_radii_nm"):
            attrs["wind_radii_nm"] = pt["wind_radii_nm"]
        common = dict(
            source_id=source_id, dataset="forecast_track", series_id=storm_id,
            entity_id=storm_id, observation_time=pt["valid_time"],
            available_at=retrieved_at, retrieved_at=retrieved_at,
            availability_precision=AvailabilityPrecision.EXACT_TIMESTAMP,
            parser_version=parser_version, source_release_time=issuance,
            forecast_issue_time=issuance, forecast_valid_time=pt["valid_time"],
        )
        if pt.get("lat") is not None:
            observations.append(Observation(metric="track_lat", value=pt["lat"], unit="deg",
                                             attrs=dict(attrs), **common))
        if pt.get("lon") is not None:
            observations.append(Observation(metric="track_lon", value=pt["lon"], unit="deg",
                                             attrs=dict(attrs), **common))
        if pt.get("max_wind_kt") is not None:
            observations.append(Observation(metric="track_max_wind_kt", value=pt["max_wind_kt"],
                                             unit="kt", attrs=dict(attrs), **common))
    return observations


def parse_current_storms(storms_json: dict, retrieved_at: datetime,
                          exposure_regions: list[dict] | None = None,
                          advisory_texts: dict[str, str] | None = None,
                          source_id: str = "nhc_storms",
                          parser_version: str = PARSER_VERSION) -> list[Observation]:
    """CurrentStorms.json -> Observations je aktivem Sturm.

    `advisory_texts` (optional): {storm_id: roher HTML/Text-Body des
    Forecast/Advisory-Produkts (TCM), bereits abgerufen -- I/O passiert NUR
    im Connector, diese Funktion bleibt pur/testbar}. Wenn für einen Sturm
    sowohl ein 'forecastAdvisory'-Objekt ({advNum, issuance, url}) im JSON
    ALS AUCH ein Eintrag in advisory_texts vorliegt, wird daraus der echte
    Forecast-Track (FORECAST/OUTLOOK VALID-Punkte, siehe
    parse_forecast_advisory_text) geparst und archiviert; min_track_distance_km
    + hours_until_closest_approach werden über Track-Punkte + aktuelle
    Position berechnet. min_distance_to_exposure_km bleibt (unverändert)
    die Distanz NUR der aktuellen Sturmposition.

    CurrentStorms.json.forecastTrack selbst ist LIVE nur ein GIS-Verweis
    (KMZ/ZIP), keine Punktliste -- wird hier nicht mehr verwendet.

    Robust gegenüber unerwarteten Formen: ein nicht-dict Top-Level-Objekt
    oder ein 'activeStorms', das keine Liste ist, führt zu NhcSchemaError
    (SCHEMA_CHANGED) statt einer AttributeError/TypeError in der Pipeline.
    Einzelne Nicht-dict-Einträge in activeStorms (z.B. rohe ID-Strings statt
    Objekten) werden übersprungen, nicht als Crash behandelt -- eine leere
    activeStorms-Liste liefert weiterhin PASS mit 0 Observations. Ein
    unlesbarer Advisory-Text bricht NUR den Track dieses EINEN Sturms
    (attrs['advisory_status']='SCHEMA_CHANGED' auf den Track-Distanz-
    Observations), nie die ganze Funktion."""
    if not isinstance(storms_json, dict):
        raise NhcSchemaError(
            f"CurrentStorms.json: Top-Level ist kein Objekt, sondern {type(storms_json).__name__}.",
            diagnostics={"body_snippet": str(storms_json)[:600]},
        )
    active_storms = storms_json.get("activeStorms", [])
    if active_storms is None:
        active_storms = []
    if not isinstance(active_storms, list):
        try:
            body_snippet = json.dumps(storms_json)[:600]
        except (TypeError, ValueError):
            body_snippet = str(storms_json)[:600]
        raise NhcSchemaError(
            f"CurrentStorms.json: 'activeStorms' ist kein Array, sondern {type(active_storms).__name__}.",
            diagnostics={"body_snippet": body_snippet},
        )

    retrieved_at = ensure_utc(retrieved_at) or utc_now()
    observations: list[Observation] = []
    exposure_regions = exposure_regions or []
    advisory_texts = advisory_texts or {}

    for storm in active_storms:
        if not isinstance(storm, dict):
            # z.B. eine rohe ID-Zeichenkette statt eines Sturm-Objekts --
            # überspringen statt mit AttributeError zu crashen.
            continue
        storm_id = storm.get("id") or storm.get("binNumber")
        if not storm_id:
            continue
        advisory = storm.get("publicAdvisory")
        if not isinstance(advisory, dict):
            advisory = {}   # offizielles Schema: Objekt {advNum, issuance, url}; alles andere ignorieren
        try:
            issuance = ensure_utc(advisory.get("issuance")) or ensure_utc(storm.get("lastUpdate"))
        except (TypeError, ValueError):
            issuance = None
        if issuance is None:
            continue

        def _num(x):
            if x is None:
                return None
            try:
                return float(re.sub(r"[^0-9.\-]", "", str(x)))
            except ValueError:
                return None

        # Offizielles Schema liefert latitudeNumeric/longitudeNumeric (signiert)
        # sowie latitude/longitude als Text ("25.1N"); ältere Fixtures lat/lon.
        lat = _num(storm.get("latitudeNumeric"))
        lon = _num(storm.get("longitudeNumeric"))
        if lat is None:
            lat = _signed_latlon(storm.get("lat") or storm.get("latitude"))
        if lon is None:
            lon = _signed_latlon(storm.get("lon") or storm.get("longitude"))

        fields = {
            "intensity_kt": _num(storm.get("intensity")),
            "pressure_mb": _num(storm.get("pressure")),
            "lat": lat, "lon": lon,
            "movement_speed_kt": _num(storm.get("movementSpeed")),
        }
        for metric, value in fields.items():
            observations.append(Observation(
                source_id=source_id, dataset="active_storms", series_id=storm_id,
                entity_id=storm_id, metric=metric, value=value,
                unit="kt" if "kt" in metric else ("mb" if metric == "pressure_mb" else "deg"),
                observation_time=issuance, available_at=retrieved_at, retrieved_at=retrieved_at,
                availability_precision=AvailabilityPrecision.EXACT_TIMESTAMP,
                parser_version=parser_version, source_release_time=issuance,
                forecast_issue_time=issuance,
                attrs={"name": storm.get("name"), "classification": storm.get("classification")},
            ))

        wsp = storm.get("windSpeedProbabilities")

        # ── aktuelle Position -> nächste Expositionsregion (unverändert) ────
        current_min_distance_km = None
        current_nearest_region = None
        if lat is not None and lon is not None:
            for region in exposure_regions:
                d = haversine_km(lat, lon, region["lat"], region["lon"])
                if current_min_distance_km is None or d < current_min_distance_km:
                    current_min_distance_km = d
                    current_nearest_region = region.get("code")

        observations.append(Observation(
            source_id=source_id, dataset="active_storms", series_id=storm_id,
            entity_id=storm_id, metric="min_distance_to_exposure_km",
            value=current_min_distance_km, unit="km",
            observation_time=issuance, available_at=retrieved_at, retrieved_at=retrieved_at,
            availability_precision=AvailabilityPrecision.EXACT_TIMESTAMP if current_min_distance_km is not None
                else AvailabilityPrecision.UNKNOWN,
            parser_version=parser_version, source_release_time=issuance, forecast_issue_time=issuance,
            attrs={
                "nearest_region": current_nearest_region,
                "limitation": None if lat is not None and lon is not None else "keine aktuelle Sturmposition",
                "wind_speed_probability_product_url": (wsp or {}).get("url") if isinstance(wsp, dict) else None,
                "wind_speed_probability_issuance": (wsp or {}).get("issuance") if isinstance(wsp, dict) else None,
            },
        ))

        # ── Forecast-Track (aus Forecast/Advisory-TEXT) -> min. Distanz +
        # Stunden bis zur größten Annäherung ────────────────────────────────
        fc_adv = storm.get("forecastAdvisory")
        adv_issuance = issuance
        adv_num = None
        advisory_status = None
        advisory_error = None
        track_points: list[dict] = []
        if isinstance(fc_adv, dict):
            adv_issuance = ensure_utc(fc_adv.get("issuance")) or issuance
            adv_num = fc_adv.get("advNum")
            raw_text = advisory_texts.get(storm_id)
            if raw_text:
                try:
                    track_obs = parse_forecast_advisory_text(
                        raw_text, storm_id, adv_issuance, adv_num, retrieved_at,
                        source_id=source_id, parser_version=parser_version,
                    )
                    observations.extend(track_obs)
                    body = _extract_advisory_pre_text(raw_text)
                    track_points = parse_forecast_advisory_points(body, adv_issuance)
                except NhcAdvisoryParseError as e:
                    advisory_status = "SCHEMA_CHANGED"
                    advisory_error = str(e)
            elif fc_adv.get("url"):
                advisory_status = "NO_TEXT_FETCHED"
                advisory_error = "forecastAdvisory.url vorhanden, aber kein Advisory-Text abgerufen/übergeben"

        track_min_distance_km = None
        track_nearest_region = None
        closest_time = None
        candidate_points: list[tuple[float, float, datetime]] = []
        if lat is not None and lon is not None:
            candidate_points.append((lat, lon, adv_issuance or retrieved_at))
        for pt in track_points:
            if pt.get("lat") is not None and pt.get("lon") is not None:
                candidate_points.append((pt["lat"], pt["lon"], pt["valid_time"]))
        if candidate_points and exposure_regions:
            for plat, plon, ptime in candidate_points:
                for region in exposure_regions:
                    d = haversine_km(plat, plon, region["lat"], region["lon"])
                    if track_min_distance_km is None or d < track_min_distance_km:
                        track_min_distance_km = d
                        track_nearest_region = region.get("code")
                        closest_time = ptime

        hours_until_closest_approach = None
        if closest_time is not None:
            hours_until_closest_approach = (closest_time - retrieved_at).total_seconds() / 3600.0

        track_attrs = {
            "nearest_region": track_nearest_region,
            "advisory": adv_num,
            "advisory_status": advisory_status,
            "advisory_error": advisory_error,
            "n_track_points": len(track_points),
        }
        observations.append(Observation(
            source_id=source_id, dataset="active_storms", series_id=storm_id,
            entity_id=storm_id, metric="min_track_distance_km",
            value=track_min_distance_km, unit="km",
            observation_time=issuance, available_at=retrieved_at, retrieved_at=retrieved_at,
            availability_precision=AvailabilityPrecision.EXACT_TIMESTAMP if track_min_distance_km is not None
                else AvailabilityPrecision.UNKNOWN,
            parser_version=parser_version, source_release_time=issuance, forecast_issue_time=issuance,
            attrs=dict(track_attrs),
        ))
        observations.append(Observation(
            source_id=source_id, dataset="active_storms", series_id=storm_id,
            entity_id=storm_id, metric="hours_until_closest_approach",
            value=hours_until_closest_approach, unit="h",
            observation_time=issuance, available_at=retrieved_at, retrieved_at=retrieved_at,
            availability_precision=AvailabilityPrecision.EXACT_TIMESTAMP if hours_until_closest_approach is not None
                else AvailabilityPrecision.UNKNOWN,
            parser_version=parser_version, source_release_time=issuance, forecast_issue_time=issuance,
            attrs=dict(track_attrs),
        ))
    # ── Abruf-Marker: EINE Observation je Abruf (auch bei 0 Stürmen), mit
    # der Liste der in DIESEM Abruf aktiven Sturm-IDs. Der Kontext nutzt nur
    # den jüngsten bis as_of verfügbaren Marker, um aufgelöste Stürme aus
    # früheren Abrufen auszuschließen (das Archiv hält die ganze Historie).
    active_ids = sorted({o.series_id for o in observations if o.dataset == "active_storms"})
    observations.append(Observation(
        source_id=source_id, dataset=FETCH_SUMMARY_DATASET, series_id="nhc_current_storms",
        entity_id="", metric="active_storm_count", value=float(len(active_ids)), unit="count",
        observation_time=retrieved_at, available_at=retrieved_at, retrieved_at=retrieved_at,
        availability_precision=AvailabilityPrecision.EXACT_TIMESTAMP,
        parser_version=parser_version, attrs={"active_storm_ids": active_ids},
    ))
    return observations


def _signed_latlon(raw: str | None) -> float | None:
    """NHC liefert lat/lon als z.B. '25.5N'/'84.0W' -> signed float."""
    if raw is None:
        return None
    s = str(raw).strip()
    m = re.match(r"^(-?\d+(?:\.\d+)?)\s*([NSEW]?)$", s, re.IGNORECASE)
    if not m:
        try:
            return float(s)
        except ValueError:
            return None
    val = float(m.group(1))
    hemi = m.group(2).upper()
    if hemi in ("S", "W"):
        val = -val
    return val


class NhcStormsConnector(Connector):
    source_id = "nhc_storms"
    parser_version = PARSER_VERSION

    def fetch(self, now: datetime) -> ConnectorResult:
        url = self.cfg.get("current_storms_url", "https://www.nhc.noaa.gov/CurrentStorms.json")
        headers = {"User-Agent": user_agent_header(self.cfg)}
        exposure_cfg = load_weather_locations(self.cfg.get("locations_config", "config/weather_locations.yaml"))
        exposure_regions = exposure_cfg.get("coastal_exposure_regions", [])
        try:
            res = http.fetch(url, headers=headers)
        except http.FetchError as e:
            return ConnectorResult(source_id=self.source_id, status=SourceStatus.FAIL, message=str(e))

        raw = [_to_raw(self.source_id, "current_storms", res)]
        try:
            storms_json = res.json()
        except ValueError as e:
            return ConnectorResult(
                source_id=self.source_id, status=SourceStatus.SCHEMA_CHANGED, raw=raw, message=str(e),
                discovered_ids={"diagnostics": {
                    "body_snippet": res.content.decode("utf-8", errors="replace")[:600],
                    "content_type": res.content_type,
                }},
            )

        # Forecast/Advisory-TEXT je aktivem Sturm holen (I/O bleibt hier im
        # Connector; parse_current_storms bekommt nur die fertigen Texte,
        # damit sie pur/testbar bleibt). Ein einzelner fehlgeschlagener
        # Advisory-Abruf bricht nie den ganzen Fetch -- der betroffene Sturm
        # bekommt schlicht keinen Forecast-Track (siehe parse_current_storms).
        advisory_texts: dict[str, str] = {}
        active_storms_raw = storms_json.get("activeStorms") if isinstance(storms_json, dict) else None
        if isinstance(active_storms_raw, list):
            for storm in active_storms_raw:
                if not isinstance(storm, dict):
                    continue
                storm_id = storm.get("id") or storm.get("binNumber")
                fc_adv = storm.get("forecastAdvisory")
                if not storm_id or not isinstance(fc_adv, dict):
                    continue
                adv_url = fc_adv.get("url")
                if not adv_url:
                    continue
                try:
                    adv_res = http.fetch(adv_url, headers=headers)
                except http.FetchError:
                    continue
                raw.append(_to_raw(self.source_id, f"forecast_advisory:{storm_id}", adv_res))
                try:
                    advisory_texts[storm_id] = adv_res.content.decode("utf-8", errors="replace")
                except Exception:
                    continue

        try:
            observations = parse_current_storms(storms_json, res.retrieved_at, exposure_regions,
                                                 advisory_texts=advisory_texts,
                                                 source_id=self.source_id, parser_version=self.parser_version)
        except NhcSchemaError as e:
            return ConnectorResult(source_id=self.source_id, status=SourceStatus.SCHEMA_CHANGED,
                                    raw=raw, message=str(e), discovered_ids={"diagnostics": e.diagnostics})

        latest_release = max((o.source_release_time for o in observations if o.source_release_time), default=None)
        return ConnectorResult(source_id=self.source_id, status=SourceStatus.PASS,
                                observations=observations, raw=raw,
                                message="keine aktiven Zyklone" if not observations else "",
                                latest_release_time=latest_release)


# --------------------------------------------------------------------------- #
# noaa_storm_events – nur Backfill, default AUS (siehe Registry enabled: false)
# --------------------------------------------------------------------------- #

def parse_storm_events_counts(csv_text: str) -> dict[tuple[str, str], int]:
    """Zählt Events je (STATE, EVENT_TYPE) aus einer Storm-Events
    'details'-CSV (offizielles NCEI-Spaltenschema STATE, EVENT_TYPE, ...)."""
    counts: dict[tuple[str, str], int] = {}
    reader = csv.DictReader(io.StringIO(csv_text))
    for row in reader:
        state = (row.get("STATE") or "").strip()
        event_type = (row.get("EVENT_TYPE") or "").strip()
        if not state or not event_type:
            continue
        key = (state, event_type)
        counts[key] = counts.get(key, 0) + 1
    return counts


class NoaaStormEventsConnector(Connector):
    source_id = "noaa_storm_events"
    parser_version = PARSER_VERSION

    def fetch(self, now: datetime) -> ConnectorResult:
        if not self.cfg.get("backfill_mode"):
            return ConnectorResult(
                source_id=self.source_id, status=SourceStatus.DEFERRED,
                message="noaa_storm_events ist standardmäßig deaktiviert (große Bulk-CSV); "
                        "nur mit cfg['backfill_mode']=True aktiv.",
            )

        base_url = self.cfg.get("base_url", "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles")
        headers = {"User-Agent": user_agent_header(self.cfg)}
        year = self.cfg.get("year") or (now.year - 1)  # laufendes Jahr meist unvollständig gepflegt
        raw: list[RawRecord] = []
        try:
            listing_res = http.fetch(base_url, headers=headers)
            raw.append(_to_raw(self.source_id, "listing", listing_res))
            filename = self._pick_details_filename(listing_res.content.decode("utf-8", "ignore"), year)
            if filename is None:
                return ConnectorResult(source_id=self.source_id, status=SourceStatus.SCHEMA_CHANGED,
                                        raw=raw, message=f"keine details-Datei für {year} im Verzeichnis gefunden")
            file_res = http.fetch(f"{base_url}/{filename}", headers=headers)
            raw.append(_to_raw(self.source_id, "details", file_res))
        except http.FetchError as e:
            return ConnectorResult(source_id=self.source_id, status=SourceStatus.FAIL, raw=raw, message=str(e))

        csv_text = file_res.content.decode("utf-8", "ignore")
        counts = parse_storm_events_counts(csv_text)
        retrieved_at = file_res.retrieved_at
        obs_time = datetime(year, 1, 1, tzinfo=timezone.utc)
        observations = [
            Observation(
                source_id=self.source_id, dataset="storm_events_details", series_id=filename,
                entity_id=state, metric="event_count", value=float(n), unit="count",
                observation_time=obs_time, available_at=retrieved_at, retrieved_at=retrieved_at,
                availability_precision=AvailabilityPrecision.CONSERVATIVE_DATE,
                parser_version=self.parser_version, source_release_time=None,
                attrs={"event_type": event_type, "year": year},
            )
            for (state, event_type), n in counts.items()
        ]
        return ConnectorResult(source_id=self.source_id, status=SourceStatus.PASS,
                                observations=observations, raw=raw,
                                message=f"{len(counts)} state/event_type-Zähler aus {filename}")

    @staticmethod
    def _pick_details_filename(listing_html_or_text: str, year: int) -> str | None:
        pattern = re.compile(rf"StormEvents_details-ftp_v1\.0_d{year}_[a-z0-9]+\.csv(?:\.gz)?")
        m = pattern.search(listing_html_or_text)
        return m.group(0) if m else None


# --------------------------------------------------------------------------- #
# ecmwf_open_data – Registry-only, DEFERRED (kein aktiver Connector)
# --------------------------------------------------------------------------- #

class EcmwfOpenDataConnector(Connector):
    source_id = "ecmwf_open_data"
    parser_version = PARSER_VERSION

    def fetch(self, now: datetime) -> ConnectorResult:
        return ConnectorResult(
            source_id=self.source_id, status=SourceStatus.DEFERRED,
            message="ecmwf_open_data: Registry-Eintrag only (CC-BY-4.0, OK, "
                    "status_override DEFERRED – FORWARD_ARCHIVE_ONLY, optional). "
                    "Kein Ingestion-Code implementiert.",
        )


# --------------------------------------------------------------------------- #
# Exposure-Hierarchie: ticker_override > industry > sector > unknown
#
# WICHTIG: der HQ-Standort eines Tickers wird an keiner Stelle dieser
# Funktion (oder ihrer Config-Eingaben) verwendet – config/weather_exposures.
# yaml enthält ausschließlich operative Footprint-Daten (Hubs), niemals ein
# HQ-Feld, und resolve_weather_relevance() nimmt gar keinen HQ-Parameter an.
# --------------------------------------------------------------------------- #

def load_weather_exposures(rel_path: str = "config/weather_exposures.yaml") -> dict:
    return _load_yaml(rel_path)


def load_industry_exposure(rel_path: str = "config/industry_exposure.yaml") -> dict:
    return _load_yaml(rel_path)


def resolve_exposure_hub_codes(ticker: str, exposures_cfg: dict | None = None) -> list[str] | None:
    """Nur ticker_overrides definieren konkrete Hub-Codes (operativer
    Footprint). Kein Ticker-Override -> None (nicht []): wir wissen schlicht
    nichts über den Footprint, das ist kein "kein Exposure"."""
    cfg = exposures_cfg if exposures_cfg is not None else load_weather_exposures()
    override = (cfg.get("ticker_overrides") or {}).get(ticker)
    if not override:
        return None
    return list(override.get("hub_codes") or [])


def resolve_weather_relevance(ticker: str | None = None, yf_sector: str | None = None,
                               yf_industry: str | None = None,
                               exposures_cfg: dict | None = None,
                               industry_cfg: dict | None = None) -> dict:
    """Exposure-Hierarchie: ticker_override > industry > sector > unknown.

    Gibt {"level": "ticker"|"industry"|"sector"|"unknown",
          "weather_relevance": "HIGH"|"MEDIUM"|"LOW"|"NONE"|None,
          "hub_codes": list[str]|None, "industry": str|None} zurück.
    "unknown" liefert weather_relevance=None (NIE "NONE" – "NONE" ist eine
    positive Aussage 'kein Impact', None heißt 'wir wissen es nicht')."""
    exp_cfg = exposures_cfg if exposures_cfg is not None else load_weather_exposures()
    ind_cfg = industry_cfg if industry_cfg is not None else load_industry_exposure()

    if ticker:
        override = (exp_cfg.get("ticker_overrides") or {}).get(ticker)
        if override:
            return {"level": "ticker", "weather_relevance": "HIGH",
                    "hub_codes": list(override.get("hub_codes") or []),
                    "industry": None, "source": override.get("source")}

    industries = ind_cfg.get("industries") or {}
    industry_map = ind_cfg.get("yfinance_industry_map") or {}
    if yf_industry and yf_industry in industry_map:
        industry_name = industry_map[yf_industry]
        rel = (industries.get(industry_name) or {}).get("weather_relevance")
        return {"level": "industry", "weather_relevance": rel, "hub_codes": None, "industry": industry_name}

    sector_fallback = ind_cfg.get("sector_fallback") or {}
    if yf_sector and yf_sector in sector_fallback:
        industry_name = sector_fallback[yf_sector]
        rel = (industries.get(industry_name) or {}).get("weather_relevance")
        return {"level": "sector", "weather_relevance": rel, "hub_codes": None, "industry": industry_name}

    return {"level": "unknown", "weather_relevance": None, "hub_codes": None, "industry": None}


CONNECTORS: dict[str, type[Connector]] = {
    "nws_forecast": NwsForecastConnector,
    "nws_alerts": NwsAlertsConnector,
    "ncei_normals": NceiNormalsConnector,
    "nhc_storms": NhcStormsConnector,
    "noaa_storm_events": NoaaStormEventsConnector,
    "ecmwf_open_data": EcmwfOpenDataConnector,
}

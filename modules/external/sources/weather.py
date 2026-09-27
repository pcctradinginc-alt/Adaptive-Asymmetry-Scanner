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
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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

    def fetch(self, now: datetime) -> ConnectorResult:
        base_url = self.cfg.get("base_url", "https://api.weather.gov")
        elements = self.cfg.get("elements", DEFAULT_GRID_ELEMENTS)
        headers = {"User-Agent": self._user_agent(), "Accept": "application/geo+json"}

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

                obs = parse_grid_response(points_json, grid_json, loc, elements,
                                           retrieved_at=grid_res.retrieved_at,
                                           source_id=self.source_id,
                                           parser_version=self.parser_version)
                observations.extend(obs)
                discovered[loc["code"]] = {
                    "gridId": points_json["properties"].get("gridId"),
                    "gridX": points_json["properties"].get("gridX"),
                    "gridY": points_json["properties"].get("gridY"),
                }
                for o in obs:
                    if latest_release is None or (o.source_release_time and o.source_release_time > latest_release):
                        latest_release = o.source_release_time
            except (http.FetchError, ValueError, KeyError) as e:
                failures += 1
                continue

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

def parse_current_storms(storms_json: dict, retrieved_at: datetime,
                          exposure_regions: list[dict] | None = None,
                          source_id: str = "nhc_storms",
                          parser_version: str = PARSER_VERSION) -> list[Observation]:
    """CurrentStorms.json -> Observations je aktivem Sturm. Wenn
    exposure_regions übergeben werden UND der Sturm Forecast-Track-Punkte
    enthält (Feld 'forecastTrack', Liste von {lat, lon, validTime}), wird die
    minimale Distanz zu jeder Region berechnet; sonst bleibt distance_km auf
    None (Limitation: CurrentStorms.json selbst liefert i.d.R. keine
    strukturierten Forecast-Punkte – volles GIS-Parsing der Advisory-
    Produkte ist hier bewusst NICHT implementiert, siehe Registry-Notes)."""
    retrieved_at = ensure_utc(retrieved_at) or utc_now()
    observations: list[Observation] = []
    exposure_regions = exposure_regions or []

    for storm in storms_json.get("activeStorms", []):
        storm_id = storm.get("id") or storm.get("binNumber")
        if not storm_id:
            continue
        advisory = storm.get("publicAdvisory") or {}
        issuance = ensure_utc(advisory.get("issuance")) or ensure_utc(storm.get("lastUpdate"))
        if issuance is None:
            continue

        def _num(x):
            if x is None:
                return None
            try:
                return float(re.sub(r"[^0-9.\-]", "", str(x)))
            except ValueError:
                return None

        lat = _signed_latlon(storm.get("lat"))
        lon = _signed_latlon(storm.get("lon"))

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
        forecast_track = storm.get("forecastTrack")
        min_distance_km = None
        nearest_region = None
        if forecast_track and exposure_regions:
            for pt in forecast_track:
                plat, plon = pt.get("lat"), pt.get("lon")
                if plat is None or plon is None:
                    continue
                for region in exposure_regions:
                    d = haversine_km(float(plat), float(plon), region["lat"], region["lon"])
                    if min_distance_km is None or d < min_distance_km:
                        min_distance_km = d
                        nearest_region = region.get("code")

        observations.append(Observation(
            source_id=source_id, dataset="active_storms", series_id=storm_id,
            entity_id=storm_id, metric="min_distance_to_exposure_km",
            value=min_distance_km, unit="km",
            observation_time=issuance, available_at=retrieved_at, retrieved_at=retrieved_at,
            availability_precision=AvailabilityPrecision.EXACT_TIMESTAMP if min_distance_km is not None
                else AvailabilityPrecision.UNKNOWN,
            parser_version=parser_version, source_release_time=issuance, forecast_issue_time=issuance,
            attrs={
                "nearest_region": nearest_region,
                "limitation": None if forecast_track else (
                    "CurrentStorms.json enthielt keine forecastTrack-Punkte; "
                    "volles GIS/Text-Parsing der Advisory-Produkte ist nicht implementiert"
                ),
                "wind_speed_probability_product_url": (wsp or {}).get("url") if isinstance(wsp, dict) else None,
                "wind_speed_probability_issuance": (wsp or {}).get("issuance") if isinstance(wsp, dict) else None,
            },
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
            return ConnectorResult(source_id=self.source_id, status=SourceStatus.SCHEMA_CHANGED,
                                    raw=raw, message=str(e))

        observations = parse_current_storms(storms_json, res.retrieved_at, exposure_regions,
                                             source_id=self.source_id, parser_version=self.parser_version)
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

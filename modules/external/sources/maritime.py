"""
modules/external/sources/maritime.py – IMF PortWatch (maritime freight) Konnektor.

Datenquelle: IMF PortWatch (https://portwatch.imf.org), betrieben von IMF &
University of Oxford. Veröffentlicht als ArcGIS Hub Feature Layers:
  - "Daily Ports Data"        – tägliche Port-Calls / Import-/Export-Aktivität
  - "Daily Chokepoints Data"  – tägliche Transits durch maritime Chokepoints
  - "Ports"                    – Referenz-Layer (Port-ID <-> Name/Land)
  - "Chokepoints"               – Referenz-Layer (Chokepoint-ID <-> Name)

WICHTIG (ehrlich dokumentierte Annahmen, die LIVE verifiziert werden müssen –
in diesem Sandbox gibt es keinen Internetzugriff auf ArcGIS/PortWatch):
  - Exakte Feldnamen der Daily-Ports/-Chokepoints-Layer (z.B. ob "import" oder
    "import_total" heißt) sind aus Fachwissen zum PortWatch-Datensatz
    rekonstruiert, NICHT live gegen die API verifiziert. Der Schema-Validator
    prüft daher nur ein konservatives Kern-Feld-Set + Präfix-Muster
    (portcalls_*, import*, export* bzw. n_*) und schlägt LAUT mit
    SCHEMA_CHANGED fehl, wenn diese fehlen – er rät NIE ein umbenanntes Feld.
  - Die exakte ArcGIS-Organisation/Owner-ID von PortWatch wird nicht fest
    verdrahtet (Gefahr, eine erratene ID zu "erfinden"); Discovery sucht per
    Titel über die öffentliche ArcGIS-Sharing-Search-API und muss in
    preflight() gegen die echte API bestätigt werden.
  - License-Status ist REVIEW_REQUIRED (siehe maritime.yaml) bis ein Mensch
    die PortWatch-Nutzungsbedingungen live geprüft hat.

Alle Netzwerk-Aufrufe laufen über `modules.external.http.fetch` (injizierbar
für Tests). Es werden NIE IDs erfunden: Port-/Chokepoint-IDs kommen
ausschließlich aus den offiziellen Referenz-Layern.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Iterable

import yaml

from modules.external import http
from modules.external.http import FetchError
from modules.external.pit import (
    AvailabilityPrecision,
    Observation,
    request_fingerprint,
    utc_now,
)
from modules.external.sources.base import (
    Connector,
    ConnectorResult,
    RawRecord,
    SourceStatus,
)

FetchFn = Callable[..., Any]

# --------------------------------------------------------------------------
# Konstanten / Konfiguration
# --------------------------------------------------------------------------

ARCGIS_SEARCH_URL = "https://www.arcgis.com/sharing/rest/search"
PORTWATCH_HOMEPAGE = "https://portwatch.imf.org"

# Titel der ArcGIS-Hub-Items, wie sie öffentlich auf portwatch.imf.org /
# ArcGIS Online geführt werden (VERIFY LIVE — Titel können sich ändern).
DATASET_TITLES: dict[str, str] = {
    "ports_daily": "Daily Ports Data",
    "chokepoints_daily": "Daily Chokepoints Data",
    "ports_reference": "Ports",
    "chokepoints_reference": "Chokepoints",
}

DEFAULT_CACHE_PATH = "outputs/external_data/manifests/portwatch_endpoints.json"
DEFAULT_PORT_UNIVERSE_PATH = "config/port_universe.yaml"
CACHE_TTL_DAYS = 30

# Vor 2019 gibt es laut PortWatch-Dokumentation (VERIFY LIVE) keine
# durchgängige Historie -> konservativer Backfill-Start.
BACKFILL_START = date(2019, 1, 1)
INCREMENTAL_WINDOW_DAYS = 21

# Kern-Felder, die im Daily-Ports-Layer vorhanden sein MÜSSEN, sonst
# SCHEMA_CHANGED (nie stillschweigend umbenannte Felder mappen).
CORE_FIELDS_PORTS = {"date", "portid", "portname"}
CARGO_PREFIXES_PORTS = ("portcalls", "import", "export")

CORE_FIELDS_CHOKEPOINTS = {"date"}
# mögliche Namens-/ID-/Zähl-Feldvarianten (VERIFY LIVE welche tatsächlich existieren)
CHOKEPOINT_ID_CANDIDATES = ("chokepointid", "choke_id", "portid", "id", "objectid")
CHOKEPOINT_NAME_CANDIDATES = ("chokepointname", "portname", "name")
CHOKEPOINT_COUNT_PREFIXES = ("n_", "vessel", "transit", "capacity")


class DiscoveryError(Exception):
    """Endpoint konnte nicht über die ArcGIS-Sharing-Search-API gefunden werden."""


class SchemaError(Exception):
    """Layer-Metadaten passen nicht zum erwarteten Kern-Schema -> SCHEMA_CHANGED."""


# --------------------------------------------------------------------------
# Discovery (ArcGIS Hub / Online Sharing Search API)
# --------------------------------------------------------------------------

def _search_arcgis_item(title: str, http_fetch: FetchFn, extra_query: str = "") -> dict:
    """Sucht ein ArcGIS-Item per Titel. Wählt bei Mehrdeutigkeit den zuletzt
    geänderten "Feature Service"-Treffer, markiert das Ergebnis aber als
    `ambiguous`, damit ein Preflight-Check das laut melden kann."""
    q = f'title:"{title}"'
    if extra_query:
        q += f" {extra_query}"
    res = http_fetch(ARCGIS_SEARCH_URL, params={"q": q, "f": "json", "num": 25})
    data = res.json()
    results = data.get("results", [])
    if not results:
        raise DiscoveryError(f"ArcGIS-Suche ohne Treffer für title={title!r}")

    def _is_exact(r: dict) -> bool:
        return r.get("title", "").strip().lower() == title.strip().lower()

    exact_services = [r for r in results if _is_exact(r) and r.get("type") == "Feature Service"]
    services = exact_services or [r for r in results if r.get("type") == "Feature Service"]
    candidates = services or results
    ambiguous = len(candidates) > 1
    chosen = sorted(candidates, key=lambda r: r.get("modified", 0), reverse=True)[0]
    return {
        "item_id": chosen.get("id"),
        "title": chosen.get("title"),
        "service_url": (chosen.get("url") or "").rstrip("/"),
        "ambiguous": ambiguous,
        "n_candidates": len(candidates),
    }


def _resolve_layer(service_url: str, http_fetch: FetchFn) -> dict:
    """Fragt den FeatureServer selbst (f=json) ab, um herauszufinden, welcher
    Layer-Index die eigentlichen Daten trägt (meist 0, aber nie annehmen)."""
    res = http_fetch(service_url, params={"f": "json"})
    data = res.json()
    layers = data.get("layers", [])
    if not layers:
        # manche Services geben den Layer direkt zurück statt einer Liste
        return {"layer_id": 0}
    return {"layer_id": layers[0].get("id", 0)}


def discover_endpoints(
    http_fetch: FetchFn | None = None,
    cache_path: str = DEFAULT_CACHE_PATH,
    force_refresh: bool = False,
    config_override: dict | None = None,
    now: datetime | None = None,
) -> dict:
    """Ermittelt die vier PortWatch-Service-URLs (Daily Ports, Daily
    Chokepoints, Ports-Referenz, Chokepoints-Referenz) via ArcGIS-Suche,
    cached das Ergebnis unter `cache_path` und respektiert eine Config-
    Override (machine_endpoint) für das Ports-Daily-Layer -- die dann als
    "verify_in_preflight" markiert wird, weil sie ungeprüft übernommen wurde.
    """
    fetch_fn = http_fetch or http.fetch
    now = now or utc_now()

    cached = _load_endpoint_cache(cache_path)
    if cached is not None and not force_refresh:
        discovered_at = cached.get("discovered_at")
        if discovered_at:
            age = now - datetime.fromisoformat(discovered_at)
            if age <= timedelta(days=CACHE_TTL_DAYS):
                endpoints = cached.get("endpoints", {})
                return _apply_override(endpoints, config_override)

    endpoints: dict[str, dict] = {}
    for key, title in DATASET_TITLES.items():
        item = _search_arcgis_item(title, fetch_fn)
        layer = _resolve_layer(item["service_url"], fetch_fn)
        endpoints[key] = {
            "service_url": item["service_url"],
            "layer_id": layer["layer_id"],
            "item_id": item["item_id"],
            "title": item["title"],
            "ambiguous": item["ambiguous"],
        }

    _write_endpoint_cache(cache_path, endpoints, now)
    return _apply_override(endpoints, config_override)


def _apply_override(endpoints: dict, config_override: dict | None) -> dict:
    endpoints = {k: dict(v) for k, v in endpoints.items()}
    if config_override and config_override.get("ports_daily_service_url"):
        endpoints.setdefault("ports_daily", {})
        endpoints["ports_daily"]["service_url"] = config_override["ports_daily_service_url"]
        endpoints["ports_daily"]["layer_id"] = config_override.get("ports_daily_layer_id", 0)
        endpoints["ports_daily"]["verify_in_preflight"] = True
    if config_override and config_override.get("chokepoints_daily_service_url"):
        endpoints.setdefault("chokepoints_daily", {})
        endpoints["chokepoints_daily"]["service_url"] = config_override["chokepoints_daily_service_url"]
        endpoints["chokepoints_daily"]["layer_id"] = config_override.get("chokepoints_daily_layer_id", 0)
        endpoints["chokepoints_daily"]["verify_in_preflight"] = True
    return endpoints


def _load_endpoint_cache(cache_path: str) -> dict | None:
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _write_endpoint_cache(cache_path: str, endpoints: dict, now: datetime) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    payload = {"discovered_at": now.isoformat(), "endpoints": endpoints}
    with open(cache_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


# --------------------------------------------------------------------------
# Layer-Metadaten & Schema-Validierung
# --------------------------------------------------------------------------

def fetch_layer_metadata(service_url: str, layer_id: int, http_fetch: FetchFn) -> dict:
    url = f"{service_url}/{layer_id}"
    res = http_fetch(url, params={"f": "json"})
    meta = res.json()
    if "fields" not in meta:
        raise SchemaError(f"Layer-Metadaten ohne 'fields': {url}")
    return meta


def _field_names(meta: dict) -> set[str]:
    return {f["name"] for f in meta.get("fields", [])}


def validate_ports_schema(meta: dict) -> list[str]:
    """Prüft Kern-Felder + mind. ein Cargo-Feld. Gibt die Liste aller
    Cargo-/Import-/Export-Felder zurück (das ist das, was tatsächlich
    beobachtet wird -- nichts wird umbenannt/erraten)."""
    fields = _field_names(meta)
    missing = CORE_FIELDS_PORTS - fields
    if missing:
        raise SchemaError(f"SCHEMA_CHANGED: Kernfelder fehlen im Ports-Layer: {sorted(missing)}")
    cargo_fields = sorted(f for f in fields if f.startswith(CARGO_PREFIXES_PORTS))
    if not cargo_fields:
        raise SchemaError("SCHEMA_CHANGED: keine portcalls_/import_/export_-Felder im Ports-Layer gefunden")
    return cargo_fields


def validate_chokepoints_schema(meta: dict) -> dict:
    fields = _field_names(meta)
    missing = CORE_FIELDS_CHOKEPOINTS - fields
    if missing:
        raise SchemaError(f"SCHEMA_CHANGED: Kernfelder fehlen im Chokepoints-Layer: {sorted(missing)}")
    id_field = next((f for f in CHOKEPOINT_ID_CANDIDATES if f in fields), None)
    name_field = next((f for f in CHOKEPOINT_NAME_CANDIDATES if f in fields), None)
    count_fields = sorted(f for f in fields if f.lower().startswith(CHOKEPOINT_COUNT_PREFIXES))
    if id_field is None or name_field is None:
        raise SchemaError(
            "SCHEMA_CHANGED: kein bekanntes ID-/Name-Feld im Chokepoints-Layer "
            f"(Felder vorhanden: {sorted(fields)})"
        )
    if not count_fields:
        raise SchemaError("SCHEMA_CHANGED: keine Transit-/Kapazitäts-Felder im Chokepoints-Layer gefunden")
    return {"id_field": id_field, "name_field": name_field, "count_fields": count_fields}


# --------------------------------------------------------------------------
# Paginierte Query gegen einen FeatureServer-Layer
# --------------------------------------------------------------------------

MAX_PAGES_SAFETY = 500


def build_where_clause(start: date, end: date, date_field: str = "date") -> str:
    return (
        f"{date_field} >= TIMESTAMP '{start.isoformat()} 00:00:00' "
        f"AND {date_field} <= TIMESTAMP '{end.isoformat()} 23:59:59'"
    )


def query_features(
    service_url: str,
    layer_id: int,
    where: str,
    out_fields: Iterable[str] | str,
    http_fetch: FetchFn,
    order_by: str | None = None,
    max_record_count: int | None = None,
    return_geometry: bool = False,
) -> list[dict]:
    """Robuste Pagination über resultOffset/resultRecordCount. Nimmt NIE eine
    fixe Pagegröße an -- nutzt maxRecordCount aus den Layer-Metadaten. Stoppt,
    wenn `exceededTransferLimit` nicht (mehr) gesetzt ist oder keine Features
    mehr zurückkommen."""
    out = ",".join(out_fields) if out_fields != "*" else "*"
    page_size = max_record_count or 2000
    query_url = f"{service_url}/{layer_id}/query"
    offset = 0
    all_features: list[dict] = []
    for _ in range(MAX_PAGES_SAFETY):
        params = {
            "where": where,
            "outFields": out,
            "resultOffset": offset,
            "resultRecordCount": page_size,
            "returnGeometry": str(return_geometry).lower(),
            "f": "json",
        }
        if order_by:
            params["orderByFields"] = order_by
        res = http_fetch(query_url, params=params)
        data = res.json()
        if "error" in data:
            raise FetchError(f"ArcGIS-Query-Fehler: {data['error']}")
        feats = data.get("features", [])
        all_features.extend(feats)
        exceeded = bool(data.get("exceededTransferLimit"))
        if not exceeded or not feats:
            break
        offset += len(feats)
    else:  # pragma: no cover - Sicherheitsnetz gegen Endlos-Pagination
        raise FetchError(f"Pagination-Limit ({MAX_PAGES_SAFETY} Seiten) erreicht für {query_url}")
    return all_features


def incremental_window(now: datetime, days: int = INCREMENTAL_WINDOW_DAYS) -> tuple[date, date]:
    end = now.date()
    start = end - timedelta(days=days)
    return start, end


# --------------------------------------------------------------------------
# Name-Normalisierung & Universe-Resolution (nie IDs erfinden)
# --------------------------------------------------------------------------

_PAREN_RE = re.compile(r"[()\-–,.]")
_WS_RE = re.compile(r"\s+")


def normalize_name(name: str) -> str:
    if not name:
        return ""
    s = name.lower()
    s = _PAREN_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    for prefix in ("port of ", "the "):
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s


@dataclass
class ResolvedEntity:
    name: str
    country: str | None
    entity_id: str | None
    status: str  # "resolved" | "ambiguous" | "no_match"
    candidates: list[dict]


def resolve_entities_by_name(
    wanted: list[dict],
    reference_features: list[dict],
    id_field: str,
    name_field: str,
    country_field: str | None = "country",
) -> list[ResolvedEntity]:
    """Matched eine gewünschte Liste {name, country?} gegen Referenz-Layer-
    Features (exakter, normalisierter Name; bei mehreren Treffern zusätzlich
    nach Land gefiltert). Kein Fuzzy-Raten -- unklare Fälle werden als
    'ambiguous' bzw. 'no_match' zurückgegeben, nie mit einer geratenen ID."""
    index: dict[str, list[dict]] = {}
    for feat in reference_features:
        attrs = feat.get("attributes", feat)
        norm = normalize_name(str(attrs.get(name_field, "")))
        index.setdefault(norm, []).append(attrs)

    results: list[ResolvedEntity] = []
    for item in wanted:
        name = item["name"]
        country = item.get("country")
        norm = normalize_name(name)
        matches = index.get(norm, [])
        if not matches:
            results.append(ResolvedEntity(name, country, None, "no_match", []))
            continue
        if len(matches) > 1 and country and country_field:
            country_norm = normalize_name(country)
            filtered = [
                m for m in matches
                if normalize_name(str(m.get(country_field, ""))) == country_norm
            ]
            if len(filtered) == 1:
                m = filtered[0]
                results.append(ResolvedEntity(name, country, str(m.get(id_field)), "resolved", []))
                continue
            matches = filtered or matches
        if len(matches) == 1:
            m = matches[0]
            results.append(ResolvedEntity(name, country, str(m.get(id_field)), "resolved", []))
        else:
            results.append(ResolvedEntity(name, country, None, "ambiguous", matches))
    return results


def load_port_universe(path: str = DEFAULT_PORT_UNIVERSE_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def flatten_port_universe(universe: dict) -> list[dict]:
    """groups: {GROUP: [{name, country}, ...]} -> flache Liste mit group-Tag."""
    flat = []
    for group, ports in (universe.get("groups") or {}).items():
        for p in ports:
            flat.append({"name": p["name"], "country": p.get("country"), "group": group})
    return flat


def flatten_chokepoint_universe(universe: dict) -> list[dict]:
    return [{"name": c["name"], "country": c.get("country")} for c in (universe.get("chokepoints") or [])]


# --------------------------------------------------------------------------
# Observation-Building
# --------------------------------------------------------------------------

_METRIC_RENAME = {"portcalls": "portcalls_total", "import": "import_total", "export": "export_total"}


def _metric_name(field: str) -> str:
    return _METRIC_RENAME.get(field, field)


def _vessel_type_of(metric: str) -> str:
    for prefix in ("portcalls_", "import_", "export_"):
        if metric.startswith(prefix) and metric not in ("portcalls_total", "import_total", "export_total"):
            return metric[len(prefix):]
    return "all"


def _arcgis_date_to_utc(raw: Any) -> datetime | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return datetime.fromtimestamp(raw / 1000.0, tz=timezone.utc)
    return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))


def build_port_observations(
    attrs: dict,
    retrieved_at: datetime,
    source_id: str,
    dataset: str,
    cargo_fields: list[str],
    parser_version: str,
    is_backfill: bool,
) -> list[Observation]:
    obs_time = _arcgis_date_to_utc(attrs.get("date"))
    if obs_time is None:
        return []
    port_id = str(attrs.get("portid"))
    port_name = attrs.get("portname")
    country = attrs.get("country") or attrs.get("ISO3")
    precision = AvailabilityPrecision.UNKNOWN if is_backfill else AvailabilityPrecision.CONSERVATIVE_DATE
    result = []
    for field in cargo_fields:
        value = attrs.get(field)
        if value is None:
            continue
        metric = _metric_name(field)
        entity_attrs = {"port_name": port_name, "country": country, "vessel_type": _vessel_type_of(metric)}
        if is_backfill:
            entity_attrs["historical_backfill"] = True
        result.append(Observation(
            source_id=source_id, dataset=dataset, series_id=field, entity_id=port_id,
            metric=metric, value=float(value), unit="count",
            observation_time=obs_time, available_at=retrieved_at, retrieved_at=retrieved_at,
            availability_precision=precision, parser_version=parser_version,
            attrs=entity_attrs,
        ))
    return result


def build_chokepoint_observations(
    attrs: dict,
    schema: dict,
    retrieved_at: datetime,
    source_id: str,
    dataset: str,
    parser_version: str,
    is_backfill: bool,
) -> list[Observation]:
    obs_time = _arcgis_date_to_utc(attrs.get("date"))
    if obs_time is None:
        return []
    entity_id = str(attrs.get(schema["id_field"]))
    name = attrs.get(schema["name_field"])
    precision = AvailabilityPrecision.UNKNOWN if is_backfill else AvailabilityPrecision.CONSERVATIVE_DATE
    result = []
    for field in schema["count_fields"]:
        value = attrs.get(field)
        if value is None:
            continue
        entity_attrs = {"port_name": name, "country": attrs.get("country"), "vessel_type": _vessel_type_of(field)}
        if is_backfill:
            entity_attrs["historical_backfill"] = True
        result.append(Observation(
            source_id=source_id, dataset=dataset, series_id=field, entity_id=entity_id,
            metric=field, value=float(value), unit="count",
            observation_time=obs_time, available_at=retrieved_at, retrieved_at=retrieved_at,
            availability_precision=precision, parser_version=parser_version,
            attrs=entity_attrs,
        ))
    return result


# --------------------------------------------------------------------------
# Connectors
# --------------------------------------------------------------------------

class _PortWatchConnectorBase(Connector):
    """Gemeinsame Logik für Ports- und Chokepoints-Konnektor."""

    def __init__(self, source_cfg: dict | None = None, http_fetch: FetchFn | None = None,
                 cache_path: str = DEFAULT_CACHE_PATH, port_universe_path: str = DEFAULT_PORT_UNIVERSE_PATH):
        super().__init__(source_cfg)
        self._fetch = http_fetch or http.fetch
        self._cache_path = cache_path
        self._universe_path = port_universe_path

    def _discover(self) -> dict:
        override = (self.cfg or {}).get("machine_endpoint_override")
        return discover_endpoints(http_fetch=self._fetch, cache_path=self._cache_path, config_override=override)

    def _raw_record(self, url: str, where: str, retrieved_at: datetime) -> RawRecord:
        return RawRecord(
            source_id=self.source_id, dataset=self.dataset, url=url,
            fingerprint=request_fingerprint("GET", url, {"where": where}),
            retrieved_at=retrieved_at, status_code=200, content_type="application/json",
            content_hash="", bytes=0,
        )


class PortWatchPortsConnector(_PortWatchConnectorBase):
    source_id = "imf_portwatch_ports"
    dataset = "daily_ports"
    parser_version = "1"

    def fetch(self, now: datetime, backfill: bool = False) -> ConnectorResult:
        try:
            endpoints = self._discover()
            ep = endpoints["ports_daily"]
            ref_ep = endpoints["ports_reference"]

            meta = fetch_layer_metadata(ep["service_url"], ep["layer_id"], self._fetch)
            cargo_fields = validate_ports_schema(meta)
            max_rc = meta.get("maxRecordCount", 2000)

            start, end = (BACKFILL_START, now.date()) if backfill else incremental_window(now)
            where = build_where_clause(start, end)
            out_fields = sorted({"date", "portid", "portname", "country", "ISO3"} | set(cargo_fields))
            raw_features = query_features(
                ep["service_url"], ep["layer_id"], where, out_fields, self._fetch,
                order_by="date ASC", max_record_count=max_rc,
            )

            ref_meta = fetch_layer_metadata(ref_ep["service_url"], ref_ep["layer_id"], self._fetch)
            ref_features = query_features(
                ref_ep["service_url"], ref_ep["layer_id"], "1=1",
                ["portid", "portname", "country"], self._fetch,
                max_record_count=ref_meta.get("maxRecordCount", 2000),
            )
            universe = load_port_universe(self._universe_path)
            wanted = flatten_port_universe(universe)
            resolved = resolve_entities_by_name(wanted, ref_features, "portid", "portname", "country")
            discovered_ids = {r.name: r.entity_id for r in resolved if r.status == "resolved"}
            unresolved = {r.name: r.status for r in resolved if r.status != "resolved"}

            retrieved_at = now
            observations: list[Observation] = []
            for feat in raw_features:
                observations.extend(build_port_observations(
                    feat.get("attributes", {}), retrieved_at, self.source_id, self.dataset,
                    cargo_fields, self.parser_version, is_backfill=backfill,
                ))

            raw = [self._raw_record(f"{ep['service_url']}/{ep['layer_id']}/query", where, retrieved_at)]
            latest = max((o.observation_time for o in observations), default=None)
            status = SourceStatus.WARN if unresolved else SourceStatus.PASS
            return ConnectorResult(
                source_id=self.source_id, status=status, observations=observations, raw=raw,
                message=f"{len(observations)} observations; {len(unresolved)} unresolved ports: {sorted(unresolved)}",
                latest_observation_time=latest,
                discovered_ids={"ports": discovered_ids, "unresolved_ports": unresolved},
            )
        except SchemaError as e:
            return ConnectorResult(source_id=self.source_id, status=SourceStatus.SCHEMA_CHANGED, message=str(e))
        except DiscoveryError as e:
            return ConnectorResult(source_id=self.source_id, status=SourceStatus.FAIL, message=str(e))
        except FetchError as e:
            return ConnectorResult(source_id=self.source_id, status=SourceStatus.FAIL, message=str(e))


class PortWatchChokepointsConnector(_PortWatchConnectorBase):
    source_id = "imf_portwatch_chokepoints"
    dataset = "daily_chokepoints"
    parser_version = "1"

    def fetch(self, now: datetime, backfill: bool = False) -> ConnectorResult:
        try:
            endpoints = self._discover()
            ep = endpoints["chokepoints_daily"]
            ref_ep = endpoints["chokepoints_reference"]

            meta = fetch_layer_metadata(ep["service_url"], ep["layer_id"], self._fetch)
            schema = validate_chokepoints_schema(meta)
            max_rc = meta.get("maxRecordCount", 2000)

            start, end = (BACKFILL_START, now.date()) if backfill else incremental_window(now)
            where = build_where_clause(start, end)
            out_fields = sorted({"date", schema["id_field"], schema["name_field"]} | set(schema["count_fields"]))
            raw_features = query_features(
                ep["service_url"], ep["layer_id"], where, out_fields, self._fetch,
                order_by="date ASC", max_record_count=max_rc,
            )

            ref_meta = fetch_layer_metadata(ref_ep["service_url"], ref_ep["layer_id"], self._fetch)
            ref_features = query_features(
                ref_ep["service_url"], ref_ep["layer_id"], "1=1",
                [schema["id_field"], schema["name_field"]], self._fetch,
                max_record_count=ref_meta.get("maxRecordCount", 2000),
            )
            universe = load_port_universe(self._universe_path)
            wanted = flatten_chokepoint_universe(universe)
            resolved = resolve_entities_by_name(
                wanted, ref_features, schema["id_field"], schema["name_field"], country_field=None,
            )
            discovered_ids = {r.name: r.entity_id for r in resolved if r.status == "resolved"}
            unresolved = {r.name: r.status for r in resolved if r.status != "resolved"}

            retrieved_at = now
            observations: list[Observation] = []
            for feat in raw_features:
                observations.extend(build_chokepoint_observations(
                    feat.get("attributes", {}), schema, retrieved_at, self.source_id, self.dataset,
                    self.parser_version, is_backfill=backfill,
                ))

            raw = [self._raw_record(f"{ep['service_url']}/{ep['layer_id']}/query", where, retrieved_at)]
            latest = max((o.observation_time for o in observations), default=None)
            status = SourceStatus.WARN if unresolved else SourceStatus.PASS
            return ConnectorResult(
                source_id=self.source_id, status=status, observations=observations, raw=raw,
                message=f"{len(observations)} observations; {len(unresolved)} unresolved chokepoints: {sorted(unresolved)}",
                latest_observation_time=latest,
                discovered_ids={"chokepoints": discovered_ids, "unresolved_chokepoints": unresolved},
            )
        except SchemaError as e:
            return ConnectorResult(source_id=self.source_id, status=SourceStatus.SCHEMA_CHANGED, message=str(e))
        except DiscoveryError as e:
            return ConnectorResult(source_id=self.source_id, status=SourceStatus.FAIL, message=str(e))
        except FetchError as e:
            return ConnectorResult(source_id=self.source_id, status=SourceStatus.FAIL, message=str(e))


CONNECTORS: dict[str, type[Connector]] = {
    PortWatchPortsConnector.source_id: PortWatchPortsConnector,
    PortWatchChokepointsConnector.source_id: PortWatchChokepointsConnector,
}

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

import difflib
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
# Alias-Tabelle: erlaubt case-insensitive Matching gegen tatsächlich in der
# Layer-Metadata gefundene Feldnamen (z.B. "portId" statt "portid") -- es
# wird NIE ein Feld erraten, das nicht mindestens einem dokumentierten Alias
# entspricht. outFields wird ausschließlich aus der Schnittmenge
# {gewünschte logische Felder} x {tatsächliche Metadaten-Felder} gebaut.
FIELD_ALIASES_PORTS: dict[str, tuple[str, ...]] = {
    "date": ("date", "Date", "DATE"),
    "portid": ("portid", "portId", "PortId", "PORTID", "port_id"),
    "portname": ("portname", "portName", "PortName", "PORTNAME", "port_name"),
    "country": ("country", "Country", "COUNTRY"),
    "iso3": ("ISO3", "iso3", "Iso3"),
}
CARGO_PREFIXES_PORTS = ("portcalls", "import", "export")

FIELD_ALIASES_CHOKEPOINTS: dict[str, tuple[str, ...]] = {
    "date": ("date", "Date", "DATE"),
    "chokepointid": ("chokepointid", "choke_id", "chokePointId", "CHOKEPOINTID", "portid", "id", "objectid"),
    "chokepointname": ("chokepointname", "choke_name", "chokePointName", "CHOKEPOINTNAME", "portname", "name"),
}
CHOKEPOINT_COUNT_PREFIXES = ("n_", "vessel", "transit", "capacity")

# Rückwärtskompatible Namen (werden von älteren Aufrufern ggf. importiert).
CORE_FIELDS_PORTS = {"date", "portid", "portname"}
CORE_FIELDS_CHOKEPOINTS = {"date"}
CHOKEPOINT_ID_CANDIDATES = FIELD_ALIASES_CHOKEPOINTS["chokepointid"]
CHOKEPOINT_NAME_CANDIDATES = FIELD_ALIASES_CHOKEPOINTS["chokepointname"]


class DiscoveryError(Exception):
    """Endpoint konnte nicht über die ArcGIS-Sharing-Search-API gefunden werden."""


class SchemaError(Exception):
    """Layer-Metadaten passen nicht zum erwarteten Kern-Schema -> SCHEMA_CHANGED.

    `diagnostics` trägt sichere, öffentliche Debug-Infos (Feldliste,
    maxRecordCount, Service-URL) -- NIE Request-Parameter/Credentials."""

    def __init__(self, message: str, diagnostics: dict | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


# --------------------------------------------------------------------------
# Discovery (ArcGIS Hub / Online Sharing Search API)
# --------------------------------------------------------------------------

def _search_arcgis_item(
    title: str,
    http_fetch: FetchFn,
    extra_query: str = "",
    require_owner: str | None = None,
    require_orgid: str | None = None,
) -> dict:
    """Sucht ein ArcGIS-Item per Titel. Wählt bei Mehrdeutigkeit den zuletzt
    geänderten "Feature Service"-Treffer, markiert das Ergebnis aber als
    `ambiguous`, damit ein Preflight-Check das laut melden kann.

    `require_owner`/`require_orgid` schränken die Suche auf den bereits
    verifizierten PortWatch-Publisher ein (aus einem zuvor erfolgreich
    aufgelösten Daily-Layer-Item) UND filtern das Ergebnis client-seitig
    erneut nach Owner/OrgId -- so wird nie ein Item aus fremder Organisation
    akzeptiert, selbst wenn die serverseitige q-Restriktion aus irgendeinem
    Grund nicht greift (z.B. die USDA-"Maritime_Ports_Ag_Trade"-Layer, die
    live fälschlich für den Titel "Ports" zurückkam)."""
    q = f'title:"{title}"'
    if require_orgid:
        q += f" orgid:{require_orgid}"
    elif require_owner:
        q += f" owner:{require_owner}"
    if extra_query:
        q += f" {extra_query}"
    res = http_fetch(ARCGIS_SEARCH_URL, params={"q": q, "f": "json", "num": 25})
    data = res.json()
    results = data.get("results", [])
    if require_orgid or require_owner:
        results = [
            r for r in results
            if (not require_orgid or r.get("orgId") == require_orgid)
            and (not require_owner or r.get("owner") == require_owner)
        ]
    if not results:
        raise DiscoveryError(
            f"ArcGIS-Suche ohne (Org-verifizierten) Treffer für title={title!r} "
            f"(require_owner={require_owner!r}, require_orgid={require_orgid!r})"
        )

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
        "owner": chosen.get("owner"),
        "orgId": chosen.get("orgId"),
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


# Referenz-Layer-Key -> zugehöriger Daily-Layer-Key, dessen bereits
# verifiziertes Owner/OrgId die Referenz-Suche einschränkt (item 1 im Fix:
# "Ports"/"Chokepoints" sind zu generische Titel, um sie ungeschützt gegen
# die globale ArcGIS-Suche laufen zu lassen -- live griff das z.B. eine
# USDA-Ag-Trade-Layer für "Ports" ab).
REFERENCE_TO_DAILY_KEY: dict[str, str] = {
    "ports_reference": "ports_daily",
    "chokepoints_reference": "chokepoints_daily",
}


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

    Reihenfolge: zuerst werden die beiden "Daily ..."-Layer aufgelöst (ihre
    Titel sind spezifisch genug, um ohne Owner-Restriktion sicher zu sein).
    Aus deren Item-Metadaten (owner/orgId) wird dann die Suche nach den
    generischen Referenz-Titeln ("Ports"/"Chokepoints") eingeschränkt -- ein
    Treffer aus fremder Organisation wird NIE akzeptiert. Schlägt die
    Org-eingeschränkte Referenz-Suche fehl, wird kein Fallback-Owner
    erraten; stattdessen wird der Referenz-Layer als
    `fallback_to_daily_layer=True` markiert, damit der Konnektor
    Port-/Chokepoint-Id/Name/Land per Distinct-Values-Query direkt aus dem
    Daily-Layer selbst ableitet.
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
    for key in ("ports_daily", "chokepoints_daily"):
        item = _search_arcgis_item(DATASET_TITLES[key], fetch_fn)
        layer = _resolve_layer(item["service_url"], fetch_fn)
        endpoints[key] = {
            "service_url": item["service_url"],
            "layer_id": layer["layer_id"],
            "item_id": item["item_id"],
            "title": item["title"],
            "ambiguous": item["ambiguous"],
            "owner": item.get("owner"),
            "orgId": item.get("orgId"),
        }

    for ref_key, daily_key in REFERENCE_TO_DAILY_KEY.items():
        title = DATASET_TITLES[ref_key]
        require_owner = endpoints[daily_key].get("owner")
        require_orgid = endpoints[daily_key].get("orgId")
        try:
            item = _search_arcgis_item(
                title, fetch_fn, require_owner=require_owner, require_orgid=require_orgid,
            )
            layer = _resolve_layer(item["service_url"], fetch_fn)
            endpoints[ref_key] = {
                "service_url": item["service_url"],
                "layer_id": layer["layer_id"],
                "item_id": item["item_id"],
                "title": item["title"],
                "ambiguous": item["ambiguous"],
                "owner": item.get("owner"),
                "orgId": item.get("orgId"),
                "fallback_to_daily_layer": False,
            }
        except DiscoveryError:
            daily_ep = endpoints[daily_key]
            endpoints[ref_key] = {
                "service_url": daily_ep["service_url"],
                "layer_id": daily_ep["layer_id"],
                "item_id": daily_ep["item_id"],
                "title": daily_ep["title"],
                "ambiguous": False,
                "owner": daily_ep.get("owner"),
                "orgId": daily_ep.get("orgId"),
                "fallback_to_daily_layer": True,
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
    if not isinstance(meta, dict) or "fields" not in meta:
        body_snippet = json.dumps(meta)[:600] if isinstance(meta, (dict, list)) else str(meta)[:600]
        raise SchemaError(
            f"Layer-Metadaten ohne 'fields': {url}",
            diagnostics={"service_url": url, "body_snippet": body_snippet},
        )
    return meta


def _field_names(meta: dict) -> set[str]:
    return {f["name"] for f in meta.get("fields", []) if isinstance(f, dict) and f.get("name")}


def _all_field_names(meta: dict) -> list[str]:
    return sorted(_field_names(meta))


def _field_index(meta: dict) -> dict[str, dict]:
    """lower(Feldname) -> Feld-Metadaten-Dict (name, type, ...)."""
    return {
        f["name"].lower(): f for f in meta.get("fields", [])
        if isinstance(f, dict) and f.get("name")
    }


def _match_alias(field_index: dict[str, dict], aliases: tuple[str, ...]) -> dict | None:
    """Case-insensitives Alias-Matching gegen die tatsächliche Layer-
    Metadata -- rät NIE ein Feld, das keinem dokumentierten Alias entspricht."""
    for alias in aliases:
        f = field_index.get(alias.lower())
        if f is not None:
            return f
    return None


def _layer_diagnostics(meta: dict, service_url: str = "", layer_id: int | str = "") -> dict:
    """Sichere Debug-Infos für ConnectorResult.discovered_ids['diagnostics']
    im Fehlerfall (nie Request-Parameter/Credentials)."""
    return {
        "field_names": _all_field_names(meta),
        "maxRecordCount": meta.get("maxRecordCount"),
        "service_url": f"{service_url}/{layer_id}" if service_url != "" else None,
    }


def resolve_ports_field_schema(meta: dict, service_url: str = "", layer_id: int | str = "") -> dict:
    """Löst die logischen Felder (date/id/name/country/iso3/cargo) case-
    insensitiv gegen die tatsächliche Ports-Layer-Metadata auf. outFields
    wird ausschließlich aus dieser Schnittmenge gebaut -- nie hartcodiert."""
    field_index = _field_index(meta)
    date_f = _match_alias(field_index, FIELD_ALIASES_PORTS["date"])
    id_f = _match_alias(field_index, FIELD_ALIASES_PORTS["portid"])
    name_f = _match_alias(field_index, FIELD_ALIASES_PORTS["portname"])
    country_f = _match_alias(field_index, FIELD_ALIASES_PORTS["country"])
    iso3_f = _match_alias(field_index, FIELD_ALIASES_PORTS["iso3"])
    cargo_fields = sorted(
        f["name"] for lower, f in field_index.items() if lower.startswith(CARGO_PREFIXES_PORTS)
    )
    if date_f is None or id_f is None or name_f is None or not cargo_fields:
        raise SchemaError(
            "SCHEMA_CHANGED: Kern-Felder (date/portid/portname/mind. 1 Cargo-Feld) nicht im "
            f"Ports-Layer gefunden. Vorhandene Felder: {_all_field_names(meta)}",
            diagnostics=_layer_diagnostics(meta, service_url, layer_id),
        )
    return {
        "date_field": date_f["name"], "date_field_type": date_f.get("type", ""),
        "id_field": id_f["name"], "name_field": name_f["name"],
        "country_field": country_f["name"] if country_f else None,
        "iso3_field": iso3_f["name"] if iso3_f else None,
        "cargo_fields": cargo_fields,
    }


def validate_ports_schema(meta: dict) -> list[str]:
    """Rückwärtskompatibler Wrapper: gibt nur die Cargo-Feldnamen zurück."""
    return resolve_ports_field_schema(meta)["cargo_fields"]


def resolve_ports_reference_schema(meta: dict, service_url: str = "", layer_id: int | str = "") -> dict:
    """Löst id/name/country für den Ports-Referenz-Layer auf (separate
    Metadata -- kein Feld wird zwischen Daily- und Referenz-Layer geraten)."""
    field_index = _field_index(meta)
    id_f = _match_alias(field_index, FIELD_ALIASES_PORTS["portid"])
    name_f = _match_alias(field_index, FIELD_ALIASES_PORTS["portname"])
    country_f = _match_alias(field_index, FIELD_ALIASES_PORTS["country"])
    if id_f is None or name_f is None:
        raise SchemaError(
            "SCHEMA_CHANGED: Kern-Felder (portid/portname) nicht im Ports-Referenz-Layer gefunden. "
            f"Vorhandene Felder: {_all_field_names(meta)}",
            diagnostics=_layer_diagnostics(meta, service_url, layer_id),
        )
    return {
        "id_field": id_f["name"], "name_field": name_f["name"],
        "country_field": country_f["name"] if country_f else None,
    }


def resolve_chokepoints_field_schema(meta: dict, service_url: str = "", layer_id: int | str = "") -> dict:
    """Löst die logischen Felder (date/id/name/count) case-insensitiv gegen
    die tatsächliche Chokepoints-Layer-Metadata auf."""
    field_index = _field_index(meta)
    date_f = _match_alias(field_index, FIELD_ALIASES_CHOKEPOINTS["date"])
    id_f = _match_alias(field_index, FIELD_ALIASES_CHOKEPOINTS["chokepointid"])
    name_f = _match_alias(field_index, FIELD_ALIASES_CHOKEPOINTS["chokepointname"])
    count_fields = sorted(
        f["name"] for lower, f in field_index.items() if lower.startswith(CHOKEPOINT_COUNT_PREFIXES)
    )
    if date_f is None or id_f is None or name_f is None or not count_fields:
        raise SchemaError(
            "SCHEMA_CHANGED: Kern-Felder (date/id/name/mind. 1 Zähl-Feld) nicht im "
            f"Chokepoints-Layer gefunden. Vorhandene Felder: {_all_field_names(meta)}",
            diagnostics=_layer_diagnostics(meta, service_url, layer_id),
        )
    return {
        "date_field": date_f["name"], "date_field_type": date_f.get("type", ""),
        "id_field": id_f["name"], "name_field": name_f["name"], "count_fields": count_fields,
    }


def resolve_chokepoints_reference_schema(meta: dict, service_url: str = "", layer_id: int | str = "") -> dict:
    field_index = _field_index(meta)
    id_f = _match_alias(field_index, FIELD_ALIASES_CHOKEPOINTS["chokepointid"])
    name_f = _match_alias(field_index, FIELD_ALIASES_CHOKEPOINTS["chokepointname"])
    if id_f is None or name_f is None:
        raise SchemaError(
            "SCHEMA_CHANGED: Kern-Felder (id/name) nicht im Chokepoints-Referenz-Layer gefunden. "
            f"Vorhandene Felder: {_all_field_names(meta)}",
            diagnostics=_layer_diagnostics(meta, service_url, layer_id),
        )
    return {"id_field": id_f["name"], "name_field": name_f["name"]}


def validate_chokepoints_schema(meta: dict) -> dict:
    """Rückwärtskompatibler Wrapper ohne date_field im Rückgabewert."""
    schema = resolve_chokepoints_field_schema(meta)
    return {"id_field": schema["id_field"], "name_field": schema["name_field"],
            "count_fields": schema["count_fields"]}


# --------------------------------------------------------------------------
# Paginierte Query gegen einen FeatureServer-Layer
# --------------------------------------------------------------------------

MAX_PAGES_SAFETY = 500


def build_where_clause(start: date, end: date, date_field: str = "date",
                        date_field_type: str = "esriFieldTypeDate") -> str:
    """Baut die WHERE-Klausel passend zum tatsächlichen ArcGIS-Feldtyp:
    esriFieldTypeDate -> TIMESTAMP-Literal (epoch-ms intern), sonst
    (z.B. esriFieldTypeString mit ISO-Datum) -> einfacher String-Vergleich.
    Nie den falschen Vergleichsoperator für den Feldtyp erraten."""
    if date_field_type and date_field_type != "esriFieldTypeDate":
        return f"{date_field} >= '{start.isoformat()}' AND {date_field} <= '{end.isoformat()}'"
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
        if isinstance(data, dict) and "error" in data:
            raise FetchError(
                f"ArcGIS-Query-Fehler: {data['error']} (outFields={out}, service_url={service_url}/{layer_id})"
            )
        feats = data.get("features", [])
        all_features.extend(feats)
        exceeded = bool(data.get("exceededTransferLimit"))
        if not exceeded or not feats:
            break
        offset += len(feats)
    else:  # pragma: no cover - Sicherheitsnetz gegen Endlos-Pagination
        raise FetchError(f"Pagination-Limit ({MAX_PAGES_SAFETY} Seiten) erreicht für {query_url}")
    return all_features


def query_distinct_values(
    service_url: str,
    layer_id: int,
    out_fields: Iterable[str],
    http_fetch: FetchFn,
    max_record_count: int | None = None,
) -> list[dict]:
    """Fallback-Ableitung von Referenzdaten (item 1 im Fix): fragt distinkte
    Kombinationen von `out_fields` (typischerweise id/name/land) direkt aus
    dem Daily-Layer ab (`returnDistinctValues=true`), statt eines dedizierten
    Referenz-Layers. Wird genutzt, wenn der Referenz-Layer nicht org-
    verifiziert gefunden werden kann. Paginiert wie `query_features`."""
    out = ",".join(out_fields)
    page_size = max_record_count or 2000
    query_url = f"{service_url}/{layer_id}/query"
    offset = 0
    all_features: list[dict] = []
    for _ in range(MAX_PAGES_SAFETY):
        params = {
            "where": "1=1",
            "outFields": out,
            "returnDistinctValues": "true",
            "returnGeometry": "false",
            "resultOffset": offset,
            "resultRecordCount": page_size,
            "f": "json",
        }
        res = http_fetch(query_url, params=params)
        data = res.json()
        if isinstance(data, dict) and "error" in data:
            raise FetchError(
                f"ArcGIS-Distinct-Query-Fehler: {data['error']} (outFields={out}, "
                f"service_url={service_url}/{layer_id})"
            )
        feats = data.get("features", [])
        all_features.extend(feats)
        exceeded = bool(data.get("exceededTransferLimit"))
        if not exceeded or not feats:
            break
        offset += len(feats)
    else:  # pragma: no cover - Sicherheitsnetz gegen Endlos-Pagination
        raise FetchError(f"Distinct-Pagination-Limit ({MAX_PAGES_SAFETY} Seiten) erreicht für {query_url}")
    # Dedupliziert nach den Attribut-Werten selbst -- ArcGIS liefert
    # `returnDistinctValues` nicht immer serverseitig verlustfrei über die
    # Pagination hinweg.
    seen: set[tuple] = set()
    deduped: list[dict] = []
    for feat in all_features:
        attrs = feat.get("attributes", feat)
        key = tuple(sorted(attrs.items()))
        if key not in seen:
            seen.add(key)
            deduped.append(feat if "attributes" in feat else {"attributes": attrs})
    return deduped


def incremental_window(now: datetime, days: int = INCREMENTAL_WINDOW_DAYS) -> tuple[date, date]:
    end = now.date()
    start = end - timedelta(days=days)
    return start, end


# --------------------------------------------------------------------------
# Name-Normalisierung & Universe-Resolution (nie IDs erfinden)
# --------------------------------------------------------------------------

_PAREN_RE = re.compile(r"[()\-–,.]")
_WS_RE = re.compile(r"\s+")


_STRAIT_OF_RE = re.compile(r"^strait of (.+)$")


def normalize_name(name: str) -> str:
    if not name:
        return ""
    s = name.lower()
    s = _PAREN_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    for prefix in ("port of ", "the "):
        if s.startswith(prefix):
            s = s[len(prefix):]
    # "Strait of X" <-> "X Strait" ist reine Wortstellung, kein Alias --
    # wird direkt normalisiert (item 2 im Fix), spelling-Varianten
    # (Bosporus/Bosphorus etc.) laufen weiterhin über die dokumentierte
    # Alias-Tabelle in config/port_universe.yaml.
    m = _STRAIT_OF_RE.match(s)
    if m:
        s = f"{m.group(1)} strait"
    return s


def close_name_matches(name: str, candidate_names: Iterable[str], k: int = 3) -> list[str]:
    """Top-`k` Kandidaten aus `candidate_names` nach normalisierter
    String-Ähnlichkeit zu `name` -- rein diagnostisch (item 3 im Fix), wird
    NIE zum automatischen Matchen benutzt, nur zum Anzeigen in Diagnostics."""
    target = normalize_name(name)
    scored = sorted(
        {c for c in candidate_names if c},
        key=lambda c: difflib.SequenceMatcher(None, target, normalize_name(c)).ratio(),
        reverse=True,
    )
    return scored[:k]


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
    alias_map: dict[str, list[str]] | None = None,
) -> list[ResolvedEntity]:
    """Matched eine gewünschte Liste {name, country?} gegen Referenz-Layer-
    Features (exakter, normalisierter Name; bei mehreren Treffern zusätzlich
    nach Land gefiltert). Kein Fuzzy-Raten -- unklare Fälle werden als
    'ambiguous' bzw. 'no_match' zurückgegeben, nie mit einer geratenen ID.

    `alias_map` (kuratierter Name -> Liste dokumentierter Alias-Namen, siehe
    config/port_universe.yaml `chokepoint_aliases`) wird NUR gegen Namen
    gematcht, die tatsächlich im offiziellen Referenz-Layer vorkommen --
    es wird nie eine ID vergeben, ohne dass ein normalisierter Name (Original
    oder dokumentierter Alias) exakt in `reference_features` gefunden wurde."""
    alias_map = alias_map or {}
    index: dict[str, list[dict]] = {}
    for feat in reference_features:
        attrs = feat.get("attributes", feat)
        norm = normalize_name(str(attrs.get(name_field, "")))
        index.setdefault(norm, []).append(attrs)

    results: list[ResolvedEntity] = []
    for item in wanted:
        name = item["name"]
        country = item.get("country")
        candidate_norms = [normalize_name(name)] + [
            normalize_name(a) for a in alias_map.get(name, [])
        ]
        matches: list[dict] = []
        norm = candidate_norms[0]
        for cand_norm in candidate_norms:
            matches = index.get(cand_norm, [])
            if matches:
                norm = cand_norm
                break
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


def load_port_aliases(universe: dict) -> dict[str, list[str]]:
    """Dokumentierte offizielle Hafennamen aus config/port_universe.yaml
    (`port_aliases`), z.B. "Busan" -> "Pusan". Nur Namen, die im Live-
    Preflight im offiziellen PortWatch-Layer beobachtet wurden; Matching
    weiterhin ausschließlich gegen tatsächlich vorhandene Layer-Namen."""
    raw = universe.get("port_aliases") or {}
    return {k: list(v) for k, v in raw.items()}


def load_chokepoint_aliases(universe: dict) -> dict[str, list[str]]:
    """Dokumentierte Namens-Alias-Tabelle aus config/port_universe.yaml
    (`chokepoint_aliases`) -- z.B. Schreibvarianten wie Bosporus/Bosphorus,
    die reine Normalisierung nicht auflösen kann. Wird NIE erraten, nur
    gegen tatsächlich im offiziellen Chokepoints-Layer vorhandene Namen
    gematcht (siehe resolve_entities_by_name)."""
    raw = universe.get("chokepoint_aliases") or {}
    return {k: list(v) for k, v in raw.items()}


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
    schema: dict,
    parser_version: str,
    is_backfill: bool,
) -> list[Observation]:
    cargo_fields = schema["cargo_fields"]
    obs_time = _arcgis_date_to_utc(attrs.get(schema["date_field"]))
    if obs_time is None:
        return []
    port_id = str(attrs.get(schema["id_field"]))
    port_name = attrs.get(schema["name_field"])
    country = (
        (attrs.get(schema["country_field"]) if schema.get("country_field") else None)
        or (attrs.get(schema["iso3_field"]) if schema.get("iso3_field") else None)
    )
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
    obs_time = _arcgis_date_to_utc(attrs.get(schema.get("date_field", "date")))
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
            schema = resolve_ports_field_schema(meta, ep["service_url"], ep["layer_id"])
            max_rc = meta.get("maxRecordCount", 2000)

            start, end = (BACKFILL_START, now.date()) if backfill else incremental_window(now)
            where = build_where_clause(start, end, date_field=schema["date_field"],
                                        date_field_type=schema["date_field_type"])
            # outFields = ausschließlich die Schnittmenge aus gewünschten
            # logischen Feldern und tatsächlicher Layer-Metadata (nie
            # hartcodierte Feldnamen an ArcGIS senden -> vermeidet den
            # 'outFields parameter is invalid' 400er).
            out_fields = sorted({
                schema["date_field"], schema["id_field"], schema["name_field"],
                *( [schema["country_field"]] if schema.get("country_field") else [] ),
                *( [schema["iso3_field"]] if schema.get("iso3_field") else [] ),
                *schema["cargo_fields"],
            })
            raw_features = query_features(
                ep["service_url"], ep["layer_id"], where, out_fields, self._fetch,
                order_by=f"{schema['date_field']} ASC", max_record_count=max_rc,
            )

            if ref_ep.get("fallback_to_daily_layer"):
                # Ports-Referenz-Layer konnte nicht org-verifiziert gefunden
                # werden (item 1 im Fix) -> Id/Name/Land werden per
                # Distinct-Values-Query direkt aus dem Daily-Ports-Layer
                # abgeleitet (der Layer trägt portid & portname ohnehin).
                ref_schema = {
                    "id_field": schema["id_field"], "name_field": schema["name_field"],
                    "country_field": schema.get("country_field"),
                }
                ref_out_fields = [ref_schema["id_field"], ref_schema["name_field"]]
                if ref_schema.get("country_field"):
                    ref_out_fields.append(ref_schema["country_field"])
                ref_features = query_distinct_values(
                    ep["service_url"], ep["layer_id"], ref_out_fields, self._fetch,
                    max_record_count=max_rc,
                )
            else:
                ref_meta = fetch_layer_metadata(ref_ep["service_url"], ref_ep["layer_id"], self._fetch)
                ref_schema = resolve_ports_reference_schema(ref_meta, ref_ep["service_url"], ref_ep["layer_id"])
                ref_out_fields = [ref_schema["id_field"], ref_schema["name_field"]]
                if ref_schema.get("country_field"):
                    ref_out_fields.append(ref_schema["country_field"])
                ref_features = query_features(
                    ref_ep["service_url"], ref_ep["layer_id"], "1=1",
                    ref_out_fields, self._fetch,
                    max_record_count=ref_meta.get("maxRecordCount", 2000),
                )
            universe = load_port_universe(self._universe_path)
            wanted = flatten_port_universe(universe)
            resolved = resolve_entities_by_name(
                wanted, ref_features, ref_schema["id_field"], ref_schema["name_field"],
                country_field=ref_schema.get("country_field"),
                alias_map=load_port_aliases(universe),
            )
            discovered_ids = {r.name: r.entity_id for r in resolved if r.status == "resolved"}
            unresolved = {r.name: r.status for r in resolved if r.status != "resolved"}
            available_ports = sorted({
                str(f.get("attributes", f).get(ref_schema["name_field"], ""))
                for f in ref_features
            } - {""})
            diagnostics = {
                "available_ports": available_ports[:60],
                "n_available_ports": len(available_ports),
                "ports_reference_fallback_to_daily_layer": bool(ref_ep.get("fallback_to_daily_layer")),
                "unresolved_ports_close_matches": {
                    name: close_name_matches(name, available_ports) for name in unresolved
                },
            }

            retrieved_at = now
            observations: list[Observation] = []
            for feat in raw_features:
                observations.extend(build_port_observations(
                    feat.get("attributes", {}), retrieved_at, self.source_id, self.dataset,
                    schema, self.parser_version, is_backfill=backfill,
                ))

            raw = [self._raw_record(f"{ep['service_url']}/{ep['layer_id']}/query", where, retrieved_at)]
            latest = max((o.observation_time for o in observations), default=None)
            status = SourceStatus.WARN if unresolved else SourceStatus.PASS
            return ConnectorResult(
                source_id=self.source_id, status=status, observations=observations, raw=raw,
                message=f"{len(observations)} observations; {len(unresolved)} unresolved ports: {sorted(unresolved)}",
                latest_observation_time=latest,
                discovered_ids={
                    "ports": discovered_ids, "unresolved_ports": unresolved,
                    "diagnostics": diagnostics,
                },
            )
        except SchemaError as e:
            return ConnectorResult(source_id=self.source_id, status=SourceStatus.SCHEMA_CHANGED,
                                    message=str(e), discovered_ids={"diagnostics": e.diagnostics})
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
            schema = resolve_chokepoints_field_schema(meta, ep["service_url"], ep["layer_id"])
            max_rc = meta.get("maxRecordCount", 2000)

            start, end = (BACKFILL_START, now.date()) if backfill else incremental_window(now)
            where = build_where_clause(start, end, date_field=schema["date_field"],
                                        date_field_type=schema["date_field_type"])
            out_fields = sorted({schema["date_field"], schema["id_field"], schema["name_field"]}
                                 | set(schema["count_fields"]))
            raw_features = query_features(
                ep["service_url"], ep["layer_id"], where, out_fields, self._fetch,
                order_by=f"{schema['date_field']} ASC", max_record_count=max_rc,
            )

            if ref_ep.get("fallback_to_daily_layer"):
                # Chokepoints-Referenz-Layer nicht org-verifiziert gefunden
                # -> Id/Name per Distinct-Values-Query aus dem
                # Daily-Chokepoints-Layer selbst ableiten.
                ref_schema = {"id_field": schema["id_field"], "name_field": schema["name_field"]}
                ref_features = query_distinct_values(
                    ep["service_url"], ep["layer_id"],
                    [ref_schema["id_field"], ref_schema["name_field"]], self._fetch,
                    max_record_count=max_rc,
                )
            else:
                ref_meta = fetch_layer_metadata(ref_ep["service_url"], ref_ep["layer_id"], self._fetch)
                ref_schema = resolve_chokepoints_reference_schema(ref_meta, ref_ep["service_url"], ref_ep["layer_id"])
                ref_features = query_features(
                    ref_ep["service_url"], ref_ep["layer_id"], "1=1",
                    [ref_schema["id_field"], ref_schema["name_field"]], self._fetch,
                    max_record_count=ref_meta.get("maxRecordCount", 2000),
                )
            universe = load_port_universe(self._universe_path)
            wanted = flatten_chokepoint_universe(universe)
            alias_map = load_chokepoint_aliases(universe)
            resolved = resolve_entities_by_name(
                wanted, ref_features, ref_schema["id_field"], ref_schema["name_field"], country_field=None,
                alias_map=alias_map,
            )
            discovered_ids = {r.name: r.entity_id for r in resolved if r.status == "resolved"}
            unresolved = {r.name: r.status for r in resolved if r.status != "resolved"}
            available_chokepoints = sorted({
                str(f.get("attributes", f).get(ref_schema["name_field"], ""))
                for f in ref_features
            } - {""})
            diagnostics = {
                "available_chokepoints": available_chokepoints[:60],
                "n_available_chokepoints": len(available_chokepoints),
                "chokepoints_reference_fallback_to_daily_layer": bool(ref_ep.get("fallback_to_daily_layer")),
                "unresolved_chokepoints_close_matches": {
                    name: close_name_matches(name, available_chokepoints) for name in unresolved
                },
            }

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
                discovered_ids={
                    "chokepoints": discovered_ids, "unresolved_chokepoints": unresolved,
                    "diagnostics": diagnostics,
                },
            )
        except SchemaError as e:
            return ConnectorResult(source_id=self.source_id, status=SourceStatus.SCHEMA_CHANGED,
                                    message=str(e), discovered_ids={"diagnostics": e.diagnostics})
        except DiscoveryError as e:
            return ConnectorResult(source_id=self.source_id, status=SourceStatus.FAIL, message=str(e))
        except FetchError as e:
            return ConnectorResult(source_id=self.source_id, status=SourceStatus.FAIL, message=str(e))


CONNECTORS: dict[str, type[Connector]] = {
    PortWatchPortsConnector.source_id: PortWatchPortsConnector,
    PortWatchChokepointsConnector.source_id: PortWatchChokepointsConnector,
}

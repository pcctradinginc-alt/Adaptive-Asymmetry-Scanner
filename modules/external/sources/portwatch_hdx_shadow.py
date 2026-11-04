"""Read-only OCHA/HDX mirror check for IMF PortWatch; NOT a production fallback.

OCHA-DAP/hdx-portwatch-viz/download_data.py documents these HDX resource IDs.
Ports are an eight-country subset, NOT the global 2,065-port universe. This
module never emits ConnectorResult/Observation, updates source health, or feeds
trading/risk decisions. Unverified metadata, redirects, empty or old CSVs fail.
"""
from __future__ import annotations

import csv
import hashlib
import io
import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from urllib.parse import urljoin, urlsplit

import requests

HDX_API = "https://data.humdata.org/api/3/action"
ALLOWED_HOSTS = frozenset({"data.humdata.org"})
USER_AGENT = "AdaptiveAsymmetryScanner/1.0 (research; IMF-PortWatch-HDX-shadow)"
MAX_CSV_BYTES = 96 * 1024 * 1024
MAX_ROWS = 1_000_000
MAX_AGE_DAYS = 21
REDIRECT_LIMIT = 3


@dataclass(frozen=True)
class ResourceSpec:
    name: str
    resource_id: str
    filename: str
    family: str
    metric: str


# Independently documented identifiers, never guessed from HDX search ranking.
RESOURCES = {
    "djibouti": ResourceSpec("djibouti", "96a7bdea-8297-49b7-ad6c-adb1d1f48e6d", "djibouti-daily-port-activity-data-and-shipment-estimates.csv", "ports", "portcalls"),
    "kenya": ResourceSpec("kenya", "e0bde7c9-2cb7-4eab-ae71-3e2c2e225207", "kenya-daily-port-activity-data-and-shipment-estimates.csv", "ports", "portcalls"),
    "lebanon": ResourceSpec("lebanon", "8d1d8984-fd6d-4f8e-af84-32b0ab8bacd3", "lebanon-daily-port-activity-data-and-shipment-estimates.csv", "ports", "portcalls"),
    "mozambique": ResourceSpec("mozambique", "9c88af9b-9868-44a6-af3b-2df6db2e8c58", "mozambique-daily-port-activity-data-and-shipment-estimates.csv", "ports", "portcalls"),
    "oman": ResourceSpec("oman", "381b41b9-12c8-4bb3-b056-65ef6ddf1c6b", "oman-daily-port-activity-data-and-shipment-estimates.csv", "ports", "portcalls"),
    "somalia": ResourceSpec("somalia", "8bf8a420-56b3-492a-96b0-e8acb7f14c04", "somalia-daily-port-activity-data-and-shipment-estimates.csv", "ports", "portcalls"),
    "sudan": ResourceSpec("sudan", "6ce7a041-ce33-4e83-bc66-7fd1d1fa4af8", "sudan-daily-port-activity-data-and-shipment-estimates.csv", "ports", "portcalls"),
    "yemen": ResourceSpec("yemen", "5b130c44-8c57-485c-b1bf-6ab07aa12ce9", "yemen-daily-port-activity-data-and-shipment-estimates.csv", "ports", "portcalls"),
    "chokepoints": ResourceSpec("chokepoints", "cbd17f06-8ab2-4ba4-a666-f7a4bcc9ea3f", "daily-chokepoint-transit-calls-and-shipment-volume-estimates.csv", "chokepoints", "n_total"),
}
# Safe, bounded default canaries; the complete 8-country mirror is opt-in.
DEFAULT_KEYS = ("djibouti", "chokepoints")


class HDXShadowError(ValueError):
    """Input/data/provenance failure; shadow status FAIL, never bypass Safe Mode."""


def _validate_url(url: str) -> str:
    if not isinstance(url, str) or len(url) > 2048:
        raise HDXShadowError("UNTRUSTED_DOWNLOAD_URL")
    try:
        p = urlsplit(url)
        if (p.scheme != "https" or p.hostname not in ALLOWED_HOSTS
                or p.username is not None or p.password is not None
                or p.port not in (None, 443) or not p.path.startswith("/")):
            raise HDXShadowError("UNTRUSTED_DOWNLOAD_URL")
    except ValueError as exc:
        raise HDXShadowError("UNTRUSTED_DOWNLOAD_URL") from exc
    return url


def _safe_get(session, url: str, **kwargs):
    """Restrict every hop, not only the initial HDX URL (SSRF/redirect guard)."""
    for _ in range(REDIRECT_LIMIT + 1):
        _validate_url(url)
        response = session.get(
            url, allow_redirects=False,
            headers={"User-Agent": USER_AGENT, "Accept": "text/csv,application/json,*/*"},
            **kwargs,
        )
        if response.status_code in (301, 302, 303, 307, 308):
            location = response.headers.get("Location", "")
            response.close()
            if not location:
                raise HDXShadowError("REDIRECT_WITHOUT_LOCATION")
            url = _validate_url(urljoin(url, location))
            continue
        response.raise_for_status()
        return response
    raise HDXShadowError("REDIRECT_LIMIT_EXCEEDED")


def _ckan_result(session, action: str, resource_id: str) -> dict:
    response = _safe_get(session, f"{HDX_API}/{action}",
                         params={"id": resource_id}, timeout=(10, 30))
    try:
        try:
            payload = response.json()
        except (ValueError, TypeError) as exc:
            raise HDXShadowError(f"HDX_{action.upper()}_INVALID_JSON") from exc
    finally:
        response.close()
    if (not isinstance(payload, dict) or payload.get("success") is not True
            or not isinstance(payload.get("result"), dict)):
        raise HDXShadowError(f"HDX_{action.upper()}_INVALID_RESPONSE")
    return payload["result"]


def _resolve_resource(session, spec: ResourceSpec, package_cache: dict) -> tuple[str, dict]:
    meta = _ckan_result(session, "resource_show", spec.resource_id)
    if meta.get("id") != spec.resource_id:
        raise HDXShadowError("RESOURCE_ID_MISMATCH")
    if str(meta.get("format") or "").strip().lower() != "csv":
        raise HDXShadowError("RESOURCE_NOT_CSV")
    name = str(meta.get("name") or "").lower()
    if (not name or "daily" not in name
            or ("chokepoint" if spec.family == "chokepoints" else "port") not in name
            or (spec.family == "ports" and spec.name not in name)):
        raise HDXShadowError("RESOURCE_NAME_MISMATCH")
    package_id = meta.get("package_id")
    if not isinstance(package_id, str) or not package_id:
        raise HDXShadowError("RESOURCE_PACKAGE_MISSING")
    if package_id not in package_cache:
        package_cache[package_id] = _ckan_result(session, "package_show", package_id)
    pkg = package_cache[package_id]
    if pkg.get("id") != package_id:
        raise HDXShadowError("RESOURCE_PACKAGE_ID_MISMATCH")
    org = pkg.get("organization") or {}
    if not isinstance(org, dict) or org.get("name") != "portwatch":
        raise HDXShadowError("HDX_PUBLISHER_MISMATCH")
    if spec.resource_id not in {x.get("id") for x in pkg.get("resources", [])
                               if isinstance(x, dict)}:
        raise HDXShadowError("RESOURCE_PACKAGE_LINK_MISMATCH")
    return _validate_url(meta.get("url")), {"publisher": "HDX/PortWatch", "package_id": package_id,
                                           "resource_id": spec.resource_id, "name": meta["name"]}


def _read_limited(session, url: str, max_bytes: int) -> bytes:
    response = _safe_get(session, url, timeout=(10, 60), stream=True)
    data = bytearray()
    try:
        for chunk in response.iter_content(chunk_size=65536):
            if not chunk:
                continue
            if len(data) + len(chunk) > max_bytes:
                raise HDXShadowError("CSV_SIZE_LIMIT")
            data.extend(chunk)
    finally:
        response.close()
    if not data:
        raise HDXShadowError("EMPTY_CSV_DOWNLOAD")
    # Reject HTML error pages erroneously returned with HTTP 200.
    if bytes(data[:256]).lstrip().lower().startswith((b"<!doctype html", b"<html", b"{\"error\"")):
        raise HDXShadowError("NOT_CSV_PAYLOAD")
    return bytes(data)


def _parse_day(raw: str) -> date:
    value = str(raw or "").strip()
    if not value:
        raise HDXShadowError("CSV_INVALID_DATE")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError as exc:
        raise HDXShadowError("CSV_INVALID_DATE") from exc


def _validate_csv(data: bytes, spec: ResourceSpec, now: datetime) -> dict:
    try:
        contents = data.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as exc:
        raise HDXShadowError("CSV_NOT_UTF8") from exc
    reader = csv.DictReader(io.StringIO(contents, newline=""))
    fields = set(reader.fieldnames or [])
    required = {"date", "portname", spec.metric}
    if (not required.issubset(fields) or None in fields or len(fields) > 256
            or len(fields) != len(reader.fieldnames or [])):
        raise HDXShadowError("CSV_SCHEMA_MISMATCH")
    count = 0
    latest = None
    unique_days = set()
    for row in reader:
        count += 1
        if count > MAX_ROWS:
            raise HDXShadowError("CSV_ROW_LIMIT")
        if None in row or not isinstance(row.get("portname"), str) or not row["portname"].strip():
            raise HDXShadowError("CSV_INVALID_ROW")
        observed = _parse_day(row["date"])
        if observed > now.date():
            raise HDXShadowError("CSV_FUTURE_OBSERVATION")
        try:
            n = float(row[spec.metric])
            if not math.isfinite(n) or n < 0:
                raise ValueError()
        except (ValueError, TypeError) as exc:
            raise HDXShadowError("CSV_INVALID_METRIC") from exc
        unique_days.add(observed)
        latest = observed if latest is None or observed > latest else latest
    if count < 7 or len(unique_days) < 7 or latest is None:
        raise HDXShadowError("CSV_INSUFFICIENT_HISTORY")
    age = (now.date() - latest).days
    if age > MAX_AGE_DAYS:
        raise HDXShadowError("CSV_STALE")
    return {"rows": count, "unique_days": len(unique_days), "latest_observation_date": latest.isoformat(),
            "data_age_days": age, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "metric": spec.metric}


def check_resources(now: datetime | None = None, keys=DEFAULT_KEYS, session=None) -> dict:
    """Read-only shadow report. No method here produces a primary-source PASS."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    now = now.astimezone(timezone.utc)
    selected = tuple(keys)
    if not selected or len(set(selected)) != len(selected) or any(k not in RESOURCES for k in selected):
        raise ValueError("unknown, empty or duplicate resource selection")
    owned_session = session is None
    session = session or requests.Session()
    cache = {}
    report = {"checked_at": now.isoformat(), "mode": "SHADOW_ONLY", "source": "OCHA/HDX IMF PortWatch",
              "production_fallback_enabled": False, "safe_mode_override": False,
              "ports_global_coverage": False, "resources": {}}
    try:
        for key in selected:
            spec = RESOURCES[key]
            entry = {"status": "FAIL", "resource_id": spec.resource_id, "family": spec.family,
                     "scope": "COUNTRY_ONLY" if spec.family == "ports" else "GLOBAL_CHOKEPOINTS"}
            try:
                url, provenance = _resolve_resource(session, spec, cache)
                payload = _read_limited(session, url, MAX_CSV_BYTES)
                details = _validate_csv(payload, spec, now)
                entry.update({"status": "PASS", "provenance": provenance, **details})
            except (HDXShadowError, requests.RequestException, UnicodeError, csv.Error, OSError) as exc:
                entry["error"] = (str(exc) if isinstance(exc, HDXShadowError)
                                  else f"HTTP_OR_IO_{type(exc).__name__}")
            report["resources"][key] = entry
    finally:
        if owned_session:
            session.close()
    report["status"] = "PASS_SHADOW" if all(
        x["status"] == "PASS" for x in report["resources"].values()) else "FAIL_SHADOW"
    return report

"""
modules/external/registry.py – Quellen-Registry, Gating, Health, Readiness.

Lädt config/external_sources/*.yaml (je Datenfamilie von den Konnektor-Agenten
gepflegt), entdeckt Konnektoren via pkgutil in modules.external.sources und
setzt die Zugriffsregeln durch:
  - license_status == REVIEW_REQUIRED ODER status_override gesetzt
    → NIE automatisch abgerufen/archiviert
  - requires_auth UND Env-Var fehlt → AUTH_MISSING (kein Call)

SourceHealth wird nach health/source_health.json persistiert; DataReadiness
wird ausschließlich aus Schema/PIT/Provenienz/Frische/Coverage/Zeitspanne/
Beobachtungszahl abgeleitet — NIE aus Returns/P&L.
"""

from __future__ import annotations

import re

import importlib
import pkgutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from modules.external.pit import ensure_utc, utc_now
from modules.external.sources.base import Connector, SourceStatus

DEFAULT_SOURCES_DIR = "config/external_sources"

REQUIRED_FIELDS = (
    "source_id", "display_name", "family", "authority", "official_homepage",
    "machine_endpoint", "access_method", "requires_auth", "auth_env_variable",
    "license_reference", "frequency", "expected_update_cadence",
    "supports_history", "supports_vintages", "supports_release_time",
    "pit_quality", "enabled", "criticality", "license_status",
)


class RegistryError(ValueError):
    """Ungültiger Eintrag in config/external_sources/*.yaml."""


class DataReadiness(str, Enum):
    DISABLED            = "DISABLED"
    COLLECTING          = "COLLECTING"
    SCHEMA_VALIDATED    = "SCHEMA_VALIDATED"
    PIT_VALIDATED       = "PIT_VALIDATED"
    EXPLORATORY_READY   = "EXPLORATORY_READY"
    CHALLENGER_READY    = "CHALLENGER_READY"
    PRODUCTION_ELIGIBLE = "PRODUCTION_ELIGIBLE"
    DEGRADED            = "DEGRADED"
    BLOCKED             = "BLOCKED"


# ── Konfigurations-Loader ────────────────────────────────────────────────────

def load_source_configs(sources_dir: str | Path = DEFAULT_SOURCES_DIR) -> dict[str, dict]:
    """Lädt und validiert alle config/external_sources/*.yaml. Jede Datei ist
    eine Liste von Quellen-Dicts. Fehlt der Ordner/ist er leer → {}."""
    d = Path(sources_dir)
    out: dict[str, dict] = {}
    if not d.exists():
        return out
    for path in sorted(d.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text()) or []
        if isinstance(raw, dict):
            raw = raw.get("sources", [])
        for entry in raw:
            _validate_source_entry(entry, path)
            sid = entry["source_id"]
            if sid in out:
                raise RegistryError(f"Doppelte source_id '{sid}' (auch in {path})")
            out[sid] = dict(entry)
    return out


def _validate_source_entry(entry: dict, path: Path) -> None:
    if not isinstance(entry, dict):
        raise RegistryError(f"Ungültiger Eintrag (kein Dict) in {path}")
    missing = [f for f in REQUIRED_FIELDS if f not in entry]
    if missing:
        raise RegistryError(
            f"{path}: Quelle '{entry.get('source_id', '?')}' fehlen Pflichtfelder: {missing}"
        )
    if entry["license_status"] not in ("OK", "REVIEW_REQUIRED"):
        raise RegistryError(
            f"{path}: license_status muss OK oder REVIEW_REQUIRED sein, war "
            f"{entry['license_status']!r}"
        )
    if entry["criticality"] not in ("low", "medium", "high"):
        raise RegistryError(
            f"{path}: criticality muss low/medium/high sein, war {entry['criticality']!r}"
        )


def discover_connectors(package: str = "modules.external.sources") -> dict[str, type]:
    """Importiert jedes Modul in modules.external.sources und sammelt dessen
    CONNECTORS: dict[source_id, Connector-Subklasse] ein."""
    pkg = importlib.import_module(package)
    out: dict[str, type] = {}
    for modinfo in pkgutil.iter_modules(pkg.__path__, prefix=f"{package}."):
        if modinfo.name.endswith(".base"):
            continue
        mod = importlib.import_module(modinfo.name)
        connectors = getattr(mod, "CONNECTORS", None)
        if not connectors:
            continue
        for sid, cls in connectors.items():
            if sid in out:
                raise RegistryError(f"Doppelte CONNECTORS-source_id '{sid}' in {modinfo.name}")
            out[sid] = cls
    return out


# ── Gating ────────────────────────────────────────────────────────────────────

def gate_source(source_cfg: dict) -> tuple[bool, str]:
    """Prüft License/Override/Auth-Gates. Rückgabe (fetchable, reason)."""
    if source_cfg.get("status_override"):
        return False, f"status_override={source_cfg['status_override']}"
    if source_cfg.get("license_status") == "REVIEW_REQUIRED":
        return False, "license_status=REVIEW_REQUIRED"
    if not source_cfg.get("enabled", False):
        return False, "disabled"
    if source_cfg.get("requires_auth"):
        env_var = source_cfg.get("auth_env_variable")
        import os
        if not env_var or not os.environ.get(env_var):
            return False, "AUTH_MISSING"
    return True, "ok"


# ── SourceHealth ─────────────────────────────────────────────────────────────

@dataclass
class SourceHealth:
    source_id: str
    last_attempt: str | None = None
    last_success: str | None = None
    status: str = SourceStatus.PASS.value
    latest_observation: str | None = None
    latest_release: str | None = None
    retrieval_lag: float | None = None       # Sekunden: retrieved_at - available_at
    expected_cadence: str | None = None
    staleness: str = "UNKNOWN"                # FRESH | STALE | UNKNOWN
    missingness: float = 0.0
    parse_failures: int = 0
    revision_count: int = 0
    duplicate_count: int = 0
    rows_ingested: int = 0
    bytes_downloaded: int = 0
    consecutive_failures: int = 0
    message: str = ""
    criticality: str = "low"
    auth_optional: bool = False   # optionale Zugangsdaten: AUTH_MISSING ohne Alert-Mail
    pit_integrity_failures: int = 0
    archive_error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SourceHealth":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# Maximales Alter der jüngsten Beobachtung je Frequenz (Referenzperiode +
# Publikationsverzug). EINE Quelle der Wahrheit für Source-Health UND die
# Feature-Frische in modules/external/context.py.
MAX_AGE_DAYS_BY_FREQUENCY = {"hourly": 3, "daily": 21, "monthly": 150, "monthly_lagged": 240,
                             "quarterly": 400, "annual": 800}


def parse_cadence(cadence: str) -> timedelta:
    """'7d' / '6h' / '2w' / ISO-8601 'P1D' / 'P7D' / 'P1M' / 'P1Y' -> timedelta.
    Unbekanntes Format -> 1 Tag (konservativ). Vorher verstand die Funktion
    nur 'Nd': 'P1M' usw. fielen auf 1 Tag -> alle Monatsquellen dauerhaft STALE
    -> DataReadiness DEGRADED (Audit 2026-09-29)."""
    c = str(cadence or "").strip().upper()
    m = re.fullmatch(r"(\d+)\s*([DHW])", c)
    if m:
        n, u = int(m.group(1)), m.group(2)
        return {"D": timedelta(days=n), "H": timedelta(hours=n), "W": timedelta(weeks=n)}[u]
    m = re.fullmatch(r"P(?:(\d+)Y)?(?:(\d+)M)?(?:(\d+)W)?(?:(\d+)D)?(?:T(?:(\d+)H)?)?", c)
    if m and any(m.groups()):
        y, mo, w, d, h = (int(g) if g else 0 for g in m.groups())
        return timedelta(days=365 * y + 30 * mo + 7 * w + d, hours=h)
    return timedelta(days=1)


def evaluate_staleness(latest_observation: datetime | None, cadence: str | None,
                        now: datetime | None = None, frequency: str | None = None,
                        max_age_days: float | None = None) -> str:
    """STALE, wenn die jüngste Beobachtung älter ist als erlaubt: explizites
    max_age_days (Registry max_staleness_days) > Frequenz-Tabelle
    (Referenzperiode + Publikationsverzug) > 2x erwartete Update-Kadenz."""
    if latest_observation is None or (not cadence and not frequency and not max_age_days):
        return "UNKNOWN"
    now = ensure_utc(now) or utc_now()
    age = now - ensure_utc(latest_observation)
    if max_age_days:
        limit = timedelta(days=float(max_age_days))
    elif frequency in MAX_AGE_DAYS_BY_FREQUENCY:
        limit = timedelta(days=MAX_AGE_DAYS_BY_FREQUENCY[frequency])
    else:
        limit = 2 * parse_cadence(cadence)
    return "STALE" if age > limit else "FRESH"


def load_health(archive_root: str | Path) -> dict[str, dict]:
    path = Path(archive_root) / "health" / "source_health.json"
    if not path.exists():
        return {}
    import json
    return json.loads(path.read_text())


def save_health(archive_root: str | Path, health: dict[str, dict]) -> Path:
    import json
    path = Path(archive_root) / "health" / "source_health.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(health, indent=2, default=str))
    return path


# ── Readiness ─────────────────────────────────────────────────────────────────

_DEFAULT_READINESS_CFG = {
    "daily":   {"exploratory_min_obs": 60,  "exploratory_min_days": 90,
                "challenger_min_obs": 250, "challenger_min_days": 365},
    "monthly": {"exploratory_min_obs": 24,  "exploratory_min_days": 0,
                "challenger_min_obs": 60,  "challenger_min_days": 0},
    "event":   {"exploratory_min_obs": 20,  "exploratory_min_days": 0,
                "challenger_min_obs": 50,  "challenger_min_days": 0},
}


def _readiness_config() -> dict:
    try:
        from modules.config import cfg
        r = getattr(getattr(cfg, "external_context", None), "readiness", None)
        if r:
            merged = {k: dict(v) for k, v in _DEFAULT_READINESS_CFG.items()}
            for freq, vals in dict(r).items():
                merged.setdefault(freq, {}).update(dict(vals))
            return merged
    except Exception:
        pass
    return _DEFAULT_READINESS_CFG


def _promoted_features() -> list[str]:
    try:
        from modules.config import cfg
        learning = getattr(getattr(cfg, "external_context", None), "learning", None)
        return list(getattr(learning, "promoted_external_features", []) or [])
    except Exception:
        return []


def compute_readiness(source_cfg: dict, health: dict, stats: dict | None = None) -> DataReadiness:
    """Leitet die Reifestufe NUR aus Schema/PIT/Provenienz/Frische/Coverage/
    Zeitspanne/Beobachtungszahl ab — NIE aus Trading-Returns.

    stats (optional, vom Aufrufer gemessen):
      observation_count, confirmatory_observation_count, calendar_span_days,
      schema_ok (bool), pit_ok (bool)
    """
    stats = stats or {}
    fetchable, reason = gate_source(source_cfg)
    status = health.get("status", SourceStatus.PASS.value)

    if reason in ("status_override", "license_status=REVIEW_REQUIRED"):
        return DataReadiness.BLOCKED
    if not source_cfg.get("enabled", False):
        return DataReadiness.DISABLED
    if reason == "AUTH_MISSING" or status == SourceStatus.AUTH_MISSING.value:
        return DataReadiness.DISABLED
    if status in (SourceStatus.SCHEMA_CHANGED.value, SourceStatus.BLOCKED.value,
                  SourceStatus.REVIEW_REQUIRED.value, SourceStatus.DEFERRED.value):
        return DataReadiness.BLOCKED

    obs_count = int(stats.get("observation_count", health.get("rows_ingested", 0)) or 0)
    if obs_count <= 0:
        return DataReadiness.COLLECTING

    if not stats.get("schema_ok", health.get("parse_failures", 0) == 0):
        return DataReadiness.SCHEMA_VALIDATED  # Schema (noch) nicht sauber genug für PIT

    confirmatory = int(stats.get("confirmatory_observation_count", 0) or 0)
    pit_ok = bool(stats.get("pit_ok", confirmatory > 0
                             and health.get("pit_integrity_failures", 0) == 0))
    if not pit_ok:
        return DataReadiness.SCHEMA_VALIDATED

    frequency = source_cfg.get("frequency", "daily")
    thresholds = _readiness_config().get(frequency, _DEFAULT_READINESS_CFG["daily"])
    span_days = int(stats.get("calendar_span_days", 0) or 0)

    meets_exploratory = (obs_count >= thresholds.get("exploratory_min_obs", 10**9)
                          and span_days >= thresholds.get("exploratory_min_days", 0))
    meets_challenger = (obs_count >= thresholds.get("challenger_min_obs", 10**9)
                         and span_days >= thresholds.get("challenger_min_days", 0))

    if not meets_exploratory:
        return DataReadiness.PIT_VALIDATED

    degraded = (health.get("staleness") == "STALE"
                or int(health.get("consecutive_failures", 0) or 0) >= 3)

    readiness = DataReadiness.CHALLENGER_READY if meets_challenger else DataReadiness.EXPLORATORY_READY

    if readiness == DataReadiness.CHALLENGER_READY:
        promoted = _promoted_features()
        source_id = source_cfg.get("source_id", "")
        if source_id in promoted:
            readiness = DataReadiness.PRODUCTION_ELIGIBLE

    if degraded and readiness != DataReadiness.PRODUCTION_ELIGIBLE:
        return DataReadiness.DEGRADED
    return readiness


# ── Registry-Facade ───────────────────────────────────────────────────────────

class SourceRegistry:
    def __init__(self, sources_dir: str | Path = DEFAULT_SOURCES_DIR,
                 archive_root: str | Path = "outputs/external_data"):
        self.sources_dir = sources_dir
        self.archive_root = archive_root
        self.sources = load_source_configs(sources_dir)
        self.connectors = discover_connectors()

    def families(self) -> set[str]:
        return {s.get("family") for s in self.sources.values()}

    def iter_sources(self, family: str | None = None) -> list[dict]:
        return [s for s in self.sources.values() if family is None or s.get("family") == family]

    def connector_for(self, source_id: str) -> type | None:
        return self.connectors.get(source_id)

    def build_connector(self, source_id: str) -> Connector | None:
        cls = self.connector_for(source_id)
        if cls is None:
            return None
        return cls(self.sources.get(source_id))

    def load_health(self) -> dict[str, dict]:
        return load_health(self.archive_root)

    def save_health(self, health: dict[str, dict]) -> Path:
        return save_health(self.archive_root, health)

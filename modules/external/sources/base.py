"""
modules/external/sources/base.py – gemeinsame Konnektor-Schnittstelle.

Jeder Konnektor:
  - hat eine source_id, die in config/external_sources.yaml registriert ist
  - holt NUR offizielle maschinenlesbare Daten (modules.external.http.fetch)
  - entdeckt offizielle IDs zur Laufzeit aus Metadaten (nie erfinden)
  - liefert ConnectorResult: normalisierte Observations + Raw-Metadaten + Status
  - wirft NIE in die Pipeline hinein: Fehler landen im Result (status FAIL/
    AUTH_MISSING/SCHEMA_CHANGED/...), der Orchestrator entscheidet
  - Schemaänderungen → status SCHEMA_CHANGED (laut), nie stillschweigend umdeuten
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from modules.external.pit import Observation


class SourceStatus(str, Enum):
    PASS            = "PASS"
    WARN            = "WARN"
    FAIL            = "FAIL"
    STALE           = "STALE"
    AUTH_MISSING    = "AUTH_MISSING"
    SCHEMA_CHANGED  = "SCHEMA_CHANGED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    BLOCKED         = "BLOCKED"
    DEFERRED        = "DEFERRED"


@dataclass
class RawRecord:
    """Metadaten eines Abrufs (Payload selbst optional/komprimiert im Archiv)."""
    source_id: str
    dataset: str
    url: str
    fingerprint: str
    retrieved_at: datetime
    status_code: int
    content_type: str
    content_hash: str
    bytes: int
    content: bytes | None = None      # vom Archiv je nach raw_payload_policy gespeichert


@dataclass
class ConnectorResult:
    source_id: str
    status: SourceStatus
    observations: list[Observation] = field(default_factory=list)
    raw: list[RawRecord] = field(default_factory=list)
    message: str = ""
    latest_observation_time: datetime | None = None
    latest_release_time: datetime | None = None
    discovered_ids: dict = field(default_factory=dict)   # z.B. {"Rotterdam": "port1234"}
    parse_failures: int = 0


class Connector:
    source_id: str = ""
    parser_version: str = "1"

    def __init__(self, source_cfg: dict | None = None):
        self.cfg = source_cfg or {}

    def fetch(self, now: datetime) -> ConnectorResult:  # pragma: no cover - interface
        raise NotImplementedError

    def preflight(self, now: datetime) -> dict:
        """Live-Check ohne Archivierung: Endpoint, HTTP-Status, Schema-Kurzinfo,
        letzte Beobachtung, Release-Metadaten, Auth-Bedarf. Nie werfen."""
        try:
            res = self.fetch(now)
            return {
                "source_id": self.source_id, "status": res.status.value,
                "n_observations": len(res.observations),
                "latest_observation_time": res.latest_observation_time.isoformat()
                    if res.latest_observation_time else None,
                "endpoints": sorted({r.url for r in res.raw}),
                "http_status": sorted({r.status_code for r in res.raw}),
                "message": res.message, "discovered_ids": res.discovered_ids,
            }
        except Exception as e:  # noqa: BLE001 - preflight darf nie werfen
            return {"source_id": self.source_id, "status": "FAIL", "message": repr(e)}

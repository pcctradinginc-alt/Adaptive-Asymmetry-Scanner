"""
modules/external/pit.py – Point-in-Time-Datenmodell für externe Beobachtungen.

Kernregel: Für ein Signal zum Zeitpunkt T darf nur verwendet werden, was
`available_at <= T` erfüllt — NIE `observation_time <= T` als Ersatz.

Zeitbegriffe (alle UTC-aware):
  observation_time     – Zeitpunkt/Periode, die der Wert beschreibt
                         (z.B. Aktivitätstag eines Hafens, Monat einer Statistik)
  source_release_time  – offizielle Veröffentlichung (falls bekannt)
  available_at         – frühester Zeitpunkt, zu dem der Wert für uns
                         verwendbar war. Prospektiv ohne offizielle
                         Release-Zeit = erster erfolgreicher Abruf (retrieved_at).
  retrieved_at         – Zeitpunkt unseres Abrufs
  vintage_time         – Zeitpunkt, ab dem DIESE Version (Revision) galt
  forecast_issue_time  – Ausgabezeit einer Prognose
  forecast_valid_time  – Zeitpunkt, für den die Prognose gilt
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable


class AvailabilityPrecision(str, Enum):
    EXACT_TIMESTAMP   = "EXACT_TIMESTAMP"    # offizieller Zeitstempel existiert
    EXACT_DATE        = "EXACT_DATE"         # offizielles Veröffentlichungsdatum existiert
    CONSERVATIVE_DATE = "CONSERVATIVE_DATE"  # Datum bekannt, konservative Regel nötig
    INFERRED          = "INFERRED"           # aus belastbarer Evidenz rekonstruiert
    UNKNOWN           = "UNKNOWN"            # historische Verfügbarkeit nicht belegbar


# UNKNOWN darf nie stillschweigend als bestätigende Evidenz zählen.
CONFIRMATORY_PRECISIONS = frozenset({
    AvailabilityPrecision.EXACT_TIMESTAMP,
    AvailabilityPrecision.EXACT_DATE,
    AvailabilityPrecision.CONSERVATIVE_DATE,
    AvailabilityPrecision.INFERRED,
})


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_utc(value: Any) -> datetime | None:
    """datetime/ISO-String → tz-aware UTC datetime. Naive Werte werden als UTC
    interpretiert (Konnektoren müssen lokale Zeiten VORHER konvertieren)."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat(timespec="seconds") if dt is not None else None


@dataclass
class Observation:
    """Eine normalisierte externe Beobachtung (eine Zahl, eine Version)."""
    source_id:   str                 # z.B. "destatis_truck_toll"
    dataset:     str                 # z.B. "daily_index"
    series_id:   str                 # offizielle Serien-ID (nie erfunden)
    entity_id:   str                 # Region/Hafen/Station/Land; "" wenn global
    metric:      str                 # z.B. "index_sa", "portcalls_container"
    value:       float | None        # None = offiziell fehlend (nie 0 als Ersatz)
    unit:        str
    observation_time: datetime       # beschriebene Periode (Start), UTC
    available_at:     datetime
    retrieved_at:     datetime
    availability_precision: AvailabilityPrecision
    parser_version: str
    source_release_time: datetime | None = None
    vintage_time:        datetime | None = None
    forecast_issue_time: datetime | None = None
    forecast_valid_time: datetime | None = None
    payload_hash:        str = ""
    attrs: dict = field(default_factory=dict)   # z.B. vessel_type, lat/lon, unit notes

    def __post_init__(self) -> None:
        for name in ("observation_time", "available_at", "retrieved_at",
                     "source_release_time", "vintage_time",
                     "forecast_issue_time", "forecast_valid_time"):
            setattr(self, name, ensure_utc(getattr(self, name)))
        if not isinstance(self.availability_precision, AvailabilityPrecision):
            self.availability_precision = AvailabilityPrecision(self.availability_precision)
        if self.vintage_time is None:
            self.vintage_time = self.available_at
        if self.available_at is None or self.retrieved_at is None or self.observation_time is None:
            raise ValueError("observation_time, available_at und retrieved_at sind Pflicht")
        if self.forecast_valid_time is not None and self.forecast_issue_time is None:
            raise ValueError("Prognose ohne forecast_issue_time")

    # Identität einer Beobachtung OHNE Wert/Vintage — gleiche Identität +
    # anderer Wert = neue Revision (Vintage), gleicher Wert = Duplikat.
    def identity_key(self) -> str:
        parts = [self.source_id, self.dataset, self.series_id, self.entity_id, self.metric,
                 _iso(self.observation_time) or "", _iso(self.forecast_valid_time) or "",
                 _iso(self.forecast_issue_time) or ""]
        return "|".join(parts)

    def is_confirmatory(self) -> bool:
        return self.availability_precision in CONFIRMATORY_PRECISIONS

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, datetime):
                d[k] = _iso(v)
        d["availability_precision"] = self.availability_precision.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Observation":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def available_as_of(observations: Iterable[Observation], t: datetime) -> list[Observation]:
    """PIT-Filter: nur Beobachtungen mit available_at <= t; pro identity_key die
    jüngste bis t verfügbare Version (Vintage)."""
    t = ensure_utc(t)
    best: dict[str, Observation] = {}
    for o in observations:
        if o.available_at > t:
            continue
        k = o.identity_key()
        cur = best.get(k)
        if cur is None or (o.vintage_time or o.available_at) > (cur.vintage_time or cur.available_at):
            best[k] = o
    return list(best.values())


def payload_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def request_fingerprint(method: str, url: str, params: dict | None = None) -> str:
    """Stabiler Fingerabdruck einer Anfrage (ohne Credentials!)."""
    safe = {k: v for k, v in sorted((params or {}).items())
            if not any(s in k.lower() for s in ("key", "token", "appid", "secret", "password"))}
    raw = json.dumps([method.upper(), url, safe], sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

"""modules/expectation_alpha/schemas.py – Kern-Schemas des Expectation-Alpha-Layers (V1, SHADOW).

Alle Werte tragen ihre Provenienz. Fehlende Daten bleiben fehlend (None + Status), nie 0.
Statuswerte sind getrennt: ERROR (technischer Fehler / Datenfehler) wird nie als ABSTAIN gelesen.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field

SCHEMA_VERSION = "ea-schema-v1"
STAGE = "EA_NEWS_CANDIDATE"                 # Population der EA-Verträge (hypothesis_contract)

# Datenstatus je Merkmal / Gap / Confirmation
OK = "OK"
STALE = "STALE"                             # jüngster Wert älter als max_age -> als fehlend behandelt
INSUFFICIENT_DATA = "INSUFFICIENT_DATA"     # zu wenig PIT-Historie für z/Perzentil
UNAVAILABLE = "UNAVAILABLE"                 # Quelle/Reihe existiert (PIT) nicht – nie simuliert
DATA_STATUSES = (OK, STALE, INSUFFICIENT_DATA, UNAVAILABLE)

# Research-Entscheidung (deterministisch, kein LLM)
TRADE, WAIT, ABSTAIN, ERROR = "TRADE", "WAIT", "ABSTAIN", "ERROR"
DECISION_STATUSES = (TRADE, WAIT, ABSTAIN, ERROR)

# Fehlerklassen (regelbasiert, nach Outcome)
FAILURE_CLASSES = ("THESIS_WRONG", "TIMING_WRONG", "EXPRESSION_WRONG", "SIZING_WRONG", "CATALYST_WRONG",
                   "REGIME_CHANGED", "DATA_BAD", "EXECUTION_BAD", "UNKNOWN")

NEWS_STRONG, NEWS_WEAK = "STRONG", "WEAK"
GROUPS = ("A", "B", "C", "D", "E", "X")


def finite(x) -> float | None:
    """float oder None (NaN/inf/None/nicht numerisch -> None, nie 0)."""
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def rnd(x, nd: int = 6) -> float | None:
    v = finite(x)
    return None if v is None else round(v, nd)


@dataclass
class FeatureValue:
    """Ein Merkmal mit vollständiger Provenienz (PIT-nachvollziehbar)."""
    name: str
    value: float | None
    unit: str
    source: str
    observed_at: str | None = None      # beschriebene Periode (Start)
    published_at: str | None = None     # offizielle Veröffentlichung (falls bekannt)
    available_at: str | None = None     # ab wann dem System bekannt (PIT-Grenze)
    retrieved_at: str | None = None     # Abrufzeit des Archivs
    vintage: str | None = None          # Vintage-Zeitpunkt (ALFRED realtime_start)
    transformation: str = "level"
    freshness_days: float | None = None
    confidence: float | None = None     # 0..1 (Frische, Vintage-Qualität, Historie)
    status: str = OK

    def __post_init__(self) -> None:
        self.value = finite(self.value)
        if self.value is None and self.status == OK:
            self.status = UNAVAILABLE

    def to_dict(self) -> dict:
        d = asdict(self)
        d["value"] = rnd(d["value"])
        d["freshness_days"] = rnd(d["freshness_days"], 2)
        d["confidence"] = rnd(d["confidence"], 3)
        return d


@dataclass
class RunErrors:
    """Sammelt technische Fehler eines Laufs. Nie still verschluckt: landen in Ledger und Report."""
    items: list[dict] = field(default_factory=list)

    def add(self, where: str, exc: BaseException | str) -> None:
        msg = exc if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
        self.items.append({"where": where, "error": str(msg)[:300]})

    def __bool__(self) -> bool:
        return bool(self.items)


def canonical_hash(obj, n: int = 16) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False)
                          .encode()).hexdigest()[:n]

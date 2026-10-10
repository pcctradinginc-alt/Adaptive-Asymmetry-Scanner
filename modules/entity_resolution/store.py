"""Point-in-Time-Entity-Store.

Grundregeln (Auftrag + Audit):
  * Jede Zuordnung trägt mapping_source, mapping_confidence, valid_from, valid_to.
  * valid_from ist der FRÜHESTE BELEGTE Zeitpunkt: ein offizielles Datum der
    Quelle (z.B. GLEIF-Beziehungsbeginn, SEC formerNames.from) oder – ohne
    offizielles Datum – der erste Abruf. Nie wird eine heutige Konzernstruktur
    rückwirkend gültig gemacht.
  * Append-only: Änderungen schließen den alten Datensatz (valid_to) und
    öffnen einen neuen; Historie wird nie überschrieben.
  * Unbekannt != falsch: ohne gültigen Datensatz zum Stichtag liefert
    resolve() None (nie eine "beste Vermutung").

Ausnahme mit Kennzeichnung: Ticker→CIK für die HISTORISCHE Research-
Verknüpfung. Die CIK ist die dauerhafte SEC-Registranten-ID; die Kursdaten
(yfinance) sind ebenfalls nach heutigem Ticker geschlüsselt. Diese Nutzung
heißt usage="research_ticker_identity" und trägt confidence MEDIUM
(Ticker-Wiederverwendungsrisiko), siehe docs/ENTITY_RESOLUTION.md.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_PATH = Path("outputs/entity/entity_map.jsonl")
CONFIDENCE = ("HIGH", "MEDIUM", "LOW")

_SUFFIXES = r"\b(incorporated|inc|corporation|corp|company|co|limited|ltd|plc|llc|lp|l\.p|holdings?|group|" \
            r"the|n\.?v|s\.?a|ag|se|sa)\b"


def normalize_name(name: str | None) -> str:
    """Rechtsform-/Interpunktions-neutraler Vergleichsname (deterministisch)."""
    if not name:
        return ""
    s = name.lower().replace("&", " and ")
    s = re.sub(r"[.,'’()/\\-]", " ", s)
    s = re.sub(_SUFFIXES, " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _iso(d) -> str | None:
    if d is None or d == "":
        return None
    if isinstance(d, datetime):
        return d.astimezone(timezone.utc).date().isoformat()
    return str(d)[:10]


@dataclass
class EntityRecord:
    entity_id: str                      # stabil: "cik:0000320193" bzw. "lei:..." ohne CIK
    ticker: str | None = None
    cik: str | None = None              # 10-stellig, führende Nullen
    lei: str | None = None
    canonical_name: str = ""
    aliases: list[str] = field(default_factory=list)
    parent_entity: str | None = None    # entity_id / LEI des direkten Mutterunternehmens
    ultimate_parent: str | None = None
    jurisdiction: str | None = None
    country: str | None = None
    industry: str | None = None         # z.B. SIC-Code + Beschreibung
    domains: list[str] = field(default_factory=list)
    mapping_source: str = ""
    mapping_confidence: str = "LOW"
    mapping_score: float | None = None
    valid_from: str | None = None       # ISO-Datum; None = "seit vor Beginn der Aufzeichnung" NUR mit offiziellem Beleg
    valid_to: str | None = None         # exklusiv; None = gilt weiter
    usage: str = "pit"                  # "pit" | "research_ticker_identity"
    evidence: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.mapping_confidence not in CONFIDENCE:
            raise ValueError(f"mapping_confidence muss {CONFIDENCE} sein")
        if not self.mapping_source:
            raise ValueError("Zuordnung ohne mapping_source")
        if self.cik:
            self.cik = str(int(self.cik)).zfill(10)
        self.valid_from, self.valid_to = _iso(self.valid_from), _iso(self.valid_to)
        if self.valid_from and self.valid_to and self.valid_to <= self.valid_from:
            raise ValueError("valid_to <= valid_from")

    def valid_at(self, as_of) -> bool:
        d = _iso(as_of)
        return (self.valid_from is None or self.valid_from <= d) and (self.valid_to is None or d < self.valid_to)

    def key(self) -> str:
        """Inhalt ohne Gültigkeit/Evidenz – gleicher Key = gleiche Aussage."""
        core = {k: v for k, v in asdict(self).items() if k not in ("valid_from", "valid_to", "evidence")}
        return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()[:16]


class EntityStore:
    """Append-only JSONL-Store. Jede Zeile: {"op": "open"|"close", "record": {...}, "at": ...}."""

    def __init__(self, path: Path | str = DEFAULT_PATH):
        self.path = Path(path)
        self.records: list[EntityRecord] = []
        if self.path.exists():
            from modules.atomic_io import read_jsonl          # abgeschnittene letzte Zeile (Abbruch) tolerieren
            for ev in read_jsonl(self.path):
                self._apply(ev)

    def _apply(self, ev: dict) -> None:
        rec = EntityRecord(**ev["record"])
        if ev["op"] == "open":
            self.records.append(rec)
        elif ev["op"] == "close":
            for r in self.records:
                if r.key() == rec.key() and r.valid_from == rec.valid_from and r.valid_to is None:
                    r.valid_to = rec.valid_to

    def _write(self, ev: dict) -> None:
        from modules.atomic_io import append_jsonl            # Zeilenreparatur nach Abbruch (Timeout-Safety)
        append_jsonl(self.path, [ev], ensure_ascii=False, sort_keys=True)

    def open_records(self, entity_id: str, usage: str | None = None) -> list[EntityRecord]:
        return [r for r in self.records if r.entity_id == entity_id and r.valid_to is None
                and (usage is None or r.usage == usage)]

    def upsert(self, rec: EntityRecord, as_of) -> str:
        """Neue Aussage zum Stichtag as_of. Gleiche offene Aussage -> nichts.
        Abweichende offene Aussage derselben Entität/Nutzung/Quelle -> schließen (valid_to=as_of)
        und neue öffnen. -> "unchanged" | "opened" | "replaced"."""
        as_of = _iso(as_of)
        # Reihe = Entität + Nutzung + Quelle + Ticker: eine CIK kann zugleich mehrere
        # Ticker haben (Aktiengattungen, z.B. GOOG/GOOGL) – das ist keine Änderung.
        same = [r for r in self.open_records(rec.entity_id, rec.usage)
                if r.mapping_source == rec.mapping_source and (r.ticker or "") == (rec.ticker or "")]
        if any(r.key() == rec.key() for r in same):
            return "unchanged"
        status = "opened"
        for r in same:
            if r.valid_from is not None and as_of <= r.valid_from:
                raise ValueError("Rückdatierte Änderung verboten (as_of <= valid_from des offenen Datensatzes)")
            r.valid_to = as_of
            self._write({"op": "close", "record": asdict(r), "at": as_of})
            status = "replaced"
        if rec.valid_from is None and not rec.evidence.get("official_valid_from"):
            rec.valid_from = as_of                  # ohne offiziellen Beleg: ab erstem Abruf, nie rückwirkend
        self.records.append(rec)
        self._write({"op": "open", "record": asdict(rec), "at": as_of})
        return status

    def close(self, rec: EntityRecord, as_of) -> None:
        """Offenen Datensatz beenden (z.B. Ticker nicht mehr in der SEC-Liste). Nie rückwirkend."""
        as_of = _iso(as_of)
        if rec.valid_to is not None:
            return
        if rec.valid_from is not None and as_of <= rec.valid_from:
            raise ValueError("Rückdatiertes Schließen verboten")
        rec.valid_to = as_of
        self._write({"op": "close", "record": asdict(rec), "at": as_of})

    def resolve(self, *, ticker: str | None = None, cik: str | None = None, lei: str | None = None,
                as_of=None, usage: str = "pit", min_confidence: str = "LOW") -> EntityRecord | None:
        """Gültiger Datensatz zum Stichtag (None = unbekannt). Bei mehreren: höchste Konfidenz."""
        if cik:
            cik = str(int(cik)).zfill(10)
        rank = {c: i for i, c in enumerate(CONFIDENCE)}
        cands = [r for r in self.records if r.usage == usage and r.valid_at(as_of)
                 and (ticker is None or (r.ticker or "").upper() == ticker.upper())
                 and (cik is None or r.cik == cik) and (lei is None or r.lei == lei)
                 and rank[r.mapping_confidence] <= rank[min_confidence]]
        if not cands:
            return None
        return sorted(cands, key=lambda r: (rank[r.mapping_confidence], -(r.mapping_score or 0)))[0]

    def profile(self, *, ticker: str | None = None, cik: str | None = None, lei: str | None = None,
                as_of=None, usage: str = "pit", min_confidence: str = "MEDIUM") -> dict | None:
        """Zusammengeführte Sicht aller zum Stichtag gültigen Datensätze einer
        Entität (SEC + GLEIF + Eltern). Je Feld der Wert aus dem Datensatz mit
        der höchsten Konfidenz; Herkunft je Feld in "sources". None = unbekannt."""
        anchor = self.resolve(ticker=ticker, cik=cik, lei=lei, as_of=as_of, usage=usage, min_confidence=min_confidence)
        if anchor is None:
            return None
        rank = {c: i for i, c in enumerate(CONFIDENCE)}
        recs = sorted([r for r in self.records if r.entity_id == anchor.entity_id and r.usage == usage
                       and r.valid_at(as_of) and rank[r.mapping_confidence] <= rank[min_confidence]],
                      key=lambda r: rank[r.mapping_confidence])
        out: dict = {"entity_id": anchor.entity_id, "sources": {}}
        for f in ("ticker", "cik", "lei", "canonical_name", "parent_entity", "ultimate_parent", "jurisdiction",
                  "country", "industry"):
            for r in recs:
                v = getattr(r, f)
                if v:
                    out[f] = v
                    out["sources"][f] = {"source": r.mapping_source, "confidence": r.mapping_confidence,
                                         "valid_from": r.valid_from}
                    break
            else:
                out[f] = None
        out["aliases"] = sorted({a for r in recs for a in r.aliases})
        out["mapping_confidence"] = recs[0].mapping_confidence if recs else None
        return out

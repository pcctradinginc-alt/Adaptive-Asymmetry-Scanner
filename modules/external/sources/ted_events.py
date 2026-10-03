"""TED (Tenders Electronic Daily) – Zuschlagsbekanntmachungen der EU als PIT-Events.

Quelle: TED Search API v3 (Amt für Veröffentlichungen der EU), POST
https://api.ted.europa.eu/v3/notices/search, JSON-Body {query, fields, page, limit, ...}.
Keine Anmeldung für die Suche nötig. Nur offizielle API, kein Scraping.

PIT: available_at = Veröffentlichungstag im Amtsblatt (publication-date) + 1 Tag, 00:00 UTC
(CONSERVATIVE_DATE – die Tagesausgabe erscheint morgens MEZ; der Folgetag ist sicher).

Zuordnung Gewinner -> Emittent (streng, nie geraten):
  HIGH    normalisierter Gewinnername == normalisierter Firmen-/Altname
  MEDIUM  Gewinnername beginnt mit dem Firmennamen + Leerzeichen UND der Firmenname ist
          unterscheidbar (>= 2 Wörter oder >= 8 Zeichen), z.B. "IBM Belgium" -> nein (3 Zeichen),
          "Accenture Technology Solutions" -> "accenture" (9 Zeichen) ja
  sonst   keine Zuordnung (LOW wird verworfen)
Tochtergesellschaften mit anderem Namen werden NICHT erfasst (dokumentierte Lücke).
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from modules.entity_resolution.store import normalize_name
from modules.external.http import SchemaError
from modules.external.pit import AvailabilityPrecision, Observation

PARSER_VERSION = "ted-v1"
SEARCH_URL = "https://api.ted.europa.eu/v3/notices/search"
FIELDS = ["publication-number", "publication-date", "notice-type", "winner-name", "total-value",
          "total-value-cur", "classification-cpv", "buyer-country"]
# Expert-Query: Zuschlagsbekanntmachungen eines Gewinners im Zeitfenster
QUERY_TEMPLATE = 'winner-name ~ ("{name}") AND publication-date >= {start} AND publication-date <= {end}'
PROBE_TEMPLATE = "form-type IN (result) AND publication-date >= {start} AND publication-date <= {end}"
PAGE_LIMIT = 100
MAX_PAGES = 10                       # je Entität und Jahr; darüber -> Jahr als unvollständig markiert


def body(query: str, page: int = 1, limit: int = PAGE_LIMIT, fields: list[str] | None = None) -> dict:
    return {"query": query, "fields": fields or FIELDS, "page": page, "limit": limit, "scope": "ALL",
            "paginationMode": "PAGE_NUMBER", "checkQuerySyntax": False}


def query_for(name: str, year: int) -> str:
    clean = re.sub(r'["\\()]', " ", name).strip()
    return QUERY_TEMPLATE.format(name=clean, start=f"{year}0101", end=f"{year}1231")


def _flat(v) -> list[str]:
    """TED-Felder sind teils Listen, teils sprachabhängige Dicts {'eng': [...], 'deu': [...]}."""
    if v is None:
        return []
    if isinstance(v, (str, int, float)):
        return [str(v)]
    if isinstance(v, list):
        return [x for e in v for x in _flat(e)]
    if isinstance(v, dict):
        pref = v.get("eng") or v.get("ENG")
        return _flat(pref) if pref else [x for e in v.values() for x in _flat(e)]
    return []


def parse_search(payload: dict) -> tuple[list[dict], int]:
    """-> (Notices als {feld: [werte]}, totalNoticeCount)."""
    if not isinstance(payload, dict) or "notices" not in payload:
        raise SchemaError(f"TED search: 'notices' fehlt (Schlüssel: {sorted(payload)[:8] if isinstance(payload, dict) else type(payload)})")
    out = []
    for n in payload["notices"]:
        if not isinstance(n, dict):
            raise SchemaError("TED search: Notice ist kein Objekt")
        out.append({k: _flat(v) for k, v in n.items()})
    total = payload.get("totalNoticeCount")
    return out, int(total) if isinstance(total, (int, float)) else len(out)


def _date(s: str | None) -> datetime | None:
    if not s:
        return None
    m = re.match(r"(\d{4})-?(\d{2})-?(\d{2})", str(s))
    return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), tzinfo=timezone.utc) if m else None


def _distinct(alias: str) -> bool:
    return len(alias.split()) >= 2 or len(alias) >= 8


def match_winner(winner: str, aliases: list[str]) -> str | None:
    """-> 'HIGH' | 'MEDIUM' | None."""
    w = normalize_name(winner)
    if not w:
        return None
    norm = {normalize_name(a) for a in aliases if a}
    norm.discard("")
    if w in norm:
        return "HIGH"
    if any(_distinct(a) and w.startswith(a + " ") for a in norm):
        return "MEDIUM"
    return None


def notices_to_observations(cik: str, notices: list[dict], aliases: list[str], retrieved_at: datetime) -> list[Observation]:
    out = []
    for n in notices:
        n = {k: (v if isinstance(v, list) and all(isinstance(x, str) for x in v) else _flat(v)) for k, v in n.items()}
        pub = (n.get("publication-number") or [None])[0]
        pdate = _date((n.get("publication-date") or [None])[0])
        if not pub or pdate is None:
            continue
        conf, matched = None, None
        for w in n.get("winner-name") or []:
            c = match_winner(w, aliases)
            if c and (conf is None or c == "HIGH"):
                conf, matched = c, w
        if conf is None:
            continue
        raw = (n.get("total-value") or [None])[0]
        cur = (n.get("total-value-cur") or [None])[0]
        val = None                           # Betrag nur bei ausgewiesener Währung EUR (nie umrechnen/raten)
        if raw not in (None, "") and cur == "EUR":
            try:
                val = float(raw)
            except (TypeError, ValueError):
                val = None
        out.append(Observation(
            source_id="ted_awards", dataset="contract_award", series_id=f"ted:{pub}:{cik}", entity_id=cik,
            metric="award_eur", value=val, unit="EUR", observation_time=pdate,
            available_at=pdate + timedelta(days=1), retrieved_at=retrieved_at,
            availability_precision=AvailabilityPrecision.CONSERVATIVE_DATE, parser_version=PARSER_VERSION,
            attrs={"notice_type": (n.get("notice-type") or [None])[0],
                   "cpv2": ((n.get("classification-cpv") or [""])[0] or "")[:2] or None,
                   "buyer_country": (n.get("buyer-country") or [None])[0],
                   "matched_name": matched, "confidence": conf}))
    return out


def probe_winner_share(notices: list[dict]) -> float | None:
    """Anteil der Notices mit Gewinnernamen (Feld-Abdeckung eines Jahres)."""
    if not notices:
        return None
    return sum(1 for n in notices if n.get("winner-name")) / len(notices)

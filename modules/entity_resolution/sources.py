"""Quellen der Entity Resolution: SEC (Ticker/CIK/Name/SIC) und GLEIF (LEI, Eltern).

Nur offizielle APIs über modules.external.http.fetch (Retries, Fingerprint,
Credential-Redaktion). Parser sind reine Funktionen (fixture-testbar), die bei
Schemaabweichung SchemaError werfen statt zu raten.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from modules.entity_resolution.store import EntityRecord, normalize_name
from modules.external.http import SchemaError, fetch

log = logging.getLogger(__name__)

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
GLEIF_BASE = "https://api.gleif.org/api/v1"
# SEC Fair Access verlangt einen identifizierenden User-Agent mit Kontakt.
US_STATES = frozenset("AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH "
                      "NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY PR".split())
SEC_HEADERS = {"User-Agent": "AdaptiveAsymmetryScanner research@pcctrading.com", "Accept-Encoding": "gzip, deflate"}


# ── SEC ───────────────────────────────────────────────────────────────────────

def parse_company_tickers(payload: dict) -> list[dict]:
    if not isinstance(payload, dict):
        raise SchemaError("company_tickers.json: Objekt erwartet")
    out = []
    for v in payload.values():
        if not {"cik_str", "ticker", "title"} <= set(v):
            raise SchemaError(f"company_tickers.json: Felder fehlen in {v}")
        out.append({"cik": str(int(v["cik_str"])).zfill(10), "ticker": str(v["ticker"]).upper().replace(".", "-"),
                    "title": v["title"]})
    return out


def parse_submissions_entity(payload: dict) -> dict:
    """Entitätsfelder aus data.sec.gov/submissions/CIK##########.json."""
    if "cik" not in payload or "name" not in payload:
        raise SchemaError("submissions: cik/name fehlen")
    addr = ((payload.get("addresses") or {}).get("business") or {})
    return {"cik": str(int(payload["cik"])).zfill(10), "name": payload["name"],
            "former_names": [{"name": f.get("name"), "from": (f.get("from") or "")[:10] or None,
                              "to": (f.get("to") or "")[:10] or None} for f in payload.get("formerNames") or []],
            "tickers": [str(t).upper().replace(".", "-") for t in payload.get("tickers") or []],
            "sic": payload.get("sic"), "sic_description": payload.get("sicDescription"),
            "state_of_incorporation": payload.get("stateOfIncorporation"),
            "business_state_or_country": addr.get("stateOrCountry"),
            "website": payload.get("website") or None}


def sec_records(tickers_rows: list[dict], submissions: dict[str, dict] | None = None) -> list[EntityRecord]:
    """-> je CIK ein PIT-Datensatz (gültig ab Abruf) UND ein als solcher
    gekennzeichneter Research-Identitätsdatensatz (Ticker = heutiger Schlüssel
    der Kursdaten, CIK dauerhaft)."""
    out = []
    for row in tickers_rows:
        sub = (submissions or {}).get(row["cik"]) or {}
        aliases = sorted({row["title"], sub.get("name") or row["title"]}
                         | {f["name"] for f in sub.get("former_names", []) if f.get("name")})
        loc = sub.get("business_state_or_country")
        base = dict(entity_id=f"cik:{row['cik']}", ticker=row["ticker"], cik=row["cik"],
                    canonical_name=sub.get("name") or row["title"], aliases=aliases,
                    jurisdiction=sub.get("state_of_incorporation"),
                    country="US" if loc in US_STATES else None,          # SEC-Codes für Ausland sind keine ISO-Länder
                    industry=(f"SIC {sub['sic']} {sub.get('sic_description') or ''}".strip() if sub.get("sic") else None),
                    domains=[sub["website"]] if sub.get("website") else [],
                    mapping_source="sec_company_tickers")
        out.append(EntityRecord(**base, mapping_confidence="HIGH", mapping_score=1.0, usage="pit",
                                evidence={"former_names": sub.get("former_names", [])}))
        out.append(EntityRecord(**base, mapping_confidence="MEDIUM", mapping_score=0.8,
                                usage="research_ticker_identity", valid_from=None,
                                evidence={"official_valid_from": "CIK ist dauerhafte SEC-Registranten-ID",
                                          "risk": "Ticker-Wiederverwendung; Kursdaten ebenfalls nach heutigem Ticker"}))
    return out


def fetch_sec_tickers() -> list[dict]:
    return parse_company_tickers(fetch(SEC_TICKERS_URL, headers=SEC_HEADERS).json())


def fetch_sec_submissions(cik: str) -> dict:
    return fetch(SEC_SUBMISSIONS_URL.format(cik=str(int(cik)).zfill(10)), headers=SEC_HEADERS).json()


# ── GLEIF ─────────────────────────────────────────────────────────────────────

def parse_lei_records(payload: dict) -> list[dict]:
    if not isinstance(payload, dict) or "data" not in payload:
        raise SchemaError("GLEIF lei-records: 'data' fehlt")
    out = []
    for d in payload["data"]:
        a = d.get("attributes") or {}
        ent, reg = a.get("entity") or {}, a.get("registration") or {}
        if not a.get("lei") or "legalName" not in ent:
            raise SchemaError("GLEIF lei-record ohne lei/legalName")
        out.append({"lei": a["lei"], "legal_name": (ent.get("legalName") or {}).get("name"),
                    "other_names": [n.get("name") for n in ent.get("otherNames") or [] if n.get("name")],
                    "jurisdiction": ent.get("jurisdiction"),
                    "country": (ent.get("legalAddress") or {}).get("country"),
                    "entity_status": ent.get("status"), "registration_status": reg.get("status"),
                    "initial_registration": (reg.get("initialRegistrationDate") or "")[:10] or None})
    return out


def parse_relationship(payload: dict) -> dict | None:
    """direct-/ultimate-parent-relationship -> {parent_lei, start, end, status} (None = keine Beziehung)."""
    data = (payload or {}).get("data")
    if not data:
        return None
    rel = ((data.get("attributes") or {}).get("relationship") or {})
    end = (rel.get("endNode") or {}).get("id")
    if not end:
        raise SchemaError("GLEIF relationship ohne endNode")
    periods = rel.get("periods") or rel.get("relationshipPeriods") or []
    rp = next((p for p in periods if p.get("type", p.get("periodType")) == "RELATIONSHIP_PERIOD"), None) or \
        (periods[0] if periods else {})
    return {"parent_lei": end, "start": (rp.get("startDate") or "")[:10] or None,
            "end": (rp.get("endDate") or "")[:10] or None, "status": rel.get("status")}


def match_lei(name_candidates: list[str], records: list[dict], country: str | None = "US") -> dict:
    """Deterministische Zuordnung -> {lei, confidence, score, reason} oder confidence LOW/None.
    HIGH: genau EIN aktiver Datensatz mit identischem Normnamen und passendem Land.
    MEDIUM: identischer Normname, Land abweichend/unbekannt, eindeutig.
    LOW: nur Teilübereinstimmung oder mehrdeutig (nie für Features nutzbar)."""
    wanted = {normalize_name(n) for n in name_candidates if n}
    exact = [r for r in records if (r.get("entity_status") in (None, "ACTIVE"))
             and ({normalize_name(r["legal_name"])} | {normalize_name(n) for n in r["other_names"]}) & wanted]
    same_country = [r for r in exact if country and (r.get("country") == country
                                                     or (r.get("jurisdiction") or "").startswith(country))]
    if len(same_country) == 1:
        return {"lei": same_country[0]["lei"], "confidence": "HIGH", "score": 1.0, "record": same_country[0],
                "reason": "exakter Normname + Land"}
    if len(exact) == 1:
        return {"lei": exact[0]["lei"], "confidence": "MEDIUM", "score": 0.7, "record": exact[0],
                "reason": "exakter Normname, Land abweichend/unbekannt"}
    if len(exact) > 1:
        return {"lei": None, "confidence": "LOW", "score": 0.0, "record": None,
                "reason": f"mehrdeutig ({len(exact)} Treffer)"}
    return {"lei": None, "confidence": "LOW", "score": 0.0, "record": None, "reason": "kein exakter Treffer"}


def name_queries(name: str) -> list[str]:
    """Suchbegriffe: Volltext auf dem Kernnamen (ohne Rechtsform) + exakter Name.
    Hintergrund (Live 2026-10-03): filter[entity.legalName] ist ein exakter Abgleich –
    SEC-Kurzformen wie "NVIDIA CORP" treffen "NVIDIA CORPORATION" nicht (126/150 LOW).
    Die Zuordnung selbst bleibt streng (match_lei: identischer Normname)."""
    import re
    core = normalize_name(name)
    expanded = re.sub(r"\bCORP\.?$", "CORPORATION", name.strip(), flags=re.I)
    expanded = re.sub(r"\bINC$", "INC.", expanded, flags=re.I)
    expanded = re.sub(r"\bCO$", "COMPANY", expanded, flags=re.I)
    return [q for q in dict.fromkeys([core, name, expanded]) if q]


def fetch_gleif_by_name(name: str) -> list[dict]:
    hdr = {"Accept": "application/vnd.api+json"}
    out: dict[str, dict] = {}
    for i, q in enumerate(name_queries(name)):
        params = ({"filter[fulltext]": q, "page[size]": 50} if i == 0 else
                  {"filter[entity.legalName]": q, "page[size]": 10})
        for r in parse_lei_records(fetch(f"{GLEIF_BASE}/lei-records", params=params, headers=hdr).json()):
            out.setdefault(r["lei"], r)
        if any(normalize_name(r["legal_name"]) == normalize_name(name) for r in out.values()):
            break                                         # exakter Normname gefunden -> keine weiteren Abrufe
    return list(out.values())


def parse_child_relationships(payload: dict) -> list[dict]:
    """direct-child-relationships -> [{child_lei, start, end, status}] (startNode = Tochter)."""
    if not isinstance(payload, dict) or "data" not in payload:
        raise SchemaError("GLEIF child-relationships: 'data' fehlt")
    out = []
    for d in payload["data"] or []:
        rel = ((d.get("attributes") or {}).get("relationship") or {})
        child = (rel.get("startNode") or {}).get("id")
        if not child:
            raise SchemaError("GLEIF child-relationship ohne startNode")
        periods = rel.get("periods") or rel.get("relationshipPeriods") or []
        rp = next((p for p in periods if p.get("type", p.get("periodType")) == "RELATIONSHIP_PERIOD"), None) or {}
        out.append({"child_lei": child, "start": (rp.get("startDate") or "")[:10] or None,
                    "end": (rp.get("endDate") or "")[:10] or None, "status": rel.get("status")})
    return out


MAX_CHILD_PAGES = 3


def fetch_gleif_children(lei: str) -> tuple[list[dict], dict[str, dict]]:
    """-> (Beziehungen, {child_lei: lei-record}). Beide Aufrufe offiziell; 404 = keine Töchter."""
    hdr = {"Accept": "application/vnd.api+json"}
    rels: list[dict] = []
    for page in range(1, MAX_CHILD_PAGES + 1):
        try:
            res = fetch(f"{GLEIF_BASE}/lei-records/{lei}/direct-child-relationships",
                        params={"page[size]": 100, "page[number]": page}, headers=hdr, retries=2)
        except Exception as e:  # noqa: BLE001 – 404 = keine gemeldeten Töchter
            if "404" in str(e):
                break
            raise
        batch = parse_child_relationships(res.json())
        rels += batch
        if len(batch) < 100:
            break
    infos: dict[str, dict] = {}
    ids = [r["child_lei"] for r in rels]
    for i in range(0, len(ids), 100):
        res = fetch(f"{GLEIF_BASE}/lei-records", params={"filter[lei]": ",".join(ids[i:i + 100]), "page[size]": 100},
                    headers=hdr)
        infos.update({r["lei"]: r for r in parse_lei_records(res.json())})
    return rels, infos


def child_records(issuer: EntityRecord, issuer_lei: str, issuer_conf: str, rels: list[dict],
                  infos: dict[str, dict]) -> list[EntityRecord]:
    """Tochtergesellschaften als zeitabhängige Exposures (PIT): gültig ab offiziellem
    Beziehungsbeginn (GLEIF RELATIONSHIP_PERIOD), sonst ab Abruf – nie rückwirkend.
    Konfidenz nie höher als die der LEI-Zuordnung des Emittenten."""
    out = []
    for r in rels:
        if r.get("status") not in (None, "ACTIVE"):
            continue
        info = infos.get(r["child_lei"]) or {}
        if not info.get("legal_name"):
            continue                                   # ohne offiziellen Namen kein Datensatz
        if r.get("start") and r.get("end") and r["end"] <= r["start"]:
            continue
        out.append(EntityRecord(
            entity_id=f"lei:{r['child_lei']}", lei=r["child_lei"], canonical_name=info["legal_name"],
            aliases=sorted({info["legal_name"], *info.get("other_names", [])}), parent_entity=issuer.entity_id,
            ultimate_parent=issuer.entity_id, jurisdiction=info.get("jurisdiction"), country=info.get("country"),
            mapping_source="gleif_child", mapping_confidence=issuer_conf, mapping_score=None,
            valid_from=r.get("start"), valid_to=r.get("end"), usage="exposure_subsidiary",
            evidence={"official_valid_from": "GLEIF relationship period start" if r.get("start") else None,
                      "parent_lei": issuer_lei, "parent_cik": issuer.cik}))
    return out


def fetch_gleif_parent(lei: str, kind: str = "direct") -> dict | None:
    url = f"{GLEIF_BASE}/lei-records/{lei}/{'direct' if kind == 'direct' else 'ultimate'}-parent-relationship"
    try:
        res = fetch(url, headers={"Accept": "application/vnd.api+json"}, retries=2)
    except Exception as e:  # noqa: BLE001 – 404 = keine gemeldete Beziehung
        if "404" in str(e):
            return None
        raise
    return parse_relationship(res.json())


def gleif_records(sec_rec: EntityRecord, match: dict, direct: dict | None,
                  ultimate: dict | None) -> list[EntityRecord]:
    """LEI-Verknüpfung und Konzernbeziehung als GETRENNTE PIT-Datensätze:
    LEI gilt ab GLEIF-Erstregistrierung (offiziell); Eltern gelten erst ab
    ihrem eigenen Beziehungsbeginn (offiziell) bzw. ab Abruf – nie rückwirkend
    über die LEI-Gültigkeit."""
    if not match.get("lei"):
        return []
    r = match["record"]
    common = dict(entity_id=sec_rec.entity_id, ticker=sec_rec.ticker, cik=sec_rec.cik, lei=match["lei"],
                  canonical_name=r["legal_name"], aliases=sorted(set(sec_rec.aliases) | set(r["other_names"])),
                  jurisdiction=r.get("jurisdiction"), country=r.get("country"), industry=sec_rec.industry,
                  mapping_confidence=match["confidence"], mapping_score=match["score"], usage="pit")
    out = [EntityRecord(**common, mapping_source="gleif_name_match", valid_from=r.get("initial_registration"),
                        evidence={"official_valid_from": "GLEIF initialRegistrationDate" if r.get("initial_registration")
                                  else None, "reason": match["reason"]})]
    parent = direct["parent_lei"] if direct and direct.get("status") in (None, "ACTIVE") else None
    up = ultimate["parent_lei"] if ultimate and ultimate.get("status") in (None, "ACTIVE") else None
    if parent or up:
        start = (direct or {}).get("start") or (ultimate or {}).get("start")
        out.append(EntityRecord(**common, mapping_source="gleif_parent",
                                parent_entity=f"lei:{parent}" if parent else None,
                                ultimate_parent=f"lei:{up}" if up else None,
                                valid_from=start, valid_to=(direct or {}).get("end"),
                                evidence={"official_valid_from": "GLEIF relationship period start" if start else None}))
    return out


def utc_today() -> str:
    return datetime.now(timezone.utc).date().isoformat()

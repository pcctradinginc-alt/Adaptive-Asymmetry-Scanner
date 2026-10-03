"""SEC Deep Event Intelligence – Konnektoren (Research/SHADOW, nie Produktion).

Zwei offizielle, kostenlose, revisionsfreie Quellen:

1. Form-3/4/5-Datensätze der SEC (vierteljährliche ZIPs, ab 2006):
   https://www.sec.gov/files/structureddata/data/form-345-data-sets/{YYYY}q{Q}_form345.zip
   Tabellen SUBMISSION / REPORTINGOWNER / NONDERIV_TRANS.
   -> Open-Market-Käufe (P) und -Verkäufe (S) je Emittent.
   PIT: FILING_DATE ist ein Datum ohne Uhrzeit -> available_at = FILING_DATE + 1 Tag
   (CONSERVATIVE_DATE). Ursprungs-Filings zählen, Amendments (4/A) nicht
   (sie ersetzen nachträglich – wir verwenden, was damals bekannt war).

2. data.sec.gov/submissions/CIK##########.json (+ ältere Seiten "files"):
   jedes Filing mit acceptanceDateTime (offizieller Zeitstempel -> EXACT_TIMESTAMP),
   form, filingDate, reportDate, items (8-K-Items).
   Filings sind unveränderlich (Korrekturen sind eigene /A-Filings).

Keine Texte als Alpha-Faktor; Textänderungen (10-K/10-Q) sind ein eigener,
späterer Schritt (docs/SEC_EVENT_INTELLIGENCE.md).
"""
from __future__ import annotations

import csv
import io
import logging
import zipfile
from datetime import datetime, timedelta, timezone

from modules.external.http import SchemaError
from modules.external.pit import AvailabilityPrecision, Observation, ensure_utc, payload_hash

log = logging.getLogger(__name__)

PARSER_VERSION = "sec-events-1"
FORM345_URL = "https://www.sec.gov/files/structureddata/data/form-345-data-sets/{year}q{q}_form345.zip"
# Offizielle Übersichtsseiten der Insider-Datensätze; die ZIP-Links werden von dort gelesen
# (CI 2026-10-03: der feste URL-Aufbau lieferte für alle Quartale 404).
FORM345_INDEX_PAGES = ("https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets",
                       "https://www.sec.gov/dera/data/form-345")
SUBMISSIONS_URL = "https://data.sec.gov/submissions/{name}"
OPEN_MARKET_CODES = {"P": "insider_buy_usd", "S": "insider_sell_usd"}
PERIODIC_FORMS = {"10-K", "10-Q"}
LATE_FORMS = {"NT 10-K", "NT 10-Q"}
EVENT_FORMS = {"8-K"}

_DATE_FORMATS = ("%d-%b-%Y", "%Y-%m-%d", "%m/%d/%Y", "%d-%b-%y")


def parse_date(s: str | None) -> datetime | None:
    if not s:
        return None
    s = str(s).strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s[:11] if fmt == "%d-%b-%Y" else s[:10], fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _tsv(zf: zipfile.ZipFile, name: str) -> list[dict]:
    match = [n for n in zf.namelist() if n.split("/")[-1].upper() == name.upper()]
    if not match:
        raise SchemaError(f"form345: {name} fehlt im ZIP")
    with zf.open(match[0]) as fh:
        return list(csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8", errors="replace"), delimiter="\t"))


def parse_form345_zip(content: bytes, issuer_ciks: set[str] | None, retrieved_at: datetime) -> list[Observation]:
    """ZIP-Bytes -> Observations (je Open-Market-Transaktion eine). issuer_ciks
    (10-stellig) begrenzt auf das Research-Universum (None = alle)."""
    zf = zipfile.ZipFile(io.BytesIO(content))
    subs = _tsv(zf, "SUBMISSION.tsv")
    need = {"ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK"}
    if subs and not need <= set(subs[0]):
        raise SchemaError(f"form345 SUBMISSION: Spalten {need - set(subs[0])} fehlen")
    sub_by_acc = {}
    for s in subs:
        cik = str(s.get("ISSUERCIK") or "").strip()
        if not cik.isdigit():
            continue
        cik = cik.zfill(10)
        if s.get("DOCUMENT_TYPE", "").strip() != "4" or (issuer_ciks is not None and cik not in issuer_ciks):
            continue                                       # nur Original-Form-4 (keine /A, keine Form 3/5)
        sub_by_acc[s["ACCESSION_NUMBER"]] = {**s, "_cik": cik}
    if not sub_by_acc:
        return []
    owners: dict[str, list[dict]] = {}
    for o in _tsv(zf, "REPORTINGOWNER.tsv"):
        if o.get("ACCESSION_NUMBER") in sub_by_acc:
            owners.setdefault(o["ACCESSION_NUMBER"], []).append(o)
    trans = _tsv(zf, "NONDERIV_TRANS.tsv")
    tneed = {"ACCESSION_NUMBER", "TRANS_CODE", "TRANS_DATE", "TRANS_SHARES", "TRANS_PRICEPERSHARE"}
    if trans and not tneed <= set(trans[0]):
        raise SchemaError(f"form345 NONDERIV_TRANS: Spalten {tneed - set(trans[0])} fehlen")
    out: list[Observation] = []
    for t in trans:
        acc = t.get("ACCESSION_NUMBER")
        s = sub_by_acc.get(acc)
        code = (t.get("TRANS_CODE") or "").strip()
        if s is None or code not in OPEN_MARKET_CODES:
            continue
        try:
            shares, price = float(t["TRANS_SHARES"]), float(t["TRANS_PRICEPERSHARE"])
        except (TypeError, ValueError):
            continue                                       # fehlender Preis/Stückzahl -> kein Wert erfinden
        filed = parse_date(s["FILING_DATE"])
        tdate = parse_date(t.get("TRANS_DATE")) or filed
        if filed is None:
            continue
        own = owners.get(acc) or []
        out.append(Observation(
            source_id="sec_form345", dataset="nonderiv_trans",
            series_id=f"{acc}#{t.get('NONDERIV_TRANS_SK') or len(out)}", entity_id=f"cik:{s['_cik']}",
            metric=OPEN_MARKET_CODES[code], value=round(shares * price, 2), unit="USD",
            observation_time=tdate, available_at=filed + timedelta(days=1), retrieved_at=retrieved_at,
            availability_precision=AvailabilityPrecision.CONSERVATIVE_DATE, parser_version=PARSER_VERSION,
            source_release_time=filed, payload_hash=payload_hash(f"{acc}|{t.get('NONDERIV_TRANS_SK')}".encode()),
            attrs={"accession": acc, "owner_ciks": sorted({o.get("RPTOWNERCIK", "") for o in own}),
                   "owner_relationship": sorted({o.get("RPTOWNER_RELATIONSHIP", "") for o in own}),
                   "shares": shares, "price": price, "code": code}))
    return out


PUBLICATION_LAG_DAYS = 30                # SEC stellt das Quartals-ZIP erst einige Zeit nach Quartalsende bereit


def parse_form345_index(html: str, base: str = "https://www.sec.gov") -> dict[tuple[int, int], str]:
    """ZIP-Links der Übersichtsseite -> {(Jahr, Quartal): absolute URL}."""
    import re
    out: dict[tuple[int, int], str] = {}
    for href in re.findall(r'href=["\']([^"\']+?\.zip)["\']', html, flags=re.I):
        m = re.search(r"(\d{4})q([1-4])[^/]*form[_-]?345[^/]*\.zip$", href, flags=re.I) \
            or re.search(r"form[_-]?345[^/]*?(\d{4})q([1-4])[^/]*\.zip$", href, flags=re.I)
        if not m:
            continue
        url = href if href.startswith("http") else base.rstrip("/") + "/" + href.lstrip("/")
        out[(int(m.group(1)), int(m.group(2)))] = url
    return out


def form345_quarters(start_year: int, today: datetime) -> list[tuple[int, int]]:
    """Alle Quartale, deren Ende mindestens PUBLICATION_LAG_DAYS zurückliegt."""
    out = []
    for y in range(start_year, today.year + 1):
        for q in range(1, 5):
            q_end = datetime(y + (q == 4), (3 * q) % 12 + 1, 1, tzinfo=timezone.utc)
            if q_end + timedelta(days=PUBLICATION_LAG_DAYS) <= today:
                out.append((y, q))
    return out


# ── Submissions (8-K, Periodenberichte, Spätmeldungen) ───────────────────────

def parse_submissions_filings(payload: dict) -> list[dict]:
    """Spaltenarrays (filings.recent ODER eine ältere Seite) -> Zeilen."""
    block = (payload.get("filings") or {}).get("recent") if "filings" in payload else payload
    if not isinstance(block, dict) or "form" not in block or "accessionNumber" not in block:
        raise SchemaError("submissions: Spaltenblock mit form/accessionNumber fehlt")
    cols = ("accessionNumber", "filingDate", "reportDate", "acceptanceDateTime", "form", "items")
    n = len(block["form"])
    if any(len(block.get(c) or [None] * n) != n for c in cols if c in block):
        raise SchemaError("submissions: Spaltenlängen ungleich")
    return [{c: (block.get(c) or [None] * n)[i] for c in cols} for i in range(n)]


def submissions_pages(payload: dict) -> list[str]:
    return [f["name"] for f in ((payload.get("filings") or {}).get("files") or []) if f.get("name")]


def filings_to_observations(cik: str, rows: list[dict], retrieved_at: datetime,
                            since: datetime | None = None) -> list[Observation]:
    cik = str(int(cik)).zfill(10)
    out = []
    for r in rows:
        form = (r.get("form") or "").strip()
        if form not in EVENT_FORMS | PERIODIC_FORMS | LATE_FORMS:
            continue
        acc_t = ensure_utc(r["acceptanceDateTime"]) if r.get("acceptanceDateTime") else None
        filed = parse_date(r.get("filingDate"))
        if acc_t is None and filed is None:
            continue
        avail = acc_t or (filed + timedelta(days=1))
        if since is not None and avail < since:
            continue
        report = parse_date(r.get("reportDate"))
        delay = (filed - report).days if (filed and report and form in PERIODIC_FORMS) else None
        out.append(Observation(
            source_id="sec_submissions", dataset="filings", series_id=r["accessionNumber"], entity_id=f"cik:{cik}",
            metric="filing", value=float(delay) if delay is not None else 1.0,
            unit="days_after_period" if delay is not None else "count",
            observation_time=acc_t or filed, available_at=avail, retrieved_at=retrieved_at,
            availability_precision=(AvailabilityPrecision.EXACT_TIMESTAMP if acc_t
                                    else AvailabilityPrecision.CONSERVATIVE_DATE),
            parser_version=PARSER_VERSION, source_release_time=acc_t or filed,
            payload_hash=payload_hash(r["accessionNumber"].encode()),
            attrs={"form": form, "items": [i.strip() for i in str(r.get("items") or "").split(",") if i.strip()],
                   "report_date": (r.get("reportDate") or None), "filing_date": r.get("filingDate")}))
    return out

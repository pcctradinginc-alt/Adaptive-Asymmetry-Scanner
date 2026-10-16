"""Unternehmensbezogene öffentliche Ereignisdaten (RESEARCH/SHADOW, 2026-10-15) – nur Archiv, keine Champion-Wirkung.

  clinicaltrials_sponsors  B8  ClinicalTrials.gov API v2: laufende Studien je börsennotiertem Sponsor (Healthcare-SIC)
                               – aktive Studien, aktive Phase-3, Primärabschluss in 90 T, Aktualisierungen in 7 T
  usaspending_contracts    B9  USAspending.gov API v2: Bundesaufträge (Contracts A–D) mit Aktivität in den letzten
                               7 Tagen, Empfänger -> Ticker über die Entity-Map (Konzern + GLEIF-Töchter)
  sec_13f_holdings         B10 SEC Form-13F-Datensätze (DERA, Quartals-ZIPs): je Emittent und Berichtsquartal Zahl
                               der Institutionen und gehaltene Stücke, nur fristgerechte 13F-HR (<= 45 T)

Zuordnung Name -> Emittent: normalize_name der Entity-Map (exakt bzw. Präfix bei Sponsoren); Abdeckung wird je Lauf
gemeldet, unsichere Treffer werden verworfen statt geraten. PIT: Momentaufnahmen gelten ab retrieved_at; 13F-Aggregate
ab dem Tag nach dem letzten einbezogenen Filing (die Einzel-Filings waren dann öffentlich). Fehlendes bleibt fehlend.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import re
import time
import zipfile
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone

from modules.external import http
from modules.external.pit import AvailabilityPrecision, Observation
from modules.external.sources.base import Connector, ConnectorResult, RawRecord, SourceStatus

log = logging.getLogger(__name__)
PARSER_VERSION = "company-events-1"
SEC_HEADERS = {"User-Agent": "AdaptiveAsymmetryScanner research@pcctrading.com", "Accept-Encoding": "gzip, deflate"}
HEALTHCARE_SIC = {"2833", "2834", "2835", "2836", "3841", "3842", "3843", "3844", "3845", "3851", "8731", "8071"}


def _raw(source_id: str, dataset: str, res: http.FetchResult) -> RawRecord:
    return RawRecord(source_id=source_id, dataset=dataset, url=res.url, fingerprint=res.fingerprint,
                     retrieved_at=res.retrieved_at, status_code=res.status, content_type=res.content_type,
                     content_hash=res.content_hash, bytes=res.bytes)


def _day(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


def _norm(name: str | None) -> str:
    from modules.entity_resolution.store import normalize_name
    s = normalize_name(name)
    return re.sub(r"(\s+(del|new|nv|md|cl [abc]|class [abc]|com))+$", "", s).strip()


def entity_index(records=None) -> dict:
    """-> {"tickers": {cik: [ticker]}, "names": {norm_name: cik}, "subsidiaries": {norm_name: cik},
    "sic": {cik: "2834"}}. Mehrdeutige Namen (zwei Emittenten) werden verworfen."""
    if records is None:
        from modules.entity_resolution.store import DEFAULT_PATH, EntityStore
        records = EntityStore(DEFAULT_PATH).records
    tickers: dict[str, set] = defaultdict(set)
    for r in records:
        if r.usage == "research_ticker_identity" and r.valid_to is None and r.ticker and r.cik:
            tickers[f"cik:{r.cik}"].add(r.ticker)
    names: dict[str, set] = defaultdict(set)
    subs: dict[str, set] = defaultdict(set)
    sic: dict[str, str] = {}
    for r in records:
        if r.valid_to is not None:
            continue
        if r.usage == "exposure_subsidiary" and r.parent_entity in tickers:
            for n in [r.canonical_name, *(r.aliases or [])]:
                if _norm(n):
                    subs[_norm(n)].add(r.parent_entity)
        elif r.entity_id in tickers:
            for n in [r.canonical_name, *(r.aliases or [])]:
                if _norm(n):
                    names[_norm(n)].add(r.entity_id)
            m = re.match(r"SIC (\d{4})", r.industry or "")
            if m:
                sic.setdefault(r.entity_id, m.group(1))
    return {"tickers": {k: sorted(v) for k, v in tickers.items()},
            "names": {n: next(iter(c)) for n, c in names.items() if len(c) == 1},
            "subsidiaries": {n: next(iter(c)) for n, c in subs.items() if len(c) == 1 and n not in names},
            "sic": sic}


def match_name(name: str, idx: dict) -> tuple[str | None, str]:
    """-> (cik, "direct"|"subsidiary") oder (None, "")."""
    n = _norm(name)
    if n in idx["names"]:
        return idx["names"][n], "direct"
    if n in idx["subsidiaries"]:
        return idx["subsidiaries"][n], "subsidiary"
    return None, ""


# ── B8: ClinicalTrials.gov ───────────────────────────────────────────────────
CT_URL = "https://clinicaltrials.gov/api/v2/studies"
CT_ACTIVE = {"RECRUITING", "ACTIVE_NOT_RECRUITING", "NOT_YET_RECRUITING", "ENROLLING_BY_INVITATION"}
CT_FIELDS = ("NCTId,OverallStatus,Phase,PrimaryCompletionDate,LastUpdatePostDate,LeadSponsorName")


def _ct_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(s) if len(s) == 10 else date.fromisoformat(s + "-28")   # "2027-03" -> Monatsende
    except ValueError:
        return None


def parse_studies(payload: dict) -> tuple[list[dict], str | None]:
    studies = (payload or {}).get("studies")
    if not isinstance(studies, list):
        raise ValueError("clinicaltrials: 'studies' fehlt")
    out = []
    for s in studies:
        ps = (s or {}).get("protocolSection") or {}
        st = ps.get("statusModule") or {}
        out.append({"nct": (ps.get("identificationModule") or {}).get("nctId"),
                    "status": st.get("overallStatus"),
                    "phases": (ps.get("designModule") or {}).get("phases") or [],
                    "primary_completion": _ct_date((st.get("primaryCompletionDateStruct") or {}).get("date")),
                    "last_update": _ct_date((st.get("lastUpdatePostDateStruct") or {}).get("date")),
                    "sponsor": ((ps.get("sponsorCollaboratorsModule") or {}).get("leadSponsor") or {}).get("name")})
    return out, payload.get("nextPageToken")


def sponsor_matches(sponsor: str | None, company_norm: str) -> bool:
    s = _norm(sponsor)
    return bool(s) and bool(company_norm) and (s == company_norm or s.startswith(company_norm + " ")
                                                or company_norm.startswith(s + " "))


def trial_metrics(studies: list[dict], today: date) -> dict[str, float]:
    act = [s for s in studies if s["status"] in CT_ACTIVE]
    return {"ct_active_trials": float(len(act)),
            "ct_active_phase3": float(sum(1 for s in act if "PHASE3" in s["phases"])),
            "ct_primary_completion_90d": float(sum(1 for s in act if s["primary_completion"]
                                                   and today <= s["primary_completion"] <= today + timedelta(days=90))),
            "ct_updated_7d": float(sum(1 for s in studies if s["last_update"]
                                       and s["last_update"] >= today - timedelta(days=7)))}


class ClinicalTrialsConnector(Connector):
    source_id = "clinicaltrials_sponsors"
    parser_version = PARSER_VERSION
    MAX_PAGES = 5

    def fetch(self, now: datetime) -> ConnectorResult:
        try:
            idx = entity_index()
        except Exception as e:  # noqa: BLE001
            return ConnectorResult(self.source_id, SourceStatus.FAIL, message=f"Entity-Map fehlt: {e}")
        companies = {cik: n for n, cik in idx["names"].items() if idx["sic"].get(cik) in HEALTHCARE_SIC}
        # je Emittent der kürzeste Name (meist der Kurzname des Sponsors)
        best: dict[str, str] = {}
        for cik, n in companies.items():
            if cik not in best or len(n) < len(best[cik]):
                best[cik] = n
        obs, raw, unmatched = [], [], 0
        for cik, name in sorted(best.items()):
            studies, token = [], None
            for _ in range(self.MAX_PAGES):
                params = {"query.spons": name, "fields": CT_FIELDS, "pageSize": 1000, "format": "json"}
                if token:
                    params["pageToken"] = token
                try:
                    res = http.fetch(CT_URL, params=params, retries=2)
                    page, token = parse_studies(res.json())
                except http.FetchError as e:
                    return ConnectorResult(self.source_id, SourceStatus.FAIL, raw=raw, message=f"ClinicalTrials: {e}")
                except (ValueError, json.JSONDecodeError) as e:
                    return ConnectorResult(self.source_id, SourceStatus.SCHEMA_CHANGED, raw=raw, message=str(e))
                if not raw:
                    raw.append(_raw(self.source_id, "studies", res))
                studies += [s for s in page if sponsor_matches(s["sponsor"], name)]
                if not token:
                    break
            if not studies:
                unmatched += 1
                continue
            t = _day(res.retrieved_at.date())
            for metric, val in trial_metrics(studies, res.retrieved_at.date()).items():
                for tk in idx["tickers"].get(cik, []):
                    obs.append(Observation(
                        source_id=self.source_id, dataset="sponsor_pipeline", series_id=f"{cik}:{metric}",
                        entity_id=tk, metric=metric, value=val, unit="count", observation_time=t,
                        available_at=res.retrieved_at, retrieved_at=res.retrieved_at,
                        availability_precision=AvailabilityPrecision.EXACT_TIMESTAMP,
                        parser_version=self.parser_version, attrs={"sponsor_query": name, "cik": cik}))
            time.sleep(float(self.cfg.get("request_pause_seconds", 0.2)))
        msg = f"{len(best) - unmatched}/{len(best)} Healthcare-Emittenten mit Studien, {len(obs)} Kennzahlen"
        return ConnectorResult(self.source_id, SourceStatus.PASS if obs else SourceStatus.WARN, observations=obs,
                               raw=raw, message=msg,
                               latest_observation_time=max((o.observation_time for o in obs), default=None))


# ── B9: USAspending.gov ──────────────────────────────────────────────────────
USASPENDING_URL = "https://api.usaspending.gov/api/v2/search/spending_by_award/"
AWARD_FIELDS = ["Award ID", "Recipient Name", "Award Amount", "Awarding Agency", "generated_internal_id"]


def parse_awards(payload: dict) -> tuple[list[dict], bool]:
    res = (payload or {}).get("results")
    if not isinstance(res, list):
        raise ValueError("usaspending: 'results' fehlt")
    if res and not all(f in res[0] for f in ("Recipient Name", "Award Amount")):
        raise ValueError(f"usaspending: Felder fehlen ({sorted(res[0])[:10]})")
    return res, bool(((payload.get("page_metadata") or {}).get("hasNext")))


class UsaSpendingConnector(Connector):
    source_id = "usaspending_contracts"
    parser_version = PARSER_VERSION

    def fetch(self, now: datetime) -> ConnectorResult:
        try:
            idx = entity_index()
        except Exception as e:  # noqa: BLE001
            return ConnectorResult(self.source_id, SourceStatus.FAIL, message=f"Entity-Map fehlt: {e}")
        days = int(self.cfg.get("window_days", 7))
        end, start = now.date() - timedelta(days=1), now.date() - timedelta(days=days)
        awards, raw = [], []
        for page in range(1, int(self.cfg.get("max_pages", 10)) + 1):
            body = {"filters": {"award_type_codes": ["A", "B", "C", "D"],
                                "time_period": [{"start_date": start.isoformat(), "end_date": end.isoformat()}]},
                    "fields": AWARD_FIELDS, "page": page, "limit": 100, "sort": "Award Amount", "order": "desc",
                    "subawards": False}
            try:
                res = http.fetch(USASPENDING_URL, method="POST", json_body=body, retries=2, timeout=60)
                rows, has_next = parse_awards(res.json())
            except http.FetchError as e:
                return ConnectorResult(self.source_id, SourceStatus.FAIL, raw=raw, message=f"USAspending: {e}")
            except (ValueError, json.JSONDecodeError) as e:
                return ConnectorResult(self.source_id, SourceStatus.SCHEMA_CHANGED, raw=raw, message=str(e))
            if not raw:
                raw.append(_raw(self.source_id, "spending_by_award", res))
            awards += rows
            if not has_next:
                break
        agg: dict[str, dict] = defaultdict(lambda: {"usd": 0.0, "n": 0, "via_subsidiary": 0, "ids": []})
        for a in awards:
            cik, how = match_name(a.get("Recipient Name"), idx)
            if not cik or a.get("Award Amount") is None:
                continue
            g = agg[cik]
            g["usd"] += float(a["Award Amount"])
            g["n"] += 1
            g["via_subsidiary"] += how == "subsidiary"
            if len(g["ids"]) < 10:
                g["ids"].append(a.get("Award ID"))
        t = _day(now.date())
        retrieved = raw[0].retrieved_at if raw else now
        obs = []
        for cik, g in sorted(agg.items()):
            for tk in idx["tickers"].get(cik, []):
                for metric, val, unit in ((f"federal_contracts_active_{days}d_usd", round(g["usd"], 2), "usd"),
                                          (f"federal_contracts_active_{days}d_count", float(g["n"]), "count")):
                    obs.append(Observation(
                        source_id=self.source_id, dataset="contract_awards", series_id=f"{cik}:{metric}",
                        entity_id=tk, metric=metric, value=val, unit=unit, observation_time=t,
                        available_at=retrieved, retrieved_at=retrieved,
                        availability_precision=AvailabilityPrecision.EXACT_TIMESTAMP,
                        parser_version=self.parser_version,
                        attrs={"cik": cik, "via_subsidiary": g["via_subsidiary"], "award_ids": g["ids"],
                               "window": [start.isoformat(), end.isoformat()]}))
        return ConnectorResult(self.source_id, SourceStatus.PASS if awards else SourceStatus.WARN, observations=obs,
                               raw=raw, message=f"{len(awards)} Aufträge, {len(agg)} Emittenten zugeordnet",
                               latest_observation_time=t if obs else None)


# ── B10: SEC Form-13F-Datensätze ─────────────────────────────────────────────
F13_INDEX_PAGES = ("https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets",
                   "https://www.sec.gov/dera/data/form-13f")
TIMELY_DAYS = 45
_MON = {m: i for i, m in enumerate(("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov",
                                    "dec"), start=1)}


def parse_13f_index(html: str, base: str = "https://www.sec.gov") -> dict[str, str]:
    """Dateiname -> absolute URL aller *_form13f.zip-Links."""
    out = {}
    for href in re.findall(r'href="([^"]*?form13f[^"]*?\.zip)"', html or "", flags=re.I):
        url = href if href.startswith("http") else base + href
        out[url.rsplit("/", 1)[-1]] = url
    return out


def set_range(name: str) -> tuple[date, date] | None:
    """'01jun2024-31aug2024_form13f.zip' bzw. '2023q4_form13f.zip' -> Filing-Zeitraum (inklusive)."""
    m = re.match(r"(\d{2})([a-z]{3})(\d{4})-(\d{2})([a-z]{3})(\d{4})_form13f", name.lower())
    if m:
        try:
            return (date(int(m.group(3)), _MON[m.group(2)], int(m.group(1))),
                    date(int(m.group(6)), _MON[m.group(5)], int(m.group(4))))
        except (KeyError, ValueError):
            return None
    m = re.match(r"(\d{4})q([1-4])_form13f", name.lower())
    if m:
        y, q = int(m.group(1)), int(m.group(2))
        start = date(y, 3 * q - 2, 1)
        end = date(y + (q == 4), (3 * q) % 12 + 1, 1) - timedelta(days=1)
        return start, end
    return None


def _d(s: str | None) -> date | None:
    from modules.external.sources.sec_events import parse_date
    t = parse_date(s)
    return t.date() if t else None


def _tsv(zf: zipfile.ZipFile, suffix: str):
    name = next((n for n in zf.namelist() if n.upper().endswith(suffix)), None)
    if name is None:
        raise ValueError(f"13F-ZIP ohne {suffix}")
    with zf.open(name) as fh:
        yield from csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8", errors="replace"), delimiter="\t")


def aggregate_13f(content: bytes, rng: tuple[date, date], idx: dict) -> tuple[list[dict], dict]:
    """-> ([{cik, period, holders, shares, available}], Statistik). Nur 13F-HR (keine /A, keine NT), nur
    fristgerecht (Periode < FILING_DATE <= Periode + 45 T) und nur Perioden, deren gesamtes Fristfenster im
    Datensatz-Zeitraum liegt; Optionen (PUTCALL) und Nicht-Aktien (SSHPRNAMTTYPE != SH) ausgeschlossen."""
    zf = zipfile.ZipFile(io.BytesIO(content))
    subs = {}
    for r in _tsv(zf, "SUBMISSION.TSV"):
        if (r.get("SUBMISSIONTYPE") or "").strip() != "13F-HR":
            continue
        filed, period = _d(r.get("FILING_DATE")), _d(r.get("PERIODOFREPORT"))
        if not filed or not period or not (period < filed <= period + timedelta(days=TIMELY_DAYS)):
            continue
        if not (rng[0] <= period + timedelta(days=1) and period + timedelta(days=TIMELY_DAYS) <= rng[1]):
            continue
        subs[r["ACCESSION_NUMBER"]] = (period, filed, (r.get("CIK") or "").strip())
    by_cusip: dict[tuple, dict] = {}
    for r in _tsv(zf, "INFOTABLE.TSV"):
        sub = subs.get(r.get("ACCESSION_NUMBER"))
        if sub is None or (r.get("PUTCALL") or "").strip() or (r.get("SSHPRNAMTTYPE") or "").strip() != "SH":
            continue
        try:
            sh = float(r.get("SSHPRNAMT") or "")
        except ValueError:
            continue
        g = by_cusip.setdefault((sub[0], (r.get("CUSIP") or "").strip().upper()),
                                {"names": Counter(), "filers": set(), "shares": 0.0, "last_filed": sub[1]})
        g["names"][r.get("NAMEOFISSUER") or ""] += 1
        g["filers"].add(sub[2])
        g["shares"] += sh
        g["last_filed"] = max(g["last_filed"], sub[1])
    per_cik: dict[tuple, dict] = {}
    mapped = 0
    for (period, _cusip), g in by_cusip.items():
        cik, how = match_name(g["names"].most_common(1)[0][0], idx)
        if not cik or how != "direct":
            continue
        mapped += 1
        a = per_cik.setdefault((cik, period), {"filers": set(), "shares": 0.0, "last_filed": g["last_filed"]})
        a["filers"] |= g["filers"]
        a["shares"] += g["shares"]
        a["last_filed"] = max(a["last_filed"], g["last_filed"])
    rows = [{"cik": cik, "period": period, "holders": float(len(a["filers"])), "shares": a["shares"],
             "available": a["last_filed"] + timedelta(days=1)} for (cik, period), a in sorted(per_cik.items())]
    return rows, {"filings": len(subs), "cusips": len(by_cusip), "cusips_mapped": mapped, "issuers": len(per_cik)}


class Sec13FConnector(Connector):
    source_id = "sec_13f_holdings"
    parser_version = PARSER_VERSION

    def _processed(self) -> set[str]:
        """Bereits archivierte Datensätze (Marker-Zeilen im Archiv) – Preflight archiviert nichts, zählt also nie."""
        try:
            from modules.external.archive import DEFAULT_ROOT, ExternalArchive
            rows = ExternalArchive(self.cfg.get("_archive_root") or DEFAULT_ROOT).load(self.source_id)
        except Exception as e:  # noqa: BLE001
            log.warning("13F: Archiv nicht lesbar (%r) – behandle als Erstimport", e)
            return set()
        return {o.series_id.split(":", 1)[1] for o in rows if o.metric == "dataset_processed"}

    def fetch(self, now: datetime) -> ConnectorResult:
        try:
            idx = entity_index()
        except Exception as e:  # noqa: BLE001
            return ConnectorResult(self.source_id, SourceStatus.FAIL, message=f"Entity-Map fehlt: {e}")
        sets, raw = {}, []
        for page in F13_INDEX_PAGES:
            try:
                res = http.fetch(page, headers=SEC_HEADERS, retries=2)
            except http.FetchError as e:
                log.warning("13F-Index %s: %s", page, e)
                continue
            raw.append(_raw(self.source_id, "index", res))
            sets.update(parse_13f_index(res.content.decode("utf-8", "replace")))
        sets = {n: u for n, u in sets.items() if set_range(n)}
        if not sets:
            return ConnectorResult(self.source_id, SourceStatus.SCHEMA_CHANGED, raw=raw,
                                   message="keine *_form13f.zip-Links auf den SEC-Indexseiten")
        done = self._processed()
        newest = sorted(sets, key=lambda n: set_range(n)[1], reverse=True)[:int(self.cfg.get("history_sets", 8))]
        todo = [n for n in newest if n not in done][:int(self.cfg.get("max_sets_per_run", 1))]
        # Datenstand = jüngster verarbeiteter Datensatz (inkl. dieses Laufs): der Backfill läuft vom neuesten zum ältesten
        # Satz, ein älterer Satz darf den gemeldeten Stand nicht zurücksetzen (sonst Source Health STALE, Live 2026-10-16)
        latest = max((_day(set_range(n)[1]) for n in done | set(todo) if set_range(n)), default=None)
        if not todo:
            return ConnectorResult(self.source_id, SourceStatus.PASS, raw=raw,
                                   message=f"kein neuer 13F-Datensatz ({len(done)} verarbeitet)",
                                   latest_observation_time=latest)
        obs, stats = [], {}
        for name in todo:
            try:
                res = http.fetch(sets[name], headers=SEC_HEADERS, retries=2, timeout=300)
                rows, stats[name] = aggregate_13f(res.content, set_range(name), idx)
            except http.FetchError as e:
                return ConnectorResult(self.source_id, SourceStatus.FAIL, raw=raw, message=f"{name}: {e}")
            except (ValueError, zipfile.BadZipFile, KeyError) as e:
                return ConnectorResult(self.source_id, SourceStatus.SCHEMA_CHANGED, raw=raw, message=f"{name}: {e}")
            raw.append(_raw(self.source_id, "dataset", res))
            obs.append(Observation(                              # Marker: Datensatz vollständig verarbeitet
                source_id=self.source_id, dataset="form13f", series_id=f"set:{name}", entity_id="SEC13F",
                metric="dataset_processed", value=float(stats[name]["issuers"]), unit="count",
                observation_time=_day(set_range(name)[1]), available_at=res.retrieved_at,
                retrieved_at=res.retrieved_at, availability_precision=AvailabilityPrecision.EXACT_TIMESTAMP,
                parser_version=self.parser_version, attrs={k: v for k, v in stats[name].items()}))
            for r in rows:
                for metric, val, unit in (("inst_holders", r["holders"], "count"), ("inst_shares", r["shares"],
                                                                                   "shares")):
                    obs.append(Observation(
                        source_id=self.source_id, dataset="form13f", series_id=f"{r['cik']}:{metric}",
                        entity_id=r["cik"], metric=metric, value=val, unit=unit,
                        observation_time=_day(r["period"]), available_at=_day(r["available"]),
                        retrieved_at=res.retrieved_at, availability_precision=AvailabilityPrecision.CONSERVATIVE_DATE,
                        parser_version=self.parser_version,
                        attrs={"tickers": idx["tickers"].get(r["cik"], []), "dataset": name}))
        return ConnectorResult(self.source_id, SourceStatus.PASS if obs else SourceStatus.WARN, observations=obs,
                               raw=raw, message=f"{todo}: {stats}", latest_observation_time=latest)


CONNECTORS: dict[str, type[Connector]] = {
    "clinicaltrials_sponsors": ClinicalTrialsConnector,
    "usaspending_contracts": UsaSpendingConnector,
    "sec_13f_holdings": Sec13FConnector,
}

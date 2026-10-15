"""Makro-Veröffentlichungskalender (B7, RESEARCH/SHADOW, 2026-10-15) – geplante Termine als PIT-Momentaufnahme.

  fomc_calendar          Federal Reserve: FOMC-Sitzungen (HTML fomccalendars.htm); Entscheid = letzter Sitzungstag
                         14:00 New York; Sitzungen mit Projektionen (SEP, "*") markiert
  bls_release_calendar   BLS: Veröffentlichungsplan als iCalendar (bls.ics) – CPI, Arbeitsmarkt, PPI, …
  bea_release_calendar   BEA: Veröffentlichungsplan als iCalendar – BIP, PCE/Persönliches Einkommen, …

PIT: Ein Termin ist ab dem Abruf bekannt (available_at = retrieved_at). observation_time = geplanter Zeitpunkt
(UTC). Verschiebungen werden als neuer Wert/neue Zeile sichtbar, nie rückwirkend. Keine Produktionswirkung –
Zweck: Event-Fenster für Research/Labels (z. B. "Signal innerhalb von 24 h vor CPI/FOMC").
"""
from __future__ import annotations

import logging
import re
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from modules.external import http
from modules.external.pit import AvailabilityPrecision, Observation
from modules.external.sources.base import Connector, ConnectorResult, RawRecord, SourceStatus

log = logging.getLogger(__name__)
PARSER_VERSION = "calendars-1"
NY = ZoneInfo("America/New_York")
FOMC_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
BLS_ICS_URL = "https://www.bls.gov/schedule/news_release/bls.ics"
BEA_ICS_URL = "https://www.bea.gov/news/schedule/ical"
CAL_HEADERS = {"User-Agent": "AdaptiveAsymmetryScanner/1.0 (research; contact: research@pcctrading.com)",
               "Accept": "text/html,text/calendar,*/*"}
MONTHS = {m: i for i, m in enumerate(("january", "february", "march", "april", "may", "june", "july", "august",
                                      "september", "october", "november", "december"), start=1)}
MONTHS.update({k[:3]: v for k, v in list(MONTHS.items())})


def _raw(source_id: str, dataset: str, res: http.FetchResult) -> RawRecord:
    return RawRecord(source_id=source_id, dataset=dataset, url=res.url, fingerprint=res.fingerprint,
                     retrieved_at=res.retrieved_at, status_code=res.status, content_type=res.content_type,
                     content_hash=res.content_hash, bytes=res.bytes)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:80]


# ── FOMC ─────────────────────────────────────────────────────────────────────
_FOMC_YEAR = re.compile(r"(\d{4})\s+FOMC\s+Meetings", re.I)
_FOMC_MEETING = re.compile(
    r'fomc-meeting__month[^>]*>\s*(?:<strong>)?\s*([A-Za-z/]+)\s*(?:</strong>)?\s*</div>\s*'
    r'<div[^>]*fomc-meeting__date[^>]*>\s*([^<]+?)\s*</div>', re.S)


def parse_fomc(html: str) -> list[dict]:
    """-> [{decision_et (datetime NY), sep (bool), label}] für alle Jahres-Panels der Seite."""
    out = []
    years = [(m.start(), int(m.group(1))) for m in _FOMC_YEAR.finditer(html)]
    if not years:
        raise ValueError("FOMC: keine Jahres-Überschrift 'YYYY FOMC Meetings'")
    for m in _FOMC_MEETING.finditer(html):
        year = next((y for pos, y in reversed(years) if pos < m.start()), None)
        if year is None:
            continue
        months = [MONTHS.get(x.strip().lower()[:3]) for x in m.group(1).split("/")]
        raw_days = m.group(2)
        days = [int(x) for x in re.findall(r"\d{1,2}", raw_days)]
        if not days or None in months:
            continue
        if "notation" in raw_days.lower() or "unscheduled" in raw_days.lower():
            continue                                             # keine regulären Sitzungen
        month = months[-1] if len(months) > 1 and len(days) > 1 and days[-1] < days[0] else months[0]
        try:
            d = date(year, month, days[-1])
        except ValueError:
            continue
        out.append({"decision_et": datetime.combine(d, time(14, 0), NY), "sep": "*" in raw_days,
                    "label": f"FOMC {d.isoformat()}"})
    if not out:
        raise ValueError("FOMC: keine Sitzungen erkannt (HTML-Struktur geändert?)")
    return sorted(out, key=lambda e: e["decision_et"])


# ── iCalendar (BLS, BEA) ─────────────────────────────────────────────────────
def _unfold(text: str) -> list[str]:
    out: list[str] = []
    for ln in text.replace("\r\n", "\n").split("\n"):
        if ln.startswith((" ", "\t")) and out:
            out[-1] += ln[1:]
        else:
            out.append(ln)
    return out


def _ics_dt(prop: str, value: str) -> datetime | None:
    tzid = re.search(r"TZID=([^;:]+)", prop)
    v = value.strip()
    try:
        if len(v) == 8:                                          # ganztägig -> 08:30 New York (übliche Uhrzeit)
            return datetime.combine(datetime.strptime(v, "%Y%m%d").date(), time(8, 30), NY)
        if v.endswith("Z"):
            return datetime.strptime(v, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        naive = datetime.strptime(v[:15], "%Y%m%dT%H%M%S")
    except ValueError:
        return None
    try:
        tz = ZoneInfo(tzid.group(1)) if tzid else NY
    except (KeyError, ValueError):                               # z. B. "US-Eastern": US-Statistikämter = New York
        tz = NY
    return naive.replace(tzinfo=tz)


def parse_ics(text: str) -> list[dict]:
    """VEVENTs -> [{start (aware), summary, uid}]; ohne VCALENDAR -> ValueError."""
    if "BEGIN:VCALENDAR" not in (text or ""):
        raise ValueError("kein iCalendar (BEGIN:VCALENDAR fehlt)")
    events, cur = [], None
    for ln in _unfold(text):
        if ln.startswith("BEGIN:VEVENT"):
            cur = {}
        elif ln.startswith("END:VEVENT") and cur is not None:
            if cur.get("start") and cur.get("summary"):
                events.append(cur)
            cur = None
        elif cur is not None and ":" in ln:
            prop, value = ln.split(":", 1)
            name = prop.split(";")[0].upper()
            if name == "DTSTART":
                cur["start"] = _ics_dt(prop, value)
            elif name == "SUMMARY":
                cur["summary"] = value.replace("\\,", ",").replace("\\;", ";").strip()
            elif name == "UID":
                cur["uid"] = value.strip()
    return events


class _CalendarConnector(Connector):
    parser_version = PARSER_VERSION
    URL = ""
    DATASET = ""
    HORIZON_DAYS = 400                                          # nur Termine im Fenster [heute − 7, heute + 400]

    def events(self, res: http.FetchResult) -> list[dict]:  # pragma: no cover - Interface
        raise NotImplementedError

    def fetch(self, now: datetime) -> ConnectorResult:
        try:
            res = http.fetch(self.URL, headers=CAL_HEADERS, retries=2)
        except http.AuthError as e:
            return ConnectorResult(self.source_id, SourceStatus.FAIL, message=f"Zugriff verweigert ({e}); "
                                   "Quelle blockiert evtl. den User-Agent")
        except http.FetchError as e:
            return ConnectorResult(self.source_id, SourceStatus.FAIL, message=str(e))
        raw = [_raw(self.source_id, self.DATASET, res)]
        try:
            evs = self.events(res)
        except ValueError as e:
            return ConnectorResult(self.source_id, SourceStatus.SCHEMA_CHANGED, raw=raw, message=str(e))
        lo, hi = now - timedelta(days=7), now + timedelta(days=self.HORIZON_DAYS)
        obs = []
        for e in evs:
            t = e["start"].astimezone(timezone.utc)
            if not lo <= t <= hi:
                continue
            obs.append(Observation(
                source_id=self.source_id, dataset=self.DATASET, series_id=e["series"], entity_id="US",
                metric="scheduled_release", value=float(e.get("value", 1.0)), unit="event",
                observation_time=t, available_at=res.retrieved_at, retrieved_at=res.retrieved_at,
                availability_precision=AvailabilityPrecision.EXACT_TIMESTAMP, parser_version=self.parser_version,
                attrs={"title": e["title"], "local_time_et": e["start"].astimezone(NY).isoformat()}))
        if not obs:
            return ConnectorResult(self.source_id, SourceStatus.SCHEMA_CHANGED, raw=raw,
                                   message=f"keine Termine im Fenster ({len(evs)} insgesamt)")
        return ConnectorResult(self.source_id, SourceStatus.PASS, observations=obs, raw=raw,
                               message=f"{len(obs)} Termine im Fenster",
                               latest_observation_time=max(o.observation_time for o in obs))


class FomcCalendarConnector(_CalendarConnector):
    source_id = "fomc_calendar"
    URL, DATASET = FOMC_URL, "fomc_meetings"

    def events(self, res):
        return [{"start": e["decision_et"], "series": "FOMC_DECISION", "title": e["label"] + (" (SEP)" if e["sep"]
                                                                                             else ""),
                 "value": 2.0 if e["sep"] else 1.0} for e in parse_fomc(res.content.decode("utf-8", "replace"))]


class _IcsConnector(_CalendarConnector):
    def events(self, res):
        out = []
        for e in parse_ics(res.content.decode("utf-8", "replace")):
            if e.get("start") is None:
                continue
            out.append({"start": e["start"], "series": _slug(re.sub(r"\s+for\s+.*$", "", e["summary"], flags=re.I)),
                        "title": e["summary"]})
        return out


class BlsReleaseCalendarConnector(_IcsConnector):
    source_id = "bls_release_calendar"
    URL, DATASET = BLS_ICS_URL, "bls_schedule"


class BeaReleaseCalendarConnector(_IcsConnector):
    source_id = "bea_release_calendar"
    URL, DATASET = BEA_ICS_URL, "bea_schedule"


CONNECTORS: dict[str, type[Connector]] = {
    "fomc_calendar": FomcCalendarConnector,
    "bls_release_calendar": BlsReleaseCalendarConnector,
    "bea_release_calendar": BeaReleaseCalendarConnector,
}

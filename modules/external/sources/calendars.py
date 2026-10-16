"""Makro-Veröffentlichungskalender (B7, RESEARCH/SHADOW, 2026-10-15) – geplante Termine als PIT-Momentaufnahme.

  fomc_calendar          Federal Reserve: FOMC-Sitzungen (HTML fomccalendars.htm); Entscheid = letzter Sitzungstag
                         14:00 New York; Sitzungen mit Projektionen (SEP, "*") markiert
  bls_release_calendar   BLS: Veröffentlichungsplan als iCalendar (bls.ics) – CPI, Arbeitsmarkt, PPI, …
  bea_release_calendar   BEA: release_dates.json bzw. Tabelle der Plan-Seite – BIP, PCE/Persönliches Einkommen, …

PIT: Ein Termin ist ab dem Abruf bekannt (available_at = retrieved_at). observation_time = geplanter Zeitpunkt
(UTC). Der tatsächliche Ankündigungszeitpunkt ist nicht belegt und wird nie früher angesetzt: vor dem ersten Abruf
gilt ein Termin als unbekannt (für Backtests gesperrt). Keine Produktionswirkung – Zweck: Event-Fenster für
Research/Labels (z. B. "Signal innerhalb von 24 h vor CPI/FOMC").

Terminänderungen (2026-10-16): Eine Terminzeile allein kann eine Verschiebung oder Absage nicht abbilden (der alte
Termin bleibt im Archiv, unveränderte Wiederholungen werden dedupliziert). Daher legt jeder Abruf, der den Plan
ändert, eine Momentaufnahme (series_id "schedule_snapshot") mit allen Terminen im Fenster an – inkl. hinzugekommener
und zurückgezogener Termine. schedule_as_of(source_id, t) liefert den Plan, wie er zum Zeitpunkt t bekannt war.
"""
from __future__ import annotations

import json
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
BEA_JSON_URL = "https://apps.bea.gov/API/signup/release_dates.json"
BEA_SCHEDULE_URL = "https://www.bea.gov/news/schedule"
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


SNAPSHOT_SERIES = "schedule_snapshot"


def _event_key(series: str, t: datetime, value: float) -> str:
    return f"{series}|{t.astimezone(timezone.utc).isoformat()}|{value:g}"


def _parse_event_key(key: str) -> dict:
    series, iso, value = key.rsplit("|", 2)
    return {"series": series, "time": datetime.fromisoformat(iso), "value": float(value)}


def schedule_as_of(source_id: str, t: datetime, archive=None) -> list[dict] | None:
    """PIT: Plan einer Kalenderquelle, wie er zum Zeitpunkt t bekannt war (jüngste Momentaufnahme mit
    available_at <= t) -> [{series, time (UTC), value}]. None = kein Nachweis vor t (für Backtests gesperrt; wird
    nie aus späteren Terminzeilen rekonstruiert)."""
    from modules.external.archive import ExternalArchive
    from modules.external.pit import ensure_utc
    t = ensure_utc(t)
    archive = archive or ExternalArchive()
    snaps = [o for o in archive.load(source_id) if o.series_id == SNAPSHOT_SERIES and o.available_at <= t]
    if not snaps:
        return None
    snap = max(snaps, key=lambda o: o.available_at)
    return [_parse_event_key(k) for k in (snap.attrs or {}).get("events") or []]


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
                cur["all_day"] = len(value.strip()) == 8               # Uhrzeit 08:30 New York angenommen
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
            attrs = {"title": e["title"], "local_time_et": e["start"].astimezone(NY).isoformat()}
            if e.get("time_assumed"):
                attrs["time_assumed"] = True                    # ganztägiger Eintrag: Uhrzeit nicht von der Quelle
            obs.append(Observation(
                source_id=self.source_id, dataset=self.DATASET, series_id=e["series"], entity_id="US",
                metric="scheduled_release", value=float(e.get("value", 1.0)), unit="event",
                observation_time=t, available_at=res.retrieved_at, retrieved_at=res.retrieved_at,
                availability_precision=AvailabilityPrecision.EXACT_TIMESTAMP, parser_version=self.parser_version,
                attrs=attrs))
        if not obs:
            return ConnectorResult(self.source_id, SourceStatus.SCHEMA_CHANGED, raw=raw,
                                   message=f"keine Termine im Fenster ({len(evs)} insgesamt)")
        snap, change = self._snapshot(obs, res, lo, hi)
        return ConnectorResult(self.source_id, SourceStatus.PASS, observations=obs + snap, raw=raw,
                               message=f"{len(obs)} Termine im Fenster{change}",
                               latest_observation_time=max(o.observation_time for o in obs))

    def _last_snapshot(self) -> Observation | None:
        root = self.cfg.get("_archive_root")
        if not root:
            return None
        try:
            from modules.external.archive import ExternalArchive
            rows = [o for o in ExternalArchive(root).load(self.source_id) if o.series_id == SNAPSHOT_SERIES]
        except Exception as e:  # noqa: BLE001 – ohne lesbares Archiv: neue Momentaufnahme statt stiller Annahme
            log.warning("%s: Archiv nicht lesbar (%r) – neue Momentaufnahme", self.source_id, e)
            return None
        return max(rows, key=lambda o: o.available_at, default=None)

    def _snapshot(self, obs: list, res, lo: datetime, hi: datetime) -> tuple[list, str]:
        """Momentaufnahme nur bei geändertem Plan (oder ohne Vorgänger). Zurückgezogen = im alten Plan, im aktuellen
        Fenster, aber nicht mehr geliefert (verschoben/abgesagt); aus dem Fenster gealterte Termine zählen nicht."""
        keys = sorted({_event_key(o.series_id, o.observation_time, o.value) for o in obs})
        prev = self._last_snapshot()
        prev_keys = set((prev.attrs or {}).get("events") or []) if prev else set()
        added = sorted(set(keys) - prev_keys) if prev else []
        withdrawn = sorted(k for k in prev_keys - set(keys) if lo <= _parse_event_key(k)["time"] <= hi)
        if prev and not added and not withdrawn:
            return [], ""
        import hashlib
        digest = hashlib.sha256("\n".join(keys).encode()).hexdigest()[:12]
        snap = Observation(                                  # entity_id je Plan: zwei Pläne in derselben Sekunde mit
            source_id=self.source_id, dataset=self.DATASET,  # gleicher Anzahl wären sonst ein "Duplikat"
            series_id=SNAPSHOT_SERIES, entity_id=f"plan:{digest}",
            metric="scheduled_events", value=float(len(keys)), unit="count", observation_time=res.retrieved_at,
            available_at=res.retrieved_at, retrieved_at=res.retrieved_at,
            availability_precision=AvailabilityPrecision.EXACT_TIMESTAMP, parser_version=self.parser_version,
            attrs={"events": keys, "window": [lo.isoformat(), hi.isoformat()], "added": added,
                   "withdrawn": withdrawn, "previous_snapshot_at": prev.available_at.isoformat() if prev else None})
        change = (f"; Plan geändert: +{len(added)} / −{len(withdrawn)} (zurückgezogen)" if prev
                  else "; erste Momentaufnahme")
        return [snap], change


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
                        "title": e["summary"], "time_assumed": bool(e.get("all_day"))})
        return out


class BlsReleaseCalendarConnector(_IcsConnector):
    source_id = "bls_release_calendar"
    URL, DATASET = BLS_ICS_URL, "bls_schedule"


def parse_bea_json(payload) -> list[dict]:
    """release_dates.json: {Datensatz: {"release_dates": [ISO-Zeitpunkte]}} (auch verschachtelt) -> Termine."""
    out = []

    def walk(name, node):
        if isinstance(node, dict):
            dates = node.get("release_dates") or node.get("releaseDates")
            if isinstance(dates, list):
                for d in dates:
                    try:
                        t = datetime.fromisoformat(str(d).replace("Z", "+00:00"))
                    except ValueError:
                        continue
                    out.append({"start": t if t.tzinfo else t.replace(tzinfo=NY), "summary": name})
            for k, v in node.items():
                if k not in ("release_dates", "releaseDates"):
                    walk(k if isinstance(v, (dict, list)) else name, v)
        elif isinstance(node, list):
            for v in node:
                walk(name, v)

    walk("BEA", payload)
    if not out:
        raise ValueError("BEA release_dates.json: keine release_dates gefunden")
    return out


_TIME_RE = re.compile(r"(\d{1,2}):(\d{2})\s*([AP])\.?M", re.I)
_DATE_RE = re.compile(r"(January|February|March|April|May|June|July|August|September|October|November|December)"
                      r"\s+(\d{1,2})(?:,\s*(\d{4}))?", re.I)


def parse_bea_schedule_html(html: str, today: date) -> list[dict]:
    """Tabellenzeilen des BEA-Veröffentlichungsplans: Datum (+ Jahr aus Zeile/Seite), Uhrzeit ET, Titel."""
    from lxml import html as lh
    doc = lh.fromstring(html.encode("utf-8"))
    out = []
    for tr in doc.xpath("//tr"):
        cells = [" ".join(c.text_content().split()) for c in tr.xpath("./td|./th")]
        row = " | ".join(cells)
        dm, tm = _DATE_RE.search(row), _TIME_RE.search(row)
        if not dm or not tm:
            continue
        title = max((c for c in cells if not _DATE_RE.fullmatch(c.strip()) and not _TIME_RE.search(c)),
                    key=len, default="")
        if not title:
            continue
        month, day = MONTHS[dm.group(1).lower()[:3]], int(dm.group(2))
        hour = int(tm.group(1)) % 12 + (12 if tm.group(3).upper() == "P" else 0)
        years = [int(dm.group(3))] if dm.group(3) else [today.year, today.year + 1]
        for y in years:
            try:
                d = date(y, month, day)
            except ValueError:
                continue
            if dm.group(3) or today - timedelta(days=31) <= d <= today + timedelta(days=334):
                out.append({"start": datetime.combine(d, time(hour, int(tm.group(2))), NY), "summary": title})
                break
    if not out:
        raise ValueError("BEA-Plan: keine Zeilen mit Datum + Uhrzeit erkannt")
    return out


class BeaReleaseCalendarConnector(_CalendarConnector):
    """BEA: zuerst release_dates.json (maschinenlesbar), sonst die Tabelle der Plan-Seite."""
    source_id = "bea_release_calendar"
    URL, DATASET = BEA_JSON_URL, "bea_schedule"

    def fetch(self, now: datetime) -> ConnectorResult:
        self.URL = BEA_JSON_URL
        res_json = super().fetch(now)
        if res_json.status == SourceStatus.PASS:
            return res_json
        self.URL = BEA_SCHEDULE_URL
        res_html = super().fetch(now)
        if res_html.status != SourceStatus.PASS:
            res_html.message = f"JSON: {res_json.message} | HTML: {res_html.message}"
            res_html.raw = res_json.raw + res_html.raw
        return res_html

    def events(self, res):
        if self.URL == BEA_JSON_URL:
            try:
                evs = parse_bea_json(json.loads(res.content.decode("utf-8", "replace")))
            except json.JSONDecodeError as e:
                raise ValueError(f"BEA release_dates.json kein JSON ({e}): "
                                 f"{res.content[:120]!r}") from e
        else:
            evs = parse_bea_schedule_html(res.content.decode("utf-8", "replace"), res.retrieved_at.date())
        return [{"start": e["start"], "series": _slug(e["summary"]), "title": e["summary"]} for e in evs]


CONNECTORS: dict[str, type[Connector]] = {
    "fomc_calendar": FomcCalendarConnector,
    "bls_release_calendar": BlsReleaseCalendarConnector,
    "bea_release_calendar": BeaReleaseCalendarConnector,
}

"""Unabhaengige, adversariale Regressionstests fuer die P1-Aenderungen vom 2026-10-16 (nur SYNTHETISCHE Daten).

Geprueft wird gegen die Spezifikation, nicht gegen die vorhandenen Tests:
  1. Kalender (fomc/bls/bea): available_at == retrieved_at == Abrufzeit, Momentaufnahme nur bei geaendertem Plan,
     Fenster [now-7d, now+400d] (Grenzen inklusiv), gealterte Termine sind nicht "zurueckgezogen",
     schedule_as_of() = Plan zum Zeitpunkt t bzw. None vor der ersten Momentaufnahme, DST/Zeitzonen
  2. data_quality.assess: allow_future_observations nur bis max_future_days, sonst unveraendert (2-Tage-Regel)
  3. source_health (Registry-Pfad): PIT-Verletzung nur aus dem Orchestrator (pit_integrity_failures)
  4. Orchestrator: statische Quelle wird bei Parser-Wechsel neu abgerufen (Health bzw. Manifest)
  5. NOAA Storm Events: series_id "<Jahr>:<Ereignistyp>", keine Kollision, idempotenter Reimport
  6. SEC 13F: Datenstand ist nie rueckwaerts gerichtet; Quartals-/Monatsfenster (set_range)
  7. Nasdaq-100: Wikipedia -> nasdaq.com -> statisch; UNIVERSE_V1 unveraendert
Plus: Archiv-Zaehler new/revision/duplicate, historische PIT-Abfragen, ops_monitor.data_health.

Ehemals mit `xfail(strict=True, reason="FINDING: ...")` markierte Befunde (4) sind behoben (2026-10-16) und laufen als normale Regressionstests.
"""
from __future__ import annotations

import gzip
import io
import json
import logging
import zipfile
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from modules import ops_monitor
from modules import source_health as sh
from modules import universe as uv
from modules import universe_v2
from modules.external import data_quality as dq
from modules.external import http
from modules.external import orchestrator as orch
from modules.external import registry as reg
from modules.external.alerts import check_pit_integrity
from modules.external.archive import ExternalArchive
from modules.external.pit import AvailabilityPrecision, Observation
from modules.external.registry import SourceHealth, SourceRegistry
from modules.external.sources import calendars as cal
from modules.external.sources import company_events as ce
from modules.external.sources import market_public as mp
from modules.external.sources import weather as wx
from modules.external.sources.base import Connector, ConnectorResult, RawRecord, SourceStatus

UTC = timezone.utc
NY = ZoneInfo("America/New_York")
T0 = datetime(2026, 10, 15, 12, 0, tzinfo=UTC)
DAY = timedelta(days=1)
EXACT = AvailabilityPrecision.EXACT_TIMESTAMP


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Kein echter Netzwerkzugriff: jeder nicht gemockte Abruf scheitert laut."""
    def boom(*a, **k):
        raise AssertionError("Netzwerkzugriff im Test")
    monkeypatch.setattr(http, "fetch", boom)
    monkeypatch.setattr("requests.sessions.Session.request", boom)


def fr(body, when, url="https://fixture", ctype="text/plain") -> http.FetchResult:
    raw = body if isinstance(body, bytes) else body.encode()
    return http.FetchResult(url=url, status=200, content=raw, content_type=ctype, retrieved_at=when,
                            content_hash="h", fingerprint="fp", bytes=len(raw))


def fake_registry(root, sources, connectors):
    r = SourceRegistry.__new__(SourceRegistry)
    r.sources_dir = "unused"
    r.archive_root = root
    r.sources = sources
    r.connectors = connectors
    return r


VALID_SOURCE = dict(
    source_id="x", display_name="X", family="maritime", authority="Amt X", official_homepage="https://x.example",
    machine_endpoint="https://x.example/api", access_method="rest", requires_auth=False, auth_env_variable=None,
    license_reference="https://x.example/license", frequency="daily", expected_update_cadence="1d",
    supports_history=True, supports_vintages=False, supports_release_time=True, pit_quality="EXACT_DATE",
    enabled=True, criticality="medium", license_status="OK")


# ═══ 1. Kalender ════════════════════════════════════════════════════════════════════════════════════════════════
BLS, FOMC, BEA = cal.BlsReleaseCalendarConnector, cal.FomcCalendarConnector, cal.BeaReleaseCalendarConnector
BLS_ID = "bls_release_calendar"


def zt(t: datetime) -> str:
    return ":" + t.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def ics(*events: tuple[str, str]) -> str:
    body = "".join(f"BEGIN:VEVENT\r\nUID:{i}\r\nDTSTART{dt}\r\nSUMMARY:{s}\r\nEND:VEVENT\r\n"
                   for i, (dt, s) in enumerate(events))
    return "BEGIN:VCALENDAR\r\nVERSION:2.0\r\n" + body + "END:VCALENDAR\r\n"


def fomc_html(*meetings: tuple[int, str, str]) -> str:
    out, last = "", None
    for year, month, days in meetings:
        if year != last:
            out += f'<h4><a id="{year}">{year} FOMC Meetings</a></h4>'
            last = year
        out += ('<div class="row fomc-meeting"><div class="fomc-meeting__month col-xs-5">'
                f'<strong>{month}</strong></div><div class="fomc-meeting__date col-xs-4">{days}</div></div>')
    return out


NYT = ";TZID=America/New_York:"
CPI_NOV = (NYT + "20261113T083000", "Consumer Price Index for October 2026")
JOBS_NOV = (NYT + "20261106T083000", "Employment Situation for October 2026")
JOBS_NOV_MOVED = (NYT + "20261109T083000", "Employment Situation for October 2026")
PPI_DEC = (NYT + "20261210T083000", "Producer Price Index for November 2026")
KEY_JOBS = "employment_situation|2026-11-06T13:30:00+00:00|1"
KEY_JOBS_MOVED = "employment_situation|2026-11-09T13:30:00+00:00|1"
KEY_CPI = "consumer_price_index|2026-11-13T13:30:00+00:00|1"
KEY_PPI = "producer_price_index|2026-12-10T13:30:00+00:00|1"

CAL_CASES = [
    ("fomc_calendar", FOMC, fomc_html((2026, "December", "8-9*"), (2027, "January", "26-27"))),
    (BLS_ID, BLS, ics(CPI_NOV, JOBS_NOV)),
    ("bea_release_calendar", BEA, json.dumps(
        {"GDP": {"release_dates": ["2026-10-29T08:30:00", "2026-11-25T08:30:00"]}})),
]


def fetch_cal(cls, body, when, root=None, *, now=None, store=True):
    """Ein Abruf (Konnektor liest seinen letzten Plan aus dem Archiv) + Ablage wie im Orchestrator."""
    conn = cls({"_archive_root": str(root)} if root is not None else {})
    with patch.object(cal.http, "fetch", side_effect=lambda url, **kw: fr(body, when, url=url)):
        res = conn.fetch(now or when)
    counts = None
    if store and root is not None and res.observations:
        counts = ExternalArchive(root).store_observations(res.observations)
    return res, counts


def snaps(res):
    return [o for o in res.observations if o.series_id == cal.SNAPSHOT_SERIES]


def events(res):
    return [o for o in res.observations if o.metric == "scheduled_release"]


def plan_days(sid, t, arch):
    p = cal.schedule_as_of(sid, t, arch)
    return None if p is None else sorted((x["series"], x["time"].day) for x in p)


@pytest.mark.parametrize("sid,cls,body", CAL_CASES, ids=[c[0] for c in CAL_CASES])
def test_announced_future_events_are_known_from_fetch_time_only(sid, cls, body, tmp_path):
    when = T0 + timedelta(hours=3)
    res, _ = fetch_cal(cls, body, when, tmp_path)
    assert res.status == SourceStatus.PASS
    ev = events(res)
    assert len(ev) >= 2 and all(o.observation_time > when + timedelta(days=2) for o in ev)
    assert all(o.available_at == o.retrieved_at == when for o in res.observations)           # nie frueher als der Abruf
    assert all(o.availability_precision == EXACT for o in res.observations)
    assert check_pit_integrity(res.observations) == []
    d = dq.assess(res.observations, SourceRegistry().sources[sid], when)
    assert d["issues"] == [] and not d["severe"] and d["details"]["scheduled_future"] == len(ev)
    arch = ExternalArchive(tmp_path)
    assert arch.as_of(sid, when - timedelta(microseconds=1)) == []
    assert cal.schedule_as_of(sid, when - timedelta(microseconds=1), arch) is None
    assert len(cal.schedule_as_of(sid, when, arch)) == len(ev)                               # inklusiv ab dem Abruf


def test_no_earlier_availability_is_invented_from_publisher_timestamps(tmp_path):
    """DTSTAMP/CREATED/LAST-MODIFIED und vergangene Terminzeiten duerfen available_at nicht vorverlegen."""
    body = ("BEGIN:VCALENDAR\r\nBEGIN:VEVENT\r\nUID:1\r\nDTSTAMP:20250101T000000Z\r\nCREATED:20250101T000000Z\r\n"
            "LAST-MODIFIED:20250102T000000Z\r\nDTSTART;TZID=America/New_York:20261113T083000\r\n"
            "SUMMARY:Consumer Price Index for October 2026\r\nEND:VEVENT\r\n"
            "BEGIN:VEVENT\r\nUID:2\r\nDTSTART;TZID=America/New_York:20261012T083000\r\n"       # 3 T vor dem Abruf
            "SUMMARY:Retail Sales for September 2026\r\nEND:VEVENT\r\nEND:VCALENDAR\r\n")
    res, _ = fetch_cal(BLS, body, T0, tmp_path)
    assert {o.available_at for o in res.observations} == {T0} and len(events(res)) == 2
    past = next(o for o in events(res) if o.series_id == "retail_sales")
    assert past.observation_time < T0 == past.available_at
    arch = ExternalArchive(tmp_path)
    for t in (datetime(2025, 6, 1, tzinfo=UTC), datetime(2026, 10, 12, 13, 0, tzinfo=UTC), T0 - timedelta(seconds=1)):
        assert arch.as_of(BLS_ID, t) == [] and cal.schedule_as_of(BLS_ID, t, arch) is None


def test_window_boundaries_are_inclusive_and_exact(tmp_path):
    lo, hi = T0 - timedelta(days=7), T0 + timedelta(days=400)
    sec = timedelta(seconds=1)
    body = ics((zt(lo), "At Lower Bound"), (zt(lo - sec), "Just Before Lower"), (zt(hi), "At Upper Bound"),
               (zt(hi + sec), "Just After Upper"))
    res, _ = fetch_cal(BLS, body, T0, tmp_path)
    assert sorted(o.series_id for o in events(res)) == ["at_lower_bound", "at_upper_bound"]
    [s] = snaps(res)
    assert s.attrs["window"] == [lo.isoformat(), hi.isoformat()] and s.value == 2.0
    d = dq.assess(res.observations, SourceRegistry().sources[BLS_ID], T0)                  # hi = 400 T <= 410 T
    assert "FUTURE_OBSERVATION" not in d["issues"]


def test_snapshot_only_when_plan_changes(tmp_path):
    r0, c0 = fetch_cal(BLS, ics(JOBS_NOV, CPI_NOV), T0, tmp_path)
    [s0] = snaps(r0)
    assert s0.attrs["previous_snapshot_at"] is None and s0.attrs["added"] == [] and s0.attrs["withdrawn"] == []
    assert s0.value == 2.0 and sorted(s0.attrs["events"]) == [KEY_CPI, KEY_JOBS]
    assert s0.observation_time == s0.available_at == s0.retrieved_at == T0 and "erste Momentaufnahme" in r0.message
    assert c0 == {"new": 3, "duplicate": 0, "revision": 0}
    r1, c1 = fetch_cal(BLS, ics(CPI_NOV, JOBS_NOV), T0 + DAY, tmp_path)                    # andere Reihenfolge, gleicher Plan
    assert snaps(r1) == [] and c1 == {"new": 0, "duplicate": 2, "revision": 0} and "Plan geändert" not in r1.message
    r2, _ = fetch_cal(BLS, ics(JOBS_NOV, CPI_NOV, PPI_DEC), T0 + 2 * DAY, tmp_path)
    [s2] = snaps(r2)
    assert s2.attrs["added"] == [KEY_PPI] and s2.attrs["withdrawn"] == [] and "+1 / −0" in r2.message
    assert s2.attrs["previous_snapshot_at"] == T0.isoformat()
    r3, _ = fetch_cal(BLS, ics(CPI_NOV, PPI_DEC), T0 + 3 * DAY, tmp_path)                  # JOBS verschwindet (im Fenster)
    [s3] = snaps(r3)
    assert s3.attrs["added"] == [] and s3.attrs["withdrawn"] == [KEY_JOBS] and "+0 / −1" in r3.message
    assert s3.attrs["previous_snapshot_at"] == (T0 + 2 * DAY).isoformat()
    r4, _ = fetch_cal(BLS, ics(CPI_NOV, PPI_DEC), T0 + 4 * DAY, tmp_path)
    assert snaps(r4) == []


def test_withdrawal_uses_the_current_window_lower_bound(tmp_path):
    """Ein verschwundener Termin ist genau dann 'zurueckgezogen', wenn er im AKTUELLEN Fenster liegt."""
    x = T0 + timedelta(days=30)
    keeper = (zt(T0 + timedelta(days=60)), "Keeper Event")
    first = ics((zt(x), "Event X"), keeper)
    key_x = f"event_x|{x.isoformat()}|1"
    inside, outside = tmp_path / "inside", tmp_path / "outside"
    fetch_cal(BLS, first, T0, inside)
    fetch_cal(BLS, first, T0, outside)
    r_in, _ = fetch_cal(BLS, ics(keeper), x + timedelta(days=7), inside)                    # x == untere Fenstergrenze
    assert [s.attrs["withdrawn"] for s in snaps(r_in)] == [[key_x]]
    r_out, _ = fetch_cal(BLS, ics(keeper), x + timedelta(days=7, seconds=1), outside)       # x eine Sekunde ausserhalb
    assert snaps(r_out) == []                                                               # gealtert, nicht zurueckgezogen


def test_future_event_vanishing_far_inside_window_is_withdrawn_but_not_after_aging(tmp_path):
    a, b = (zt(T0 + 5 * DAY), "Soon Event"), (zt(T0 + 200 * DAY), "Later Event")
    fetch_cal(BLS, ics(a, b), T0, tmp_path)
    r, _ = fetch_cal(BLS, ics(b), T0 + DAY, tmp_path)                                       # Soon-Event faellt aus dem Plan
    assert snaps(r)[0].attrs["withdrawn"][0].startswith("soon_event|")
    later = T0 + 30 * DAY                                                                    # Soon-Event laengst > 7 T alt
    r2, _ = fetch_cal(BLS, ics(b), later, tmp_path)
    assert snaps(r2) == []                                                                   # Plan seit Tag 1 unveraendert


def test_moved_cancelled_and_added_dates_are_traceable_via_historical_queries(tmp_path):
    arch = ExternalArchive(tmp_path)
    t1, t2, t3 = T0 + 2 * DAY, T0 + 3 * DAY, T0 + 4 * DAY
    fetch_cal(BLS, ics(CPI_NOV, JOBS_NOV), T0, tmp_path)
    fetch_cal(BLS, ics(CPI_NOV, JOBS_NOV_MOVED, PPI_DEC), t1, tmp_path)                    # verschoben + neu
    fetch_cal(BLS, ics(CPI_NOV, JOBS_NOV_MOVED), t2, tmp_path)                              # PPI abgesagt
    fetch_cal(BLS, ics(JOBS_NOV_MOVED), t3, tmp_path)                                       # CPI abgesagt
    cpi, jobs, ppi = ("consumer_price_index", 13), ("employment_situation", 6), ("producer_price_index", 10)
    jobs2 = ("employment_situation", 9)
    assert plan_days(BLS_ID, T0 - timedelta(seconds=1), arch) is None
    assert plan_days(BLS_ID, T0, arch) == [cpi, jobs]
    assert plan_days(BLS_ID, t1 - timedelta(seconds=1), arch) == [cpi, jobs]
    assert plan_days(BLS_ID, t1, arch) == [cpi, jobs2, ppi]
    assert plan_days(BLS_ID, t2, arch) == [cpi, jobs2]
    assert plan_days(BLS_ID, t3, arch) == [jobs2]
    assert plan_days(BLS_ID, t3 + 365 * DAY, arch) == [jobs2]
    # Terminzeilen allein behalten abgesagte/verschobene Termine (daher die Momentaufnahme)
    rows = {(o.series_id, o.observation_time.day) for o in arch.as_of(BLS_ID, t3) if o.metric == "scheduled_release"}
    assert rows == {cpi, jobs, jobs2, ppi}


def test_flip_flop_plan_and_exact_pit_boundaries(tmp_path):
    arch = ExternalArchive(tmp_path)
    t1, t2, t3 = T0, T0 + 2 * DAY, T0 + 4 * DAY
    fetch_cal(BLS, ics(JOBS_NOV), t1, tmp_path)
    fetch_cal(BLS, ics(JOBS_NOV_MOVED), t2, tmp_path)
    _, c3 = fetch_cal(BLS, ics(JOBS_NOV), t3, tmp_path)                                     # zurueck zum alten Termin
    assert c3 == {"new": 1, "duplicate": 1, "revision": 0}                                  # nur die Momentaufnahme ist neu
    micro = timedelta(microseconds=1)
    jobs, moved = [("employment_situation", 6)], [("employment_situation", 9)]
    assert plan_days(BLS_ID, t1 - micro, arch) is None and plan_days(BLS_ID, t1, arch) == jobs
    assert plan_days(BLS_ID, t2 - micro, arch) == jobs and plan_days(BLS_ID, t2, arch) == moved
    assert plan_days(BLS_ID, t3 - micro, arch) == moved and plan_days(BLS_ID, t3, arch) == jobs


def test_same_day_refetch_with_changed_plan_hours_apart(tmp_path):
    arch = ExternalArchive(tmp_path)
    t_morning, t_evening = T0, T0 + timedelta(hours=9)
    fetch_cal(BLS, ics(JOBS_NOV), t_morning, tmp_path)
    r, c = fetch_cal(BLS, ics(JOBS_NOV_MOVED), t_evening, tmp_path)                         # gleicher Tag, gleiche Anzahl
    assert len(snaps(r)) == 1 and c["new"] == 2                                              # verschobener Termin + Snapshot
    assert plan_days(BLS_ID, t_morning + timedelta(hours=8), arch) == [("employment_situation", 6)]
    assert plan_days(BLS_ID, t_evening, arch) == [("employment_situation", 9)]


# FINDING (low, latent) der unabhängigen Prüfung – behoben 2026-10-16; bleibt als Regressionstest.
def test_changed_plan_with_same_count_in_same_second_is_not_lost(tmp_path):
    fetch_cal(BLS, ics(JOBS_NOV), T0, tmp_path)
    fetch_cal(BLS, ics(JOBS_NOV_MOVED), T0, tmp_path)
    stored = [o for o in ExternalArchive(tmp_path).load(BLS_ID) if o.series_id == cal.SNAPSHOT_SERIES]
    assert any(KEY_JOBS_MOVED in o.attrs["events"] for o in stored)


def test_value_only_change_is_a_revision_and_a_snapshot(tmp_path):
    """FOMC: Sitzung erhaelt nachtraeglich das SEP-Kennzeichen (value 1 -> 2): gleiche Terminzeile, neue Vintage."""
    sid, base = "fomc_calendar", fomc_html((2026, "December", "8-9"))
    r1, c1 = fetch_cal(FOMC, base, T0, tmp_path)
    r2, c2 = fetch_cal(FOMC, base.replace("8-9", "8-9*"), T0 + DAY, tmp_path)
    assert c1 == {"new": 2, "duplicate": 0, "revision": 0} and c2 == {"new": 1, "duplicate": 0, "revision": 1}
    [s] = snaps(r2)
    assert s.attrs["added"] == ["FOMC_DECISION|2026-12-09T19:00:00+00:00|2"]
    assert s.attrs["withdrawn"] == ["FOMC_DECISION|2026-12-09T19:00:00+00:00|1"]
    arch = ExternalArchive(tmp_path)
    val = lambda t: [o.value for o in arch.as_of(sid, t) if o.metric == "scheduled_release"]  # noqa: E731
    assert val(T0) == [1.0] and val(T0 + DAY - timedelta(seconds=1)) == [1.0] and val(T0 + DAY) == [2.0]
    assert [p["value"] for p in cal.schedule_as_of(sid, T0, arch)] == [1.0]
    assert [p["value"] for p in cal.schedule_as_of(sid, T0 + DAY, arch)] == [2.0]
    assert "DUPLICATE_CONFLICT" not in dq.assess(r2.observations, SourceRegistry().sources[sid], T0 + DAY)["issues"]


def test_empty_missing_and_unconfigured_archive(tmp_path):
    missing = tmp_path / "does" / "not" / "exist"
    arch = ExternalArchive(missing)
    assert arch.load(BLS_ID) == [] and arch.as_of(BLS_ID, T0) == []
    assert cal.schedule_as_of(BLS_ID, T0, arch) is None                       # leeres Archiv (truthy) -> kein Default-Archiv
    r, _ = fetch_cal(BLS, ics(JOBS_NOV), T0, missing)
    assert r.status == SourceStatus.PASS and len(snaps(r)) == 1 and "erste Momentaufnahme" in r.message
    r_a, _ = fetch_cal(BLS, ics(JOBS_NOV), T0, None)                          # ohne _archive_root: nie stille Annahme
    r_b, _ = fetch_cal(BLS, ics(JOBS_NOV), T0 + DAY, None)
    assert len(snaps(r_a)) == 1 and len(snaps(r_b)) == 1
    f = tmp_path / "root_is_a_file"
    f.write_text("x")
    r_f, _ = fetch_cal(BLS, ics(JOBS_NOV), T0, f, store=False)
    assert r_f.status == SourceStatus.PASS and len(snaps(r_f)) == 1


def test_unreadable_archive_gives_fresh_snapshot_instead_of_crash(tmp_path, caplog):
    fetch_cal(BLS, ics(JOBS_NOV), T0, tmp_path)
    for p in (tmp_path / "normalized" / BLS_ID).glob("*"):
        p.write_text("kein json\n{")
    with caplog.at_level(logging.WARNING, logger=cal.log.name):
        r, _ = fetch_cal(BLS, ics(JOBS_NOV), T0 + DAY, tmp_path, store=False)
    assert r.status == SourceStatus.PASS and len(snaps(r)) == 1 and "erste Momentaufnahme" in r.message
    assert any("Archiv nicht lesbar" in rec.getMessage() for rec in caplog.records)


@pytest.mark.parametrize("failure,status", [("fetch_error", SourceStatus.FAIL), ("auth_error", SourceStatus.FAIL),
                                            ("garbage", SourceStatus.SCHEMA_CHANGED),
                                            ("empty_window", SourceStatus.SCHEMA_CHANGED)])
def test_failed_fetch_emits_no_snapshot_and_keeps_state(tmp_path, failure, status):
    arch = ExternalArchive(tmp_path)
    fetch_cal(BLS, ics(JOBS_NOV, CPI_NOV), T0, tmp_path)
    before = len(arch.load(BLS_ID))
    t1 = T0 + DAY
    outcome = {"fetch_error": http.FetchError("boom"), "auth_error": http.AuthError("403"),
               "garbage": fr("<html>kein Kalender</html>", t1),
               "empty_window": fr(ics((zt(datetime(2020, 1, 1, tzinfo=UTC)), "Ancient Event")), t1)}[failure]
    with patch.object(cal.http, "fetch", side_effect=[outcome]):                                # Ausnahme oder Antwort
        res = BLS({"_archive_root": str(tmp_path)}).fetch(t1)
    assert res.status == status and res.observations == []
    assert len(arch.load(BLS_ID)) == before
    r2, c2 = fetch_cal(BLS, ics(JOBS_NOV, CPI_NOV), T0 + 2 * DAY, tmp_path)                 # Erholung, Plan unveraendert
    assert snaps(r2) == [] and c2["new"] == 0
    assert plan_days(BLS_ID, T0 + 2 * DAY, arch) == [("consumer_price_index", 13), ("employment_situation", 6)]
    r3, _ = fetch_cal(BLS, ics(JOBS_NOV_MOVED, CPI_NOV), T0 + 3 * DAY, tmp_path)
    assert snaps(r3)[0].attrs["previous_snapshot_at"] == T0.isoformat()                      # Vorgaenger = letzter ERFOLGREICHER


def test_bea_in_feed_duplicates(tmp_path):
    payload = {"GDP": {"release_dates": ["2026-10-29T08:30:00", "2026-10-29T08:30:00"]},
               "Personal Income and Outlays": {"release_dates": ["2026-10-29T08:30:00"]}}
    res, counts = fetch_cal(BEA, json.dumps(payload), T0, tmp_path)
    assert sorted(o.series_id for o in events(res)) == ["gdp", "gdp", "personal_income_and_outlays"]
    [s] = snaps(res)
    assert s.value == 2.0 and len(s.attrs["events"]) == 2                                      # Schluessel dedupliziert
    assert counts == {"new": 3, "duplicate": 1, "revision": 0}
    d = dq.assess(res.observations, SourceRegistry().sources["bea_release_calendar"], T0)
    assert d["duplicate_conflicts"] == 0 and "DUPLICATE_CONFLICT" not in d["issues"]
    assert len(cal.schedule_as_of("bea_release_calendar", T0, ExternalArchive(tmp_path))) == 2


BEA_HTML = ("<html><body><table>"
            "<tr><td>November 1, 2026</td><td>8:30 AM</td><td>Gross Domestic Product (Third Estimate)</td></tr>"
            "<tr><td>March 14, 2027</td><td>8:30 AM</td><td>Personal Income and Outlays, February 2027</td></tr>"
            "<tr><th>Kopfzeile ohne Datum</th></tr></table></body></html>")


@pytest.mark.parametrize("json_failure", ["not_json", "fetch_error"])
def test_bea_falls_back_to_html_schedule_and_snapshots(tmp_path, json_failure):
    def fake(url, **kw):
        if url == cal.BEA_JSON_URL:
            if json_failure == "fetch_error":
                raise http.FetchError("404")
            return fr("<html>Moved</html>", T0, url=url)
        return fr(BEA_HTML, T0, url=url)
    with patch.object(cal.http, "fetch", side_effect=fake):
        res = BEA({"_archive_root": str(tmp_path)}).fetch(T0)
    assert res.status == SourceStatus.PASS and len(snaps(res)) == 1
    got = {o.series_id: o for o in events(res)}
    assert got["gross_domestic_product_third_estimate"].observation_time == datetime(2026, 11, 1, 13, 30, tzinfo=UTC)
    assert got["personal_income_and_outlays_february_2027"].observation_time == datetime(2027, 3, 14, 12, 30, tzinfo=UTC)
    assert check_pit_integrity(res.observations) == []


# ── Zeitzonen und Sommerzeit ─────────────────────────────────────────────────────────────────────────────────
DST_ICS_CASES = [
    ("20261031T083000", (12, 30)),     # EDT (UTC-4), Vortag der Zeitumstellung
    ("20261101T083000", (13, 30)),     # EST (UTC-5), Tag der Umstellung
    ("20261101T013000", (5, 30)),      # mehrdeutig (01:30 doppelt) -> erstes Auftreten (EDT)
    ("20270313T083000", (13, 30)),     # EST, Vortag der Umstellung
    ("20270314T083000", (12, 30)),     # EDT
    ("20270314T033000", (7, 30)),      # erste Stunde nach der Luecke
]


@pytest.mark.parametrize("local,expected", DST_ICS_CASES)
def test_ics_new_york_dst_transitions(local, expected):
    [ev] = cal.parse_ics(ics((NYT + local, "Some Release")))
    t = ev["start"].astimezone(UTC)
    assert (t.hour, t.minute) == expected and t.tzinfo is not None


def test_ics_nonexistent_local_time_in_spring_gap_does_not_crash():
    [ev] = cal.parse_ics(ics((NYT + "20270314T023000", "Gap Release")))                     # 02:30 existiert nicht
    assert ev["start"].astimezone(UTC) in (datetime(2027, 3, 14, 6, 30, tzinfo=UTC), datetime(2027, 3, 14, 7, 30, tzinfo=UTC))


@pytest.mark.parametrize("tzid,expected_utc_hour", [
    ("America/New_York", 13), ("US-Eastern", 13), ("Mars/Olympus_Mons", 13), ("Europe/Berlin", 7), ("UTC", 8)])
def test_ics_tzid_unknown_falls_back_to_new_york(tzid, expected_utc_hour):
    [ev] = cal.parse_ics(ics((f";TZID={tzid}:20261113T083000", "Zone Release")))
    assert ev["start"].astimezone(UTC).hour == expected_utc_hour


def test_ics_without_tzid_utc_suffix_and_all_day():
    evs = {e["summary"]: e for e in cal.parse_ics(ics(
        (":20261113T083000", "Floating"), (":20261113T083000Z", "Zulu"), (";VALUE=DATE:20261101", "AllDay Fall"),
        (";VALUE=DATE:20270314", "AllDay Spring")))}
    assert evs["Floating"]["start"].astimezone(UTC).hour == 13                                # ohne TZID = New York
    assert evs["Zulu"]["start"].astimezone(UTC).hour == 8                                      # Z = UTC, nicht umgedeutet
    assert evs["AllDay Fall"]["start"].astimezone(UTC) == datetime(2026, 11, 1, 13, 30, tzinfo=UTC)     # 08:30 EST
    assert evs["AllDay Spring"]["start"].astimezone(UTC) == datetime(2027, 3, 14, 12, 30, tzinfo=UTC)   # 08:30 EDT
    assert evs["AllDay Fall"]["all_day"] and not evs["Zulu"]["all_day"]


def test_calendar_connector_flags_assumed_all_day_times_and_shows_dst_offsets(tmp_path):
    res, _ = fetch_cal(BLS, ics((";VALUE=DATE:20261101", "Daily Release"), (NYT + "20261031T083000", "Edt Release"),
                                (NYT + "20261101T083000", "Est Release")), T0, tmp_path)
    ev = {o.series_id: o for o in events(res)}
    assert ev["daily_release"].attrs["time_assumed"] is True and "time_assumed" not in ev["est_release"].attrs
    assert ev["edt_release"].attrs["local_time_et"].endswith("08:30:00-04:00")
    assert ev["est_release"].attrs["local_time_et"].endswith("08:30:00-05:00")
    assert ev["edt_release"].observation_time.hour == 12 and ev["est_release"].observation_time.hour == 13


def test_fomc_decision_time_across_dst_transitions(tmp_path):
    body = fomc_html((2026, "October", "27-28"), (2026, "November", "1"), (2027, "March", "13"), (2027, "March", "14"))
    got = {m["decision_et"].astimezone(UTC): m for m in cal.parse_fomc(body)}
    assert sorted(got) == [datetime(2026, 10, 28, 18, 0, tzinfo=UTC),                          # 14:00 EDT
                           datetime(2026, 11, 1, 19, 0, tzinfo=UTC),                           # 14:00 EST (Umstellungstag)
                           datetime(2027, 3, 13, 19, 0, tzinfo=UTC),                           # 14:00 EST
                           datetime(2027, 3, 14, 18, 0, tzinfo=UTC)]                           # 14:00 EDT (Umstellungstag)
    res, _ = fetch_cal(FOMC, body, T0, tmp_path)
    offs = {o.observation_time: o.attrs["local_time_et"][-6:] for o in events(res)}
    assert offs[datetime(2026, 11, 1, 19, 0, tzinfo=UTC)] == "-05:00"
    assert offs[datetime(2027, 3, 14, 18, 0, tzinfo=UTC)] == "-04:00"


def test_bea_naive_times_are_new_york_but_explicit_offsets_are_respected():
    dates = ["2026-10-31T08:30:00", "2026-11-01T08:30:00", "2027-03-13T08:30:00", "2027-03-14T08:30:00",
             "2026-11-01T08:30:00-05:00", "2026-11-01T13:30:00Z", "2026-11-01T08:30:00-04:00"]
    got = [(e["start"].astimezone(UTC).hour, e["start"].astimezone(UTC).minute)
           for e in cal.parse_bea_json({"GDP": {"release_dates": dates}})]
    assert got == [(12, 30), (13, 30), (13, 30), (12, 30), (13, 30), (13, 30), (12, 30)]


def test_schedule_as_of_accepts_non_utc_and_naive_timestamps(tmp_path):
    arch = ExternalArchive(tmp_path)
    fetch_cal(BLS, ics(JOBS_NOV), T0, tmp_path)
    want = [("employment_situation", 6)]
    assert plan_days(BLS_ID, T0.astimezone(NY), arch) == want                                  # selber Zeitpunkt, andere Zone
    assert plan_days(BLS_ID, (T0 - timedelta(seconds=1)).astimezone(NY), arch) is None
    assert plan_days(BLS_ID, T0.replace(tzinfo=None), arch) == want                            # naiv = UTC (ensure_utc)


def test_events_and_snapshots_are_idempotent_on_reimport_and_survive_compaction(tmp_path):
    arch = ExternalArchive(tmp_path)
    far = [(zt(T0 + 200 * DAY), "Far Release A"), (zt(T0 + 300 * DAY), "Far Release B")]
    res, c1 = fetch_cal(BLS, ics(*far), T0, tmp_path)
    assert c1 == {"new": 3, "duplicate": 0, "revision": 0}
    n = len(arch.load(BLS_ID))
    assert arch.store_observations(res.observations) == {"new": 0, "duplicate": 3, "revision": 0}
    assert len(arch.load(BLS_ID)) == n
    packed = arch.compact_cold_partitions(now=T0 + 120 * DAY, cold_after_days=1)
    assert packed["compacted"] and not packed["failures"]
    assert list((tmp_path / "normalized" / BLS_ID).glob("*.jsonl.gz"))
    assert len(arch.load(BLS_ID)) == n
    assert sorted((p["series"], p["time"]) for p in cal.schedule_as_of(BLS_ID, T0, arch)) == [
        ("far_release_a", T0 + 200 * DAY), ("far_release_b", T0 + 300 * DAY)]
    r2, c2 = fetch_cal(BLS, ics(*far), T0 + 120 * DAY, tmp_path)                            # Vorgaenger liegt jetzt in .jsonl.gz
    assert snaps(r2) == [] and c2 == {"new": 0, "duplicate": 2, "revision": 0}


# ═══ 2. data_quality ═══════════════════════════════════════════════════════════════════════════════════════════
NOW_DQ = datetime(2026, 10, 16, 12, 0, tzinfo=UTC)
CAL_CFG = {"allow_future_observations": True, "max_future_days": 410}


def dq_obs(delta, *, value=1.0, series="x", **kw):
    base = dict(source_id="s", dataset="d", series_id=series, entity_id="US", metric="m", value=value, unit="e",
                observation_time=NOW_DQ + delta, available_at=NOW_DQ, retrieved_at=NOW_DQ,
                availability_precision=EXACT, parser_version="t")
    base.update(kw)
    return Observation(**base)


@pytest.mark.parametrize("delta,flagged", [
    (timedelta(days=2, seconds=1), False), (timedelta(days=410), False), (timedelta(days=410, seconds=1), True),
    (timedelta(days=4000), True)])
def test_allow_future_observations_only_up_to_max_future_days(delta, flagged):
    r = dq.assess([dq_obs(delta)], CAL_CFG, NOW_DQ)
    assert ("FUTURE_OBSERVATION" in r["issues"]) is flagged and r["severe"] is flagged
    assert r["future_observations"] == int(flagged)
    assert (r["details"].get("future_example") is not None) is flagged
    assert r["details"]["scheduled_future"] == int(not flagged)


@pytest.mark.parametrize("cfg", [{}, {"allow_future_observations": False},
                                 {"max_future_days": 410}, {"allow_future_observations": None, "max_future_days": 410}])
def test_other_sources_keep_the_two_day_tolerance(cfg):
    assert dq.assess([dq_obs(timedelta(days=2))], cfg, NOW_DQ)["issues"] == []
    r = dq.assess([dq_obs(timedelta(days=2, seconds=1))], cfg, NOW_DQ)
    assert "FUTURE_OBSERVATION" in r["issues"] and r["future_observations"] == 1 and r["severe"]
    assert "scheduled_future" not in r["details"]


def test_forecasts_are_exempt_with_and_without_allow_flag():
    f = dq_obs(timedelta(days=900), forecast_issue_time=NOW_DQ, forecast_valid_time=NOW_DQ + timedelta(days=900))
    for cfg in ({}, CAL_CFG):
        r = dq.assess([f], cfg, NOW_DQ)
        assert r["issues"] == [] and r["future_observations"] == 0
    mixed = dq.assess([f, dq_obs(timedelta(days=900), series="y")], {}, NOW_DQ)               # nur der Nicht-Forecast zaehlt
    assert mixed["future_observations"] == 1


def test_mixed_batch_counts_scheduled_and_flagged_separately():
    batch = [dq_obs(timedelta(days=d), series=f"s{d}") for d in (1, 3, 30, 400, 500, 900)]
    r = dq.assess(batch, CAL_CFG, NOW_DQ)
    assert r["future_observations"] == 2 and r["details"]["scheduled_future"] == 3             # 1 T liegt in der Toleranz
    assert r["details"]["future_example"] == (NOW_DQ + timedelta(days=500)).isoformat()


def test_default_horizon_is_550_days_when_max_future_days_is_omitted():
    cfg = {"allow_future_observations": True}
    assert dq.assess([dq_obs(timedelta(days=550))], cfg, NOW_DQ)["issues"] == []
    assert "FUTURE_OBSERVATION" in dq.assess([dq_obs(timedelta(days=551))], cfg, NOW_DQ)["issues"]
    assert dq.DEFAULT_MAX_FUTURE_DAYS == 550


def test_registry_future_policy_is_limited_to_the_three_calendars():
    srcs = SourceRegistry().sources
    allowed = {k for k, v in srcs.items() if v.get("allow_future_observations")}
    assert allowed == {"fomc_calendar", BLS_ID, "bea_release_calendar"}
    assert {srcs[k].get("max_future_days") for k in allowed} == {410}
    assert 410 > cal._CalendarConnector.HORIZON_DAYS >= 400                                    # DQ-Grenze > Konnektor-Fenster


def test_duplicate_conflict_vs_identical_duplicates_vs_vintages():
    a, b = dq_obs(timedelta(days=30), value=1.0), dq_obs(timedelta(days=30), value=2.0)
    r = dq.assess([a, b], CAL_CFG, NOW_DQ)
    assert r["issues"] == ["DUPLICATE_CONFLICT"] and r["duplicate_conflicts"] == 1 and r["severe"]
    assert dq.assess([a, dq_obs(timedelta(days=30), value=1.0)], CAL_CFG, NOW_DQ)["issues"] == []
    later = NOW_DQ + timedelta(hours=1)
    rev = dq_obs(timedelta(days=30), value=2.0, available_at=later, retrieved_at=later)       # andere Vintage = Revision
    assert dq.assess([a, rev], CAL_CFG, NOW_DQ)["issues"] == []


def test_naive_now_is_treated_as_utc():
    assert dq.assess([dq_obs(timedelta(days=30))], CAL_CFG, NOW_DQ.replace(tzinfo=None))["issues"] == []
    assert "FUTURE_OBSERVATION" in dq.assess([dq_obs(timedelta(days=30))], {}, NOW_DQ.replace(tzinfo=None))["issues"]


# FINDING (low, robustness) der unabhängigen Prüfung – behoben 2026-10-16; bleibt als Regressionstest.
def test_null_max_future_days_falls_back_to_default_instead_of_crashing():
    r = dq.assess([dq_obs(timedelta(days=30))], {"allow_future_observations": True, "max_future_days": None}, NOW_DQ)
    assert r["issues"] == []


# ═══ 3. Genuiner Lookahead und Source Health ════════════════════════════════════════════════════════════════════
def test_check_pit_integrity_catches_available_at_after_retrieved_at():
    def mk(avail, retr=T0, obs=T0 + 30 * DAY):
        return Observation(source_id=BLS_ID, dataset="d", series_id="EV", entity_id="US", metric="scheduled_release",
                           value=1.0, unit="event", observation_time=obs, available_at=avail, retrieved_at=retr,
                           availability_precision=EXACT, parser_version="t")
    assert check_pit_integrity([mk(T0)]) == []                                                # gleichzeitig: ok
    assert check_pit_integrity([mk(T0 - DAY)]) == []                                          # frueher bekannt: ok
    for bad in (mk(T0 + timedelta(microseconds=1)), mk(T0 + DAY, obs=T0 - DAY)):             # egal wo observation_time liegt
        errs = check_pit_integrity([bad])
        assert len(errs) == 1 and "available_at > retrieved_at" in errs[0]
    assert len(check_pit_integrity([mk(T0 + DAY), mk(T0), mk(T0 + 2 * DAY)])) == 2


def _event_obs(now, days, *, avail_delta=timedelta(0), sid="cal_x"):
    return Observation(source_id=sid, dataset="d", series_id=f"EV{days}", entity_id="US", metric="scheduled_release",
                       value=1.0, unit="event", observation_time=now + timedelta(days=days),
                       available_at=now + avail_delta, retrieved_at=now, availability_precision=EXACT,
                       parser_version="cal-test-1")


class _CalLike(Connector):
    source_id = "cal_x"
    parser_version = "cal-test-1"
    lookahead = False

    def fetch(self, now):
        obs = [_event_obs(now, 30)]
        if self.lookahead:
            obs.append(_event_obs(now, 45, avail_delta=DAY))                                  # Veroeffentlichung erst morgen
        raw = [RawRecord(source_id="cal_x", dataset="d", url="u", fingerprint="fp", retrieved_at=now, status_code=200,
                         content_type="x", content_hash="h", bytes=3)]
        return ConnectorResult("cal_x", SourceStatus.PASS, observations=obs, raw=raw,
                               latest_observation_time=obs[-1].observation_time)


class _Honest(_CalLike):
    pass


class _Lookahead(_CalLike):
    lookahead = True


def _health_checks(tmp_path, sources, now):
    cfg = dict(sh.load_config(), core_sources=[], alt_sources=[])
    checks = {c["source_id"]: c for c in sh.gather(cfg, now, probes=False, registry_health=reg.load_health(tmp_path),
                                                   contracts=sources)}
    return cfg, checks


CAL_X = dict(VALID_SOURCE, source_id="cal_x", family="public_research", frequency="weekly",
             expected_update_cadence="P7D", max_staleness_days=14, allow_future_observations=True, max_future_days=410)


def test_genuine_lookahead_is_flagged_by_orchestrator_and_surfaces_as_pit_violation_in_source_health(tmp_path):
    sources = {"cal_x": dict(CAL_X)}
    s = orch.run_ingestion(T0, registry=fake_registry(tmp_path, sources, {"cal_x": _Lookahead}))
    assert s["sources"]["cal_x"]["pit_errors"] == 1
    h = reg.load_health(tmp_path)["cal_x"]
    assert h["pit_integrity_failures"] == 1 and "available_at > retrieved_at" in h["message"]
    assert "FUTURE_OBSERVATION" not in h["dq"]["issues"]               # Zeitpunkte liegen im erlaubten Horizont (DQ blind)
    cfg, checks = _health_checks(tmp_path, sources, T0)
    st = checks["cal_x"]["store"]
    assert (st["future_timestamps"], st["future_periods"]) == (1, 0)
    c = sh.classify(checks["cal_x"], None, cfg, T0)
    assert c["raw_status"] == sh.BROKEN and c["failed"] is True
    assert [r for r in c["reasons"] if r.startswith("PIT-Verletzung: 1 Beobachtungen mit available_at > retrieved_at")]
    assert not any("Zeitpunkt in der Zukunft" in r for r in c["reasons"])


def test_honest_announced_calendar_is_not_a_pit_violation_in_source_health(tmp_path):
    sources = {"cal_x": dict(CAL_X)}
    s = orch.run_ingestion(T0, registry=fake_registry(tmp_path, sources, {"cal_x": _Honest}))
    assert s["sources"]["cal_x"]["pit_errors"] == 0 and s["sources"]["cal_x"]["dq_issues"] == []
    h = reg.load_health(tmp_path)["cal_x"]
    assert h["pit_integrity_failures"] == 0 and h["dq"]["details"]["scheduled_future"] == 1
    assert h["parser_version"] == "cal-test-1"
    cfg, checks = _health_checks(tmp_path, sources, T0)
    c = sh.classify(checks["cal_x"], None, cfg, T0)
    assert not any("PIT" in r or "Zukunft" in r for r in c["reasons"]) and c["raw_status"] == sh.HEALTHY


def test_real_registry_calendar_configs_end_to_end_over_three_runs(tmp_path):
    """Echte Registry-Eintraege (DQ-Horizont 410 T, Kadenz, Staleness) + Orchestrator + Source Health."""
    real = SourceRegistry().sources
    cases = {sid: (cls, body) for sid, cls, body in CAL_CASES}
    moved = {BLS_ID: ics(CPI_NOV, JOBS_NOV_MOVED)}
    sources = {sid: dict(real[sid]) for sid in cases}
    conns = {sid: cls for sid, (cls, _) in cases.items()}
    urls = {cal.FOMC_URL: "fomc_calendar", cal.BLS_ICS_URL: BLS_ID, cal.BEA_JSON_URL: "bea_release_calendar"}
    outs = []
    for i in range(3):
        now = T0 + i * DAY
        bodies = {sid: (moved.get(sid, b) if i == 2 else b) for sid, (_, b) in cases.items()}
        with patch.object(cal.http, "fetch", side_effect=lambda url, bodies=bodies, now=now, **kw: fr(
                bodies[urls[url]], now, url=url)):
            outs.append(orch.run_ingestion(now, registry=fake_registry(tmp_path, sources, conns))["sources"])
    for sid in cases:
        assert [o[sid]["status"] for o in outs] == ["PASS"] * 3
        assert all(o[sid]["pit_errors"] == 0 and o[sid]["dq_issues"] == [] for o in outs)
        assert outs[0][sid]["counts"] == {"new": 3, "duplicate": 0, "revision": 0}          # 2 Termine + Momentaufnahme
        assert outs[1][sid]["counts"] == {"new": 0, "duplicate": 2, "revision": 0}          # idempotent
    assert outs[2][BLS_ID]["counts"] == {"new": 2, "duplicate": 1, "revision": 0}           # verschobener Termin + Snapshot
    assert outs[2]["fomc_calendar"]["counts"]["new"] == outs[2]["bea_release_calendar"]["counts"]["new"] == 0
    arch = ExternalArchive(tmp_path)
    assert plan_days(BLS_ID, T0 + DAY, arch) == [("consumer_price_index", 13), ("employment_situation", 6)]
    assert plan_days(BLS_ID, T0 + 2 * DAY, arch) == [("consumer_price_index", 13), ("employment_situation", 9)]
    cfg, checks = _health_checks(tmp_path, sources, T0 + 2 * DAY)
    for sid in cases:
        c = sh.classify(checks[sid], None, cfg, T0 + 2 * DAY)
        assert c["raw_status"] == sh.HEALTHY and c["reasons"] == [], (sid, c)


@pytest.mark.parametrize("source", ["fomc_calendar", "fred_liquidity"], ids=["allow_future", "strict"])
@pytest.mark.parametrize("pit,dq_block,exp_ts,exp_per", [
    (0, None, 0, 0),
    (None, None, 0, 0),
    (0, {"issues": [], "future_observations": 0}, 0, 0),
    (3, {"issues": []}, 3, 0),
    (0, {"issues": ["FUTURE_OBSERVATION"], "future_observations": 2}, 0, 2),
    (3, {"issues": ["FUTURE_OBSERVATION"], "future_observations": 2}, 3, 2),
    (0, {"issues": ["OUT_OF_RANGE"], "future_observations": 5}, 0, 0),    # Zaehler ohne Befund-Flag wird ignoriert
    (0, {"issues": ["FUTURE_OBSERVATION"]}, 0, 0),                          # Befund ohne Zaehler
])
def test_source_health_separates_pit_violations_from_future_periods(source, pit, dq_block, exp_ts, exp_per):
    srcs = SourceRegistry().sources
    entry = {"status": "PASS", "latest_observation": "2026-10-14T00:00:00+00:00", "last_success": T0.isoformat()}
    if pit is not None:
        entry["pit_integrity_failures"] = pit
    if dq_block is not None:
        entry["dq"] = dq_block
    cfg = dict(sh.load_config(), core_sources=[], alt_sources=[])
    chk = next(c for c in sh.gather(cfg, T0, probes=False, registry_health={source: entry},
                                    contracts={source: srcs[source]}) if c["source_id"] == source)
    assert (chk["store"]["future_timestamps"], chk["store"]["future_periods"]) == (exp_ts, exp_per)
    c = sh.classify(chk, None, cfg, T0)
    pit_r = [r for r in c["reasons"] if r.startswith("PIT-Verletzung:")]
    per_r = [r for r in c["reasons"] if "Zeitpunkt in der Zukunft" in r]
    assert bool(pit_r) is bool(exp_ts) and bool(per_r) is bool(exp_per)
    if per_r:
        assert "keine PIT-Verletzung" in per_r[0] and not per_r[0].startswith("PIT-Verletzung")
    assert (c["raw_status"] == sh.BROKEN) is bool(exp_ts or exp_per)
    assert c["failed"] is bool(exp_ts or exp_per)


# ═══ 4. Orchestrator: Neuabruf statischer Quellen bei Parser-Wechsel ═══════════════════════════════════════════
def make_connector(parser, calls, *, status=SourceStatus.PASS):
    class _C(Connector):
        source_id = "ok_src"
        parser_version = parser

        def fetch(self, now):
            calls.append((parser, now))
            obs = [Observation(source_id="ok_src", dataset="d", series_id="S", entity_id="E", metric="m", value=1.0,
                               unit="u", observation_time=now, available_at=now, retrieved_at=now,
                               availability_precision=EXACT, parser_version=parser)]
            raw = [RawRecord(source_id="ok_src", dataset="d", url="u", fingerprint="fp", retrieved_at=now,
                             status_code=200, content_type="x", content_hash="h", bytes=3)]
            if status == SourceStatus.FAIL:
                return ConnectorResult("ok_src", status, message="down")
            return ConnectorResult("ok_src", status, observations=obs, raw=raw, latest_observation_time=now)
    return _C


STATIC = {"ok_src": dict(VALID_SOURCE, source_id="ok_src", min_refetch_days=30)}
S0 = datetime(2026, 2, 1, tzinfo=UTC)


def run_static(tmp_path, cls, when, **kw):
    return orch.run_ingestion(when, registry=fake_registry(tmp_path, STATIC, {"ok_src": cls}), **kw)["sources"]["ok_src"]


def static_health(tmp_path):
    return reg.load_health(tmp_path)["ok_src"]


def test_parser_change_refetches_once_then_stays_static(tmp_path):
    calls = []
    v1, v2 = make_connector("p-1", calls), make_connector("p-2", calls)
    run_static(tmp_path, v1, S0)
    assert static_health(tmp_path)["parser_version"] == "p-1"
    assert run_static(tmp_path, v1, S0 + 2 * DAY)["skipped"] == "min_refetch_days" and len(calls) == 1
    assert run_static(tmp_path, v2, S0 + 3 * DAY)["status"] == "PASS" and [c[0] for c in calls] == ["p-1", "p-2"]
    assert static_health(tmp_path)["parser_version"] == "p-2"
    assert run_static(tmp_path, v2, S0 + 4 * DAY)["skipped"] == "min_refetch_days" and len(calls) == 2
    run_static(tmp_path, v1, S0 + 5 * DAY)                                                   # Rollback ist ebenfalls ein Wechsel
    assert [c[0] for c in calls] == ["p-1", "p-2", "p-1"] and static_health(tmp_path)["parser_version"] == "p-1"


def test_parser_change_keeps_retrying_until_the_new_parser_succeeds(tmp_path):
    calls = []
    run_static(tmp_path, make_connector("p-1", calls), S0)
    for i, status in enumerate((SourceStatus.FAIL, SourceStatus.FAIL, SourceStatus.WARN), start=1):
        out = run_static(tmp_path, make_connector("p-2", calls, status=status), S0 + i * DAY)
        assert "skipped" not in out and static_health(tmp_path)["parser_version"] == "p-1"       # nur PASS merkt den Parser
    assert len(calls) == 4
    assert run_static(tmp_path, make_connector("p-2", calls), S0 + 4 * DAY)["status"] == "PASS"
    assert static_health(tmp_path)["parser_version"] == "p-2" and len(calls) == 5
    assert run_static(tmp_path, make_connector("p-2", calls), S0 + 5 * DAY)["skipped"] == "min_refetch_days"


def test_dry_run_parser_change_does_not_consume_the_refetch(tmp_path):
    calls = []
    run_static(tmp_path, make_connector("p-1", calls), S0)
    run_static(tmp_path, make_connector("p-2", calls), S0 + DAY, dry_run=True)
    assert len(calls) == 2 and static_health(tmp_path)["parser_version"] == "p-1"
    run_static(tmp_path, make_connector("p-2", calls), S0 + 2 * DAY)
    assert len(calls) == 3 and static_health(tmp_path)["parser_version"] == "p-2"


def test_age_beyond_min_refetch_days_still_refetches_with_same_parser(tmp_path):
    calls = []
    v1 = make_connector("p-1", calls)
    run_static(tmp_path, v1, S0)
    assert run_static(tmp_path, v1, S0 + timedelta(days=29, hours=23))["skipped"] == "min_refetch_days"
    assert "skipped" not in run_static(tmp_path, v1, S0 + 30 * DAY) and len(calls) == 2


def _drop_health_parser(tmp_path):
    p = tmp_path / "health" / "source_health.json"
    h = json.loads(p.read_text())
    h["ok_src"].pop("parser_version")
    p.write_text(json.dumps(h))


def test_legacy_health_without_parser_version_uses_the_newest_manifest_entry(tmp_path):
    calls = []
    run_static(tmp_path, make_connector("p-1", calls), S0)
    _drop_health_parser(tmp_path)
    assert run_static(tmp_path, make_connector("p-1", calls), S0 + DAY)["skipped"] == "min_refetch_days"
    assert run_static(tmp_path, make_connector("p-2", calls), S0 + 2 * DAY)["status"] == "PASS" and len(calls) == 2
    assert static_health(tmp_path)["parser_version"] == "p-2"


def test_legacy_health_without_any_parser_evidence_stays_skipped(tmp_path):
    calls = []
    run_static(tmp_path, make_connector("p-1", calls), S0)
    _drop_health_parser(tmp_path)
    for p in (tmp_path / "manifests").rglob("*.json"):
        p.unlink()
    assert run_static(tmp_path, make_connector("p-2", calls), S0 + DAY)["skipped"] == "min_refetch_days"


def test_last_manifest_parser_version_is_robust_and_prefers_newest(tmp_path):
    m = tmp_path / "manifests"
    for d in ("2026-10-01", "2026-10-02", "2026-10-03"):
        (m / d).mkdir(parents=True)
    ent = lambda v: json.dumps({"entries": [{"source_id": "s", "parser_version": v}]})  # noqa: E731
    (m / "2026-10-01" / "a.json").write_text(ent("old"))
    (m / "2026-10-02" / "a.json").write_text("{kaputt")
    (m / "2026-10-02" / "b.json").write_text(json.dumps([1, 2]))
    (m / "2026-10-02" / "c.json").write_text(json.dumps({"entries": None}))
    (m / "2026-10-02" / "d.json").write_text(json.dumps({"entries": ["x", {"source_id": "s", "parser_version": None},
                                                                      {"source_id": "s"}, {"source_id": "t",
                                                                                           "parser_version": "other"}]}))
    (m / "stray.txt").write_text("x")
    assert orch._last_manifest_parser_version(tmp_path, "s") == "old"                          # ueberspringt Muell/ohne Version
    (m / "2026-10-03" / "a.json").write_text(ent("new"))
    assert orch._last_manifest_parser_version(tmp_path, "s") == "new"
    (m / "2026-10-03" / "z.json").write_text(ent("newest-run-in-day"))                         # spaeterer Run-Name = juenger
    assert orch._last_manifest_parser_version(tmp_path, "s") == "newest-run-in-day"
    assert orch._last_manifest_parser_version(tmp_path, "t") == "other"
    assert orch._last_manifest_parser_version(tmp_path / "leer", "s") is None


def test_source_health_dataclass_parser_version_roundtrip_and_legacy_dict():
    h = SourceHealth.from_dict({"source_id": "a", "last_success": "x", "unbekanntes_feld": 1})
    assert h.parser_version is None and "parser_version" in h.to_dict()
    assert SourceHealth.from_dict(SourceHealth(source_id="a", parser_version="p-3").to_dict()).parser_version == "p-3"


def test_noaa_storm_events_legacy_health_is_refetched_on_first_run_after_the_fix(tmp_path):
    """Produktion: Health ohne parser_version, letzter Erfolg vor 5 Tagen (< min_refetch_days=30), Manifest 'alt'."""
    cfg = dict(SourceRegistry().sources["noaa_storm_events"])
    assert cfg["min_refetch_days"] == 30 and wx.NoaaStormEventsConnector.parser_version == "storm-events-2"
    now = datetime(2026, 10, 16, 12, 0, tzinfo=UTC)
    reg.save_health(tmp_path, {"noaa_storm_events": {"source_id": "noaa_storm_events", "status": "WARN",
                                                     "last_success": (now - 5 * DAY).isoformat()}})
    day = tmp_path / "manifests" / "2026-10-11"
    day.mkdir(parents=True)
    (day / "20261011T120000Z-x.json").write_text(json.dumps(
        {"entries": [{"source_id": "noaa_storm_events", "parser_version": "1"}]}))
    registry = fake_registry(tmp_path, {"noaa_storm_events": cfg}, {"noaa_storm_events": wx.NoaaStormEventsConnector})
    with storm_http(_storm_csv([("TEXAS", "Hail"), ("TEXAS", "Tornado")]), now):
        out = orch.run_ingestion(now, registry=registry)["sources"]["noaa_storm_events"]
        again = orch.run_ingestion(now + DAY, registry=registry)["sources"]["noaa_storm_events"]
    assert out["status"] == "PASS" and out["counts"]["new"] == 2 and out["dq_issues"] == []
    assert reg.load_health(tmp_path)["noaa_storm_events"]["parser_version"] == "storm-events-2"
    assert again["skipped"] == "min_refetch_days"


# ═══ 5. NOAA Storm Events ═════════════════════════════════════════════════════════════════════════════════════
NOW_S = datetime(2026, 10, 16, 12, 0, tzinfo=UTC)
STORM_LISTING = (b'<a href="StormEvents_details-ftp_v1.0_d2024_c20250101.csv.gz">a</a>'
                 b'<a href="StormEvents_details-ftp_v1.0_d2025_c20260101.csv.gz">b</a>')


def _storm_csv(rows) -> str:
    return "STATE,EVENT_TYPE,EPISODE_ID\n" + "".join(f"{s},{e},1\n" for s, e in rows)


def storm_http(csv_text, when, *, gz=True, listing=STORM_LISTING):
    body = gzip.compress(csv_text.encode()) if gz else csv_text.encode()

    def fake(url, **kw):
        return fr(listing if url.endswith("csvfiles") else body, when, url=url)
    return patch.object(wx.http, "fetch", side_effect=fake)


def storm_fetch(rows, when=NOW_S, *, gz=True, cfg=None, now=NOW_S):
    with storm_http(_storm_csv(rows), when, gz=gz):
        return wx.NoaaStormEventsConnector(dict({"backfill_mode": True}, **(cfg or {}))).fetch(now)


STORM_ROWS = ([("TEXAS", "Hail")] * 3 + [("TEXAS", "Tornado"), ("OKLAHOMA", "Hail"), ("OKLAHOMA", "Tornado"),
                                         ("TEXAS", "Marine Hail"), (" TEXAS ", "Hail"), ("", "Hail"), ("TEXAS", "")])


def test_storm_series_id_carries_year_and_event_type_and_pairs_never_collide():
    r = storm_fetch(STORM_ROWS)
    assert r.status == SourceStatus.PASS
    got = {(o.entity_id, o.series_id): o.value for o in r.observations}
    assert got == {("TEXAS", "2025:Hail"): 4.0, ("TEXAS", "2025:Tornado"): 1.0, ("TEXAS", "2025:Marine Hail"): 1.0,
                   ("OKLAHOMA", "2025:Hail"): 1.0, ("OKLAHOMA", "2025:Tornado"): 1.0}
    assert len({o.identity_key() for o in r.observations}) == len(r.observations) == 5
    d = dq.assess(r.observations, SourceRegistry().sources["noaa_storm_events"], NOW_S)
    assert d["issues"] == [] and d["duplicate_conflicts"] == 0
    assert all(o.attrs["event_type"] == o.series_id.split(":", 1)[1] and o.attrs["year"] == 2025
               for o in r.observations)
    assert all(o.observation_time == datetime(2025, 1, 1, tzinfo=UTC) and o.available_at == o.retrieved_at == NOW_S
               for o in r.observations)
    assert check_pit_integrity(r.observations) == []


def test_storm_old_identity_scheme_would_have_been_flagged_by_the_dq_gate():
    """Sanity: der Test oben haette den alten Fehler (series_id = Dateiname) erkannt."""
    r = storm_fetch(STORM_ROWS)
    old = []
    for o in r.observations:
        d = o.to_dict()
        d["series_id"] = o.attrs["file"]
        old.append(Observation.from_dict(d))
    res = dq.assess(old, SourceRegistry().sources["noaa_storm_events"], NOW_S)
    assert "DUPLICATE_CONFLICT" in res["issues"] and res["duplicate_conflicts"] >= 1


@pytest.mark.parametrize("gz", [True, False])
def test_storm_reimport_of_identical_file_is_idempotent(tmp_path, gz):
    arch = ExternalArchive(tmp_path)
    r1 = storm_fetch(STORM_ROWS, gz=gz)
    assert arch.store_observations(r1.observations) == {"new": 5, "duplicate": 0, "revision": 0}
    r2 = storm_fetch(STORM_ROWS, when=NOW_S + 30 * DAY, gz=gz, now=NOW_S + 30 * DAY)          # spaeterer Abruf, gleiche Datei
    assert arch.store_observations(r2.observations) == {"new": 0, "duplicate": 5, "revision": 0}
    assert len(arch.load("noaa_storm_events")) == 5
    changed = [("TEXAS", "Hail")] + STORM_ROWS
    r3 = storm_fetch(changed, when=NOW_S + 60 * DAY, gz=gz, now=NOW_S + 60 * DAY)             # NOAA ergaenzt einen Hagel-Fall
    assert arch.store_observations(r3.observations) == {"new": 0, "duplicate": 4, "revision": 1}
    rows = {o.entity_id + "|" + o.series_id: o.value for o in arch.as_of("noaa_storm_events", NOW_S + 45 * DAY)}
    assert rows["TEXAS|2025:Hail"] == 4.0
    rows = {o.entity_id + "|" + o.series_id: o.value for o in arch.as_of("noaa_storm_events", NOW_S + 61 * DAY)}
    assert rows["TEXAS|2025:Hail"] == 5.0 and arch.as_of("noaa_storm_events", NOW_S - DAY) == []


def test_storm_year_selection_and_failure_modes():
    r = storm_fetch(STORM_ROWS, cfg={"year": 2024})
    assert {o.attrs["file"] for o in r.observations} == {"StormEvents_details-ftp_v1.0_d2024_c20250101.csv.gz"}
    assert all(o.series_id.startswith("2024:") for o in r.observations)
    with storm_http(_storm_csv(STORM_ROWS), NOW_S, listing=b'<a href="StormEvents_details-ftp_v1.0_d2019_c1.csv">x</a>'):
        assert wx.NoaaStormEventsConnector({"backfill_mode": True}).fetch(NOW_S).status == SourceStatus.SCHEMA_CHANGED
    assert storm_fetch([("", ""), ("", "Hail")]).status == SourceStatus.SCHEMA_CHANGED         # keine verwertbaren Zeilen
    with storm_http("", NOW_S) as m:
        m.side_effect = lambda url, **kw: fr(STORM_LISTING if url.endswith("csvfiles") else b"\x1f\x8bkaputt", NOW_S, url=url)
        assert wx.NoaaStormEventsConnector({"backfill_mode": True}).fetch(NOW_S).status == SourceStatus.SCHEMA_CHANGED
    assert wx.NoaaStormEventsConnector({}).fetch(NOW_S).status == SourceStatus.DEFERRED       # ohne backfill_mode inaktiv


# ═══ 6. SEC 13F ═══════════════════════════════════════════════════════════════════════════════════════════════
def _rec(entity_id, ticker=None, usage="pit", name="", cik=None):
    return SimpleNamespace(entity_id=entity_id, ticker=ticker, cik=cik, usage=usage, canonical_name=name, aliases=[],
                           parent_entity=None, industry=None, valid_to=None)


IDX = ce.entity_index([_rec("cik:3", "GOOGL", "research_ticker_identity", cik="3"),
                       _rec("cik:3", "GOOG", "research_ticker_identity", cik="3"), _rec("cik:3", name="ALPHABET INC")])
NOW_F = datetime(2026, 10, 15, 12, 0, tzinfo=UTC)


def f13(d: date) -> str:
    return d.strftime("%d-%b-%Y").upper()


def zip13f(filings, holdings=None) -> bytes:
    """filings: [(acc, filed, type, cik, period)] mit date-Objekten; holdings: [(acc, shares)] (Alphabet, Aktien)."""
    sub = "ACCESSION_NUMBER\tFILING_DATE\tSUBMISSIONTYPE\tCIK\tPERIODOFREPORT\n" + "".join(
        f"{a}\t{f13(f)}\t{t}\t{c}\t{f13(p)}\n" for a, f, t, c, p in filings)
    info = ("ACCESSION_NUMBER\tNAMEOFISSUER\tTITLEOFCLASS\tCUSIP\tVALUE\tSSHPRNAMT\tSSHPRNAMTTYPE\tPUTCALL\n" + "".join(
        f"{a}\tALPHABET INC\tCAP STK CL A\t02079K305\t1\t{sh_}\tSH\t\n"
        for a, sh_ in (holdings if holdings is not None else [(f[0], 100) for f in filings])))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("SUBMISSION.tsv", sub)
        z.writestr("INFOTABLE.tsv", info)
    return buf.getvalue()


def _links(*names):
    return "".join(f'<a href="/files/x/{n}">x</a>' for n in names)


@pytest.mark.parametrize("name,want", [
    ("2023q1_form13f.zip", (date(2023, 1, 1), date(2023, 3, 31))),
    ("2023q2_form13f.zip", (date(2023, 4, 1), date(2023, 6, 30))),
    ("2023q3_form13f.zip", (date(2023, 7, 1), date(2023, 9, 30))),
    ("2023q4_form13f.zip", (date(2023, 10, 1), date(2023, 12, 31))),
    ("2020Q4_FORM13F.ZIP", (date(2020, 10, 1), date(2020, 12, 31))),
    ("01jun2026-31aug2026_form13f.zip", (date(2026, 6, 1), date(2026, 8, 31))),
    ("01dec2023-29feb2024_form13f.zip", (date(2023, 12, 1), date(2024, 2, 29))),
    ("01dec2025-28feb2026_form13f.zip", (date(2025, 12, 1), date(2026, 2, 28))),
    ("2023q5_form13f.zip", None), ("2023q0_form13f.zip", None), ("01dec2023-30feb2024_form13f.zip", None),
    ("01xyz2023-31dec2023_form13f.zip", None), ("form13f.zip", None)])
def test_13f_set_range_for_both_naming_schemes(name, want):
    assert ce.set_range(name) == want


def _scheme_sets():
    quarterly = [f"{y}q{q}_form13f.zip" for y in range(2022, 2028) for q in (1, 2, 3, 4)]
    monthly = []
    for y in range(2022, 2028):
        leap = (y + 1) % 4 == 0
        monthly += [f"01mar{y}-31may{y}_form13f.zip", f"01jun{y}-31aug{y}_form13f.zip",
                    f"01sep{y}-30nov{y}_form13f.zip", f"01dec{y}-{29 if leap else 28}feb{y + 1}_form13f.zip"]
    return {"quarterly": quarterly, "monthly": monthly}


@pytest.mark.parametrize("scheme", ["quarterly", "monthly"])
def test_13f_every_quarter_end_period_is_claimed_by_exactly_one_set(scheme):
    sets = _scheme_sets()[scheme]
    assert all(ce.set_range(n) for n in sets)
    ends = [date(y, m, d) for y in range(2023, 2027) for m, d in ((3, 31), (6, 30), (9, 30), (12, 31))]
    for period in ends:
        content = zip13f([("A1", period + timedelta(days=10), "13F-HR", "900", period)])
        claimed = [n for n in sets if ce.aggregate_13f(content, ce.set_range(n), IDX)[0]]
        assert len(claimed) == 1, (period, claimed)


def test_13f_timeliness_window_and_range_edges():
    p = date(2026, 6, 30)
    rng = (date(2026, 6, 1), date(2026, 8, 31))
    filings = [("A0", p, "13F-HR", "900", p),                                        # am Stichtag: nicht nach Periodenende
               ("A1", p + timedelta(days=1), "13F-HR", "901", p),                    # erster Tag
               ("A2", p + timedelta(days=45), "13F-HR", "902", p),                   # letzter fristgerechter Tag
               ("A3", p + timedelta(days=46), "13F-HR", "903", p),                   # verspaetet
               ("A4", p + timedelta(days=5), "13F-HR/A", "904", p),                  # Amendment
               ("A5", p + timedelta(days=5), "NT 13F", "905", p)]
    rows, st = ce.aggregate_13f(zip13f(filings, [(f[0], 100) for f in filings]), rng, IDX)
    [r] = rows
    assert r["holders"] == 2 and r["shares"] == 200 and st["filings"] == 2
    assert r["available"] == p + timedelta(days=46)                                  # Tag nach dem letzten einbezogenen Filing
    content = zip13f(filings, [(f[0], 100) for f in filings])
    assert ce.aggregate_13f(content, (rng[0], date(2026, 8, 13)), IDX)[0] == []      # Fristfenster endet am 14.08.
    assert ce.aggregate_13f(content, (date(2026, 7, 2), rng[1]), IDX)[0] == []       # Fenster beginnt vor Datensatzbeginn


def _run13(tmp_path, html, monkeypatch, zips, when=NOW_F, cfg=None):
    monkeypatch.setattr(ce, "entity_index", lambda: IDX)

    def fake(url, **kw):
        name = url.rsplit("/", 1)[-1]
        return fr(zips.get(name, b"") if url.endswith(".zip") else html, when, url=url)
    with patch.object(ce.http, "fetch", side_effect=fake):
        res = ce.Sec13FConnector(dict({"_archive_root": str(tmp_path)}, **(cfg or {}))).fetch(when)
    ExternalArchive(tmp_path).store_observations(res.observations)
    return res


def _markers(res):
    return [o.series_id for o in res.observations if o.metric == "dataset_processed"]


Q3 = date(2026, 6, 30)
ZIP_Q3 = zip13f([("A1", date(2026, 8, 1), "13F-HR", "900", Q3), ("A2", date(2026, 8, 14), "13F-HR", "901", Q3)])


def test_13f_backfill_newest_first_never_moves_the_data_stand_backwards(tmp_path, monkeypatch):
    names = ["2026q3_form13f.zip", "01dec2025-28feb2026_form13f.zip", "2026q2_form13f.zip", "2025q4_form13f.zip"]
    html, zips = _links(*names), {n: ZIP_Q3 for n in names}
    seen, order = [], []
    for i in range(6):
        res = _run13(tmp_path, html, monkeypatch, zips, when=NOW_F + i * DAY)
        seen.append(res.latest_observation_time)
        order += [m.split(":", 1)[1] for m in _markers(res)]
        assert res.status == SourceStatus.PASS
    assert order == ["2026q3_form13f.zip", "2026q2_form13f.zip", "01dec2025-28feb2026_form13f.zip",
                     "2025q4_form13f.zip"]                                                  # nach Fensterende absteigend
    assert seen == [datetime(2026, 9, 30, tzinfo=UTC)] * 6                                 # Stand bleibt der neueste Satz
    assert "kein neuer" in res.message and res.observations == []


def test_13f_mixed_naming_orders_by_range_end(tmp_path, monkeypatch):
    names = ["2023q4_form13f.zip", "01dec2023-29feb2024_form13f.zip"]
    res = _run13(tmp_path, _links(*names), monkeypatch, {n: ZIP_Q3 for n in names})
    assert _markers(res) == ["set:01dec2023-29feb2024_form13f.zip"]
    assert res.latest_observation_time == datetime(2024, 2, 29, tzinfo=UTC)


def test_13f_new_quarter_advances_the_stand_during_backfill(tmp_path, monkeypatch):
    zips = {n: ZIP_Q3 for n in ("2026q1_form13f.zip", "2026q2_form13f.zip", "2026q3_form13f.zip")}
    r1 = _run13(tmp_path, _links("2026q2_form13f.zip", "2026q1_form13f.zip"), monkeypatch, zips)
    r2 = _run13(tmp_path, _links("2026q3_form13f.zip", "2026q2_form13f.zip", "2026q1_form13f.zip"), monkeypatch, zips,
                when=NOW_F + DAY)
    r3 = _run13(tmp_path, _links("2026q3_form13f.zip", "2026q2_form13f.zip", "2026q1_form13f.zip"), monkeypatch, zips,
                when=NOW_F + 2 * DAY)
    assert [_markers(r)[0] for r in (r1, r2, r3)] == ["set:2026q2_form13f.zip", "set:2026q3_form13f.zip",
                                                      "set:2026q1_form13f.zip"]
    assert [r.latest_observation_time for r in (r1, r2, r3)] == [datetime(2026, 6, 30, tzinfo=UTC),
                                                                 datetime(2026, 9, 30, tzinfo=UTC),
                                                                 datetime(2026, 9, 30, tzinfo=UTC)]


def test_13f_max_sets_per_run_and_idempotent_reimport(tmp_path, monkeypatch):
    names = ["2026q3_form13f.zip", "2026q2_form13f.zip", "2026q1_form13f.zip"]
    zips = {n: ZIP_Q3 for n in names}
    res = _run13(tmp_path, _links(*names), monkeypatch, zips, cfg={"max_sets_per_run": 2})
    assert _markers(res) == ["set:2026q3_form13f.zip", "set:2026q2_form13f.zip"]
    assert res.latest_observation_time == datetime(2026, 9, 30, tzinfo=UTC)
    arch = ExternalArchive(tmp_path)
    assert arch.store_observations(res.observations) == {"new": 0, "duplicate": len(res.observations), "revision": 0}
    inst = [o for o in arch.load("sec_13f_holdings") if o.metric == "inst_holders"]
    assert {(o.entity_id, o.observation_time) for o in inst} == {("cik:3", datetime(2026, 6, 30, tzinfo=UTC))}


def test_13f_row_pit_availability_and_historical_queries(tmp_path, monkeypatch):
    res = _run13(tmp_path, _links("2026q3_form13f.zip"), monkeypatch, {"2026q3_form13f.zip": ZIP_Q3})
    assert check_pit_integrity(res.observations) == []
    holders = next(o for o in res.observations if o.metric == "inst_holders")
    assert holders.available_at == datetime(2026, 8, 15, tzinfo=UTC) and holders.retrieved_at == NOW_F
    assert holders.value == 2.0 and holders.attrs["tickers"] == ["GOOG", "GOOGL"]
    arch = ExternalArchive(tmp_path)
    assert arch.as_of("sec_13f_holdings", datetime(2026, 8, 14, 23, 59, 59, tzinfo=UTC)) == []
    on_day = arch.as_of("sec_13f_holdings", datetime(2026, 8, 15, tzinfo=UTC))
    assert {o.metric for o in on_day} == {"inst_holders", "inst_shares"}                   # Marker erst ab Abruf
    assert {o.metric for o in arch.as_of("sec_13f_holdings", NOW_F)} == {"inst_holders", "inst_shares", "dataset_processed"}


def test_13f_filing_dated_after_retrieval_is_caught_as_genuine_lookahead(tmp_path, monkeypatch):
    p = date(2026, 9, 30)
    z = zip13f([("A1", date(2026, 10, 20), "13F-HR", "900", p)])                           # Filing nach dem Abrufzeitpunkt
    res = _run13(tmp_path, _links("2026q4_form13f.zip"), monkeypatch, {"2026q4_form13f.zip": z})
    errs = check_pit_integrity(res.observations)
    assert len(errs) == 2 and all("available_at > retrieved_at" in e for e in errs)


def test_13f_failed_set_leaves_no_marker_and_is_retried(tmp_path, monkeypatch):
    names = ["2026q3_form13f.zip"]
    res = _run13(tmp_path, _links(*names), monkeypatch, {"2026q3_form13f.zip": b"kein zip"})
    assert res.status == SourceStatus.SCHEMA_CHANGED and res.observations == [] and res.latest_observation_time is None
    monkeypatch.setattr(ce, "entity_index", lambda: IDX)
    with patch.object(ce.http, "fetch", side_effect=lambda url, **kw: (_ for _ in ()).throw(http.FetchError("503"))
                      if url.endswith(".zip") else fr(_links(*names), NOW_F, url=url)):
        failed = ce.Sec13FConnector({"_archive_root": str(tmp_path)}).fetch(NOW_F)
    assert failed.status == SourceStatus.FAIL and failed.observations == []
    ok = _run13(tmp_path, _links(*names), monkeypatch, {"2026q3_form13f.zip": ZIP_Q3}, when=NOW_F + DAY)
    assert _markers(ok) == ["set:2026q3_form13f.zip"]


# FINDING (medium) der unabhängigen Prüfung – behoben 2026-10-16; bleibt als Regressionstest.
def test_13f_steady_state_without_new_dataset_is_not_a_warning(tmp_path, monkeypatch):
    cfg = dict(SourceRegistry().sources["sec_13f_holdings"])
    monkeypatch.setattr(ce, "entity_index", lambda: IDX)
    html = _links("2026q3_form13f.zip")
    outs = []
    with patch.object(ce.http, "fetch", side_effect=lambda url, **kw: fr(
            ZIP_Q3 if url.endswith(".zip") else html, NOW_F, url=url)):
        for i in range(3):
            registry = fake_registry(tmp_path, {"sec_13f_holdings": cfg}, {"sec_13f_holdings": ce.Sec13FConnector})
            outs.append(orch.run_ingestion(NOW_F + i * DAY, registry=registry)["sources"]["sec_13f_holdings"])
    assert outs[0]["status"] == "PASS" and outs[0]["counts"]["new"] > 0                       # Import des Datensatzes
    assert [o["status"] for o in outs[1:]] == ["PASS", "PASS"]                               # idempotente Folgelaeufe
    assert all(o["dq_issues"] == [] for o in outs[1:])
    cfg_sh, checks = _health_checks(tmp_path, {"sec_13f_holdings": cfg}, NOW_F + 2 * DAY)
    assert sh.classify(checks["sec_13f_holdings"], None, cfg_sh, NOW_F + 2 * DAY)["raw_status"] == sh.HEALTHY


def test_13f_unreadable_archive_is_treated_as_first_import_with_warning(tmp_path, monkeypatch, caplog):
    names = ["2026q3_form13f.zip", "2026q2_form13f.zip"]
    zips = {n: ZIP_Q3 for n in names}
    _run13(tmp_path, _links(*names), monkeypatch, zips)
    for p in (tmp_path / "normalized" / "sec_13f_holdings").glob("*"):
        p.write_text("kaputt{")
    monkeypatch.setattr(ce, "entity_index", lambda: IDX)
    with caplog.at_level(logging.WARNING, logger=ce.log.name), patch.object(
            ce.http, "fetch", side_effect=lambda url, **kw: fr(ZIP_Q3 if url.endswith(".zip") else _links(*names),
                                                                NOW_F, url=url)):
        res = ce.Sec13FConnector({"_archive_root": str(tmp_path)}).fetch(NOW_F)
    assert _markers(res) == ["set:2026q3_form13f.zip"] and res.latest_observation_time == datetime(2026, 9, 30, tzinfo=UTC)
    assert any("Archiv nicht lesbar" in r.getMessage() for r in caplog.records)


# ═══ 7. Nasdaq-100: Wikipedia -> nasdaq.com -> statisch ═════════════════════════════════════════════════════════
def _sym(i: int) -> str:
    return f"Q{chr(65 + i // 26)}{chr(65 + i % 26)}"


def ndx_table(n, col="Ticker", *, tid=None, last="BRK.B", tickers=None):
    head = f"<tr><th>Company</th><th>{col}</th></tr>"
    syms = tickers if tickers is not None else [_sym(i) for i in range(n - 1)] + [last]
    body = "".join(f"<tr><td>F{i}</td><td>{s}</td></tr>" for i, s in enumerate(syms))
    return f"<table{f' id={tid!r}' if tid else ''} class='wikitable'>{head}{body}</table>"


def changes_table():
    return ("<table class='wikitable'><tr><th rowspan=2>Date</th><th colspan=2>Added</th><th colspan=2>Removed</th></tr>"
            "<tr><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th></tr>"
            "<tr><td>2026-09-22</td><td>NEW1</td><td>New One</td><td>OLD1</td><td>Old One</td></tr></table>")


@pytest.fixture
def official(monkeypatch):
    box = {"list": [], "calls": 0}

    def fake():
        box["calls"] += 1
        return list(box["list"])
    monkeypatch.setattr(mp, "fetch_nasdaq100", fake)
    return box


@pytest.fixture
def serve(monkeypatch):
    def _serve(*tables):
        html = "<html><body>" + "".join(tables) + "</body></html>"
        monkeypatch.setattr(uv.requests, "get", lambda *a, **k: SimpleNamespace(text=html, raise_for_status=lambda: None))
    return _serve


OFFICIAL = ["AAPL", "BRK-B", "SHOP"] + [_sym(i) for i in range(98)]


@pytest.mark.parametrize("col", ["Ticker", "Symbol", "Ticker symbol"])
def test_wikipedia_table_with_any_accepted_ticker_column(serve, official, col):
    official["list"] = OFFICIAL
    serve(changes_table(), ndx_table(101, col))
    lst, src = uv._nasdaq100_live()
    assert src == "wikipedia" and len(lst) == 101 and "BRK-B" in lst and "NEW1" not in lst
    assert official["calls"] == 0                                                           # Wikipedia genuegt: nasdaq.com unberuehrt


@pytest.mark.parametrize("n,accepted", [(79, False), (80, True), (120, True), (121, False)])
def test_wikipedia_plausibility_bounds_80_to_120_rows(serve, official, n, accepted):
    official["list"] = OFFICIAL
    serve(ndx_table(n, tid="constituents"))
    lst, src = uv._nasdaq100_live()
    assert (src, len(lst)) == (("wikipedia", n) if accepted else ("nasdaq_official", len(OFFICIAL)))


def test_wikipedia_multirow_thead_and_blank_tickers(serve, official):
    official["list"] = OFFICIAL
    thead = "<thead><tr><th rowspan=2>Company</th><th>Index</th></tr><tr><th>Ticker</th></tr></thead>"
    body = "".join(f"<tr><td>F{i}</td><td>{_sym(i)}</td></tr>" for i in range(101))
    serve(f"<table class='wikitable'>{thead}<tbody>{body}</tbody></table>")
    assert uv._nasdaq100_live()[1] == "wikipedia"
    serve(ndx_table(101, tickers=[_sym(i) if i % 2 == 0 else "" for i in range(101)]))       # nur 51 echte Ticker
    assert uv._nasdaq100_live() == (OFFICIAL, "nasdaq_official")


# FINDING (low) der unabhängigen Prüfung – behoben 2026-10-16; bleibt als Regressionstest.
def test_wikipedia_table_of_repeated_tickers_is_not_plausible(serve, official):
    official["list"] = OFFICIAL
    serve(ndx_table(101, tickers=["AAPL"] * 101))
    assert uv._nasdaq100_live() == (OFFICIAL, "nasdaq_official")


def test_official_list_is_used_when_wikipedia_fails_or_has_no_table(serve, official, monkeypatch):
    official["list"] = OFFICIAL
    serve(changes_table(), "<table><tr><th>Year</th><th>Level</th></tr><tr><td>2025</td><td>1</td></tr></table>")
    assert uv._nasdaq100_live() == (OFFICIAL, "nasdaq_official")
    assert uv._fetch_nasdaq100() == OFFICIAL                                                 # get_universe sieht nur die Liste

    def boom(*a, **k):
        raise RuntimeError("Wikipedia down")
    monkeypatch.setattr(uv.requests, "get", boom)
    assert uv._nasdaq100_live() == (OFFICIAL, "nasdaq_official")


@pytest.mark.parametrize("official_behaviour", ["empty", "none", "raises"])
def test_static_fallback_only_when_both_sources_fail(serve, official, monkeypatch, official_behaviour):
    serve(changes_table())
    if official_behaviour == "none":
        monkeypatch.setattr(mp, "fetch_nasdaq100", lambda: None)
    elif official_behaviour == "raises":
        monkeypatch.setattr(mp, "fetch_nasdaq100", lambda: (_ for _ in ()).throw(RuntimeError("nasdaq.com")))
    assert uv._nasdaq100_live() == ([], "none") and uv._fetch_nasdaq100() == []
    monkeypatch.setattr(uv, "_fetch_sp500", lambda: [])
    uv.get_universe.cache_clear()
    try:
        assert uv.get_universe("nasdaq100") == uv._clean(list(uv._NASDAQ100_STATIC))
    finally:
        uv.get_universe.cache_clear()


def test_get_universe_uses_the_first_available_tier(serve, official, monkeypatch):
    monkeypatch.setattr(uv, "_fetch_sp500", lambda: [])
    official["list"] = ["SHOP"] + [_sym(i) for i in range(100)]

    def universe():
        uv.get_universe.cache_clear()
        try:
            return uv.get_universe("nasdaq100")
        finally:
            uv.get_universe.cache_clear()
    serve(ndx_table(101, tid="constituents"))
    assert _sym(5) in universe() and "SHOP" not in universe() and "BRK-B" in universe()      # Tier 1: Wikipedia
    serve(changes_table())
    assert "SHOP" in universe() and "EA" not in universe()                                   # Tier 2: nasdaq.com


def test_real_official_fetch_never_raises_and_validates_size():
    def payload(n):
        return {"data": {"data": {"rows": [{"symbol": f"T{i:03d}"} for i in range(n)]}}}
    for n, ok in ((94, False), (95, True), (110, True), (111, False)):
        with patch.object(mp.http, "fetch", return_value=fr(json.dumps(payload(n)), T0)):
            got = mp.fetch_nasdaq100()
        assert (len(got) == n) is ok and (got == []) is (not ok)
    with patch.object(mp.http, "fetch", side_effect=http.FetchError("down")):
        assert mp.fetch_nasdaq100() == []
    with patch.object(mp.http, "fetch", return_value=fr("<html>kein json</html>", T0)):
        assert mp.fetch_nasdaq100() == []


@pytest.mark.parametrize("live,expected_source", [
    ((["AAPL", "BRK-B"], "wikipedia"), "wikipedia"), ((["AAPL", "SHOP"], "nasdaq_official"), "nasdaq_official"),
    (([], "none"), "static_fallback")])
def test_production_list_mirrors_the_source_chain(monkeypatch, live, expected_source):
    monkeypatch.setattr(uv, "_nasdaq100_live", lambda: live)
    lst, src = mp.Nasdaq100MembersConnector._production_list()
    assert src == expected_source and lst == sorted(lst)
    if live[0]:
        assert lst == sorted(live[0])


def test_universe_v1_code_hash_is_unchanged():
    assert universe_v2.v1_unchanged() is True


# ═══ 8. ops_monitor.data_health ═════════════════════════════════════════════════════════════════════════════
def _paths(tmp_path, source_health):
    p = tmp_path / "sh.json"
    if source_health is not None:
        p.write_text(json.dumps(source_health))
    return {"source_health": p, "external_health": tmp_path / "ext" / "health" / "h.json", "quotes": tmp_path / "q"}


def test_data_health_explains_unvalidated_without_changing_counts(tmp_path):
    srcs = {"a": {"status": "HEALTHY"},
            "b": {"status": "UNVALIDATED", "reasons": ["nicht aktiv: status_override=DEFERRED"]},
            "c": {"status": "UNVALIDATED", "reasons": ["nicht aktiv: status_override=REVIEW_REQUIRED"]},
            "d": {"status": "UNVALIDATED", "reasons": ["nicht aktiv: disabled"]},
            "e": {"status": "UNVALIDATED", "reasons": ["kein Beleg fuer erfolgreiche Lieferung"]},     # ungeklaert
            "f": {"status": "UNVALIDATED"}, "g": {"status": "UNVALIDATED", "reasons": None},
            "h": {"status": "BROKEN", "reasons": ["nicht aktiv: disabled"]},                             # nur UNVALIDATED zaehlt
            "i": None}
    for shape in (srcs, list(srcs.values())):
        P = _paths(tmp_path, {"generated": "g", "sources": shape})
        out = ops_monitor.data_health(P, datetime(2026, 10, 16, tzinfo=UTC))["source_health"]
        assert out["by_status"] == {"HEALTHY": 1, "UNVALIDATED": 6, "BROKEN": 1, None: 1}
        assert out["inactive_by_design"] == {"DEFERRED": 1, "REVIEW_REQUIRED": 1, "disabled": 1}
    out = ops_monitor.data_health(_paths(tmp_path / "leer", None), datetime(2026, 10, 16, tzinfo=UTC))["source_health"]
    assert out == {"generated": None, "by_status": {}, "inactive_by_design": {}}


# ═══ 9. Archiv: Zaehler und historische PIT-Abfragen ════════════════════════════════════════════════════════════
def test_archive_counts_and_as_of_for_value_a_b_a(tmp_path):
    arch = ExternalArchive(tmp_path)

    def mk(value, avail):
        return Observation(source_id="s", dataset="d", series_id="S", entity_id="E", metric="m", value=value, unit="u",
                           observation_time=T0, available_at=avail, retrieved_at=avail, availability_precision=EXACT,
                           parser_version="t")
    t1, t2, t3 = T0 + DAY, T0 + 2 * DAY, T0 + 3 * DAY
    assert arch.store_observations([mk(1.0, t1)]) == {"new": 1, "duplicate": 0, "revision": 0}
    assert arch.store_observations([mk(1.0, t2)]) == {"new": 0, "duplicate": 1, "revision": 0}
    assert arch.store_observations([mk(2.0, t2)]) == {"new": 0, "duplicate": 0, "revision": 1}
    assert arch.store_observations([mk(2.0, t3), mk(1.0, t3)]) == {"new": 0, "duplicate": 1, "revision": 1}  # A -> B -> A
    micro = timedelta(microseconds=1)
    val = lambda t: [o.value for o in arch.as_of("s", t)]  # noqa: E731
    assert val(t1 - micro) == [] and val(t1) == [1.0] and val(t2 - micro) == [1.0] and val(t2) == [2.0]
    assert val(t3 - micro) == [2.0] and val(t3) == [1.0] and val(t3 + 400 * DAY) == [1.0]
    assert val(t3.astimezone(NY)) == [1.0]
    assert [o.value for o in arch.vintage_history("s", "S", "E", "m", T0)] == [1.0, 2.0, 1.0]
    assert ExternalArchive(tmp_path / "nirgendwo").as_of("s", T0) == []

"""PIT-Semantik der Kalenderquellen (fomc/bls/bea_release_calendar, P1 2026-10-16) – nur SYNTHETISCHE Fixtures.

Live-Befund: Source Health meldete 9/29/66 "PIT-Verletzungen" – tatsächlich angekündigte Zukunftstermine (DQ-Zähler
"Zeitpunkt in der Zukunft" im PIT-Feld). Geprüft:
  - angekündigte Termine sind bekannt ab dem Abruf (available_at = retrieved_at), nie früher
  - Terminänderungen (verschoben/abgesagt) bleiben als Momentaufnahme nachvollziehbar; schedule_as_of(t) liefert den
    Plan, wie er zum Zeitpunkt t bekannt war; vor der ersten Momentaufnahme -> None (für Backtests gesperrt)
  - unveränderte Wiederholungen sind idempotent (keine neuen Zeilen)
  - Zeitpunkte jenseits des erlaubten Horizonts bleiben FUTURE_OBSERVATION; Nicht-Kalenderquellen unverändert
  - echte Lookahead-Fehler (available_at > retrieved_at) werden weiter erkannt
  - Zeitzonen/DST (FOMC 14:00 New York, BEA ohne Offset = New York), ganztägige BLS-Einträge gekennzeichnet
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from modules.external import data_quality as dq
from modules.external import http
from modules.external.alerts import check_pit_integrity
from modules.external.archive import ExternalArchive
from modules.external.pit import AvailabilityPrecision, Observation
from modules.external.registry import SourceRegistry
from modules.external.sources import calendars as cal
from modules.external.sources.base import SourceStatus

UTC = timezone.utc
T0 = datetime(2026, 10, 15, 12, 0, tzinfo=UTC)


def resp(text: str, when: datetime) -> http.FetchResult:
    body = text.encode()
    return http.FetchResult(url="https://fixture", status=200, content=body, content_type="text/calendar",
                            retrieved_at=when, content_hash="h", fingerprint="fp", bytes=len(body))


def ics(*events: tuple[str, str]) -> str:
    body = "".join(f"BEGIN:VEVENT\r\nUID:{i}\r\nDTSTART{dt}\r\nSUMMARY:{s}\r\nEND:VEVENT\r\n"
                   for i, (dt, s) in enumerate(events))
    return "BEGIN:VCALENDAR\r\nVERSION:2.0\r\n" + body + "END:VCALENDAR\r\n"


CPI_NOV = (";TZID=America/New_York:20261113T083000", "Consumer Price Index for October 2026")
JOBS_NOV = (";TZID=America/New_York:20261106T083000", "Employment Situation for October 2026")
JOBS_NOV_MOVED = (";TZID=America/New_York:20261109T083000", "Employment Situation for October 2026")
PPI_DEC = (";TZID=America/New_York:20261210T083000", "Producer Price Index for November 2026")


def ingest(root, text: str, when: datetime):
    """Ein Abruf + Archivierung wie im Orchestrator (Konnektor liest seinen letzten Plan aus dem Archiv)."""
    conn = cal.BlsReleaseCalendarConnector({"_archive_root": str(root)})
    with patch.object(cal.http, "fetch", return_value=resp(text, when)):
        r = conn.fetch(when)
    assert r.status == SourceStatus.PASS
    counts = ExternalArchive(root).store_observations(r.observations)
    return r, counts


def snaps(r):
    return [o for o in r.observations if o.series_id == cal.SNAPSHOT_SERIES]


def test_announced_future_events_known_from_retrieval_only(tmp_path):
    r, _ = ingest(tmp_path, ics(CPI_NOV, JOBS_NOV), T0)
    ev = [o for o in r.observations if o.metric == "scheduled_release"]
    assert len(ev) == 2 and all(o.observation_time > T0 for o in ev)
    assert all(o.available_at == o.retrieved_at == T0 for o in r.observations)          # nie früher als der Abruf
    assert check_pit_integrity(r.observations) == []                                      # keine PIT-Verletzung
    arch = ExternalArchive(tmp_path)
    assert arch.as_of("bls_release_calendar", T0 - timedelta(seconds=1)) == []           # vorher unbekannt
    assert cal.schedule_as_of("bls_release_calendar", T0 - timedelta(seconds=1), arch) is None   # gesperrt
    plan = cal.schedule_as_of("bls_release_calendar", T0, arch)
    assert {(p["series"], p["time"]) for p in plan} == {
        ("consumer_price_index", datetime(2026, 11, 13, 13, 30, tzinfo=UTC)),
        ("employment_situation", datetime(2026, 11, 6, 13, 30, tzinfo=UTC))}


def test_unchanged_refetch_is_idempotent(tmp_path):
    ingest(tmp_path, ics(CPI_NOV, JOBS_NOV), T0)
    r2, counts = ingest(tmp_path, ics(CPI_NOV, JOBS_NOV), T0 + timedelta(days=1))
    assert snaps(r2) == [] and counts["new"] == 0 and counts["revision"] == 0             # nichts Neues archiviert
    assert "Plan geändert" not in r2.message


def test_moved_and_cancelled_events_are_traceable(tmp_path):
    arch = ExternalArchive(tmp_path)
    ingest(tmp_path, ics(CPI_NOV, JOBS_NOV), T0)
    t1 = T0 + timedelta(days=2)
    r1, _ = ingest(tmp_path, ics(CPI_NOV, JOBS_NOV_MOVED, PPI_DEC), t1)                  # verschoben + neu
    [s1] = snaps(r1)
    assert s1.attrs["withdrawn"] == ["employment_situation|2026-11-06T13:30:00+00:00|1"]
    assert sorted(s1.attrs["added"]) == ["employment_situation|2026-11-09T13:30:00+00:00|1",
                                         "producer_price_index|2026-12-10T13:30:00+00:00|1"]
    assert "+2 / −1" in r1.message
    t2 = t1 + timedelta(days=1)
    r2, _ = ingest(tmp_path, ics(CPI_NOV, JOBS_NOV_MOVED), t2)                           # PPI abgesagt
    assert snaps(r2)[0].attrs["withdrawn"] == ["producer_price_index|2026-12-10T13:30:00+00:00|1"]

    def plan(t):
        return sorted((p["series"], p["time"].day) for p in cal.schedule_as_of("bls_release_calendar", t, arch))
    assert plan(T0 + timedelta(hours=1)) == [("consumer_price_index", 13), ("employment_situation", 6)]
    assert plan(t1) == [("consumer_price_index", 13), ("employment_situation", 9), ("producer_price_index", 10)]
    assert plan(t2) == [("consumer_price_index", 13), ("employment_situation", 9)]
    # Terminzeilen allein (as_of) behalten den alten Termin – genau deshalb die Momentaufnahme
    rows = [o for o in arch.as_of("bls_release_calendar", t2) if o.metric == "scheduled_release"]
    assert len(rows) == 4


def test_events_aging_out_of_window_are_not_withdrawals(tmp_path):
    ingest(tmp_path, ics(JOBS_NOV, PPI_DEC), T0)
    later = datetime(2026, 11, 20, 12, 0, tzinfo=UTC)                                      # Nov-Termin > 7 T alt
    r, _ = ingest(tmp_path, ics(JOBS_NOV, PPI_DEC), later)
    assert snaps(r) == []                                                                  # kein "zurückgezogen"


def test_horizon_keeps_far_future_and_non_calendar_detection():
    reg = SourceRegistry()
    cfg = reg.sources["bls_release_calendar"]
    assert cfg["allow_future_observations"] is True and cfg["max_future_days"] == 410

    def ob(days):
        t = T0 + timedelta(days=days)
        return Observation(source_id="bls_release_calendar", dataset="bls_schedule", series_id="x", entity_id="US",
                           metric="scheduled_release", value=1.0, unit="event", observation_time=t, available_at=T0,
                           retrieved_at=T0, availability_precision=AvailabilityPrecision.EXACT_TIMESTAMP, parser_version="t")
    ok = dq.assess([ob(30), ob(400)], cfg, T0)
    assert "FUTURE_OBSERVATION" not in ok["issues"] and ok["details"]["scheduled_future"] == 2
    far = dq.assess([ob(30), ob(4000)], cfg, T0)                                          # z. B. falsches Jahr
    assert "FUTURE_OBSERVATION" in far["issues"] and far["future_observations"] == 1 and far["severe"]
    other = dq.assess([ob(30)], reg.sources["fred_liquidity"], T0)
    assert "FUTURE_OBSERVATION" in other["issues"]                                         # Gate unverändert


def test_genuine_lookahead_still_detected():
    bad = Observation(source_id="bls_release_calendar", dataset="bls_schedule", series_id="x", entity_id="US",
                      metric="scheduled_release", value=1.0, unit="event", observation_time=T0 + timedelta(days=5),
                      available_at=T0 + timedelta(days=1), retrieved_at=T0,
                      availability_precision=AvailabilityPrecision.EXACT_TIMESTAMP, parser_version="t")
    assert len(check_pit_integrity([bad])) == 1                                            # available_at > retrieved_at


def test_dst_and_all_day_entries(tmp_path):
    allday = (";VALUE=DATE:20261201", "Job Openings and Labor Turnover Survey for October 2026")
    r, _ = ingest(tmp_path, ics(("" + ";TZID=America/New_York:20261030T083000", "GDP Q3"),
                                (";TZID=America/New_York:20261104T083000", "Productivity Q3"), allday), T0)
    ev = {o.series_id: o for o in r.observations if o.metric == "scheduled_release"}
    assert ev["gdp_q3"].observation_time.hour == 12                                        # EDT: 08:30 = 12:30 UTC
    assert ev["productivity_q3"].observation_time.hour == 13                               # EST nach 01.11.
    jolts = ev["job_openings_and_labor_turnover_survey"]
    assert jolts.attrs.get("time_assumed") is True and "time_assumed" not in ev["gdp_q3"].attrs
    bea = cal.parse_bea_json({"GDP": {"release_dates": ["2026-10-29T08:30:00", "2027-03-26T08:30:00"]}})
    assert [e["start"].astimezone(UTC).hour for e in bea] == [12, 12]                      # EDT / EDT (nach 14.03.)
    bea_w = cal.parse_bea_json({"GDP": {"release_dates": ["2026-12-23T08:30:00"]}})
    assert bea_w[0]["start"].astimezone(UTC).hour == 13                                    # EST


def test_fomc_sep_flag_change_creates_new_snapshot(tmp_path):
    html1 = ('<h4><a id="1">2026 FOMC Meetings</a></h4><div class="row fomc-meeting"><div class="fomc-meeting__month '
             'col-xs-5"><strong>December</strong></div><div class="fomc-meeting__date col-xs-4">8-9</div></div>')
    html2 = html1.replace("8-9", "8-9*")
    conn = cal.FomcCalendarConnector({"_archive_root": str(tmp_path)})
    arch = ExternalArchive(tmp_path)
    for i, h in enumerate((html1, html2)):
        with patch.object(cal.http, "fetch", return_value=resp(h, T0 + timedelta(days=i))):
            r = conn.fetch(T0 + timedelta(days=i))
        arch.store_observations(r.observations)
    plan = cal.schedule_as_of("fomc_calendar", T0 + timedelta(days=1), arch)
    assert [p["value"] for p in plan] == [2.0]                                             # SEP nachträglich markiert
    assert [p["value"] for p in cal.schedule_as_of("fomc_calendar", T0, arch)] == [1.0]

"""Tests für modules/ops_workflows.py (Fehler-Digest der GitHub-Actions-Läufe, Scanner-Startzeit).

Nur SYNTHETISCHE API-Antworten über eine Fake-Session – kein Netzwerk. Geprüft: Fehlerzeilen-Extraktion (Rauschen,
Zeitstempel, Duplikate), Zeitfenster seit dem vorherigen Handelstag, fehlgeschlagene Läufe mit Jobs/Schritten/Logs,
Log-Limit, Spätstart-Erkennung in New-York-Zeit und dass das Modul ohne Token oder bei API-Fehlern nie wirft.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from modules import ops_workflows as ow

REPO = "owner/repo"
ENV = {"GITHUB_TOKEN": "t0k", "GITHUB_REPOSITORY": REPO}
NOW = datetime(2026, 10, 14, 21, 0, tzinfo=timezone.utc)          # Mittwoch, 17:00 New York (EDT)
API = f"{ow.API}/repos/{REPO}"


class Resp:
    def __init__(self, payload=None, text: str = "", status: int = 200):
        self._payload, self.text, self.status_code = payload, text, status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    """url -> Resp | Exception; protokolliert alle Aufrufe."""

    def __init__(self, routes: dict):
        self.routes, self.headers, self.calls = routes, {}, []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        r = self.routes.get(url)
        if r is None:
            return Resp(status=404)
        if isinstance(r, Exception):
            raise r
        return r


def run(rid: int, name: str, path: str, conclusion: str | None, created: str = "2026-10-14T17:30:00Z") -> dict:
    return {"id": rid, "name": name, "path": path, "conclusion": conclusion, "created_at": created,
            "event": "schedule", "html_url": f"https://github.com/{REPO}/actions/runs/{rid}",
            "jobs_url": f"{API}/actions/runs/{rid}/jobs"}


def job(jid: int, name: str, conclusion: str, steps=(), started: str | None = None) -> dict:
    return {"id": jid, "name": name, "conclusion": conclusion, "started_at": started,
            "steps": [{"name": s, "conclusion": c} for s, c in steps]}


LOG = """2026-10-14T17:31:00.1234567Z ##[warning]Node.js 20 is deprecated; please update
2026-10-14T17:31:01.0000000Z Traceback (most recent call last):
2026-10-14T17:31:02.0000000Z ImportError: cannot import name 'x' from 'modules.y'
2026-10-14T17:31:03.0000000Z (node:123) [DEP0040] DeprecationWarning: The punycode module is deprecated
2026-10-14T17:31:04.0000000Z ##[error]Process completed with exit code 1.
"""


def scanner_routes(scan_started: str = "2026-10-14T17:35:00Z") -> dict:
    runs = [run(1, "Adaptive Asymmetry Scanner", ow.SCANNER_FILE, "failure"),
            run(2, "Options-Capture (RTH, nur Quote-Archiv)", ".github/workflows/options_capture.yml", "success"),
            run(3, "Source Health", ".github/workflows/source_health.yml", "cancelled")]
    return {
        f"{API}/actions/runs": Resp({"workflow_runs": runs}),
        f"{API}/actions/runs/1/jobs": Resp({"jobs": [
            job(11, "guard", "success", started="2026-10-14T17:31:00Z"),
            job(12, ow.SCAN_JOB, "failure", steps=[("Checkout", "success"), ("Scanner ausführen", "failure"),
                                                    ("Commit", "skipped")], started=scan_started)]}),
        f"{API}/actions/runs/2/jobs": Resp({"jobs": [job(21, "capture", "success")]}),
        f"{API}/actions/jobs/12/logs": Resp(text=LOG),
    }


# ── Fehlerzeilen ────────────────────────────────────────────────────────────
def test_error_lines_strip_timestamps_and_filter_noise():
    assert ow.error_lines(LOG) == ["Traceback (most recent call last):",
                                   "ImportError: cannot import name 'x' from 'modules.y'",
                                   "##[error]Process completed with exit code 1."]


def test_error_lines_dedup_keeps_latest_order_limit_and_truncation():
    text = "\n".join(["ValueError: a", "KeyError: b", "ValueError: a", "Fehler: c", "x" * 300 + " Error"])
    out = ow.error_lines(text, n=3)
    assert out[:2] == ["ValueError: a", "Fehler: c"] and len(out) == 3 and len(out[2]) == 200
    assert ow.error_lines("") == [] and ow.error_lines(None) == []
    assert ow.error_lines("alles gut\nfertig") == []


# ── Zeitfenster ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("now, hours", [
    (datetime(2026, 10, 14, 21, 0, tzinfo=timezone.utc), 26),     # Mittwoch -> seit Dienstag
    (datetime(2026, 10, 19, 21, 0, tzinfo=timezone.utc), 74),     # Montag -> seit Freitag (Wochenende)
    (datetime(2026, 11, 27, 18, 0, tzinfo=timezone.utc), 50),     # Freitag nach Thanksgiving -> seit Mittwoch
    (datetime(2026, 10, 15, 2, 0, tzinfo=timezone.utc), 26),      # 22:00 New York am 14.: NY-Datum zählt
])
def test_window_hours_since_previous_trading_day(now, hours):
    assert ow.window_hours(now) == hours


# ── collect ─────────────────────────────────────────────────────────────────
def test_collect_without_token_or_repo_is_unavailable_and_makes_no_call():
    s = FakeSession({})
    for env in ({}, {"GITHUB_TOKEN": "x"}, {"GITHUB_REPOSITORY": REPO}):
        res = ow.collect(NOW, env=env, session=s)
        assert res["status"] == "UNAVAILABLE" and "GITHUB_TOKEN" in res["reason"]
    assert s.calls == []


def test_collect_failed_runs_jobs_steps_and_log_lines():
    s = FakeSession(scanner_routes())
    res = ow.collect(NOW, env=ENV, session=s)
    assert res["status"] == "OK" and res["runs_total"] == 3 and res["window_hours"] == 26
    assert res["since"] == "2026-10-13T19:00:00Z"
    assert s.calls[0] == (f"{API}/actions/runs", {"created": ">=2026-10-13T19:00:00Z", "per_page": 100})
    assert s.headers["Authorization"] == "Bearer t0k"
    [f] = res["failed"]                                       # cancelled zählt nicht als Fehler
    assert f["workflow"] == "Adaptive Asymmetry Scanner" and f["run_id"] == 1 and f["conclusion"] == "failure"
    assert f["failed_jobs"] == [ow.SCAN_JOB] and f["failed_steps"] == ["Scanner ausführen"]
    assert f["error_lines"][-1] == "##[error]Process completed with exit code 1."
    assert "ImportError: cannot import name 'x' from 'modules.y'" in f["error_lines"]
    assert f["url"].endswith("/actions/runs/1")
    assert res["failed_by_workflow"] == {"Adaptive Asymmetry Scanner": 1}


def test_collect_unreadable_log_still_reports_run():
    routes = scanner_routes()
    routes[f"{API}/actions/jobs/12/logs"] = RuntimeError("blob gone")
    [f] = ow.collect(NOW, env=ENV, session=FakeSession(routes))["failed"]
    assert f["error_lines"] == ["(Log nicht lesbar: RuntimeError)"] and f["failed_steps"] == ["Scanner ausführen"]


def test_collect_limits_log_downloads():
    runs = [run(100 + i, f"W{i}", f".github/workflows/w{i}.yml", "failure") for i in range(10)]
    routes = {f"{API}/actions/runs": Resp({"workflow_runs": runs})}
    for i in range(10):
        routes[f"{API}/actions/runs/{100 + i}/jobs"] = Resp({"jobs": [job(900 + i, "j", "failure")]})
        routes[f"{API}/actions/jobs/{900 + i}/logs"] = Resp(text="Error: boom")
    s = FakeSession(routes)
    res = ow.collect(NOW, env=ENV, session=s)
    assert len(res["failed"]) == 10
    assert sum(1 for u, _ in s.calls if u.endswith("/logs")) == ow.MAX_LOG_JOBS
    assert res["failed"][-1]["error_lines"] == []             # Logs nicht mehr geladen, Lauf trotzdem gemeldet


def test_collect_api_error_is_unavailable_never_raises():
    res = ow.collect(NOW, env=ENV, session=FakeSession({f"{API}/actions/runs": Resp(status=403)}))
    assert res["status"] == "UNAVAILABLE" and "403" in res["reason"]
    res = ow.collect(NOW, env=ENV, session=FakeSession({f"{API}/actions/runs": ConnectionError("dns")}))
    assert res["status"] == "UNAVAILABLE" and "ConnectionError" in res["reason"]


# ── Scanner-Start ───────────────────────────────────────────────────────────
def test_scanner_start_on_time_in_new_york_time():
    st = ow.collect(NOW, env=ENV, session=FakeSession(scanner_routes("2026-10-14T17:35:00Z")))["scanner_start"]
    assert st == {"started_et": "2026-10-14T13:35-04:00", "run_id": 1, "late": False, "late_after_et": "15:00",
                  "trading_day": True}


def test_scanner_start_late_after_15_et():
    st = ow.collect(NOW, env=ENV, session=FakeSession(scanner_routes("2026-10-14T19:20:00Z")))["scanner_start"]
    assert st["late"] is True and st["started_et"] == "2026-10-14T15:20-04:00"


def test_scanner_start_ignores_skipped_and_previous_day_jobs():
    routes = scanner_routes("2026-10-13T17:35:00Z")           # Scan-Job von gestern (NY)
    st = ow.collect(NOW, env=ENV, session=FakeSession(routes))["scanner_start"]
    assert st == {"started_et": None, "trading_day": True}
    routes = scanner_routes()
    routes[f"{API}/actions/runs/1/jobs"] = Resp({"jobs": [job(12, ow.SCAN_JOB, "skipped",
                                                              started="2026-10-14T17:35:00Z")]})
    st = ow.collect(NOW, env=ENV, session=FakeSession(routes))["scanner_start"]
    assert st["started_et"] is None

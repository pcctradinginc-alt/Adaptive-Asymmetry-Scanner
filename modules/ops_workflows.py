"""modules/ops_workflows.py – Fehler-Digest der GitHub-Actions-Läufe und Scanner-Startzeit (READ-ONLY, 2026-10-15).

Anlass: Der Upstream-Guard des Scanners stürzte am 12./13.10. zwei Tage lang ab (fehlender Import), der Watchdog
ebenfalls – aufgefallen ist es erst bei einer manuellen Log-Durchsicht. Dieses Modul liest über die GitHub-REST-API
(GITHUB_TOKEN / GITHUB_REPOSITORY, in Actions automatisch gesetzt) die Läufe seit dem vorherigen Handelstag:

  failed          fehlgeschlagene Läufe (failure / timed_out / startup_failure) mit fehlgeschlagenen Jobs/Schritten
                  und den letzten Fehlerzeilen aus dem Job-Log (begrenzt, Rauschen wie Node-Deprecation gefiltert)
  scanner_start   Start des heutigen Scan-Jobs in New-York-Zeit (Cron-Läufe kommen real 4–7 h verspätet; startet der
                  Scanner zu spät, fallen die RTH-Marks aus dem Fenster)

Nie schreibend, nie werfend: ohne Token oder bei API-Fehlern -> status UNAVAILABLE mit Grund.
"""
from __future__ import annotations

import logging
import os
import re
from collections import Counter
from datetime import datetime, time, timedelta, timezone

log = logging.getLogger(__name__)

API = "https://api.github.com"
FAILED = frozenset({"failure", "timed_out", "startup_failure"})
SCANNER_FILE = ".github/workflows/scanner.yml"
SCAN_JOB = "scan"
LATE_START_ET = time(15, 0)          # danach bleibt bis zum Ende des Mark-Fensters (Schluss − 10 min) kaum Zeit
MAX_RUNS, MAX_LOG_JOBS, MAX_LINES = 15, 6, 4
_TS_RE = re.compile(r"^\d{4}-\d\d-\d\dT[\d:.]+Z\s?")
_ERR_RE = re.compile(r"Traceback|Error|Exception|##\[error\]|\bFAILED\b|Fehler")
_NOISE_RE = re.compile(r"Node\.js 20 is deprecated|DeprecationWarning|punycode|##\[warning\]|trace-deprecation")


def error_lines(text: str, n: int = MAX_LINES) -> list[str]:
    """Die letzten n unterschiedlichen Fehlerzeilen eines Job-Logs (Zeitstempel entfernt, je ≤ 200 Zeichen)."""
    out: list[str] = []
    for raw in (text or "").splitlines():
        line = _TS_RE.sub("", raw).strip()
        if not line or _NOISE_RE.search(line) or not _ERR_RE.search(line):
            continue
        line = line[:200]
        if line in out:
            out.remove(line)
        out.append(line)
    return out[-n:]


def window_hours(now: datetime) -> int:
    """Stunden seit dem vorherigen NYSE-Handelstag (+2 h Puffer): montags/nach Feiertagen das ganze Wochenende."""
    from modules import market_snapshot as ms
    d = now.astimezone(ms.NY_TZ).date()
    prev = d - timedelta(days=1)
    for _ in range(10):
        if ms.is_trading_day(prev):
            break
        prev -= timedelta(days=1)
    return 24 * (d - prev).days + 2


def _get(session, url: str, params: dict | None = None):
    resp = session.get(url, params=params, timeout=20)
    resp.raise_for_status()
    return resp


def collect(now: datetime | None = None, env: dict | None = None, session=None) -> dict:
    """-> {status, window_hours, runs_total, failed[...], failed_by_workflow{}, scanner_start{...}}; wirft nie."""
    now = now or datetime.now(timezone.utc)
    env = os.environ if env is None else env
    token, repo = env.get("GITHUB_TOKEN"), env.get("GITHUB_REPOSITORY")
    if not token or not repo:
        return {"status": "UNAVAILABLE", "reason": "GITHUB_TOKEN/GITHUB_REPOSITORY fehlen (nur in GitHub Actions gesetzt)"}
    try:
        if session is None:
            import requests
            session = requests.Session()
        session.headers.update({"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                                "X-GitHub-Api-Version": "2022-11-28"})
        hours = window_hours(now)
        since = (now - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
        runs = _get(session, f"{API}/repos/{repo}/actions/runs",
                    {"created": f">={since}", "per_page": 100}).json().get("workflow_runs") or []
        failed_runs = [r for r in runs if r.get("conclusion") in FAILED][:MAX_RUNS]
        failed, logs_left = [], MAX_LOG_JOBS
        for r in failed_runs:
            jobs = (_get(session, r["jobs_url"]).json().get("jobs") or []) if r.get("jobs_url") else []
            fj = [j for j in jobs if j.get("conclusion") in FAILED]
            lines: list[str] = []
            for j in fj:
                if logs_left <= 0:
                    break
                logs_left -= 1
                try:
                    lines += error_lines(_get(session, f"{API}/repos/{repo}/actions/jobs/{j['id']}/logs").text)
                except Exception as e:  # noqa: BLE001 – Log nicht lesbar: Lauf trotzdem melden
                    lines.append(f"(Log nicht lesbar: {type(e).__name__})")
            failed.append({"workflow": r.get("name"), "file": r.get("path"), "run_id": r.get("id"),
                           "url": r.get("html_url"), "created_at": r.get("created_at"), "event": r.get("event"),
                           "conclusion": r.get("conclusion"), "failed_jobs": [j.get("name") for j in fj],
                           "failed_steps": [s.get("name") for j in fj for s in (j.get("steps") or [])
                                            if s.get("conclusion") in FAILED],
                           "error_lines": lines[-MAX_LINES:]})
        return {"status": "OK", "window_hours": hours, "since": since, "runs_total": len(runs), "failed": failed,
                "failed_by_workflow": dict(Counter(f["workflow"] for f in failed)),
                "scanner_start": _scanner_start(session, runs, now)}
    except Exception as e:  # noqa: BLE001 – Monitoring darf nie scheitern
        return {"status": "UNAVAILABLE", "reason": f"GitHub-API: {type(e).__name__}: {str(e)[:160]}"}


def _scanner_start(session, runs: list[dict], now: datetime) -> dict:
    """Start des heutigen Scan-Jobs (New-York-Datum); der Guard-Job allein zählt nicht (SKIP-Läufe)."""
    from modules import market_snapshot as ms
    today = now.astimezone(ms.NY_TZ).date()
    for r in sorted((r for r in runs if r.get("path") == SCANNER_FILE), key=lambda r: r.get("created_at") or ""):
        jobs = (_get(session, r["jobs_url"]).json().get("jobs") or []) if r.get("jobs_url") else []
        for j in jobs:
            if j.get("name") != SCAN_JOB or j.get("conclusion") == "skipped" or not j.get("started_at"):
                continue
            t = datetime.fromisoformat(j["started_at"].replace("Z", "+00:00")).astimezone(ms.NY_TZ)
            if t.date() == today:
                return {"started_et": t.isoformat(timespec="minutes"), "run_id": r.get("id"),
                        "late": t.time() > LATE_START_ET, "late_after_et": LATE_START_ET.strftime("%H:%M"),
                        "trading_day": ms.is_trading_day(today)}
    return {"started_et": None, "trading_day": ms.is_trading_day(today)}

"""modules/ops_workflows.py – Fehler-Digest der GitHub-Actions-Läufe und Scanner-Startzeit (READ-ONLY, 2026-10-15).

Anlass: Der Upstream-Guard des Scanners stürzte am 12./13.10. zwei Tage lang ab (fehlender Import), der Watchdog
ebenfalls – aufgefallen ist es erst bei einer manuellen Log-Durchsicht. Dieses Modul liest über die GitHub-REST-API
(GITHUB_TOKEN / GITHUB_REPOSITORY, in Actions automatisch gesetzt) die Läufe auf main seit dem vorherigen Handelstag:

  failed          fehlgeschlagene Läufe (failure / timed_out / startup_failure) mit fehlgeschlagenen Jobs/Schritten
                  und den letzten Fehlerzeilen aus dem Job-Log (begrenzt, Rauschen wie Node-Deprecation gefiltert)
  scanner_start   Start des heutigen Scan-Jobs in New-York-Zeit (Cron-Läufe kommen real 4–7 h verspätet; startet der
                  Scanner zu spät, fallen die RTH-Marks aus dem Fenster)

  governance      (Audit 2026-11-07) Branch-Schutz von main und gemergte PRs der letzten Tage, die CODEOWNERS-Pfade
                  (Bewertungsprotokolle, Gates, Datenkatalog, Workflows) ohne freigebendes Review geändert haben.
                  CODEOWNERS selbst sagt: "Wirksam erst mit Branch-Protection 'Require review from Code Owners'".
                  Live 07.11.: main protected=false; #165 (research_lab.py) und #166 (data_catalog.yaml) ohne Review gemergt.

Nie schreibend, nie werfend: ohne Token oder bei API-Fehlern -> status UNAVAILABLE mit Grund.
"""
from __future__ import annotations

import logging
import os
import re
from collections import Counter
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

log = logging.getLogger(__name__)

API = "https://api.github.com"
FAILED = frozenset({"failure", "timed_out", "startup_failure"})
BRANCH = "main"                      # Betrieb läuft nur auf main; Feature-Branch-/PR-Läufe (Tests vor dem Merge) sind kein
                                     # Betriebsfehler (Fehlalarm 2026-10-15: rote Tests eines Feature-Branch-Pushs)
SCANNER_FILE = ".github/workflows/scanner.yml"
SCAN_JOB = "scan"
LATE_START_ET = time(15, 0)          # danach bleibt bis zum Ende des Mark-Fensters (Schluss − 10 min) kaum Zeit
MAX_RUNS, MAX_LOG_JOBS, MAX_LINES, MAX_PAGES = 15, 6, 4, 3
CODEOWNERS = Path(".github/CODEOWNERS")
GOVERNANCE_WINDOW_DAYS, GOVERNANCE_MAX_PRS = 14, 15
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


# Recent green Actions runs are not evidence of execution: the scanner may have
# skipped its scan job, and optional research stages may have failed. Keep these
# outcomes separate from the existing hard-failure digest.
OUTCOME_WORKFLOWS = frozenset({
    ".github/workflows/scanner.yml", ".github/workflows/ml_research.yml",
    ".github/workflows/options_capture.yml", ".github/workflows/source_health.yml",
    ".github/workflows/feedback.yml", ".github/workflows/improvement_notify.yml",
})
MAX_RECENT_OUTCOMES = 6


def summarize_run_outcome(run: dict, jobs: list[dict]) -> dict:
    """Technical run status only; never asserts that trades, labels or quotes exist."""
    conclusion = str(run.get("conclusion") or "unknown")
    step_failures = [(j.get("name"), st.get("name")) for j in jobs
                     for st in (j.get("steps") or []) if st.get("conclusion") in FAILED]
    skipped_jobs = [str(j.get("name")) for j in jobs if j.get("conclusion") == "skipped"]
    if conclusion in FAILED:
        status = "FAILED"
    elif conclusion == "cancelled":
        status = "CANCELLED"
    elif not jobs:
        status = "UNVERIFIED"
    elif run.get("path") == SCANNER_FILE and any(j.get("name") == SCAN_JOB and
                                                 j.get("conclusion") == "skipped" for j in jobs):
        status = "SKIPPED_SCAN"
    elif step_failures:
        status = "DEGRADED_STEP_FAILURE"
    elif all(j.get("conclusion") == "skipped" for j in jobs):
        status = "SKIPPED_ALL_JOBS"
    elif conclusion == "success":
        status = "TECHNICALLY_SUCCESSFUL"
    else:
        status = "UNVERIFIED"
    return {"workflow": run.get("name"), "run_id": run.get("id"), "url": run.get("html_url"),
            "conclusion": conclusion, "effective_status": status,
            "skipped_jobs": skipped_jobs, "failed_steps": step_failures,
            "business_evidence": "NOT_EVALUATED"}


def _repo_from_jobs_url(url: str | None) -> str:
    """The REST job-list URL is emitted by GitHub; avoid guessing repository identity."""
    m = re.search(r"/repos/([^/]+/[^/]+)/actions/runs/\d+/jobs(?:[?].*)?$", url or "")
    if not m:
        raise ValueError("No GitHub repository in jobs_url")
    return m.group(1)


STAGE_LEDGER_PATHS = frozenset({".github/workflows/ml_research.yml", ".github/workflows/feedback.yml"})


def stage_failure_names(job_log: str) -> list[str]:
    """Extract machine-reported failures from the CI stage-balance, not generic log errors.

    GitHub marks `continue-on-error` steps as successful in the job API, although the
    stage-balance contains `commodity_evaluate=failure` or a matching warning.
    Only the authoritative CRITICAL/DEGRADED balance is interpreted as a failure.
    """
    failed = set()
    for line in (job_log or "").splitlines():
        if "  CRITICAL:" in line or "  DEGRADED:" in line:
            failed.update(re.findall(r"\b([a-z][a-z0-9_]+)=failure\b", line))
        if "##[warning]Stufe '" in line and "' ausgefallen" in line:
            m = re.search(r"Stufe '([^']+)' ausgefallen", line)
            if m:
                failed.add(m.group(1))
    return sorted(failed)


def recent_run_outcomes(session, runs: list[dict]) -> list[dict]:
    """One recent run per core workflow; individual job API failures do not mask other runs."""
    latest = {}
    for r in sorted(runs, key=lambda x: str(x.get("created_at") or ""), reverse=True):
        path = r.get("path")
        if path in OUTCOME_WORKFLOWS and path not in latest:
            latest[path] = r
        if len(latest) >= MAX_RECENT_OUTCOMES:
            break
    out = []
    for r in latest.values():
        try:
            jobs = (_get(session, r["jobs_url"]).json().get("jobs") or []) if r.get("jobs_url") else []
            record = summarize_run_outcome(r, jobs)
            # These jobs run `continue-on-error` stages: REST reports "success"
            # even when `ci_stage_check.sh` identified a failed substage.
            if r.get("path") in STAGE_LEDGER_PATHS and record["effective_status"] == "TECHNICALLY_SUCCESSFUL":
                try:
                    primary = next(j for j in jobs if j.get("name") in {"ml", "feedback"})
                    report = _get(session, f"{API}/repos/{_repo_from_jobs_url(r.get('jobs_url'))}/actions/jobs/{primary['id']}/logs").text
                    failures = stage_failure_names(report)
                    if failures:
                        record["effective_status"] = "DEGRADED_STEP_FAILURE"
                        record["failed_steps"] += [(primary["name"], name) for name in failures]
                    record["stage_balance"] = "FAILURE" if failures else "NO_FAILURE_REPORTED"
                except Exception as exc:  # noqa: BLE001
                    record["effective_status"] = "UNVERIFIED"
                    record["stage_balance"] = "UNAVAILABLE"
                    record["stage_error"] = type(exc).__name__
            out.append(record)
        except Exception as exc:  # noqa: BLE001 - missing job metadata must be visible, not silently green
            out.append({"workflow": r.get("name"), "run_id": r.get("id"),
                        "url": r.get("html_url"), "conclusion": r.get("conclusion"),
                        "effective_status": "UNVERIFIED", "error": type(exc).__name__,
                        "business_evidence": "NOT_EVALUATED"})
    return out


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
        runs: list[dict] = []
        for page in range(1, MAX_PAGES + 1):                  # Montags (74 h) können es > 100 Läufe sein
            chunk = _get(session, f"{API}/repos/{repo}/actions/runs",
                         {"created": f">={since}", "branch": BRANCH, "per_page": 100, "page": page}).json().get("workflow_runs") or []
            runs += chunk
            if len(chunk) < 100:
                break
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
                "scanner_start": _scanner_start(session, runs, now),
                "recent_outcomes": recent_run_outcomes(session, runs),
                "governance": governance(session, repo, now)}
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


def codeowner_patterns(text: str) -> list[str]:
    """Pfadmuster aus CODEOWNERS (erste Spalte, ohne Kommentare); führender Slash entfernt."""
    out = []
    for line in (text or "").splitlines():
        line = line.split("#", 1)[0].strip()
        if line and len(line.split()) >= 2:
            out.append(line.split()[0].lstrip("/"))
    return out


def owned(path: str, patterns: list[str]) -> bool:
    """Grobe CODEOWNERS-Zuordnung: Verzeichnis-Muster (endet auf /) als Präfix, sonst exakter Pfad, '*' als Joker."""
    import fnmatch
    for pat in patterns:
        if pat.endswith("/") and path.startswith(pat):
            return True
        if path == pat or ("*" in pat and fnmatch.fnmatch(path, pat)):
            return True
    return False


def governance(session, repo: str, now: datetime, patterns: list[str] | None = None) -> dict:
    """Branch-Schutz von main + gemergte PRs (letzte GOVERNANCE_WINDOW_DAYS Tage) mit CODEOWNERS-Pfaden ohne
    APPROVED-Review. Nur lesend; eigener Fehlerpfad (der Fehler-Digest bleibt unberührt)."""
    try:
        if patterns is None:
            patterns = codeowner_patterns(CODEOWNERS.read_text(encoding="utf-8")) if CODEOWNERS.exists() else []
        protected = bool(_get(session, f"{API}/repos/{repo}/branches/{BRANCH}").json().get("protected"))
        since = now - timedelta(days=GOVERNANCE_WINDOW_DAYS)
        pulls = _get(session, f"{API}/repos/{repo}/pulls", {"state": "closed", "base": BRANCH, "sort": "updated",
                                                            "direction": "desc", "per_page": 50}).json() or []
        merged = [p for p in pulls if p.get("merged_at")
                  and datetime.fromisoformat(p["merged_at"].replace("Z", "+00:00")) >= since][:GOVERNANCE_MAX_PRS]
        unreviewed = []
        for p in merged:
            n = p.get("number")
            files = [f.get("filename") for f in (_get(session, f"{API}/repos/{repo}/pulls/{n}/files",
                                                      {"per_page": 100}).json() or [])]
            hit = sorted(f for f in files if f and owned(f, patterns))
            if not hit:
                continue
            reviews = _get(session, f"{API}/repos/{repo}/pulls/{n}/reviews").json() or []
            if not any(r.get("state") == "APPROVED" for r in reviews):
                unreviewed.append({"number": n, "title": (p.get("title") or "")[:120], "merged_at": p.get("merged_at"),
                                   "url": p.get("html_url"), "codeowner_paths": hit[:10], "n_codeowner_paths": len(hit)})
        return {"status": "OK", "branch": BRANCH, "protected": protected, "window_days": GOVERNANCE_WINDOW_DAYS,
                "merged_checked": len(merged), "codeowner_patterns": len(patterns),
                "unreviewed_codeowner_merges": unreviewed}
    except Exception as e:  # noqa: BLE001 – Monitoring darf nie scheitern
        return {"status": "UNAVAILABLE", "reason": f"GitHub-API: {type(e).__name__}: {str(e)[:160]}"}

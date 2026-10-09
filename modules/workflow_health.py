"""modules/workflow_health.py – Freshness-Gate und Missed-Run-Watchdog für die kritischen Workflows
(Reliability-Maintenance 2026-10-09).

Vorher hing die Reihenfolge External Data -> Source Health -> Scanner allein an versetzten Cron-Zeiten,
obwohl GitHub-Crons hier 6–7 h verspätet starten und am 2026-10-05 Scanner + Source Health mangels Runner
ganz ausfielen. Jetzt:
  upstream_ready()  Scanner startet nur mit einem Source-Health-Snapshot aus dem aktuellen Tageszyklus
                    (sonst UPSTREAM_NOT_READY, nie still mit alten Daten), nie vor seiner Handelszeit
                    und nie zweimal am Tag (Tagesreport existiert -> SKIP_ALREADY_RAN).
  plan_recovery()   erkennt fehlende Läufe (Erwartung + max. Verzögerung + Tagesartefakt) und plant je
                    Workflow und Tag höchstens EINEN Recovery-Dispatch; solange ein Pflicht-Upstream
                    fehlt, wird der Downstream nicht angestoßen. Recovery ist idempotent, weil die Jobs
                    es sind (Guard: Tagesreport existiert -> Scanner läuft nicht erneut; Archiv dedupliziert).
Reine Funktionen (testbar); IO in scripts/upstream_guard.py und scripts/workflow_watchdog.py.
"""
from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import yaml

CONFIG = Path("config/workflow_schedule.yaml")
STATUS = Path("outputs/state/workflow_status.json")
ACTIVE_RUN_STATES = ("queued", "in_progress", "waiting", "requested", "pending")


def load_cfg(path: Path | None = None) -> dict:
    return yaml.safe_load(Path(path or CONFIG).read_text(encoding="utf-8"))


def _read(p: Path):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _ts(value) -> datetime | None:
    """ISO-Zeitstempel -> aware UTC; reines Datum -> 00:00 UTC; unlesbar -> None (nie geraten)."""
    if not value:
        return None
    s = str(value).replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _at(day: date, hhmm: str) -> datetime:
    h, m = (int(x) for x in str(hhmm).split(":"))
    return datetime.combine(day, time(h, m), tzinfo=timezone.utc)


def latest_timestamp(spec: dict, root: Path = Path(".")) -> datetime | None:
    """Zeitstempel eines Upstream-Artefakts (Feld `field`; any_source: jüngster Wert über alle Quellen)."""
    doc = _read(root / spec["path"])
    if not isinstance(doc, dict):
        return None
    field = spec.get("field") or "generated"
    if spec.get("any_source") or spec.get("kind") == "json_any_source_date":
        srcs = doc.get("sources", doc)
        vals = [_ts((v or {}).get(field)) for v in srcs.values() if isinstance(v, dict)] if isinstance(srcs, dict) else []
        vals = [v for v in vals if v is not None]
        return max(vals) if vals else None
    return _ts(doc.get(field))


def artifact_ok(spec: dict | None, day: date, root: Path = Path(".")) -> bool | None:
    """Ist der Tag `day` durch das Artefakt versorgt? None = kein Artefakt konfiguriert (Run-Status zählt)."""
    if not spec:
        return None
    if spec.get("kind") == "file":
        return (root / spec["path"].format(date=day.isoformat())).exists()
    ts = latest_timestamp(spec, root)
    return ts is not None and ts.date() == day


def upstream_ready(job: str, now: datetime, root: Path = Path("."), cfg: dict | None = None) -> dict:
    """-> {'run': bool, 'status': RUN|SKIP_NOT_TRADING_DAY|SKIP_BEFORE_WINDOW|SKIP_ALREADY_RAN|
    UPSTREAM_NOT_READY, 'reason': str, 'checks': [...], 'advisory': [...]}"""
    cfg = cfg or load_cfg()
    wf = (cfg.get("workflows") or {}).get(job) or {}
    gate = (cfg.get("gates") or {}).get(job) or {}
    today = now.date()

    def out(run, status, reason, checks=(), advisory=()):
        return {"job": job, "run": run, "status": status, "reason": reason, "checked_at": now.isoformat(timespec="seconds"),
                "checks": list(checks), "advisory": list(advisory)}

    if wf.get("days") is not None and today.weekday() not in wf["days"]:
        return out(False, "SKIP_NOT_TRADING_DAY", f"{today} ist kein geplanter Tag")
    if gate.get("not_before_utc") and now < _at(today, gate["not_before_utc"]):
        return out(False, "SKIP_BEFORE_WINDOW",
                   f"vor {gate['not_before_utc']} UTC – der geplante Lauf übernimmt (Handelszeit unverändert)")
    if artifact_ok(wf.get("artifact"), today, root):
        return out(False, "SKIP_ALREADY_RAN", f"Tag {today} bereits versorgt (idempotent)")

    def check(spec):
        ts = latest_timestamp(spec, root)
        age_h = None if ts is None else round((now - ts).total_seconds() / 3600, 1)
        if ts is None:
            st = "MISSING"
        elif ts > now + timedelta(minutes=5):
            st = "FUTURE_TIMESTAMP"              # unplausibel -> nicht als frisch behandeln
        elif age_h > float(spec.get("max_age_hours", 24)):
            st = "STALE"
        else:
            st = "FRESH"
        return {"name": spec.get("name") or spec["path"], "timestamp": ts.isoformat() if ts else None,
                "age_hours": age_h, "max_age_hours": spec.get("max_age_hours"), "status": st}

    checks = [check(s) for s in gate.get("requires") or []]
    advisory = [check(s) for s in gate.get("advisory") or []]
    bad = [c for c in checks if c["status"] != "FRESH"]
    if bad:
        why = "; ".join(f"{c['name']} {c['status']} (Stand {c['timestamp'] or 'nie'}, Alter {c['age_hours']} h, "
                        f"max {c['max_age_hours']} h)" for c in bad)
        return out(False, "UPSTREAM_NOT_READY", why, checks, advisory)
    return out(True, "RUN", "Pflicht-Upstream frisch", checks, advisory)


def plan_recovery(now: datetime, runs: dict[str, list[dict] | None], cfg: dict | None = None,
                  artifacts: dict[str, bool | None] | None = None,
                  recoveries: dict[str, int] | None = None) -> dict:
    """Plan für den heutigen UTC-Tag.

    runs:       {job: [{created_at, status, conclusion, event}]} heutiger Tag (GitHub-API); None = API-Fehler
    artifacts:  {job: True/False/None} Tag durch Artefakt versorgt (None = kein Artefakt konfiguriert)
    recoveries: {job: n} bereits ausgelöste Recovery-Dispatches heute (Status-Datei, gegen Doppel-Dispatch)
    -> {job: {expected, actual_last_run, delay_hours, recovered, state, action, stale_upstream}} mit state
       OK | PENDING | RUNNING | MISSED | RECOVERY_EXHAUSTED | NOT_SCHEDULED | UNKNOWN.
    action == "DISPATCH" genau dann, wenn verpasst, Recovery-Budget frei und Pflicht-Upstream versorgt."""
    cfg = cfg or load_cfg()
    artifacts = artifacts or {}
    recoveries = recoveries or {}
    max_rec = int(cfg.get("max_recoveries_per_day", 1))
    today = now.date()
    out = {}
    for job, wf in (cfg.get("workflows") or {}).items():
        exp = _at(today, wf["cron_utc"])
        api_ok = runs.get(job) is not None
        rs = sorted(runs.get(job) or [], key=lambda r: r.get("created_at") or "")
        n_rec = max(sum(1 for r in rs if r.get("event") == "workflow_dispatch"), int(recoveries.get(job, 0)))
        first = _ts(rs[0].get("created_at")) if rs else None
        row = {"file": wf["file"], "expected": exp.isoformat(timespec="minutes"),
               "actual_last_run": rs[-1].get("created_at") if rs else None,
               "conclusion": rs[-1].get("conclusion") if rs else None,
               "delay_hours": round((first - exp).total_seconds() / 3600, 1) if first else None,
               "runs_today": len(rs), "recovered": n_rec > 0, "recoveries_today": n_rec,
               "artifact_ok": artifacts.get(job), "action": None, "stale_upstream": []}
        art = artifacts.get(job)
        done = art if art is not None else any(r.get("conclusion") == "success" for r in rs)
        if today.weekday() not in wf.get("days", list(range(7))):
            row["state"] = "NOT_SCHEDULED"
        elif done:
            row["state"] = "OK"
        elif any(r.get("status") in ACTIVE_RUN_STATES for r in rs):
            row["state"] = "RUNNING"
        elif now < exp + timedelta(hours=float(wf.get("max_delay_hours", 8))):
            row["state"] = "PENDING"
        elif not api_ok:
            row["state"] = "UNKNOWN"                      # ohne Run-Liste nie blind dispatchen
        elif n_rec >= max_rec:
            row["state"] = "RECOVERY_EXHAUSTED"           # kein Endlos-Retry
        else:
            row["state"], row["action"] = "MISSED", "DISPATCH"
        out[job] = row
    # Pflicht-Upstream: Downstream nicht anstoßen, solange der Upstream selbst nicht versorgt ist
    for job, wf in (cfg.get("workflows") or {}).items():
        stale = [d for d in wf.get("depends_on") or [] if out.get(d, {}).get("state") not in ("OK", "NOT_SCHEDULED")]
        out[job]["stale_upstream"] = stale
        if stale and out[job]["action"] == "DISPATCH":
            out[job]["action"] = None
            out[job]["state"] = "MISSED"
    return out


def merge_status(prev: dict | None, today: date, plan: dict, dispatched: dict[str, int],
                 keep_days: int = 21, version: str = "wf-sched-v1") -> dict:
    """Fortschreibung von outputs/state/workflow_status.json (je Tag letzter Plan + Recovery-Zähler).
    Enthält bewusst keinen Prüfzeitpunkt: unveränderter Zustand -> unveränderte Datei -> kein Commit."""
    prev = prev if isinstance(prev, dict) else {}
    days = dict(prev.get("days") or {})
    rec = {d: dict(v) for d, v in (prev.get("recoveries") or {}).items()}
    key = today.isoformat()
    for job, n in (dispatched or {}).items():
        rec.setdefault(key, {})[job] = rec.get(key, {}).get(job, 0) + int(n)
    days[key] = plan
    cutoff = (today - timedelta(days=keep_days)).isoformat()
    days = {d: v for d, v in sorted(days.items()) if d >= cutoff}
    rec = {d: v for d, v in sorted(rec.items()) if d >= cutoff}
    return {"version": version, "days": days, "recoveries": rec}


def weekly_summary(status: dict | None, start: date, end: date) -> dict:
    """Kompakte Wochensicht je Workflow für den Montagsreport (Zeitraum inkl. start..end)."""
    days = {d: v for d, v in ((status or {}).get("days") or {}).items() if start.isoformat() <= d <= end.isoformat()}
    jobs: dict[str, dict] = {}
    for d, plan in sorted(days.items()):
        for job, row in (plan or {}).items():
            j = jobs.setdefault(job, {"expected_last": None, "actual_last": None, "delays": [], "recovered": 0,
                                      "missed": 0, "exhausted": 0, "stale_upstream_days": 0, "days": 0})
            if row.get("state") == "NOT_SCHEDULED":
                continue
            j["days"] += 1
            j["expected_last"] = row.get("expected")
            if row.get("actual_last_run"):
                j["actual_last"] = row["actual_last_run"]
            if row.get("delay_hours") is not None:
                j["delays"].append(row["delay_hours"])
            j["recovered"] += 1 if row.get("recovered") else 0
            j["missed"] += 1 if row.get("state") in ("MISSED", "RECOVERY_EXHAUSTED") else 0
            j["exhausted"] += 1 if row.get("state") == "RECOVERY_EXHAUSTED" else 0
            j["stale_upstream_days"] += 1 if row.get("stale_upstream") else 0
    for j in jobs.values():
        ds = sorted(j.pop("delays"))
        j["delay_hours_median"] = ds[len(ds) // 2] if ds else None
        j["delay_hours_max"] = ds[-1] if ds else None
    return {"days_covered": len(days), "jobs": jobs}

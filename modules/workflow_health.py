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
Recovery-Härtung 2026-10-19 (Scanner-Ausfall Mo 19.10.: Source-Health-Cron kam nicht, Watchdog-Frist 9 h
lag hinter Börsenschluss): spätester Scanner-Start je NY-Handelstag (`latest_start_before_close_minutes`,
danach MISSED_WINDOW_CLOSED statt Produktionsscan mit Kursen außerhalb der Sitzung) und Recovery-Frist,
die nie hinter diesem Start liegt (`deadline()`).
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
RECOVERY_LEAD = timedelta(hours=1)   # Recovery-Dispatch spätestens so lange vor dem spätesten Start (Puffer für
                                     # Upstream-Lauf + Runner-Wartezeit; Dispatch-Läufe starten ohne Cron-Verzug)


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


def _ny(now: datetime) -> datetime:
    from modules.market_snapshot import NY_TZ
    return (now if now.tzinfo else now.replace(tzinfo=timezone.utc)).astimezone(NY_TZ)


def _hhmm(v: str) -> time:
    h, m = (int(x) for x in str(v).split(":"))
    return time(h, m)


def _market_open_day(now: datetime) -> bool:
    """NYSE-Handelstag (New-York-Datum; Feiertage, Sonderschließungen) – einzige Kalenderquelle market_snapshot."""
    from modules.market_snapshot import is_trading_day
    return is_trading_day(_ny(now).date())


def _at(day: date, hhmm: str) -> datetime:
    h, m = (int(x) for x in str(hhmm).split(":"))
    return datetime.combine(day, time(h, m), tzinfo=timezone.utc)


def latest_start(gate: dict, ref: datetime) -> datetime | None:
    """Spätester sinnvoller Start (UTC) am NY-Handelstag von `ref`: Schluss der regulären Sitzung (verkürzte
    Sitzungen inklusive) minus `latest_start_before_close_minutes`. None = nicht konfiguriert (kein Limit)."""
    mins = gate.get("latest_start_before_close_minutes")
    if mins is None:
        return None
    from modules.market_snapshot import NY_TZ, market_close_time
    d = _ny(ref).date()
    close = datetime.combine(d, market_close_time(d), tzinfo=NY_TZ).astimezone(timezone.utc)
    return close - timedelta(minutes=int(mins))


def deadline(wf: dict, gate: dict, day: date) -> tuple[datetime, datetime | None]:
    """(Recovery-Frist, spätester Start). Frist = Cron + max_delay_hours, aber nie später als spätester Start
    minus RECOVERY_LEAD – sonst käme die Recovery erst, wenn kein gültiger Lauf mehr möglich ist."""
    exp = _at(day, wf["cron_utc"])
    due = exp + timedelta(hours=float(wf.get("max_delay_hours", 8)))
    cut = latest_start(gate, exp)
    return (min(due, cut - RECOVERY_LEAD) if cut is not None else due), cut


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
    """-> {'run': bool, 'status': RUN|SKIP_NOT_TRADING_DAY|SKIP_MARKET_CLOSED|SKIP_BEFORE_WINDOW|SKIP_ALREADY_RAN|
    MISSED_WINDOW_CLOSED|UPSTREAM_NOT_READY, 'reason': str, 'checks': [...], 'advisory': [...]}"""
    cfg = cfg or load_cfg()
    wf = (cfg.get("workflows") or {}).get(job) or {}
    gate = (cfg.get("gates") or {}).get(job) or {}
    today = now.date()

    def out(run, status, reason, checks=(), advisory=()):
        return {"job": job, "run": run, "status": status, "reason": reason, "checked_at": now.isoformat(timespec="seconds"),
                "checks": list(checks), "advisory": list(advisory)}

    if wf.get("days") is not None and today.weekday() not in wf["days"]:
        return out(False, "SKIP_NOT_TRADING_DAY", f"{today} ist kein geplanter Tag")
    if wf.get("trading_days_only") and not _market_open_day(now):
        return out(False, "SKIP_MARKET_CLOSED", f"{_ny(now).date()} ist kein NYSE-Handelstag "
                   "(Feiertag/Sonderschließung, config/market_calendar.yaml)")
    if gate.get("not_before_ny") and _ny(now).time() < _hhmm(gate["not_before_ny"]):
        return out(False, "SKIP_BEFORE_WINDOW",
                   f"vor {gate['not_before_ny']} New York ({_ny(now):%H:%M} ET) – ein späterer Trigger übernimmt "
                   "(Handelszeit in NY-Zeit, DST-fest)")
    if gate.get("not_before_utc") and now < _at(today, gate["not_before_utc"]):
        return out(False, "SKIP_BEFORE_WINDOW",
                   f"vor {gate['not_before_utc']} UTC – der geplante Lauf übernimmt (Handelszeit unverändert)")
    if artifact_ok(wf.get("artifact"), today, root):
        return out(False, "SKIP_ALREADY_RAN", f"Tag {today} bereits versorgt (idempotent)")
    cut = latest_start(gate, now)
    if cut is not None and now >= cut:
        return out(False, "MISSED_WINDOW_CLOSED",
                   f"MISSED: {_ny(now):%H:%M} ET liegt nach dem spätesten Start {_ny(cut):%H:%M} ET (Sitzungsschluss − "
                   f"{gate['latest_start_before_close_minutes']} min) – kein verspäteter Produktionsscan mit Kursen "
                   "außerhalb der regulären Sitzung; der Tag bleibt als verpasst dokumentiert")

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
    -> {job: {expected, deadline, actual_last_run, delay_hours, recovered, state, action, stale_upstream, reason}}
       mit state OK | PENDING | RUNNING | MISSED | RECOVERY_EXHAUSTED | NOT_SCHEDULED | UNKNOWN.
    action == "DISPATCH" genau dann, wenn verpasst, Recovery-Budget frei, Pflicht-Upstream versorgt und der
    späteste Start (Gate) noch nicht erreicht ist; danach MISSED mit window_closed (dokumentiert, kein Dispatch)."""
    cfg = cfg or load_cfg()
    artifacts = artifacts or {}
    recoveries = recoveries or {}
    max_rec = int(cfg.get("max_recoveries_per_day", 1))
    today = now.date()
    out = {}
    for job, wf in (cfg.get("workflows") or {}).items():
        exp = _at(today, wf["cron_utc"])
        due, cut = deadline(wf, (cfg.get("gates") or {}).get(job) or {}, today)
        api_ok = runs.get(job) is not None
        rs = sorted(runs.get(job) or [], key=lambda r: r.get("created_at") or "")
        n_rec = max(sum(1 for r in rs if r.get("event") == "workflow_dispatch"), int(recoveries.get(job, 0)))
        first = _ts(rs[0].get("created_at")) if rs else None
        row = {"file": wf["file"], "expected": exp.isoformat(timespec="minutes"),
               "deadline": due.isoformat(timespec="minutes"),
               "actual_last_run": rs[-1].get("created_at") if rs else None,
               "conclusion": rs[-1].get("conclusion") if rs else None,
               "delay_hours": round((first - exp).total_seconds() / 3600, 1) if first else None,
               "runs_today": len(rs), "recovered": n_rec > 0, "recoveries_today": n_rec,
               "artifact_ok": artifacts.get(job), "action": None, "stale_upstream": [], "reason": None}
        art = artifacts.get(job)
        done = art if art is not None else any(r.get("conclusion") == "success" for r in rs)
        if today.weekday() not in wf.get("days", list(range(7))):
            row["state"] = "NOT_SCHEDULED"
        elif wf.get("trading_days_only") and not _market_open_day(exp):
            row["state"], row["market_closed"] = "NOT_SCHEDULED", True   # Börsenfeiertag: kein Lauf erwartet
        elif done:
            row["state"] = "OK"
        elif any(r.get("status") in ACTIVE_RUN_STATES for r in rs):
            row["state"] = "RUNNING"
        elif now < due:
            row["state"] = "PENDING"
        elif not api_ok:
            row["state"], row["reason"] = "UNKNOWN", "Run-Liste nicht lesbar – kein Dispatch"   # nie blind dispatchen
        elif n_rec >= max_rec:
            row["state"] = "RECOVERY_EXHAUSTED"           # kein Endlos-Retry
            row["reason"] = f"{n_rec} Recovery-Versuch(e) heute ohne versorgten Tag – kein weiterer Retry"
        elif cut is not None and now >= cut:
            row["state"], row["window_closed"] = "MISSED", True
            row["reason"] = (f"spätester Start {cut.isoformat(timespec='minutes')} überschritten – kein verspäteter "
                             "Lauf außerhalb der regulären Sitzung")
        else:
            row["state"], row["action"] = "MISSED", "DISPATCH"
            row["reason"] = f"Frist {row['deadline']} überschritten, Tag nicht versorgt"
        out[job] = row
    # Pflicht-Upstream: Downstream nicht anstoßen, solange der Upstream selbst nicht versorgt ist
    for job, wf in (cfg.get("workflows") or {}).items():
        stale = [d for d in wf.get("depends_on") or [] if out.get(d, {}).get("state") not in ("OK", "NOT_SCHEDULED")]
        out[job]["stale_upstream"] = stale
        if stale and out[job]["action"] == "DISPATCH":
            out[job]["action"] = None
            out[job]["state"] = "MISSED"
            out[job]["reason"] = f"Pflicht-Upstream {', '.join(stale)} nicht versorgt – kein Dispatch"
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

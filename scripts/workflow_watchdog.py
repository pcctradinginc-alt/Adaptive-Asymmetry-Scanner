"""scripts/workflow_watchdog.py – Missed-Run-/Recovery-Watchdog für die kritischen Workflows (CI).

Je Workflow aus config/workflow_schedule.yaml: heutige Runs (GitHub-API) + Tagesartefakt ->
modules.workflow_health.plan_recovery(). Verpasst (Frist überschritten, Tag nicht versorgt, kein Run aktiv,
Pflicht-Upstream versorgt) -> genau EIN workflow_dispatch je Workflow und UTC-Tag. Doppelschutz gegen
Mehrfach-Dispatch: Zähler in outputs/state/workflow_status.json UND workflow_dispatch-Runs des Tages.
Die Jobs selbst sind idempotent (Scanner-Guard: Tagesreport existiert -> SKIP_ALREADY_RAN).
    python scripts/workflow_watchdog.py [--dry-run]
Ohne GITHUB_TOKEN/GITHUB_REPOSITORY: Run-Liste unbekannt -> Status UNKNOWN, nie blind dispatchen.
Jeder nicht versorgte Tag nach Ablauf der Frist (MISSED ohne Dispatch, RECOVERY_EXHAUSTED, UNKNOWN) erzeugt ein
::warning:: mit Grund; die Step-Summary zeigt je Workflow Zustand, Frist und Grund (kein stilles "recovered=False").
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules.atomic_io import atomic_write_json  # noqa: E402
from modules.workflow_health import STATUS, _read, artifact_ok, load_cfg, merge_status, plan_recovery  # noqa: E402

API = "https://api.github.com"


def _api(method: str, path: str, body: dict | None = None):
    req = urllib.request.Request(API + path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
                                          "Accept": "application/vnd.github+json",
                                          "X-GitHub-Api-Version": "2022-11-28"})
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read()
        return r.status, (json.loads(raw) if raw else None)


def list_runs(repo: str, file: str, day: str) -> list[dict] | None:
    try:
        _, data = _api("GET", f"/repos/{repo}/actions/workflows/{file}/runs?created=%3E%3D{day}&per_page=50")
    except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
        print(f"::warning::Run-Liste {file} nicht lesbar ({type(e).__name__}) -> UNKNOWN, kein Dispatch")
        return None
    return [{"created_at": r.get("created_at"), "status": r.get("status"), "conclusion": r.get("conclusion"),
             "event": r.get("event")} for r in (data or {}).get("workflow_runs") or []
            if str(r.get("created_at") or "")[:10] == day]


def dispatch(repo: str, file: str, ref: str) -> bool:
    try:
        st, _ = _api("POST", f"/repos/{repo}/actions/workflows/{file}/dispatches", {"ref": ref})
        return st == 204
    except (urllib.error.URLError, OSError) as e:
        print(f"::error::Recovery-Dispatch {file} fehlgeschlagen ({type(e).__name__}) – kein erneuter Versuch heute")
        return False


def _alerts(plan: dict) -> list[str]:
    """Verständliche Warnungen für Tage, die (vorerst) nicht versorgt werden – nie still."""
    out = []
    for job, row in plan.items():
        st = row.get("state")
        if st == "MISSED" and row.get("action") != "DISPATCH":
            out.append(f"MISSED {job}: {row.get('reason')}")
        elif st == "RECOVERY_EXHAUSTED":
            out.append(f"RECOVERY_EXHAUSTED {job}: {row.get('reason')} – manuell prüfen")
        elif st == "UNKNOWN":
            out.append(f"UNKNOWN {job}: {row.get('reason')} (Frist {row.get('deadline')} überschritten)")
    return out


def _summary(plan: dict, alerts: list[str], day: str) -> None:
    p = os.environ.get("GITHUB_STEP_SUMMARY")
    if not p:
        return
    lines = [f"### Workflow-Watchdog {day}: {'⚠️ ' + str(len(alerts)) + ' Problem(e)' if alerts else 'OK'}", "",
             "| Workflow | Zustand | Frist (UTC) | letzter Lauf | Recovery | Grund |", "|---|---|---|---|---|---|"]
    for job, row in plan.items():
        lines.append(f"| {job} | {row.get('state')} | {row.get('deadline')} | {row.get('actual_last_run') or '–'} | "
                     f"{row.get('recoveries_today', 0)} | {row.get('reason') or '–'} |")
    with open(p, "a", encoding="utf-8") as f:
        f.write("\n".join(lines + [""] + [f"- {m}" for m in alerts]) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_cfg()
    now = datetime.now(timezone.utc)
    day = now.date().isoformat()
    repo, token = os.environ.get("GITHUB_REPOSITORY"), os.environ.get("GITHUB_TOKEN")
    ref = os.environ.get("WATCHDOG_REF") or "main"
    wfs = cfg.get("workflows") or {}
    runs = {job: (list_runs(repo, wf["file"], day) if repo and token else None) for job, wf in wfs.items()}
    artifacts = {job: artifact_ok(wf.get("artifact"), now.date()) for job, wf in wfs.items()}
    prev = _read(STATUS)
    already = ((prev or {}).get("recoveries") or {}).get(day) or {}
    plan = plan_recovery(now, runs, cfg, artifacts=artifacts, recoveries=already)
    dispatched: dict[str, int] = {}
    for job, row in plan.items():
        if row["action"] != "DISPATCH":
            continue
        if a.dry_run:
            print(f"[dry-run] würde {row['file']} genau einmal dispatchen")
            continue
        # Zähler zählt den VERSUCH (auch fehlgeschlagen) -> nie ein zweiter Dispatch am selben Tag
        dispatched[job] = 1
        ok = dispatch(repo, row["file"], ref)
        row["state"] = "RECOVERY_TRIGGERED" if ok else "RECOVERY_EXHAUSTED"
        if not ok:
            row["reason"] = "Recovery-Dispatch fehlgeschlagen (GitHub API) – kein weiterer Versuch heute"
        row["recovered"], row["recoveries_today"] = ok, row["recoveries_today"] + 1
        print(f"::warning::MISSED {job} (erwartet {row['expected']}) -> Recovery-Dispatch {'OK' if ok else 'FEHLER'}")
    alerts = _alerts(plan)
    for msg in alerts:
        print(f"::warning::{msg}")
    _summary(plan, alerts, day)
    for job, row in plan.items():
        print(f"{job:14s} {row['state']:20s} erwartet {row['expected']}  letzter Lauf {row['actual_last_run']}  "
              f"Verzögerung {row['delay_hours']} h  recovered={row['recovered']}  stale_upstream={row['stale_upstream']}")
    if not a.dry_run:
        STATUS.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(STATUS, merge_status(prev, now.date(), plan, dispatched,
                                               keep_days=int(cfg.get("status_keep_days", 21)),
                                               version=cfg.get("version", "wf-sched-v1")), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())

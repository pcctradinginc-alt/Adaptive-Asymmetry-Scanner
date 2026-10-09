"""scripts/upstream_guard.py <job> – Freshness-Gate vor einem kritischen Downstream-Job (CI).

Schreibt `run=true|false` und `status=<STATUS>` nach $GITHUB_OUTPUT und eine Zeile in die Step-Summary.
UPSTREAM_NOT_READY ist nie still: ::warning:: mit Grund (Upstream-Stand, Alter, Grenze). Der Lauf bleibt
grün, weil ein verfrühter/verspäteter Cron kein Systemfehler ist – fehlt der Tag am Ende trotzdem,
meldet ihn der Watchdog (scripts/workflow_watchdog.py) als MISSED.
    python scripts/upstream_guard.py scanner [--force]
--force (manueller Dispatch mit Input force=true): Gate wird übersprungen, aber protokolliert.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules.workflow_health import upstream_ready  # noqa: E402


def _emit(key: str, value: str) -> None:
    p = os.environ.get("GITHUB_OUTPUT")
    if p:
        with open(p, "a", encoding="utf-8") as f:
            f.write(f"{key}={value}\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    res = upstream_ready(a.job, datetime.now(timezone.utc))
    res["event"] = os.environ.get("GITHUB_EVENT_NAME")
    if a.force and not res["run"]:
        print(f"::warning::{a.job}: Guard manuell übersteuert (force) – Status wäre {res['status']}: {res['reason']}")
        res = {**res, "run": True, "status": "FORCED", "forced_from": res["status"]}
    print(json.dumps(res, indent=2, ensure_ascii=False))
    if res["status"] == "UPSTREAM_NOT_READY":
        print(f"::warning::UPSTREAM_NOT_READY {a.job}: {res['reason']}")
    for c in res.get("advisory") or []:
        if c["status"] != "FRESH":
            print(f"::notice::{a.job} Hinweis (nicht blockierend): {c['name']} {c['status']} (Alter {c['age_hours']} h)")
    _emit("run", "true" if res["run"] else "false")
    _emit("status", res["status"])
    s = os.environ.get("GITHUB_STEP_SUMMARY")
    if s:
        with open(s, "a", encoding="utf-8") as f:
            f.write(f"### Upstream-Guard {a.job}: `{res['status']}`\n\n{res['reason']}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

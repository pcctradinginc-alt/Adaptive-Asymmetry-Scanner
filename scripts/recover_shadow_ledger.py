"""scripts/recover_shadow_ledger.py – einmalige, idempotente Rückführung verdrängter Schatten-Trades
in das Shadow-Ledger (Audit 2026-10-09).

Quellen (Originaldaten, unverändert):
  1. history.json shadow_trades (aktuelle Ansicht)
  2. outputs/shadow_trades_archive.jsonl (seit 2026-10-03 verdrängte Records)
  3. --git: alle früheren Stände von outputs/history.json in der Git-Historie (vor dem Archiv wurden
     verdrängte Records ohne Archiv verworfen)
Records ohne ticker/entry_date werden nicht geschätzt -> der Worker klassifiziert sie OUTCOME_UNAVAILABLE.
Bewertung fälliger Horizonte erfolgt im regulären Feedback-Lauf (Kursdaten).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from modules import shadow_ledger as sl
from modules.atomic_io import read_jsonl


def _git_shadows(path: str = "outputs/history.json", ref: str = "HEAD") -> list[dict]:
    commits = subprocess.run(["git", "log", "--format=%H", ref, "--", path],
                             capture_output=True, text=True, check=True).stdout.split()
    seen: dict[tuple, dict] = {}
    for c in commits:
        raw = subprocess.run(["git", "show", f"{c}:{path}"], capture_output=True, text=True).stdout
        try:
            js = json.loads(raw)
        except ValueError:
            continue
        for t in (js.get("shadow_trades") or []) if isinstance(js, dict) else []:
            k = sl.key_of(t)
            if k not in seen or (seen[k].get("outcome") is None and t.get("outcome") is not None):
                seen[k] = t
    return list(seen.values())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--git", action="store_true", help="auch Git-Historie von history.json durchsuchen")
    ap.add_argument("--history", default="outputs/history.json")
    ap.add_argument("--archive", default="outputs/shadow_trades_archive.jsonl")
    a = ap.parse_args(argv)
    now = datetime.now(timezone.utc)
    today = now.date()
    hist = json.loads(Path(a.history).read_text())
    out = {"view": sl.register(hist.get("shadow_trades") or [], "history_view", now)}
    if Path(a.archive).exists():
        out["archive"] = sl.recover(read_jsonl(a.archive), "recovery:shadow_trades_archive", today)
    if a.git:
        out["git"] = sl.recover(_git_shadows(a.history), "recovery:git_history", today)
    out["health"] = sl.health(today, archive_path=Path(a.archive), view=hist.get("shadow_trades") or [])
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

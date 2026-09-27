"""
scripts/trace_candidate.py – verfolgt Kandidaten aus dem Candidate Ledger
durch den gesamten Pfad und prüft PIT + Reproduzierbarkeit:

  externe Quelle (Snapshot: Quellen, latest_observation, available_at_max)
  -> normalisierte Beobachtung / abgeleitetes Feature (Neuaufbau zum
     Signalzeitpunkt aus dem HEUTIGEN Archiv, Vergleich mit eingefrorenen
     Werten: Abweichung = PIT-Leck oder Versionswechsel)
  -> external_context (eingefroren im Ledger)
  -> Kandidat / Ledger-Status / Richtung
  -> Shadow- oder Real-Trade (history.json) -> reifes Outcome
  -> Feedback (feature_stats_external)

Aufruf: python scripts/trace_candidate.py [--date YYYY-MM-DD] [--ticker T] [--limit 3]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LEDGER_DIR = Path("outputs/candidate_ledger")
ARCHIVE_ROOT = Path("outputs/external_data")


def _rows(date=None, ticker=None):
    for f in sorted(LEDGER_DIR.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not r.get("external"):
                continue
            if date and str(r.get("date"))[:10] != date:
                continue
            if ticker and r.get("ticker") != ticker:
                continue
            yield r


def _snapshot(snapshot_id):
    hits = list((ARCHIVE_ROOT / "snapshots").glob(f"*/{snapshot_id}.json"))
    return json.loads(hits[0].read_text()) if hits else None


def _close(a, b, tol=1e-9):
    if a is None or b is None:
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=tol)
    return a == b


def trace(r: dict) -> dict:
    from modules.external.archive import ExternalArchive
    from modules.external.context import build_external_context
    ext = r["external"]
    ts_raw = r.get("signal_timestamp") or ext.get("available_at")
    ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00")) if ts_raw else None
    snap = _snapshot(ext.get("snapshot_id")) if ext.get("snapshot_id") else None
    out = {"ticker": r.get("ticker"), "date": r.get("date"), "signal_timestamp": ts_raw,
           "status": r.get("status"), "reject_reason": r.get("reject_reason"),
           "direction": r.get("direction"), "snapshot_id": ext.get("snapshot_id"),
           "feature_version": ext.get("feature_version"), "policy": ext.get("policy")}

    # 1) Quellen + PIT: available_at_max aller genutzten Quellen <= Signalzeit
    if snap and ts:
        srcs = {}
        for sid, info in (snap.get("sources") or {}).items():
            am = info.get("available_at_max")
            srcs[sid] = {"latest_observation": info.get("latest_observation"), "available_at_max": am,
                         "pit_ok": (am is None) or datetime.fromisoformat(am) <= ts.astimezone(timezone.utc)}
        out["sources"] = srcs
        out["pit_violations"] = [s for s, v in srcs.items() if not v["pit_ok"]]

    # 2) Reproduzierbarkeit: Neuaufbau zum Signalzeitpunkt aus dem heutigen Archiv
    if ts:
        rebuilt = build_external_context(ts, archive=ExternalArchive(root=str(ARCHIVE_ROOT)))
        frozen_p, new_p = ext.get("primitives") or {}, rebuilt.get("primitives") or {}
        diffs = {k: {"frozen": frozen_p.get(k), "rebuilt": new_p.get(k)}
                 for k in sorted(set(frozen_p) | set(new_p)) if not _close(frozen_p.get(k), new_p.get(k))}
        out["reproducible"] = not diffs
        out["reproduction_diffs"] = diffs
        out["feature_version_rebuilt"] = rebuilt.get("feature_versions")

    # 3) external_context (eingefroren)
    out["states"] = ext.get("states")
    out["ticker_exposure"] = ext.get("ticker_exposure")
    out["relation"] = ext.get("relation")

    # 4) Outcome + Feedback
    out["outcomes"] = r.get("outcomes")
    try:
        h = json.loads(Path("outputs/history.json").read_text())
        for kind in ("closed_trades", "active_trades", "shadow_trades"):
            for t in h.get(kind, []):
                if t.get("ticker") == r.get("ticker") and str(t.get("entry_date", ""))[:10] == str(r.get("date"))[:10]:
                    out["trade"] = {"kind": kind, "outcome": t.get("outcome"),
                                    "has_external_context_entry": bool(t.get("external_context_entry")),
                                    "entry_context_matches_ledger":
                                        (t.get("external_context_entry") or {}).get("snapshot_id") == ext.get("snapshot_id")}
        out["feature_stats_external_buckets"] = len(h.get("feature_stats_external") or {})
    except Exception as e:  # noqa: BLE001
        out["trade_error"] = repr(e)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date")
    ap.add_argument("--ticker")
    ap.add_argument("--limit", type=int, default=3)
    a = ap.parse_args()
    rows = list(_rows(a.date, a.ticker))[: a.limit]
    if not rows:
        print("Keine Ledger-Zeilen mit externem Kontext gefunden (Ledger startet 2026-09-28).")
        return 0
    for r in rows:
        print(json.dumps(trace(r), indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

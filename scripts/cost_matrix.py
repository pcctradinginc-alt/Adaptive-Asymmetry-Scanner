"""scripts/cost_matrix.py – forensische API-Kostenmatrix + Monatsprojektion aus dem Kosten-Ledger.

    python scripts/cost_matrix.py [--out outputs/costs/cost_matrix.json]

Quelle: outputs/costs/ledger-*.jsonl (gemessene usage x Listenpreis; Daten-APIs: Request-Zähler je Lauf).
Schreibt JSON (Matrix sortiert nach Kostenanteil, Top-Treiber, Projektion, Request-Cache-Klassen).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules import cost_telemetry as ct  # noqa: E402
from modules.atomic_io import atomic_write_json  # noqa: E402


def build(today: date | None = None) -> dict:
    today = today or datetime.now(timezone.utc).date()
    rows = ct.load_ledger()
    m = ct.cost_matrix(rows)
    proj = ct.projection(rows, today)
    full_month = (round(m["cost_per_run_usd"] * 22, 2) if m["cost_per_run_usd"] else None)
    return {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "as_of": today.isoformat(),
            "matrix": m, "projection": proj, "full_month_estimate_usd_22_scanner_days": full_month,
            "monthly_api_budget_usd": (ct.policy().get("api_budget") or {}).get("monthly_api_budget_usd")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/costs/cost_matrix.json")
    a = ap.parse_args(argv)
    res = build()
    atomic_write_json(Path(a.out), res, indent=2, default=str)
    p = res["projection"]
    print(f"Kosten je Scanner-Lauf ${res['matrix']['cost_per_run_usd']:.2f} | MTD ${p['cost_month_to_date_usd']:.2f} | "
          f"Projected monthly API cost ${p['projected_month_usd']} (Ziel ≤ ${res['monthly_api_budget_usd']})")
    for r in res["matrix"]["rows"]:
        print(f"  {r['provider']:13s} {r['consumer']:18s} {str(r.get('model') or ''):28s} req/Lauf {r['requests_per_run']:>7} "
              f"$/Lauf {r.get('cost_per_run_usd')}  Anteil {r.get('share')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

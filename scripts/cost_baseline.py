"""
scripts/cost_baseline.py – SCHÄTZUNG der LLM-Kosten-Baseline vor Einführung der Telemetrie.

Quelle: vorhandene Tagesreports (outputs/daily_reports/*.json, Funnel-Zähler) x Token-Annahmen
(unten, aus Prompt-Längen abgeleitet) x offizielle Preise (config/cost_policy.yaml).
Das ist KEINE Messung: sobald outputs/costs/ledger-*.jsonl einen vollen Monat abdeckt,
ersetzt die gemessene Telemetrie (modules/cost_telemetry.month_summary) diese Schätzung.

    python scripts/cost_baseline.py            # Tabelle je Monat
    python scripts/cost_baseline.py --json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules import cost_telemetry as ct  # noqa: E402

# Token-Annahmen je Call (Zeichen/3 der realen Prompt-Vorlagen + typische Füllung, Stand 2026-10-03)
ASSUME = {
    "deep_analysis": {"model": "claude-sonnet-4-6", "input_tokens": 2000, "output_tokens": 900},
    "prescreening": {"model": "claude-haiku-4-5-20251001", "input_tokens": 2700, "output_tokens": 800},  # je 20er-Batch
    "shadow_relation": {"model": "claude-haiku-4-5-20251001", "input_tokens": 900, "output_tokens": 250},
}
BATCH = 20
SHADOW_MAX = 25


def call_cost(stage: str) -> float:
    a = ASSUME[stage]
    return ct.compute_cost(a["model"], {"input_tokens": a["input_tokens"], "output_tokens": a["output_tokens"]}) or 0.0


def estimate(reports_dir: str = "outputs/daily_reports") -> dict:
    months: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    for f in sorted(glob.glob(f"{reports_dir}/*.json")):
        try:
            d = json.load(open(f))
        except Exception as e:  # noqa: BLE001
            print(f"übersprungen: {f} ({e})", file=sys.stderr)
            continue
        s = d.get("stats") or {}
        rej = d.get("rejects") or {}
        key = Path(f).stem[:7]
        m = months[key]
        m["scan_runs"] += 1
        cands = s.get("candidates") or 0
        sector_ok = s.get("sector_ok") or cands
        m["prescreen_calls"] += math.ceil(min(cands, sector_ok) / BATCH) if cands else 0
        vetoes = ((rej.get("deep_analysis_veto_or_invalid") or {}).get("count") or 0)
        sonnet = (s.get("analyzed") or 0) + vetoes
        m["sonnet_calls"] += sonnet
        m["shadow_calls"] += min(SHADOW_MAX, s.get("analyzed") or 0)
        m["final_trades"] += s.get("trades") or 0
    out = {}
    for k, m in sorted(months.items()):
        cost = {"deep_analysis": m["sonnet_calls"] * call_cost("deep_analysis"),
                "prescreening": m["prescreen_calls"] * call_cost("prescreening"),
                "shadow_relation": m["shadow_calls"] * call_cost("shadow_relation")}
        total = sum(cost.values())
        out[k] = {**{x: int(v) for x, v in m.items()}, "cost_usd": {x: round(v, 2) for x, v in cost.items()},
                  "total_usd": round(total, 2),
                  "cost_per_scan": round(total / m["scan_runs"], 3) if m["scan_runs"] else None,
                  "cost_per_final_trade": round(total / m["final_trades"], 2) if m["final_trades"] else None}
    return {"kind": "ESTIMATE", "assumptions": ASSUME, "months": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    res = estimate()
    if a.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return
    print("| Monat | Läufe | Sonnet | Haiku-Prescreen | Haiku-Shadow | Sonnet $ | Haiku $ | Gesamt $ | $/Lauf | Trades |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for k, m in res["months"].items():
        c = m["cost_usd"]
        print(f"| {k} | {m['scan_runs']} | {m['sonnet_calls']} | {m['prescreen_calls']} | {m['shadow_calls']} | "
              f"{c['deep_analysis']:.2f} | {c['prescreening'] + c['shadow_relation']:.2f} | {m['total_usd']:.2f} | "
              f"{m['cost_per_scan']} | {m['final_trades']} |")


if __name__ == "__main__":
    main()

"""Ursachenanalyse der echten Paper-Performance des täglichen Scanners
(Audit P1-9). Nur gespeicherte Paper-Trades aus outputs/history.json, kein
Backtest. Nur zuverlässige Outcomes (ohne delta_approx-Rekonstruktion).

    python scripts/paper_performance_analysis.py
-> outputs/research/paper_performance_analysis.{json,md}
"""
from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

HIST = Path("outputs/history.json")
OUT = Path("outputs/research")


def summary(trades: list[dict]) -> dict | None:
    o = [t["outcome"] for t in trades]
    if not o:
        return None
    w, l = [x for x in o if x > 0], [x for x in o if x <= 0]
    pf = sum(w) / -sum(l) if l and sum(l) < 0 else None
    return {"n": len(o), "win_rate": round(len(w) / len(o), 3), "mean": round(st.mean(o), 4),
            "median": round(st.median(o), 4), "profit_factor": round(pf, 2) if pf is not None else None}


def group(trades: list[dict], key) -> dict:
    g: dict = defaultdict(list)
    for t in trades:
        g[str(key(t))].append(t)
    return {k: summary(v) for k, v in sorted(g.items())}


def mc_bucket(t: dict) -> str | None:
    hr = (t.get("simulation") or {}).get("hit_rate")
    if hr is None:
        return None
    return "<0.55" if hr < 0.55 else "0.55-0.65" if hr < 0.65 else "0.65-0.75" if hr < 0.75 else ">=0.75"


def analyse(hist: dict) -> dict:
    closed = [t for t in hist.get("closed_trades") or [] if isinstance(t.get("outcome"), (int, float))]
    rel = [t for t in closed if t.get("outcome_method_reconstructed") != "delta_approx"]
    mc = defaultdict(list)
    for t in rel:
        b = mc_bucket(t)
        if b:
            mc[b].append(t)
    mc_cal = {b: {**summary(v), "predicted_hit_rate": round(st.mean(t["simulation"]["hit_rate"] for t in v), 3)}
              for b, v in sorted(mc.items())}
    return {"n_closed": len(closed), "n_reliable": len(rel), "overall": summary(rel),
            "by_strategy": group(rel, lambda t: t.get("strategy")),
            "by_entry_month": group(rel, lambda t: t.get("entry_date", "")[:7]),
            "by_close_reason": group(rel, lambda t: t.get("close_reason") or "offen_bis_Bewertung"),
            "by_catalyst": group(rel, lambda t: t.get("catalyst_type") or "keiner"),
            "by_llm_impact": group(rel, lambda t: (t.get("features") or {}).get("impact")),
            "by_llm_surprise": group(rel, lambda t: (t.get("features") or {}).get("surprise")),
            "mc_hit_rate_calibration": mc_cal}


def render(r: dict) -> str:
    L = ["# Paper-Performance des täglichen Scanners – Ursachenanalyse", "",
         f"Quelle: outputs/history.json, nur zuverlässige Outcomes: n={r['n_reliable']} von {r['n_closed']}.", "",
         f"Gesamt: {r['overall']}", ""]
    for k in ("mc_hit_rate_calibration", "by_strategy", "by_entry_month", "by_close_reason", "by_catalyst",
              "by_llm_impact", "by_llm_surprise"):
        L += [f"## {k}", "", "| Gruppe | n | Trefferquote | Ø | Median | PF | vorhergesagt |", "|---|---|---|---|---|---|---|"]
        for g, s in r[k].items():
            if s:
                L.append(f"| {g} | {s['n']} | {s['win_rate']} | {s['mean']} | {s['median']} | {s['profit_factor']} | "
                         f"{s.get('predicted_hit_rate', '')} |")
        L.append("")
    return "\n".join(L)


def main() -> int:
    r = analyse(json.loads(HIST.read_text()))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "paper_performance_analysis.json").write_text(json.dumps(r, indent=1, ensure_ascii=False))
    (OUT / "paper_performance_analysis.md").write_text(render(r), encoding="utf-8")
    print(render(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

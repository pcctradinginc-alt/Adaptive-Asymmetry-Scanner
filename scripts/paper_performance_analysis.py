"""Ursachenanalyse der echten Paper-Performance des täglichen Scanners
(Audit P1-9). Nur gespeicherte Paper-Trades aus outputs/history.json, kein
Backtest. Nur RELIABLE-Outcomes (modules/outcomes.py; ohne UNKNOWN/RECONSTRUCTED/APPROXIMATED).

    python scripts/paper_performance_analysis.py
-> outputs/research/paper_performance_analysis.{json,md}
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))      # Aufruf als Skript (CI)

HIST = Path("outputs/history.json")
OUT = Path("outputs/research")
CAL_MIN_N = 10          # Band-Kalibrierung erst ab n Trades im Band (wie reports/weekly.CAL_MIN_N)


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


def _ece(pairs: list[tuple[float, float]], bins: int = 5) -> float | None:
    if not pairs:
        return None
    err = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [(p, y) for p, y in pairs if lo <= p < hi or (b == bins - 1 and p == 1.0)]
        if sel:
            err += len(sel) / len(pairs) * abs(st.mean(p for p, _ in sel) - st.mean(y for _, y in sel))
    return round(err, 4)


def calibration_oos(rel: list[dict]) -> dict:
    """Prequentielle (cross-fitted) Bewertung: für jeden Trade werden die Bänder NUR aus Trades
    geschätzt, die VOR seinem Entry geschlossen waren – Fit und Bewertung nie auf denselben
    Beobachtungen. Vergleicht rohe MC-Quote mit der kalibrierten Band-Quote (Brier, ECE)."""
    ts = sorted((t for t in rel if mc_bucket(t) and t.get("entry_date") and t.get("close_date")),
                key=lambda t: t["entry_date"])
    raw, cal, skipped = [], [], 0
    for t in ts:
        past = [u for u in ts if str(u["close_date"])[:10] < str(t["entry_date"])[:10] and mc_bucket(u) == mc_bucket(t)]
        if len(past) < CAL_MIN_N:
            skipped += 1
            continue
        y = 1.0 if t["outcome"] > 0 else 0.0
        raw.append((float(t["simulation"]["hit_rate"]), y))
        cal.append((sum(1 for u in past if u["outcome"] > 0) / len(past), y))

    def brier(pairs):
        return round(st.mean((p - y) ** 2 for p, y in pairs), 4) if pairs else None
    return {"method": "prequential: Band-Win-Rate nur aus vor dem Entry geschlossenen Trades (min n "
                      f"{CAL_MIN_N} je Band)", "n_evaluated": len(raw), "n_skipped_insufficient_history": skipped,
            "brier_raw": brier(raw), "brier_calibrated": brier(cal), "ece_raw": _ece(raw), "ece_calibrated": _ece(cal),
            "calibrated_better": (brier(cal) < brier(raw)) if raw else None}


def analyse(hist: dict) -> dict:
    closed = [t for t in hist.get("closed_trades") or [] if isinstance(t.get("outcome"), (int, float))]
    from modules.outcomes import is_reliable_outcome, reliability_stamp   # eine Definition für alle Auswertungen
    rel = [t for t in closed if is_reliable_outcome(t)]
    mc = defaultdict(list)
    for t in rel:
        b = mc_bucket(t)
        if b:
            mc[b].append(t)
    mc_cal = {b: {**summary(v), "predicted_hit_rate": round(st.mean(t["simulation"]["hit_rate"] for t in v), 3)}
              for b, v in sorted(mc.items())}
    return {"n_closed": len(closed), "n_reliable": len(rel), **reliability_stamp(closed), "overall": summary(rel),
            "by_strategy": group(rel, lambda t: t.get("strategy")),
            "by_entry_month": group(rel, lambda t: t.get("entry_date", "")[:7]),
            "by_close_reason": group(rel, lambda t: t.get("close_reason") or "offen_bis_Bewertung"),
            "by_catalyst": group(rel, lambda t: t.get("catalyst_type") or "keiner"),
            "by_llm_impact": group(rel, lambda t: (t.get("features") or {}).get("impact")),
            "by_llm_surprise": group(rel, lambda t: (t.get("features") or {}).get("surprise")),
            "mc_hit_rate_calibration": mc_cal,
            "mc_hit_rate_calibration_scope": "in-sample über alle RELIABLE-Trades (deskriptiv); für neue "
                                             "Kandidaten nur aus der Vergangenheit -> gültig; LIVE_FORWARD_CALIBRATION siehe calibration_oos",
            "calibration_oos": calibration_oos(rel)}


def render(r: dict) -> str:
    L = ["# Paper-Performance des täglichen Scanners – Ursachenanalyse", "",
         f"Quelle: outputs/history.json, nur RELIABLE-Outcomes: n={r['n_reliable']} von {r['n_closed']} (Klassen {r.get('outcome_classes')}).", "",
         f"Gesamt: {r['overall']}", "", f"Kalibrierung out-of-sample (prequential): {r.get('calibration_oos')}", ""]
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

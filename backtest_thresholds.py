"""
backtest_thresholds.py – Schwellen-Validierung auf historischen Trades

Replay über closed_trades (+ bewertete shadow_trades) aus history.json:
Für jede Gate-Schwelle wird gezeigt, wie Win-Rate / Ø-Return / Trade-Anzahl
sich verändern, wenn die Schwelle anders gesetzt wäre. Kein API-Call nötig.

Nutzung:
    python backtest_thresholds.py
"""

import json
import statistics
import sys
from pathlib import Path

HISTORY_PATH = Path("outputs/history.json")


def trade_rows(history: dict) -> list[dict]:
    """closed_trades + bewertete shadow_trades als flache Zeilen."""
    rows = []
    for src, trades in [("real", history.get("closed_trades", [])),
                        ("shadow", history.get("shadow_trades", []))]:
        for t in trades:
            o = t.get("outcome")
            if o is None:
                continue
            sim = t.get("simulation") or {}
            da  = t.get("deep_analysis") or {}
            opt = t.get("option") or {}
            rows.append({
                "source":   src,
                "ticker":   t.get("ticker"),
                "outcome":  float(o),
                "mismatch": (t.get("features") or {}).get("mismatch"),
                "impact":   da.get("impact"),
                "surprise": da.get("surprise"),
                "hit_rate": sim.get("hit_rate"),
                "dte":      opt.get("dte"),
                "score":    t.get("trade_score"),
                "strategy": t.get("strategy", ""),
            })
    return rows


def summarize(rows: list[dict]) -> str:
    if not rows:
        return "n=0"
    outs = [r["outcome"] for r in rows]
    wins = sum(1 for o in outs if o > 0)
    tl   = sum(1 for o in outs if o <= -0.99)
    return (
        f"n={len(outs):3d}  win={wins/len(outs):5.0%}  "
        f"mean={statistics.mean(outs):+7.2%}  "
        f"median={statistics.median(outs):+7.2%}  totalloss={tl}"
    )


def sweep(rows: list[dict], label: str, field: str,
          thresholds: list[float], mode: str = "min") -> None:
    """Zeigt Statistik wenn nur Trades mit field >=/<= threshold genommen würden."""
    print(f"\n── {label} ({'≥' if mode == 'min' else '≤'}) " + "─" * 30)
    valid = [r for r in rows if isinstance(r.get(field), (int, float))]
    if not valid:
        print(f"  (keine Daten für '{field}')")
        return
    print(f"  alle              {summarize(valid)}")
    for th in thresholds:
        if mode == "min":
            sel = [r for r in valid if r[field] >= th]
        else:
            sel = [r for r in valid if r[field] <= th]
        print(f"  {th:<17} {summarize(sel)}")


def main() -> None:
    if not HISTORY_PATH.exists():
        print("outputs/history.json nicht gefunden.")
        sys.exit(1)
    history = json.loads(HISTORY_PATH.read_text())
    rows    = trade_rows(history)
    real    = [r for r in rows if r["source"] == "real"]
    shadow  = [r for r in rows if r["source"] == "shadow"]

    print("=" * 70)
    print("Schwellen-Backtest auf historischen Outcomes")
    print(f"  Echte Trades:     {summarize(real)}")
    if shadow:
        print(f"  Schatten-Trades:  {summarize(shadow)}")
    print("=" * 70)

    sweep(rows, "DTE-Floor", "dte", [30, 45, 60, 90, 120], mode="min")
    sweep(rows, "Mismatch-Cap", "mismatch", [5, 6, 7, 8], mode="max")
    sweep(rows, "Impact-Floor", "impact", [4, 5, 6, 7], mode="min")
    sweep(rows, "Surprise-Floor", "surprise", [3, 4, 5, 6], mode="min")
    sweep(rows, "MC-Hit-Rate-Floor", "hit_rate", [0.45, 0.50, 0.55, 0.60], mode="min")
    sweep(rows, "Trade-Score-Floor", "score", [40, 50, 55, 60, 70], mode="min")

    print(
        "\nHinweis: Kleine n → Zufall dominiert. Schwellen erst ändern, wenn"
        "\nein Muster über ≥20 Trades stabil ist (Schatten-Trades zählen mit)."
    )


if __name__ == "__main__":
    main()

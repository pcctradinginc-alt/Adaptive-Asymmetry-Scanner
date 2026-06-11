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
                "catalyst": t.get("catalyst_type"),
            })
    return rows


def catalyst_breakdown(rows: list[dict]) -> None:
    """Win-Rate pro Katalysator-Typ — welcher Event-Typ liefert die Gewinner?"""
    print("\n── Katalysator-Typ " + "─" * 35)
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r.get("catalyst") or "unbekannt", []).append(r)
    if set(groups) == {"unbekannt"}:
        print("  (catalyst_type wird erst seit Juni 2026 getrackt — Daten folgen)")
        return
    for cat, g in sorted(groups.items(), key=lambda x: -len(x[1])):
        print(f"  {cat:<17} {summarize(g)}")


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


# ── Tuning-Vorschläge (für monthly_report.py) ────────────────────────────────

# Tunable Gates: (Label, Feld, Modus, Kandidaten, aktueller Wert via cfg/Code)
TUNABLES = [
    ("DTE-Floor",         "dte",      "min", [30, 45, 60, 90, 120]),
    ("Mismatch-Cap",      "mismatch", "max", [4, 5, 6, 7, 8]),
    ("Impact-Floor",      "impact",   "min", [4, 5, 6]),
    ("Surprise-Floor",    "surprise", "min", [3, 4, 5]),
    ("Trade-Score-Floor", "score",    "min", [40, 50, 55, 60, 70]),
]

MIN_N_FOR_SUGGESTION = 20     # Guardrail: nie auf dünner Datenbasis vorschlagen
MIN_WINRATE_GAIN_PP  = 5.0    # Mindest-Verbesserung in Prozentpunkten


def _select(rows: list[dict], field: str, th: float, mode: str) -> list[dict]:
    if mode == "min":
        return [r for r in rows if r[field] >= th]
    return [r for r in rows if r[field] <= th]


def _stats(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    outs = [r["outcome"] for r in rows]
    wins = sum(1 for o in outs if o > 0)
    return {"n": len(outs), "win_rate": wins / len(outs),
            "mean": statistics.mean(outs)}


def suggest_thresholds(history: dict, current: dict) -> list[dict]:
    """
    Vergleicht für jedes tunable Gate den aktuellen Schwellwert mit
    Alternativen über echte + Schatten-Trade-Outcomes.

    Guardrails (kein Overfitting auf Kleinst-Stichproben):
      - Alternative braucht n ≥ MIN_N_FOR_SUGGESTION Trades
      - Win-Rate-Verbesserung ≥ MIN_WINRATE_GAIN_PP Prozentpunkte
      - Ø-Return darf sich nicht verschlechtern

    `current`: {"dte": 45, "mismatch": 7, "impact": 4, "surprise": 3, "score": 55}
    Returns Liste von Vorschlägen (kann leer sein) — es wird NICHTS
    automatisch geändert, nur empfohlen.
    """
    rows = trade_rows(history)
    suggestions = []
    for label, field, mode, candidates in TUNABLES:
        cur_th = current.get(field)
        if cur_th is None:
            continue
        valid = [r for r in rows if isinstance(r.get(field), (int, float))]
        cur_stats = _stats(_select(valid, field, cur_th, mode))
        if cur_stats is None:
            continue
        best = None
        for th in candidates:
            if th == cur_th:
                continue
            s = _stats(_select(valid, field, th, mode))
            if s is None or s["n"] < MIN_N_FOR_SUGGESTION:
                continue
            gain_pp = (s["win_rate"] - cur_stats["win_rate"]) * 100
            if gain_pp < MIN_WINRATE_GAIN_PP or s["mean"] < cur_stats["mean"]:
                continue
            if best is None or s["win_rate"] > best["stats"]["win_rate"]:
                best = {"threshold": th, "stats": s, "gain_pp": gain_pp}
        if best:
            suggestions.append({
                "gate":        label,
                "field":       field,
                "mode":        mode,
                "current":     cur_th,
                "suggested":   best["threshold"],
                "gain_pp":     round(best["gain_pp"], 1),
                "current_stats":   cur_stats,
                "suggested_stats": best["stats"],
            })
    return suggestions


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
    catalyst_breakdown(rows)

    print(
        "\nHinweis: Kleine n → Zufall dominiert. Schwellen erst ändern, wenn"
        "\nein Muster über ≥20 Trades stabil ist (Schatten-Trades zählen mit)."
    )


if __name__ == "__main__":
    main()

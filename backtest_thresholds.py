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
                # v7.2: Short-Interest-Features (nur bei Trades ab 2026-07 vorhanden)
                "short_pct_float": (t.get("features") or {}).get("short_pct_float"),
                "short_mom":       (t.get("features") or {}).get("short_mom"),
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


# ── Slot-Analyse (Kandidaten für den nächsten Tuning-Slot) ──────────────────
#
# Nur EINE Schwellen-Änderung pro Monat ist erlaubt ("Tuning-Slot") — Effekte
# sollen sich sonst nicht mehr sauber einer Ursache zuordnen lassen. Parallel
# laufen mehrere Schatten-Messungen (Trailing-Stop-Sim, Spread- vs.
# Long-Call-Vergleich, Sektor-Konzentration), die als Entscheidungsvorlage
# für den nächsten Slot dienen sollen. Diese Funktion fasst sie zusammen —
# sie ändert NICHTS, sie schlägt nur vor (der User entscheidet).

SLOT_MIN_N_TRAILING     = 5     # Guardrail: Trailing-Vergleich erst ab n≥5 "ready"
SLOT_MIN_DELTA_PP       = 5.0   # Mindest-Delta in Prozentpunkten für Empfehlung
SLOT_MIN_N_STRATEGY     = 10    # Guardrail: Spread-vs.-Long erst ab n≥10 je Gruppe
SLOT_MIN_GAP_PP         = 10.0  # Mindest-Ø-Return-Rückstand der Spreads in pp
SLOT_MIN_N_SECTOR_TOTAL = 20    # Guardrail: Sektor-Analyse erst ab n≥20 gesamt
SLOT_MIN_N_SECTOR_GROUP = 3     # ein Sektor zählt erst ab n≥3 mit


def _mean(xs: list[float]) -> float | None:
    return statistics.mean(xs) if xs else None


def _median(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def _trailing_vs_hard_tp(history: dict) -> dict:
    """Analyse 1: virtuelle Trailing-Stop-Weiterführung vs. echter harter TP."""
    name = "Trailing-Stop vs. harter Take-Profit"
    try:
        sims = [
            s for s in (history.get("trailing_sim") or [])
            if isinstance(s, dict) and s.get("trailing_outcome") is not None
        ]
    except Exception:
        sims = []

    n = len(sims)
    if n == 0:
        return {
            "name": name, "n": 0,
            "finding": "Noch keine abgeschlossenen Trailing-Simulationen.",
            "recommendation": "weiter Daten sammeln", "ready": False,
        }

    tp_outs, trail_outs, deltas, trail_better = [], [], [], 0
    for s in sims:
        try:
            tp  = float(s.get("tp_outcome") or 0)
            trl = float(s.get("trailing_outcome"))
        except (TypeError, ValueError):
            continue
        tp_outs.append(tp)
        trail_outs.append(trl)
        deltas.append(trl - tp)
        if trl > tp:
            trail_better += 1

    n = len(deltas)
    if n == 0:
        return {
            "name": name, "n": 0,
            "finding": "Trailing-Simulationen ohne verwertbare Outcomes.",
            "recommendation": "weiter Daten sammeln", "ready": False,
        }

    mean_tp, mean_trail   = _mean(tp_outs), _mean(trail_outs)
    median_tp, median_trl = _median(tp_outs), _median(trail_outs)
    mean_delta   = _mean(deltas)
    share_better = trail_better / n

    finding = (
        f"n={n}: Ø TP={mean_tp:+.1%} (Median {median_tp:+.1%}) vs. "
        f"Ø Trailing={mean_trail:+.1%} (Median {median_trl:+.1%}) — "
        f"Ø-Delta {mean_delta:+.1%}, Trailing besser bei {share_better:.0%} der Trades."
    )
    ready = n >= SLOT_MIN_N_TRAILING
    recommendation = None
    if ready and mean_delta is not None and mean_delta * 100 >= SLOT_MIN_DELTA_PP:
        recommendation = (
            "Trailing-Stop statt hartem Take-Profit für den nächsten Slot erwägen "
            f"(Ø-Delta {mean_delta:+.1%} über n={n})."
        )
    elif not ready:
        recommendation = "Weiter Daten sammeln (n < 5)."

    return {"name": name, "n": n, "finding": finding,
            "recommendation": recommendation, "ready": ready}


def _spreads_vs_long_calls(history: dict) -> dict:
    """Analyse 2: Spread-Strategien vs. reine Long Calls über closed_trades."""
    name = "Spreads vs. Long Calls"
    try:
        closed = [
            t for t in (history.get("closed_trades") or [])
            if isinstance(t, dict) and t.get("outcome") is not None
        ]
    except Exception:
        closed = []

    spreads, longs = [], []
    for t in closed:
        try:
            outcome = float(t["outcome"])
        except (TypeError, ValueError, KeyError):
            continue
        strategy = str(t.get("strategy") or "").upper()
        if "SPREAD" in strategy:
            spreads.append(outcome)
        elif "LONG" in strategy:
            longs.append(outcome)

    def group_stats(outs: list[float]) -> dict | None:
        if not outs:
            return None
        wins = sum(1 for o in outs if o > 0)
        return {"n": len(outs), "win_rate": wins / len(outs),
                "mean": _mean(outs), "median": _median(outs)}

    ss, ls = group_stats(spreads), group_stats(longs)
    n_total = len(spreads) + len(longs)

    if ss is None or ls is None:
        finding = (
            f"Zu wenig Daten für einen Vergleich (Spreads n={len(spreads)}, "
            f"Long Calls n={len(longs)})."
        )
        return {"name": name, "n": n_total, "finding": finding,
                "recommendation": "weiter Daten sammeln", "ready": False}

    finding = (
        f"Spreads: n={ss['n']}, Win={ss['win_rate']:.0%}, Ø {ss['mean']:+.1%}, "
        f"Median {ss['median']:+.1%} | Long Calls: n={ls['n']}, "
        f"Win={ls['win_rate']:.0%}, Ø {ls['mean']:+.1%}, Median {ls['median']:+.1%}."
    )
    ready = ss["n"] >= SLOT_MIN_N_STRATEGY and ls["n"] >= SLOT_MIN_N_STRATEGY
    recommendation = None
    gap_pp = (ls["mean"] - ss["mean"]) * 100
    if ready and gap_pp >= SLOT_MIN_GAP_PP:
        recommendation = (
            "IV_SPREAD_GATE anheben prüfen "
            f"(Spreads liegen {gap_pp:.0f}pp unter Long Calls im Ø-Return)."
        )
    elif not ready:
        recommendation = (
            f"Weiter Daten sammeln (Spreads n={ss['n']}, Long Calls n={ls['n']}, "
            f"Ziel je ≥{SLOT_MIN_N_STRATEGY})."
        )

    return {"name": name, "n": n_total, "finding": finding,
            "recommendation": recommendation, "ready": ready}


def _to_float_or_none(x) -> float | None:
    """Wandelt x in float um, tolerant gegenüber None/kaputten Werten."""
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _sector_concentration(history: dict) -> dict:
    """Analyse 3: Sektor-Konzentration über closed_trades + Final-MC-Survivor."""
    name = "Sektor-Konzentration"
    rows: list[tuple[str, float | None]] = []
    for t in (history.get("closed_trades") or []):
        if not isinstance(t, dict):
            continue
        etf = t.get("sector_etf")
        if not etf:
            continue
        rows.append((etf, _to_float_or_none(t.get("outcome"))))
    for t in (history.get("shadow_trades") or []):
        if not isinstance(t, dict) or t.get("reject_reason") != "final_mc_survivor":
            continue
        etf = t.get("sector_etf")
        if not etf:
            continue
        rows.append((etf, _to_float_or_none(t.get("outcome"))))

    n_total = len(rows)
    if n_total < SLOT_MIN_N_SECTOR_TOTAL:
        finding = f"Zu wenig Trades mit sector_etf (n={n_total}) für eine belastbare Aussage."
        return {"name": name, "n": n_total, "finding": finding,
                "recommendation": "weiter Daten sammeln", "ready": False}

    groups: dict[str, list[float]] = {}
    for etf, outcome in rows:
        if outcome is None:
            continue
        groups.setdefault(etf, []).append(outcome)

    stats_by_etf = {}
    for etf, outs in groups.items():
        if len(outs) < SLOT_MIN_N_SECTOR_GROUP:
            continue
        wins = sum(1 for o in outs if o > 0)
        stats_by_etf[etf] = {
            "n": len(outs), "win_rate": wins / len(outs), "mean": _mean(outs),
        }

    if not stats_by_etf:
        finding = f"n={n_total} mit sector_etf, aber kein Sektor erreicht n≥{SLOT_MIN_N_SECTOR_GROUP}."
        return {"name": name, "n": n_total, "finding": finding,
                "recommendation": None, "ready": True}

    best  = max(stats_by_etf.items(), key=lambda kv: kv[1]["mean"])
    worst = min(stats_by_etf.items(), key=lambda kv: kv[1]["mean"])

    positive_outcomes = [o for outs in groups.values() for o in outs if o > 0]
    dominant_note = ""
    if positive_outcomes:
        pos_by_etf = {
            etf: sum(1 for o in outs if o > 0) for etf, outs in groups.items()
        }
        top_etf, top_pos_count = max(pos_by_etf.items(), key=lambda kv: kv[1])
        share = top_pos_count / len(positive_outcomes)
        if share > 0.5:
            dominant_note = f" {top_etf} trägt {share:.0%} aller positiven Outcomes."

    finding = (
        f"n={n_total} mit sector_etf, {len(stats_by_etf)} Sektoren mit n≥{SLOT_MIN_N_SECTOR_GROUP}. "
        f"Bester: {best[0]} (n={best[1]['n']}, Win={best[1]['win_rate']:.0%}, "
        f"Ø {best[1]['mean']:+.1%}) — Schlechtester: {worst[0]} (n={worst[1]['n']}, "
        f"Win={worst[1]['win_rate']:.0%}, Ø {worst[1]['mean']:+.1%})."
        f"{dominant_note}"
    )

    recommendation = None
    positive_etfs = [etf for etf, s in stats_by_etf.items() if s["mean"] is not None and s["mean"] > 0]
    negative_etfs = [etf for etf, s in stats_by_etf.items() if s["mean"] is not None and s["mean"] <= 0]
    if len(positive_etfs) == 1 and negative_etfs and len(negative_etfs) == len(stats_by_etf) - 1:
        recommendation = (
            f"Sektor-Gate erwägen — nur {positive_etfs[0]} ist im Ø positiv, "
            f"alle anderen Sektoren (n≥{SLOT_MIN_N_SECTOR_GROUP}) sind es nicht."
        )

    return {"name": name, "n": n_total, "finding": finding,
            "recommendation": recommendation, "ready": True}


def slot_candidate_analysis(history: dict) -> list[dict]:
    """
    Fasst die laufenden Schatten-Messungen als Entscheidungsvorlage für den
    nächsten Tuning-Slot zusammen (Trailing-Stop, Spreads vs. Long Calls,
    Sektor-Konzentration). Ändert NICHTS — reines Reporting.

    Jeder Eintrag: {"name", "n", "finding", "recommendation", "ready"}.
    `ready=False` heißt: Datenbasis zu dünn, die `recommendation` lautet dann
    explizit "weiter Daten sammeln". Wirft nie eine Exception nach außen —
    fehlende/kaputte history-Daten führen höchstens zu ready=False-Einträgen.
    """
    if not isinstance(history, dict):
        history = {}

    analyses = []
    for fn in (_trailing_vs_hard_tp, _spreads_vs_long_calls, _sector_concentration):
        try:
            analyses.append(fn(history))
        except Exception as e:
            analyses.append({
                "name": getattr(fn, "__name__", "?"), "n": 0,
                "finding": f"Analyse fehlgeschlagen: {e}",
                "recommendation": "weiter Daten sammeln", "ready": False,
            })
    return analyses


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

    print("\n── Slot-Analyse — Kandidaten für den nächsten Tuning-Slot " + "─" * 10)
    for a in slot_candidate_analysis(history):
        flag = "✅" if a["ready"] else "⏳"
        print(f"  {flag} {a['name']} (n={a['n']})")
        print(f"      {a['finding']}")
        if a["recommendation"]:
            print(f"      → {a['recommendation']}")

    print(
        "\nHinweis: Kleine n → Zufall dominiert. Schwellen erst ändern, wenn"
        "\nein Muster über ≥20 Trades stabil ist (Schatten-Trades zählen mit)."
    )


if __name__ == "__main__":
    main()

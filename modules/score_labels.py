"""modules/score_labels.py – einheitliche Bezeichnung der Monte-Carlo-Kursziel-Kennzahl (Reporting 2026-10-09).

simulation.hit_rate misst den Anteil simulierter Pfade, in denen das UNDERLYING das Kursziel erreicht.
Sie misst NICHT, ob die Option bzw. der Spread nach Bid/Ask, Slippage, Zeitwert, IV-Effekt, Gebühren
und Exit-Ausführung profitabel ist (historisch, explorativ: Ø 75 % Score vs. 31 % profitable Trades).
In allen Reports heißt sie daher "Target-Hit Score (unkalibriert)" – ein Ranking-Signal, keine
Gewinnwahrscheinlichkeit. Berechnung, Gates und Ranking bleiben unverändert (nur Bezeichnung).
"""
from __future__ import annotations

TARGET_HIT_LABEL = "Target-Hit Score"
TARGET_HIT_LABEL_DE = "Kursziel-Treffer-Score"
UNCALIBRATED = "uncalibrated"
TARGET_HIT_EXPLANATION = (
    "Der Target-Hit Score misst die relative Wahrscheinlichkeit, dass das Underlying das simulierte "
    "Kursziel erreicht. Er ist keine kalibrierte Gewinnwahrscheinlichkeit des Options-Trades. "
    "Höherer Score = relativ attraktiver im Ranking, nicht automatisch höhere reale Gewinnchance "
    "in gleicher Prozenthöhe.")


def calibration_status(paper_perf: dict | None) -> str:
    """'uncalibrated' solange keine belastbare LIVE_FORWARD_CALIBRATION existiert (sonst 'calibrated')."""
    from modules.learning_health import live_forward_calibration
    lf = live_forward_calibration(paper_perf if isinstance(paper_perf, dict) else None)
    return "calibrated" if lf["status"] == "CALIBRATED" else UNCALIBRATED


def score_text(value, paper_perf: dict | None = None, digits: int = 0) -> str:
    """z. B. 'Target-Hit Score: 70 % (uncalibrated) – Ranking signal only'."""
    try:
        v = f"{float(value) * 100:.{digits}f} %"
    except (TypeError, ValueError):
        v = "n/a"
    return f"{TARGET_HIT_LABEL}: {v} ({calibration_status(paper_perf)}) – Ranking signal only"

"""
modules/trade_scorer.py  —  Adaptive Asymmetry-Scanner v9.0

Score-System (0-100 Punkte total):

A. SIGNAL-QUALITÄT (0-40 Punkte)
   √(Impact × Surprise) / 10 × 40

B. OPTIONEN-QUALITÄT (0-30 Punkte)
   - ROI netto:         max 15 Pkt  (DTE-aware, kurze DTE mit hohem ROI = Penalty)
   - Bid-Ask Spread:    max 10 Pkt
   - Open Interest:     max  5 Pkt

C. RISIKO-ABZÜGE (0 bis -30 Punkte)
   - Bear Case Severity ≥7:   bis -15 Pkt
   - IV-Rank ≥85%:            bis -10 Pkt
   - 48h-Move ≥5%:            bis  -5 Pkt

D. KONTEXT-BONUS (0-30 Punkte)
   - Makro expansiv (sektor-gewichtet): +10 × Sektor-Sensitivität
   - Sektor-Momentum positiv:           +10
   - KI-Konsistenz:                     +5
   - Richtungskonflikt:                 -5

Änderungen v9.0:
   #4  ROI-Scoring DTE-aware: >50% ROI bei <30 DTE → Penalty für Leverage-Artefakt.
       Unterscheidet jetzt zwischen realistischem ROI und kurzfristigem Hebel-Effekt.

   #8  Makro-Bonus sektor-spezifisch: Finanzwerte profitieren mehr von steiler Kurve
       als Biotech/Healthcare. Multiplikator 0.1 (Healthcare) bis 1.0 (Financials).

Finale Empfehlung:
   >=75: STRONG BUY | 60-74: BUY | 45-59: WATCH | <45: AVOID
"""

from __future__ import annotations
import math
import logging

log = logging.getLogger(__name__)

# v9.0 #8: Sektor-spezifische Makro-Sensitivität
# Wie stark profitiert ein Sektor von einem expansiven Makro-Regime (steile Zinskurve)?
# Basis: empirische Korrelation zwischen 10Y-2Y Spread und Sektorperformance
MACRO_SECTOR_SENSITIVITY: dict[str, float] = {
    "Financial Services":     1.00,  # Direkter NIM-Effekt
    "Financials":             1.00,
    "Real Estate":            0.85,  # Rate-sensitiv (Refinanzierung)
    "Consumer Cyclical":      0.70,  # Konjunktursensitiv
    "Industrials":            0.70,
    "Technology":             0.60,  # Wachstums-Diskontierung
    "Communication Services": 0.50,
    "Basic Materials":        0.50,
    "Energy":                 0.40,  # Rohstoff-getrieben, weniger Makro
    "Utilities":              0.55,  # Defensive, moderat rate-sensitiv
    "Consumer Defensive":     0.25,  # Stabil, kaum Makro-Beta
    "Healthcare":             0.20,  # Pipeline-getrieben, makro-unabhängig
    "Biotechnology":          0.10,  # Fast ausschließlich katalysatorgetrieben
}


def compute_trade_score(proposal: dict) -> dict:
    da       = proposal.get("deep_analysis", {}) or {}
    option   = proposal.get("option", {}) or {}
    roi_data = proposal.get("roi_analysis", {}) or {}
    features = proposal.get("features", {}) or {}
    red_team = da.get("red_team", {}) or {}
    ticker   = proposal.get("ticker", "?")
    sector   = proposal.get("sector", "") or ""

    # ── A: SIGNAL-QUALITÄT (0-40) ─────────────────────────────────────────────
    impact   = float(da.get("impact", 0) or 0)
    surprise = float(da.get("surprise", 0) or 0)

    if impact > 0 and surprise > 0:
        signal_raw = math.sqrt(impact * surprise) / 10.0
    else:
        signal_raw = 0.0
    signal_pts = round(signal_raw * 40, 1)

    # ── B: OPTIONEN-QUALITÄT (0-30) ───────────────────────────────────────────
    roi_net    = float(roi_data.get("roi_net", 0) or 0)
    spread_pct = float(roi_data.get("spread_pct", 1) or 1)
    oi         = int(option.get("open_interest", 0) or 0)
    dte        = int(roi_data.get("dte", 120) or 120)

    # v9.0 #4: DTE-aware ROI-Scoring
    # Kurze DTE + hoher ROI = Leverage-Artefakt, kein echter Alpha-Edge.
    # Penalty verhindert, dass 16-DTE 135%-ROI denselben Score bekommt wie
    # ein 90-DTE 40%-ROI (der probability-weighted realistischer ist).
    roi_penalty = 0.0
    if roi_net > 0.50 and dte < 30:
        # Extremer Leverage-Artefakt: 135% ROI in 16 Tagen
        roi_for_scoring = 0.20
        roi_penalty     = -6.0
        log.debug(f"  [{ticker}] ROI-Penalty: {roi_net:.0%} bei {dte}d → score={roi_for_scoring:.0%} -6pts")
    elif roi_net > 0.80 and dte < 60:
        # Hohes Leverage bei Short-Term
        roi_for_scoring = 0.25
        roi_penalty     = -3.0
        log.debug(f"  [{ticker}] ROI-Penalty: {roi_net:.0%} bei {dte}d → score={roi_for_scoring:.0%} -3pts")
    else:
        # Realistischer Bereich: cap auf 40% (leicht höher als v8 wegen mc_weight)
        roi_for_scoring = min(max(0.0, roi_net), 0.40)

    roi_pts = round((roi_for_scoring / 0.40) * 15, 1)

    if spread_pct <= 0.05:
        liq_pts = 10.0
    elif spread_pct <= 0.10:
        liq_pts = 7.0
    elif spread_pct <= 0.20:
        liq_pts = 4.0
    else:
        liq_pts = 1.0

    if oi >= 1000:
        oi_pts = 5.0
    elif oi >= 500:
        oi_pts = 3.0
    elif oi >= 100:
        oi_pts = 1.5
    else:
        oi_pts = 0.5

    # v10.0 #5: Edge vs. Market-Implied Expected Move
    # edge_vs_implied in % (z.B. 4.2 = Modell erwartet 4.2% mehr als Markt impliziert)
    # Positiver Edge → Signal hat echte Informationsasymmetrie
    # Negativer Edge → Markt hat das Kurs-Ziel bereits eingepreist
    edge_vs_implied = proposal.get("edge_vs_implied")  # % oder None
    edge_pts = 0.0
    if edge_vs_implied is not None:
        if edge_vs_implied < -2.0:
            edge_pts = -6.0   # Modell-Move unter Market-Implied → kein Edge
        elif edge_vs_implied < 0.0:
            edge_pts = -2.0   # leicht negativ
        elif edge_vs_implied > 5.0:
            edge_pts = 4.0    # starker positiver Edge

    options_pts = min(round(roi_pts + liq_pts + oi_pts + roi_penalty + edge_pts, 1), 30.0)
    options_pts = max(options_pts, 0.0)

    # ── C: RISIKO-ABZÜGE (0 bis -30) ─────────────────────────────────────────
    bear_sev  = float(da.get("bear_case_severity", 0) or 0)
    iv_rank   = float(proposal.get("iv_rank", 50) or 50)
    _raw_move = float(features.get("price_change_48h", 0) or 0)
    move_48h  = abs(_raw_move / 100.0 if abs(_raw_move) > 1.0 else _raw_move)

    if bear_sev <= 6:
        bear_deduct = 0.0
    elif bear_sev <= 7:
        bear_deduct = -(bear_sev - 5) * 5.0
    else:
        bear_deduct = -10.0 - (bear_sev - 7) * 2.5

    surprise_high = surprise >= 8.0
    if iv_rank >= 95:
        iv_deduct = -5.0 if surprise_high else -10.0
    elif iv_rank >= 85:
        iv_deduct = -2.5 if surprise_high else -5.0
    elif iv_rank >= 70:
        iv_deduct = -1.0 if surprise_high else -2.0
    else:
        iv_deduct = 0.0

    if move_48h >= 0.10:
        move_deduct = -5.0
    elif move_48h >= 0.05:
        move_deduct = -3.0
    else:
        move_deduct = 0.0

    risk_pts = round(bear_deduct + iv_deduct + move_deduct, 1)

    # ── D: KONTEXT-BONUS (0-30) ───────────────────────────────────────────────
    macro_regime = da.get("macro_regime", "neutral")
    dir_conflict = bool(da.get("direction_conflict", False))
    sector_info  = proposal.get("sector_momentum", {}) or {}
    sector_rs    = float(sector_info.get("rel_strength", 0) or 0)

    # v9.0 #8: Sektor-spezifische Makro-Sensitivität
    # Makro-Bonus wird mit der Sektor-Korrelation zur Zinskurve gewichtet.
    macro_sensitivity = MACRO_SECTOR_SENSITIVITY.get(sector, 0.50)
    if macro_regime == "expansive":
        macro_bonus_base = 10.0
    elif macro_regime in ("neutral", "unknown"):
        macro_bonus_base = 5.0
    else:
        macro_bonus_base = 0.0
    macro_bonus = round(macro_bonus_base * macro_sensitivity, 1)

    if sector_rs >= 0.03:
        sector_bonus = 10.0
    elif sector_rs >= 0.0:
        sector_bonus = 5.0
    else:
        sector_bonus = 0.0

    ki_bonus     = 5.0 if not dir_conflict else 0.0
    conflict_pen = 0.0 if not dir_conflict else -5.0

    context_pts = min(round(macro_bonus + sector_bonus + ki_bonus + conflict_pen, 1), 30.0)

    # ── TOTAL ─────────────────────────────────────────────────────────────────
    total = max(0, min(100, int(round(signal_pts + options_pts + risk_pts + context_pts, 0))))

    # ── GRADE ─────────────────────────────────────────────────────────────────
    if total >= 75:
        grade = "STRONG BUY"
        emoji = "🟢"
    elif total >= 60:
        grade = "BUY"
        emoji = "🟡"
    elif total >= 45:
        grade = "WATCH"
        emoji = "🟠"
    else:
        grade = "AVOID"
        emoji = "🔴"

    grade_short = grade
    grade_full  = f"{grade} {emoji}"

    # ── REASONING ─────────────────────────────────────────────────────────────
    strengths  = []
    weaknesses = []

    if signal_pts >= 25:
        strengths.append(f"starkes Signal (Impact={impact:.0f}/Surprise={surprise:.0f})")
    elif signal_pts >= 15:
        strengths.append(f"solides Signal (Impact={impact:.0f}/Surprise={surprise:.0f})")
    else:
        weaknesses.append(f"schwaches Signal (Impact={impact:.0f}/Surprise={surprise:.0f})")

    if roi_net >= 0.20:
        strengths.append(f"guter ROI ({roi_net:.0%})")
    elif roi_net < 0:
        weaknesses.append(f"negativer ROI ({roi_net:.0%})")

    if roi_penalty < 0:
        weaknesses.append(f"ROI-Penalty: {roi_net:.0%} bei {dte}d (Leverage-Artefakt)")

    if edge_vs_implied is not None:
        implied_pct = proposal.get("implied_move_pct", 0) or 0
        model_pct   = proposal.get("model_move_pct", 0) or 0
        if edge_vs_implied < -2.0:
            weaknesses.append(
                f"Kein IV-Edge: Model +{model_pct:.1f}% ≤ Markt ±{implied_pct:.1f}% (bereits eingepreist)"
            )
        elif edge_vs_implied > 5.0:
            strengths.append(
                f"Klarer Edge: Model +{model_pct:.1f}% vs. Markt ±{implied_pct:.1f}% (+{edge_vs_implied:.1f}%)"
            )

    if iv_rank >= 85:
        weaknesses.append(f"hohe IV ({iv_rank:.0f}%) — Optionen teuer")
    elif iv_rank <= 40:
        strengths.append(f"günstige IV ({iv_rank:.0f}%)")

    if move_48h >= 0.05:
        weaknesses.append(f"48h-Move +{move_48h:.0%} — Alpha teilw. eingepreist")

    if bear_sev >= 7:
        weaknesses.append(f"hohes Bear-Risiko ({bear_sev:.0f}/10)")

    if sector_rs >= 0.03:
        strengths.append("positive Sektor-Relative-Stärke")
    elif sector_rs < -0.03:
        weaknesses.append("Sektor underperformt Markt")

    if macro_regime == "expansive" and macro_bonus > 5:
        strengths.append(f"expansives Makro (Sektor-Sensitivität {macro_sensitivity:.0%})")
    elif macro_regime == "expansive" and macro_bonus <= 3:
        weaknesses.append(f"Makro expansiv aber Sektor-Korrelation gering ({macro_sensitivity:.0%})")

    reasoning_parts = []
    if strengths:
        reasoning_parts.append("Stärken: " + ", ".join(strengths[:3]))
    if weaknesses:
        reasoning_parts.append("Risiken: " + ", ".join(weaknesses[:3]))

    best_for     = (da.get("asymmetry_reasoning", "") or "Strukturelles Signal erkannt")[:700]
    best_against = (red_team.get("argument_1", "") or "n/a")[:700]

    log.info(
        f"  [{ticker}] Score: {total}/100 ({grade_short}) | "
        f"Signal={signal_pts} Optionen={options_pts} "
        f"Risiko={risk_pts} Kontext={context_pts} "
        f"(MacroSens={macro_sensitivity:.0%} Sektor='{sector}')"
    )

    return {
        "total":       total,
        "grade":       grade_full,
        "grade_short": grade_short,
        "components": {
            "signal_quality":    signal_pts,
            "options_quality":   options_pts,
            "risk_deductions":   risk_pts,
            "context_bonus":     context_pts,
            "roi_penalty":       roi_penalty,
            "edge_pts":          edge_pts,
            "macro_sensitivity": macro_sensitivity,
        },
        "reasoning":             " | ".join(reasoning_parts),
        "best_argument_for":     best_for,
        "best_argument_against": best_against,
    }


def rank_proposals(proposals: list[dict]) -> list[dict]:
    """Rankt Trade-Vorschläge nach Score (höchster zuerst)."""
    for p in proposals:
        p["trade_score"] = compute_trade_score(p)

    proposals.sort(key=lambda x: x["trade_score"]["total"], reverse=True)

    for i, p in enumerate(proposals):
        p["trade_rank"] = i + 1

    return proposals

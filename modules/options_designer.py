"""
modules/options_designer.py v10.4

Änderungen v10.4:
    #EDGE-GATE: Break-even statt straddle/2 als Vergleichsbasis.
        Long Call:        BEP = (strike + ask  - S₀) / S₀
        Bull Call Spread: BEP = (strike + net_debit - S₀) / S₀
        Fallback:         BEP = straddle / 2 (wenn keine Prämie verfügbar)
        Straddle weiterhin geloggt als Markt-Kontext.
        trade_bep_pct im Return-Dict für Reporting.

Änderungen v10.3:
    #MC-PNL Phase 2: Stochastische IV via Ornstein-Uhlenbeck-Prozess.
        history wird an simulate_option_pnl() weitergegeben.
        OU-Parameter: 2-Layer Hybrid (Heuristik bei iv_rank + lineare Regression
        nach ≥30 Tagen eigener IV-Historie). iv_crush_factor entfernt.

Änderungen v10.2:
    #MC-PNL: Options-P&L Monte Carlo (Phase 1).
        Nach ROI-Gate und Edge-Gate: simulate_option_pnl() aus MirofishSimulation.
        roi_net wird durch MC-basierten expected_pnl_pct ersetzt.
        Theta und IV-Crush wirken via Haltepunkt-Repricing bei ~45% der Laufzeit.
        iv_crush_factor: 0.22 wenn IV-Rank ≥ 60, sonst 0.12.

Änderungen v10.1:
    #EDGE-GATE: Hard-Rejection wenn Edge vs. Market-Implied Move ≤ 1%.
        Bisher: Straddle-Daten wurden berechnet, geloggt, aber nicht zum Filtern genutzt.
        Jetzt: Trade wird pro Tier verworfen wenn model_move - implied_move ≤ 1%.
        Platzierung: nach ROI-Gate, nach Straddle-Berechnung, vor return.

Änderungen v9.0:
    #1  DTE-Architektur: Erste passende Tier-Logik ersetzt durch Catalyst-aligned DTE.
        time_to_materialization aus deep_analysis bestimmt jetzt das DTE-Minimum.
        Tiers unterhalb des Minimums werden übersprungen (nicht nur vega_loss-Check).

    #2  IV-Spread-Gate: 85% → 52%.
        BULL_CALL_SPREAD wird ab IV-Rank ≥ 52% gewählt (nicht erst bei 85%).
        Bei normaler IV (< 52%) bleibt LONG_CALL, weil Vega-Exposure akzeptabel.

    #3  Probability-weighted ROI: roi_delta × mc_hit_rate.
        MC Hit-Rate aus quick_mc wird als Wahrscheinlichkeitsgewicht genutzt.
        Ergebnis: 135%-ROI bei 0.55 Hit-Rate → ~74% realistischer Erwartungswert.

    #6  time_to_materialization wird jetzt explizit aus deep_analysis
        ausgelesen und an _compute_roi + Proposal weitergegeben.

    #13 Theta-Decay-Gate: theta_daily_pct > 3%/Tag bei Short-Term → erzwingt
        Upgrade auf Mid-Term. Verhindert dass Options mit >3% Tagesverlust
        durch Zeitwert bei Short-Term akzeptiert werden.

Änderungen v8.1:
    1. Adaptives Strike-Fenster bei hohem Aktienkurs
    2. Delta-Filter im Tradier-Pfad

Änderungen v8.0:
    - Tradier Live-API als primäre Datenquelle
    - yfinance Fallback

Änderungen v7.5:
    - IV-Rank Kalibrierung
"""

from __future__ import annotations
import logging
import math
import os
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timezone
from typing import Optional

import yfinance as yf

from modules.config              import cfg
from modules.macro_context       import get_macro_regime_multiplier, get_macro_context
from modules.mirofish_simulation import MirofishSimulation

log = logging.getLogger(__name__)

# IV-Gate: ab diesem Wert wird SPREAD statt LONG_CALL gewählt
# v9.0: 85% → 52% (Spread schützt bereits bei mittlerer IV)
IV_SPREAD_GATE = 52.0

# IV-Gate für Scoring-Log (behalten für Kompatibilität)
IV_RANK_GATE = IV_SPREAD_GATE

DTE_TIERS = [
    {"label": "Short-Term", "dte_min": 14,  "dte_max": 60,  "min_roi": 0.15},
    {"label": "Mid-Term",   "dte_min": 61,  "dte_max": 149, "min_roi": 0.12},
    {"label": "Long-Term",  "dte_min": 150, "dte_max": 365, "min_roi": 0.10},
]

# v9.0 #6: time_to_materialization → DTE-Minimum
# Verhindert 16-DTE-Option bei 3-Monats-Thesis
TTM_TO_DTE_MIN: dict[str, int] = {
    "4-8 Wochen":  45,   # Min 45d: Short-Thesis braucht trotzdem Puffer
    "2-3 Monate":  55,   # Mid-Term Minimum (kein 16-DTE!)
    "6 Monate":   140,   # Long-Term Minimum
}

# v9.0 #13: Theta-Decay-Gate
# Wenn Zeitwertverlust > X%/Tag des Prämienpreises → Short-Term überspringen
THETA_DAILY_PCT_GATE = 0.030   # 3 % pro Tag

# v10.0 #2: Event-typ-spezifische IV-Crush-Schätzungen (Basis-Rate, vor IV-Rank-Scaling)
# Earnings:  IV crush hoch (42%/18%/6%), Event fix terminiert
# FDA:       Sehr hoch (60%/22%/8%), binäres Outcome kollabiert Post-Event
# M&A:       Mittel (25%/10%/3%), bereits teilweise eingepreist
# Insider:   Niedrig (10%/4%/1%), kein konkretes Event-Datum
# Other:     Standard (~22%/10%/3%)
# Scaling:   × (0.5 + iv_rank/200) — bei iv_rank=50: ×0.75, bei iv_rank=100: ×1.0
EVENT_IV_CRUSH: dict[str, dict[str, float]] = {
    "EARNINGS": {"short": 0.42, "mid": 0.18, "long": 0.06},
    "FDA":      {"short": 0.60, "mid": 0.22, "long": 0.08},
    "MA":       {"short": 0.25, "mid": 0.10, "long": 0.03},
    "INSIDER":  {"short": 0.10, "mid": 0.04, "long": 0.01},
    "OTHER":    {"short": 0.22, "mid": 0.10, "long": 0.03},
}

SECTOR_ETF = {
    "Technology": "XLK", "Healthcare": "XLV", "Biotechnology": "XBI",
    "Financial Services": "XLF", "Financials": "XLF", "Energy": "XLE",
    "Consumer Cyclical": "XLY", "Consumer Defensive": "XLP",
    "Industrials": "XLI", "Basic Materials": "XLB",
    "Real Estate": "XLRE", "Utilities": "XLU",
    "Communication Services": "XLC", "default": "SPY",
}

RELATIVE_STRENGTH_MIN = -0.15  # -0.08→-0.15: weniger aggressiv, rettet Underreaction-Signale

TRADIER_BASE    = "https://api.tradier.com/v1"
TRADIER_TIMEOUT = 10

MIN_STRIKE_DOLLAR_RANGE = 15.0


def _classify_catalyst_type(s: dict) -> str:
    """
    Klassifiziert den Katalysator-Typ aus alpha_signals + deep_analysis.
    Priorität: FDA > Earnings > M&A > Insider > Other
    Wird in _compute_roi() für event-spezifischen IV-Crush genutzt.
    """
    alpha        = s.get("alpha_signals", {}) or {}
    da           = s.get("deep_analysis", {}) or {}
    catalyst_txt = (da.get("catalyst", "") or "").lower()

    if alpha.get("fda_catalyst"):
        return "FDA"
    # eps_drift ist ein Float — nur als EARNINGS klassifizieren wenn signifikant (>5%)
    eps_drift_significant = abs(float(alpha.get("eps_drift") or 0)) > 0.05
    if eps_drift_significant or any(k in catalyst_txt for k in ("earnings", " eps", "ergebnis", "guidance")):
        return "EARNINGS"
    if any(k in catalyst_txt for k in ("merger", "acquisition", "takeover", "buyout", "übernahme", "deal", "m&a")):
        return "MA"
    if alpha.get("insider_cluster") or "insider" in catalyst_txt:
        return "INSIDER"
    return "OTHER"


# ── Tradier Hilfsfunktionen ───────────────────────────────────────────────────

def _tradier_headers() -> dict:
    api_key = os.environ.get("TRADIER_API_KEY", "")
    return {
        "Authorization": f"Bearer {api_key}",
        "Accept":        "application/json",
    }


def _tradier_expirations(symbol: str) -> list[str]:
    try:
        resp = requests.get(
            f"{TRADIER_BASE}/markets/options/expirations",
            params={"symbol": symbol, "includeAllRoots": "true"},
            headers=_tradier_headers(),
            timeout=TRADIER_TIMEOUT,
        )
        resp.raise_for_status()
        data  = resp.json()
        dates = data.get("expirations", {}).get("date", []) or []
        if isinstance(dates, str):
            dates = [dates]
        return sorted(dates)
    except Exception as e:
        log.debug(f"Tradier Expirations [{symbol}]: {e}")
        return []


def _tradier_chain(symbol: str, expiration: str) -> list[dict]:
    try:
        resp = requests.get(
            f"{TRADIER_BASE}/markets/options/chains",
            params={
                "symbol":     symbol,
                "expiration": expiration,
                "greeks":     "true",
            },
            headers=_tradier_headers(),
            timeout=TRADIER_TIMEOUT,
        )
        resp.raise_for_status()
        data    = resp.json()
        options = data.get("options", {}).get("option", []) or []
        if isinstance(options, dict):
            options = [options]
        return options
    except Exception as e:
        log.debug(f"Tradier Chain [{symbol} {expiration}]: {e}")
        return []


def _tradier_chain_to_df(options: list[dict], option_type: str) -> pd.DataFrame:
    rows = []
    for o in options:
        if o.get("option_type") != option_type:
            continue
        greeks = o.get("greeks") or {}

        iv = (
            greeks.get("mid_iv")
            or greeks.get("smv_vol")
            or 0.30
        )
        if not isinstance(iv, (int, float)) or iv <= 0.01:
            iv = 0.30

        delta = greeks.get("delta")
        delta = float(delta) if isinstance(delta, (int, float)) else 0.0

        rows.append({
            "strike":            float(o.get("strike", 0)),
            "bid":               float(o.get("bid") or 0),
            "ask":               float(o.get("ask") or 0),
            "openInterest":      int(o.get("open_interest") or 0),
            "impliedVolatility": float(iv),
            "delta":             delta,
            "volume":            int(o.get("volume") or 0),
            "_option_symbol":    o.get("symbol", ""),
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# ── Strike-Fenster Berechnung ─────────────────────────────────────────────────

def _strike_window(current: float, dte_min: int, dte_max: int) -> tuple[float, float]:
    """
    Berechnet OTM/ITM-Multiplikatoren für das Strike-Fenster.

    v8.1: Adaptiv bei hohem Aktienkurs.
    Mindestens MIN_STRIKE_DOLLAR_RANGE ($15) Spielraum pro Seite.

    Beispiele:
        AME  $310, Short-Term: 3% = $9.3  → zu eng → $15 = 4.8% → otm=1.048
        AAPL $210, Short-Term: 3% = $6.3  → zu eng → $15 = 7.1% → otm=1.071
        NVDA $900, Short-Term: 3% = $27.0 → ok      → otm=1.03 bleibt
    """
    days_mid = (dte_min + dte_max) / 2
    if days_mid <= 60:
        base_otm, base_itm = 1.03, 0.97
    elif days_mid <= 149:
        base_otm, base_itm = 1.08, 0.96
    else:
        base_otm, base_itm = 1.12, 0.95

    if current > 0 and current * (base_otm - 1.0) < MIN_STRIKE_DOLLAR_RANGE:
        adj     = MIN_STRIKE_DOLLAR_RANGE / current
        otm_max = round(1.0 + adj, 4)
        itm_max = round(1.0 - adj, 4)
        log.debug(
            f"  Strike-Fenster adaptiv: ${current:.0f} × {base_otm-1:.0%} "
            f"= ${current*(base_otm-1):.1f} < ${MIN_STRIKE_DOLLAR_RANGE} "
            f"→ otm={otm_max:.4f} itm={itm_max:.4f}"
        )
    else:
        otm_max, itm_max = base_otm, base_itm

    return otm_max, itm_max


# ── Haupt-Klasse ──────────────────────────────────────────────────────────────

class OptionsDesigner:

    def __init__(self, gates, history: dict = None):
        self.gates   = gates
        self.history = history or {}
        self._use_tradier = bool(os.environ.get("TRADIER_API_KEY", "").strip())
        if self._use_tradier:
            log.info("OptionsDesigner: Tradier Live-API aktiv (Primary)")
        else:
            log.warning("OptionsDesigner: TRADIER_API_KEY fehlt → yfinance Fallback")
        # VIX Term Structure: einmal laden, gecacht für alle Ticker dieses Runs
        self._vix_ts = get_macro_context().get("vix_term_structure", {})

    def run(self, signals: list[dict]) -> list[dict]:
        proposals = []
        for s in signals:
            ticker = s.get("ticker", "")
            if not self._bear_case_ok(s):
                continue
            t_obj = yf.Ticker(ticker)
            if not self._sector_momentum_ok(s, t=t_obj):
                log.info(f"  [{ticker}] SECTOR-GATE → verworfen")
                continue
            proposal = self._design_with_adaptive_dte(s, t_obj)
            if proposal:
                proposals.append(proposal)
        return proposals

    def _design_with_adaptive_dte(self, s: dict, t=None) -> Optional[dict]:
        ticker    = s["ticker"]
        direction = s.get("deep_analysis", {}).get("direction", "BULLISH")
        sim       = s.get("simulation", {})
        current   = sim.get("current_price", 0)

        if current <= 0:
            return None

        if self.gates.has_upcoming_earnings(ticker):
            log.info(f"  [{ticker}] EARNINGS-GATE → blockiert")
            return None

        if t is None:
            t = yf.Ticker(ticker)
        iv_rank = self._get_iv_rank(ticker, t)

        # v9.0 #6: time_to_materialization → DTE-Minimum
        da  = s.get("deep_analysis", {})
        ttm = da.get("time_to_materialization", "4-8 Wochen") or "4-8 Wochen"
        dte_floor = TTM_TO_DTE_MIN.get(ttm, 14)
        log.info(
            f"  [{ticker}] TTM='{ttm}' → DTE-Minimum={dte_floor}d "
            f"(Catalyst-aligned DTE Gate)"
        )

        # v9.0 #3: MC Hit-Rate für probability-weighted ROI
        # v10.0 #6: Kein unterer Clamp mehr — 0.30 war künstlich und verbesserte
        # schlechte Signale. Mirofish filtert bereits < threshold (0.45/0.50).
        qmc         = s.get("quick_mc", {}) or {}
        mc_hit_rate = float(qmc.get("hit_rate", 0.65) or 0.65)
        mc_hit_rate = min(0.95, mc_hit_rate)   # nur obere Grenze, kein Floor

        # v10.0 #2: Katalysator-Typ für event-spezifischen IV-Crush
        catalyst_type = _classify_catalyst_type(s)

        # Dealer-Gamma aus alpha_signals (bereits in v9.0 berechnet)
        dealer_gamma = (
            s.get("alpha_signals", {}).get("dealer_gamma", {}) or {}
        )

        mirofish_sim     = MirofishSimulation()
        results_per_tier = []

        for tier in DTE_TIERS:
            label = tier["label"]

            # v9.0 #1: Überspringe Tiers die unter dem Catalyst-DTE-Minimum liegen.
            # Korrekte Bedingung: dte_min des Tiers < dte_floor
            # Beispiel: "2-3 Monate" → dte_floor=55
            #   Short-Term dte_min=14 < 55 → übersprungen ✓
            #   Mid-Term   dte_min=61 < 55 → False → evaluiert  ✓
            if tier["dte_min"] < dte_floor:
                log.info(
                    f"  [{ticker}] {label}: dte_min={tier['dte_min']}d < "
                    f"dte_floor={dte_floor}d (Thesis='{ttm}') → übersprungen"
                )
                continue

            strategy = self._select_strategy(ticker, direction, iv_rank, dealer_gamma)
            option   = self._find_option_for_dte(
                ticker, strategy, current, tier["dte_min"], tier["dte_max"], t
            )

            if not option:
                log.info(f"  [{ticker}] {label}: kein Kontrakt verfügbar")
                if "SPREAD" in strategy:
                    fallback = "LONG_CALL" if "BULL" in strategy else "LONG_PUT"
                    option   = self._find_option_for_dte(
                        ticker, fallback, current, tier["dte_min"], tier["dte_max"], t
                    )
                    if option:
                        strategy = fallback
                        log.info(f"  [{ticker}] {label}: Fallback → {fallback}")
                if not option:
                    continue

            roi = self._compute_roi(option, sim, iv_rank, tier, strategy, mc_hit_rate, catalyst_type)

            try:
                dte_safe       = max(int(option.get("dte") or 1), 1)
                roi_net_safe   = float(roi["roi_net"].real if isinstance(roi["roi_net"], complex) else roi["roi_net"])
                annualized_roi = float((1 + roi_net_safe) ** (365 / dte_safe) - 1)
                annualized_roi = min(annualized_roi, 9.99)
            except Exception:
                annualized_roi = 0.0

            log.info(
                f"  [{ticker}] {label} ({int(option['dte'] or 0)}d): "
                f"ROI={roi['roi_net']:.1%} (mc_weighted) "
                f"theta={roi.get('theta_daily_pct', 0):.1%}/d "
                f"(ann.={annualized_roi:.1%}) "
                f"{'✅ PASS' if roi['passes_roi_gate'] else '❌ FAIL'}"
            )

            results_per_tier.append({
                "tier": label, "dte": option["dte"],
                "option": option, "roi": roi,
                "annualized_roi": round(annualized_roi, 4),
                "strategy": strategy,
            })

            # v9.0 #13: Theta-Decay-Gate — bei Short-Term und hohem Theta → upgrade
            if label == "Short-Term":
                theta_pct = roi.get("theta_daily_pct", 0.0)
                vega_loss = roi.get("vega_loss", 0)

                if theta_pct > THETA_DAILY_PCT_GATE:
                    log.info(
                        f"  [{ticker}] {label}: Theta={theta_pct:.1%}/d > "
                        f"{THETA_DAILY_PCT_GATE:.0%} Gate → Zeitwertverlust zu hoch, "
                        f"versuche längere Laufzeit"
                    )
                    continue

                if vega_loss > 0.35:
                    log.info(
                        f"  [{ticker}] {label}: Vega-Loss={vega_loss:.0%} > 35% "
                        f"→ zu hohes IV-Crush-Risiko, versuche längere Laufzeit"
                    )
                    continue

            if roi["passes_roi_gate"]:
                tier_idx = DTE_TIERS.index(tier)
                if tier_idx > 0:
                    prev_label = DTE_TIERS[0]["label"]
                    prev_roi   = results_per_tier[0]["roi"]["roi_net"] if results_per_tier else None
                    if prev_roi is not None:
                        log.info(
                            f"  [{ticker}] ⚡ LAUFZEIT-RETTUNG: "
                            f"{prev_label} ROI={prev_roi:.1%} (FAIL) → "
                            f"{label} ROI={roi['roi_net']:.1%} (PASS) — "
                            f"Trade akzeptiert mit {option['dte']}d Laufzeit"
                        )

                # v10.4 EDGE-GATE: Trade-spezifischer Break-even statt straddle/2
                # Break-even bei Verfall (konservativ):
                #   Long Call:        (strike + ask  - S₀) / S₀
                #   Bull Call Spread: (strike + net_debit - S₀) / S₀
                # Beim Haltepunkt (~50% DTE) liegt der effektive BEP noch darunter
                # (Zeitwert der verbleibenden Laufzeit hilft) → Gate bleibt konservativ.
                # Straddle wird weiterhin geloggt als Markt-Kontext, aber nicht als Gate.
                implied_move = self._get_atm_straddle(ticker, current, option["expiry"], t)
                model_move   = (
                    (sim.get("target_price", 0) - current) / current
                    if current > 0 and sim.get("target_price", 0) > current else 0.0
                )

                _ask    = float(option.get("ask", 0) or 0)
                _strike = float(option.get("strike", 0) or 0)
                _nd     = float(option.get("net_debit", 0) or 0) if "SPREAD" in strategy else 0.0

                if current > 0 and _strike > 0 and ("SPREAD" in strategy and _nd > 0):
                    trade_bep = (_strike + _nd - current) / current
                    bep_desc  = f"Spread-BEP +{trade_bep:.1%} (net=${_nd:.2f})"
                elif current > 0 and _strike > 0 and _ask > 0:
                    trade_bep = (_strike + _ask - current) / current
                    bep_desc  = f"Call-BEP +{trade_bep:.1%} (ask=${_ask:.2f})"
                elif implied_move is not None:
                    trade_bep = implied_move / 2   # Fallback wenn keine Prämie vorhanden
                    bep_desc  = f"BEP ≈ straddle/2 +{trade_bep:.1%} (fallback)"
                else:
                    trade_bep = None
                    bep_desc  = "BEP n/a"

                edge_vs_implied = (model_move - trade_bep) if trade_bep is not None else None

                if edge_vs_implied is not None:
                    has_edge = edge_vs_implied > 0.005  # 0.5% Mindest-Edge über dem Trade-BEP
                    straddle_ctx = f"Straddle ±{implied_move:.1%} | " if implied_move is not None else ""
                    log.info(
                        f"  [{ticker}] EDGE-CHECK {label}: "
                        f"{straddle_ctx}{bep_desc} | Model +{model_move:.1%} | "
                        f"Edge {edge_vs_implied:+.1%} "
                        f"({'✅ Edge' if has_edge else '❌ kein Edge → verworfen'})"
                    )
                    # Hard-Gate: Trade-BEP > Modell-Ziel → verwerfen
                    if not has_edge:
                        continue

                # v10.3 MC-PNL: Options-P&L Monte Carlo (Phase 2 — stochastische IV)
                mc_result = mirofish_sim.simulate_option_pnl(
                    candidate        = s,
                    option           = option,
                    days_to_expiry   = option.get("dte", 45),
                    history          = self.history,
                    n_paths          = 5000,
                    iv_rank          = iv_rank,
                )
                if "error" not in mc_result:
                    roi["mc_pnl_pct"]      = mc_result["expected_pnl_pct"]
                    roi["roi_net"]         = mc_result["expected_pnl_pct"] - (roi.get("spread_pct", 0.0) * 2)
                    roi["passes_roi_gate"] = roi["roi_net"] >= tier["min_roi"]
                    ou_tag = mc_result.get("ou_method", "heuristic")
                    log.info(
                        f"  [{ticker}] MC-P&L {label} [{ou_tag}]: "
                        f"median={mc_result['expected_pnl_pct']:.1%} "
                        f"σ={mc_result['pnl_std']:.1%} "
                        f"hold={mc_result['hold_days']}d "
                        f"({'✅ PASS' if roi['passes_roi_gate'] else '❌ FAIL → verworfen'})"
                    )
                    if not roi["passes_roi_gate"]:
                        continue

                return {
                    "ticker":              ticker,
                    "strategy":            strategy,
                    "iv_rank":             iv_rank,
                    "iv_gate_applied":     iv_rank >= IV_SPREAD_GATE,
                    "direction":           direction,
                    "option":              option,
                    "roi_analysis":        roi,
                    "dte_tier":            label,
                    "annualized_roi":      round(annualized_roi, 4),
                    "all_tiers_tried":     results_per_tier,
                    "features":            s.get("features", {}),
                    "simulation":          s.get("simulation", {}),
                    "deep_analysis":       s.get("deep_analysis", {}),
                    "sector_momentum":     s.get("sector_momentum", {}),
                    "final_score":         s.get("final_score", 0),
                    "mc_hit_rate":         mc_hit_rate,
                    "time_to_maturation":  ttm,
                    "sector":              s.get("info", {}).get("sector", ""),
                    "catalyst_type":       catalyst_type,
                    "implied_move_pct":    round(implied_move * 100, 2) if implied_move is not None else None,
                    "model_move_pct":      round(model_move * 100, 2),
                    "edge_vs_implied":     round(edge_vs_implied * 100, 2) if edge_vs_implied is not None else None,
                    "mc_pnl_pct":          mc_result.get("expected_pnl_pct"),
                    "mc_pnl_std":          mc_result.get("pnl_std"),
                    "mc_hold_days":        mc_result.get("hold_days"),
                    "mc_ou_method":        mc_result.get("ou_method"),
                    "mc_ou_n_days":        mc_result.get("ou_n_days"),
                    "trade_bep_pct":       round(trade_bep * 100, 2) if trade_bep is not None else None,
                }

        tried = ", ".join(
            f"{r['tier']}={r['roi']['roi_net']:.1%}" for r in results_per_tier
        )
        log.info(f"  [{ticker}] Alle Laufzeiten unter ROI-Gate: {tried} → verworfen")
        return None

    def _select_strategy(
        self,
        ticker:       str,
        direction:    str,
        iv_rank:      float,
        dealer_gamma: dict | None = None,
    ) -> str:
        """
        Strategie-Auswahl mit drei Signalen:
        1. IV-Rank: ≥ 52% → Spread (Vega-Schutz)
        2. Dealer-Gamma: negative Gamma → aggressiver (LONG_CALL auch bei mittlerer IV)
        3. Kombination: negative Gamma + niedrige IV → LONG_CALL
                        positive Gamma + hohe IV   → SPREAD (doppelter Schutz)
        """
        dealer_gamma  = dealer_gamma or {}
        gamma_sign    = dealer_gamma.get("net_gamma_sign", "neutral")
        gamma_ok      = dealer_gamma.get("data_available", False)

        is_bullish = direction == "BULLISH"

        if gamma_ok and gamma_sign == "negative":
            effective_gate = min(IV_SPREAD_GATE + 13, 65.0)
            reason = f"Dealer-Gamma negativ (Trend-Verstärkung) → IV-Gate {effective_gate:.0f}%"
        elif gamma_ok and gamma_sign == "positive":
            effective_gate = max(IV_SPREAD_GATE - 12, 40.0)
            reason = f"Dealer-Gamma positiv (Mean-Reversion) → IV-Gate {effective_gate:.0f}%"
        else:
            effective_gate = IV_SPREAD_GATE
            reason = f"Dealer-Gamma neutral/unbekannt → Standard IV-Gate {IV_SPREAD_GATE:.0f}%"

        # VIX Term Structure: Backwardation → kurzfristige Angst hoch →
        # Vol-Crush nach Event wahrscheinlicher → Spread bevorzugen (Gate -8%)
        vix_structure = self._vix_ts.get("structure", "unknown") if self._vix_ts else "unknown"
        if vix_structure == "backwardation":
            effective_gate = max(effective_gate - 8.0, 35.0)
            reason += f" | VIX Backwardation → Gate -{8:.0f}% ({effective_gate:.0f}%)"
        elif vix_structure == "contango":
            pass  # Contango: kein Adjustment — Standardlogik reicht

        if iv_rank >= effective_gate:
            s = "BULL_CALL_SPREAD" if is_bullish else "BEAR_PUT_SPREAD"
            log.info(
                f"  [{ticker}] IV={iv_rank:.0f}% ≥ {effective_gate:.0f}% → {s} "
                f"(Spread | {reason})"
            )
        else:
            s = "LONG_CALL" if is_bullish else "LONG_PUT"
            log.info(
                f"  [{ticker}] IV={iv_rank:.0f}% < {effective_gate:.0f}% → {s} "
                f"(Naked | {reason})"
            )
        return s

    def _compute_roi(
        self,
        option:        dict,
        sim:           dict,
        iv_rank:       float,
        tier:          dict,
        strategy:      str   = "",
        mc_hit_rate:   float = 0.65,   # v9.0 #3: probability weight
        catalyst_type: str   = "OTHER", # v10.0 #2: event-spezifischer IV-Crush
    ) -> dict:
        bid     = option.get("bid", 0) or 0
        ask     = option.get("ask", 0) or 0
        strike  = option.get("strike", 0) or 0
        iv      = option.get("implied_vol", 0.30) or 0.30
        dte     = int(option.get("dte", 120) or 120)
        current = sim.get("current_price", 0) or 0
        target  = sim.get("target_price", 0) or 0
        min_roi = tier["min_roi"]

        is_spread = "SPREAD" in strategy
        cost = option.get("net_debit", ask) if is_spread else ask

        if cost <= 0 or current <= 0:
            return {"roi_net": 0.0, "passes_roi_gate": False,
                    "roi_gross": 0.0, "spread_pct": 0.0,
                    "vega_loss": 0.0, "theta_daily_pct": 0.0,
                    "breakeven": 0.0, "breakeven_pct": 0.0,
                    "delta": 0.0, "min_roi_threshold": min_roi}

        if iv < 0.05 or iv > 3.0:
            iv = 0.30

        spread_pct = (ask - bid) / ask if ask > 0 else 0.0
        T          = max(dte / 365.0, 1 / 365.0)  # mindestens 1 Tag

        tradier_delta = option.get("delta")
        if tradier_delta and 0.01 <= abs(float(tradier_delta)) <= 0.99:
            delta = float(tradier_delta)
            vega  = 0.0
            try:
                d1   = (math.log(current / strike) + 0.5 * iv**2 * T) / (iv * math.sqrt(T))
                nd1  = math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
                vega = current * nd1 * math.sqrt(T)
            except Exception:
                nd1 = 0.0
        else:
            try:
                d1   = (math.log(current / strike) + 0.5 * iv**2 * T) / (iv * math.sqrt(T))
                nd1  = math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
                delta = (1.0 + math.erf(d1 / math.sqrt(2.0))) / 2.0
                vega  = current * nd1 * math.sqrt(T)
            except Exception:
                delta, vega, nd1 = 0.5, 0.0, 0.0

        if is_spread:
            delta *= 0.80

        # v9.0 #13: Theta-Decay (vereinfachte BS-Formel, r≈0)
        # theta_per_year = -current * N'(d1) * sigma / (2 * sqrt(T))
        # = -vega * sigma / (2 * T)
        try:
            theta_per_year = -(current * nd1 * iv) / (2.0 * math.sqrt(T))
            theta_daily    = theta_per_year / 365.0
            theta_daily_pct = abs(theta_daily) / cost if cost > 0 else 0.0
        except Exception:
            theta_daily     = 0.0
            theta_daily_pct = 0.0

        # Breakeven bei Expiry
        breakeven     = strike + ask if not is_spread else strike + cost
        breakeven_pct = (breakeven - current) / current if current > 0 else 0.0

        leverage      = current / cost if cost > 0 else 1.0
        expected_move = (target - current) / current if target > current else 0.0

        # v9.0 #3 / v10.0 #6: Probability-weighted ROI, kein unterer Clamp mehr
        mc_weight = min(0.95, mc_hit_rate)
        roi_delta = expected_move * delta * leverage * mc_weight

        # v10.0 #2: Event-typ-spezifischer IV-Crush + IV-Rank-Scaling
        # Basis-Rate aus EVENT_IV_CRUSH, dann skaliert mit IV-Rank:
        # iv_rank=0  → 0.50× (kaum IV, kaum Crush)
        # iv_rank=50 → 0.75× (mittlere IV)
        # iv_rank=100→ 1.00× (maximale IV, maximaler Crush)
        crush_rates  = EVENT_IV_CRUSH.get(catalyst_type, EVENT_IV_CRUSH["OTHER"])
        dte_key      = "short" if dte <= 60 else ("mid" if dte <= 149 else "long")
        iv_drop_base = crush_rates[dte_key]
        iv_rank_scale = 0.5 + (iv_rank / 200.0)

        # VIX Term Structure: Backwardation → erhöhter Crush (×1.20)
        #                     Contango      → geringerer Crush (×0.85, Markt ruhig)
        vix_structure = self._vix_ts.get("structure", "unknown") if self._vix_ts else "unknown"
        vix_crush_scale = 1.20 if vix_structure == "backwardation" else (
                          0.85 if vix_structure == "contango" else 1.00)

        iv_drop = iv_drop_base * iv_rank_scale * vix_crush_scale

        vega_loss = min((vega * iv * iv_drop) / cost, 0.50) if cost > 0 else 0.0
        roi_net   = roi_delta - (spread_pct * 2) - vega_loss
        passes    = roi_net >= min_roi

        def _safe_float(v):
            if isinstance(v, complex): return float(v.real)
            try: return float(v)
            except: return 0.0

        return {
            "roi_gross":         round(_safe_float(roi_delta), 4),
            "roi_net":           round(_safe_float(roi_net), 4),
            "spread_pct":        round(_safe_float(spread_pct), 4),
            "vega_loss":         round(_safe_float(vega_loss), 4),
            "theta_daily":       round(_safe_float(theta_daily), 4),
            "theta_daily_pct":   round(_safe_float(theta_daily_pct), 4),
            "delta":             round(_safe_float(delta), 4),
            "breakeven":         round(_safe_float(breakeven), 2),
            "breakeven_pct":     round(_safe_float(breakeven_pct), 4),
            "iv_drop_assumed":   _safe_float(iv_drop),
            "iv_drop_base":      _safe_float(iv_drop_base),
            "catalyst_type":     catalyst_type,
            "cost_basis":        round(float(cost), 4),
            "mc_weight":         round(mc_weight, 3),
            "is_spread":         is_spread,
            "passes_roi_gate":   passes,
            "min_roi_threshold": min_roi,
            "dte":               int(dte),
        }

    # ── Option Chain Abruf: Tradier Primary, yfinance Fallback ───────────────

    def _find_option_for_dte(
        self, ticker: str, strategy: str, current: float,
        dte_min: int, dte_max: int,
        t: Optional[object] = None,
    ) -> Optional[dict]:
        if self._use_tradier:
            result = self._find_option_tradier(ticker, strategy, current, dte_min, dte_max)
            if result is not None:
                return result
            log.debug(f"  [{ticker}] Tradier Chain leer → yfinance Fallback")

        return self._find_option_yfinance(ticker, strategy, current, dte_min, dte_max, t)

    def _find_option_tradier(
        self, ticker: str, strategy: str, current: float,
        dte_min: int, dte_max: int,
    ) -> Optional[dict]:
        try:
            all_dates = _tradier_expirations(ticker)
            if not all_dates:
                return None

            dates = [d for d in all_dates if dte_min <= self._days_to(d) <= dte_max]
            if not dates:
                return None

            best_expiry = dates[0]
            is_call     = "CALL" in strategy or "BULL" in strategy
            option_type = "call" if is_call else "put"

            raw = _tradier_chain(ticker, best_expiry)
            if not raw:
                return None

            opts = _tradier_chain_to_df(raw, option_type)
            if opts.empty:
                return None

            otm_max, itm_max = _strike_window(current, dte_min, dte_max)

            min_oi = max(
                getattr(getattr(cfg, "risk", None), "min_open_interest", 100), 50
            )
            filtered = opts[
                (opts["strike"] >= current * itm_max) &
                (opts["strike"] <= current * otm_max) &
                (opts["openInterest"] >= min_oi)
            ].copy()

            if filtered.empty:
                return None

            filtered["spread_ratio"] = (
                (filtered["ask"] - filtered["bid"]) /
                filtered["ask"].clip(lower=0.01)
            )
            filtered = filtered[filtered["spread_ratio"] <= cfg.risk.max_bid_ask_ratio]
            if filtered.empty:
                return None

            delta_min = getattr(getattr(cfg, "options", None), "delta_target_low",  0.50)
            delta_max = getattr(getattr(cfg, "options", None), "delta_target_high", 0.75)
            if "delta" in filtered.columns:
                delta_filtered = filtered[
                    (filtered["delta"].abs() >= delta_min) &
                    (filtered["delta"].abs() <= delta_max)
                ]
                if not delta_filtered.empty:
                    filtered = delta_filtered
                else:
                    log.debug(
                        f"  [{ticker}] Delta-Filter ({delta_min:.2f}–{delta_max:.2f}): "
                        f"kein Match → alle behalten"
                    )

            best = filtered.sort_values("openInterest", ascending=False).iloc[0]
            dte  = self._days_to(best_expiry)

            result = {
                "expiry":        best_expiry,
                "strike":        float(best["strike"]),
                "bid":           float(best["bid"]),
                "ask":           float(best["ask"]),
                "open_interest": int(best["openInterest"]),
                "implied_vol":   float(best["impliedVolatility"]),
                "spread_ratio":  round(float(best["spread_ratio"]), 4),
                "dte":           int(dte),
                "delta":         float(best.get("delta", 0)),
                "data_source":   "tradier",
            }

            log.info(
                f"  [{ticker}] Tradier Chain: expiry={best_expiry} "
                f"strike={result['strike']:.1f} IV={result['implied_vol']:.1%} "
                f"delta={result['delta']:.2f} OI={result['open_interest']}"
            )

            if "SPREAD" in strategy:
                spread_leg = self._find_spread_leg(opts, best["strike"])
                result["spread_leg"] = spread_leg
                if spread_leg:
                    result["net_debit"] = round(
                        result["ask"] - spread_leg.get("bid", 0), 2
                    )
                    if spread_leg.get("bid", 0) <= 0:
                        log.debug(f"  [{ticker}] Short-Leg hat keine Liquidität → kein Spread")
                        result.pop("spread_leg", None)
                        result.pop("net_debit", None)

            return result

        except Exception as e:
            log.debug(f"Tradier _find_option [{ticker}] {dte_min}-{dte_max}d: {e}")
            return None

    def _find_option_yfinance(
        self, ticker: str, strategy: str, current: float,
        dte_min: int, dte_max: int,
        t: Optional[object] = None,
    ) -> Optional[dict]:
        try:
            if t is None:
                t = yf.Ticker(ticker)
            dates = [
                d for d in (t.options or [])
                if dte_min <= self._days_to(d) <= dte_max
            ]
            if not dates:
                return None

            best_expiry = dates[0]
            chain       = t.option_chain(best_expiry)
            is_call     = "CALL" in strategy or "BULL" in strategy
            opts        = chain.calls if is_call else chain.puts

            otm_max, itm_max = _strike_window(current, dte_min, dte_max)

            filtered = opts[
                (opts["strike"] >= current * itm_max) &
                (opts["strike"] <= current * otm_max) &
                (opts["openInterest"] >= max(
                    getattr(getattr(cfg, "risk", None), "min_open_interest", 100), 50
                ))
            ].copy()

            if filtered.empty:
                return None

            filtered["spread_ratio"] = (
                (filtered["ask"] - filtered["bid"]) /
                filtered["ask"].clip(lower=0.01)
            )
            filtered = filtered[filtered["spread_ratio"] <= cfg.risk.max_bid_ask_ratio]
            if filtered.empty:
                return None

            best = filtered.sort_values("openInterest", ascending=False).iloc[0]
            dte  = self._days_to(best_expiry)

            result = {
                "expiry":        best_expiry,
                "strike":        float(best["strike"]),
                "bid":           float(best["bid"]),
                "ask":           float(best["ask"]),
                "open_interest": int(best["openInterest"]),
                "implied_vol":   float(best.get("impliedVolatility", 0.30)),
                "spread_ratio":  round(float(best["spread_ratio"]), 4),
                "dte":           int(dte),
                "data_source":   "yfinance",
            }

            if "SPREAD" in strategy:
                spread_leg = self._find_spread_leg(opts, best["strike"])
                result["spread_leg"] = spread_leg
                if spread_leg:
                    result["net_debit"] = round(
                        result["ask"] - spread_leg.get("bid", 0), 2
                    )
                    if spread_leg.get("bid", 0) <= 0:
                        log.debug(f"  [{ticker}] Short-Leg hat keine Liquidität → kein Spread")
                        result.pop("spread_leg", None)
                        result.pop("net_debit", None)

            return result

        except Exception as e:
            log.debug(f"yfinance _find_option [{ticker}] {dte_min}-{dte_max}d: {e}")
            return None

    def _find_spread_leg(self, opts: pd.DataFrame, long_strike: float) -> Optional[dict]:
        candidates = opts[
            (opts["strike"] >= long_strike * 1.05) &
            (opts["strike"] <= long_strike * 1.20)
        ]
        if candidates.empty:
            return None
        best = candidates.iloc[
            (candidates["strike"] - long_strike * 1.10).abs().argsort()
        ].iloc[0]
        return {"strike": float(best["strike"]),
                "bid": float(best["bid"]), "ask": float(best["ask"])}

    # ── ATM Straddle (Market-Implied Expected Move) ───────────────────────────

    def _get_atm_straddle(
        self, ticker: str, current: float, expiry: str, t=None
    ) -> Optional[float]:
        """
        Berechnet den ATM Straddle-Preis (Call Ask + Put Ask) für eine Expiry.
        Gibt ihn als Anteil des Aktienkurses zurück (= Market-Implied Expected Move).

        Beispiel: Straddle $10 bei Kurs $100 → implied_move = 0.10 (±10%).
        """
        # Tradier: Chain bereits für diese Expiry gecacht
        if self._use_tradier:
            try:
                raw = _tradier_chain(ticker, expiry)
                if raw:
                    atm_call = min(
                        (o for o in raw if o.get("option_type") == "call"),
                        key=lambda o: abs(float(o.get("strike", 1e9)) - current),
                        default=None,
                    )
                    atm_put = min(
                        (o for o in raw if o.get("option_type") == "put"),
                        key=lambda o: abs(float(o.get("strike", 1e9)) - current),
                        default=None,
                    )
                    if atm_call and atm_put:
                        ca = float(atm_call.get("ask", 0) or 0)
                        pa = float(atm_put.get("ask", 0) or 0)
                        if ca > 0 and pa > 0 and current > 0:
                            return (ca + pa) / current
            except Exception as e:
                log.debug(f"  [{ticker}] ATM Straddle Tradier Fehler: {e}")

        # yfinance Fallback
        try:
            if t is None:
                t = yf.Ticker(ticker)
            chain = t.option_chain(expiry)
            c = chain.calls.iloc[(chain.calls["strike"] - current).abs().argsort()].iloc[0]
            p = chain.puts.iloc[(chain.puts["strike"] - current).abs().argsort()].iloc[0]
            ca = float(c.get("ask", 0) or 0)
            pa = float(p.get("ask", 0) or 0)
            if ca > 0 and pa > 0 and current > 0:
                return (ca + pa) / current
        except Exception as e:
            log.debug(f"  [{ticker}] ATM Straddle yfinance Fehler: {e}")

        return None

    # ── IV-Rank ───────────────────────────────────────────────────────────────

    def _get_iv_rank(self, ticker: str, t: Optional[object] = None) -> float:
        try:
            if t is None:
                t = yf.Ticker(ticker)

            rv_score = 50.0
            info     = t.info
            current  = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)

            hist = t.history(period="1y")
            if not hist.empty and len(hist) >= 60:
                rets    = hist["Close"].pct_change().dropna()
                roll_rv = rets.rolling(21).std().dropna() * (252 ** 0.5)
                if len(roll_rv) >= 20:
                    rv_current = float(roll_rv.iloc[-1])
                    rv_min     = float(roll_rv.quantile(0.05))
                    rv_max     = float(roll_rv.quantile(0.95))
                    if rv_max > rv_min:
                        rv_score = ((rv_current - rv_min) / (rv_max - rv_min)) * 100
                        rv_score = max(0.0, min(100.0, rv_score))

            if current <= 0:
                return round(rv_score, 1)

            term_score = 20.0
            iv_pts     = self._get_term_structure_iv(ticker, current, t)

            if len(iv_pts) >= 2:
                iv_pts.sort()
                iv_short = iv_pts[0][1]
                iv_long  = iv_pts[-1][1]
                if iv_long > 0:
                    slope      = (iv_short / iv_long) - 1.0
                    term_score = max(0.0, min(80.0, (slope + 0.05) * 100))

            combined = round(rv_score * 0.80 + term_score * 0.20, 1)
            log.info(
                f"  [{ticker}] IV-Rank: rv={rv_score:.0f} term={term_score:.0f} "
                f"→ combined={combined:.0f} "
                f"({'SPREAD' if combined >= IV_SPREAD_GATE else 'LONG'})"
            )
            return combined

        except Exception:
            return 50.0

    def _get_term_structure_iv(
        self, ticker: str, current: float, t=None
    ) -> list[tuple[int, float]]:
        if self._use_tradier:
            iv_pts = self._term_structure_tradier(ticker, current)
            if len(iv_pts) >= 2:
                return iv_pts
            log.debug(f"  [{ticker}] Term-Structure Tradier unvollständig → yfinance")
        return self._term_structure_yfinance(ticker, current, t)

    def _term_structure_tradier(
        self, ticker: str, current: float
    ) -> list[tuple[int, float]]:
        iv_pts = []
        try:
            dates = _tradier_expirations(ticker)
            for d in dates[:3]:
                dte = self._days_to(d)
                if dte < 7:
                    continue
                raw = _tradier_chain(ticker, d)
                if not raw:
                    continue
                atm_ivs = []
                for o in raw:
                    if o.get("option_type") != "call":
                        continue
                    strike = float(o.get("strike", 0))
                    if not (current * 0.93 <= strike <= current * 1.07):
                        continue
                    greeks = o.get("greeks") or {}
                    iv = greeks.get("mid_iv") or greeks.get("smv_vol")
                    if iv and isinstance(iv, (int, float)) and iv > 0.05:
                        atm_ivs.append(float(iv))
                if atm_ivs:
                    iv_pts.append((dte, float(np.median(atm_ivs))))
        except Exception as e:
            log.debug(f"  [{ticker}] Term-Structure Tradier Fehler: {e}")
        return iv_pts

    def _term_structure_yfinance(
        self, ticker: str, current: float, t=None
    ) -> list[tuple[int, float]]:
        iv_pts = []
        try:
            if t is None:
                t = yf.Ticker(ticker)
            dates = t.options or []
            for d in dates[:3]:
                try:
                    dte = self._days_to(d)
                    if dte < 7:
                        continue
                    ch  = t.option_chain(d)
                    atm = ch.calls[
                        (ch.calls["strike"] >= current * 0.93) &
                        (ch.calls["strike"] <= current * 1.07) &
                        (ch.calls["impliedVolatility"] > 0.05)
                    ]
                    if not atm.empty:
                        iv_pts.append((dte, float(atm["impliedVolatility"].median())))
                except Exception:
                    continue
        except Exception as e:
            log.debug(f"  [{ticker}] Term-Structure yfinance Fehler: {e}")
        return iv_pts

    # ── Sektor-Momentum ───────────────────────────────────────────────────────

    def _sector_momentum_ok(self, s: dict, t=None) -> bool:
        ticker    = s.get("ticker", "")
        sector    = s.get("info", {}).get("sector", "default")
        direction = s.get("deep_analysis", {}).get("direction", "BULLISH")
        etf       = SECTOR_ETF.get(sector, SECTOR_ETF["default"])
        try:
            ticker_obj = t if t is not None else yf.Ticker(ticker)
            sh = ticker_obj.history(period="35d")
            eh = yf.Ticker(etf).history(period="35d")
            if sh.empty or eh.empty or len(sh) < 5:
                return True
            sr = float((sh["Close"].iloc[-1] - sh["Close"].iloc[0]) / sh["Close"].iloc[0])
            er = float((eh["Close"].iloc[-1] - eh["Close"].iloc[0]) / eh["Close"].iloc[0])
            rs = sr - er
            s["sector_momentum"] = {"etf": etf, "rel_strength": round(rs, 4)}
            return rs >= RELATIVE_STRENGTH_MIN if direction == "BULLISH" else rs <= -RELATIVE_STRENGTH_MIN
        except Exception:
            return True

    def _bear_case_ok(self, s: dict) -> bool:
        sev = s.get("deep_analysis", {}).get("bear_case_severity", 0)
        thr = getattr(getattr(cfg, "risk", None), "max_bear_case_severity", 8)
        if sev >= thr:
            log.info(f"  [{s['ticker']}] BEAR-CASE={sev} ≥ {thr} → blockiert")
            return False
        return True

    def _days_to(self, expiry_str: str) -> int:
        try:
            expiry = datetime.strptime(expiry_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            now    = datetime.now(timezone.utc)
            return max(0, (expiry - now).days)
        except Exception:
            return 0

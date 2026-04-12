"""
modules/data_validator.py – Daten-Validierung & Bid-Ask ROI

Priorität 3+6: Bid-Ask Strafzins + Alpha Vantage EPS Cross-Check

Verhindert:
  - Trades bei denen Transaktionskosten das Alpha auffressen
  - Claude-Halluzinationen aufgrund falscher yfinance-EPS-Daten
"""

from __future__ import annotations
import logging
import os
import time
from typing import Optional

import requests

log = logging.getLogger(__name__)

# Alpha Vantage Free Tier: 25 Requests/Tag, 5 Requests/Minute
_AV_BASE    = "https://www.alphavantage.co/query"
_AV_DELAY   = 12.5   # Sekunden zwischen Calls (5/min = 12s Pause)
_last_av_call: float = 0.0


# ── Bid-Ask ROI Kalkulation ───────────────────────────────────────────────────

def compute_option_roi(option: dict, simulation: dict) -> dict:
    """
    Berechnet den realistischen ROI nach Abzug des vollen Bid-Ask-Spreads.

    Formel:
        entry_cost  = ask_price (worst case: wir zahlen Ask)
        exit_price  = bid_price × (1 + expected_move) (vereinfacht)
        roi         = (exit_price - entry_cost) / entry_cost

    Alternativ vereinfacht:
        roi_gross   = expected_move × delta (aus simulation hit_rate)
        roi_net     = roi_gross - spread_cost_pct

    Args:
        option:     Dict mit bid, ask, last, implied_vol, strike
        simulation: Dict mit hit_rate, current_price, target_price

    Returns:
        {
            "roi_gross":     float,   # ROI vor Spread-Kosten
            "roi_net":       float,   # ROI nach Spread-Kosten
            "spread_pct":    float,   # Spread als % des Ask-Preises
            "spread_cost":   float,   # Absoluter Spread in USD
            "passes_roi_gate": bool,  # True wenn roi_net > min_roi
        }
    """
    bid     = option.get("bid", 0.0) or 0.0
    ask     = option.get("ask", 0.0) or 0.0
    last    = option.get("last", 0.0) or 0.0

    if ask <= 0:
        return _roi_no_data()

    # Spread-Kosten
    spread_abs = ask - bid
    spread_pct = spread_abs / ask if ask > 0 else 0.0

    # Entry: wir kaufen zu Ask
    entry_cost = ask

    # Erwarteter Exit-Preis basierend auf Simulation
    hit_rate     = simulation.get("hit_rate", 0.7)
    current      = simulation.get("current_price", 0.0)
    target       = simulation.get("target_price", 0.0)
    strike       = option.get("strike", current)

    if current > 0 and target > current and strike > 0:
        # Intrinsic Value bei Target-Price (vereinfacht, kein Greeks-Modell)
        expected_intrinsic = max(target - strike, 0)
        # Time value geschätzt als 20% des aktuellen Ask (verbleibt)
        expected_time_val  = ask * 0.20
        exit_estimate      = expected_intrinsic + expected_time_val
    else:
        # Fallback: Einfaches P&L basierend auf Hit-Rate
        exit_estimate = ask * (1 + hit_rate * 0.5)

    # ROI Berechnung
    roi_gross = (exit_estimate - entry_cost) / entry_cost if entry_cost > 0 else 0.0

    # Spread abziehen (Round-trip: einmal beim Kauf, einmal beim Verkauf)
    roi_net   = roi_gross - (spread_pct * 2)

    # Gate: Mindest-ROI nach Transaktionskosten
    min_roi   = float(os.getenv("MIN_OPTION_ROI", "0.15"))   # Default: 15%
    passes    = roi_net >= min_roi

    if not passes:
        log.info(
            f"    BID-ASK ROI GATE: roi_net={roi_net:.1%} < {min_roi:.0%} "
            f"(spread={spread_pct:.1%}) → Trade abgelehnt."
        )
    else:
        log.info(
            f"    ROI: gross={roi_gross:.1%} spread={spread_pct:.1%} "
            f"net={roi_net:.1%} ✅"
        )

    return {
        "roi_gross":       round(roi_gross, 4),
        "roi_net":         round(roi_net, 4),
        "spread_pct":      round(spread_pct, 4),
        "spread_cost":     round(spread_abs, 2),
        "entry_cost":      round(entry_cost, 2),
        "exit_estimate":   round(exit_estimate, 2),
        "passes_roi_gate": passes,
    }


def _roi_no_data() -> dict:
    return {
        "roi_gross":       0.0,
        "roi_net":         0.0,
        "spread_pct":      0.0,
        "spread_cost":     0.0,
        "entry_cost":      0.0,
        "exit_estimate":   0.0,
        "passes_roi_gate": False,
    }


# ── Alpha Vantage EPS Cross-Check ─────────────────────────────────────────────

def _av_rate_limit():
    """Respektiert Alpha Vantage Free Tier: max 5 Calls/Minute."""
    global _last_av_call
    elapsed = time.time() - _last_av_call
    if elapsed < _AV_DELAY:
        time.sleep(_AV_DELAY - elapsed)
    _last_av_call = time.time()


def fetch_eps_alphavantage(ticker: str) -> Optional[float]:
    """
    Ruft EPS via Alpha Vantage OVERVIEW Endpoint ab.
    Free Tier: 25 Calls/Tag.

    Gibt `EPS` (TTM) zurück oder None bei Fehler.
    """
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    if not api_key:
        return None

    _av_rate_limit()

    try:
        resp = requests.get(
            _AV_BASE,
            params={
                "function": "OVERVIEW",
                "symbol":   ticker,
                "apikey":   api_key,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        # Alpha Vantage gibt leeres Dict zurück wenn Limit erreicht
        if "Note" in data or "Information" in data:
            log.warning(f"Alpha Vantage Rate-Limit erreicht für {ticker}")
            return None

        eps_str = data.get("EPS", "")
        if eps_str and eps_str not in ("None", "-", ""):
            return float(eps_str)

        return None

    except Exception as e:
        log.debug(f"Alpha Vantage EPS Fehler für {ticker}: {e}")
        return None


def cross_check_eps(
    ticker: str,
    yfinance_eps: float,
    tolerance: float = 0.10,
) -> dict:
    """
    Cross-Checked EPS zwischen yfinance und Alpha Vantage.

    Args:
        ticker:        Aktien-Ticker
        yfinance_eps:  EPS aus yfinance (forwardEps oder trailingEps)
        tolerance:     Maximale erlaubte Abweichung (default: 10%)

    Returns:
        {
            "av_eps":          float or None,
            "yf_eps":          float,
            "deviation_pct":   float,
            "consistent":      bool,   # True wenn < tolerance abweichung
            "confidence":      str,    # "high" | "medium" | "low"
        }

    Wenn consistent=False → EPS-Daten unzuverlässig → Signal-Qualität niedriger
    """
    av_eps = fetch_eps_alphavantage(ticker)

    if av_eps is None:
        # Kein Cross-Check möglich → neutral
        return {
            "av_eps":        None,
            "yf_eps":        yfinance_eps,
            "deviation_pct": None,
            "consistent":    True,   # Benefit of the doubt
            "confidence":    "medium",
        }

    if yfinance_eps == 0:
        return {
            "av_eps":        av_eps,
            "yf_eps":        yfinance_eps,
            "deviation_pct": None,
            "consistent":    False,
            "confidence":    "low",
        }

    deviation = abs(av_eps - yfinance_eps) / abs(yfinance_eps)
    consistent = deviation <= tolerance

    confidence = "high" if deviation < 0.02 else ("medium" if consistent else "low")

    if not consistent:
        log.warning(
            f"  [{ticker}] EPS Cross-Check FAILED: "
            f"yfinance={yfinance_eps:.2f} vs AlphaVantage={av_eps:.2f} "
            f"Abweichung={deviation:.1%} > {tolerance:.0%} → Daten-Warnung."
        )
    else:
        log.info(
            f"  [{ticker}] EPS Cross-Check OK: "
            f"yf={yfinance_eps:.2f} av={av_eps:.2f} "
            f"Abw={deviation:.1%} conf={confidence}"
        )

    return {
        "av_eps":        round(av_eps, 4),
        "yf_eps":        round(yfinance_eps, 4),
        "deviation_pct": round(deviation, 4),
        "consistent":    consistent,
        "confidence":    confidence,
    }


def validate_candidate_data(candidate: dict) -> dict:
    """
    Führt alle Daten-Validierungen für einen Post-Prescreening-Kandidaten durch:
      1. EPS Cross-Check (yfinance vs Alpha Vantage)

    Fügt candidate["data_validation"] hinzu.
    Setzt candidate["data_confidence"] auf "high" | "medium" | "low".
    """
    info      = candidate.get("info", {})
    ticker    = candidate.get("ticker", "")
    yf_eps    = info.get("trailingEps") or info.get("forwardEps") or 0.0

    eps_check = cross_check_eps(ticker, yf_eps)

    candidate["data_validation"] = {
        "eps_cross_check": eps_check,
    }
    candidate["data_confidence"] = eps_check["confidence"]

    if not eps_check["consistent"]:
        log.warning(
            f"  [{ticker}] Daten-Inkonsistenz → Signal wird mit "
            f"reduziertem Vertrauen weitergeführt (nicht verworfen)."
        )

    return candidate

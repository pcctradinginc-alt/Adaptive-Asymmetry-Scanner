"""
Stufe 4: Normalisierter Mismatch-Score (Quant-Validierung)

Fixes:
  H-03: Negative Mismatch-Werte (Markt hat überreagiert) wurden als "weak"
        klassifiziert und weiterverarbeitet. Fix: expliziter Filter, der
        Überreaktionen vor der Simulation aussortiert.
  M-01: EPS-Drift-Bins nutzten hardcodierte Werte statt config.yaml.
        Fix: cfg.eps_drift.* Thresholds.
"""

import logging
import numpy as np
import yfinance as yf

from modules.config import cfg

log = logging.getLogger(__name__)


def _bin_impact(impact: float) -> str:
    if impact <= 4:
        return "low"
    if impact <= 7:
        return "mid"
    return "high"


def _bin_mismatch(mismatch: float) -> str:
    """
    FIX H-03: Negative Werte bedeuten Überreaktion (Markt hat mehr reagiert
    als die News rechtfertigen). Diese werden NICHT mehr als "weak" behandelt
    – sie werden durch den expliziten Filter in _score() bereits entfernt.
    Der verbleibende Wertebereich ist immer ≥ 0.
    """
    if mismatch < 3:
        return "weak"
    if mismatch <= 6:
        return "good"
    return "strong"


def _bin_eps_drift(drift: float) -> str:
    """FIX M-01: Thresholds aus config.yaml statt hartcodiert."""
    abs_drift = abs(drift)
    if abs_drift > cfg.eps_drift.massive_threshold:
        return "massive"
    if abs_drift > cfg.eps_drift.relevant_threshold:
        return "relevant"
    return "noise"


class MismatchScorer:

    def run(self, analyses: list[dict]) -> list[dict]:
        scored = []
        for a in analyses:
            result = self._score(a)
            if result:
                scored.append(result)
        return scored

    def _score(self, a: dict) -> dict | None:
        ticker = a["ticker"]
        da     = a.get("deep_analysis", {})
        impact = da.get("impact", 0)
        r_2d   = abs(a.get("price_move_48h", 0))

        sigma = self._compute_sigma(ticker)
        if sigma == 0:
            log.warning(f"  [{ticker}] σ30d = 0, übersprungen.")
            return None

        z_score  = r_2d / sigma
        mismatch = impact - (z_score * 5)

        # FIX H-03: Negative Mismatch explizit filtern.
        # Negatives Mismatch = Markt hat MEHR reagiert als die News rechtfertigen
        # = Überreaktion. Das ist das Gegenteil des gesuchten Signals (Underreaction).
        # Solche Ticker werden nicht weiter verarbeitet.
        if mismatch <= 0:
            log.info(
                f"  [{ticker}] Mismatch={mismatch:.2f} ≤ 0 "
                f"→ Markt hat überreagiert, gefiltert."
            )
            return None

        eps_drift_val = a.get("eps_drift", {}).get("drift", 0.0)

        features = {
            "impact":    impact,
            "surprise":  da.get("surprise", 0),
            "mismatch":  round(mismatch, 3),
            "z_score":   round(z_score, 3),
            "sigma_30d": round(sigma, 4),
            "eps_drift": round(eps_drift_val, 4),
            "bin_impact":    _bin_impact(impact),
            "bin_mismatch":  _bin_mismatch(mismatch),
            "bin_eps_drift": _bin_eps_drift(eps_drift_val),
        }

        log.info(
            f"  [{ticker}] Mismatch={mismatch:.2f} "
            f"Z={z_score:.2f} σ={sigma:.4f}"
        )

        return {**a, "features": features}

    def _compute_sigma(self, ticker: str) -> float:
        """30-Tages Standardabweichung der täglichen Returns."""
        try:
            hist = yf.Ticker(ticker).history(period="35d")
            if len(hist) < 10:
                return 0.0
            returns = hist["Close"].pct_change().dropna()
            return float(np.std(returns))
        except Exception as e:
            log.debug(f"Sigma-Berechnung Fehler für {ticker}: {e}")
            return 0.0

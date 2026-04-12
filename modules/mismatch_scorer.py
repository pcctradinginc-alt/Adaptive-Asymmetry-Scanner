"""
Stufe 4: Normalisierter Mismatch-Score (Quant-Validierung)
- σ30d: Standardabweichung der 30-Tages-Returns
- Z-Score der 48h-Bewegung: Z = |R_2d| / σ30d
- Mismatch = impact - (Z × 5)
- Hohes Mismatch = Underreaction = Alpha-Signal
"""

import logging
import numpy as np
import yfinance as yf

log = logging.getLogger(__name__)


def _bin_impact(impact: float) -> str:
    if impact <= 4:
        return "low"
    if impact <= 7:
        return "mid"
    return "high"


def _bin_mismatch(mismatch: float) -> str:
    if mismatch < 3:
        return "weak"
    if mismatch <= 6:
        return "good"
    return "strong"


def _bin_eps_drift(drift: float) -> str:
    if abs(drift) < 0.02:
        return "noise"
    if abs(drift) <= 0.10:
        return "relevant"
    return "massive"


class MismatchScorer:
    def run(self, analyses: list[dict]) -> list[dict]:
        scored = []
        for a in analyses:
            result = self._score(a)
            if result:
                scored.append(result)
        return scored

    def _score(self, a: dict) -> dict | None:
        ticker   = a["ticker"]
        da       = a.get("deep_analysis", {})
        impact   = da.get("impact", 0)
        r_2d     = abs(a.get("price_move_48h", 0))

        # σ30d berechnen
        sigma = self._compute_sigma(ticker)
        if sigma == 0:
            log.warning(f"  [{ticker}] σ30d = 0, übersprungen.")
            return None

        # Z-Score
        z_score = r_2d / sigma

        # Mismatch-Score
        mismatch = impact - (z_score * 5)

        eps_drift_val = a.get("eps_drift", {}).get("drift", 0.0)

        features = {
            "impact":    impact,
            "surprise":  da.get("surprise", 0),
            "mismatch":  round(mismatch, 3),
            "z_score":   round(z_score, 3),
            "sigma_30d": round(sigma, 4),
            "eps_drift": round(eps_drift_val, 4),
            # Bins für Quasi-ML
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

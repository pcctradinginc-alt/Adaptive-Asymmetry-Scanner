"""
Stufe 5: Normalisierter Mismatch-Score (Quant-Validierung)

Fixes:
  H-03: Negative Mismatch-Werte (Markt hat überreagiert) wurden als "weak"
        klassifiziert und weiterverarbeitet. Fix: expliziter Filter, der
        Überreaktionen vor der Simulation aussortiert.
  M-01: EPS-Drift-Bins nutzten hardcodierte Werte statt config.yaml.
        Fix: cfg.eps_drift.* Thresholds.

  v8.2 FIX-A: price_move_48h wurde NIRGENDS im Produktionscode gesetzt.
        r_2d war IMMER 0 → Z-Score IMMER 0 → Mismatch = Impact.
        Fix: Mismatch-Scorer berechnet 48h-Move selbst via yfinance.

  v8.2 FIX-B: EPS-Drift wurde unter falschem Key gelesen.
        a.get("eps_drift", {}).get("drift", 0.0) → Feld existiert nie.
        Korrekter Pfad: a["data_validation"]["eps_cross_check"]["deviation_pct"]
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

        # ── FIX v8.2-A: 48h-Move selbst berechnen ────────────────────────────
        # Vorher: r_2d = abs(a.get("price_move_48h", 0))
        # Problem: "price_move_48h" wurde NIRGENDS im Produktionscode gesetzt.
        #          Nur in Tests als Mock-Daten vorhanden.
        #          → r_2d war IMMER 0 → Z-Score IMMER 0 → Mismatch = Impact.
        # Jetzt:   Eigene Berechnung, gleiche Logik wie deep_analysis v8.2.
        r_2d = abs(self._compute_48h_move(ticker))

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
                f"(Impact={impact}, Z={z_score:.2f}, 48h-Move={r_2d:.3f}) "
                f"→ Markt hat überreagiert, gefiltert."
            )
            return None

        # ── FIX v8.2-B: EPS-Drift aus korrektem Pfad lesen ───────────────────
        # Vorher: eps_drift_val = a.get("eps_drift", {}).get("drift", 0.0)
        # Problem: Feld "eps_drift" mit Subkey "drift" existiert nie.
        #          data_validator.py schreibt unter:
        #          candidate["data_validation"]["eps_cross_check"]["deviation_pct"]
        # Jetzt:   Korrekter Zugriffspfad.
        eps_check     = a.get("data_validation", {}).get("eps_cross_check", {})
        eps_drift_val = eps_check.get("deviation_pct", 0.0) or 0.0

        features = {
            "impact":        impact,
            "surprise":      da.get("surprise", 0),
            "mismatch":      round(mismatch, 3),
            "z_score":       round(z_score, 3),
            "sigma_30d":     round(sigma, 4),
            "price_move_48h": round(r_2d, 4),   # NEU: für Reports & Debugging
            "eps_drift":     round(eps_drift_val, 4),
            "bin_impact":    _bin_impact(impact),
            "bin_mismatch":  _bin_mismatch(mismatch),
            "bin_eps_drift": _bin_eps_drift(eps_drift_val),
        }

        log.info(
            f"  [{ticker}] Mismatch={mismatch:.2f} "
            f"Z={z_score:.2f} σ={sigma:.4f} "
            f"48h-Move={r_2d:.3f} EPS-Drift={eps_drift_val:.4f}"
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

    def _compute_48h_move(self, ticker: str) -> float:
        """
        Berechnet die Preisbewegung der letzten 2 vollen Handelstage.

        Nutzt period='10d' und vergleicht volle Handelstage:
          close[-2] = gestriger Close (letzter abgeschlossener Tag)
          close[-4] = vor-3-Tage-Close (48h-Fenster)

        Gleiche Logik wie deep_analysis.py v8.2 _get_48h_move().
        """
        try:
            hist  = yf.Ticker(ticker).history(period="10d")
            close = hist["Close"]
            if hasattr(close, "iloc"):
                close = close.squeeze()
            if len(close) < 5:
                return 0.0
            return float((close.iloc[-2] - close.iloc[-4]) / close.iloc[-4])
        except Exception:
            return 0.0

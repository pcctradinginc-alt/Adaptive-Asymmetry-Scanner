"""
Stufe 6: Adaptive Quasi-ML Scoring (Selbstlern-Kern)
- Feature-Binning: impact, mismatch, eps_drift
- Scoring basierend auf historischen avg_returns pro Bin
- FinalScore = Σ(Bin_Avg_Return_i × Current_Weight_i)
- Sortierung nach FinalScore → Top-Signal(e) des Tages
"""

import logging

log = logging.getLogger(__name__)

# Mindest-Daten für ML-Scoring (darunter: Fallback auf einfache Gewichtung)
MIN_BIN_COUNT = 3


class QuasiML:
    def __init__(self, history: dict):
        self.history      = history
        self.feat_stats   = history.get("feature_stats", {})
        self.weights      = history.get("model_weights", {
            "impact":    0.35,
            "mismatch":  0.45,
            "eps_drift": 0.20,
        })

    def run(self, simulated: list[dict]) -> list[dict]:
        scored = []
        for s in simulated:
            final_score = self._compute_final_score(s)
            scored.append({**s, "final_score": round(final_score, 4)})
            log.info(
                f"  [{s['ticker']}] FinalScore={final_score:.4f} "
                f"(Weights: {self.weights})"
            )

        # Absteigend nach FinalScore
        scored.sort(key=lambda x: x["final_score"], reverse=True)
        return scored

    def _compute_final_score(self, s: dict) -> float:
        features = s.get("features", {})
        total    = 0.0

        feature_bins = {
            "impact":    features.get("bin_impact"),
            "mismatch":  features.get("bin_mismatch"),
            "eps_drift": features.get("bin_eps_drift"),
        }

        for feat_name, bin_label in feature_bins.items():
            if bin_label is None:
                continue
            weight   = self.weights.get(feat_name, 0.0)
            avg_ret  = self._get_bin_avg_return(feat_name, bin_label)
            total   += avg_ret * weight

        # Wenn noch keine Historydaten → fallback auf normalisierte Features
        if total == 0.0:
            total = self._fallback_score(features)

        return total

    def _get_bin_avg_return(self, feature: str, bin_label: str) -> float:
        try:
            stats = self.feat_stats.get(feature, {}).get(bin_label, {})
            count = stats.get("count", 0)
            if count < MIN_BIN_COUNT:
                return self._prior_return(feature, bin_label)
            return stats.get("avg_return", 0.0)
        except Exception:
            return 0.0

    def _prior_return(self, feature: str, bin_label: str) -> float:
        """Bayesianischer Prior bevor genug Daten vorhanden sind."""
        priors = {
            "impact":    {"low": -0.02, "mid": 0.04, "high": 0.12},
            "mismatch":  {"weak": -0.01, "good": 0.05, "strong": 0.15},
            "eps_drift": {"noise": 0.00, "relevant": 0.04, "massive": 0.10},
        }
        return priors.get(feature, {}).get(bin_label, 0.0)

    def _fallback_score(self, features: dict) -> float:
        """Einfacher Fallback ohne Historydaten."""
        impact   = features.get("impact", 0) / 10.0
        mismatch = min(features.get("mismatch", 0) / 10.0, 1.0)
        drift    = min(abs(features.get("eps_drift", 0)), 0.2) / 0.2
        return (
            impact   * self.weights.get("impact",    0.35) +
            mismatch * self.weights.get("mismatch",  0.45) +
            drift    * self.weights.get("eps_drift", 0.20)
        )

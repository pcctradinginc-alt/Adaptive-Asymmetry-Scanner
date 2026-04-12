"""
Tests für den Adaptive Asymmetry-Scanner
Ausführen: pytest tests/ -v
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def empty_history():
    return {
        "feature_stats": {
            "impact":    {"low": {"count": 0, "avg_return": 0.0}, "mid": {"count": 0, "avg_return": 0.0}, "high": {"count": 0, "avg_return": 0.0}},
            "mismatch":  {"weak": {"count": 0, "avg_return": 0.0}, "good": {"count": 0, "avg_return": 0.0}, "strong": {"count": 0, "avg_return": 0.0}},
            "eps_drift": {"noise": {"count": 0, "avg_return": 0.0}, "relevant": {"count": 0, "avg_return": 0.0}, "massive": {"count": 0, "avg_return": 0.0}},
        },
        "active_trades": [],
        "closed_trades": [],
        "model_weights": {"impact": 0.35, "mismatch": 0.45, "eps_drift": 0.20},
    }


@pytest.fixture
def sample_candidate():
    return {
        "ticker": "AAPL",
        "info": {
            "marketCap": 3_000_000_000_000,
            "averageVolume10days": 50_000_000,
            "forwardEps": 7.5,
            "recommendationMean": 1.8,
            "sector": "Technology",
        },
        "eps_drift": {"drift": 0.12, "current_eps": 7.5, "stored_eps": 6.7, "drift_weight": "massive", "rec_mean": 1.8},
        "news": ["Apple announces revolutionary AI chip", "iPhone sales beat estimates by 15%"],
        "prescreen_reason": "Technologischer Durchbruch – neue Produktkategorie",
    }


@pytest.fixture
def sample_analysis(sample_candidate):
    return {
        **sample_candidate,
        "price_move_48h": 0.015,
        "deep_analysis": {
            "ticker":                  "AAPL",
            "impact":                  8,
            "surprise":                7,
            "mispricing_logic":        "Markt unterschätzt langfristige Margenwirkung des AI-Chips.",
            "catalyst":                "Earnings Q2 2026 – 14. August",
            "time_to_materialization": "2-3 Monate",
            "bear_case":               "Makro-Abschwung könnte Consumer-Spending drücken.",
            "bear_case_severity":      4,
            "direction":               "BULLISH",
        },
    }


# ── Mismatch Scorer ───────────────────────────────────────────────────────────

class TestMismatchScorer:
    def test_high_impact_low_move_gives_high_mismatch(self):
        from modules.mismatch_scorer import MismatchScorer
        scorer = MismatchScorer()

        with patch.object(scorer, "_compute_sigma", return_value=0.02):
            candidate = {
                "ticker": "AAPL",
                "info": {},
                "eps_drift": {"drift": 0.12},
                "news": [],
                "deep_analysis": {"impact": 9, "surprise": 8, "direction": "BULLISH"},
                "price_move_48h": 0.01,   # Kleine 48h-Bewegung
            }
            result = scorer._score(candidate)

        assert result is not None
        mismatch = result["features"]["mismatch"]
        assert mismatch > 3, f"Erwartetes hohes Mismatch, aber: {mismatch}"
        assert result["features"]["bin_mismatch"] in ("good", "strong")

    def test_low_impact_high_move_gives_low_mismatch(self):
        from modules.mismatch_scorer import MismatchScorer
        scorer = MismatchScorer()

        with patch.object(scorer, "_compute_sigma", return_value=0.02):
            candidate = {
                "ticker": "MSFT",
                "info": {},
                "eps_drift": {"drift": 0.01},
                "news": [],
                "deep_analysis": {"impact": 3, "surprise": 2, "direction": "BULLISH"},
                "price_move_48h": 0.08,   # Große 48h-Bewegung → Markt hat reagiert
            }
            result = scorer._score(candidate)

        assert result is not None
        mismatch = result["features"]["mismatch"]
        assert mismatch < 3, f"Erwartetes niedriges Mismatch, aber: {mismatch}"

    def test_zero_sigma_returns_none(self):
        from modules.mismatch_scorer import MismatchScorer
        scorer = MismatchScorer()

        with patch.object(scorer, "_compute_sigma", return_value=0.0):
            result = scorer._score({"ticker": "X", "deep_analysis": {"impact": 5}, "price_move_48h": 0.0, "eps_drift": {"drift": 0}, "info": {}, "news": []})
        assert result is None


# ── Quasi-ML ─────────────────────────────────────────────────────────────────

class TestQuasiML:
    def test_fallback_scoring_without_history(self, empty_history):
        from modules.quasi_ml import QuasiML
        qml = QuasiML(history=empty_history)

        signal = {
            "ticker": "NVDA",
            "features": {
                "impact": 9, "surprise": 8,
                "mismatch": 7.0, "z_score": 0.4, "sigma_30d": 0.02, "eps_drift": 0.15,
                "bin_impact": "high", "bin_mismatch": "strong", "bin_eps_drift": "massive",
            },
            "deep_analysis": {"direction": "BULLISH"},
            "simulation": {"hit_rate": 0.82},
        }
        result = qml.run([signal])
        assert len(result) == 1
        assert result[0]["final_score"] > 0

    def test_signals_sorted_by_score(self, empty_history):
        from modules.quasi_ml import QuasiML
        qml = QuasiML(history=empty_history)

        low = {"ticker": "A", "features": {"impact": 2, "mismatch": 1, "eps_drift": 0, "bin_impact": "low", "bin_mismatch": "weak", "bin_eps_drift": "noise"}, "deep_analysis": {}, "simulation": {}}
        high = {"ticker": "B", "features": {"impact": 9, "mismatch": 8, "eps_drift": 0.15, "bin_impact": "high", "bin_mismatch": "strong", "bin_eps_drift": "massive"}, "deep_analysis": {}, "simulation": {}}

        result = qml.run([low, high])
        assert result[0]["ticker"] == "B"


# ── Risk Gates ────────────────────────────────────────────────────────────────

class TestRiskGates:
    def test_vix_below_threshold_passes(self):
        from modules.risk_gates import RiskGates
        gates = RiskGates()
        with patch.object(gates, "_get_vix", return_value=20.0):
            assert gates.global_ok() is True

    def test_vix_above_threshold_blocks(self):
        from modules.risk_gates import RiskGates
        gates = RiskGates()
        with patch.object(gates, "_get_vix", return_value=40.0):
            assert gates.global_ok() is False


# ── Mirofish Simulation ───────────────────────────────────────────────────────

class TestMirofishSimulation:
    def test_strong_signal_passes_gate(self):
        from modules.mirofish_simulation import MirofishSimulation
        sim = MirofishSimulation()

        candidate = {
            "ticker": "NVDA",
            "features": {"mismatch": 8.0, "impact": 9},
            "deep_analysis": {
                "direction": "BULLISH",
                "time_to_materialization": "2-3 Monate",
            },
        }
        with patch.object(sim, "_get_market_params", return_value=(0.02, 500.0, "Technology")):
            result = sim._simulate(candidate)

        # Mit hohem Mismatch-Drift sollte die Simulation passieren
        assert result is not None
        assert result["simulation"]["hit_rate"] >= 0.70

    def test_zero_price_returns_none(self):
        from modules.mirofish_simulation import MirofishSimulation
        sim = MirofishSimulation()

        candidate = {
            "ticker": "BROKEN",
            "features": {"mismatch": 5.0, "impact": 5},
            "deep_analysis": {"direction": "BULLISH", "time_to_materialization": "2-3 Monate"},
        }
        with patch.object(sim, "_get_market_params", return_value=(0.02, 0.0, "default")):
            result = sim._simulate(candidate)
        assert result is None


# ── Feedback Loop ─────────────────────────────────────────────────────────────

class TestFeedbackLoop:
    def test_bin_update_running_average(self):
        import feedback
        stats = {}
        feedback.update_bin(stats, "impact", "high", 0.20)
        feedback.update_bin(stats, "impact", "high", 0.10)
        assert stats["impact"]["high"]["count"] == 2
        assert abs(stats["impact"]["high"]["avg_return"] - 0.15) < 1e-6

    def test_bin_to_num_mapping(self):
        import feedback
        assert feedback._bin_to_num("impact",   "high")     == 1.0
        assert feedback._bin_to_num("mismatch", "weak")     == 0.0
        assert feedback._bin_to_num("eps_drift","relevant") == 0.5

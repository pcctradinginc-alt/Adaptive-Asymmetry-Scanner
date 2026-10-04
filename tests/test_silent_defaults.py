"""Regression Audit 2026-09-29: fehlende Daten dürfen nicht als gültiger
Wert (0-Move, neutrales Sentiment) in Scores eingehen."""
from __future__ import annotations

from unittest.mock import patch

from modules import finbert_sentiment as fb
from modules import sentiment_tracker as st
from modules.mismatch_scorer import MismatchScorer


def _analysis(ticker="ZZZ", impact=8):
    return {"ticker": ticker, "deep_analysis": {"impact": impact, "surprise": 5}}


def test_missing_price_history_is_rejected_not_scored_as_zero_move():
    sc = MismatchScorer()
    with patch.object(sc, "_compute_sigma", return_value=0.02), \
         patch.object(sc, "_compute_48h_move", return_value=None):
        assert sc.run([_analysis()]) == []
    assert sc.data_missing == ["ZZZ"]


def test_yfinance_error_returns_none():
    sc = MismatchScorer()
    with patch("modules.mismatch_scorer.yf.Ticker", side_effect=RuntimeError("down")):
        assert sc._compute_48h_move("ZZZ") is None


def test_research_features_signed_and_scaled():
    sc = MismatchScorer()
    with patch.object(sc, "_compute_sigma", return_value=0.02), \
         patch.object(sc, "_compute_48h_move", return_value=-0.01):
        out = sc.run([_analysis()])[0]["features"]
    assert out["price_move_48h_signed"] == -0.01 and out["price_move_48h"] == 0.01
    assert out["z_score"] == 0.5 and out["z_score_2d_scaled"] == round(0.01 / (0.02 * 2 ** 0.5), 3)


def test_finbert_fallback_is_flagged():
    assert fb.score_headlines([])["sentiment_status"] == "no_headlines"
    with patch.object(fb, "_load_model", return_value=False):
        r = fb.score_headlines(["x"])
    assert r["sentiment_score"] == 0.0 and r["sentiment_status"] == "model_unavailable"


def test_fallback_sentiment_not_written_to_history():
    h = {}
    st.enrich_with_sentiment_drift({"ticker": "ZZZ", "news": ["a"],
                                    "features": {"sentiment_score": 0.0,
                                                 "sentiment_status": "model_unavailable"}}, h)
    assert not h.get("sentiment_history", {}).get("ZZZ")
    st.enrich_with_sentiment_drift({"ticker": "ZZZ", "news": ["a"],
                                    "features": {"sentiment_score": 0.4, "sentiment_status": "ok"}}, h)
    assert h["sentiment_history"]["ZZZ"][-1]["score"] == 0.4


def test_deep_analysis_48h_move_missing_is_none():
    from modules.deep_analysis import DeepAnalysis
    da = DeepAnalysis.__new__(DeepAnalysis)
    with patch("modules.deep_analysis.yf.Ticker", side_effect=RuntimeError("down")):
        assert da._get_48h_move("ZZZ") is None


def _hist(n_days, rho_sign=1.0, seed=3):
    import random
    rnd = random.Random(seed)
    closed = []
    for d in range(n_days):
        imp = rnd.choice(["low", "mid", "high"])
        base = {"low": 0.0, "mid": 0.5, "high": 1.0}[imp]
        closed.append({"entry_date": f"2026-{1 + d // 28:02d}-{1 + d % 28:02d}",
                       "outcome": rho_sign * base + rnd.gauss(0, 0.2),
                       "features": {"bin_impact": imp, "bin_mismatch": rnd.choice(["weak", "good", "strong"]),
                                    "bin_eps_drift": "noise"}, "outcome_method": "option_quote"})
    return {"closed_trades": closed, "model_weights": {"impact": 0.35, "mismatch": 0.45, "eps_drift": 0.20}}


def test_pearson_weights_ignore_weak_noise_correlations():
    import feedback
    h = _hist(12)                       # zu wenige Tage -> keine Anpassung trotz echter Korrelation
    assert feedback.compute_pearson_weights(h) == {"impact": 0.35, "mismatch": 0.45, "eps_drift": 0.2}
    strong = feedback.compute_pearson_weights(_hist(120))
    assert strong["impact"] > 0.35      # belastbare Evidenz -> Gewicht steigt (langsam)


def test_pearson_weights_unchanged_on_current_history():
    import json
    import feedback
    h = json.load(open("outputs/history.json"))
    assert feedback.compute_pearson_weights(h) == {"impact": 0.35, "mismatch": 0.45, "eps_drift": 0.2}


def test_mc_skips_when_price_history_missing():
    import modules.mirofish_simulation as ms
    ms._get_hist_params.cache_clear()
    with patch("modules.mirofish_simulation.yf.download", side_effect=RuntimeError("down")):
        ms._get_hist_params("ZZZ")
    assert ms.PARAM_SOURCE["ZZZ"] == "default_error"
    cand = {"ticker": "ZZZ", "current_price": 100.0, "deep_analysis": {"impact": 7, "surprise": 5}}
    assert ms.MirofishSimulation().run_for_dte(cand, days_to_expiry=30) is None
    ms._get_hist_params.cache_clear()


def test_iv_rank_components_flag_unmeasured_history():
    from modules.options_designer import compute_iv_rank_components
    assert compute_iv_rank_components([100.0] * 10, [])["rv_measured"] is False
    closes = [100 * (1 + 0.01 * ((i * 7919) % 13 - 6) / 6) for i in range(260)]
    assert compute_iv_rank_components(closes, [])["rv_measured"] is True


def test_iv_rank_error_is_none_not_50():
    from modules.options_designer import OptionsDesigner
    od = OptionsDesigner.__new__(OptionsDesigner)

    class _Boom:
        @property
        def info(self):
            raise RuntimeError("down")
    assert od._get_iv_rank("ZZZ", _Boom()) is None


def test_earnings_unknown_is_flagged_not_silent(caplog):
    from modules import alpha_sources as al
    with patch.object(al, "get_earnings_date_finnhub", return_value=None), \
         patch("yfinance.Ticker", side_effect=RuntimeError("down")):
        caplog.set_level("WARNING")
        assert al.has_earnings_within_days("ZZZ", use_finnhub=False) == (False, None)
    assert "unbekannt" in caplog.text

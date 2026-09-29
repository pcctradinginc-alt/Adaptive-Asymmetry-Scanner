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

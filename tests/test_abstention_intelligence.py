"""Abstention Intelligence: Risikovektor je Champion-Trade – nur Messung, fehlend bleibt None."""
from __future__ import annotations

import pytest

from modules import abstention_intelligence as ai

CAL = {">=0.75": {"n": 29, "win_rate": 0.414}, "0.65-0.75": {"n": 27, "win_rate": 0.222},
       "<0.55": {"n": 1, "win_rate": 0.0}}


def _extras(**kw):
    e = {"mc_calibration": CAL, "data_quality": 0.9, "drift_level": "MODERATE", "share_deteriorating": 0.333,
         "disagreement_dist": [i / 100 for i in range(100)]}
    e.update(kw)
    return e


def _p(score=60, mc=0.80, roi=0.30, hurdle=0.07):
    return {"trade_score": {"total": score}, "mc_hit_rate": mc, "roi_analysis": {"roi_net": roi, "min_roi_threshold": hurdle}}


def test_components_from_measured_inputs():
    env = {"champion_probability": 0.80, "ml_disagreement_sd": 0.90, "blind_spot_sector_match": 1}
    v = ai.risk_vector(_p(), env, {}, _extras(), trade_score_min=55, mc_threshold=0.5)
    assert v["p_model_wrong"] == pytest.approx(0.586)          # 1 - realisierte Win Rate Band >=0.75
    assert v["data_quality"] == pytest.approx(0.1)
    assert v["regime_mismatch"] == pytest.approx(2 / 3, abs=1e-3)
    assert v["unknown_risk"] == 1.0 and v["model_disagreement"] == pytest.approx(0.91)
    assert v["alpha_decay"] == pytest.approx(0.333)
    assert v["counterfactual_fragility"] == pytest.approx(1 / 3, abs=1e-3)   # Score 60 vs 55 -> knapp
    assert v["n_known"] == 7 and 0 < v["abstain_score"] < 1


def test_missing_never_zero():
    v = ai.risk_vector({}, {}, {}, _extras(mc_calibration=None, data_quality=None, drift_level=None,
                                           share_deteriorating=None, disagreement_dist=[]))
    assert all(v[k] is None for k in ai.COMPONENTS)
    assert v["abstain_score"] is None and v["n_known"] == 0


def test_thin_calibration_band_is_unknown():
    v = ai.risk_vector(_p(mc=0.5), {"champion_probability": 0.5}, {}, _extras())
    assert v["p_model_wrong"] is None                           # Band <0.55 nur n=1


def test_fragility_only_counts_passed_gates():
    assert ai.fragility(_p(score=90, mc=0.9, roi=0.5), 55, 0.5) == 0.0
    assert ai.fragility(_p(score=56, mc=0.52, roi=0.075), 55, 0.5) == 1.0
    assert ai.fragility({}, 55, 0.5) is None


def test_as_features_prefix():
    f = ai.as_features({"p_model_wrong": 0.5, "abstain_score": 0.4})
    assert f["risk_p_model_wrong"] == 0.5 and f["risk_abstain_score"] == 0.4 and f["risk_unknown_risk"] is None


def test_adapter_records_risk_vector_without_changing_decisions(tmp_path):
    from modules import production_intelligence_adapter as pia
    props = [{"ticker": "AAA", "features": {"impact": 5}, "trade_score": {"total": 60},
              "simulation": {"hit_rate": 0.8}, "sector": "Technology"}]
    ctx = {"safe_mode_active": 0, "blind_spot_sectors": ["Energy"], "ml_cards": {}}
    kept, blocked, recs = pia.apply_to_proposals(props, vix=18, today="2026-10-06", context=ctx, contracts=[],
                                                 state_path=tmp_path / "s.json", registry=tmp_path / "r.jsonl",
                                                 transitions=tmp_path / "t.jsonl", ledger_dir=tmp_path,
                                                 policy={}, trade_score_min=55, mc_threshold=0.5)
    assert [p["ticker"] for p in kept] == ["AAA"] and not blocked
    rv = recs[0]["risk_vector"]
    assert rv["unknown_risk"] == 0.0 and rv["n_known"] >= 1
    assert kept[0]["features"]["risk_unknown_risk"] == 0.0     # -> Trade-Record -> abstention_proposals


def test_abstention_proposals_consider_risk_features():
    from modules import abstention_proposals as ap
    assert {"risk_abstain_score", "risk_counterfactual_fragility", "risk_p_model_wrong"} <= set(ap.FEATURES)

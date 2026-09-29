"""Meta-Cognition: Alpha-Decay/CUSUM, Research-Value, Safe-Mode-Auslöser,
metrisch begründeter Zustand. Kein Netz."""
from __future__ import annotations

import numpy as np
import pandas as pd

from modules import meta_cognition as mc


def _ic(first, second, n=120, seed=0):
    rnd = np.random.default_rng(seed)
    v = np.r_[rnd.normal(first, 0.05, n // 2), rnd.normal(second, 0.05, n - n // 2)]
    return pd.Series(v, index=pd.date_range("2018-01-05", periods=n, freq="W-FRI"))


def test_decay_and_structural_break_detected():
    d = mc.decay_profile(_ic(0.08, -0.02))
    assert d["status"] in ("decaying", "break") and d["structural_break"]["break"]
    s = mc.decay_profile(_ic(0.03, 0.03, seed=1))
    assert s["status"] == "stable"
    assert mc.decay_profile(_ic(0.1, 0.1, n=30))["status"] == "insufficient_history"


def test_research_value_efficiency():
    hyp = {"hypotheses": {"A": {"source": "literature", "canonical_status": "ACCEPTED", "walk_forward": {"base": {"mean": 0.004}}},
                          "B": {"source": "literature", "canonical_status": "REJECTED"},
                          **{f"C{i}": {"source": "human", "canonical_status": "REJECTED"} for i in range(6)}},
           "discovery": {"n_tested": 117, "n_survivors": 0}}
    rv = mc.research_value(hyp, {}, {})
    assert rv["literature"]["research_efficiency"] == "HIGH" and rv["human"]["research_efficiency"] == "LOW"
    assert rv["discovery_engine"]["experiments"] == 117


def test_safe_mode_triggers():
    meta = {"drift": {"feature_drift_flag": True, "feature_drift": {"tnx": {"out_of_range": True}}},
            "model_intelligence": {"a": {"trend": "deteriorating"}, "b": {"trend": "stable"}},
            "disagreement": {"current_level": "NORMAL"}}
    sm = mc.safe_mode(meta, {"current": {"uncertainty": 0.3}}, {"x": {"status": "PASS"}}, [0.01] * 12,
                      {"calibration": {"interval_calibrated": True}})
    assert sm["active"] and any("DRIFT" in r for r in sm["reasons"])
    calm = mc.safe_mode({"drift": {}, "model_intelligence": {"a": {"trend": "stable"}}}, {"current": {"uncertainty": 0.2}},
                        {}, None, {"calibration": {"interval_calibrated": True}})
    assert not calm["active"]
    bad = mc.safe_mode({}, {}, {}, [-0.05 + 0.001 * i for i in range(12)], {})
    assert any("BREAKDOWN" in r for r in bad["reasons"])


def test_machine_state_only_metric_statements():
    meta = {"approaches": {"static_equal": {"metrics": {"expectancy": 0.003, "sharpe": 0.28, "hit_rate": 0.47, "ece": 0.04},
                                            "by_regime": {"vix_ge_20": {"expectancy": 0.012}}}},
            "calibration_buckets": {"static_equal": [{"bucket": "55–60 %", "n": 7868, "predicted": 0.55, "win_rate": 0.42,
                                                      "flag": "overconfident"}]},
            "model_intelligence": {"m": {"trend": "deteriorating", "prior_ic": 0.07, "recent_ic": -0.1, "trend_t": -2.6,
                                         "contribution": -0.001}}}
    st = mc.machine_state({"meta": meta, "ml": {"calibration": {"interval_calibrated": True, "coverage": 0.65}}},
                          {}, {}, {}, {"active": False, "reasons": []})
    assert any("Überkonfident" in x for x in st["where_are_we_systematically_wrong"])
    assert any("m:" in x for x in st["which_models_are_redundant"])
    assert st["self_assessment"]["overall_calibration"] == "GOOD"
    assert mc.render_md(st)

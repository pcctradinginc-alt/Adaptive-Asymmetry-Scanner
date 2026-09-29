"""Nächste Intelligenz-Stufe: Counterfactual/Stress, Blind Spots, Decision
Intelligence, Gesamtvalidierung A–G mit Abstinenz, Ablation und Gate.
Synthetisch, kein Netz; Protokoll-Pin."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_ml_research as T  # noqa: E402

from modules import blind_spots as bs  # noqa: E402
from modules import counterfactual as cf  # noqa: E402
from modules import decision_intel as di  # noqa: E402
from modules import meta_learning as meta  # noqa: E402
from modules import next_intelligence as ni  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EXPECTED_SHA = "5b4c3c8e1dd5c058074066c455f49cb8b72e8b837188fbc9d979752b6f13a4ad"


def test_next_protocol_hash_pinned():
    h = hashlib.sha256((ROOT / "config" / "next_protocol.yaml").read_bytes()).hexdigest()
    assert h == EXPECTED_SHA, "config/next_protocol.yaml geändert – nur strenger zulässig"


def test_apply_case_shifts_and_flips():
    df = pd.DataFrame({"vix": [15.0], "vix_chg_21": [0.0], "mom_12_1": [0.3], "vol_60": [-0.2], "tnx": [3.0]})
    assert cf.apply_case(df, "vol_spike")["vix"].iloc[0] == 25.0
    assert cf.apply_case(df, "rate_shock")["tnx"].iloc[0] == 4.0
    assert cf.apply_case(df, "momentum_reversal")["mom_12_1"].iloc[0] == -0.3
    assert df["vix"].iloc[0] == 15.0                                   # Original unverändert


def test_fragility_flags_single_assumption():
    df = pd.DataFrame({"date": ["d"] * 3, "base": [0.95, 0.95, 0.5], "a": [0.96, 0.5, 0.5], "b": [0.97, 0.93, 0.4]})
    fr = cf.fragility(df, "base", {"A": "a", "B": "b"})
    assert list(fr["cf_fragile"]) == [False, True, False]
    assert fr["cf_worst_case"].iloc[1] == "A"


@pytest.fixture(scope="module")
def last():
    p = T._panel(n_days=3150, n_stocks=60, signal=True)
    rnd = np.random.default_rng(2)
    vix = pd.Series(rnd.uniform(12, 30, p["date"].nunique()), index=sorted(p["date"].unique()))
    p["vix"] = p["date"].map(vix)
    p["sector"] = np.where(p["ticker"].str[-1].astype(int) % 2 == 0, "Tech", "Health")
    specs = [{"id": "mom", "model": "rule", "rule_feature": "mom_3m", "target": "fwd_xs_20"},
             {"id": "enet", "model": "elastic_net", "target": "fwd_xs_20", "features": "all",
              "params_grid": {"alpha": [0.001], "l1_ratio": [0.5]}}]
    meta.evaluate(p, specs, latest_ranks={})
    return dict(meta.LAST_RUN), p


def test_counterfactual_columns_exist_historically(last):
    lr, _ = last
    cols = [c for c in lr["res"].columns if c.startswith("cfrank_")]
    assert len(cols) == len(cf.ALL_CASES)
    assert lr["res"][cols].stack().between(0, 1).all()


def test_blind_spot_clusters_and_walk_forward(last):
    lr, _ = last
    res = lr["res"].copy()
    top = bs.top_rows(res, "s_static_equal")
    # künstlicher blinder Fleck: Health-Titel im Top-Dezil verlieren massiv
    res.loc[top.index[top["sector"] == "Health"], meta.PROB_TARGET] -= 0.2
    cl = bs.find_clusters(bs.top_rows(res, "s_static_equal"))
    assert any(c["properties"].get("sector") == "Health" for c in cl)
    d = bs.describe(cl, {})
    assert d[0]["id"] == "UNKNOWN_CLUSTER_001" and d[0]["existing_model_coverage"] == "LOW"


def test_blind_spot_filter_uses_only_past(last):
    lr, _ = last
    out = bs.filter_walk_forward(lr["res"], lr["base"], "s_static_equal")
    assert out["verdict"] in ("KEEP", "MODIFY", "REJECT")
    assert {r["fold"] for r in out["per_fold"]} == set(lr["res"]["fold"].unique())


def test_greedy_selection_respects_sector_cap():
    c = pd.DataFrame({"exp": [0.9, 0.8, 0.7, 0.6, 0.5], "sector": ["A", "A", "A", "B", "C"]},
                     index=["t1", "t2", "t3", "t4", "t5"])
    sel = di.greedy_select(c, None, k=4, max_sector_share=0.5)
    assert sum(1 for t in sel if c.loc[t, "sector"] == "A") <= 2 and len(sel) == 4


def test_shrunk_cov_and_profile(last):
    _, p = last
    tick = sorted(p["ticker"].unique())[:8]
    prof = di.candidate_profile(tick, p, {t: {"exp": 0.01, "prob": 0.55} for t in tick})
    assert set(prof) == set(tick)
    assert all(v["avg_corr_to_candidates"] is not None for v in prof.values())
    assert abs(sum(v["tail_risk_contribution"] for v in prof.values()) - 1.0) < 1e-6


def test_full_validation_end_to_end(last, monkeypatch):
    lr, p = last
    monkeypatch.setattr(ni, "causal_tilt_scores", lambda res: None)
    rep = ni.evaluate(dict(lr), p)
    for k in ("A", "B", "F", "A_abstention", "G"):
        assert k in rep["metrics"], k
    m = rep["metrics"]["A"]
    for key in ("cagr", "sharpe", "sortino", "calmar", "max_dd", "es5_monthly", "hit_rate", "profit_factor",
                "expectancy", "avg_winner", "avg_loser", "precision_at_k", "brier", "ece", "log_loss", "turnover",
                "n_trades", "hc_hit_rate", "stability_sd_yearly", "regime_min_expectancy"):
        assert key in m, key
    assert rep["decision"] in ("PROMOTE", "KEEP_CHAMPION")
    assert "D/E" in rep["notes"]
    if not rep["G_components"]:
        assert rep["decision"] == "KEEP_CHAMPION"
    assert ni.render_md(rep)


def test_abstention_zero_months_and_gate_logic():
    months = pd.period_range("2021-01", "2021-06", freq="M")
    pos = pd.DataFrame({"date": pd.to_datetime(["2021-02-05", "2021-02-12"]), "ret": [0.02, 0.01]})
    s = ni.monthly_full(pos, months)
    assert len(s) == 6 and (s.drop(pd.Period("2021-02")) == 0).all()
    ref = {"n_cohorts": 200, "brier": 0.25, "ece": 0.02, "sharpe": 0.3, "regime_min_expectancy": -0.002,
           "hc_hit_rate": 0.5, "max_dd": -0.1}
    good = {**ref, "sharpe": 0.6, "hc_hit_rate": 0.55}
    g = ni.gate(good, ref, {"ci_monthly_mean": [0.001, 0.01], "trimmed_mean": 0.002}, {"x": 0.001}, True)
    assert g["pass"]
    g2 = ni.gate(good, ref, {"ci_monthly_mean": [0.001, 0.01], "trimmed_mean": 0.002}, {"x": -0.001}, True)
    assert not g2["pass"] and "10_complexity_pays" in g2["failed"]

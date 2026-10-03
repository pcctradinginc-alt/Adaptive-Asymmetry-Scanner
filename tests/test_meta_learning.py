"""Meta-Learning: kein Stacking-Leakage (Basis- und Meta-Folds), PIT-Historie,
Regime-Abhängigkeit wird gelernt, Rauschen wird nicht promoted, Kalibrierungs-
Buckets erkennen Überkonfidenz, Gate/Safe-Mode/HC-Regel. Synthetisch, kein Netz."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_ml_research as T  # noqa: E402

from modules import meta_learning as meta  # noqa: E402
from modules import ml_research as ml  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PINNED_META_SHA256 = hashlib.sha256((ROOT / "config" / "meta_protocol.yaml").read_bytes()).hexdigest()


def test_meta_protocol_hash_pinned():
    # Pin: bei Änderung muss dieser Wert bewusst im Review angepasst werden
    assert PINNED_META_SHA256 == EXPECTED_META_SHA256, "config/meta_protocol.yaml geändert – nur strenger zulässig"


EXPECTED_META_SHA256 = "6e56c1c3a4b93ddb3d5d3686ae722be69f3e8dafa7d18679d1bc39ceccfad2a1"


def _synthetic_base(regime_dependent=True, n_dates=560, n_tick=80, seed=1):
    """Zwei Basismodelle: A trägt bei VIX >= 20, B bei VIX < 20 (oder Rauschen)."""
    rnd = np.random.default_rng(seed)
    dates = pd.bdate_range("2014-06-06", periods=n_dates, freq="W-FRI")
    vix = pd.Series(np.where(np.sin(np.arange(n_dates) / 9.0) > 0, 26.0, 14.0), index=dates)
    rows = []
    for i, d in enumerate(dates):
        a, b = rnd.uniform(size=n_tick), rnd.uniform(size=n_tick)
        if regime_dependent:
            sig = (a - 0.5) if vix[d] >= 20 else (b - 0.5)
        else:
            sig = np.zeros(n_tick)
        y = 0.06 * sig + rnd.normal(0, 0.03, n_tick)
        le = dates[i + 4] if i + 4 < n_dates else pd.NaT
        for k in range(n_tick):
            rows.append({"date": d, "ticker": f"T{k}", "r_A": a[k], "r_B": b[k], "p_A": a[k], "p_B": b[k],
                         "fwd_xs_20": y[k] if le is not pd.NaT else np.nan, "label_end_20": le,
                         "mfe_20": abs(y[k]), "mae_20": -abs(y[k]), "vix": vix[d], "vix_chg_21": 0.0,
                         "spy_trend_200": 0.05, "spy_mom_63": 0.01, "tnx": 3.0, "curve_10y_3m": 0.5,
                         "sector": ["Tech", "Health"][k % 2], "log_dollar_vol": rnd.uniform(-0.5, 0.5),
                         "vol_60": rnd.uniform(-0.5, 0.5)})
    df = pd.DataFrame(rows)
    df = df[df["label_end_20"].notna()].reset_index(drop=True)
    df["fold"] = df["date"].dt.year.astype(str)
    df.loc[df["date"] >= ml.LOCKED_FROM, "fold"] = "locked"
    df = df[(df["date"] >= "2019-01-01")].reset_index(drop=True)
    df["sector_code"] = df["sector"].astype("category").cat.codes.astype(float)
    return df


@pytest.fixture(scope="module")
def regime_case():
    df, meta_d = meta.build_meta_frame(_synthetic_base(True), ["A", "B"])
    res, prov = meta.meta_walk_forward(df, meta_d, ["A", "B"])
    return df, meta_d, res, prov


def test_trailing_stats_only_use_finished_labels():
    df, meta_d = meta.build_meta_frame(_synthetic_base(True, n_dates=400), ["A", "B"])
    d = meta_d.index[100]
    tr = meta.trailing_stats(meta_d, ["A"], 52, [d])
    past = meta_d[(meta_d.index < d) & (meta_d["label_end"] < d)].tail(52)
    assert tr.loc[d, "tic_A"] == pytest.approx(past["A"].mean())
    assert (past["label_end"] < d).all()


def test_no_stacking_leakage(regime_case):
    _, _, res, prov = regime_case
    ok, issues = meta.leakage_checks({}, prov["folds"])
    assert ok, issues
    for f, p in prov["folds"].items():
        assert pd.Timestamp(p["meta_train_max_label_end"]) < pd.Timestamp(p["meta_train_label_end_before"])


def test_meta_learns_regime_dependence(regime_case):
    df, meta_d, res, _ = regime_case
    dev = res[res["fold"] != "locked"]
    e_meta = meta.portfolio_metrics(meta.positions(dev, "s_meta_regime_weights"))["expectancy"]
    e_static = meta.portfolio_metrics(meta.positions(dev, "s_static_equal"))["expectancy"]
    e_stack = meta.portfolio_metrics(meta.positions(dev, "s_meta_stacking"))["expectancy"]
    assert e_meta > e_static + 0.003 and e_stack > e_static
    w = meta.regime_weights(meta.regime_weight_model(meta_d, ["A", "B"], pd.Timestamp("2024-01-01")), meta_d,
                            [meta_d.index[meta_d["vix"] >= 20][-1], meta_d.index[meta_d["vix"] < 20][-1]])
    hi, lo = w[w["date"] == w["date"].iloc[0]].set_index("model"), w[w["date"] == w["date"].iloc[-1]].set_index("model")
    assert hi.loc["A", "weight"] > hi.loc["B", "weight"] and lo.loc["B", "weight"] > lo.loc["A", "weight"]
    assert 0 <= hi.loc["A", "p_adds_value"] <= 1


def test_noise_is_not_promoted():
    df, meta_d = meta.build_meta_frame(_synthetic_base(False, seed=4), ["A", "B"])
    res, prov = meta.meta_walk_forward(df, meta_d, ["A", "B"])
    pm, pr = meta.positions(res, "s_meta_regime_weights"), meta.positions(res, "s_static_equal")
    cal = {"brier": 0.25, "ece": 0.01}
    g = meta.promotion_gate(res, "meta_regime_weights", "static_equal", pm, pr, cal, cal, True)
    assert g["verdict"] in ("REJECT", "NEED_MORE_DATA")
    assert not g["criteria"]["bootstrap_ci_positive"]["pass"]


def test_calibration_buckets_flag_overconfidence():
    rnd = np.random.default_rng(0)
    n = 5000
    cal = pd.DataFrame({"prob": np.full(n, 0.82), "rel20": np.where(rnd.uniform(size=n) < 0.61, 0.02, -0.02),
                        "mae_20": -0.03, "exp_xs20": 0.01})
    b = [x for x in meta.calibration_buckets(cal) if x["n"]][0]
    assert b["bucket"].startswith("80") and b["flag"] == "overconfident"
    assert b["calibration_error"] == pytest.approx(0.61 - 0.82, abs=0.03)


def test_lagged_calibration_uses_previous_fold_only(regime_case):
    _, _, res, _ = regime_case
    cal = meta.lagged_calibration(res, "s_static_equal")
    first = sorted(f for f in res["fold"].unique() if f != "locked")[0]
    assert first not in set(cal["fold"])                 # erstes Jahr hat kein Vorjahr -> keine Wahrscheinlichkeit
    assert cal["prob"].between(0, 1).all()


def test_hc_rule_disabled_without_edge_and_uses_only_calibration_years():
    df, meta_d = meta.build_meta_frame(_synthetic_base(False, seed=5), ["A", "B"])
    res, _ = meta.meta_walk_forward(df, meta_d, ["A", "B"])
    cal = meta.lagged_calibration(res, "s_static_equal")
    r = meta.calibrate_hc_rule(cal, res, "static_equal", use_agreement=False)
    assert not r["enabled"]
    if r.get("calibration_folds"):
        assert "locked" not in r["calibration_folds"]


def test_hc_rule_enabled_with_real_edge(regime_case):
    _, _, res, _ = regime_case
    cal = meta.lagged_calibration(res, "s_meta_regime_weights")
    r = meta.calibrate_hc_rule(cal, res, "meta_regime_weights", use_agreement=False)
    assert r["enabled"], r
    assert r["validation"]["mean"] > 0 and r["n_rules_tested"] == len(meta.MP["high_confidence"]["prob_grid"])


def test_disagreement_test_structure(regime_case):
    _, _, res, _ = regime_case
    d = meta.disagreement_test(res)
    assert "use_in_score" in d and "dis_rank_sd" in d


def test_end_to_end_evaluate_on_small_panel(tmp_path, monkeypatch):
    p = T._panel(n_days=3150, n_stocks=55, signal=True)
    p["vix"] = 15.0 + 10 * (p["date"].dt.month % 2)
    specs = [{"id": "mom", "model": "rule", "rule_feature": "mom_3m", "target": "fwd_xs_20"},
             {"id": "rev", "model": "rule", "rule_feature": "rev_1m", "target": "fwd_xs_20"},
             {"id": "enet", "model": "elastic_net", "target": "fwd_xs_20", "features": "all",
              "params_grid": {"alpha": [0.001], "l1_ratio": [0.5]}}]
    rep = meta.evaluate(p, specs, latest_ranks={})
    assert rep["leakage_checks"]["ok"], rep["leakage_checks"]
    assert rep["decision"]["verdict"] in ("PROMOTE", "REJECT", "NEED_MORE_DATA")
    for n in ("static_equal", "meta_regime_weights", "meta_stacking", "best_single_ex_ante", "trailing_ic_weighted"):
        assert n in rep["approaches"]
    for k in ("cagr", "sharpe", "sortino", "max_dd", "calmar", "profit_factor", "hit_rate", "avg_winner",
              "avg_loser", "payoff", "expectancy", "brier", "log_loss", "ece", "precision_at_k", "recall_strong",
              "turnover", "exposure", "n_trades"):
        assert k in rep["approaches"]["static_equal"]["metrics"], k
    assert "meta_regime_weights__no_regime" in rep["ablations"]
    assert meta.render_md(rep)

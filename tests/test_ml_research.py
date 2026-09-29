"""ML-Research-Kreislauf: keine Zukunftsdaten in Features, korrekte Labels
(Entry Open_{t+1}, MFE/MAE), Purge im Walk-Forward, Locked-Holdout,
Registry-Hash, Prognose-Ledger, Entscheidungsregel. Synthetisch, kein Netz."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from modules import ml_research as ml


def _spy(n):
    idx = pd.bdate_range("2014-01-01", periods=n)
    c = pd.Series(200 * np.cumprod(1 + np.random.default_rng(0).normal(0.0003, 0.008, n)), index=idx)
    return pd.DataFrame({"Open": c * 0.999, "Close": c})


def _stock(n, seed, signal=None):
    rnd = np.random.default_rng(seed)
    idx = pd.bdate_range("2014-01-01", periods=n)
    r = rnd.normal(0, 0.015, n)
    if signal is not None:
        r = r + signal
    c = 50 * np.cumprod(1 + r)
    o = c / (1 + rnd.normal(0, 0.002, n))
    return pd.DataFrame({"Open": o, "High": np.maximum(o, c) * 1.01, "Low": np.minimum(o, c) * 0.99,
                         "Close": c, "Volume": rnd.integers(1e6, 2e6, n).astype(float)}, index=idx)


def test_labels_entry_next_open_and_excursions():
    spy = _spy(400)
    df = _stock(400, 1)
    f = ml.ticker_frame(df, spy, horizons=(20,))
    t = 300
    entry = df["Open"].iloc[t + 1]
    assert math.isclose(f["fwd_ret_20"].iloc[t], df["Close"].iloc[t + 20] / entry - 1)
    assert math.isclose(f["mfe_20"].iloc[t], df["High"].iloc[t + 1:t + 21].max() / entry - 1)
    assert math.isclose(f["mae_20"].iloc[t], df["Low"].iloc[t + 1:t + 21].min() / entry - 1)
    assert f["label_end_20"].iloc[t] == spy.index[t + 20]
    assert np.isnan(f["fwd_ret_20"].iloc[-5]) and np.isnan(f["mfe_20"].iloc[-5])   # unvollständig -> NaN


def test_features_do_not_use_future_data():
    spy = _spy(500)
    df = _stock(500, 2)
    f1 = ml.ticker_frame(df, spy)
    g = df.copy()
    g.iloc[405:, g.columns.get_loc("Close")] *= 1.5     # nur Zukunft (Exit-Fenster) ändern
    f2 = ml.ticker_frame(g, spy)
    for col in ml.STOCK_FEATURES:
        a, b = f1[col].iloc[400], f2[col].iloc[400]
        assert (np.isnan(a) and np.isnan(b)) or math.isclose(a, b), col
    assert not math.isclose(f1["fwd_ret_20"].iloc[400], f2["fwd_ret_20"].iloc[400])


def test_macro_features_only_published_before_day():
    from datetime import datetime, timezone
    from modules.external.pit import AvailabilityPrecision, Observation
    obs = []
    for m in range(1, 25):
        t = datetime(2015 + (m - 1) // 12, (m - 1) % 12 + 1, 1, tzinfo=timezone.utc)
        avail = datetime(t.year + (t.month == 12), t.month % 12 + 1, 15, tzinfo=timezone.utc)
        obs.append(Observation(source_id="fred_regime_macro", dataset="d", series_id="CPIAUCSL", entity_id="US",
                               metric="us_cpi", value=100.0 * (1.02 ** (m / 12)) * (1.1 if m == 24 else 1.0), unit="i", observation_time=t,
                               available_at=avail, retrieved_at=avail,
                               availability_precision=AvailabilityPrecision.CONSERVATIVE_DATE, parser_version="1"))
    d_before, d_after = pd.Timestamp("2017-01-15"), pd.Timestamp("2017-01-16")
    mf = ml.macro_features(obs, [d_before, d_after])
    # Dezember-2016-Sprung (veröffentlicht 2017-01-15) erst am Folgetag sichtbar
    assert math.isclose(mf.loc[d_before, "cpi_yoy"], 0.02, rel_tol=1e-6)
    assert math.isclose(mf.loc[d_after, "cpi_yoy"], 1.02 * 1.1 - 1, rel_tol=1e-6)


def _panel(n_days=1700, n_stocks=60, signal=True):
    spy = _spy(n_days)
    frames = {}
    rnd = np.random.default_rng(5)
    for i in range(n_stocks):
        frames[f"T{i:02d}"] = _stock(n_days, 100 + i)
    p = ml.build_panel(frames, spy, start="2014-06-01")
    if signal:     # künstlicher, lernbarer Zusammenhang: mom_3m-Rang -> Forward-Rendite
        p["fwd_xs_20"] = p["mom_3m"] * 0.06 + rnd.normal(0, 0.02, len(p))
        p.loc[p["label_end_20"].isna(), "fwd_xs_20"] = np.nan
    return p


def test_panel_ranks_and_columns():
    p = _panel(n_days=600, n_stocks=55, signal=False)
    g = p[p["date"] == p["date"].max()]
    assert g["mom_12_1"].between(-0.5, 0.5).all() or g["mom_12_1"].isna().all()
    for col in ml.ALL_FEATURES:
        assert col in p.columns
    assert p.groupby("date")["ticker"].nunique().max() == 55


def test_walk_forward_purges_training_and_respects_lock(monkeypatch):
    p = _panel()
    locked = pd.Timestamp("2020-06-01")
    seen = []
    orig = ml.fit

    def spy_fit(spec, train, params=None):
        seen.append(train["label_end_20"].max())
        return orig(spec, train, params)
    monkeypatch.setattr(ml, "fit", spy_fit)
    spec = {"id": "x", "model": "elastic_net", "target": "fwd_xs_20", "features": "all",
            "params_grid": {"alpha": [0.001], "l1_ratio": [0.5]}}
    wf = ml.walk_forward(p, spec, locked, first_test_year=2018, importance=False)
    assert wf["status"] == "ok"
    oos = wf["oos"]
    assert (oos["date"] < locked).all() and (oos["label_end_20"] < locked).all()
    for y in wf["params_by_year"]:
        assert all(le < locked for le in seen)
    # jedes Trainings-Label endet vor dem jeweiligen Testjahr
    years = sorted(wf["params_by_year"])
    assert seen[0] < pd.Timestamp(year=years[0], month=1, day=1)


def test_learnable_signal_is_found_and_noise_is_not():
    p = _panel()
    spec = {"id": "e", "model": "elastic_net", "target": "fwd_xs_20", "features": "all",
            "params_grid": {"alpha": [0.0003], "l1_ratio": [0.5]}}
    ev = ml.evaluate_oos(ml.walk_forward(p, spec, pd.Timestamp("2020-06-01"), 2017, importance=False)["oos"])
    assert ev["ic"]["mean_ic"] > 0.3 and ev["base"]["mean"] > 0
    q = _panel(signal=False)
    ev0 = ml.evaluate_oos(ml.walk_forward(q, spec, pd.Timestamp("2020-06-01"), 2017, importance=False)["oos"])
    assert abs(ev0["ic"]["mean_ic"]) < 0.1


def test_permutation_importance_finds_signal_feature():
    p = _panel()
    spec = {"id": "e", "model": "elastic_net", "target": "fwd_xs_20", "features": "all",
            "params_grid": {"alpha": [0.0003], "l1_ratio": [0.5]}}
    train = ml.purged(p, pd.Timestamp("2019-01-01"))
    test = p[(p["date"] >= "2019-01-01") & p["fwd_xs_20"].notna()]
    imp = ml.permutation_importance(ml.fit(spec, train, spec["params_grid"] and {"alpha": 0.0003, "l1_ratio": 0.5}),
                                    test, "fwd_xs_20")
    assert max(imp, key=imp.get) == "mom_3m"


def test_registry_hash_detects_modified_spec(tmp_path):
    reg = {"models": [{"id": "a", "model": "hist_gbm", "target": "fwd_xs_20", "params_grid": {"max_iter": [10]}}]}
    logp = tmp_path / "log.json"
    st, book = ml.check_registry(reg, logp)
    assert st["a"] == "valid"
    logp.write_text(json.dumps(book))
    reg["models"][0]["params_grid"] = {"max_iter": [999]}
    st2, _ = ml.check_registry(reg, logp)
    assert st2["a"] == "invalid_modified"


def test_registry_file_is_valid_and_has_no_champion():
    reg = ml.load_registry()
    assert reg["champion"] is None and "promotion_criteria" not in reg and "locked_from" not in reg
    assert ml.LOCKED_FROM < pd.Timestamp.now()
    ids = [m["id"] for m in reg["models"]]
    assert len(ids) == len(set(ids)) and any(m.get("role") == "benchmark" for m in reg["models"])
    for m in reg["models"]:
        assert m["model"] in ("rule", "elastic_net", "hist_gbm")
        if m["model"] != "rule":
            ml._make_model(m["model"], ml._grid(m)[0])


def test_decide_requires_all_criteria_and_forward():
    crit = ml.PROTOCOL["promotion_criteria"]
    good = {"base": {"mean": 0.004, "t_months": 3.0, "sharpe_ann": 1.2, "max_dd": -0.05, "years_positive_share": 0.8},
            "stress": {"mean": 0.001}, "ic": {"mean_ic": 0.03, "t_months": 3.0}}
    bench = {"base": {"sharpe_ann": 0.5, "max_dd": -0.10}}
    locked = {"base": {"mean": 0.002}}
    assert ml.decide(good, bench, crit, None, locked)["verdict"] == "running_forward"
    fwd = {"base": {"n_cohorts": 30, "mean": 0.003}}
    assert ml.decide(good, bench, crit, fwd, locked)["verdict"] == "promote_recommended"
    assert ml.decide(good, bench, crit, fwd, {"base": {"mean": -0.001}})["verdict"] == "rejected_so_far"
    weak = {**good, "base": {**good["base"], "t_months": 1.2}}
    assert ml.decide(weak, bench, crit, fwd, locked)["verdict"] == "rejected_so_far"


def test_prediction_ledger_roundtrip_and_forward_eval(tmp_path):
    p = _panel(n_days=900, n_stocks=60)
    reg = {"models": [{"id": "mom", "role": "benchmark", "model": "rule", "rule_feature": "mom_3m",
                       "target": "fwd_xs_20", "registered_at": "2000-01-01T00:00:00Z"}]}
    st, _ = ml.check_registry(reg, tmp_path / "log.json")
    d0 = p["date"].unique()[-40]
    past = p[p["date"] <= d0]
    new = ml.predict_latest(past, reg, st, pred_dir=tmp_path)
    assert len(new) == 1 and new[0]["prediction_date"] == str(pd.Timestamp(d0).date())
    assert ml.predict_latest(past, reg, st, pred_dir=tmp_path) == []            # idempotent
    fwd = ml.forward_eval(p, reg, pred_dir=tmp_path)
    assert fwd["mom"]["base"]["n_cohorts"] == 1
    stored = ml._read_predictions(tmp_path)[0]
    assert stored["realized"]["top_vs_univ_gross"] is not None
    # latest_scores liefert Ränge nur für frische Prognosen
    assert ml.latest_scores(pred_dir=tmp_path, today=pd.Timestamp(d0) + pd.Timedelta(days=3))["mom"]
    assert ml.latest_scores(pred_dir=tmp_path, today=pd.Timestamp(d0) + pd.Timedelta(days=30)) == {}


def test_forward_eval_ignores_predictions_before_registration(tmp_path):
    p = _panel(n_days=900, n_stocks=60)
    reg = {"models": [{"id": "mom", "model": "rule", "rule_feature": "mom_3m", "target": "fwd_xs_20",
                       "registered_at": "2999-01-01T00:00:00Z"}]}
    st, _ = ml.check_registry(reg, tmp_path / "log.json")
    ml.predict_latest(p[p["date"] <= p["date"].unique()[-40]], reg, st, pred_dir=tmp_path)
    assert ml.forward_eval(p, reg, pred_dir=tmp_path) == {}


def test_perf_costs_and_asymmetry():
    coh = pd.DataFrame({"date": pd.bdate_range("2020-01-03", periods=60, freq="W-FRI"),
                        "top_vs_univ": 0.004, "long_short": 0.01, "top_mfe": 0.08, "top_mae": -0.04,
                        "univ_mfe": 0.06, "univ_mae": -0.05})
    b, s = ml.perf(coh, ml.COST_BASE), ml.perf(coh, ml.COST_STRESS)
    assert math.isclose(b["mean"], 0.002) and math.isclose(s["mean"], -0.001)
    assert b["top_asymmetry"] == 2.0


@pytest.mark.parametrize("target", ["fwd_xs_20", "asym_20"])
def test_hgb_trains_on_both_targets(target):
    p = _panel(n_days=700, n_stocks=55)
    spec = {"model": "hist_gbm", "target": target, "features": "all",
            "params_grid": {"max_iter": [20], "max_depth": [2], "min_samples_leaf": [50]}}
    m = ml.fit(spec, ml.purged(p, p["date"].max()), {"max_iter": 20, "max_depth": 2, "min_samples_leaf": 50})
    assert np.isfinite(m.predict(p.tail(100))).all()


def test_feature_groups_in_spec():
    cols = ml.feature_list({"features": ["group:momentum", "vix", "group:nope", "fwd_xs_20"]})
    assert cols[:6] == list(ml.FEATURE_GROUPS["momentum"]) and cols[-1] == "vix" and "fwd_xs_20" not in cols


def _unc_panel():
    p = _panel(n_days=2000, n_stocks=60, signal=False)
    rnd = np.random.default_rng(3)
    # heteroskedastisch: Streuung wächst mit vol_60-Rang -> Intervall muss breiter werden
    scale = 0.05 + 0.2 * (p["vol_60"].fillna(0) + 0.5)
    p["fwd_ret_60"] = rnd.normal(0.01, 1, len(p)) * scale
    p["mae_60"] = -np.abs(rnd.normal(0, 1, len(p))) * scale
    p.loc[p["label_end_60"].isna(), ["fwd_ret_60", "mae_60"]] = np.nan
    return p


def test_uncertainty_interval_calibrated_and_adapts():
    p = _unc_panel()
    cal = ml.calibration_wf(p, pd.Timestamp("2021-06-01"), first_test_year=2018)
    assert cal["status"] == "ok"
    assert abs(cal["coverage"] - 0.8) < 0.06
    train = ml.purged(p, pd.Timestamp("2019-01-01"), 60)
    m = ml.fit_uncertainty(train)
    test = p[(p["date"] >= "2019-01-01") & p["vol_60"].notna()].head(3000)
    u = ml.predict_uncertainty(m, test)
    width = u["q_hi"] - u["q_lo"]
    assert width[test["vol_60"] > 0.3].mean() > 1.5 * width[test["vol_60"] < -0.3].mean()
    assert (u["q_lo"] <= u["q_mid"]).all() and (u["q_mid"] <= u["q_hi"]).all() and (u["mae_mid"] <= 0).all()


def test_cards_fields_analogs_and_counterfactual(tmp_path):
    p = _unc_panel()
    latest = p["date"].max()
    tick = sorted(p.loc[p["date"] == latest, "ticker"])
    ranks = {"a": {t: i / len(tick) for i, t in enumerate(tick)},
             "b": {t: 1 - i / len(tick) for i, t in enumerate(tick)}}
    cards = ml.build_cards(p, ranks, {"interval_calibrated": True, "p_up_skill": 0.01}, top_n_drivers=5)
    c = cards["cards"][tick[0]]
    for k in ("expected_return_60", "interval_80", "expected_drawdown_60", "p_return_gt_10", "model_disagreement",
              "data_quality", "regime_confidence", "analogs"):
        assert k in c, k
    assert c["model_disagreement"] == "HIGH"                     # Modelle widersprechen sich maximal
    assert c["interval_80"][0] <= c["expected_return_60"] <= c["interval_80"][1]
    assert sum("counterfactual" in v for v in cards["cards"].values()) == 5
    path = tmp_path / "cards.json"
    path.write_text(json.dumps(cards, default=str))
    assert ml.latest_cards(path=path, today=latest + pd.Timedelta(days=2))[tick[0]]
    assert ml.latest_cards(path=path, today=latest + pd.Timedelta(days=20)) == {}


def test_analogs_never_use_unfinished_labels():
    p = _unc_panel()
    latest = p["date"].max()
    hist = ml.purged(p, latest + pd.Timedelta(days=1), 60)
    assert (hist["label_end_60"] <= latest).all()
    a = ml.analogs(hist, p[p["date"] == latest], 10)
    assert a["analog_share_positive"].between(0, 1).all()


def test_conformal_correction_repairs_overconfident_intervals():
    """Streuung wächst über die Jahre: rohe Intervalle (aus der Vergangenheit
    gelernt) werden zu eng; die Vorjahres-Konformalkorrektur muss näher an 80 %."""
    p = _panel(n_days=2600, n_stocks=60, signal=False)
    rnd = np.random.default_rng(8)
    years = p["date"].dt.year - 2014
    scale = 0.03 * (1.25 ** years)
    p["fwd_ret_60"] = rnd.normal(0, 1, len(p)) * scale
    p["mae_60"] = -np.abs(rnd.normal(0, 1, len(p))) * scale
    p.loc[p["label_end_60"].isna(), ["fwd_ret_60", "mae_60"]] = np.nan
    cal = ml.calibration_wf(p, pd.Timestamp("2023-06-01"), first_test_year=2018)
    assert cal["coverage"] < 0.75                                   # roh überkonfident
    assert abs(cal["coverage_conformal"] - 0.8) < abs(cal["coverage"] - 0.8)
    assert cal["correction"]["qhat"] > 0 and cal["correction"]["from_year"] >= 2022

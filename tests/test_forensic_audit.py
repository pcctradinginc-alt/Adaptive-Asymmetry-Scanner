"""Forensischer Acceptance-Audit (docs/FORENSIC_ACCEPTANCE_AUDIT.md).

Diese Tests prüfen das GEWÜNSCHTE Verhalten end-to-end über hc_scanner.run()
(Dry-Run, Test-Transport, kein Netz) bzw. greifen Leakage-/Safe-Mode-Pfade an.
Während des Audits wird NICHTS repariert: Tests, die heute einen Defekt
belegen, sind xfail(strict=True) mit Audit-ID markiert. Wird der Defekt
behoben, schlägt der xfail fehl und muss entfernt werden.
"""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from modules import counterfactual as cf
from modules import hc_scanner as hc
from modules import meta_cognition as mc
from modules import ml_research as ml

TODAY = date(2026, 10, 5)
SIGNAL = "2026-10-02"
TICKERS = [f"T{i:03d}" for i in range(1, 101)]


# ── Harness: vollständiges out_dir wie nach einem CI-Lauf ───────────────────

def _card(dq="HIGH"):
    return {"expected_return_60": 0.06, "interval_80": [-0.10, 0.22], "expected_drawdown_60": -0.07,
            "data_quality": dq, "regime_confidence": "HIGH", "interval_calibrated": True,
            "analogs": {"analog_n": 30, "analog_share_positive": 0.7, "analog_median_mfe_60": 0.16,
                        "analog_median_mae_60": -0.07, "analog_median_ret_60": 0.04}}


def _write_env(d, *, rule_enabled=True, prob_validated=True, ece=0.02, interval_calibrated=True, safe_mode={"active": False, "reasons": []},
               signal_date=SIGNAL, dq="HIGH", cards=True):
    rule = {"enabled": rule_enabled, "disabled_reason": None if rule_enabled else "keine Regel",
            "active_ece": ece, "rule": {"prob": 0.55}, "agreement_used": False, "feature_drift_flag": False,
            "probability_validated": prob_validated,
            "prob_map": {"prob": {"x": [0.0, 0.9, 1.0], "y": [0.40, 0.52, 0.62]},
                         "exp_xs20": {"x": [0.0, 1.0], "y": [-0.01, 0.015]}},
            "models": ["a", "b"], "meta_version": "meta-test", "active_ensemble": "static_equal"}
    (d / "hc_thresholds.json").write_text(json.dumps(rule))
    (d / "ml_cards.json").write_text(json.dumps({"date": signal_date,
                                                 "cards": {t: _card(dq) for t in TICKERS} if cards else {}}))
    (d / "ml_research.json").write_text(json.dumps({"calibration": {"interval_calibrated": interval_calibrated}}))
    (d / "next_validation.json").write_text(json.dumps({"abstention_confirmation": {"confirmed": True}}))
    (d / "meta_learning.json").write_text(json.dumps({"model_intelligence": {}}))
    if safe_mode is not None:
        (d / "safe_mode.json").write_text(json.dumps(safe_mode))
    pdir = d / "ml_predictions"
    pdir.mkdir(exist_ok=True)
    rows = [{"model_id": m, "prediction_date": signal_date, "spec_hash": m,
             "rank_pct": {t: (i + 1) / 100 for i, t in enumerate(TICKERS)}} for m in ("a", "b")]
    (pdir / f"{signal_date[:7]}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _panel(vix=25.0, trend=-0.02):
    return pd.DataFrame({"date": pd.Timestamp(SIGNAL), "ticker": TICKERS, "log_dollar_vol": 0.3,
                         "vix": vix, "spy_trend_200": trend})


@pytest.fixture
def robust_world(monkeypatch):
    """Counterfactual robust, kein KG-Widerspruch, kein Blind-Spot, positiver Portfolio-Nutzen."""
    monkeypatch.setattr(ml, "load_registry", lambda *a, **k: {"models": [{"id": "a"}]})
    monkeypatch.setattr(ml, "check_registry", lambda reg, *a, **k: ({"a": "valid"}, {}))
    monkeypatch.setattr(cf, "latest_counterfactuals",
                        lambda panel, specs: {"per_ticker": {t: {"fragile": False} for t in TICKERS}})
    from modules import decision_intel as dint
    from modules import knowledge_graph as kg
    monkeypatch.setattr(kg, "load_inputs", lambda: (_ for _ in ()).throw(OSError("kein KG im Test")))
    monkeypatch.setattr(dint, "candidate_profile",
                        lambda tickers, panel, info: {t: {"portfolio_utility": 0.01} for t in tickers})
    from modules import mailer
    monkeypatch.setattr(mailer, "send_mail", lambda subj, html, text, dry_run=False, **k:
                        {"status": "dry_run" if dry_run else "sent"})     # Test-Transport


def _run(d, **kw):
    return hc.run(send=False, dry_run=True, today=TODAY, out_dir=d, earnings_fn=lambda t, x: 40,
                  name_fn=lambda t: t, **kw)


# ── 16. Alert-Audit (end-to-end über run) ───────────────────────────────────

def test_A16_good_candidate_produces_alert(tmp_path, robust_world):
    _write_env(tmp_path)
    res = _run(tmp_path, panel=_panel())
    assert res["enabled"] and res["candidates"], res.get("disabled_reason")
    assert res["sent"] and all(s["status"] for s in res["sent"])
    assert not (tmp_path / "alerts_state.json").exists()            # Dry-Run ändert keinen Zustand


def test_A16_calm_regime_no_alert(tmp_path, robust_world):
    _write_env(tmp_path)
    res = _run(tmp_path, panel=_panel(vix=15.0, trend=0.05))
    assert res["candidates"] == [] and "nichts tun" in res["disabled_reason"]


@pytest.mark.parametrize("kw, needle", [
    ({"rule_enabled": False}, "keine OOS-validierte Regel"),
    ({"prob_validated": False}, "Wahrscheinlichkeiten nicht validiert"),
    ({"ece": 0.09}, "Kalibrierung"),
    ({"interval_calibrated": False}, "intervalle nicht kalibriert"),
    ({"safe_mode": {"active": True, "reasons": ["TEST"]}}, "SAFE MODE"),
    ({"signal_date": "2026-09-20"}, "veraltet"),
    ({"cards": False}, "Unsicherheitskarten"),
])
def test_A16_A18_blocking_conditions(tmp_path, robust_world, kw, needle):
    _write_env(tmp_path, **kw)
    res = _run(tmp_path, panel=_panel())
    assert res["candidates"] == [] and res["sent"] == []
    assert needle in (res["disabled_reason"] or "")


def test_A16_bad_data_quality_card_no_alert(tmp_path, robust_world):
    _write_env(tmp_path, dq="LOW")
    assert _run(tmp_path, panel=_panel())["candidates"] == []


def test_A16_fragile_counterfactual_blocks(tmp_path, robust_world, monkeypatch):
    _write_env(tmp_path)
    monkeypatch.setattr(cf, "latest_counterfactuals",
                        lambda panel, specs: {"per_ticker": {t: {"fragile": True, "worst_case": "crash"} for t in TICKERS}})
    res = _run(tmp_path, panel=_panel())
    assert res["candidates"] == [] and res["rejected_by_intelligence"]


def _send(d, panel, day=TODAY):
    return hc.run(send=True, dry_run=False, today=day, out_dir=d, panel=panel,
                  earnings_fn=lambda t, x: 40, name_fn=lambda t: t)


def test_A16_duplicate_suppressed_and_material_change_realerted(tmp_path, robust_world):
    _write_env(tmp_path)
    assert _send(tmp_path, _panel())["sent"]
    assert _send(tmp_path, _panel())["sent"] == []                  # Duplikat
    re = _send(tmp_path, _panel(vix=25, trend=0.05))                 # Regimewechsel = materielle Änderung
    assert re["sent"] and {s["reason"] for s in re["sent"]} == {"Regimewechsel"}


def test_A16_failed_delivery_is_retried(tmp_path, robust_world, monkeypatch):
    """Remediation P1-3 (Audit F10)."""
    from modules import mailer
    _write_env(tmp_path)
    monkeypatch.setattr(mailer, "send_mail", lambda *a, **k: {"status": "failed"})
    first = _send(tmp_path, _panel())                                # nicht zugestellt
    assert first["sent"] and all(s["status"] != "sent" for s in first["sent"])
    monkeypatch.setattr(mailer, "send_mail", lambda *a, **k: {"status": "sent"})
    again = _send(tmp_path, _panel())                                # erneuter Versuch
    assert again["sent"] and {s["reason"] for s in again["sent"]} == {"neu"}
    assert _send(tmp_path, _panel())["sent"] == []                   # jetzt zugestellt -> Duplikat


# ── 18. Safe-Mode-Angriffe ──────────────────────────────────────────────────

def test_A18_corrupt_threshold_file_fails_closed(tmp_path, robust_world):
    _write_env(tmp_path)
    (tmp_path / "hc_thresholds.json").write_text("{kaputt")
    res = _run(tmp_path, panel=_panel())
    assert res["candidates"] == [] and not res["enabled"]


def test_A18_corrupt_safe_mode_file_must_fail_closed(tmp_path, robust_world):
    _write_env(tmp_path)
    (tmp_path / "safe_mode.json").write_text("{kaputt")
    res = _run(tmp_path, panel=_panel())
    assert res["candidates"] == [] and "SAFE MODE unbekannt" in res["disabled_reason"]


def test_A18_missing_safe_mode_file_must_fail_closed(tmp_path, robust_world):
    """Remediation P1-2 (Audit F07)."""
    _write_env(tmp_path)
    (tmp_path / "safe_mode.json").unlink()
    res = _run(tmp_path, panel=_panel())
    assert res["candidates"] == [] and "SAFE MODE unbekannt" in res["disabled_reason"]


@pytest.mark.parametrize("meta, world, health, fwd, mlr, needle", [
    ({"drift": {"feature_drift_flag": True, "feature_drift": {"vix": {"out_of_range": True}}}}, {}, {}, None, {}, "DRIFT"),
    ({"model_intelligence": {"a": {"trend": "deteriorating"}, "b": {"trend": "deteriorating"}}}, {}, {}, None, {}, "MODEL DRIFT"),
    ({}, {}, {}, None, {"calibration": {"interval_calibrated": False}}, "CALIBRATION"),
    ({}, {}, {"a": {"status": "FAIL"}, "b": {"status": "FAIL"}, "c": {"status": "PASS"}}, None, {}, "PIPELINE"),
    ({"disagreement": {"current_level": "HIGH"}}, {}, {}, None, {}, "DISAGREEMENT"),
    ({}, {"current": {"uncertainty": 0.9}}, {}, None, {}, "WORLD MODEL"),
])
def test_A18_safe_mode_triggers(meta, world, health, fwd, mlr, needle):
    sm = mc.safe_mode(meta, world, health, fwd, mlr)
    assert sm["active"] and any(needle in r for r in sm["reasons"])


def test_A18_stale_sources_trigger_safe_mode():
    """Remediation P2-2 (Audit F08)."""
    health = {k: {"status": "PASS", "staleness": "STALE"} for k in "abcd"}
    assert mc.safe_mode({}, {}, health, None, {})["active"]
    one_high = {"a": {"status": "PASS", "staleness": "STALE", "criticality": "high"},
                **{k: {"status": "PASS", "staleness": "FRESH"} for k in "bcdefgh"}}
    assert any("STALE" in r for r in mc.safe_mode({}, {}, one_high, None, {})["reasons"])
    fresh = {k: {"status": "PASS", "staleness": "FRESH"} for k in "abcd"}
    assert not mc.safe_mode({}, {}, fresh, None, {})["active"]


def test_A18_corrupt_artifact_triggers_safe_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(mc, "OUT", tmp_path)
    (tmp_path / "hc_thresholds.json").write_text("{kaputt")
    mc.CORRUPT.clear()
    mc._load("hc_thresholds.json", {})
    sm = mc.safe_mode({}, {}, {}, None, {}, corrupt=list(mc.CORRUPT))
    assert sm["active"] and any("CORRUPT" in r for r in sm["reasons"])


# ── 14. Meta-Cognition: Aussage muss zur Metrik passen ──────────────────────

def test_A14_calibration_label_respects_overconfident_buckets():
    meta = {"approaches": {"static_equal": {"metrics": {"ece": 0.006}}},
            "calibration_buckets": {"static_equal": [
                {"bucket": "55–60 %", "n": 356, "predicted": 0.57, "win_rate": 0.46, "flag": "overconfident"}]}}
    st = mc.machine_state({"meta": meta, "ml": {"calibration": {"interval_calibrated": True}}},
                          {}, {}, {}, {"active": False, "reasons": []})
    assert st["self_assessment"]["overall_calibration"] != "GOOD"


# ── 5. Leakage-Angriffe ─────────────────────────────────────────────────────

def _toy_prices(n=400, seed=0):
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range("2020-01-01", periods=n)
    c = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=cal)
    df = pd.DataFrame({"Open": c.shift(1).fillna(100), "High": c * 1.01, "Low": c * 0.99, "Close": c,
                       "Volume": 1e6}, index=cal)
    return df


def test_A5_features_invariant_to_future_prices():
    """Ändert man alle Preise NACH t, dürfen sich Features zu t nicht ändern."""
    df, spy = _toy_prices(seed=1), _toy_prices(seed=2)
    t = df.index[300]
    base = ml.ticker_frame(df, spy)
    df2, spy2 = df.copy(), spy.copy()
    later = df.index[310]                                           # Einstieg t+1 unverändert, Ausstieg t+20 verändert
    df2.loc[df2.index > later, ["Open", "High", "Low", "Close"]] *= 3.0
    spy2.loc[spy2.index > later, ["Open", "Close"]] *= 0.3
    alt = ml.ticker_frame(df2, spy2)
    feats = list(ml.STOCK_FEATURES)
    pd.testing.assert_series_equal(base.loc[t, feats], alt.loc[t, feats], check_names=False)
    assert not np.isclose(base.loc[t, "fwd_ret_20"], alt.loc[t, "fwd_ret_20"])   # Label hängt an Zukunft


def test_A5_labels_start_at_next_open():
    df, spy = _toy_prices(seed=3), _toy_prices(seed=4)
    f = ml.ticker_frame(df, spy)
    t = df.index[100]
    exp = df["Close"].shift(-20).loc[t] / df["Open"].shift(-1).loc[t] - 1
    assert np.isclose(f.loc[t, "fwd_ret_20"], exp)
    assert f.loc[t, "label_end_20"] == df.index[120]


def test_A5_purge_excludes_labels_overlapping_test():
    p = pd.DataFrame({"date": pd.to_datetime(["2020-12-01", "2020-12-20", "2020-12-28"]),
                      "label_end_20": pd.to_datetime(["2020-12-29", "2021-01-15", "2021-01-26"])})
    tr = ml.purged(p, pd.Timestamp("2021-01-01"), 20, start="2014-01-01")
    assert list(tr["date"]) == [pd.Timestamp("2020-12-01")]


def test_A5_lagged_calibration_purges_overlapping_labels():
    from modules import meta_learning as mlm
    rows = []
    rng = np.random.default_rng(0)
    for fold, dates in (("2021", pd.date_range("2021-01-01", "2021-12-31", freq="W-FRI")),
                        ("2022", pd.date_range("2022-01-07", "2022-03-31", freq="W-FRI"))):
        for d in dates:
            for k in range(60):
                rows.append({"date": d, "ticker": f"X{k}", "fold": fold, "score": rng.normal(),
                             "rel20": rng.normal(), "fwd_xs_20": rng.normal(), "mae_20": -0.05,
                             "label_end_20": d + pd.Timedelta(days=28)})
    res = pd.DataFrame(rows)
    used = {}
    import sklearn.isotonic as iso_mod
    orig = iso_mod.IsotonicRegression.fit

    def spy_fit(self, X, y, *a, **k):
        used.setdefault("n", []).append(len(X))
        return orig(self, X, y, *a, **k)
    iso_mod.IsotonicRegression.fit = spy_fit
    try:
        mlm.lagged_calibration(res, "score")
    finally:
        iso_mod.IsotonicRegression.fit = orig
    leaking = res[(res["fold"] == "2021") & (res["label_end_20"] >= pd.Timestamp("2022-01-07"))]
    n_prev = int((res["fold"] == "2021").sum())
    assert used["n"][0] <= n_prev - len(leaking)


def _changes_table():
    cols = pd.MultiIndex.from_tuples([("Effective Date", "Effective Date"), ("Added", "Ticker"), ("Added", "Security"),
                                      ("Removed", "Ticker"), ("Removed", "Security"), ("Reason", "Reason")])
    return pd.DataFrame([["January 10, 2020", "CCC", "C Corp", "XXX", "X Corp", "Marktkap."],
                         ["May 1, 2018", "BRK.B", "B Corp", "YYY", "Y Inc", "Übernahme[1]"],
                         ["June 3, 2016", np.nan, np.nan, "ZZZ", "Z Inc", "Insolvenz"]], columns=cols)


def test_A5_sp500_changes_parser_and_pit_membership():
    """Remediation P0-2 (Audit F01): Mitgliedschaft je Stichtag statt heutiger Liste."""
    from modules import universe as u
    ch = u.parse_sp500_changes(_changes_table())
    assert ch[0] == {"date": "2016-06-03", "added": None, "removed": "ZZZ"}
    assert ch[1]["added"] == "BRK-B" and ch[2]["removed"] == "XXX"
    iv = u.membership_intervals(["AAA", "BRK-B", "CCC"], ch)
    at = lambda d: sorted(t for t, x in iv.items() if u.is_member(x, d))
    assert at("2021-03-01") == ["AAA", "BRK-B", "CCC"]
    assert at("2019-03-01") == ["AAA", "BRK-B", "XXX"]          # CCC noch nicht, XXX noch Mitglied
    assert at("2017-03-01") == ["AAA", "XXX", "YYY"]
    assert at("2015-03-01") == ["AAA", "XXX", "YYY", "ZZZ"]
    assert at("2020-01-10") == ["AAA", "BRK-B", "CCC"]          # Stichtag der Änderung: neue Zusammensetzung


def test_A5_research_universe_includes_removed_names(monkeypatch):
    from modules import universe as u
    monkeypatch.setattr(u, "sp500_history", lambda: {"intervals": u.membership_intervals(
        ["AAA", "BRK-B", "CCC"], u.parse_sp500_changes(_changes_table())), "n_changes": 3,
        "first_change": "2016-06-03", "source": "test"})
    uni = u.research_universe("2014-01-01")
    assert set(uni["tickers"]) == {"AAA", "BRK-B", "CCC", "XXX", "YYY", "ZZZ"}
    assert set(uni["removed_since_start"]) == {"XXX", "YYY", "ZZZ"}
    assert u.research_universe("2019-01-01")["removed_since_start"] == ["XXX"]


def test_A5_panel_rows_only_during_membership():
    spy = _toy_prices(n=400, seed=9)
    frames = {"IN": _toy_prices(seed=10), "OUT": _toy_prices(seed=11)}
    cut = str(spy.index[300].date())
    p = ml.build_panel(frames, spy, start=str(spy.index[0].date()),
                       membership={"IN": [[None, None]], "OUT": [[None, cut]]})
    late = p[p["date"] >= pd.Timestamp(cut)]
    assert set(late["ticker"]) == {"IN"} and "OUT" in set(p[p["date"] < pd.Timestamp(cut)]["ticker"])
    # Querschnittsränge nach dem Filter: einziger Titel -> Rang 0.5 (pct=1.0 - 0.5)
    assert (late["mom_3m"].dropna() == 0.5).all()


def test_A5_research_panel_never_falls_back_to_todays_list(monkeypatch):
    from modules import universe as u
    monkeypatch.setattr(u, "sp500_history", lambda: (_ for _ in ()).throw(RuntimeError("keine Historie")))
    with pytest.raises(RuntimeError):
        ml.build_research_panel("weekly")


# ── Remediation P0-4 (Audit F03): Abstinenz nur vorwärts bestätigbar ────────

def test_P04_contaminated_confirmation_never_counts():
    from modules import next_intelligence as ni
    assert ni.NP["abstention"]["confirmation_status"] == "CONTAMINATED"
    nv = {"abstention_confirmation": {"confirmed": False, "status": "CONTAMINATED", "in_sample_rule_holds": True,
                                      "forward": {"confirmed": False, "status": "ACCUMULATING"}},
          "approaches": {}}
    ok, why = hc.regime_compatible({}, nv, {"vix": 15.0, "spy_trend_200": 0.05})
    assert "Abstinenz-Regel (bestätigt)" not in why


def _fwd_panel(dates, vix):
    rows = []
    rng = np.random.default_rng(5)
    for d, v in zip(dates, vix):
        shock = rng.normal(0, 0.01)                         # Kohorten-Rauschen
        for i, t in enumerate(TICKERS):
            rows.append({"date": pd.Timestamp(d), "ticker": t, "vix": v, "spy_trend_200": 0.05,
                         "fwd_xs_20": 0.002 * (i - 50) / 50 + (0.03 + shock if (v >= 20 and i >= 90) else
                                                               (shock if i >= 90 else 0.0)),
                         "label_end_20": pd.Timestamp(d) + pd.Timedelta(days=28)})
    return pd.DataFrame(rows)


def test_P04_forward_abstention_ledger_counts_only_matured_cohorts():
    from modules import next_intelligence as ni
    dates = list(pd.date_range("2026-10-02", periods=70, freq="W-FRI").strftime("%Y-%m-%d"))
    vix = [25.0 if k % 2 else 15.0 for k in range(len(dates))]
    panel = _fwd_panel(dates, vix)
    preds = [{"model_id": m, "prediction_date": d, "rank_pct": {t: (i + 1) / 100 for i, t in enumerate(TICKERS)}}
             for d in dates for m in ("a", "b")] + \
            [{"model_id": "a", "prediction_date": "2026-09-01", "rank_pct": {"T001": 1.0}}]   # vor forward_from: ignoriert
    r = ni.forward_abstention(panel, preds)
    assert r["pending_cohorts"] >= 4                       # jüngste Labels noch offen -> nicht gezählt
    assert r["active_cohorts"] + r["inactive_cohorts"] + r["pending_cohorts"] == len(dates)
    assert r["active_expectancy"] > r["inactive_expectancy"]
    assert r["status"] == "CONFIRMED" and r["confirmed"]
    short = ni.forward_abstention(_fwd_panel(dates[:10], vix[:10]), preds)
    assert short["status"] == "ACCUMULATING" and not short["confirmed"]


def test_P14_probability_validation_requires_monotone_calibrated_buckets():
    from modules import meta_learning as mlm
    real = [{"bucket": "50–55 %", "n": 15659, "win_rate": 0.4606, "predicted": 0.5053, "calibration_error": -0.0447},
            {"bucket": "55–60 %", "n": 356, "win_rate": 0.4635, "predicted": 0.5717, "calibration_error": -0.1082},
            {"bucket": "65–70 %", "n": 61, "win_rate": 0.5738, "predicted": 0.6818, "calibration_error": -0.108}]
    v = mlm.probability_validation(real)                           # echte Buckets vom 2026-10-02
    assert not v["probability_validated"] and any("55–60" in r for r in v["probability_validation_reasons"])
    good = [{"bucket": "50–55 %", "n": 500, "win_rate": 0.52, "calibration_error": 0.0},
            {"bucket": "55–60 %", "n": 300, "win_rate": 0.57, "calibration_error": 0.0},
            {"bucket": "60–65 %", "n": 150, "win_rate": 0.62, "calibration_error": 0.0}]
    assert mlm.probability_validation(good)["probability_validated"]
    nonmono = [dict(good[0]), dict(good[1], win_rate=0.50, calibration_error=-0.04)]
    assert not mlm.probability_validation(nonmono)["probability_validated"]


# ── Remediation P2-5 (Audit F14): Determinismus bei gleichem Panel ──────────

def test_P25_meta_evaluate_is_deterministic_on_same_panel():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import test_ml_research as T
    from modules import meta_learning as mlm
    p = T._panel(n_days=2600, n_stocks=55, signal=True)
    p["vix"] = 15.0 + 10 * (p["date"].dt.month % 2)
    specs = [{"id": "mom", "model": "rule", "rule_feature": "mom_3m", "target": "fwd_xs_20"},
             {"id": "hgb", "model": "hist_gbm", "target": "fwd_xs_20", "features": "all",
              "params_grid": {"max_iter": [30], "learning_rate": [0.1], "max_depth": [2], "min_samples_leaf": [50]}}]
    a = mlm.evaluate(p.copy(), specs, latest_ranks={})
    b = mlm.evaluate(p.copy(), specs, latest_ranks={})
    import importlib.util
    spec = importlib.util.spec_from_file_location("repro_check", Path(__file__).resolve().parent.parent / "scripts" / "repro_check.py")
    rc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rc)
    flat_a, flat_b = {}, {}
    for name, ap in a["approaches"].items():
        rc._flatten(name, ap.get("metrics"), flat_a)
    for name, ap in b["approaches"].items():
        rc._flatten(name, ap.get("metrics"), flat_b)
    assert flat_a and rc.compare(flat_a, flat_b, 1e-12) == []
    assert a["panel_hash"] == b["panel_hash"] and a["decision"]["verdict"] == b["decision"]["verdict"]

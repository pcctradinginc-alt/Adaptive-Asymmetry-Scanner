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


def _write_env(d, *, rule_enabled=True, ece=0.02, interval_calibrated=True, safe_mode=None,
               signal_date=SIGNAL, dq="HIGH", cards=True):
    rule = {"enabled": rule_enabled, "disabled_reason": None if rule_enabled else "keine Regel",
            "active_ece": ece, "rule": {"prob": 0.55}, "agreement_used": False, "feature_drift_flag": False,
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
    return hc.run(send=True, dry_run=True, today=day, out_dir=d, panel=panel,
                  earnings_fn=lambda t, x: 40, name_fn=lambda t: t)


def test_A16_duplicate_suppressed_and_material_change_realerted(tmp_path, robust_world):
    _write_env(tmp_path)
    assert _send(tmp_path, _panel())["sent"]
    assert _send(tmp_path, _panel())["sent"] == []                  # Duplikat
    re = _send(tmp_path, _panel(vix=25, trend=0.05))                 # Regimewechsel = materielle Änderung
    assert re["sent"] and {s["reason"] for s in re["sent"]} == {"Regimewechsel"}


@pytest.mark.xfail(strict=True, reason="AUDIT-F10: fehlgeschlagener Versand wird im Dedup-Zustand als gemeldet "
                                       "gespeichert -> Alert geht verloren, kein Retry")
def test_A16_failed_delivery_is_retried(tmp_path, robust_world):
    _write_env(tmp_path)
    first = _send(tmp_path, _panel())                                # Mail nicht konfiguriert -> nicht zugestellt
    assert first["sent"] and all(s["status"] != "sent" for s in first["sent"])
    assert _send(tmp_path, _panel())["sent"]                         # muss erneut versucht werden


# ── 18. Safe-Mode-Angriffe ──────────────────────────────────────────────────

def test_A18_corrupt_threshold_file_fails_closed(tmp_path, robust_world):
    _write_env(tmp_path)
    (tmp_path / "hc_thresholds.json").write_text("{kaputt")
    res = _run(tmp_path, panel=_panel())
    assert res["candidates"] == [] and not res["enabled"]


@pytest.mark.xfail(strict=True, reason="AUDIT-F07: beschädigtes safe_mode.json wird als 'nicht aktiv' gelesen (fail-open)")
def test_A18_corrupt_safe_mode_file_must_fail_closed(tmp_path, robust_world):
    _write_env(tmp_path)
    (tmp_path / "safe_mode.json").write_text("{kaputt")
    res = _run(tmp_path, panel=_panel())
    assert res["candidates"] == []


@pytest.mark.xfail(strict=True, reason="AUDIT-F07: fehlendes safe_mode.json blockiert nicht (fail-open)")
def test_A18_missing_safe_mode_file_must_fail_closed(tmp_path, robust_world):
    _write_env(tmp_path)                                             # kein safe_mode.json
    res = _run(tmp_path, panel=_panel())
    assert res["candidates"] == []


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


@pytest.mark.xfail(strict=True, reason="AUDIT-F08: stale Quellen (STALE) lösen keinen Safe Mode aus, nur FAIL")
def test_A18_stale_sources_trigger_safe_mode():
    health = {k: {"status": "PASS", "staleness": "STALE"} for k in "abcd"}
    assert mc.safe_mode({}, {}, health, None, {})["active"]


# ── 14. Meta-Cognition: Aussage muss zur Metrik passen ──────────────────────

@pytest.mark.xfail(strict=True, reason="AUDIT-F09: 'overall_calibration GOOD' trotz überkonfidentem Top-Bucket")
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


@pytest.mark.xfail(strict=True, reason="AUDIT-F05: lagged_calibration nutzt Vorjahres-Labels, die nach Testbeginn enden (kein Purge)")
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


def test_A5_research_universe_is_survivorship_free():
    """Universum muss je Stichtag rekonstruiert werden (PIT-Mitgliedschaft)."""
    import inspect
    from modules import universe
    src = inspect.getsource(universe.get_universe)
    pit = "as_of" in inspect.signature(universe.get_universe).parameters
    if not pit:
        pytest.xfail("AUDIT-F01: get_universe() liefert die HEUTIGE Indexliste ohne PIT-Mitgliedschaft "
                     "(Survivorship-Bias im gesamten Research-Stack)")
    assert "delist" in src

"""High-Confidence-Scanner: keine Alerts ohne OOS-validierte Regel bzw. bei
zweifelhafter Kalibrierung/Datenlage, Mehrfachbedingungen, Deduplizierung,
Alert-Felder, keine Orderausführung, append-only Gedächtnis. Kein Netz."""
from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

from modules import hc_scanner as hc
from modules import prediction_memory as pm

TODAY = date(2026, 10, 5)


def _card(q=0.05, dq="HIGH", share=0.7, mfe=0.15, mae=-0.08):
    return {"expected_return_60": q, "interval_80": [-0.12, 0.2], "expected_drawdown_60": mae, "data_quality": dq,
            "regime_confidence": "HIGH", "interval_calibrated": True,
            "analogs": {"analog_n": 25, "analog_share_positive": share, "analog_median_mfe_60": mfe,
                        "analog_median_mae_60": mae, "analog_median_ret_60": 0.03},
            "counterfactual": {"supports": {"rev_1m": 0.01}, "weighs_against": {"vol_60": -0.004}}}


def _rule(enabled=True, ece=0.02):
    return {"enabled": enabled, "disabled_reason": None if enabled else "keine Regel", "active_ece": ece,
            "probability_validated": True,
            "rule": {"prob": 0.55, "agreement_sd": 0.2}, "agreement_used": True, "feature_drift_flag": False,
            "prob_map": {"prob": {"x": [0.0, 0.9, 1.0], "y": [0.45, 0.52, 0.6]},
                         "exp_xs20": {"x": [0.0, 1.0], "y": [-0.01, 0.012]}},
            "current_weights": {"a": {"weight": 0.6}, "b": {"weight": 0.4}},
            "failure_profiles": {"a": {"works": ["vix_lt_20 (IC 0.02, t 2.1)"], "fails": ["high_vol (IC -0.03, t -2.2)"]}},
            "models": ["a", "b"], "meta_version": "meta-v1", "active_ensemble": "static_equal"}


def _ens():
    ranks = {"a": {f"T{i}": i / 100 for i in range(1, 101)}, "b": {f"T{i}": (i / 100 + 0.01) % 1 for i in range(1, 101)}}
    return hc.ensemble_scores(ranks, ["a", "b"], None)


def test_global_gate_blocks_on_doubtful_calibration_or_data():
    assert hc.global_gate(_rule(enabled=False), {"x": 1}, {"interval_calibrated": True})
    assert hc.global_gate(_rule(ece=0.09), {"x": 1}, {"interval_calibrated": True})
    assert hc.global_gate(_rule(), {"x": 1}, {"interval_calibrated": False})
    assert hc.global_gate(_rule(), {}, {"interval_calibrated": True})
    assert not hc.global_gate(_rule(), {"x": 1}, {"interval_calibrated": True})


def test_candidates_need_all_conditions():
    ens = _ens()
    cards = {t: _card() for t in ens}
    cards["T100"] = _card(dq="MEDIUM")                              # Datenqualität
    cards["T99"] = _card(share=0.3)                                  # Analogien negativ
    liq = {t: 0.3 for t in ens}
    liq["T98"] = -0.2                                                # illiquide
    c = hc.evaluate_candidates(ens, cards, _rule(), liq, TODAY, earnings_fn=lambda t, d: 40)
    tick = {x["ticker"] for x in c}
    assert tick and not tick & {"T100", "T99", "T98"}
    assert all(x["prob"] >= 0.55 for x in c)
    assert hc.evaluate_candidates(ens, cards, _rule(), liq, TODAY, earnings_fn=lambda t, d: None) == []   # Event unbekannt
    assert hc.evaluate_candidates(ens, cards, _rule(), liq, TODAY, earnings_fn=lambda t, d: 3) == []      # Earnings nah


def test_alert_fields_and_no_false_precision():
    ens = _ens()
    c = hc.evaluate_candidates(ens, {t: _card() for t in ens}, _rule(), {t: 0.3 for t in ens}, TODAY,
                               earnings_fn=lambda t, d: 40)[0]
    a = hc.build_alert(c, _rule(), "2026-10-02", {"vix": 15, "spy_trend_200": 0.05}, {"models": {}, "meta": "m"}, "Co")
    for k in ("ticker", "company", "signal_date", "calibrated_probability", "expected_return_20d", "expected_return_60d",
              "expected_downside", "expected_max_adverse_excursion", "expected_max_favorable_excursion", "asymmetry_ratio",
              "model_agreement", "regime_compatibility", "data_quality", "number_historical_analogues",
              "historical_analogue_win_rate", "bull_case", "bear_case", "key_risk_factors", "invalidation_conditions",
              "model_version", "meta_model_version", "confidence"):
        assert k in a, k
    assert a["expected_return_120d"] is None and "nicht modelliert" in a["expected_return_120d_note"]
    assert a["no_order_execution"] is True and a["confidence"] in ("HIGH", "VERY HIGH")
    subj, html, text = hc.render_alert({**a, "alert_reason": "neu"})
    assert subj.startswith(f"[{a['confidence']} CONFIDENCE] {a['ticker']}") and "KEINE Orderausführung" in text


def test_dedup_only_on_material_change():
    a = {"ticker": "T1", "signal_id": "s1", "calibrated_probability": 0.56, "expected_return_20d": 0.01, "confidence": "HIGH"}
    send, st = hc.dedup([a], {}, TODAY, "vix_lt_20|up")
    assert [x["alert_reason"] for x in send] == ["neu"]
    send, st = hc.dedup([{**a, "calibrated_probability": 0.57}], st, date(2026, 10, 12), "vix_lt_20|up")
    assert send == []                                               # kaum verändert -> kein Spam
    send, st = hc.dedup([{**a, "calibrated_probability": 0.62}], st, date(2026, 10, 19), "vix_lt_20|up")
    assert send[0]["alert_reason"] == "Wahrscheinlichkeit deutlich gestiegen"
    send, st = hc.dedup([{**a, "calibrated_probability": 0.62}], st, date(2026, 10, 26), "vix_ge_20|up")
    assert send[0]["alert_reason"] == "Regimewechsel"
    send, st = hc.dedup([{**a, "calibrated_probability": 0.62}], st, date(2026, 12, 20), "vix_ge_20|up")
    assert send[0]["alert_reason"] == "nach Pause neu entstanden"
    assert st["T1"]["first_alert"] == "2026-12-20" and len(st["T1"]["history"]) == 5


def test_run_disabled_writes_no_alerts(tmp_path):
    (tmp_path / "hc_thresholds.json").write_text(json.dumps(_rule(enabled=False)))
    res = hc.run(send=False, dry_run=True, today=TODAY, out_dir=tmp_path)
    assert res["enabled"] is False and res["candidates"] == [] and "keine" in res["disabled_reason"]
    assert not (tmp_path / "alerts_state.json").exists()


def test_prediction_memory_append_only_and_outcomes(tmp_path):
    row = {"signal_date": "2026-01-02", "ticker": "AAA", "model_versions": {"a": "h1"}, "meta_model_version": "m1",
           "uncertainty": {"interval_80": [-0.1, 0.2]}}
    assert pm.record_predictions([row], tmp_path) == 1
    assert pm.record_predictions([row], tmp_path) == 0                     # gleiche id -> nicht doppelt
    assert pm.record_predictions([{**row, "meta_model_version": "m2"}], tmp_path) == 1   # neue Version -> neue id
    before = (tmp_path / "2026-01.jsonl").read_text()
    panel = pd.DataFrame({"date": [pd.Timestamp("2026-01-02")], "ticker": ["AAA"], "fwd_xs_20": [-0.03],
                          "fwd_ret_60": [-0.15], "mae_20": [-0.08], "mfe_20": [0.02]})
    assert pm.record_outcomes(panel, tmp_path) == 2
    assert pm.record_outcomes(panel, tmp_path) == 0
    after = (tmp_path / "2026-01.jsonl").read_text()
    assert after.startswith(before)                                         # nur angehängt
    mem = pm.load_memory(tmp_path)
    assert set(mem["error_category"]) == {"below_interval"} and set(mem["trade_outcome"]) == {"loss"}


def test_no_broker_or_order_code_anywhere():
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    for f in ("modules/hc_scanner.py", "modules/meta_learning.py", "reports/weekly.py", "modules/mailer.py"):
        src = (root / f).read_text().lower()
        for bad in ("place_order", "submit_order", "alpaca", "ib_insync", "/orders"):
            assert bad not in src, (f, bad)


def test_regime_compatibility_prefers_confirmed_abstention():
    ok, why = hc.regime_compatible({}, {"abstention_confirmation": {"confirmed": True}}, {"vix": 15, "spy_trend_200": 0.05})
    assert not ok and "nichts tun" in why
    ok2, _ = hc.regime_compatible({}, {"abstention_confirmation": {"confirmed": True}}, {"vix": 24, "spy_trend_200": 0.05})
    assert ok2
    meta = {"active_ensemble": "static_equal", "approaches": {"static_equal": {"by_regime": {
        "vix_lt_20": {"expectancy": -0.002}, "vix_ge_20": {"expectancy": 0.012},
        "spy_uptrend": {"expectancy": -0.0004}, "spy_downtrend": {"expectancy": 0.015}}}}}
    assert not hc.regime_compatible(meta, {}, {"vix": 15, "spy_trend_200": 0.05})[0]
    assert hc.regime_compatible(meta, {}, {"vix": 25, "spy_trend_200": -0.02})[0]


def test_intelligence_checks_reject_fragile_and_blindspot(tmp_path, monkeypatch):
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parent))
    import test_ml_research as T
    from modules import counterfactual as cf
    p = T._panel(n_days=900, n_stocks=55, signal=False)
    p["sector"] = "Tech"
    snap = p[p["date"] == p["date"].max()]
    ticks = list(snap["ticker"])[:3]
    monkeypatch.setattr(cf, "latest_counterfactuals", lambda panel, specs: {"per_ticker": {
        ticks[0]: {"fragile": True, "worst_case": "rate_shock"}, ticks[1]: {"fragile": False}, ticks[2]: {"fragile": False}}})
    (tmp_path / "next_validation.json").write_text(json.dumps({"blind_spot_clusters": [
        {"common_properties": {"liquidity": "liquid" if snap.set_index("ticker").at[ticks[2], "log_dollar_vol"] > 0
                               else "less_liquid"}}]}))
    cands = [{"ticker": t, "prob": 0.6, "asymmetry": 1.5, "card": _card()} for t in ticks]
    kept, rej = hc.intelligence_checks(cands, p, [{"id": "x"}], tmp_path)
    rt = {r["ticker"]: r["reasons"] for r in rej}
    assert "fragil" in rt[ticks[0]][0]
    assert any("Unknown-Risk" in x for x in rt[ticks[2]])
    assert all(k["ticker"] == ticks[1] for k in kept)

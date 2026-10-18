"""V3/V4 als SHADOW (2026-10-18): korrigierte Werte laufen neben den produktiven mit und entscheiden nichts.

V3: Options-P&L-MC mit der Drift-Regel von run_for_dte ohne 6-Monats-Trend (nur Signal-Alpha, abklingend).
V4: Berührwahrscheinlichkeit desselben Ziels mit Handelstagen statt Kalendertagen als Simulationsschritten."""
from __future__ import annotations

import math

import numpy as np
import pytest

import pipeline
from modules import mirofish_simulation as mf
from modules import options_designer as od

ADM = {"strike": 77.5, "bid": 11.5, "ask": 12.4, "implied_vol": 0.3377, "dte": 178, "net_debit": 4.9,
       "spread_leg": {"strike": 85.0, "bid": 7.5, "ask": 8.0}}


def _cand(sigma=0.0177, impact=6, surprise=5, ttm="2-3 Monate"):
    return {"ticker": "ADM", "simulation": {"current_price": 84.39}, "features": {"sigma_30d": sigma},
            "deep_analysis": {"impact": impact, "surprise": surprise, "time_to_materialization": ttm}}


# ── V4 ──────────────────────────────────────────────────────────────────────

def test_v4_uses_trading_days_and_lowers_touch_probability():
    s = mf.MirofishSimulation()
    s.shadow_rng = np.random.default_rng(11)
    v4 = s.shadow_hit_rate_trading_days(100.0, 108.0, 0.02, 0.0, 0.016, 45, 40000)
    assert v4["steps"] == 31 and v4["unit"] == "trading_days" and v4["calendar_days"] == 45
    s.shadow_rng = np.random.default_rng(11)
    as_calendar = s.shadow_hit_rate_trading_days(100.0, 108.0, 0.02, 0.0, 0.016, round(45 * 365 / 252), 40000)
    assert as_calendar["steps"] == 45                                   # so rechnet die Produktion heute
    assert v4["hit_rate"] < as_calendar["hit_rate"] - 0.05              # Streuung √(45/31) zu hoch
    assert s.shadow_hit_rate_trading_days(100.0, 108.0, 0.02, 0.01, 0.0, 45, 2000, cap=0.75)["hit_rate"] == 0.75


def test_v4_shadow_in_run_for_dte_does_not_change_production(monkeypatch):
    monkeypatch.setattr(mf, "_get_hist_params", lambda t: (0.02, 0.001))
    mf.PARAM_SOURCE["ADM"] = "measured"
    a = mf.MirofishSimulation()
    a.rng = np.random.default_rng(7)
    ra = a.run_for_dte(_cand(), days_to_expiry=45)
    b = mf.MirofishSimulation()
    b.rng = np.random.default_rng(7)
    monkeypatch.setattr(mf.MirofishSimulation, "shadow_hit_rate_trading_days",
                        lambda *x, **k: (_ for _ in ()).throw(RuntimeError("aus")))
    rb = b.run_for_dte(_cand(), days_to_expiry=45)
    assert ra["simulation"]["hit_rate"] == rb["simulation"]["hit_rate"]       # eigener Zufallsstrom
    assert ra["simulation"]["shadow_v4"]["steps"] == 31 and rb["simulation"]["shadow_v4"] is None


# ── V3 ──────────────────────────────────────────────────────────────────────

def test_v3_shadow_removes_trend_drift_on_same_paths(monkeypatch):
    monkeypatch.setattr(mf, "_get_hist_params", lambda t: (0.0177, 0.003))
    s = mf.MirofishSimulation()
    s.rng = np.random.default_rng(5)
    r = s.simulate_option_pnl(_cand(), ADM, 178, history={}, n_paths=4000)
    v3 = r["shadow_v3"]
    _, alpha, decay = mf._signal_terms(_cand()["deep_analysis"])
    assert v3["cum_drift_prod"] == pytest.approx((0.003 + alpha) * 252 * r["hold_days"] / 365, abs=1e-4)
    assert 0 < v3["cum_drift_shadow"] < alpha * 252 * r["hold_days"] / 365          # nur Signal, abklingend
    assert v3["mean_pnl_pct"] < r["mean_pnl_pct"] and v3["median_pnl_pct"] <= r["expected_pnl_pct"]
    assert v3["mean_pnl_pct"] <= 7.5 / 4.9 - 1                                     # Spread-Cap gilt auch hier


def test_v3_shadow_does_not_change_production_values(monkeypatch):
    monkeypatch.setattr(mf, "_get_hist_params", lambda t: (0.0177, 0.003))
    s = mf.MirofishSimulation()
    s.rng = np.random.default_rng(5)
    r = s.simulate_option_pnl(_cand(), ADM, 178, history={}, n_paths=3000)
    monkeypatch.setattr(mf, "_signal_terms", lambda da: (_ for _ in ()).throw(RuntimeError("aus")))
    s2 = mf.MirofishSimulation()
    s2.rng = np.random.default_rng(5)
    r2 = s2.simulate_option_pnl(_cand(), ADM, 178, history={}, n_paths=3000)
    assert r2["shadow_v3"] is None
    assert r2["mean_pnl_pct"] == pytest.approx(r["mean_pnl_pct"], rel=1e-9)


def test_signal_terms_match_previous_rule():
    st, alpha, decay = mf._signal_terms({"impact": 4, "surprise": 9, "time_to_materialization": "4-8 Wochen"})
    assert st == pytest.approx(math.sqrt(0.4 * 0.9)) and alpha == pytest.approx(st * mf.MAX_SIGNAL_ALPHA)
    assert decay == mf.NARRATIVE_DECAY["short"]
    assert mf._signal_terms({"time_to_materialization": "6 Monate"})[2] == mf.NARRATIVE_DECAY["long"]
    assert mf._signal_terms(None) == pytest.approx((0.5, 0.5 * mf.MAX_SIGNAL_ALPHA, mf.NARRATIVE_DECAY["medium"]))


# ── Designer / Pipeline: nur protokolliert ──────────────────────────────────

def test_designer_records_v3_shadow_without_changing_decision(monkeypatch):
    from tests.test_options_designer_behaviour import FakeTicker, Gates, _exp, _signal, _yf
    monkeypatch.delenv("TRADIER_API_KEY", raising=False)
    monkeypatch.setattr(od, "get_macro_context", lambda: {"vix_term_structure": {"structure": "contango"}})
    monkeypatch.setattr(od.MirofishSimulation, "simulate_option_pnl",
                        lambda self, **k: {"expected_pnl_pct": 0.25, "mean_pnl_pct": 0.6, "pnl_std": 0.8,
                                           "hold_days": 100, "ou_method": "heuristic", "ou_n_days": 0,
                                           "instrument": "LONG_CALL",
                                           "shadow_v3": {"mean_pnl_pct": -0.2, "median_pnl_pct": -0.3,
                                                         "cum_drift_prod": 0.1, "cum_drift_shadow": 0.02,
                                                         "version": "v3-shadow-1"}})
    d = od.OptionsDesigner(Gates(), history={})
    _yf(monkeypatch, {"default": FakeTicker(options=[_exp(30), _exp(90), _exp(200)])})
    out = d.run([_signal()])
    assert len(out) == 1                                                       # Produktion entscheidet wie gehabt
    roi = out[0]["roi_analysis"]
    costs = roi["spread_pct"] * 2 + roi["commission_pct"]
    assert roi["shadow_v3"]["passes"] is False and roi["passes_roi_gate"] is True
    assert roi["shadow_v3"]["mean_net"] == pytest.approx(-0.2 - costs, abs=1e-4)
    audit = pipeline._roi_audit(roi)
    assert audit["shadow_v3"] == roi["shadow_v3"] and audit["roi_model"] == "ev-v2"


def test_roi_audit_handles_missing():
    assert pipeline._roi_audit(None) is None
    assert pipeline._roi_audit({"roi_net": 0.1, "x": 1}) == {"roi_net": 0.1}

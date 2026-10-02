"""Audit P1-8: Verhaltenstests der Monte-Carlo-Simulation (mirofish) – ohne Netz."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from modules import mirofish_simulation as mf


@pytest.fixture
def sim(monkeypatch):
    def params(ticker):
        mf.PARAM_SOURCE[ticker] = {"NODATA": "default_no_data", "ERR": "default_error"}.get(ticker, "measured")
        return (0.02, 0.0) if ticker not in ("HIGHVOL",) else (0.06, 0.0)
    monkeypatch.setattr(mf, "_get_hist_params", params)
    s = mf.MirofishSimulation()
    s.rng = np.random.default_rng(42)
    return s


def _cand(t="AAA", price=100.0, impact=6, surprise=5, ttm="medium"):
    return {"ticker": t, "simulation": {"current_price": price},
            "deep_analysis": {"impact": impact, "surprise": surprise, "time_to_materialization": ttm}}


def test_dynamic_target_floor_and_vol_scaling():
    assert mf._compute_dynamic_target(100, 0.01, 120) == pytest.approx(108.0)
    assert mf._compute_dynamic_target(100, 0.04, 120) == pytest.approx(100 * (1 + 0.5 * 0.04 * math.sqrt(120)))


def test_run_for_dte_basic_and_deterministic(sim):
    r = sim.run_for_dte(_cand(), days_to_expiry=30)
    s = r["simulation"]
    assert 0.0 <= s["hit_rate"] <= mf.HIT_RATE_CAP_SHORT and s["n_paths"] == mf.QUICK_MC_PATHS
    assert s["sigma_source"] == "measured" and s["target_price"] == pytest.approx(108.0)


def test_stronger_signal_raises_hit_rate(sim):
    weak = sim.run_for_dte(_cand(impact=1, surprise=1), days_to_expiry=60)["simulation"]["hit_rate"]
    sim.rng = np.random.default_rng(42)
    strong = sim.run_for_dte(_cand(impact=10, surprise=10), days_to_expiry=60)["simulation"]["hit_rate"]
    assert strong > weak


def test_no_price_or_no_history_returns_none(sim, monkeypatch):
    import types
    monkeypatch.setattr(mf.yf, "Ticker", lambda t: types.SimpleNamespace(info={}))
    assert sim.run_for_dte(_cand(price=0.0)) is None
    assert sim.run_for_dte(_cand(t="NODATA")) is None          # keine Trefferquote aus Default-σ
    assert sim.run_for_dte(_cand(t="ERR")) is None


def test_hit_rate_is_capped(sim):
    r = sim.run_for_dte({**_cand(), "simulation": {"current_price": 100.0}}, days_to_expiry=400)
    assert r["simulation"]["hit_rate"] <= mf.HIT_RATE_CAP_LONG


def test_hist_params_measured_clamped_and_missing(monkeypatch):
    mf._get_hist_params.cache_clear() if hasattr(mf._get_hist_params, "cache_clear") else None
    rng = np.random.default_rng(0)
    good = pd.DataFrame({"Close": 100 * np.cumprod(1 + rng.normal(0, 0.02, 120))})
    wild = pd.DataFrame({"Close": 100 * np.cumprod(np.tile([1.5, 1 / 1.5], 60))})     # σ ≈ 45 %/Tag
    for name, df, src in (("G1", good, "measured"), ("W1", wild, "clamped"), ("E1", pd.DataFrame(), "default_no_data")):
        monkeypatch.setattr(mf.yf, "download", lambda *a, d=df, **k: d)
        sigma, mu = mf._get_hist_params.__wrapped__(name)
        assert mf.PARAM_SOURCE[name] == src
        assert 0.005 <= sigma <= 0.15 and -0.005 <= mu <= 0.005

    def boom(*a, **k):
        raise RuntimeError("down")
    monkeypatch.setattr(mf.yf, "download", boom)
    mf._get_hist_params.__wrapped__("X1")
    assert mf.PARAM_SOURCE["X1"] == "default_error"


def test_black_scholes_call_properties(sim):
    S = np.array([80.0, 100.0, 120.0])
    c = sim._black_scholes_call(S, 100.0, 0.5, 0.3)
    assert np.all(np.diff(c) > 0) and np.all(c >= np.maximum(S - 100, 0))
    assert np.allclose(sim._black_scholes_call(S, 100.0, 0.0, 0.3), np.maximum(S - 100, 0))


def test_gbm_paths_shape_and_start(sim):
    p = sim._generate_gbm_paths(50.0, 0.0, 0.2, 0.5, 200, 30)
    assert p.shape == (200, 31) and np.all(p[:, 0] == 50.0) and np.all(p > 0)
    assert sim._generate_gbm_paths(50.0, 0.0, 0.2, 0.5, 5, 0).shape == (5, 1)


def test_ou_calibration_and_heuristic(sim):
    rng = np.random.default_rng(1)
    iv = [0.3]
    for _ in range(80):
        iv.append(iv[-1] + 2.0 * (0.3 - iv[-1]) / 365 + 0.2 * rng.normal() / math.sqrt(365))
    ou = sim._calibrate_ou_from_history(iv)
    assert ou["method"] == "regression" and 0.3 <= ou["kappa"] <= 5.0 and 0.08 <= ou["theta"] <= 0.85
    assert sim._calibrate_ou_from_history(iv[:10]) is None
    h = sim._get_ou_parameters("T", {"iv_history": {}}, 0.4, 80)
    assert h["method"] == "heuristic" and h["kappa"] == 1.8
    reg = sim._get_ou_parameters("T", {"iv_history": {"T": [{"atm_iv": v} for v in iv]}}, 0.4, 20)
    assert reg["method"] == "regression"


def test_iv_paths_floor(sim):
    p = sim._generate_iv_paths(100, 20, 0.06, {"kappa": 0.5, "theta": 0.05, "sigma_v": 0.6})
    assert p.shape == (100, 21) and p.min() >= 0.05


def test_log_iv_today_no_duplicates(sim):
    h = {}
    sim._log_iv_today("T", 0.31, h)
    sim._log_iv_today("T", 0.35, h)
    assert len(h["iv_history"]["T"]) == 1


def test_simulate_option_pnl_valid_and_invalid(sim):
    cand = {**_cand(), "features": {"sigma_30d": 0.02}}
    opt = {"strike": 105.0, "ask": 4.0, "implied_vol": 0.3}
    r = sim.simulate_option_pnl(cand, opt, 60, history={}, n_paths=500)
    assert set(r) >= {"expected_pnl_pct", "hit_rate", "ou_method", "hold_days"} and r["hold_days"] == 30
    assert 0.0 <= r["hit_rate"] <= 1.0 and r["expected_pnl_pct"] >= -1.0
    assert sim.simulate_option_pnl(cand, {**opt, "ask": 0}, 60)["error"] == "invalid_input"
    assert sim.simulate_option_pnl(cand, opt, 3)["error"] == "invalid_input"
    assert sim.simulate_option_pnl({"ticker": "X"}, opt, 60)["error"] == "invalid_input"


def test_time_value_efficiency():
    r = mf.compute_time_value_efficiency(0.2, 100)
    assert r["roi_per_day_pct"] == pytest.approx(0.2) and r["annualized_roi"] is not None
    assert mf.compute_time_value_efficiency(3.0, 10)["annualized_roi"] is None     # unrealistisch -> None
    assert mf.compute_time_value_efficiency(0.1, 0)["dte"] == 1

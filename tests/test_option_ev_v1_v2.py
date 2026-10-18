"""V1/V2 (2026-10-18): Spread-korrekte MC-P&L und Vorfilter-ROI als Erwartungswert.

V1: simulate_option_pnl bewertet einen Bull Call Spread mit beiden Beinen ab dem Debit (Long-Ask − Short-Bid),
    Wert auf die Breite begrenzt. Vorher Long-Call ab Long-Ask -> Gate-ROI über dem Maximalertrag.
V2: _compute_roi = p · Gewinn (Netto-Delta, Cap) − (1 − p) · Zeitwertverlust − Round Trip der Kombination."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from modules import mirofish_simulation as mf
from modules import options_designer as od

ROOT = Path(__file__).resolve().parent.parent
ADM = {"strike": 77.5, "bid": 11.5, "ask": 12.4, "implied_vol": 0.3377, "dte": 178, "net_debit": 4.9,
       "spread_leg": {"strike": 85.0, "bid": 7.5, "ask": 8.0}}
S_ADM = 84.39


def _designer():
    d = od.OptionsDesigner.__new__(od.OptionsDesigner)
    d._vix_ts = {}
    return d


def _cand(sigma_daily, impact=1, surprise=1, ticker="ADM", price=S_ADM):
    return {"ticker": ticker, "simulation": {"current_price": price}, "features": {"sigma_30d": sigma_daily},
            "deep_analysis": {"impact": impact, "surprise": surprise}}


# ── Hilfsfunktionen ─────────────────────────────────────────────────────────

def test_implied_vol_roundtrip_and_bounds():
    p = mf.bs_call_r0(100, 105, 0.5, 0.32)
    assert mf.implied_vol_r0(p, 100, 105, 0.5) == pytest.approx(0.32, abs=1e-6)
    assert mf.implied_vol_r0(4.0, 100, 95, 0.5) is None          # unter innerem Wert (5)
    assert mf.implied_vol_r0(0.0, 100, 95, 0.5) is None


def test_spread_terms_valid_and_invalid():
    t = mf.bull_call_spread_terms(ADM, S_ADM, 178 / 365)
    assert t["debit"] == 4.9 and t["width"] == 7.5 and t["combo_spread"] == pytest.approx(1.4)
    assert mf.bs_call_r0(S_ADM, 77.5, 178 / 365, t["iv_long"]) == pytest.approx(11.95, abs=1e-3)   # Long-Mid
    assert mf.bs_call_r0(S_ADM, 85.0, 178 / 365, t["iv_short"]) == pytest.approx(7.75, abs=1e-3)   # Short-Mid
    assert mf.bull_call_spread_terms({**ADM, "spread_leg": {}}, S_ADM, 0.5) is None                   # kein Bein
    assert mf.bull_call_spread_terms({**ADM, "spread_leg": {"strike": 70.0, "bid": 1, "ask": 2}}, S_ADM, 0.5) is None
    assert mf.bull_call_spread_terms({**ADM, "net_debit": 0}, S_ADM, 0.5) is None


# ── V1: MC-P&L ──────────────────────────────────────────────────────────────

def test_spread_mc_starts_at_debit_and_without_drift_loses_the_crossing(monkeypatch):
    sig = 0.3377 / math.sqrt(252)                                 # realisierte Vola = IV, Drift 0
    monkeypatch.setattr(mf, "_get_hist_params", lambda t: (sig, 0.0))
    s = mf.MirofishSimulation()
    s.rng = np.random.default_rng(3)
    r = s.simulate_option_pnl(_cand(sig), ADM, 178, history={}, n_paths=20000)
    assert r["instrument"] == "BULL_CALL_SPREAD" and r["entry_price"] == 4.9
    # Mid 4,20 gegen Debit 4,90 -> −14,3 %; ohne Edge bleibt der Erwartungswert in dieser Größenordnung negativ
    assert -0.25 < r["mean_pnl_pct"] < -0.05
    assert r["mean_pnl_pct"] <= 7.5 / 4.9 - 1


def test_spread_mc_pnl_bounded_even_with_strong_drift(monkeypatch):
    monkeypatch.setattr(mf, "_get_hist_params", lambda t: (0.0177, 0.005))
    s = mf.MirofishSimulation()
    s.rng = np.random.default_rng(1)
    r = s.simulate_option_pnl(_cand(0.0177, 10, 10), ADM, 178, history={}, n_paths=5000)
    assert -1.0 <= r["expected_pnl_pct"] <= r["mean_pnl_pct"] + 1 and r["mean_pnl_pct"] <= 7.5 / 4.9 - 1


def test_long_call_path_unchanged(monkeypatch):
    monkeypatch.setattr(mf, "_get_hist_params", lambda t: (0.02, 0.0))
    opt = {k: v for k, v in ADM.items() if k not in ("spread_leg", "net_debit")}
    s = mf.MirofishSimulation()
    s.rng = np.random.default_rng(2)
    r = s.simulate_option_pnl(_cand(0.02), opt, 178, history={}, n_paths=2000)
    assert r["instrument"] == "LONG_CALL" and r["entry_price"] == 12.4


# ── V2: Vorfilter-ROI ───────────────────────────────────────────────────────

def test_compute_roi_spread_cap_net_delta_and_combo_costs():
    roi = _designer()._compute_roi(ADM, {"current_price": S_ADM, "target_price": S_ADM * 1.5, "days": 45}, 40,
                                   {"min_roi": 0.1}, "BULL_CALL_SPREAD", 0.9, "OTHER")
    assert roi["is_spread"] and roi["roi_model"] == "ev-v2"
    assert roi["gain_if_hit"] == pytest.approx(7.5 / 4.9 - 1, abs=1e-4)          # Cap greift
    assert roi["combo_spread_pct"] == pytest.approx(1.4 / 4.9, abs=1e-4)          # (12,4−7,5) − (11,5−8,0)
    assert roi["roundtrip_cost_pct"] == roi["combo_spread_pct"]
    assert 0 < roi["delta"] < 0.3                                                 # Long ~0,7 − Short ~0,55
    assert roi["roi_net"] <= roi["max_gain"]
    assert roi["commission_pct"] == pytest.approx(0.65 * 4 / 490, abs=1e-4)


def test_compute_roi_loss_branch_is_expected_value():
    opt = {"bid": 4.8, "ask": 5.0, "strike": 100, "implied_vol": 0.30, "dte": 120}
    roi = _designer()._compute_roi(opt, {"current_price": 100, "target_price": 110, "days": 45}, 50,
                                   {"min_roi": 0.1}, "LONG_CALL", 0.5, "OTHER")
    assert roi["loss_if_miss"] > 0 and roi["hold_days_assumed"] == 45
    assert roi["roi_gross"] == pytest.approx(0.5 * roi["gain_if_hit"] - 0.5 * roi["loss_if_miss"], abs=2e-4)
    assert roi["loss_if_miss"] == pytest.approx(abs(roi["theta_daily"]) * 45 / 5.0, abs=2e-3)


def test_spread_without_short_leg_is_a_long_call():
    opt = {k: v for k, v in ADM.items() if k != "spread_leg"}
    roi = _designer()._compute_roi(opt, {"current_price": S_ADM, "target_price": S_ADM * 1.08}, 40,
                                   {"min_roi": 0.1}, "BULL_CALL_SPREAD", 0.7, "OTHER")
    assert not roi["is_spread"] and roi["cost_basis"] == 12.4
    assert roi["commission_pct"] == pytest.approx(0.65 * 2 / 1240, abs=1e-4)       # 1 Leg, nicht 4 Fills (gerundet)


def test_real_champion_spreads_no_longer_pass_with_impossible_roi():
    path = ROOT / "outputs" / "history.json"
    if not path.exists():
        pytest.skip("outputs/history.json fehlt")
    trades = [t for t in json.loads(path.read_text())["closed_trades"]
              if t.get("outcome_method") and t.get("strategy") == "BULL_CALL_SPREAD"]
    if len(trades) < 3:
        pytest.skip("zu wenige Champion-Spreads")
    d = _designer()
    for t in trades:
        o, sim = t["option"], t["simulation"]
        p = min(0.95, (t.get("features") or {}).get("quick_mc_hit_rate") or sim["hit_rate"])
        roi = d._compute_roi(o, sim, 50, {"min_roi": 0.10}, t["strategy"], p, t.get("catalyst_type", "OTHER"))
        produced = t["tve"]["roi_per_day_pct"] * t["tve"]["dte"] / 100        # damaliger Gate-Wert
        assert roi["is_spread"] and roi["roi_net"] <= roi["max_gain"]
        assert roi["roi_net"] < produced and roi["roi_net"] < 0.10, t["ticker"]  # Round Trip frisst den Vorteil

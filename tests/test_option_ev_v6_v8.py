"""V6–V8 (2026-10-18): Kelly aus der P&L-Verteilung, echte Quotes im Liquiditätsfilter, ein Horizont."""
from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

from modules import mirofish_simulation as mf
from modules import options_designer as od
from modules import position_sizing as ps
from modules import reporter

ADM = {"strike": 77.5, "bid": 11.5, "ask": 12.4, "implied_vol": 0.3377, "dte": 178, "net_debit": 4.9,
       "spread_leg": {"strike": 85.0, "bid": 7.5, "ask": 8.0}}


def _cand():
    return {"ticker": "ADM", "simulation": {"current_price": 84.39}, "features": {"sigma_30d": 0.0177},
            "deep_analysis": {"impact": 5, "surprise": 5}}


# ── V6: Kelly ───────────────────────────────────────────────────────────────

def test_kelly_inputs_from_net_distribution(monkeypatch):
    monkeypatch.setattr(mf, "_get_hist_params", lambda t: (0.0177, 0.0))
    runs = {}
    for cost in (0.0, 0.2):
        s = mf.MirofishSimulation()
        s.rng = np.random.default_rng(4)
        runs[cost] = s.simulate_option_pnl(_cand(), ADM, 178, history={}, n_paths=4000, cost_pct=cost)
    k0, k2 = runs[0.0]["kelly_inputs"], runs[0.2]["kelly_inputs"]
    assert 0 <= k2["p_win"] < k0["p_win"] <= 1 and k2["cost_pct"] == 0.2
    assert 0 < k0["avg_loss"] <= 1.0 and k0["avg_win"] > 0
    assert runs[0.2]["mean_pnl_pct"] == pytest.approx(runs[0.0]["mean_pnl_pct"])   # Brutto-Werte unverändert


def test_kelly_formula_and_risk_at_full_debit():
    p = {"ticker": "T", "strategy": "BULL_CALL_SPREAD", "option": {"ask": 3.0, "net_debit": 1.5},
         "roi_analysis": {"kelly_inputs": {"p_win": 0.45, "avg_win": 0.9, "avg_loss": 0.6}}}
    s = ps.compute_position_size(p)
    assert s["kelly_raw"] == pytest.approx(0.45 / 0.6 - 0.55 / 0.9, abs=1e-4)
    assert s["max_risk_usd"] == pytest.approx(s["contracts"] * 150.0)
    assert ps.compute_position_size({**p, "roi_analysis": {"kelly_inputs": {"p_win": 0.3, "avg_win": 0.5,
                                                                              "avg_loss": 0.9}}})["contracts"] == 0


def test_size_multiplier_scales_risk_without_loss_assumption():
    base = {"position_pct": 0.04, "position_usd": 1000.0, "contracts": 4, "cost_per_contract": 250.0,
            "max_risk_usd": 1000.0}
    half = ps.apply_size_multiplier(base, 0.5)
    assert half["contracts"] == 2 and half["max_risk_usd"] == 500.0


# ── V7: Liquiditätsfilter ───────────────────────────────────────────────────

def _chain(zero_strike):
    rows = []
    for k in np.arange(80.0, 125.0, 5.0):
        ask = round(max(100 - k, 0) + 8.0, 2)
        bid, oi = round(ask * 0.95, 2), 300
        if k == zero_strike:
            bid = ask = 0.0
            oi = 9000                                              # höchstes OI -> wurde früher gewählt
        rows.append({"strike": k, "bid": bid, "ask": ask, "openInterest": oi, "impliedVolatility": 0.35})
    calls = pd.DataFrame(rows)
    return types.SimpleNamespace(calls=calls, puts=calls.copy())


def test_zero_quote_contract_never_counts_as_liquid(monkeypatch):
    from tests.test_options_designer_behaviour import FakeTicker, Gates, _exp
    monkeypatch.delenv("TRADIER_API_KEY", raising=False)
    monkeypatch.setattr(od, "get_macro_context", lambda: {"vix_term_structure": {"structure": "contango"}})
    d = od.OptionsDesigner(Gates(), history={})
    tk = FakeTicker(options=[_exp(200)], chain=_chain(100.0))
    opt = d._find_option_for_dte("ABC", "LONG_CALL", 100.0, 150, 365, tk)
    assert opt is not None and opt["strike"] != 100.0 and opt["bid"] > 0 and opt["ask"] >= opt["bid"]


# ── V8: ein Horizont ────────────────────────────────────────────────────────

def test_simulation_hold_days_capped_by_parameter(monkeypatch):
    monkeypatch.setattr(mf, "_get_hist_params", lambda t: (0.0177, 0.0))
    s = mf.MirofishSimulation()
    assert s.simulate_option_pnl(_cand(), ADM, 178, history={}, n_paths=500, hold_days=45)["hold_days"] == 45
    assert s.simulate_option_pnl(_cand(), ADM, 178, history={}, n_paths=500)["hold_days"] == 89   # Altverhalten


def test_designer_uses_max_hold_final_mc_probability_and_same_costs(monkeypatch):
    from tests.test_options_designer_behaviour import FakeTicker, Gates, _exp, _signal, _yf
    monkeypatch.delenv("TRADIER_API_KEY", raising=False)
    monkeypatch.setattr(od, "get_macro_context", lambda: {"vix_term_structure": {"structure": "contango"}})
    seen = {}

    def fake(self, **k):
        seen.update(k)
        return {"expected_pnl_pct": 0.25, "mean_pnl_pct": 0.6, "pnl_std": 0.8, "hold_days": k.get("hold_days"),
                "ou_method": "heuristic", "ou_n_days": 0, "instrument": "LONG_CALL",
                "kelly_inputs": {"p_win": 0.5, "avg_win": 0.9, "avg_loss": 0.5, "cost_pct": k.get("cost_pct")}}
    monkeypatch.setattr(od.MirofishSimulation, "simulate_option_pnl", fake)
    d = od.OptionsDesigner(Gates(), history={})
    _yf(monkeypatch, {"default": FakeTicker(options=[_exp(30), _exp(90), _exp(200)])})
    sig = _signal(simulation={"current_price": 100.0, "target_price": 140.0, "hit_rate": 0.55},
                  quick_mc={"hit_rate": 0.70})
    out = d.run([sig])
    assert len(out) == 1
    roi = out[0]["roi_analysis"]
    assert seen["hold_days"] == min(max(5, seen["days_to_expiry"] // 2), od.MAX_HOLD_DAYS) == od.MAX_HOLD_DAYS
    assert roi["mc_weight"] == 0.55 and out[0]["mc_hit_rate"] == 0.55               # Final-MC, nicht Quick-MC
    assert roi["hold_days_assumed"] == od.MAX_HOLD_DAYS and roi["mc_hold_days"] == od.MAX_HOLD_DAYS
    exp_cost = (roi["combo_spread_pct"] / 2 if roi["is_spread"] else roi["spread_pct"] * 2) + roi["commission_pct"]
    assert seen["cost_pct"] == pytest.approx(exp_cost, abs=1e-3)                      # dieselben Kosten wie das Gate
    assert roi["kelly_inputs"]["p_win"] == 0.5


def test_report_shows_max_holding_and_same_time_exit_as_feedback():
    p = {"strategy": "BULL_CALL_SPREAD", "option": {"net_debit": 2.0, "ask": 3.0, "strike": 100, "dte": 180,
                                                      "expiry": "2027-04-16", "spread_leg": {"strike": 110}}}
    r = reporter.compute_exit_rules(p)
    assert r["time_exit_dte_remaining"] == 90 and r["max_hold_days"] == od.MAX_HOLD_DAYS
    assert r["max_hold_date"] != "N/A"

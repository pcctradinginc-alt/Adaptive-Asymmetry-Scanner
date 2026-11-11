"""Forensic P0 regression tests: deterministic scores, risk caps, timing and statistics."""
from datetime import date, datetime, timezone

import pytest

from modules import champion_protocol as cp
from modules import candidate_ledger as cl
from modules import position_sizing as ps
from modules import mirofish_simulation as sim


def test_zero_variance_is_not_false_significance():
    result = cp.one_sided_test([0.1, 0.1, 0.1], 0.01)
    assert result["p_value"] is None
    assert result["lower"] is None
    assert result["upper"] is None


def test_one_contract_cannot_exceed_hard_risk_limit():
    p = {"ticker": "TOO_EXPENSIVE", "strategy": "LONG_CALL",
         "option": {"ask": 7.5}, "roi_analysis": {}}
    s = ps.compute_position_size(p)
    assert s["contracts"] == 0
    assert s["max_risk_usd"] == 0
    assert s["exceeds_max_position_pct"]
    assert "REJECT" in s["note"]


def test_directional_entry_requires_final_decision_timestamp(monkeypatch):
    from modules import timing_policy as tp
    row = {"signal_timestamp": "2026-11-11T15:00:00+00:00",
           "direction": "BULLISH", "real_option": {"entry_pending": True,
           "symbol": "SPY261218C00500000"}}
    # Missing final decision timestamp must fail closed, never reuse universe time.
    assert cl._v2_entry(row, "real_option", datetime(2026, 11, 11, 17, tzinfo=timezone.utc))
    assert row["real_option"]["entry_status"] == "UNUSABLE"
    assert row["real_option"]["entry_unusable_reason"] == "SIGNAL_TIME_MISSING"


def test_directional_entry_uses_final_decision_not_universe_time(monkeypatch):
    from modules import timing_policy as tp
    captured = {}
    def fake_find(symbols, after, now):
        captured["after"] = after
        return {"status": "PENDING", "day": "2026-11-11"}
    monkeypatch.setattr(tp, "find_fill", fake_find)
    row = {"signal_timestamp": "2026-11-11T15:00:00+00:00",
           "decision_frozen_at": "2026-11-11T16:00:00+00:00",
           "direction": "BULLISH", "real_option": {"entry_pending": True,
           "symbol": "SPY261218C00500000"}}
    assert cl._v2_entry(row, "real_option", datetime(2026, 11, 11, 17, tzinfo=timezone.utc))
    assert captured["after"] == datetime(2026, 11, 11, 16, tzinfo=timezone.utc)


def test_mc_reproducible_with_same_frozen_inputs(monkeypatch):
    monkeypatch.setattr(sim, "QUICK_MC_PATHS", 64)
    monkeypatch.setattr(sim, "_get_hist_params", lambda ticker: (0.025, 0.0001))
    monkeypatch.setattr(sim, "PARAM_SOURCE", {"SPY": "measured"})
    candidate = {"ticker": "SPY", "current_price": 600.0,
                 "deep_analysis": {"impact": 5, "surprise": 4,
                                   "time_to_materialization": "short"}}
    first = sim.MirofishSimulation().run_for_dte(candidate, days_to_expiry=5)
    second = sim.MirofishSimulation().run_for_dte(candidate, days_to_expiry=5)
    assert first["simulation"]["hit_rate"] == second["simulation"]["hit_rate"]

"""SPREAD_EXECUTION_LIQUIDITY_GATE (SHADOW_ONLY, 2026-10-09). Auslöser SNPS: Entry 25.20, sofort ausführbar
15.60 -> 38 % Immediate Liquidation Loss; Spread-Stop -50 % feuerte nach −1,7 % im Underlying."""
from __future__ import annotations

from pathlib import Path

import pytest

from modules import spread_execution as se

ROOT = Path(__file__).resolve().parent.parent
CFG = se.load_cfg()
SNPS = {"strike": 500.0, "bid": 64.2, "ask": 70.3, "open_interest": 202, "expiry": "2027-03-19", "dte": 163,
        "net_debit": 25.2, "spread_leg": {"strike": 550.0, "bid": 45.1, "ask": 48.6}}
LIQUID = {"strike": 100.0, "bid": 10.00, "ask": 10.10, "open_interest": 5000, "expiry": "2027-03-19", "dte": 160,
          "net_debit": 4.20, "spread_leg": {"strike": 110.0, "bid": 5.90, "ask": 5.98, "open_interest": 4000}}


def test_snps_regression_entry_friction_detected_and_shadow_rejects():
    a = se.assess_entry(SNPS, {"roi_gross": 0.9, "roi_net": 0.6, "spread_pct": 0.0868, "vega_loss": 0.05}, CFG)
    assert a["combo_ask"] == 25.2 and a["combo_bid"] == 15.6 and a["combo_mid"] == 20.4
    assert a["immediate_liquidation_loss_pct"] == pytest.approx(0.381, abs=1e-3)           # 1. hohe Entry-Friction
    assert a["ill_bucket"] == ">=30%"                                                        # Extrem-Bucket
    assert a["quote_quality"] in ("VERY_WIDE", "WIDE")                                       # 2. problematisch
    assert a["shadow_verdict"] == "WOULD_REJECT"                                             # 3. Shadow-Gate
    assert a["production_friction_model"] == pytest.approx(0.1736, abs=1e-3)                 # Root Cause: 17 % statt 38 %
    assert a["net_expected_roi"] < a["production_roi_net"]
    assert a["gate_mode"] == "SHADOW_ONLY" and a["production_effect"].startswith("NONE")      # 5. keine Produktionswirkung


def test_snps_stop_not_loosened_and_production_rule_unchanged():
    import feedback
    tr = {"strategy": "BULL_CALL_SPREAD", "entry_debit": 25.2, "entry_date": "2026-10-06",
          "option": {**SNPS, "expiry": "2027-03-19"}}
    assert feedback.check_exit_rules(tr, -0.5079, feedback.datetime(2026, 10, 8)) == "stop_loss"   # 4. -50 % bleibt
    assert feedback.check_exit_rules(tr, -0.49, feedback.datetime(2026, 10, 8)) is None
    src = (ROOT / "feedback.py").read_text()
    assert "sl_threshold = -0.50" in src and "sl_threshold = -0.45" in src


def test_liquid_spread_no_false_reject():
    a = se.assess_entry(LIQUID, {"roi_gross": 0.5, "roi_net": 0.4, "spread_pct": 0.01, "vega_loss": 0.02}, CFG)
    assert a["immediate_liquidation_loss_pct"] < 0.05 and a["ill_bucket"] == "<5%"
    assert a["quote_quality"] == "HEALTHY" and a["shadow_verdict"] == "PASS"


def test_illiquid_spread_flagged():
    opt = {**LIQUID, "bid": 9.0, "ask": 11.0, "spread_leg": {"strike": 110.0, "bid": 5.0, "ask": 6.5}}
    a = se.assess_entry(opt, None, CFG)
    assert a["immediate_liquidation_loss_pct"] >= 0.30 and a["shadow_verdict"] == "WOULD_REJECT"
    assert a["quote_quality"] == "VERY_WIDE"


def test_stale_quote_no_reliable_stop():
    from datetime import datetime, timezone
    now = datetime(2026, 10, 8, 20, tzinfo=timezone.utc)
    trade = {"entry_debit": 25.2, "simulation": {"current_price": 506.21}}
    obs = se.monitor(trade, {"bid": 64.2, "ask": 70.3}, {"bid": 45.1, "ask": 48.6}, 497.75, CFG, now,
                     quote_ts="2026-10-08T17:00:00+00:00")                                  # 3 h alt
    assert obs["quote_quality"] == "STALE" and obs["executable_pnl_pct"] is None and obs["fair_value_pnl_pct"] is None
    assert se.stop_shadow([obs], cfg=CFG)["shadow_stop"] == "NO_RELIABLE_QUOTE"


def test_wide_quote_stable_underlying_is_execution_problem():
    trade = {"entry_debit": 25.2, "simulation": {"current_price": 506.21}}
    # SNPS am 08.10.: Underlying -1,7 %, Spread executable -50,8 %
    o1 = se.monitor(trade, {"bid": 63.0, "ask": 69.5}, {"bid": 44.4, "ask": 50.6}, 497.75, CFG)
    st = se.stop_shadow([o1], cfg=CFG)
    assert o1["executable_pnl_pct"] <= -0.50 and o1["fair_value_pnl_pct"] > -0.50
    assert st["production_stop_hit"] is True and st["shadow_stop"] == "EXECUTION_DRIVEN"


def test_true_collapse_keeps_stop_protection():
    trade = {"entry_debit": 25.2, "simulation": {"current_price": 506.21}}
    obs = [se.monitor(trade, {"bid": 30.0, "ask": 31.0}, {"bid": 19.0, "ask": 20.0}, 430.0, CFG) for _ in range(2)]
    st = se.stop_shadow(obs, cfg=CFG)
    assert obs[-1]["fair_value_pnl_pct"] <= -0.50 and obs[-1]["underlying_return_pct"] < -0.10
    assert st["production_stop_hit"] is True and st["shadow_stop"] == "STOP_CONFIRMED"


def test_missing_quote_fail_closed():
    a = se.assess_entry({**SNPS, "spread_leg": {}}, None, CFG)
    assert a["quote_quality"] == "UNAVAILABLE" and a["immediate_liquidation_loss_pct"] is None
    assert a["shadow_verdict"] == "WOULD_REJECT" and a["combo_mid"] is None
    obs = se.monitor({"entry_debit": 25.2}, None, None, None, CFG)
    assert obs["quote_quality"] == "UNAVAILABLE" and obs["executable_pnl_pct"] is None


def test_crossed_market_invalid():
    a = se.assess_entry({**SNPS, "bid": 71.0, "ask": 70.3}, None, CFG)
    assert a["quote_quality"] == "INVALID" and a["shadow_verdict"] == "WOULD_REJECT"


def test_buckets_and_need_more_data():
    assert [se.bucket(x, CFG) for x in (0.01, 0.07, 0.12, 0.17, 0.25, 0.38)] == \
        ["<5%", "5-10%", "10-15%", "15-20%", "20-30%", ">=30%"]
    res = se.analyse([{"strategy": "BULL_CALL_SPREAD", "option": SNPS, "outcome": -0.5, "close_reason": "stop_loss",
                       "outcome_method": "spread_quote", "close_price": 497.75,
                       "simulation": {"current_price": 506.21}}], CFG)
    b = res["buckets"][">=30%"]
    assert b["n_with_outcome"] == 1 and b["status"] == "NEED_MORE_DATA" and b["false_stop_frequency"] == 1.0


def test_pipeline_shadow_never_changes_proposals(tmp_path, monkeypatch):
    import copy
    import pipeline
    monkeypatch.setattr(se, "LEDGER_DIR", tmp_path)
    props = [{"ticker": "SNPS", "strategy": "BULL_CALL_SPREAD", "option": dict(SNPS),
              "roi_analysis": {"roi_gross": 0.9, "roi_net": 0.6, "spread_pct": 0.0868}}]
    before = copy.deepcopy(props)
    pipeline._spread_execution_shadow(props, [], "2026-10-06")
    assert len(props) == 1 and props[0]["option"] == before[0]["option"]                    # nichts entfernt/verändert
    assert props[0]["spread_execution"]["shadow_verdict"] == "WOULD_REJECT"
    rec = pipeline.build_trade_record(props[0], "2026-10-06")
    assert rec["entry_debit"] == 25.2 and rec["spread_execution_entry"]["immediate_liquidation_loss_pct"] == pytest.approx(0.381, abs=1e-3)
    assert any(tmp_path.glob("*.jsonl"))

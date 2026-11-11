"""Forensic Quant Audit P0 safety and provenance regressions."""
from datetime import datetime, timezone

from modules import candidate_ledger as cl
from modules import champion_protocol as cp
from modules import mirofish_simulation as mf
from modules import position_sizing as sz
from modules import research_evidence_discovery as red


def test_candidate_fill_anchor_is_frozen_decision_not_early_news(monkeypatch):
    from modules import timing_policy as tp
    afters = []

    def capture(symbols, after, now):
        afters.append(after)
        return {"status": "PENDING", "day": "2026-11-12"}

    monkeypatch.setattr(tp, "find_fill", capture)
    row = {
        "signal_timestamp": "2026-11-11T14:00:00+00:00",
        "decision_frozen_at": "2026-11-11T16:00:00+00:00",
        "entry_anchor_policy": "decision-frozen-v1",
        "real_option": {"entry_pending": True, "symbol": "AAPL270319C00200000"},
    }
    cl._v2_entry(row, "real_option", datetime(2026, 11, 11, 17, tzinfo=timezone.utc))
    assert afters == [datetime(2026, 11, 11, 16, tzinfo=timezone.utc)]


def test_frozen_provenance_rejects_predecision_quote():
    sym = "AAPL270319C00200000"
    row = {
        "date": "2026-11-11", "ticker": "AAPL",
        "entry_anchor_policy": "decision-frozen-v1",
        "decision_frozen_at": "2026-11-11T16:00:00+00:00",
        "real_strategy": {
            "legs": [{"symbol": sym, "side": "long", "bid": 9.9, "ask": 10.0, "mid": 9.95}],
            "entry_pending": False, "entry_date": "2026-11-11", "net_debit_entry": 10.0,
        },
        "outcomes": {"real_strat_ret_20d": 0.2, "real_strat_mark_date_20d": "2026-12-01"},
    }
    assert red.provenance(row, red.quote_index([])) == "ENTRY_NOT_ARCHIVED"
    row["real_strategy"]["entry_quote_record_ids"] = ["q1"]
    q = {
        "record_id": "q1", "occ_symbol": sym, "bid": 9.9, "ask": 10.0,
        "mid": 9.95, "quote_time": "2026-11-11T15:30:00+00:00",
        "retrieved_at": "2026-11-11T15:30:00+00:00",
        "quality_status": "OK", "quality_flags": [],
    }
    assert red.provenance(row, red.quote_index([q])) == "ENTRY_BEFORE_DECISION"


def test_unit_position_cap_is_enforced():
    pc = {"size_usd": 5000.0, "max_position_pct": 0.10}
    blocked = sz._unit_sizing({}, pc, 1000.0, {})
    assert blocked["exceeds_max_position_pct"] and blocked["contracts"] == 0
    assert blocked["position_usd"] == 0
    assert sz._unit_sizing({}, pc, 300.0, {})["contracts"] == 1


def test_constant_day_means_cannot_confirm_alpha():
    t = cp.one_sided_test([0.2] * 5, 0.01)
    assert t["p_value"] is None and t["lower"] is None


def test_mc_repeats_same_candidate_results(monkeypatch):
    def params(ticker):
        mf.PARAM_SOURCE[ticker] = "measured"
        return 0.02, 0.0

    monkeypatch.setattr(mf, "_get_hist_params", params)
    cand = {
        "ticker": "AAA", "simulation": {"current_price": 100.0},
        "deep_analysis": {"impact": 6, "surprise": 5, "time_to_materialization": "medium"},
    }
    a = mf.MirofishSimulation().run_for_dte(cand, days_to_expiry=30)["simulation"]
    b = mf.MirofishSimulation().run_for_dte(cand, days_to_expiry=30)["simulation"]
    assert a["hit_rate"] == b["hit_rate"] and a["rng_seed"] == b["rng_seed"]
    assert a["trading_steps"] == round(30 * 252 / 365)


def test_v3_entry_cohort_keeps_open_and_unusable_positions_in_denominator():
    from datetime import date
    history = {
        "active_trades": [{"ticker": "AAA", "entry_date": "2026-11-12", "decision_time": "2026-11-12T15:00:00Z"}],
        "closed_trades": [{"ticker": "BBB", "entry_date": "2026-11-13",
                           "outcome_method": "option_quote", "outcome_net": 0.2}],
        "closed_trades_unusable": [{"ticker": "CCC", "entry_date": "2026-11-14"}],
    }
    cohort = cp.frozen_entry_cohort(history, date(2026, 11, 12))
    assert [x["ticker"] for x in cohort] == ["AAA", "BBB", "CCC"]
    assert [x["reliable"] for x in cohort] == [False, True, False]

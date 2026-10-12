"""Options-Outcome-Kreislauf (2026-10-11): Bewertungsstatus geschlossener Trades, Quote-Archiv-Verknüpfung,
Lernen nur bei OK. Technische Tests mit synthetischen Quotes – kein wirtschaftlicher Nachweis."""
from __future__ import annotations

import ast
import inspect
from datetime import datetime, timezone

import feedback
from modules import option_quotes as oq
from modules.atomic_io import read_jsonl
from modules.outcomes import effective_entry_debit, trade_evaluation

UTC = timezone.utc
RTH = "2026-10-08T18:30:00+00:00"          # 14:30 ET
POST = "2026-10-15T20:40:00+00:00"         # 16:40 ET (Feedback-Lauf nach Handelsschluss)
TRADE = {"ticker": "AAPL", "strategy": "LONG_CALL", "entry_date": "2026-10-08", "entry_debit": 2.0,
         "option": {"strike": 200.0, "expiry": "2026-11-20", "ask": 2.0}, "outcome": 0.25}


def q(status="OK", t=RTH, rid="r1"):
    return {"record_id": rid, "quality_status": status, "quote_time": t, "retrieved_at": t}


def test_observed_rth_reproducible_is_ok_top_tier():
    ev = trade_evaluation(TRADE, price_type="OBSERVED_QUOTE", entry_quotes=[q()], exit_quotes=[q(rid="r2")])
    assert ev["status"] == "OK" and ev["tier"] == "REALIZED_RTH_REPRODUCIBLE" and ev["reproducible"]
    assert ev["entry_session"] == ev["exit_session"] == "RTH" and ev["reasons"] == []


def test_exit_after_close_is_ok_but_flagged_not_rth():
    ev = trade_evaluation(TRADE, price_type="OBSERVED_QUOTE", entry_quotes=[q()], exit_quotes=[q(t=POST)])
    assert ev["status"] == "OK" and ev["tier"] == "OBSERVED" and "EXIT_OUTSIDE_RTH" in ev["flags"]


def test_loss_with_observed_quotes_is_a_real_loss_not_unusable():
    ev = trade_evaluation({**TRADE, "outcome": -1.0}, price_type="OBSERVED_QUOTE", entry_quotes=[q()],
                          exit_quotes=[q(rid="r2")])
    assert ev["status"] == "OK"                                              # -100 % ist ein echter Verlust


def test_modeled_or_proxy_exit_is_unusable_with_reason():
    for pt, why in (("MODELED", "EXIT_PRICE_MODELED"), ("UNDERLYING_PROXY", "EXIT_PRICE_UNDERLYING_PROXY"),
                    (None, "EXIT_PRICE_UNKNOWN")):
        ev = trade_evaluation(TRADE, price_type=pt, entry_quotes=[q()], exit_quotes=[None])
        assert ev["status"] == "UNUSABLE" and ev["reasons"] == [why] and ev["tier"] == "NOT_EVALUABLE"


def test_data_bad_on_invalid_entry():
    bad = {**TRADE, "entry_debit": 0, "option": {"strike": 200.0}}
    assert effective_entry_debit(bad) == 0
    ev = trade_evaluation(bad, price_type="OBSERVED_QUOTE", entry_quotes=[q()], exit_quotes=[q()])
    assert ev["status"] == "DATA_BAD" and "ENTRY_DEBIT_INVALID" in ev["reasons"]
    ev2 = trade_evaluation(TRADE, price_type="OBSERVED_QUOTE", entry_quotes=[q("UNUSABLE")], exit_quotes=[q()])
    assert ev2["status"] == "DATA_BAD" and ev2["reasons"] == ["ENTRY_QUOTE_UNUSABLE"]
    nan = trade_evaluation({**TRADE, "outcome": float("nan")}, price_type="OBSERVED_QUOTE")
    assert nan["status"] == "DATA_BAD" and "OUTCOME_NOT_FINITE" in nan["reasons"]


def test_legacy_debit_derived_like_compute_outcome():
    legacy = {**TRADE, "entry_debit": None}
    assert effective_entry_debit(legacy) == 2.0
    spread = {"strategy": "BULL_CALL_SPREAD", "option": {"ask": 3.0, "spread_leg": {"bid": 1.2}}}
    assert effective_entry_debit(spread) == 1.8


def test_settlement_exit_and_unarchived_entry():
    ev = trade_evaluation(TRADE, price_type="SETTLEMENT_INTRINSIC", entry_quotes=[None], exit_quotes=[])
    assert ev["status"] == "OK" and ev["exit_session"] == "SETTLEMENT" and "ENTRY_QUOTE_NOT_ARCHIVED" in ev["flags"]
    assert ev["tier"] == "OBSERVED" and not ev["reproducible"]


def test_record_trade_outcome_links_archived_quotes_and_dedupes(tmp_path):
    raw = {"symbol": "AAPL261120C00200000", "bid": 1.9, "ask": 2.0, "quote_time": RTH}
    eid = oq.capture(raw, underlying="AAPL", source="tradier", context="trade_entry", retrieved_at=RTH)
    oq.capture({**raw, "bid": 2.5, "ask": 2.6, "quote_time": POST}, underlying="AAPL", source="tradier",
               context="exit_mark", retrieved_at=POST)
    trade = {**TRADE, "option": {**TRADE["option"], "option_symbol": "AAPL261120C00200000"},
             "entry_quote_record_ids": [eid], "close_date": "2026-10-15"}
    meta = {"method": "option_quote", "price_type": "OBSERVED_QUOTE"}
    now = datetime(2026, 10, 15, 20, 45, tzinfo=UTC)
    rec = feedback.record_trade_outcome(trade, 0.25, meta, now=now, path=tmp_path / "to.jsonl")
    ev = rec["evaluation"]
    assert ev["status"] == "OK" and ev["reproducible"] and ev["entry_session"] == "RTH"
    assert ev["exit_session"] == "POST" and "EXIT_OUTSIDE_RTH" in ev["flags"] and rec["reliable"] is True
    assert rec["exit_quote_record_ids"][0] and rec["outcome_net"] < rec["outcome"]        # Kosten abgezogen
    assert trade["outcome_evaluation"] == ev and feedback.learnable_on_close(trade, meta)
    assert feedback.record_trade_outcome(trade, 0.25, meta, now=now, path=tmp_path / "to.jsonl") is None
    assert len(read_jsonl(tmp_path / "to.jsonl")) == 1                                    # genau einmal


def test_not_learnable_when_evaluation_not_ok():
    meta = {"method": "option_quote", "price_type": "OBSERVED_QUOTE"}
    assert not feedback.learnable_on_close({"outcome_evaluation": {"status": "UNUSABLE"}}, meta)
    assert not feedback.learnable_on_close({}, meta)                                     # fehlt -> konservativ
    assert not feedback.learnable_on_close({"outcome_evaluation": {"status": "OK"}}, {"method": "delta_approx"})


def test_main_records_evaluation_before_learning_and_gates_it():
    src = inspect.getsource(feedback.main)
    assert src.index("record_trade_outcome(") < src.index("learnable_on_close(") < src.index("learn_on_close(history")
    ast.parse(src)

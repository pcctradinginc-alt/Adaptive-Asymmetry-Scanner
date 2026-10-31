"""Alpha-to-Options economics (Phase B): zero-network tests.

All numbers below are synthetic fixtures with hand-computed expectations. They
test arithmetic and integrity rules and are NEVER evidence of economic success.
"""
from datetime import date, datetime, timedelta, timezone

import pytest
import yaml

from modules import option_quotes as oq
from modules.autonomous_alpha import option_economics as oe
from modules.autonomous_alpha import option_link as al
from tests.test_alpha_option_link import (
    DECIDE,
    REPO,
    _decisions,
    _exit_quote,
    _registered,
    install_market,
)

RULE = yaml.safe_load((REPO / al.CONFIG).read_text())["rule"]


@pytest.fixture
def market(monkeypatch):
    return install_market(monkeypatch)


def _leg(sym, side, bid, ask):
    return {"occ_symbol": sym, "side": side, "bid": bid, "ask": ask}


# ── independent return calculations (hand-computed) ──────────────────────────

def test_long_call_net_return_by_hand():
    entry = [_leg("X", "long", 4.9, 5.1)]
    exit_ = [_leg("X", "long", 7.0, 7.2)]
    base = oe.position_economics(entry, exit_, RULE, "BASE")
    # capital = 5.10*100 + 0.65 = 510.65; proceeds = 7.00*100 - 0.65 = 699.35
    assert base["capital_usd"] == pytest.approx(510.65)
    assert base["pnl_usd"] == pytest.approx(188.70)
    assert base["return"] == pytest.approx(188.70 / 510.65, abs=1e-6)
    adv = oe.position_economics(entry, exit_, RULE, "ADVERSE")
    # entry 5.10 + 0.25*0.20 = 5.15; exit 7.00 - 0.25*0.20 = 6.95
    assert adv["capital_usd"] == pytest.approx(515.65) and adv["pnl_usd"] == pytest.approx(178.70)
    assert adv["max_loss_usd"] == pytest.approx(515.65 + 0.65)


def test_bull_call_spread_leg_directions_by_hand():
    entry = [_leg("L", "long", 7.9, 8.1), _leg("S", "short", 1.9, 2.1)]
    exit_ = [_leg("L", "long", 12.0, 12.4), _leg("S", "short", 4.0, 4.2)]
    base = oe.position_economics(entry, exit_, RULE, "BASE")
    # entry: buy long @8.10, sell short @1.90 -> 6.20*100 + 2*0.65 = 621.30
    # exit:  sell long @12.00, buy back short @4.20 -> 7.80*100 - 2*0.65 = 778.70
    assert base["capital_usd"] == pytest.approx(621.30) and base["pnl_usd"] == pytest.approx(157.40)
    adv = oe.position_economics(entry, exit_, RULE, "ADVERSE")
    # entry: 8.15 - 1.85 = 6.30 -> 631.30; exit: 11.90 - 4.25 = 7.65 -> 763.70
    assert adv["capital_usd"] == pytest.approx(631.30) and adv["pnl_usd"] == pytest.approx(132.40)
    wrong = [dict(leg, side="short" if leg["side"] == "long" else "long") for leg in entry]
    assert oe.position_economics(wrong, exit_, RULE, "BASE")["pnl_usd"] != base["pnl_usd"]


def test_adverse_scenario_floors_bid_at_zero_and_total_loss():
    entry = [_leg("X", "long", 0.05, 0.6)]
    exit_ = [_leg("X", "long", 0.0, 0.1)]
    adv = oe.position_economics(entry, exit_, RULE, "ADVERSE")
    assert adv["exit_value_usd"] == pytest.approx(-0.65)          # worthless, still pays the exit fee
    # entry 0.60 + 0.25*0.55 = 0.7375 -> 73.75 + 0.65 = 74.40; total loss incl. both fees = 75.05
    assert adv["capital_usd"] == pytest.approx(74.40)
    assert adv["pnl_usd"] == pytest.approx(-75.05) == pytest.approx(-adv["max_loss_usd"])


# ── synthetic rows for metrics and comparisons ───────────────────────────────

def _row(hid, d, ret, *, status="SETTLED", st="LONG_CALL", ticker="AAA", group="TREATMENT", cap=500.0,
         regime="vix_lt_20|spy_uptrend"):
    exit_day = (date.fromisoformat(d) + timedelta(days=28)).isoformat()
    eco = {sc: {"capital_usd": cap, "pnl_usd": ret * cap, "return": ret, "max_loss_usd": cap + 0.65}
           for sc in oe.SCENARIOS} if status == "SETTLED" else {}
    return {"decision_id": f"{hid}{d}{ticker}{st}", "signal_id": f"s{d}{ticker}", "hypothesis_id": hid,
            "hypothesis_version": "v1", "comparison_group": group, "structure": st, "ticker": ticker,
            "pit_data_date": d, "decision_day": d, "regime": regime,
            "decision": "ABSTAIN" if status == "ABSTAIN" else "TRADE", "status": status,
            "evidence_class": "QUOTE_VERIFIED_PAPER" if status == "SETTLED" else None, "reasons": [],
            "exit_day": exit_day if status == "SETTLED" else None, "economics": eco}


def test_metrics_by_hand():
    rows = [_row("H", "2026-11-06", 0.5, ticker="A"), _row("H", "2026-11-13", -0.2, ticker="B"),
            _row("H", "2026-11-20", -0.4, ticker="A"), _row("H", "2026-11-27", 0.1, status="PENDING")]
    m = oe.metrics(rows)
    assert m["n"] == 3 and m["winrate"] == pytest.approx(1 / 3, abs=1e-4)
    assert m["avg_win_return"] == pytest.approx(0.5) and m["avg_loss_return"] == pytest.approx(-0.3)
    assert m["net_ev_return"] == pytest.approx(-0.1 / 3, abs=1e-6)
    assert m["profit_factor"] == pytest.approx(250 / 300, abs=1e-4)
    assert m["capital_weighted_return"] == pytest.approx(-50 / 1500, abs=1e-6)
    assert m["max_drawdown_usd"] == pytest.approx(-300.0) and m["max_loss_usd"] == pytest.approx(-200.0)
    assert m["concentration"]["top_ticker"] == "A" and m["concentration"]["distinct_tickers"] == 2
    assert m["independent_periods"] == 1                         # three weekly entries overlap one 28-day window
    assert m["peak_concurrent_capital_usd"] == pytest.approx(1500.0)


def test_comparisons_same_population_same_budget_missing_never_positive():
    h = [_row("H", "2026-11-06", 0.4, ticker="A"), _row("H", "2026-11-06", 0.0, status="ABSTAIN", ticker="B"),
         _row("H", "2026-12-04", 0.2, ticker="C"), _row("H", "2026-12-04", 0.0, status="EXIT_MISSING", ticker="D"),
         _row("H", "2027-01-08", 0.3, status="PENDING", ticker="E")]
    c = [_row("C", "2026-11-06", 0.1, group="CONTROL"), _row("C", "2026-12-04", -0.1, group="CONTROL"),
         _row("C", "2027-01-08", 0.0, group="CONTROL")]
    nt = oe.compare(h, None)
    # 2026-11-06: (0.4 + 0)/2 = 0.2 ; 2026-12-04: 0.2 (missing excluded) ; 2027-01-08 pending -> not a date
    assert nt["n_dates"] == 2 and nt["mean_difference"] == pytest.approx(0.2)
    cons = oe.compare(h, None, view="CONSERVATIVE")
    assert cons["mean_difference"] == pytest.approx((0.2 + (0.2 - 1.0) / 2) / 2)   # missing = total loss
    base = oe.compare(h, c)
    assert base["n_dates"] == 2 and base["mean_difference"] == pytest.approx(((0.2 - 0.1) + (0.2 + 0.1)) / 2)
    assert base["status"] == "DESCRIPTIVE_NOT_A_TEST"
    assert oe.compare(h, [_row("C", "2025-01-03", 0.1)])["status"] == "NO_COMMON_POPULATION"


def _cfg():
    return al.config(REPO)


def test_research_decision_is_never_an_order():
    rows = [_row("H", (date(2026, 11, 6) + timedelta(days=7 * i)).isoformat(), 0.3 + 0.01 * (i % 3), ticker=f"T{i}")
            for i in range(40)]
    ctrl = [_row("C", r["pit_data_date"], -0.05, group="CONTROL", ticker="Z" + r["ticker"]) for r in rows]
    m = oe.metrics(rows)
    cov = {"TRADE_DECIDED": 40, "PENDING": 0}
    vs = {"no_trade_conservative": oe.compare(rows, None, view="CONSERVATIVE"), "base_strategy": oe.compare(rows, ctrl)}
    dec = oe.research_decision(m, cov, vs, _cfg())
    assert dec["label"] == "CANDIDATE_FOR_PROSPECTIVE_VALIDATION"
    assert dec["order"] is None and dec["decision_type"] == "RESEARCH_ONLY" and dec["production_influence"] == "NONE"
    neg = [dict(r, economics={sc: {**e, "return": -0.2, "pnl_usd": -100.0} for sc, e in r["economics"].items()})
           for r in rows]
    vs_neg = {"no_trade_conservative": oe.compare(neg, None, view="CONSERVATIVE"),
              "base_strategy": oe.compare(neg, ctrl)}
    assert oe.research_decision(oe.metrics(neg), cov, vs_neg, _cfg())["label"] == "NO_ECONOMIC_EDGE_AFTER_COSTS"
    few = oe.research_decision(oe.metrics(rows[:3]), {"TRADE_DECIDED": 3, "PENDING": 0}, vs, _cfg())
    assert few["label"] == "NEED_MORE_DATA" and "TOO_FEW_VERIFIED_OUTCOMES" in few["reasons"]
    low_cov = oe.research_decision(m, {"TRADE_DECIDED": 80, "PENDING": 0}, vs, _cfg())
    assert "QUOTE_COVERAGE_BELOW_MIN" in low_cov["reasons"]


# ── end-to-end on Phase A journals ───────────────────────────────────────────

def _settled(tmp_path, market):
    _registered(tmp_path)
    al.decide(tmp_path, now=DECIDE)
    at = datetime(2026, 12, 8, 16, 0, tzinfo=timezone.utc)
    for d in _decisions(tmp_path):
        if d["decision"] != "TRADE":
            continue
        for leg in d["legs"]:
            up = 1.5 if leg["side"] == "long" else 1.2
            _exit_quote(tmp_path, leg["occ_symbol"], at, round(leg["bid"] * up, 2), round(leg["ask"] * up + 0.05, 2))
    al.settle(tmp_path, now=datetime(2026, 12, 12, 15, 0, tzinfo=timezone.utc))
    return oe.evaluate(tmp_path, now=datetime(2026, 12, 12, 15, 0, tzinfo=timezone.utc))


def test_end_to_end_uses_frozen_decisions_and_reverified_quotes(tmp_path, market):
    rep = _settled(tmp_path, market)
    rows, _ = oe.trades(tmp_path, al.read_all(tmp_path), al.config(tmp_path))
    settled = [r for r in rows if r["status"] == "SETTLED"]
    assert settled and all(r["evidence_class"] == "QUOTE_VERIFIED_PAPER" for r in settled)
    d = next(x for x in _decisions(tmp_path) if x["decision_id"] == settled[0]["decision_id"])
    o = next(x for x in al.read_all(tmp_path)["outcomes"] if x["decision_id"] == d["decision_id"])
    assert settled[0]["economics"]["ADVERSE"] == oe.position_economics(d["legs"], o["exit_legs"], RULE, "ADVERSE")
    ref = rep["hypotheses"]["momentum_12_1"]["LONG_CALL"]
    assert ref["coverage"]["settled_verified"] == ref["metrics"]["ADVERSE"]["n"] > 0
    # synthetic fixture: far too little data -> no economic statement
    assert ref["research_decision"]["label"] == "NEED_MORE_DATA" and rep["status"] == "NEED_MORE_DATA"
    assert rep["evidence_classes"]["EXECUTED"]["n"] == 0
    assert rep["order_placed"] is False and rep["production_influence"] == "NONE"
    assert rep["hypotheses"][al.CONTROL_ID]["LONG_CALL"]["research_decision"]["label"] == "BASE_STRATEGY_BENCHMARK"
    again = oe.evaluate(tmp_path, now=datetime(2026, 12, 12, 15, 0, tzinfo=timezone.utc), persist=False)
    assert again["hypotheses"] == rep["hypotheses"]                                   # deterministic


def test_missing_archive_quote_makes_outcome_unverifiable(tmp_path, market):
    _settled(tmp_path, market)
    (tmp_path / oq.ROOT / "2026-12-08.jsonl").unlink()                               # exit quotes gone
    rep = oe.evaluate(tmp_path, now=datetime(2026, 12, 12, 15, 0, tzinfo=timezone.utc), persist=False)
    ref = rep["hypotheses"]["momentum_12_1"]["LONG_CALL"]
    assert ref["coverage"]["settled_verified"] == 0 and ref["coverage"]["unverifiable"] > 0
    assert "EXIT_QUOTE_NOT_IN_ARCHIVE" in ref["coverage"]["failure_reasons"]
    assert ref["metrics"]["ADVERSE"]["n"] == 0


def test_exit_missing_is_counted_never_in_metrics(tmp_path, market):
    _registered(tmp_path)
    al.decide(tmp_path, now=DECIDE)
    al.settle(tmp_path, now=datetime(2026, 12, 17, 15, 0, tzinfo=timezone.utc))      # deadline passed, no quotes
    rep = oe.evaluate(tmp_path, now=datetime(2026, 12, 17, 15, 0, tzinfo=timezone.utc), persist=False)
    ref = rep["hypotheses"]["momentum_12_1"]["LONG_CALL"]
    assert ref["coverage"]["exit_missing"] == ref["coverage"]["trade_decided"] > 0
    assert ref["metrics"]["ADVERSE"]["n"] == 0
    assert ref["comparisons"]["no_trade"].get("n_dates", 0) == 0                       # excluded, not positive
    assert ref["comparisons"]["no_trade_conservative"]["mean_difference"] < 0           # total loss when booked


def test_structure_proposal_only_among_candidates():
    def entry(label, lo):
        return {"research_decision": {"label": label}, "comparisons": {"base_strategy": {"ci": [lo, lo + 0.1]}}}
    none = oe.propose_structure({"LONG_CALL": entry("NEED_MORE_DATA", 0.5),
                                 "BULL_CALL_SPREAD": entry("NO_ECONOMIC_EDGE_AFTER_COSTS", 0.4)})
    assert none["structure"] is None and none["order"] is None
    one = oe.propose_structure({"LONG_CALL": entry("CANDIDATE_FOR_PROSPECTIVE_VALIDATION", 0.01),
                                "BULL_CALL_SPREAD": entry("ADVANTAGE_NOT_ESTABLISHED", 0.9)})
    assert one["structure"] == "LONG_CALL" and one["selection_uses_exploratory_data"]
    tie = oe.propose_structure({"LONG_CALL": entry("CANDIDATE_FOR_PROSPECTIVE_VALIDATION", 0.02),
                                "BULL_CALL_SPREAD": entry("CANDIDATE_FOR_PROSPECTIVE_VALIDATION", 0.02)})
    assert tie["structure"] == "BULL_CALL_SPREAD"


def test_cli_option_link_runs_phase_b(tmp_path, market):
    from modules.autonomous_alpha import __main__ as cli
    _settled(tmp_path, market)
    assert cli.main(["option-link", "--root", str(tmp_path)]) == 0
    rep = (tmp_path / oe.REPORT)
    assert rep.exists() and '"order_placed": false' in rep.read_text()

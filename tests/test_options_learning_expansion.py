"""Independent research expansion: no fabricated labels, no hindsight orders."""
from __future__ import annotations

import copy
import hashlib
import json
from datetime import date, datetime, timezone

import pytest

from modules import candidate_ledger as cl
from modules import llm_census_external as lx
from modules import llm_free_challenger as lf
from modules import option_label_forensics as forensic
from modules import champion_protocol as cp


def _quote(sym: str, day: str, bid: float, ask: float) -> dict:
    when = f"{day}T16:00:00+00:00"
    return {"record_id": hashlib.sha256((sym + when).encode()).hexdigest()[:18],
            "occ_symbol": sym, "bid": bid, "ask": ask, "mid": (bid + ask) / 2,
            "retrieved_at": when, "quote_time": when,
            "quality_status": "OK", "quality_flags": []}


def _row(day="2026-11-02", ticker="AAA", *, sampled=True, directional=False):
    sym = f"{ticker}270319C00100000"
    return {"date": day, "ticker": ticker, "signal_id": ticker + ":" + day,
            "option_sample": "research_census" if sampled else "directional",
            "direction": "BULLISH" if directional else None,
            "features": {"rel_volume": 2.0, "sentiment_score": .3, "scan_day_ret": .01},
            "real_option": {"symbol": sym, "bid": 2.0, "ask": 2.2, "expiry": "2027-03-19"},
            "real_strategy": {"legs": [{"symbol": sym, "side": "long", "bid": 2.0, "ask": 2.2}],
                              "net_debit_entry": 2.2, "entry_date": day,
                              "expiry": "2027-03-19"},
            "outcomes": {"real_strat_ret_5d": .10, "real_strat_mark_date_5d": "2026-11-09"}}


def test_forensic_never_invents_entries():
    old = _row("2026-10-01")
    old["outcomes"]["real_strat_mark_date_5d"] = "2026-10-07"
    before = copy.deepcopy(old)
    q = [_quote("UNRELATED270319C00100000", "2026-10-12", 1, 2)]
    rep = forensic.audit([old], q, date(2026, 11, 10), horizons=(5,))
    assert rep["missing_by_horizon_reason"]["5d"]["ENTRY_NOT_ARCHIVED"] == 1
    assert rep["entry_missing_details"][0]["reason"] == "before_archive"
    assert rep["entry_not_archived_unique_signals"] == 1
    assert rep["recovered_labels"] == 0
    assert old == before


def test_free_challenger_ignores_directional_champion_and_is_stable():
    rows = [_row(ticker=f"T{i:03d}") for i in range(30)]
    rows += [_row(ticker="CHAMP", sampled=False, directional=True)]
    a = lf.select(rows, "2026-11-02")
    b = lf.select(list(reversed(rows)), "2026-11-02")
    assert [x["ticker"] for x in a["selected"]] == [x["ticker"] for x in b["selected"]]
    assert a["n_ranked"] == 6 and a["n_audit"] == 4
    assert a["pool_n"] == 30
    assert len({c["ticker"] for c in a["selected"]}) == 10
    assert all(c["ticker"] != "CHAMP" for c in a["selected"])
    assert all(c["cohort"] in ("fixed_rank", "random_audit") for c in a["selected"])


def test_forward_paper_requires_later_entry_and_mature_archived_exit():
    sel = lf.select([_row()], "2026-11-02")
    sel["created_at"] = "2026-11-02T18:00:00+00:00"
    sym = sel["selected"][0]["occ_symbol"]
    entry = _quote(sym, "2026-11-03", 1.9, 2.0)
    exit_quote = _quote(sym, "2026-11-09", 2.5, 2.6)
    prior = _quote(sym, "2026-11-02", 3.0, 3.1)
    assert lf.settle(sel, [prior], date(2026, 11, 10))["verified_n"] == 0
    immature = lf.settle(sel, [prior, entry, exit_quote], date(2026, 11, 4))
    assert immature["rows"][0]["status"] == "NOT_DUE"
    mature = lf.settle(sel, [prior, entry, exit_quote], date(2026, 11, 10))
    assert mature["verified_n"] == 1
    assert mature["rows"][0]["entry_day"] == "2026-11-03"
    assert mature["rows"][0]["net_return"] == pytest.approx((2.5 - 2.0 - .013) / 2)
    assert mature["verdict"] == "OBSERVATIONAL_UNVALIDATED"


def test_census_freeze_checksums_same_day_and_no_retroactive_calls(tmp_path):
    now = datetime(2026, 11, 10, 19, tzinfo=timezone.utc)
    root = tmp_path / "frozen"
    design = {"policy": "uniform-without-replacement-v1", "day": "2026-11-10",
              "n": 50, "k": 1, "pi": .02}
    saved = lx.freeze([{"ticker": "AAA", "news": ["frozen"]}], design, root=root, now=now)
    assert saved["status"] == "FROZEN"
    assert lx.freeze([], design, root=root, now=now)["status"] == "ALREADY_FROZEN"
    p = root / "2026-11-10.json"
    assert lx.read_frozen(p, now=now)["design"]["pi"] == .02
    assert lx.process("2026-11-09", inputs=root, now=now)["status"] == "STALE_DAY"
    content = json.loads(p.read_text())
    content["picks"][0]["news"] = ["future"]
    p.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="checksum"):
        lx.read_frozen(p, now=now)


def test_census_cost_fail_closed_and_observations_not_trade_notes(tmp_path):
    now = datetime(2026, 11, 10, 19, tzinfo=timezone.utc)
    inputs, outputs = tmp_path / "in", tmp_path / "out"
    design = {"policy": "uniform-without-replacement-v1", "day": "2026-11-10",
              "n": 20, "k": 1, "pi": .05}
    lx.freeze([{"ticker": "AAA", "news": ["known"]}], design, root=inputs, now=now)
    poor = {"level": "OK", "soft_pressure": False, "monthly_api_budget_usd": 10.,
            "projected_month_usd": 9., "month_usd": 1.,
            "week_usd": .5, "weekly_budget_usd": 6.}
    assert lx.process("2026-11-10", inputs=inputs, outputs=outputs,
                      now=now, allow_llm=True, budget_fn=lambda: poor)["status"] == "BUDGET_GUARD"

    class FakeAnalyzer:
        skipped_for_time = []
        def run(self, picks, deadline):
            return [{**p, "deep_analysis": {"impact": 6, "surprise": 4, "direction": "BULLISH"}}
                    for p in picks]

    healthy = {**poor, "projected_month_usd": 2.}
    good = lx.process("2026-11-10", inputs=inputs, outputs=outputs,
                      now=now, allow_llm=True, budget_fn=lambda: healthy,
                      analyzer_factory=FakeAnalyzer)
    assert good["status"] == "RECORDED" and good["analyzed"] == 1
    raw = json.loads((outputs / "2026-11-10.json").read_text())
    assert set(raw["observations"][0]) == {"ticker", "fields"}
    assert "direction" not in raw["observations"][0]["fields"]
    assert "llm_census" in raw["observations"][0]["fields"]
    assert lx.process("2026-11-10", inputs=inputs, outputs=outputs,
                      now=now, allow_llm=True, budget_fn=lambda: healthy)["status"] == "ALREADY_RECORDED"


def test_unapproved_snapshot_expansion_stays_at_champion_baseline(monkeypatch):
    monkeypatch.setenv("RESEARCH_EXTRA_OPTION_SNAPSHOTS_APPROVED", "1")
    assert cl._max_option_snapshots() == 40
    monkeypatch.delenv("RESEARCH_EXTRA_OPTION_SNAPSHOTS_APPROVED")
    assert cl._max_option_snapshots() == 40


def test_champion_rule_hash_and_trading_config_unchanged():
    assert cp.decision_config_sha() == cp.load_protocol()["decision_config_sha256"]
    assert lf.POLICY not in str(cp.decision_config())

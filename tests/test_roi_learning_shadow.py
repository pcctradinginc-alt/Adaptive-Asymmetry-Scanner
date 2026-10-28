"""ROI counterfactual learning safety tests; no LLM, no network."""
from datetime import date, datetime, timezone
from pathlib import Path
import json

import pytest
import yaml

from modules import roi_learning_shadow as rl
from modules import option_quotes as oq


def _seed(tmp_path, **config):
    params = {
        "version": "roi-learning-v1", "enabled": True,
        "max_new_per_day": 5,
        "roi_research_thresholds": [0., .05, .07],
        "horizons_calendar_days": [5, 20, 45],
        "max_entry_lag_days": 5, "max_exit_lag_days": 8,
        "max_quote_symbols_per_capture": 30,
        "commission_usd_per_leg_per_side": .65,
        "production_influence": "NONE", "allow_llm": False,
    }
    params.update(config)
    f = tmp_path / rl.CONFIG
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(yaml.safe_dump(params), encoding="utf-8")


def _reject(ticker, roi=.04, strategy="LONG_CALL"):
    return {
        "ticker": ticker, "strategy": strategy, "roi_net": roi,
        "roi_hurdle": .07, "fail_gates": {"Long-Term": "mc_pnl"},
        "option": {"strike": 100, "expiry": "2027-03-19",
                   "bid": 3.5, "ask": 4.,
                   "option_symbol": f"{ticker}270319C00100000"},
    }


def _register(root, rejects=None):
    return rl.register(rejects or [_reject("XYZ")], "2026-11-02", root=root,
                       now=datetime(2026, 11, 2, 16, 0, tzinfo=timezone.utc))


def _quote(root, d, time_, bid, ask, sym="XYZ270319C00100000"):
    p = root / "outputs/options_archive/quotes" / f"{d}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    q = {"occ_symbol": sym, "record_id": f"id-{d}-{bid}-{ask}",
         "retrieved_at": f"{d}T{time_}+00:00",
         "price_type": "OBSERVED_QUOTE", "quality_status": "OK",
         "bid": bid, "ask": ask}
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(q) + "\n")


def test_daily_research_budget_independent_of_roi_and_prod(tmp_path):
    _seed(tmp_path)
    rows = [_reject(f"T{i:02}", roi=(-.20 + i*.05)) for i in range(20)]
    report = _register(tmp_path, rows)
    assert report["registered"] == 5
    assert report["production_impact"] is False and report["llm_calls"] == 0
    assert _register(tmp_path, rows)["registered"] == 0
    frozen = rl._journal_read(tmp_path / rl.COHORTS)
    assert len(frozen) == 5
    assert all(r["production_eligible"] is False for r in frozen)
    assert all(r["evidence"] == "REJECTED_ONLY_NOT_TRADABLE" for r in frozen)
    assert all(r["original_production_hurdle"] == .07 for r in frozen)


def test_roi_counterfactual_thresholds_are_research_labels_not_permission(tmp_path):
    _seed(tmp_path)
    _register(tmp_path, [_reject("XYZ", roi=.04)])
    row = rl._journal_read(tmp_path / rl.COHORTS)[0]
    assert row["roi_only_ge"] == {"0.00": True, "0.05": False, "0.07": False}
    assert row["production_eligible"] is False
    assert row["other_production_gates"] == {"Long-Term": "mc_pnl"}


def test_conflicting_cohort_identity_fails_closed(tmp_path):
    _seed(tmp_path)
    _register(tmp_path)
    p = next((tmp_path / rl.COHORTS).glob("*.jsonl"))
    event = json.loads(p.read_text().strip())
    event["record"]["original_roi_net"] = 17.
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event) + "\n")
    with pytest.raises(rl.ResearchIntegrityError, match="content hash mismatch"):
        _register(tmp_path)


def test_no_quote_no_outcome_never_synthetic_pnl(tmp_path):
    _seed(tmp_path)
    _register(tmp_path)
    report = rl.evaluate(root=tmp_path, today=date(2026,12,20))
    assert report["n_all_marks"] == 0
    assert report["missing_quotes_by_reason"] == {} or "entry_quote_missing" in report["missing_quotes_by_reason"]
    assert report["actual_production_roi_gate_changed"] is False


def test_entry_must_follow_decision_and_exit_must_follow_horizon(tmp_path, monkeypatch):
    _seed(tmp_path)
    _register(tmp_path)
    monkeypatch.setattr(oq, "usable_rth_quote", lambda rec: True)
    _quote(tmp_path, "2026-11-02", "15:59:00", 3.5, 4.0)  # 1 min BEFORE decision: must not fill.
    _quote(tmp_path, "2026-11-02", "16:05:00", 3.4, 4.0)  # genuine first executable entry.
    _quote(tmp_path, "2026-11-05", "16:05:00", 9., 9.5)  # too early for 5d mark.
    early = rl.evaluate(root=tmp_path, today=date(2026,11,6))
    assert early["n_all_marks"] == 0
    _quote(tmp_path, "2026-11-09", "16:05:00", 5., 5.4)
    marked = rl.evaluate(root=tmp_path, today=date(2026,11,10))
    assert marked["n_new_marks"] == 1
    mark = rl._journal_read(tmp_path / rl.MARKS)[0]
    assert mark["horizon_days"] == 5
    assert mark["entry_at"] == "2026-11-02T16:05:00+00:00"
    assert mark["net_return"] == round((5.-4.-.013)/4., 6)
    assert mark["strategy_alpha_attributable"] is False
    assert mark["price_evidence"] == "OBSERVED_RTH_EXECUTABLE_BID_ASK"
    assert rl.evaluate(root=tmp_path, today=date(2026,11,10))["n_new_marks"] == 0


def test_exit_20_45_horizons_need_their_own_later_real_quotes(tmp_path, monkeypatch):
    _seed(tmp_path)
    _register(tmp_path)
    monkeypatch.setattr(oq, "usable_rth_quote", lambda rec: True)
    _quote(tmp_path,"2026-11-02","16:05:00",3.5,4.)
    _quote(tmp_path,"2026-11-09","16:05:00",5.,5.5)
    _quote(tmp_path,"2026-11-24","16:05:00",2.,2.6)
    _quote(tmp_path,"2026-12-18","16:05:00",6.,6.4)
    report = rl.evaluate(root=tmp_path,today=date(2026,12,20))
    assert report["n_all_marks"] == 3
    assert report["marks_by_horizon"] == {5: 1, 20: 1, 45: 1}
    assert report["research_roi_thresholds"]["0.00"]["realized_mark_45"] == 1


def test_mark_missing_does_not_get_filled_by_model(tmp_path, monkeypatch):
    _seed(tmp_path)
    _register(tmp_path)
    monkeypatch.setattr(oq, "usable_rth_quote", lambda rec: True)
    _quote(tmp_path,"2026-11-02","16:05:00",3.5,4.)
    res = rl.evaluate(root=tmp_path,today=date(2026,11,10))
    assert res["n_all_marks"] == 0
    assert "mark_5_quote_missing" in res["missing_quotes_by_reason"]


def test_configuration_cannot_relax_production_or_unbound_cost(tmp_path):
    for config in [
        {"max_new_per_day": 10},
        {"production_influence": "ACTIVE"},
        {"allow_llm": True},
        {"roi_research_thresholds": [-.50,.05,.07]},
        {"max_quote_symbols_per_capture": 100},
    ]:
        _seed(tmp_path, **config)
        with pytest.raises(rl.ResearchIntegrityError):
            _register(tmp_path)


def test_capture_targets_are_bounded_and_due_only(tmp_path):
    _seed(tmp_path)
    _register(tmp_path)
    assert rl.due_capture_symbols(today=date(2026,11,2),root=tmp_path) == ["XYZ270319C00100000"]
    assert rl.due_capture_symbols(today=date(2026,11,16),root=tmp_path) == []
    assert rl.due_capture_symbols(today=date(2026,11,23),root=tmp_path) == ["XYZ270319C00100000"]


def test_existing_workflows_keep_regular_cadence_without_new_llm():
    root=Path(__file__).resolve().parents[1]
    pipeline=(root/"pipeline.py").read_text()
    feedback=(root/".github/workflows/feedback.yml").read_text()
    marks=(root/"scripts/rth_marks.py").read_text()
    assert "_roi_learning.register(_roi_rejects, today)" in pipeline
    assert "python -m modules.roi_learning_shadow evaluate" in feedback
    assert "rls.due_capture_symbols" in marks
    assert "ANTHROPIC_API_KEY" not in rl.__dict__
    assert "roi_learning_shadow" in pipeline


def test_gate_acquisition_report_keeps_budget_deferred_unpriced(tmp_path):
    _seed(tmp_path)
    daily = tmp_path / "outputs/daily_reports/2026-11-02.json"
    daily.parent.mkdir(parents=True, exist_ok=True)
    daily.write_text(json.dumps({
        "date": "2026-11-02",
        "stats": {
            "sector_ok": 358, "economics_first": {
                "mode": "active", "input": 358, "probed": 8,
                "haiku": 3, "audit_haiku": 1
            },
            "analyzed": 1, "quick_mc": 0, "roi_ok": 0, "trades": 0
        },
        "rejects": {
            "economics_first_quote_probe_budget": {"count": 350},
            "economics_first_earnings_gate": {"count": 5},
            "mismatch_below_min": {"count": 1}
        }
    }), encoding="utf-8")
    report = rl.evaluate(today=date(2026, 11, 3), root=tmp_path)
    summary = report["economics_first_acquisition"]
    assert summary["days"][0]["budget_not_probed"] == 350
    assert summary["days"][0]["earnings_rejected"] == 5
    assert summary["days"][0]["quote_probes"] == 8
    assert summary["budget_not_probed_is_unprofitable"] is False
    assert summary["options_outcomes_for_unprobed_candidates"] == "UNKNOWN"
    assert report["observed_option_economics"]["horizons"]["5"]["marked"] == 0
    markdown = (tmp_path / rl.ROOT / "gate_economics.md").read_text()
    assert "350" in markdown and "keine" in markdown
    assert report["actual_production_roi_gate_changed"] is False


def test_gate_economics_only_counts_observed_rth_marks(tmp_path, monkeypatch):
    _seed(tmp_path)
    _register(tmp_path, [_reject("XYZ", roi=.04)])
    monkeypatch.setattr(oq, "usable_rth_quote", lambda rec: True)
    _quote(tmp_path, "2026-11-02", "16:05:00", 3.5, 4.0)
    _quote(tmp_path, "2026-11-09", "16:05:00", 5.0, 5.4)
    report = rl.evaluate(today=date(2026, 11, 10), root=tmp_path)
    economics = report["observed_option_economics"]
    assert economics["horizons"]["5"]["marked"] == 1
    assert economics["horizons"]["5"]["fully_matured"] == 0
    assert economics["horizons"]["5"]["mean_net"] is None
    assert economics["horizons"]["20"]["marked"] == 0
    assert economics["rejection_gates"]["mc_pnl"]["5"]["marked"] == 1
    assert economics["rejection_gates"]["mc_pnl"]["45"]["marked"] == 0
    assert economics["gate_groups_overlap"] is True
    assert economics["statistical_promotion_allowed"] is False
    assert economics["fixed_horizon_marks_not_real_stop_take_profit"] is True


def test_matured_missing_quotes_count_in_coverage_denominator(tmp_path):
    _seed(tmp_path)
    _register(tmp_path)
    result = rl.evaluate(today=date(2026, 12, 15), root=tmp_path)
    cover = result["observed_option_economics"]["horizons"]["5"]
    assert cover["registered"] == 1
    assert cover["fully_matured"] == 1
    assert cover["fully_matured_marked"] == 0
    assert cover["matured_quote_coverage"] == 0.0
    assert cover["status"] == "NEED_MORE_MATURED_QUOTES"
    assert result["observed_option_economics"]["horizons"]["45"]["fully_matured"] == 0


def test_underlying_returns_remain_stock_only_not_option_pnl(tmp_path):
    _seed(tmp_path)
    ledger = tmp_path / "outputs/candidate_ledger/2026-10.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    observations = [
        {"date": "2026-10-20", "ticker": "ABC", "signal_id": "id1",
         "status": "rejected", "reject_reason": "economics_first_quote_probe_budget",
         "direction": None, "outcomes": {"ret_5d": 0.08}},
        {"date": "2026-10-20", "ticker": "XYZ", "signal_id": "id2",
         "status": "rejected", "reject_reason": "mismatch_below_min",
         "direction": "BULLISH", "outcomes": {"ret_5d": -0.03}},
        {"date": "2026-10-20", "ticker": "XYZ", "signal_id": "id2",
         "status": "rejected", "reject_reason": "mismatch_below_min",
         "direction": "BULLISH", "outcomes": {"ret_5d": -0.03}},
        {"date": "2026-10-20", "ticker": "ERR", "signal_id": "id3",
         "status": "rejected", "reject_reason": "mismatch_below_min",
         "direction": "BULLISH",
         "outcomes": {"ret_5d": 99, "price_basis": {"status": "CHANGED"}}},
    ]
    ledger.write_text("".join(json.dumps(row) + "\n" for row in observations))
    report = rl.evaluate(today=date(2026, 10, 29), root=tmp_path)
    stock = report["candidate_underlying_by_gate"]
    assert stock["signals_read"] == 3
    assert stock["stocks_are_not_option_returns"] is True
    budget = stock["by_reject_reason"]["economics_first_quote_probe_budget"]
    assert budget["registered"] == 1 and budget["direction_known"] == 0
    assert budget["underlying"]["5"]["observed"] == 1
    assert budget["underlying"]["5"]["mean_stock_return"] is None
    rejected = stock["by_reject_reason"]["mismatch_below_min"]
    assert rejected["registered"] == 2
    assert rejected["underlying"]["5"]["observed"] == 1
    assert report["observed_option_economics"]["horizons"]["5"]["marked"] == 0

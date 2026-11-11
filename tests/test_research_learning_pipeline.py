"""No synthetic option data, no production influence, no post-label model lookup."""
from __future__ import annotations
import copy
from datetime import date
from pathlib import Path

from modules import label_recovery_audit as audit
from modules import research_snapshot_expansion as snap
from modules import llm_census_offline as census
from modules import no_llm_challenger as no_llm


def test_snapshot_provider_gate_and_hard_limits(tmp_path):
    p = snap.policy()
    assert p["enabled"] is True
    assert snap.additional_slots(40, cfg=p, environ={"TRADIER_API_KEY": "fake"}) == (0, "provider_not_approved")
    allowed = {"TRADIER_API_KEY": "fake", "TRADIER_RESEARCH_80_APPROVED": "1"}
    assert snap.additional_slots(40, cfg=p, environ=allowed) == (40, "approved")
    assert snap.additional_slots(60, cfg=p, environ=allowed) == (20, "approved")
    assert snap.additional_slots(80, cfg=p, environ=allowed) == (0, "approved")
    f = tmp_path / "bad.yaml"
    f.write_text("enabled: true\nmax_extra_snapshots: 100\nmax_total_snapshots: 100\n"
                 "require_tradier_plan_approval: true\n")
    assert snap.policy(f)["enabled"] is False
    f.write_text("enabled: true\nmax_extra_snapshots: 40\nmax_total_snapshots: 80\n"
                 "require_tradier_plan_approval: false\n")
    assert snap.policy(f)["enabled"] is False


def test_label_recovery_is_evidence_only_and_does_not_impute():
    original = {
        "date": "2026-10-01", "ticker": "ABC", "signal_id": "s1",
        "real_strategy": {"strategy": "LONG_CALL", "net_debit_entry": 1.5,
                          "entry_date": "2026-10-01", "legs": [
                              {"side": "long", "symbol": "ABC261120C00100000",
                               "bid": 1.0, "ask": 1.5}]},
        "outcomes": {"real_strat_ret_5d": 0.30, "real_strat_mark_date_5d": "2026-10-07"}
    }
    untouched = copy.deepcopy(original)
    result = audit.audit([original], [], horizon=5)
    assert result["historical_missing_entry_count"] == 1
    assert result["missing"][0]["reason"] == "ENTRY_NOT_ARCHIVED"
    assert result["restored_labels"] == 0 and result["no_synthetic_prices"] is True
    assert original == untouched


def test_no_llm_needs_model_strictly_before_day_and_ignores_outcomes():
    model = {"policy": "ts-acq-v1", "label_cutoff": "2026-11-10",
             "names": ["news_count"], "median": [0.], "mu": [0.], "sd": [1.],
             "coef": [.1], "intercept": -.05, "model_id": "m1"}
    row = {"date": "2026-11-11", "ticker": "ABC", "signal_id": "s1",
           "features": {"news_count": 4},
           "real_option": {"symbol": "ABC261120C00100000"},
           "real_strategy": {"strategy": "LONG_CALL", "legs": [{"side": "long"}], "net_debit_entry": 2.0},
           "option_sample": "research_census",
           "outcomes": {"real_strat_ret_5d": 99.0}}
    a = no_llm.decide([row], date(2026,11,11), model)
    assert a["action"] == "PAPER_ONLY" and a["orders"] == 0 and len(a["watchlist"]) == 1
    b = copy.deepcopy(row)
    b["outcomes"] = {"real_strat_ret_5d": -99.0}
    assert no_llm.decide([b], date(2026,11,11), model)["prediction_id"] == a["prediction_id"]
    assert no_llm.decide([row], date(2026,11,10), model)["reason"] == "model_not_prior_or_stale"
    assert no_llm.decide([row], date(2026,11,11), None)["action"] == "ABSTAIN"


def test_offline_census_stays_frozen_and_requires_budget(tmp_path, monkeypatch):
    p = census.policy()
    assert p["k"] == 5 and p["k"] <= 10
    pool = [{"ticker": f"T{i:02d}", "news": ["frozen"]} for i in range(30)]
    prior = copy.deepcopy(pool)
    res = census.freeze(pool, "2026-11-11", cfg=p, root=tmp_path)
    assert res["status"] == "FROZEN" and res["n"] == 30 and res["k"] == 5
    assert census.freeze(list(reversed(pool)), "2026-11-11", cfg=p, root=tmp_path)["status"] == "ALREADY_FROZEN"
    assert pool == prior
    import json
    frozen = json.loads((tmp_path / "2026-11-11.json").read_text())
    assert len(frozen["candidates"]) == 5
    good = {"level": "OK", "soft_pressure": False, "projected_month_usd": 2,
            "month_usd": 1, "monthly_api_budget_usd": 10}
    assert census.budget_ok(good, p)
    assert not census.budget_ok({**good, "projected_month_usd": 9.5}, p)

    class Analyzer:
        def run(self, picks, deadline=None):
            return [{**x, "deep_analysis": {"impact": 5, "surprise": 4, "direction": "BULLISH"}}
                    for x in picks]
    class Scorer:
        def __init__(self):
            self.shadow = {}
        def _score(self, row):
            return {"features": {"mismatch": 0.2}}
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    report = census.analyze_frozen(frozen, cfg=p, budget=good, analyzer=Analyzer(), scorer_cls=Scorer)
    assert report["status"] == "DONE" and report["analyzed"] == 5
    assert len(report["rows"]) == 5 and report["orders_created"] == 0
    assert all("census_direction" in a["fields"] for a in report["rows"])
    assert all("direction" not in a["fields"] for a in report["rows"])
    assert census.analyze_frozen(frozen, cfg=p, budget={}, analyzer=Analyzer())["reason"] == "budget_guard"


def test_no_production_config_changed_and_workflow_order():
    base = Path(__file__).resolve().parents[1]
    s = (base / ".github/workflows/scanner.yml").read_text()
    assert s.index("No-LLM prospective options ranking") < s.index("Options-Lernmodell – Prognosen")
    pipeline = (base / "pipeline.py").read_text()
    assert pipeline.index("_lco.freeze(candidates, today)") < pipeline.index("select_before_haiku(")
    assert "max_quote_probes: 8" in (base / "config/economics_first.yaml").read_text()
    assert "champion-v3-2026-11-11" in (base / "config/champion_protocol.yaml").read_text()

def test_no_llm_evaluation_has_no_imputation_or_best_outcome_selection():
    from modules import no_llm_challenger_eval as ev
    predictions = [{"day": "2026-11-11", "watchlist": [
        {"ticker": "ABC", "signal_id": "s1", "contract": "ABC261120C00100000",
         "model_net_estimate": 0.25}]}]
    candidates = [{"date": "2026-11-11", "signal_id": "s1", "ticker": "ABC",
                   "exprs": [{"contract": ["ABC261120C00100000"],
                              "status": "NO_LABEL", "y": None}]}]
    a = ev.evaluate(predictions, candidates)
    assert a["verified"] == 0 and a["unverified"] == 1 and a["mean_of_day_means"] is None
    candidates[0]["exprs"] = [{"contract": ["ABC261120C00100000"], "status": "VERIFIED",
                              "y": -0.2, "label_day": date(2026,11,17)}]
    b = ev.evaluate(predictions, candidates)
    assert b["verified"] == 1 and b["rows"][0]["net_return"] == -0.2
    assert b["claim_proven_alpha"] is False

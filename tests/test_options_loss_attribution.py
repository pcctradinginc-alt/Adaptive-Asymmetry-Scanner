"""Fail-closed tests for the read-only option loss-attribution report."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.options_loss_attribution import AuditError, audit_cohort, audit_repository


def _status(horizon=5, *, mid=0.05, cost=-0.15):
    net = mid + cost
    return {
        "horizon": horizon, "as_of": "2026-11-23",
        "options_native": {
            "components": {
                "mid_ret": {"n": 4, "mean_diff": mid},
                "cost": {"n": 4, "mean_diff": cost},
                "exec_ret": {"n": 4, "mean_diff": net},
            },
            "execution": {
                "overall": {
                    "cost": {"n": 4, "mean_diff": cost, "independent_periods": 2},
                    "exec_ret": {"n": 4, "mean_diff": net},
                },
                "by_entry_spread": {
                    "0%-2%": {"exec_ret": {"n": 1, "mean_diff": 0.01}},
                    ">=10%": {"exec_ret": {"n": 3, "mean_diff": -0.14}},
                },
            },
        },
    }


def test_identity_and_population_are_checked():
    report = audit_cohort(_status())
    assert report["n_verified_contract_labels_with_attribution"] == 4
    assert report["mean_executable_return"] == pytest.approx(-0.10)
    assert abs(report["identity_residual"]) < 1e-8
    assert report["independent_holding_periods"] == 2


@pytest.mark.parametrize("mutator", [
    lambda x: x["options_native"]["components"]["exec_ret"].update(mean_diff=0.99),
    lambda x: x["options_native"]["components"]["cost"].update(n=3),
    lambda x: x["options_native"]["execution"]["by_entry_spread"].pop(">=10%"),
    lambda x: x["options_native"]["components"]["cost"].update(mean_diff=None),
])
def test_inconsistent_evidence_fails_closed(mutator):
    row = _status()
    mutator(row)
    with pytest.raises(AuditError):
        audit_cohort(row)


def test_repository_integrates_two_horizons_without_model_or_trade_changes(tmp_path):
    for horizon in (5, 20):
        p = tmp_path / "outputs/research/option_learner" / f"status_{horizon}d.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(_status(horizon=horizon)))
    paper = tmp_path / "outputs/research/paper_performance_analysis.json"
    paper.parent.mkdir(parents=True, exist_ok=True)
    paper.write_text(json.dumps({"n_closed": 118, "n_reliable": 8,
                                 "by_strategy": {"BULL_CALL_SPREAD": {"n": 5, "win_rate": 0, "mean": -0.467}}}))
    report = audit_repository(tmp_path)
    assert report["research_only"] is True
    assert report["production_influence"] == "NONE"
    assert report["champion"]["n_unreliable_or_unassessed"] == 110
    assert report["cohorts"]["20d"]["horizon_sessions"] == 20


def test_missing_or_corrupt_data_never_turns_into_zero(tmp_path):
    with pytest.raises(AuditError):
        audit_repository(tmp_path)
    p = tmp_path / "outputs/research/option_learner/status_5d.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{corrupt")
    with pytest.raises(AuditError):
        audit_repository(tmp_path)


def test_committed_research_reports_remain_reconciled():
    # Real data regression. Never repair existing outcomes or relax tolerances.
    root = Path(__file__).resolve().parents[1]
    result = audit_repository(root)
    assert result["cohorts"]["5d"]["n_verified_contract_labels_with_attribution"] > 0
    assert result["cohorts"]["20d"]["n_verified_contract_labels_with_attribution"] > 0

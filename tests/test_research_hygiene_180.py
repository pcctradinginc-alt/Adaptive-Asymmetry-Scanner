"""PR180 regression: purely observational research census and unused-source audit.

Synthetic fixtures. Never buys options, imports an exchange client, trains a
model, invokes a research hypothesis test or changes a production protocol.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from modules import research_candidate_census as census
from modules import research_source_utilization as sources

ROOT = Path(__file__).resolve().parents[1]


def row(day, ticker, **extra):
    return {"date": day, "ticker": ticker, **extra}


def test_latest_completed_uses_only_strictly_prior_ledger_dates(tmp_path):
    root = tmp_path / "ledger"
    root.mkdir()
    records = [row("2026-11-05", "AAA"), row("2026-11-06", "BBB"),
               row("2026-11-07", "CCC"), row("2026-11-08", "DDD")]
    (root / "2026-11.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    assert census.latest_completed_day(root, "2026-11-08") == "2026-11-07"
    assert census.latest_completed_day(root, "2026-11-06") == "2026-11-05"
    assert census.latest_completed_day(root, "2026-11-09") == "2026-11-08"
    result = census.run_latest(root=root, out=tmp_path / "reports", as_of="2026-11-08")
    report = json.loads(result.read_text())
    assert report["date"] == "2026-11-07"
    assert report["signals"] == 1
    assert report["option_outcome_provenance"] == "LEDGER_REPORTED_NOT_QUOTE_VERIFIED"
    assert report["actual_option_outcome_rows"] == 0
    assert report["missing_option_marks_are_not_zero_returns"]


def test_census_does_not_fabricate_quote_verification(tmp_path):
    root = tmp_path / "ledger"
    root.mkdir()
    (root / "2026-11.jsonl").write_text(
        json.dumps(row("2026-11-06", "AAA", reject_reason="economics_first",
                       outcomes={"ret_5d": 0.04, "real_strat_ret_5d": 0.2})) + "\n"
        + json.dumps(row("2026-11-06", "BBB", outcomes={})) + "\n")
    target = census.run_latest(root=root, out=tmp_path / "out", as_of="2026-11-07")
    info = json.loads(target.read_text())
    assert info["actual_option_outcome_rows"] == 1
    assert "NOT_QUOTE_VERIFIED" in info["option_outcome_provenance"]
    assert info["signals"] == 2 and info["underlying_outcome_rows"] == 1
    assert info["reasons"]["economics_first"] == 1


def test_missing_ledger_is_hard_fail_not_green_empty_report(tmp_path):
    with pytest.raises(FileNotFoundError, match="No completed"):
        census.latest_completed_day(tmp_path, "2026-11-08")
    root = tmp_path / "ledger"
    root.mkdir()
    (root / "2026-11.jsonl").write_text("{broken-json\n")
    with pytest.raises(json.JSONDecodeError):
        census.latest_completed_day(root, "2026-11-08")


def test_source_utilization_has_no_implicit_research_reader_or_alpha_success():
    health = {"fred_liquidity": {"status": "PASS", "rows_ingested": 22},
              "noaa_storm_events": {"status": "PASS"}}
    storage = {"measured_at": "2026-11-08T12:00:00+00:00",
               "per_source": {"fred_liquidity": {"retrieval_days": 3, "rows": 22,
                                                "raw_bytes_per_day": 500, "normalized_bytes_per_day": 60}}}
    rep = sources.build(health, storage, as_of="2026-11-08")
    assert rep["n_sources"] == 8
    fred = next(x for x in rep["sources"] if x["source_id"] == "fred_liquidity")
    assert fred["source_health"] == "PASS"
    assert fred["code_linkage_status"] == "NO_ALPHA_READER_REVIEWED"
    assert fred["alpha_test_status"] == "NOT_EVALUATED"
    assert fred["feature_status"] == "NOT_ENABLED"
    assert rep["measured_daily_bytes_for_scope"] == {"raw": 500, "normalized": 60}
    assert rep["proposed_next_protocol"]["status"] == "DRAFT_ONLY_NOT_REGISTERED"
    assert not rep["no_alpha_test"] is False
    assert rep["no_new_features"] and rep["no_protocol_or_production_changes"]
    absent = next(x for x in rep["sources"] if x["source_id"] == "clinicaltrials_sponsors")
    assert absent["source_health"] == "UNOBSERVED"
    assert absent["measured_raw_bytes_per_day"] is None
    assert rep["unmeasured_cost_is_not_zero"]


def test_source_audit_missing_or_invalid_inputs_fail_closed(tmp_path):
    health = tmp_path / "health.json"
    storage = tmp_path / "storage.json"
    with pytest.raises(FileNotFoundError):
        sources.run(health_path=health, storage_path=storage, out=tmp_path / "r.json")
    health.write_text("[]")
    storage.write_text("{}")
    with pytest.raises(ValueError, match="dict"):
        sources.run(health_path=health, storage_path=storage, out=tmp_path / "r.json")
    health.write_text("{}")
    storage.write_text("{not-json")
    with pytest.raises(json.JSONDecodeError):
        sources.run(health_path=health, storage_path=storage, out=tmp_path / "r.json")


def test_workflow_uses_weekly_only_research_and_commits_outputs():
    workflow = (ROOT / ".github/workflows/ml_research.yml").read_text(encoding="utf-8")
    census_step = workflow.split("id: research_census", 1)[1].split("      - name:", 1)[0]
    source_step = workflow.split("id: source_utilization", 1)[1].split("      - name:", 1)[0]
    assert "env.MODE == 'weekly'" in census_step
    assert "env.MODE == 'weekly'" in source_step
    assert "research_candidate_census --latest-completed" in census_step
    assert "research_source_utilization" in source_step
    assert "research_census=\${{ steps.research_census.outcome }}" in workflow
    assert "source_utilization=\${{ steps.source_utilization.outcome }}" in workflow
    assert 'outputs/research/' in workflow
    assert "option_learner evaluate" in workflow  # existing learner, not replaced


def test_frozen_research_protocols_unmodified_by_pr_180():
    # No new fields in the production rules and no shadow research promotion.
    import hashlib
    pinned = (ROOT / "config/factory_protocol.yaml").read_bytes()
    assert hashlib.sha256(pinned).hexdigest() == (
        "d1d9d16412d7656687755130f612ba4019bd9eaab6ef57f79b53f63091e3ede2")

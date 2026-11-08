"""Alpha E2E v3: realistic fail-closed operations and existing-memory feedback.

All receipts and quotes in tmp_path are synthetic. Production data is never
altered and no remote service, API, SMTP or order interface is called.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from modules import research_memory as rm
from modules.autonomous_alpha import e2e_audit as audit
from modules.autonomous_alpha import paper_account as pa

NOW = datetime(2026, 11, 10, 18, 0, tzinfo=timezone.utc)
ROOT = Path(__file__).resolve().parents[1]


def put(root, rel, value):
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


@pytest.fixture
def configured(tmp_path):
    for p in ("config/alpha_option_link.yaml", "config/paper_account.yaml",
              "config/alpha_option_validation.yaml"):
        dst = tmp_path / p
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((ROOT / p).read_bytes())
    return tmp_path


def capture(root, **kwargs):
    put(root, audit.CAPTURE / "2026-11-09.json", {
        "mode": "capture", "window": "IN_WINDOW", "observed_at": "2026-11-09T18:00:00+00:00",
        "capture_status": "COMPLETE", "paper_account": {"status": "ACTIVE"},
        "alpha_link": {"status": "OK"}, **kwargs})


def test_missing_real_market_observations_are_not_errors_or_alpha(configured):
    before = set(configured.rglob("*"))
    result = audit.build(configured, now=NOW)
    assert result["status"] == "NEED_MORE_DATA"
    assert result["n_technical_defects"] == 0
    assert result["paper_account"]["activation_verified_from_events"] is False
    assert result["source_chain"]["verified_settled_economics"] == 0
    assert result["existing_learning"]["forward_confirmed_and_memorized"] == 0
    assert result["existing_learning"]["alpha_improvement_proven"] is False
    assert set(configured.rglob("*")) == before
    assert {"CAPTURE_NOT_OBSERVED", "PAPER_ACCOUNT_NOT_ACTIVATED",
            "NO_VERIFIED_ALPHA_OPTION_EXITS"}.issubset({x["id"] for x in result["issues"]})
    assert result["real_orders"] is False and result["broker_connection"] == "NONE"


def test_capture_step_errors_are_technical_defects_but_missing_quotes_are_not(configured):
    capture(configured, capture_status="PARTIAL_MISSING_QUOTES",
            paper_account={"status": "ERROR", "error": "missing journal"},
            alpha_link={"status": "ERROR", "error": "invalid quote schema"})
    result = audit.build(configured, now=NOW)
    issues = {p["id"]: p["severity"] for p in result["issues"]}
    assert issues["ALPHA_DECISION_CAPTURE_FAILED"] == "TECHNICAL_DEFECT"
    assert issues["PAPER_BOOKING_CAPTURE_FAILED"] == "TECHNICAL_DEFECT"
    assert issues["MISSING_CAPTURE_QUOTES"] == "NEED_MORE_DATA"
    assert result["n_technical_defects"] == 2


def test_weekly_sync_reuses_one_memory_entry_for_falsified_forward_plan(configured):
    capture(configured)
    put(configured, audit.VALIDATION, {"plans": [{
        "plan_id": "AOV-SYN-1", "hypothesis_id": "H-SYN-1", "structure": "LONG_CALL",
        "status": "FALSIFIED", "classification": "NOT_CONFIRMED",
        "binding": {"look": 1, "report_id": "FROZEN-SYNTHETIC"}
    }]})
    first = audit.run(configured, now=NOW, sync_memory=True)
    assert first["n_technical_defects"] == 0
    assert first["existing_learning"]["verdicts_reconciled_to_memory"] == 1
    assert first["existing_learning"]["forward_confirmed_and_memorized"] == 0
    saved = rm.load(configured / rm.MEMORY)
    assert len([e for e in saved if e["kind"] == "alpha_option_validation"]) == 1
    assert rm.latest(saved)["alpha_option_validation:AOV-SYN-1"]["status"] == "REJECTED"
    receipts = sorted((configured / audit.RECEIPTS).glob("*.json"))
    assert len(receipts) == 1
    second = audit.run(configured, now=NOW, sync_memory=True)
    assert second == first
    assert len([e for e in rm.load(configured / rm.MEMORY)
                if e["kind"] == "alpha_option_validation"]) == 1
    assert sorted((configured / audit.RECEIPTS).glob("*.json")) == receipts


def test_fake_confirmed_plan_without_verified_forward_quotes_fails_closed(configured):
    capture(configured)
    put(configured, audit.VALIDATION, {"plans": [{
        "plan_id": "AOV-FAKE", "hypothesis_id": "H", "structure": "LONG_CALL",
        "status": "CONFIRMED", "classification": "QUALIFIED_FOR_HUMAN_REVIEW",
        "start_date": "2026-11-09", "binding": {"report_id": "FROZEN-SYN"}
    }]})
    checked = audit.run(configured, now=NOW, sync_memory=True)
    assert checked["n_technical_defects"] >= 1
    # Corrupt source validation does NOT get synchronized into Research Memory.
    assert checked["existing_learning"]["verdicts_reconciled_to_memory"] == 0
    assert "alpha_option_validation:AOV-FAKE" not in rm.latest(rm.load(configured / rm.MEMORY))
    assert checked["existing_learning"]["forward_confirmed_and_memorized"] == 0
    assert checked["existing_learning"]["alpha_improvement_proven"] is False
    issue = [i for i in checked["issues"] if i["id"] == "FORWARD_CONFIRMATION_PROVENANCE_MISSING"]
    assert len(issue) == 1 and issue[0]["severity"] == "TECHNICAL_DEFECT"
    assert not (configured / "outputs/options_archive/alpha_link").exists()


def test_tampered_account_event_prevents_audit_and_never_turns_into_pnl(configured):
    pa.sync(configured, now=NOW, capture=True, persist_report=False)
    good = audit.build(configured, now=NOW)
    assert good["paper_account"]["activation_verified_from_events"]
    assert good["source_chain"]["verified_settled_economics"] == 0
    (configured / pa.JOURNAL / "PA-INIT.json").write_text('{"sha256":"broken","payload":{}}')
    with pytest.raises(ValueError, match="integrity"):
        audit.build(configured, now=NOW)
    # The scheduled CLI writes a durable failed report and returns nonzero,
    # rather than dropping the owner alert or accepting a forged learning label.
    failed = audit.run(configured, now=NOW)
    assert failed["status"] == "TECHNICAL_DEFECT"
    assert failed["n_technical_defects"] == 1
    assert failed["issues"][0]["id"] == "E2E_PRECHECK_FAILED"
    assert json.loads((configured / audit.REPORT).read_text())["status"] == "TECHNICAL_DEFECT"


def test_capture_outside_window_not_misreported_as_realtime_failure(configured):
    assert audit.build(configured, now=NOW)["capture"]["status"] == "NOT_OBSERVED"
    capture(configured, capture_status="COMPLETE")
    out = audit.build(configured, now=NOW)
    assert out["capture"]["latest_session"] == "2026-11-09"
    assert out["n_technical_defects"] == 0


def test_existing_workflow_audits_after_memory_sync_and_before_owner_digest():
    wf = yaml.safe_load((ROOT / ".github/workflows/ml_research.yml").read_text())
    stages = wf["jobs"]["ml"]["steps"]
    ids = [s.get("id") for s in stages]
    assert ids.index("research_improvement") == ids.index("option_learner") + 1
    assert ids.index("improvement_notify") == ids.index("research_improvement") + 1
    assert ids.index("alpha_data_quality") < ids.index("alpha_e2e") < ids.index("paper_report")
    assert "e2e-audit" in stages[ids.index("alpha_e2e")]["run"]
    assert "alpha_e2e=" in (ROOT / ".github/workflows/ml_research.yml").read_text()
    assert "paper_account.sync" not in (ROOT / ".github/workflows/ml_research.yml").read_text()


def test_recovered_capture_error_survives_latest_daily_overwrite(configured):
    from scripts import rth_marks
    from datetime import timedelta
    directory = configured / audit.CAPTURE
    t0 = NOW
    bad = {"mode": "capture", "window": "IN_WINDOW",
           "capture_status": "ALPHA_LINK_ERROR",
           "alpha_link": {"status": "ERROR", "error": "synthetic quote schema failure"},
           "paper_account": {"status": "ERROR", "error": "synthetic account failure"}}
    good = {"mode": "capture", "window": "IN_WINDOW",
            "capture_status": "COMPLETE",
            "alpha_link": {"status": "OK"}, "paper_account": {"status": "ACTIVE"}}
    rth_marks.persist_capture_status(bad, now=t0, root=directory)
    rth_marks.persist_capture_status(good, now=t0 + timedelta(minutes=1), root=directory)
    result = audit.build(configured, now=t0 + timedelta(minutes=2))
    issues = {x["id"]: x["severity"] for x in result["issues"]}
    assert result["capture"]["alpha_link_status"] == "OK"
    assert result["capture"]["recent_failed_attempts"] == 1
    assert issues["RECOVERED_CAPTURE_ERRORS"] == "DEGRADED"
    assert result["n_technical_defects"] == 0
    assert len(list((directory / "runs").glob("*/*.json"))) == 2


def test_corrupt_capture_attempt_receipt_fails_closed(configured):
    from scripts import rth_marks
    directory = configured / audit.CAPTURE
    rth_marks.persist_capture_status({"mode": "capture", "window": "IN_WINDOW",
                                      "capture_status": "COMPLETE"},
                                     now=NOW, root=directory)
    path = next((directory / "runs").glob("*/*.json"))
    path.write_text('{"sha256":"bad","payload":{}}')
    result = audit.build(configured, now=NOW)
    assert result["n_technical_defects"] == 1
    assert "CAPTURE_ATTEMPT_INTEGRITY" in {x["id"] for x in result["issues"]}


def test_cached_account_balances_must_equal_signed_git_journal(configured):
    pa.sync(configured, now=NOW, capture=True, persist_report=False)
    pa.sync(configured, now=NOW, capture=False, persist_report=True)
    clean = audit.build(configured, now=NOW)
    assert clean["n_technical_defects"] == 0
    file = configured / audit.ACCOUNT_REPORT
    fake = json.loads(file.read_text())
    fake["cash_usd"] += 25000
    file.write_text(json.dumps(fake))
    broken = audit.build(configured, now=NOW)
    assert broken["n_technical_defects"] == 1
    assert "ACCOUNT_SUMMARY_DIFFERS_FROM_SIGNED_LEDGER" in {i["id"] for i in broken["issues"]}
    checked = audit.run(configured, now=NOW)
    assert checked["n_technical_defects"] == 1
    assert not (configured / rm.MEMORY).exists()  # source corruption prevents memory ingestion

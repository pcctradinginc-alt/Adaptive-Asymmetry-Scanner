"""Phase-A experiments: offline, synthetic, no LLM or network requests."""
import json
import sqlite3
import subprocess

import pytest

from modules.autonomous_alpha import registry as reg
from modules.autonomous_alpha import orchestrator


def _save(root, path, payload):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload), encoding="utf-8")


def _fixtures(root):
    protocol = root / "config/research_protocol.yaml"
    protocol.parent.mkdir(parents=True, exist_ok=True)
    protocol.write_text("periods:\n  locked_status: CONTAMINATED\n", encoding="utf-8")
    _save(root, "outputs/research/hypothesis_db.json", {
        "generated": "2026-10-03T12:00:00+00:00",
        "hypotheses": {
            "H-1": {
                "id": "H-1", "signal": "rev_1m", "direction": 1,
                "canonical_status": "REJECTED",
                "walk_forward": {
                    "base": {"mean": 0.01, "sharpe_ann": 0.7, "max_dd": -0.15, "n_cohorts": 80},
                    "stress": {"mean": 0.001}
                }, "locked": {"base": {"mean": 0.10}},
                "reasons": ["stress too weak"]
            }
        },
    })
    _save(root, "outputs/research/factory_plan.json", {
        "generated": "2026-10-03T12:00:00+00:00",
        "ideas": [{"id": "FAC-1", "signal": "rank(rev_1m)", "direction": 1,
                   "family": "momentum", "plan_status": "SELECTED"}],
    })
    _save(root, "outputs/research/factory_results.json", {
        "generated": "2026-10-03T13:00:00+00:00",
        "results": {"FAC-1": {"spec": {"signal": "rank(rev_1m)",
                                       "direction": 1, "family": "momentum"},
                              "status": "REJECTED", "reasons": ["not robust"]}},
    })
    _save(root, "outputs/research/ml_forward.json", {
        "generated": "2026-10-24T12:00:00+00:00", "forward": {},
    })
    _save(root, "outputs/research/ml_research.json", {
        "generated": "2026-10-03T12:00:00+00:00",
        "models": {"model_a": {"role": "challenger", "registry_status": "valid",
                               "wf": {"base": {"mean": -0.01, "sharpe_ann": -0.2,
                                               "max_dd": -0.2}}}},
    })


def test_registry_reads_real_shapes_and_keeps_locked_nonpromotable(tmp_path):
    _fixtures(tmp_path)
    rep = reg.ingest(tmp_path, strict=True)
    assert rep["n_experiments"] == 4
    assert rep["n_prospectively_confirmed"] == 0
    assert rep["locked_holdout_policy"] == "CONTAMINATED"
    assert rep["appended"] == 4
    events = reg.read_journal(tmp_path / reg.JOURNAL)
    lab = next(x["record"] for x in events if x["record"]["experiment_key"] == "research_lab:H-1")
    assert lab["metrics"]["oos_sharpe_ann"] == 0.7
    assert lab["metrics"]["locked_mean_information_only"] == 0.1
    assert lab["promotion_eligible"] is False
    assert lab["evidence_class"] == "HISTORICAL_RESEARCH"
    rows = reg.leaderboard(events)
    assert rows and all(row["promotion_eligible"] is False for row in rows)
    conn = reg.build_sqlite(events)
    assert conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0] == 4
    conn.close()


def test_repeat_ingestion_is_idempotent(tmp_path):
    _fixtures(tmp_path)
    assert reg.ingest(tmp_path)["appended"] == 4
    again = reg.ingest(tmp_path)
    assert again["appended"] == 0
    assert len(reg.read_journal(tmp_path / reg.JOURNAL)) == 4


def test_updated_status_preserves_old_snapshot(tmp_path):
    _fixtures(tmp_path)
    reg.ingest(tmp_path)
    f = tmp_path / "outputs/research/hypothesis_db.json"
    j = json.loads(f.read_text())
    j["generated"] = "2026-10-07T12:00:00+00:00"
    j["hypotheses"]["H-1"]["canonical_status"] = "INCONCLUSIVE"
    f.write_text(json.dumps(j))
    result = reg.ingest(tmp_path)
    assert result["appended"] == 1
    events = reg.read_journal(tmp_path / reg.JOURNAL)
    latest = reg.latest_records(events)
    assert latest["research_lab:H-1"]["status"] == "INCONCLUSIVE"
    assert sum(e["record"]["experiment_key"] == "research_lab:H-1" for e in events) == 2


def test_changed_spec_requires_new_id(tmp_path):
    _fixtures(tmp_path)
    reg.ingest(tmp_path)
    f = tmp_path / "outputs/research/hypothesis_db.json"
    j = json.loads(f.read_text())
    j["hypotheses"]["H-1"]["signal"] = "mom_3m"
    f.write_text(json.dumps(j))
    with pytest.raises(reg.EvidenceIntegrityError, match="NEW experiment ID"):
        reg.ingest(tmp_path)


def test_corrupt_journal_fails_closed_without_replacement(tmp_path):
    _fixtures(tmp_path)
    reg.ingest(tmp_path)
    journal = tmp_path / reg.JOURNAL
    orig = journal.read_text()
    journal.write_text(orig + '{"event_id":"fake","record":{"schema":"alpha-experiment-v1"}}\n')
    with pytest.raises(reg.EvidenceIntegrityError, match="content-address"):
        reg.ingest(tmp_path)
    assert "fake" in journal.read_text()


def test_missing_source_informative_but_strict_fails(tmp_path):
    _fixtures(tmp_path)
    (tmp_path / "outputs/research/factory_results.json").unlink()
    report = reg.ingest(tmp_path)
    assert any("factory_results" in x for x in report["source_health"]["missing_sources"])
    with pytest.raises(reg.EvidenceIntegrityError, match="missing required"):
        reg.ingest(tmp_path, strict=True)


def test_order_independent_dedup_and_sqlite(tmp_path):
    _fixtures(tmp_path)
    reg.ingest(tmp_path)
    journal = tmp_path / reg.JOURNAL
    rows = journal.read_text().splitlines()
    journal.write_text("\n".join(list(reversed(rows)) + [rows[0]]) + "\n")
    events = reg.read_journal(journal)
    assert len(events) == 4
    assert reg.ingest(tmp_path)["appended"] == 0


def test_orchestrator_reuses_panel_and_does_not_execute_dynamic_code(tmp_path):
    _fixtures(tmp_path)
    cache = tmp_path / "trusted_panel.pkl"
    cache.write_bytes(b"synthetic-contents-unused")
    calls = []
    def fake_runner(command, **kwargs):
        calls.append(command)
        assert kwargs["env"]["ML_PANEL_CACHE"] == str(cache)
        assert command[:2] == [orchestrator.sys.executable, "-m"]
        return subprocess.CompletedProcess(command, 0, "", "")
    result = orchestrator.run_cycle(tmp_path, panel_cache=cache, runner=fake_runner)
    assert result["success"] is True
    assert len(calls) == len(orchestrator.STAGES) - 1
    assert result["journal"]["n_experiments"] == 4
    assert (tmp_path / reg.OUTPUT / "last_run.json").exists()


def test_orchestrator_stops_on_failed_stage_and_reports_failure(tmp_path):
    _fixtures(tmp_path)
    cache = tmp_path / "trusted_panel.pkl"
    cache.write_bytes(b"x")
    calls = []
    def fail_runner(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 17, "", "test failure")
    result = orchestrator.run_cycle(tmp_path, panel_cache=cache, runner=fail_runner)
    assert not result["success"]
    assert len(calls) == 1
    assert result["stages"][0]["exit_code"] == 17


def test_weekly_forward_observation_is_not_promotion(tmp_path):
    _fixtures(tmp_path)
    rep1 = reg.ingest(tmp_path, strict=True)
    assert rep1["n_prospectively_confirmed"] == 0
    _save(tmp_path, "outputs/research/ml_forward.json", {
        "generated": "2026-11-07T12:00:00+00:00",
        "forward": {"model_a": {
            "base": {"mean": 0.03, "sharpe_ann": 1.5, "max_dd": -0.04,
                     "n_cohorts": 3, "n_months": 1},
            "ic": {"mean_ic": 0.1},
        }},
    })
    report = reg.ingest(tmp_path, strict=True)
    assert report["appended"] == 1
    assert report["n_experiments"] == 5
    assert report["n_prospectively_confirmed"] == 0
    rec = reg.latest_records(reg.read_journal(tmp_path / reg.JOURNAL))["ml_forward:model_a"]
    assert rec["evidence_class"] == "PROSPECTIVE_OBSERVATIONS_UNVALIDATED"
    assert rec["metrics"]["forward_n_cohorts"] == 3
    assert rec["promotion_eligible"] is False

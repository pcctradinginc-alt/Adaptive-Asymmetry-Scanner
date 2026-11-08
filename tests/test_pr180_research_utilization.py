"""PR #180: research-only candidate-census and unused-source measurement guards.

All examples synthetic. Quotes are NOT generated, claimed or re-priced;
the already independent evidence-discovery result remains authoritative.
"""
import json
from datetime import date
from pathlib import Path

from modules import research_candidate_census as census
from modules import research_source_utilization as sources


def evidence(as_of="2026-11-08", verified=2, unverified=3):
    return {"as_of": as_of, "genuine_option_labels": verified,
            "unverified_option_labels": unverified,
            "provenance_counts": {"VERIFIED": verified, "EXIT_NOT_ARCHIVED": unverified},
            "no_option_proxy": True, "production_influence": "NONE",
            "evidence_class": "EXPLORATORY_REPEATED_HOLDOUT"}


def candidate(day, ticker, *, outcome=None, reason=None, sample=None):
    return {"date": day, "ticker": ticker, "signal_id": f"{day}:{ticker}", "reject_reason": reason,
            "option_sample": sample, "outcomes": outcome or {}}


def test_weekly_census_counts_all_rejected_candidates_and_never_confuses_marks_with_quotes():
    rows = [
        candidate("2026-11-01", "OLD", outcome={"real_strat_ret_20d": 0.5}),
        candidate("2026-11-02", "AAA", reason="economics_first_quote_probe_budget",
                  sample="research_census", outcome={"ret_5d": 0.04}),
        candidate("2026-11-05", "BBB", reason="economics_first", sample=None,
                  outcome={"real_strat_ret_5d": -0.22}),
        candidate("2026-11-07", "CCC", reason="none", sample="directional"),
        candidate("2026-11-08", "AAA", reason="economics_first"),
        candidate("2026-11-09", "FUTURE", outcome={"real_strat_ret_5d": 0.98}),
    ]
    out = census.audit_window(rows, "2026-11-08", days=7)
    assert out["window_start"] == "2026-11-02"
    assert out["candidate_rows"] == 4
    assert out["unique_tickers"] == 3
    assert out["reject_reasons"]["economics_first"] == 2
    assert out["option_sampling_classes"]["research_census"] == 1
    assert out["underlying_mark_present_rows"] == 1
    assert out["option_mark_present_unverified_rows"] == 1
    assert out["sampling_audit_is_not_quote_request"]
    assert census.audit_window(list(reversed(rows)), "2026-11-08")["sampling_audit_tickers"] == out["sampling_audit_tickers"]


def test_weekly_uses_existing_global_quote_verification_without_misattributing_to_week(tmp_path):
    ledger = tmp_path / "ledger"
    provenance = tmp_path / "evidence"
    ledger.mkdir()
    provenance.mkdir()
    (ledger / "2026-11.jsonl").write_text(
        json.dumps(candidate("2026-11-06", "AAA", outcome={"real_strat_ret_5d": 0.7})) + "\n")
    (provenance / "2026-11-08.json").write_text(json.dumps(evidence(verified=17, unverified=2)))
    path, report = census.run_week("2026-11-08", root=ledger, out=tmp_path / "out", evidence_dir=provenance)
    assert path.is_file() and report["status"] == "AUDIT_COMPLETE"
    assert report["candidate_rows"] == 1
    assert report["option_mark_present_unverified_rows"] == 1
    assert report["quote_provenance"]["global_verified_option_labels"] == 17
    assert report["quote_provenance"]["scope"] == "ALL_ELIGIBLE_HISTORICAL_CANDIDATES_NOT_THIS_WEEK"
    assert report["production_influence"] == "NONE"


def test_weekly_missing_month_or_missing_or_future_provenance_fails_closed(tmp_path):
    ledger, evidence_dir = tmp_path / "ledger", tmp_path / "ev"
    ledger.mkdir()
    evidence_dir.mkdir()
    (ledger / "2026-10.jsonl").write_text(json.dumps(candidate("2026-10-31", "X")) + "\n")
    (_, report) = census.run_week("2026-11-02", root=ledger, out=tmp_path / "reports",
                                  evidence_dir=evidence_dir)
    assert report["status"] == "DATA_GAP"
    assert report["missing_ledger_partitions"] == ["2026-11"]
    assert report["quote_provenance"]["global_verified_option_labels"] is None
    assert census.quote_evidence_summary(evidence("2026-11-09"), "2026-11-08")["status"].startswith("MISSING")
    bad = evidence()
    bad["no_option_proxy"] = False
    assert census.quote_evidence_summary(bad, "2026-11-08")["status"].startswith("MISSING")
    assert census.main(["--as-of", "2026-11-02", "--ledger", str(ledger),
                        "--evidence-dir", str(evidence_dir), "--output", str(tmp_path / "out")]) == 2


def test_original_daily_census_remains_compatible():
    row = candidate("2026-10-29", "AAA", outcome={"ret_5d": 0.1})
    assert census.audit([row], "2026-10-29")["signals"] == 1
    assert census.audit([row], "2026-10-29")["underlying_outcome_rows"] == 1


def test_source_utilization_measured_and_no_new_consumer_claims(tmp_path):
    health = {sid: {"status": "PASS", "last_success": "2026-11-08T12:00:00+00:00"}
              for sid in sources.REVIEWED_WITHOUT_ALPHA_CONSUMER}
    per_source = {sid: {"rows": 100, "raw_bytes_per_day": 2048,
                         "one_off_backfill_bytes": 4096, "retrieval_days": 8}
                  for sid in health}
    telemetry = {"measured_at": "2026-11-08T13:40:24+00:00", "per_source": per_source}
    result = sources.inventory(health, telemetry, date(2026, 11, 8))
    assert result["status"] == "MEASURED" and result["source_count"] == 8
    assert all(s["action_state"] == "RECOMMENDATION_ONLY_NOT_EXECUTED" for s in result["sources"])
    assert all(s["consumer_classification"] == "NO_RESEARCH_READER_VERIFIED_AT_CODE_AUDIT"
               and not s["consumer_audit_is_live_dependency_scan"] and not s["alpha_tested"]
               for s in result["sources"])
    assert all(s["estimated_raw_bytes_per_day"] == 2048 for s in result["sources"])
    assert not result["research_protocol_modified"] and not result["data_sources_enabled_or_disabled"]
    assert {s["source_id"] for s in result["sources"]} == set(sources.REVIEWED_WITHOUT_ALPHA_CONSUMER)
    assert {s["research_priority"] for s in result["sources"]} == {"HIGH", "MEDIUM", "LOW"}
    h, t = tmp_path / "h.json", tmp_path / "t.json"
    h.write_text(json.dumps(health))
    t.write_text(json.dumps(telemetry))
    target, persisted = sources.run(date(2026, 11, 8), h, t, tmp_path / "out")
    assert json.loads(target.read_text()) == persisted


def test_missing_or_stale_source_metrics_never_become_zero():
    stale = sources.inventory({}, {"measured_at": "2026-10-01", "per_source": {
        "fred_liquidity": {"raw_bytes_per_day": 2000}}}, "2026-11-08")
    assert stale["status"] == "DATA_GAP"
    assert all(s["estimated_raw_bytes_per_day"] is None for s in stale["sources"])
    assert stale["sources"][0]["health_status"] == "UNKNOWN"
    future = sources.inventory({}, {"measured_at": "2026-11-10", "per_source": {}}, "2026-11-08")
    assert future["status"] == "DATA_GAP"


def test_weekly_workflow_is_observational_and_keeps_all_existing_research_gates():
    workflow = Path(".github/workflows/ml_research.yml").read_text()
    assert "python -m modules.research_candidate_census --as-of" in workflow
    assert "python -m modules.research_source_utilization --as-of" in workflow
    assert workflow.count("if: env.MODE == 'weekly'") >= 1
    assert "candidate_census=\${{ steps.candidate_census.outcome }}".replace("\\", "") in workflow
    assert "source_utilization=\${{ steps.source_utilization.outcome }}".replace("\\", "") in workflow
    assert "outputs/research/" in workflow
    assert "python -m modules.research_evidence_discovery" in workflow


def test_no_new_research_signals_or_online_quote_acquisition():
    src = Path("modules/research_source_utilization.py").read_text() + Path("modules/research_candidate_census.py").read_text()
    for forbidden in ("send_order(", "market_snapshot.", "requests.get(", "httpx.get(", "select_liquid_contract("):
        assert forbidden not in src
    assert not any(s in src for s in ("config/factory_protocol.yaml", "model_registry", "hypothesis_acceptance.yaml"))

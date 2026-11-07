"""P1 quote, model feature and independent-window observability tests."""
import json
from datetime import datetime, timezone
from pathlib import Path

from modules.autonomous_alpha import data_quality as dq


NOW = datetime(2026, 11, 7, 19, 0, tzinfo=timezone.utc)


def put(root, rel, value):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value), encoding="utf-8")


def test_missing_quotes_are_counted_not_backfilled_and_source_dates_retained(tmp_path):
    put(tmp_path, dq.LINK, {"as_of": "2026-11-07", "status": "NEED_MORE_DATA",
                            "n_signals": 12, "n_decisions": 8, "n_outcomes": 0})
    put(tmp_path, dq.ECON, {"status": "NEED_MORE_DATA"})
    put(tmp_path, dq.VALIDATION, {"status": "NEED_MORE_DATA", "n_plans": 0,
                                 "learning": {"structure_labels": {
                                     "LONG_CALL": {"NEED_MORE_DATA": 5},
                                     "BULL_CALL_SPREAD": {"NEED_MORE_DATA": 5}}}})
    put(tmp_path, dq.LEARNER / "status_20d.json", {
        "as_of": "2026-11-07", "label_dates": ["2026-11-01", "2026-11-02"],
        "provenance_counts": {"VERIFIED": 30, "ENTRY_NOT_ARCHIVED": 70, "EXIT_NOT_ARCHIVED": 1},
        "feature_coverage": {"news_novelty": 0.0, "sentiment_score": 1.0},
        "evidence_requirements": {"min_independent_periods": 5}})
    put(tmp_path, dq.EVIDENCE / "2026-11-07.json", {
        "as_of": "2026-11-07", "population_rows": 999,
        "provenance_counts": {"ENTRY_NOT_ARCHIVED": 70, "EXIT_NOT_ARCHIVED": 1, "VERIFIED": 30}})
    put(tmp_path, dq.ERRORS, {"data_gaps": [{
        "kind": "independent_periods", "horizon": 20, "finding": "no independent windows"}]})
    out = dq.write(tmp_path, now=NOW)
    assert out["existing_option_learner"]["20"]["verified_quote_labels"] == 30
    assert out["existing_option_learner"]["20"]["archive_sample_verified_fraction"] == round(30 / 101, 5)
    assert out["existing_option_learner"]["20"]["independent_periods"] is None
    assert out["existing_option_learner"]["20"]["feature_coverage_below_80pct"] == {"news_novelty": 0.0}
    assert out["evidence_population_rows"] == 999
    assert out["largest_observed_count_gaps"] == ["ENTRY_NOT_ARCHIVED"]
    assert out["blocked_hypotheses_by_structure"]["LONG_CALL"] == 5
    assert all(g["recoverable_historically"] is False for g in out["data_gaps"])
    assert out["no_reconstructed_quotes"] and out["no_fake_alpha"]
    history = (tmp_path / dq.HISTORY / "2026-11-07.json").read_bytes()
    out2 = dq.write(tmp_path, now=NOW)
    assert (tmp_path / dq.HISTORY / "2026-11-07.json").read_bytes() == history
    assert out2["active_gap_ids"] == out["active_gap_ids"]


def test_missing_reports_do_not_invent_independent_periods(tmp_path):
    report = dq.build(tmp_path, now=NOW)
    assert report["link_settled_outcomes"] is None
    assert report["evidence_population_rows"] is None
    assert report["existing_option_learner"]["5"]["archive_sample_verified_fraction"] is None
    assert report["existing_option_learner"]["20"]["independent_periods"] is None
    assert report["largest_observed_count_gaps"] == []


def test_real_repository_research_evidence_is_read_without_fabrication():
    # P6 real repository-data pass, not a synthetic backtest: no external API,
    # new paper trades, changed frozen experiments or historical quote filling.
    root = Path(__file__).resolve().parents[1]
    actual = dq.build(root, now=NOW)
    assert actual["link_signals"] >= 90
    assert actual["link_decisions"] >= 40
    # Repository evidence is expected to grow after genuine RTH capture; a
    # zero-outcome snapshot must not be a permanent CI constraint.
    assert isinstance(actual["link_settled_outcomes"], int)
    assert 0 <= actual["link_settled_outcomes"] <= actual["link_decisions"]
    assert actual["alpha_option_status"] in ("NEED_MORE_DATA", "EVALUATED",
                                              "QUALIFIED_FOR_HUMAN_REVIEW")
    assert actual["existing_option_learner"]["20"]["verified_quote_labels"] >= 100
    assert actual["existing_option_learner"]["20"]["entry_not_archived"] > 0
    assert actual["existing_option_learner"]["20"]["independent_periods"] is None
    assert "ENTRY_NOT_ARCHIVED" in actual["active_gap_ids"]
    if actual["alpha_option_forward_plans"] == 0:
        assert "NO_PREREGISTERED_ALPHA_OPTION_FORWARD_PLANS" in actual["active_gap_ids"]
    else:
        assert actual["alpha_option_forward_plans"] > 0
    assert actual["no_reconstructed_quotes"] is True

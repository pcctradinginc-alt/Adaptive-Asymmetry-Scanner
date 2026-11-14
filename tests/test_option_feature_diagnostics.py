"""Offline contract tests for the read-only options feature cohort diagnosis."""
from __future__ import annotations

from copy import deepcopy
from datetime import date
import json

from modules import option_learner as ol
from modules import option_feature_diagnostics as diag


def _row(day, ticker, *, sample, features, gate="proposed"):
    return {"date": day, "ticker": ticker, "signal_id": f"{day}-{ticker}",
            "option_sample": sample, "features": features,
            "reject_reason": gate if gate != "proposed" else None,
            "decision_frozen_at": f"{day}T15:00:00+00:00" if sample == "directional" else None}


def _candidate(row, *, verified=True, expression="single"):
    raw = {"date": row["date"], "ticker": row["ticker"], "signal_id": row["signal_id"],
           "sample": row["option_sample"], "gate": row.get("reject_reason") or "proposed"}
    raw["exprs"] = [{
        "status": "VERIFIED" if verified else "NO_LABEL",
        "expression": expression,
        "contract": [f"{row['ticker']}270115C00100000"],
        "label_day": date(2026, 11, 12),
        "x": ol.features(row), "y": -0.1,
    }]
    return raw


def _cohorts():
    old = [
        _row("2026-10-20", "OLD1", sample="directional",
             features={"surprise": 4, "impact": 6, "options_skew_ratio": 1.2,
                       "options_skew_25d": 0.3, "dealer_oi_ratio": 0.7}),
        _row("2026-10-24", "OLD2", sample="directional",
             features={"surprise": 5, "impact": 7, "options_skew_ratio": 1.1,
                       "options_skew_25d": 0.4, "dealer_oi_ratio": 0.8}),
    ]
    new = [
        _row("2026-11-03", "NEW1", sample="research_census", features={},
             gate="economics_first_quote_probe_budget"),
        _row("2026-11-04", "NEW2", sample="research_census",
             features={"surprise": "not-a-number", "options_skew_25d": None},
             gate="economics_first_quote_probe_budget"),
    ]
    # Numeric values at root level are NOT valid replacements for missing
    # features: the frozen option-learner spec reads row["features"] only.
    new[-1]["options_skew_ratio"] = 999.0
    rows = old + new
    c5 = [_candidate(r) for r in rows]
    c20 = [_candidate(r) for r in old]
    # There are labels for other maturities, but the selected 5d/20d
    # training population MUST NOT include unverified/no-label rows.
    c5.append(_candidate(_row("2026-11-05", "UNVER", sample="research_census",
                              features={"surprise": 50}), verified=False))
    return rows, {5: c5, 20: c20}


def test_composition_not_a_reader_bug_when_cohorts_change():
    rows, c = _cohorts()
    before_rows, before_c = deepcopy(rows), deepcopy(c)
    five = diag.horizon_diagnostic(rows, c[5], 5)
    twenty = diag.horizon_diagnostic(rows, c[20], 20)
    assert five["total"]["labels"] == 4 and twenty["total"]["labels"] == 2
    assert five["total"]["feature_coverage"]["surprise"]["fraction"] == 0.5
    assert twenty["total"]["feature_coverage"]["surprise"]["fraction"] == 1.0
    for name in ("options_skew_ratio", "options_skew_25d", "impact", "dealer_oi_ratio"):
        assert five["total"]["feature_coverage"][name]["fraction"] == 0.5
        assert twenty["total"]["feature_coverage"][name]["fraction"] == 1.0
    assert five["by_sample"]["directional"]["labels"] == 2
    assert five["by_sample"]["research_census"]["labels"] == 2
    assert five["by_era"]["before_research_census"]["labels"] == 2
    assert five["by_era"]["from_research_census_start"]["labels"] == 2
    assert five["by_gate"]["economics_first_quote_probe_budget"]["labels"] == 2
    assert five["reader_consistency"] == twenty["reader_consistency"] == "CONSISTENT"
    assert five["source_audit"]["surprise"]["missing_key"] == 1
    assert five["source_audit"]["surprise"]["invalid_non_numeric"] == 1
    assert five["source_audit"]["options_skew_25d"]["null_value"] == 1
    assert five["source_audit"]["options_skew_ratio"]["numeric_top_level_only"] == 1
    assert five["source_audit"]["options_skew_ratio"]["reader_mismatch"] == 0
    assert five["per_feature_pit_provenance"] == "NOT_RECORDED"
    assert rows == before_rows and c == before_c


def test_reader_mismatch_is_separated_from_missing_source_values():
    rows, cohorts = _cohorts()
    c = deepcopy(cohorts[5])
    c[0]["exprs"][0]["x"].pop("surprise")  # synthetic corruption of reader output
    diagnosis = diag.horizon_diagnostic(rows, c, 5)
    assert diagnosis["reader_consistency"] == "MISMATCH"
    assert diagnosis["source_audit"]["surprise"]["reader_mismatch"] == 1
    assert diagnosis["source_audit"]["surprise"]["missing_key"] == 1


def test_multiple_expressions_do_not_masquerade_as_independent_candidates():
    rows, cohorts = _cohorts()
    c = deepcopy(cohorts[5])
    extra = dict(c[0]["exprs"][0], expression="spread", contract=["ONE", "TWO"])
    c[0]["exprs"].append(extra)
    result = diag.horizon_diagnostic(rows, c, 5)
    assert result["total"]["labels"] == 5
    assert result["total"]["unique_candidates"] == 4
    assert result["by_expression"]["spread"]["labels"] == 1
    assert result["total"]["feature_coverage"]["surprise"]["fraction"] == 0.6


def test_legacy_duplicate_rows_use_first_saved_feature_snapshot():
    rows, cohorts = _cohorts()
    old = dict(rows[0], features={"surprise": 99})
    result = diag.horizon_diagnostic(rows + [old], cohorts[5], 5)
    assert result["n_unique_ledger_keys"] == 4
    assert result["reader_consistency"] == "CONSISTENT"


def test_generated_report_is_reproducible_read_only_and_uses_canonical_labels(monkeypatch, tmp_path):
    rows, cohorts = _cohorts()
    inputs = deepcopy((rows, cohorts))
    calls = []

    def fake_load(as_of, horizon, ledger, quotes):
        calls.append((as_of, horizon, ledger, quotes))
        cs = cohorts[horizon]
        return rows, cs, ol._verified(cs)

    monkeypatch.setattr(ol, "_load", fake_load)
    out = tmp_path / "out"
    out.mkdir()
    as_of = date(2026, 11, 14)
    for h in (5, 20):
        data = ol._verified(cohorts[h])
        (out / f"status_{h}d.json").write_text(json.dumps({
            "as_of": as_of.isoformat(),
            "manifest": ol.manifest(data, as_of),
            "verified_labels": len(data),
            "feature_coverage": ol.feature_coverage(data),
        }), encoding="utf-8")

    before_id = ol.experiment_id()
    report = diag.run(as_of, ledger=tmp_path / "ledger",
                      quotes=tmp_path / "quotes", out=out)
    assert report["root_cause_status"] == "READ_PATH_CONSISTENT__COHORT_AND_STAGE_COMPOSITION_TO_REVIEW"
    assert report["horizons"]["5"]["stored_status_reconciliation"]["status"] == "REPRODUCED"
    assert report["horizons"]["20"]["stored_status_reconciliation"]["status"] == "REPRODUCED"
    assert report["raw_20d_minus_5d_coverage"]["surprise"] == 0.5
    assert report["production_influence"] == "NONE" and report["real_orders"] is False
    assert report["model_or_protocol_changed"] is False
    assert ol.experiment_id() == before_id
    assert (out / "feature_diagnostics.md").exists()
    b = (out / "feature_diagnostics.json").read_bytes()
    again = diag.run(as_of, ledger=tmp_path / "ledger",
                     quotes=tmp_path / "quotes", out=out)
    assert again == report
    assert b == (out / "feature_diagnostics.json").read_bytes()
    assert len(calls) == 4 and {h for _, h, _, _ in calls} == {5, 20}
    assert (rows, cohorts) == inputs


def test_missing_status_is_reported_not_silently_claimed_reproduced(monkeypatch, tmp_path):
    rows, cohorts = _cohorts()
    monkeypatch.setattr(ol, "_load", lambda as_of, h, ledger, quotes:
                        (rows, cohorts[h], ol._verified(cohorts[h])))
    report = diag.diagnose(date(2026, 11, 14), ledger=tmp_path / "absent",
                           quotes=tmp_path / "missing", out=tmp_path)
    assert report["horizons"]["5"]["stored_status_reconciliation"]["status"] == "NO_STORED_STATUS"


def test_diagnostics_run_after_canonical_option_learner_not_before():
    from pathlib import Path
    wf = (Path(__file__).resolve().parents[1] / ".github/workflows/ml_research.yml").read_text()
    assert wf.index("id: option_learner") < wf.index("id: option_feature_diagnostics")
    assert wf.index("id: option_feature_diagnostics") < wf.index("id: research_improvement")
    assert "option_feature_diagnostics" in wf
    assert "model_or_protocol_changed" not in wf  # research step, not a model gate

"""Regression tests for the 2026-11-04 P0/P1 operational reliability fixes.

No network, no orders, no synthetic option marks or claimed source recovery.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

from modules.external.sources import maritime as mw
from modules.external.sources.base import SourceStatus
from modules import ops_workflows as ow
from modules import candidate_ledger as cl
from modules import market_snapshot as ms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import rth_marks  # noqa: E402


NOW = datetime(2026, 11, 3, 19, 0, tzinfo=timezone.utc)


def test_commodity_cli_dispatch_occurs_after_evaluate_definition():
    """Regression: an earlier top-level __main__ raised NameError: evaluate."""
    source = (Path(__file__).resolve().parents[1] /
              "modules/commodity_intelligence.py").read_text(encoding="utf-8")
    assert source.count('if __name__ == "__main__":') == 1
    assert source.index("def evaluate(") < source.index('if __name__ == "__main__":')


def test_portwatch_rediscovers_invalid_arcgis_endpoint_once_and_fails_closed(monkeypatch):
    c = mw.PortWatchPortsConnector(http_fetch=lambda *a, **kw: None)
    state = {"current": "https://dead.example/FeatureServer", "refreshes": 0}
    seen = []

    def discover(*, force_refresh=False):
        if force_refresh:
            state["current"] = "https://new.example/FeatureServer"
            state["refreshes"] += 1
        return {"ports_daily": {"service_url": state["current"], "layer_id": 0},
                "ports_reference": {"service_url": state["current"], "layer_id": 0}}

    def bad_metadata(url, layer_id, http_fetch):
        seen.append(url)
        raise mw.SchemaError("Layer-Metadaten ohne 'fields': " + url)

    monkeypatch.setattr(c, "_discover", discover)
    monkeypatch.setattr(mw, "fetch_layer_metadata", bad_metadata)
    result = c.fetch(NOW)
    assert result.status == SourceStatus.SCHEMA_CHANGED
    assert state["refreshes"] == 1
    assert seen == ["https://dead.example/FeatureServer",
                    "https://new.example/FeatureServer"]


def test_portwatch_unchanged_endpoint_does_not_retry(monkeypatch):
    c = mw.PortWatchChokepointsConnector(http_fetch=lambda *a, **kw: None)
    refreshes = []
    monkeypatch.setattr(c, "_discover",
                        lambda *, force_refresh=False:
                        (refreshes.append(force_refresh) or {
                            "chokepoints_daily": {"service_url": "https://same.example/FeatureServer",
                                                  "layer_id": 0},
                            "chokepoints_reference": {"service_url": "https://same.example/FeatureServer",
                                                      "layer_id": 0}}))
    monkeypatch.setattr(mw, "fetch_layer_metadata",
                        lambda *_: (_ for _ in ()).throw(mw.SchemaError(
                            "Layer-Metadaten ohne 'fields'")))
    assert c.fetch(NOW).status == SourceStatus.SCHEMA_CHANGED
    assert refreshes == [False, True]


def test_portwatch_other_schema_error_does_not_autocorrect(monkeypatch):
    c = mw.PortWatchPortsConnector(http_fetch=lambda *a, **kw: None)
    refreshes = []
    monkeypatch.setattr(c, "_discover",
                        lambda *, force_refresh=False:
                        (refreshes.append(force_refresh) or {
                            "ports_daily": {"service_url": "https://same.example/FeatureServer", "layer_id": 0},
                            "ports_reference": {"service_url": "https://same.example/FeatureServer", "layer_id": 0}}))
    monkeypatch.setattr(mw, "fetch_layer_metadata",
                        lambda *_: (_ for _ in ()).throw(mw.SchemaError(
                            "SCHEMA_CHANGED: missing confirmed cargo field")))
    assert c.fetch(NOW).status == SourceStatus.SCHEMA_CHANGED
    assert refreshes == [False]


def test_capture_status_persists_only_real_rth_attempts(tmp_path):
    directory = tmp_path / "capture"
    skipped = {"mode": "capture", "window": "AFTER_WINDOW",
               "capture_status": "SKIPPED_OUTSIDE_WINDOW"}
    assert rth_marks.persist_capture_status(skipped, now=NOW, root=directory) is None
    assert not directory.exists()
    observed = {"mode": "capture", "window": "IN_WINDOW", "requested": 1,
                "quotes": 0, "missing": ["TEST"], "capture_status": "PARTIAL_MISSING_QUOTES"}
    path = rth_marks.persist_capture_status(observed, now=NOW, root=directory)
    assert path and path.exists()
    import json
    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["capture_status"] == "PARTIAL_MISSING_QUOTES"
    assert row["missing"] == ["TEST"]
    assert row["observed_at"] == NOW.isoformat()
    assert rth_marks.persist_capture_status(skipped, now=NOW, root=directory) is None
    assert json.loads(path.read_text())["capture_status"] == "PARTIAL_MISSING_QUOTES"


def test_capture_refreshes_alpha_coverage_after_decisions(monkeypatch, tmp_path):
    from modules.autonomous_alpha import option_link as al
    from modules import roi_learning_shadow as rls
    calls = []
    monkeypatch.setattr(ms, "us_market_session", lambda t: "regular")
    monkeypatch.setattr(cl, "due_mark_symbols", lambda d: [])
    monkeypatch.setattr(cl, "pending_entry_targets", lambda d: {})
    monkeypatch.setattr(rls, "due_capture_targets", lambda **kw: {})
    monkeypatch.setattr(al, "decide", lambda **kw: calls.append("decide") or
                        {"status": "OK", "n_new_decisions": 2})
    monkeypatch.setattr(al, "settle", lambda **kw: calls.append("settle") or
                        {"status": "OK", "n_new_outcomes": 0})
    monkeypatch.setattr(al, "coverage", lambda **kw: calls.append("coverage") or
                        {"status": "NEED_MORE_DATA", "n_decisions": 2})
    monkeypatch.setattr(al, "due_capture_targets", lambda **kw: {})
    monkeypatch.setattr(rth_marks, "_satisfied", lambda *args, **kw: set())
    res = rth_marks.run(NOW, tmp_path / "no-history.json", capture_only=True)
    assert res["window"] == "IN_WINDOW"
    assert calls == ["decide", "settle", "coverage"]
    assert res["alpha_link_settlement"]["n_new_outcomes"] == 0
    assert res["alpha_link_coverage"]["n_decisions"] == 2
    assert res["capture_status"] == "NO_TARGETS"


def test_technical_workflow_outcome_not_confused_with_business_evidence():
    scan = {"name": "Scanner", "path": ow.SCANNER_FILE, "conclusion": "success", "id": 1}
    skipped = ow.summarize_run_outcome(scan, [
        {"name": "guard", "conclusion": "success"},
        {"name": ow.SCAN_JOB, "conclusion": "skipped"}])
    assert skipped["effective_status"] == "SKIPPED_SCAN"
    assert skipped["business_evidence"] == "NOT_EVALUATED"

    ml = {"name": "Research", "path": ".github/workflows/ml_research.yml",
          "conclusion": "success", "id": 2}
    degraded = ow.summarize_run_outcome(ml, [{
        "name": "ml", "conclusion": "success",
        "steps": [{"name": "commodity_evaluate", "conclusion": "failure"}]}])
    assert degraded["effective_status"] == "DEGRADED_STEP_FAILURE"
    assert degraded["failed_steps"] == [("ml", "commodity_evaluate")]

    capture = {"name": "Options Capture", "path": ".github/workflows/options_capture.yml",
               "conclusion": "success", "id": 3}
    completed = ow.summarize_run_outcome(capture, [{"name": "capture", "conclusion": "success"}])
    assert completed["effective_status"] == "TECHNICALLY_SUCCESSFUL"
    assert completed["business_evidence"] == "NOT_EVALUATED"


def test_continue_on_error_stage_balance_not_misclassified_as_success():
    log = ("2026-11-03T16:04:44.421Z   CRITICAL: factory_plan=success " +
           "alpha_option_forward=success\n" +
           "2026-11-03T16:04:44.421Z   DEGRADED: alt_evaluate=success " +
           "commodity_evaluate=failure meta_learning=success\n" +
           "2026-11-03T16:04:44.428Z ##[warning]Stufe 'commodity_evaluate' ausgefallen\n")
    assert ow.stage_failure_names(log) == ["commodity_evaluate"]
    assert ow.stage_failure_names("DEGRADED: commodity_evaluate=success") == []
    assert ow._repo_from_jobs_url(
        "https://api.github.com/repos/pcctradinginc-alt/Adaptive-Asymmetry-Scanner/actions/runs/12/jobs"
    ) == "pcctradinginc-alt/Adaptive-Asymmetry-Scanner"

    class Response:
        def __init__(self, data=None, text=""):
            self.data, self.text = data, text

        def json(self):
            return self.data

        def raise_for_status(self):
            return None

    class Session:
        def get(self, url, params=None, timeout=None):
            if url.endswith("/runs/12/jobs"):
                return Response({"jobs": [{"name": "ml", "id": 5, "conclusion": "success",
                                          "steps": [{"name": "Commodity", "conclusion": "success"}]}]})
            if url.endswith("/jobs/5/logs"):
                return Response(text=log)
            raise AssertionError(url)

    outcome = ow.recent_run_outcomes(Session(), [{
        "name": "Research", "id": 12, "path": ".github/workflows/ml_research.yml",
        "conclusion": "success", "jobs_url":
        "https://api.github.com/repos/pcctradinginc-alt/Adaptive-Asymmetry-Scanner/actions/runs/12/jobs",
    }])
    assert outcome[0]["effective_status"] == "DEGRADED_STEP_FAILURE"
    assert outcome[0]["failed_steps"] == [("ml", "commodity_evaluate")]

"""Regression tests for the 2026-11-04 P0/P1 operational reliability fixes.

No network, no orders, no synthetic option marks or claimed source recovery.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

from modules import candidate_ledger as cl
from modules import market_snapshot as ms
from modules import ops_workflows as ow
from modules.external.sources import maritime as mw
from modules.external.sources.base import SourceStatus

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import rth_marks

NOW = datetime(2026, 11, 3, 19, 0, tzinfo=timezone.utc)
IMF_OWNER = "IMF-portwatch_imf_dataviz"          # Owner im produktiven Endpoint-Cache (2026-10-28)


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
        # Ersatz-Layer desselben, bereits verifizierten Herausgebers (Owner aus dem Cache)
        return {"ports_daily": {"service_url": state["current"], "layer_id": 0, "owner": IMF_OWNER},
                "ports_reference": {"service_url": state["current"], "layer_id": 0, "owner": IMF_OWNER}}

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
    from modules import roi_learning_shadow as rls
    from modules.autonomous_alpha import option_link as al
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



def test_capture_coverage_does_not_import_pandas_dependent_research(monkeypatch, tmp_path):
    """Capture runner installs only requests+pyyaml; weekly research may use pandas."""
    import shutil

    from modules.autonomous_alpha import option_link as al
    config = tmp_path / al.CONFIG
    config.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(__file__).resolve().parents[1] / al.CONFIG, config)
    (tmp_path / al.FACTOR_PRED_DIR).mkdir(parents=True, exist_ok=True)
    (tmp_path / al.FACTOR_PRED_DIR / "2026-11.jsonl").write_text("{}\\n")
    def no_heavy_sources(*args, **kwargs):
        raise AssertionError("capture must not load pandas-dependent research sources")
    monkeypatch.setattr(al, "_sources", no_heavy_sources)
    out = al.coverage(tmp_path, now=NOW, persist=False, include_source_stats=False)
    assert out["status"] == "NEED_MORE_DATA"
    assert out["source_stats"]["status"] == "SKIPPED_CAPTURE_LIGHTWEIGHT"
    assert out["n_decisions"] == 0



def test_missing_alpha_link_configuration_does_not_fake_capture_error(monkeypatch, tmp_path):
    from modules import roi_learning_shadow as rls
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ms, "us_market_session", lambda t: "regular")
    monkeypatch.setattr(cl, "due_mark_symbols", lambda d: [])
    monkeypatch.setattr(cl, "pending_entry_targets", lambda d: {})
    monkeypatch.setattr(rls, "due_capture_targets", lambda **kw: {})
    monkeypatch.setattr(rth_marks, "_satisfied", lambda *a, **kw: set())
    res = rth_marks.run(NOW, tmp_path / "no-history.json", capture_only=True)
    assert res["alpha_link"]["status"] == "NOT_CONFIGURED"
    assert "alpha_link_settlement" not in res
    assert res["capture_status"] == "NO_TARGETS"


# ── Review 2026-11-04 (Claude Code): Provenienz der Neuentdeckung, Push-Konflikt zweier Report-Schreiber ──
def _cache_with(tmp_path, owner=IMF_OWNER):
    import json
    cache = tmp_path / "portwatch_endpoints.json"
    ep = {"service_url": "https://services9.arcgis.com/imf/arcgis/rest/services/Daily_Ports_Data/FeatureServer",
          "layer_id": 0, "item_id": "imf-item", "title": "Daily_Ports_Data", "ambiguous": False,
          "owner": owner, "orgId": None}
    payload = {"discovered_at": NOW.isoformat(),
               "endpoints": {"ports_daily": ep, "ports_reference": {**ep, "fallback_to_daily_layer": True},
                             "chokepoints_daily": ep, "chokepoints_reference": ep}}
    cache.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return cache


def _foreign_search_fetch(foreign_url, calls):
    """ArcGIS-Suche liefert (wie bei einem zurückgezogenen IMF-Layer denkbar) die Kopie einer fremden Org."""
    def fetch(url, params=None, **kw):
        calls.append(url)
        if url == mw.ARCGIS_SEARCH_URL:
            return type("R", (), {"json": lambda self: {"results": [{
                "id": "foreign-item", "title": params["q"].split('"')[1], "type": "Feature Service",
                "url": foreign_url, "owner": "someone_else", "orgId": "foreignorg", "modified": 2}]}})()
        if url == foreign_url:
            return type("R", (), {"json": lambda self: {"layers": [{"id": 0}]}})()
        if url.endswith("/0") and "where" not in (params or {}):
            return type("R", (), {"json": lambda self: {"name": "x"}})()      # Metadaten ohne 'fields'
        raise AssertionError(f"unerwartete Abfrage {url} {params}")
    return fetch


def test_portwatch_rediscovery_rejects_foreign_publisher_and_restores_cache(tmp_path):
    """Ein Ersatz-Layer einer fremden Organisation wird nie genutzt, auch nicht im nächsten Lauf über den Cache."""
    cache = _cache_with(tmp_path)
    before = cache.read_text(encoding="utf-8")
    foreign = "https://services.arcgis.com/foreign/arcgis/rest/services/Daily_Ports_Data/FeatureServer"
    calls: list = []
    c = mw.PortWatchPortsConnector(http_fetch=_foreign_search_fetch(foreign, calls), cache_path=str(cache))
    result = c.fetch(NOW)
    assert result.status == SourceStatus.SCHEMA_CHANGED and not result.observations    # bleibt gesperrt
    assert foreign + "/0" not in calls                                    # fremder Layer nie abgefragt
    assert cache.read_text(encoding="utf-8") == before                    # Cache unverändert wiederhergestellt
    again = mw.PortWatchPortsConnector(http_fetch=_foreign_search_fetch(foreign, []), cache_path=str(cache))
    assert again._discover()["ports_daily"]["owner"] == IMF_OWNER         # Folge-Lauf liest weiter den IMF-Endpoint


def test_portwatch_rediscovery_without_verified_owner_fails_closed(tmp_path):
    cache = _cache_with(tmp_path, owner=None)
    foreign = "https://services.arcgis.com/foreign/arcgis/rest/services/Daily_Ports_Data/FeatureServer"
    calls: list = []
    c = mw.PortWatchPortsConnector(http_fetch=_foreign_search_fetch(foreign, calls), cache_path=str(cache))
    assert c.fetch(NOW).status == SourceStatus.SCHEMA_CHANGED and foreign + "/0" not in calls


def test_option_link_report_conflict_never_costs_the_quote_archive_push(tmp_path):
    """options_capture.yml und ml_research.yml (verschiedene concurrency-Gruppen) schreiben beide den abgeleiteten
    Report option_link.json. Ein Rebase-Konflikt darauf darf den Push der RTH-Quotes nicht abbrechen."""
    import subprocess

    from tests.test_ops_readiness_independent import _ci_sandbox
    c1, c2, ci_push, remote = _ci_sandbox(tmp_path)
    report = Path("outputs/research/autonomous_alpha/option_link.json")
    g = ["git", "-c", "user.name=t", "-c", "user.email=t@t"]
    (c2 / report).parent.mkdir(parents=True)
    (c2 / report).write_text('{"n_decisions": 1}\n', encoding="utf-8")
    assert ci_push(c2, "research base", "outputs/research/").returncode == 0
    subprocess.run(g + ["pull", "-q", "--rebase", "origin", "main"], cwd=c1, check=True, capture_output=True)
    (c2 / report).write_text('{"n_decisions": 2, "writer": "ml_research"}\n', encoding="utf-8")
    assert ci_push(c2, "ml research", "outputs/research/").returncode == 0
    quotes = c1 / "outputs/options_archive/quotes/2026-10-14.jsonl"
    quotes.write_text(quotes.read_text() + '{"id": "rth-quote"}\n', encoding="utf-8")
    (c1 / report).write_text('{"n_decisions": 3, "writer": "options_capture"}\n', encoding="utf-8")
    r = ci_push(c1, "options capture", "outputs/options_archive/",
                "outputs/research/autonomous_alpha/option_link/", str(report))
    assert r.returncode == 0, r.stdout + r.stderr
    assert '"rth-quote"' in remote("outputs/options_archive/quotes/2026-10-14.jsonl")   # Quote-Archiv gesichert
    assert "options_capture" in remote(str(report))                     # Version dieses Laufs (abgeleitet)


def test_ops_report_surfaces_green_but_incomplete_runs():
    """recent_outcomes darf nicht nur im JSON stehen: der Bericht zeigt übersprungene Scans und verdeckte
    continue-on-error-Ausfälle, statt allein 'Keine fehlgeschlagenen Läufe.' zu melden."""
    from modules import ops_report
    from tests.test_ops_report import make_status
    s = make_status()
    s["workflows"] = {"status": "OK", "since": "2026-11-03T00:00:00Z", "window_hours": 24, "runs_total": 3,
                      "failed": [], "scanner_start": {}, "recent_outcomes": [
                          {"workflow": "Adaptive Asymmetry Scanner", "effective_status": "SKIPPED_SCAN",
                           "conclusion": "success", "skipped_jobs": ["scan"], "failed_steps": []},
                          {"workflow": "ML Research", "effective_status": "DEGRADED_STEP_FAILURE",
                           "conclusion": "success", "skipped_jobs": [], "failed_steps": [["ml", "commodity_evaluate"]]},
                          {"workflow": "Options Capture", "effective_status": "TECHNICALLY_SUCCESSFUL",
                           "conclusion": "success", "skipped_jobs": [], "failed_steps": []}]}
    row = ops_report.history_row(s)
    md = ops_report.render_markdown(s, row, ops_report.deltas(row, []), [])
    assert "SKIPPED_SCAN" in md and "DEGRADED_STEP_FAILURE" in md and "ml/commodity_evaluate" in md
    assert "Options Capture" not in md.split("Grün, aber nicht vollständig ausgeführt")[1].split("##")[0]

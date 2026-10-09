"""Reliability-Maintenance 2026-10-09: Runner-Pinning, Dependency-Order/Freshness-Gate, Missed-Run-Watchdog
(einmalige Recovery, kein Endlos-Retry, Idempotenz), fairer Haiku-vs-Sonnet-Vergleich (TRUNCATED),
Wetter-Partitionierung (gleicher Bestand, keine Duplikate, PIT), Drift-Input-Freshness (nur Kennzeichnung)."""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from modules import workflow_health as wh

WF_DIR = Path(".github/workflows")
UTC = timezone.utc


def _jobs(path: Path) -> dict:
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("jobs") or {}


# ── 1. Runner ────────────────────────────────────────────────────────────────────────────────────────────
def test_all_productive_workflows_pinned_to_ubuntu_2404():
    files = sorted(WF_DIR.glob("*.yml"))
    assert files
    for f in files:
        runs_on = [ln.split("runs-on:", 1)[1].strip() for ln in f.read_text(encoding="utf-8").splitlines()
                   if "runs-on:" in ln and not ln.lstrip().startswith("#")]
        assert runs_on and all("latest" not in r for r in runs_on), f"{f.name}: {runs_on}"
        if f.name == "compat_ubuntu26.yml":
            continue
        for name, job in _jobs(f).items():
            assert job.get("runs-on") == "ubuntu-24.04", f"{f.name}:{name} -> {job.get('runs-on')}"


def test_ubuntu26_compat_check_is_non_productive():
    f = WF_DIR / "compat_ubuntu26.yml"
    doc = yaml.safe_load(f.read_text(encoding="utf-8"))
    text = f.read_text(encoding="utf-8")
    assert {j["runs-on"] for j in doc["jobs"].values()} == {"ubuntu-26.04"}
    assert doc["permissions"] == {"contents": "read"}                     # kann nichts schreiben
    assert "ci_push" not in text and "git push" not in text and "secrets." not in text
    assert "pytest" in text


# ── 2./3. Dependency-Order + Freshness-Gate ──────────────────────────────────────────────────────────────
def _root(tmp_path: Path, snap_ts: str | None = None, ext_ts: str | None = None, report_day: str | None = None) -> Path:
    if snap_ts is not None:
        p = tmp_path / "outputs/health/source_health_snapshot.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"generated": snap_ts}))
    if ext_ts is not None:
        p = tmp_path / "outputs/external_data/health/source_health.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"a": {"last_attempt": ext_ts}, "b": {"last_attempt": "2026-09-01T00:00:00+00:00"}}))
    if report_day is not None:
        p = tmp_path / f"outputs/daily_reports/{report_day}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
    return tmp_path


FRI_2000 = datetime(2026, 10, 9, 20, 0, tzinfo=UTC)          # Freitag, typische (verspätete) Scanner-Zeit


def test_scanner_blocked_when_source_health_stale(tmp_path):
    r = wh.upstream_ready("scanner", FRI_2000, _root(tmp_path, snap_ts="2026-10-08T19:00:33+00:00",
                                                     ext_ts="2026-10-09T13:21:00+00:00"))
    assert r["run"] is False and r["status"] == "UPSTREAM_NOT_READY"
    assert "source_health_snapshot STALE" in r["reason"] and "2026-10-08" in r["reason"]


def test_scanner_blocked_when_source_health_missing(tmp_path):
    r = wh.upstream_ready("scanner", FRI_2000, _root(tmp_path))
    assert r["run"] is False and r["status"] == "UPSTREAM_NOT_READY" and "MISSING" in r["reason"]


def test_future_timestamp_is_never_treated_as_fresh(tmp_path):
    r = wh.upstream_ready("scanner", FRI_2000, _root(tmp_path, snap_ts="2026-10-10T08:00:00+00:00"))
    assert r["run"] is False and r["checks"][0]["status"] == "FUTURE_TIMESTAMP"


def test_scanner_runs_with_fresh_snapshot(tmp_path):
    r = wh.upstream_ready("scanner", FRI_2000, _root(tmp_path, snap_ts="2026-10-09T19:00:00+00:00",
                                                     ext_ts="2026-10-09T13:21:00+00:00"))
    assert r["run"] is True and r["status"] == "RUN"


def test_external_data_is_advisory_only_for_scanner(tmp_path):
    """External Data ist Observability (nicht production) -> meldet, blockiert den Scanner nicht."""
    r = wh.upstream_ready("scanner", FRI_2000, _root(tmp_path, snap_ts="2026-10-09T19:00:00+00:00",
                                                     ext_ts="2026-10-05T13:00:00+00:00"))
    assert r["run"] is True and r["advisory"][0]["status"] == "STALE"


def test_workflow_run_cannot_pull_scanner_before_its_trading_time(tmp_path):
    early = datetime(2026, 10, 9, 12, 50, tzinfo=UTC)        # Source Health ohne Verzögerung fertig
    r = wh.upstream_ready("scanner", early, _root(tmp_path, snap_ts="2026-10-09T12:45:00+00:00"))
    assert r["run"] is False and r["status"] == "SKIP_BEFORE_WINDOW"


def test_scanner_not_on_weekend(tmp_path):
    sat = datetime(2026, 10, 10, 19, 5, tzinfo=UTC)
    r = wh.upstream_ready("scanner", sat, _root(tmp_path, snap_ts="2026-10-10T19:00:00+00:00"))
    assert r["status"] == "SKIP_NOT_TRADING_DAY" and r["run"] is False


def test_scanner_guard_is_idempotent_daily_report_exists(tmp_path):
    root = _root(tmp_path, snap_ts="2026-10-09T19:00:00+00:00", report_day="2026-10-09")
    for _ in range(2):                                         # Recovery zweimal -> nie ein zweiter Scan
        r = wh.upstream_ready("scanner", FRI_2000, root)
        assert r["run"] is False and r["status"] == "SKIP_ALREADY_RAN"


def test_scanner_workflow_wired_to_source_health_and_guard():
    sc = yaml.safe_load((WF_DIR / "scanner.yml").read_text(encoding="utf-8"))
    sh = yaml.safe_load((WF_DIR / "source_health.yml").read_text(encoding="utf-8"))
    on = sc[True]                                               # PyYAML liest "on" als True
    assert on["workflow_run"]["workflows"] == [sh["name"]]      # exakter Workflow-Name, sonst feuert nichts
    assert on["workflow_run"]["types"] == ["completed"]
    assert "schedule" in on                                     # Cron bleibt Fallback/Startfenster
    scan = sc["jobs"]["scan"]
    assert scan["needs"] == "guard" and scan["if"] == "needs.guard.outputs.run == 'true'"
    steps = sc["jobs"]["guard"]["steps"]
    assert any("upstream_guard.py scanner" in str(s.get("run", "")) for s in steps)
    assert sc["concurrency"]["group"] == "history-write"         # Guard läuft erst nach einem laufenden Scan


def test_upstream_guard_script_writes_outputs_and_warning(tmp_path, monkeypatch, capsys):
    import scripts.upstream_guard as ug
    out = tmp_path / "gh_out"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    monkeypatch.setattr(ug, "upstream_ready", lambda job, now: {"job": job, "run": False, "status": "UPSTREAM_NOT_READY",
                                                               "reason": "x STALE", "checks": [], "advisory": []})
    assert ug.main(["scanner"]) == 0
    assert "run=false" in out.read_text() and "status=UPSTREAM_NOT_READY" in out.read_text()
    assert "::warning::UPSTREAM_NOT_READY scanner" in capsys.readouterr().out
    out.write_text("")
    assert ug.main(["scanner", "--force"]) == 0                 # manuelle Übersteuerung: protokolliert
    assert "run=true" in out.read_text() and "status=FORCED" in out.read_text()


# ── 4. Missed-Run / Recovery ─────────────────────────────────────────────────────────────────────────────
CFG = wh.load_cfg()
LATE = datetime(2026, 10, 9, 23, 0, tzinfo=UTC)               # nach allen Fristen (Scanner 13:30 + 9 h)


def _ok(created):
    return {"created_at": created, "status": "completed", "conclusion": "success", "event": "schedule"}


def _runs(**over):
    base = {"external_data": [_ok("2026-10-09T13:20:00Z")], "source_health": [_ok("2026-10-09T19:00:00Z")],
            "scanner": [], "feedback": [_ok("2026-10-09T21:30:00Z")]}
    base.update(over)
    return base


ART = {"external_data": True, "source_health": True, "scanner": False, "feedback": None}


def test_missing_run_detected_and_exactly_one_dispatch():
    p = wh.plan_recovery(LATE, _runs(), CFG, artifacts=ART)
    assert p["scanner"]["state"] == "MISSED" and p["scanner"]["action"] == "DISPATCH"
    assert [j for j, r in p.items() if r["action"] == "DISPATCH"] == ["scanner"]
    assert p["source_health"]["delay_hours"] == pytest.approx(6.3)  # 12:41 -> 19:00 sichtbar als Verzögerung


def test_green_run_without_artifact_is_not_ok():
    """Guard-Skip (UPSTREAM_NOT_READY) ist grün, versorgt den Tag aber nicht -> trotzdem verpasst."""
    p = wh.plan_recovery(LATE, _runs(scanner=[_ok("2026-10-09T20:00:00Z")]), CFG, artifacts=ART)
    assert p["scanner"]["state"] == "MISSED" and p["scanner"]["action"] == "DISPATCH"


def test_no_endless_retry_after_one_recovery():
    failed_recovery = {"created_at": "2026-10-09T22:30:00Z", "status": "completed", "conclusion": "failure",
                       "event": "workflow_dispatch"}
    p = wh.plan_recovery(LATE, _runs(scanner=[failed_recovery]), CFG, artifacts=ART)
    assert p["scanner"]["state"] == "RECOVERY_EXHAUSTED" and p["scanner"]["action"] is None
    p2 = wh.plan_recovery(LATE, _runs(), CFG, artifacts=ART, recoveries={"scanner": 1})   # Zähler aus Statusdatei
    assert p2["scanner"]["state"] == "RECOVERY_EXHAUSTED" and p2["scanner"]["action"] is None


def test_no_dispatch_while_running_pending_or_unknown():
    running = {"created_at": "2026-10-09T22:50:00Z", "status": "in_progress", "conclusion": None, "event": "schedule"}
    assert wh.plan_recovery(LATE, _runs(scanner=[running]), CFG, artifacts=ART)["scanner"]["state"] == "RUNNING"
    early = datetime(2026, 10, 9, 20, 0, tzinfo=UTC)          # 6,5 h Verzögerung = normal, keine Panik
    assert wh.plan_recovery(early, _runs(), CFG, artifacts=ART)["scanner"]["state"] == "PENDING"
    unk = wh.plan_recovery(LATE, _runs(scanner=None), CFG, artifacts=ART)["scanner"]
    assert unk["state"] == "UNKNOWN" and unk["action"] is None                  # API-Fehler: nie blind dispatchen


def test_downstream_not_dispatched_while_required_upstream_missing():
    p = wh.plan_recovery(LATE, _runs(source_health=[]), CFG,
                         artifacts={**ART, "source_health": False})
    assert p["source_health"]["action"] == "DISPATCH"                           # zuerst der Upstream
    assert p["scanner"]["action"] is None and p["scanner"]["stale_upstream"] == ["source_health"]


def test_source_health_does_not_depend_on_observability_external_data():
    p = wh.plan_recovery(LATE, _runs(external_data=[], source_health=[]), CFG,
                         artifacts={**ART, "external_data": False, "source_health": False})
    assert p["external_data"]["action"] == "DISPATCH" and p["source_health"]["action"] == "DISPATCH"


def test_watchdog_twice_dispatches_once_and_records(tmp_path, monkeypatch):
    import scripts.workflow_watchdog as wd
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config/workflow_schedule.yaml").write_text(
        (Path(__file__).resolve().parent.parent / "config/workflow_schedule.yaml").read_text(encoding="utf-8"))
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    monkeypatch.setenv("GITHUB_TOKEN", "t")
    calls = []
    monkeypatch.setattr(wd, "list_runs", lambda repo, f, day: [])               # API zeigt den Dispatch (noch) nicht
    monkeypatch.setattr(wd, "dispatch", lambda repo, f, ref: calls.append(f) or True)
    monkeypatch.setattr(wd, "artifact_ok", lambda spec, day: False)

    class _DT(datetime):
        @classmethod
        def now(cls, tz=None):
            return LATE
    monkeypatch.setattr(wd, "datetime", _DT)
    wd.main([])
    wd.main([])                                                                  # zweiter Watchdog-Lauf
    st = json.loads((tmp_path / "outputs/state/workflow_status.json").read_text())
    assert sorted(calls) == ["external_data.yml", "source_health.yml"]          # je Workflow genau einmal
    assert st["recoveries"]["2026-10-09"] == {"external_data": 1, "source_health": 1}
    assert st["days"]["2026-10-09"]["external_data"]["state"] == "RECOVERY_EXHAUSTED"   # 2. Lauf: kein Retry
    assert st["days"]["2026-10-09"]["feedback"]["state"] == "PENDING"            # 15:30 + 8 h Frist läuft noch
    assert st["days"]["2026-10-09"]["scanner"]["stale_upstream"] == ["source_health"]


def test_watchdog_without_token_never_dispatches(tmp_path, monkeypatch):
    import scripts.workflow_watchdog as wd
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(wd, "STATUS", tmp_path / "s.json")
    monkeypatch.setattr(wd, "dispatch", lambda *a: pytest.fail("darf nicht dispatchen"))
    wd.main([])
    st = json.loads((tmp_path / "s.json").read_text())
    assert all(r["action"] is None for r in next(iter(st["days"].values())).values())


def test_status_file_has_no_volatile_timestamp_and_prunes():
    plan = {"scanner": {"state": "OK"}}
    a = wh.merge_status(None, date(2026, 10, 9), plan, {})
    b = wh.merge_status(a, date(2026, 10, 9), plan, {})
    assert a == b                                     # gleicher Zustand -> gleiche Datei -> kein Commit
    old = wh.merge_status({"days": {"2026-09-01": plan}, "recoveries": {"2026-09-01": {"scanner": 1}}},
                          date(2026, 10, 9), plan, {}, keep_days=21)
    assert "2026-09-01" not in old["days"] and "2026-09-01" not in old["recoveries"]


# ── 5. Workflow-Status im Montagsreport ──────────────────────────────────────────────────────────────────
def test_weekly_workflow_status_block():
    from reports import weekly as w
    p = wh.plan_recovery(LATE, _runs(), CFG, artifacts=ART)
    p["scanner"]["state"], p["scanner"]["recovered"] = "RECOVERY_TRIGGERED", True
    st = wh.merge_status(None, LATE.date(), p, {"scanner": 1})
    blocks = w.workflow_status_blocks({"workflow_status": wh.weekly_summary(st, date(2026, 10, 2), date(2026, 10, 9))})
    table = next(b for b in blocks if b[0] == "table")
    assert table[1][:4] == ["Workflow", "erwartet", "letzter Lauf", "Verzögerung Median/Max"]
    rows = {r[0]: r for r in table[2]}
    assert rows["scanner"][4] == "1" and rows["scanner"][5] == "0"          # recovered, nicht verpasst
    assert rows["source_health"][3].startswith("6.3")
    assert "kein Systemfehler" in blocks[-1][1]
    assert "Watchdog noch ohne Historie" in w.workflow_status_blocks({})[0][1]


# ── 6. Haiku vs. Sonnet: TRUNCATED zählt nicht ───────────────────────────────────────────────────────────
def _pair(i, gate_ref, gate_other, stop_other=None):
    from modules import model_routing as mr
    ref = {"impact": 6 if gate_ref else 2, "surprise": 5 if gate_ref else 1, "direction": "BULLISH"}
    oth = {"impact": 6 if gate_other else 2, "surprise": 5 if gate_other else 1, "direction": "BULLISH"}
    row = mr.pair_row("deep_analysis", f"T{i}", "sonnet", "haiku", ref, oth, other_stop_reason=stop_other)
    return {**row, "kind": "route_pair", "workflow": "deep_analysis", "run_id": "r1"}


def test_truncated_response_is_not_counted_as_model_error():
    from modules import model_routing as mr
    rows = ([_pair(i, True, True) for i in range(3)] + [_pair(3, True, False)]
            + [_pair(10 + i, True, False, stop_other="max_tokens") for i in range(4)])
    rep = mr.comparison_report(rows)
    assert rep["pairs_total"] == 8 and rep["valid_comparisons"] == 4 and rep["truncated"] == 4
    assert rep["challenger_stricter"] == 1 and rep["equal_gate_decision"] == 3      # nicht 5 "Fehler"
    assert all(p["comparison_status"] == "TRUNCATED" for p in rows[4:])


def test_legacy_pairs_annotated_from_llm_rows():
    from modules import model_routing as mr
    legacy = {k: v for k, v in _pair(1, True, False).items() if k not in ("comparison_status", "other_stop_reason")}
    llm = {"kind": "llm", "run_id": "r1", "ticker": "T1", "model": "haiku", "stop_reason": "max_tokens"}
    out = mr.annotate_truncation([legacy], [llm])
    assert out[0]["comparison_status"] == "TRUNCATED"
    assert mr.annotate_truncation([legacy], [])[0]["comparison_status"] == "VALID"
    assert mr.valid_pairs(out) == []


def test_deep_analysis_records_stop_reasons_on_pair(monkeypatch):
    from types import SimpleNamespace
    from modules import cost_telemetry, deep_analysis as da
    rec = []
    monkeypatch.setattr(cost_telemetry, "record", rec.append)
    obj = object.__new__(da.DeepAnalysis)
    monkeypatch.setattr(obj, "_parse_json", lambda msg, t: {"impact": 6, "surprise": 5}, raising=False)
    obj._record_pair({"ticker": "AAA", "params": {"model": "sonnet"}}, {"impact": 6, "surprise": 5},
                     SimpleNamespace(stop_reason="max_tokens"), {"other": "haiku"},
                     ref_msg=SimpleNamespace(stop_reason="end_turn"))
    assert rec[0]["comparison_status"] == "TRUNCATED" and rec[0]["other_stop_reason"] == "max_tokens"
    assert rec[0]["ref_stop_reason"] == "end_turn"


def test_token_budget_raised_only_for_challenger_ab_sample():
    from modules import model_routing as mr
    pol = yaml.safe_load(Path("config/cost_policy.yaml").read_text(encoding="utf-8"))
    da = pol["model_routing"]["deep_analysis"]
    from modules.config import cfg
    assert mr.max_tokens_for("deep_analysis", cfg.models.deep_analysis, pol) == 1600   # Produktion unverändert
    assert mr.max_tokens_for("deep_analysis", "claude-haiku-4-5-20251001", pol) == 2400
    assert da.get("mode") != "challenger"                                        # kein erzwungener Modellwechsel


# ── 7. Wetter-Partitionierung ────────────────────────────────────────────────────────────────────────────
def _wobs(day: int, value: float, retrieved: datetime):
    from modules.external.pit import AvailabilityPrecision, Observation
    t = datetime(2026, 10, day, 12, tzinfo=UTC)
    return Observation(source_id="nws_forecast", dataset="forecast", series_id="temp", entity_id="KNYC",
                       metric="temp_f", value=value, unit="F", observation_time=t, available_at=retrieved,
                       retrieved_at=retrieved, forecast_issue_time=retrieved, forecast_valid_time=t,
                       availability_precision=AvailabilityPrecision.EXACT_DATE, parser_version="1")


def test_weekly_partition_same_data_no_duplicates_pit(tmp_path):
    from modules.external.archive import ExternalArchive
    r1 = datetime(2026, 10, 1, 6, tzinfo=UTC)
    r2 = datetime(2026, 10, 9, 6, tzinfo=UTC)
    batch1 = [_wobs(d, 60.0 + d, r1) for d in (1, 5, 8, 15, 29)]
    batch2 = [_wobs(d, 70.0 + d, r2) for d in (8, 15)]                      # neue Vintages (Revision)
    mono, week = ExternalArchive(root=tmp_path / "m"), ExternalArchive(root=tmp_path / "w")
    for a, part in ((mono, "monthly"), (week, "weekly")):
        a.store_observations(batch1, partition=part)
        a.store_observations(batch2, partition=part)
        again = a.store_observations(batch1 + batch2, partition=part)       # idempotenter Re-Run
        assert again["new"] == 0 and again["revision"] == 0
    files = sorted(p.name for p in (tmp_path / "w/normalized/nws_forecast").glob("*.jsonl"))
    assert files == ["2026-10-w1.jsonl", "2026-10-w2.jsonl", "2026-10-w3.jsonl", "2026-10-w5.jsonl"]

    def key(o):
        return (o.observation_time, o.value, o.available_at)
    assert sorted(map(key, mono.load("nws_forecast"))) == sorted(map(key, week.load("nws_forecast")))
    for t in (r1 + timedelta(hours=1), r2 + timedelta(hours=1)):            # PIT-Stand identisch
        assert sorted(map(key, mono.as_of("nws_forecast", t))) == sorted(map(key, week.as_of("nws_forecast", t)))
    assert {o.value for o in week.as_of("nws_forecast", r1 + timedelta(hours=1))} == {61.0, 65.0, 68.0, 75.0, 89.0}


def test_weekly_partition_dedups_against_legacy_month_file(tmp_path):
    from modules.external.archive import ExternalArchive
    a = ExternalArchive(root=tmp_path)
    r1 = datetime(2026, 10, 1, 6, tzinfo=UTC)
    a.store_observations([_wobs(3, 61.0, r1), _wobs(9, 62.0, r1)], partition="monthly")     # Altbestand 2026-10.jsonl
    res = a.store_observations([_wobs(3, 61.0, r1), _wobs(9, 62.0, r1)], partition="weekly")
    assert res["new"] == 0 and res["duplicate"] == 2
    assert len(a.load("nws_forecast")) == 2
    assert not list((tmp_path / "normalized/nws_forecast").glob("*-w*.jsonl"))


def test_weather_source_configured_weekly_others_unchanged():
    w = yaml.safe_load(Path("config/external_sources/weather.yaml").read_text(encoding="utf-8"))
    srcs = w.get("sources") or w
    flat = srcs if isinstance(srcs, dict) else {s.get("source_id") or s.get("id"): s for s in srcs}
    assert flat["nws_forecast"]["normalized_partition"] == "weekly"
    assert all(not v.get("normalized_partition") for k, v in flat.items() if k != "nws_forecast" and isinstance(v, dict))


# ── 8. Drift-Freshness (nur Kennzeichnung) ───────────────────────────────────────────────────────────────
def _state(tmp_path, gen: str | None, now: datetime) -> dict:
    from modules import system_state as ss
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"generated": gen} if gen else {}))
    inputs = {k: tmp_path / f"missing_{k}.json" for k in ss.DEFAULT_INPUTS}
    inputs["meta_learning"] = meta
    return ss.derive(inputs, now=now)


def test_drift_input_freshness_is_labelled_only(tmp_path):
    from modules import system_state as ss
    now = datetime(2026, 10, 9, 12, tzinfo=UTC)
    fresh = _state(tmp_path, "2026-10-05T10:00:00+00:00", now)
    stale = _state(tmp_path, "2026-09-28T10:00:00+00:00", now)
    unk = _state(tmp_path, None, now)
    assert fresh["drift_state"]["drift_input_status"] == "FRESH"
    assert stale["drift_state"]["drift_input_status"] == "STALE_DRIFT_INPUT"
    assert stale["drift_state"]["drift_input_age_days"] == pytest.approx(11.1, abs=0.05)
    assert unk["drift_state"]["drift_input_status"] == "UNKNOWN"
    for k in ("level", "consequences", "reasons"):                    # Policy-Ergebnis unverändert
        assert fresh["drift_state"][k] == stale["drift_state"][k]
    assert ss.DRIFT_INPUT_MAX_AGE_DAYS == 8


def test_drift_age_alone_does_not_change_fingerprint(tmp_path):
    from modules import system_state as ss
    a = _state(tmp_path, "2026-10-05T10:00:00+00:00", datetime(2026, 10, 6, 12, tzinfo=UTC))
    b = _state(tmp_path, "2026-10-05T10:00:00+00:00", datetime(2026, 10, 7, 12, tzinfo=UTC))
    assert a["drift_state"]["drift_input_age_days"] != b["drift_state"]["drift_input_age_days"]
    assert ss._fingerprint(a) == ss._fingerprint(b)


def test_weekly_drift_line_shows_input_freshness():
    from reports import weekly as w
    t = w.drift_input_text({"drift_input_status": "STALE_DRIFT_INPUT", "drift_input_timestamp": "2026-09-28T10:00:00",
                            "drift_input_age_days": 11.1})
    assert "2026-09-28" in t and "STALE_DRIFT_INPUT" in t
    assert w.drift_input_text({}) == ""

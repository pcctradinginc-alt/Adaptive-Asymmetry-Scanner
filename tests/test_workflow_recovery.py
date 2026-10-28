"""Recovery-Härtung 2026-10-19: Am Mo 19.10. lief kein Scanner. Der Source-Health-Cron (12:41 UTC) kam nicht,
die Scanner-Crons warteten auf einen frischen Snapshot, und die Watchdog-Frist (12:41 + 9 h = 21:41 UTC) lag hinter
dem Börsenschluss (20:00 UTC); der Watchdog-Cron selbst kam nur 2 von 12 Mal. Geprüft wird die Kette
Source Health -> Scanner -> Watchdog mit der Repo-Config: kürzere Fristen, spätester Start vor Sitzungsschluss
(MISSED statt verspätetem Produktionsscan), genau ein Recovery-Dispatch, redundante Watchdog-Auslösung."""
from __future__ import annotations

import json
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from modules import workflow_health as wh

REPO = Path(__file__).resolve().parent.parent
WF_DIR = REPO / ".github" / "workflows"
CFG = wh.load_cfg(REPO / "config" / "workflow_schedule.yaml")
UTC = timezone.utc


def utc(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


def run(created: str, *, event: str = "schedule", status: str = "completed", conclusion: str | None = "success"):
    return {"created_at": created, "status": status, "conclusion": conclusion, "event": event}


def root(tmp_path: Path, *, snap: str | None = None, report_day: str | None = None) -> Path:
    if snap:
        p = tmp_path / "outputs/health/source_health_snapshot.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"generated": snap}))
    if report_day:
        p = tmp_path / f"outputs/daily_reports/{report_day}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
    return tmp_path


def guard(tmp_path: Path, now: str, **kw) -> dict:
    return wh.upstream_ready("scanner", utc(now), root=root(tmp_path, **kw), cfg=CFG)


def plan(now: str, runs: dict, artifacts: dict, recoveries: dict | None = None, cfg: dict | None = None) -> dict:
    base = {"external_data": [run(now[:10] + "T11:00:00Z")], "source_health": [], "scanner": [], "feedback": []}
    return wh.plan_recovery(utc(now), {**base, **runs}, cfg or CFG,
                            artifacts={"external_data": True, "feedback": None, **artifacts}, recoveries=recoveries)


def dispatches(p: dict) -> list[str]:
    return sorted(j for j, r in p.items() if r["action"] == "DISPATCH")


# ── Root Cause 19.10. ────────────────────────────────────────────────────────────────────────────────────
def test_root_cause_alte_frist_haette_erst_nach_boersenschluss_recovered():
    """Stand 19.10. 18:29 UTC (14:29 ET): kein Source-Health-Lauf. Mit 9 h Frist: alles PENDING, recovered=False,
    die erste mögliche Recovery 21:41 UTC = 17:41 ET – nach Börsenschluss. Mit der neuen Frist: Dispatch."""
    old = {**CFG, "workflows": {**CFG["workflows"],
                                "source_health": {**CFG["workflows"]["source_health"], "max_delay_hours": 9},
                                "scanner": {**CFG["workflows"]["scanner"], "max_delay_hours": 9}},
           "gates": {"scanner": {k: v for k, v in CFG["gates"]["scanner"].items()
                                 if k != "latest_start_before_close_minutes"}}}
    alt = plan("2026-10-19T18:29", {}, {"source_health": False, "scanner": False}, cfg=old)
    assert alt["source_health"]["state"] == "PENDING" and dispatches(alt) == []
    assert alt["source_health"]["deadline"] == "2026-10-19T21:41+00:00"         # 17:41 ET, Markt zu
    neu = plan("2026-10-19T18:29", {}, {"source_health": False, "scanner": False})
    assert dispatches(neu) == ["source_health"]


# ── 1. Normaler Tagesablauf ──────────────────────────────────────────────────────────────────────────────
def test_normaler_tagesablauf(tmp_path):
    assert guard(tmp_path, "2026-10-19T17:02", snap="2026-10-19T17:00:00+00:00")["status"] == "RUN"
    p = plan("2026-10-19T18:00", {"source_health": [run("2026-10-19T17:00:00Z")],
                                  "scanner": [run("2026-10-19T17:02:00Z", event="workflow_run")]},
             {"source_health": True, "scanner": True})
    assert {j: r["state"] for j, r in p.items()} == {"external_data": "OK", "source_health": "OK", "scanner": "OK",
                                                    "feedback": "PENDING"}
    assert dispatches(p) == []


# ── 2. Source-Health-Cron fällt aus ─────────────────────────────────────────────────────────────────────
def test_source_health_cron_faellt_aus_recovery_vor_dem_scanner(tmp_path):
    vor = plan("2026-10-19T15:30", {}, {"source_health": False, "scanner": False})
    assert vor["source_health"]["state"] == "PENDING" and dispatches(vor) == []      # 3 h Cron-Verzug sind normal
    p = plan("2026-10-19T15:45", {}, {"source_health": False, "scanner": False})
    assert dispatches(p) == ["source_health"]                                         # genau der Upstream
    assert p["scanner"]["state"] == "PENDING" and p["scanner"]["stale_upstream"] == ["source_health"]
    # Recovery-Lauf fertig -> workflow_run startet den Scanner: 11:48 ET, frischer Snapshot -> Produktionsscan
    assert guard(tmp_path, "2026-10-19T15:48", snap="2026-10-19T15:47:00+00:00")["status"] == "RUN"


def test_source_health_ausfall_nach_frist_scanner_wartet_auf_upstream():
    p = plan("2026-10-19T17:40", {}, {"source_health": False, "scanner": False}, recoveries={"source_health": 1})
    assert p["source_health"]["state"] == "RECOVERY_EXHAUSTED"
    assert p["scanner"]["state"] == "MISSED" and p["scanner"]["action"] is None
    assert "source_health" in p["scanner"]["reason"]


# ── 3. Scanner-Cron fällt aus ───────────────────────────────────────────────────────────────────────────
def test_scanner_cron_faellt_aus():
    sh = {"source_health": [run("2026-10-19T14:00:00Z")]}
    assert plan("2026-10-19T17:00", sh, {"source_health": True, "scanner": False})["scanner"]["state"] == "PENDING"
    p = plan("2026-10-19T17:31", sh, {"source_health": True, "scanner": False})
    assert dispatches(p) == ["scanner"] and p["scanner"]["deadline"] == "2026-10-19T17:30+00:00"


def test_gruener_guard_skip_versorgt_den_tag_nicht():
    """success heißt nicht versorgt: UPSTREAM_NOT_READY/SKIP_BEFORE_WINDOW enden grün ohne Tagesreport."""
    p = plan("2026-10-19T17:31", {"source_health": [run("2026-10-19T14:00:00Z")],
                                  "scanner": [run("2026-10-19T13:35:00Z"), run("2026-10-19T14:01:00Z", event="workflow_run")]},
             {"source_health": True, "scanner": False})
    assert dispatches(p) == ["scanner"]


# ── 4. Source Health läuft verspätet ────────────────────────────────────────────────────────────────────
def test_source_health_verspaetet_aber_im_fenster(tmp_path):
    assert guard(tmp_path, "2026-10-19T18:32", snap="2026-10-19T18:31:00+00:00")["status"] == "RUN"   # 14:32 ET


def test_source_health_zu_spaet_kein_produktionsscan_sondern_missed(tmp_path, capsys, monkeypatch):
    r = guard(tmp_path, "2026-10-19T19:10", snap="2026-10-19T19:08:00+00:00")       # 15:10 ET, Schluss − 50 min
    assert r["run"] is False and r["status"] == "MISSED_WINDOW_CLOSED" and "MISSED" in r["reason"]
    import scripts.upstream_guard as ug
    out = tmp_path / "gh_out"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    monkeypatch.setattr(ug, "upstream_ready", lambda job, now: r)
    ug.main(["scanner"])
    assert "run=false" in out.read_text()
    assert "::warning::MISSED_WINDOW_CLOSED scanner" in capsys.readouterr().out


# ── 5. Source-Health-Snapshot veraltet ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("snap", ["2026-10-18T16:06:26+00:00",       # Stand am 19.10. (Sonntagslauf)
                                  "2026-10-19T23:00:00+00:00"])      # Zukunft = unplausibel
def test_veralteter_snapshot_blockiert_den_scanner(tmp_path, snap):
    r = guard(tmp_path, "2026-10-19T16:00", snap=snap)
    assert r["run"] is False and r["status"] == "UPSTREAM_NOT_READY"


def test_fehlender_snapshot_blockiert_den_scanner(tmp_path):
    assert guard(tmp_path, "2026-10-19T16:00")["status"] == "UPSTREAM_NOT_READY"


# ── 6. Bereits erfolgreich ausgeführter Scanner ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("now", ["2026-10-19T17:00", "2026-10-19T19:30"])
def test_bereits_gelaufen_ist_idempotent_auch_nach_dem_fenster(tmp_path, now):
    r = guard(tmp_path, now, snap="2026-10-19T16:00:00+00:00", report_day="2026-10-19")
    assert r["status"] == "SKIP_ALREADY_RAN"                                         # nicht MISSED
    p = plan(now, {"source_health": [run("2026-10-19T16:00:00Z")], "scanner": [run("2026-10-19T16:02:00Z")]},
             {"source_health": True, "scanner": True})
    assert p["scanner"]["state"] == "OK" and dispatches(p) == []


# ── 7. Gleichzeitiger Cron- und Recovery-Trigger ────────────────────────────────────────────────────────
def test_gleichzeitiger_cron_und_recovery_trigger(tmp_path):
    sh = {"source_health": [run("2026-10-19T14:00:00Z")]}
    laufend = plan("2026-10-19T17:45", {**sh, "scanner": [run("2026-10-19T17:40:00Z", status="in_progress",
                                                               conclusion=None)]},
                   {"source_health": True, "scanner": False})
    assert laufend["scanner"]["state"] == "RUNNING" and dispatches(laufend) == []
    # Recovery bereits ausgelöst (Statusdatei verloren, API zeigt den Dispatch-Run) -> nie ein zweiter Dispatch
    p = plan("2026-10-19T18:10", {**sh, "scanner": [run("2026-10-19T17:35:00Z", event="workflow_dispatch",
                                                         conclusion="failure")]},
             {"source_health": True, "scanner": False})
    assert p["scanner"]["state"] == "RECOVERY_EXHAUSTED" and dispatches(p) == []
    # Concurrency serialisiert: der zweite Guard sieht den Tagesreport des ersten Laufs
    sc = yaml.safe_load((WF_DIR / "scanner.yml").read_text(encoding="utf-8"))
    assert sc["concurrency"] == {"group": "history-write", "cancel-in-progress": False}
    assert guard(tmp_path, "2026-10-19T18:20", snap="2026-10-19T14:00:00+00:00",
                 report_day="2026-10-19")["status"] == "SKIP_ALREADY_RAN"


# ── 8. GitHub API liefert einen Fehler ──────────────────────────────────────────────────────────────────
def _watchdog_env(tmp_path, monkeypatch, now: datetime):
    import scripts.workflow_watchdog as wd
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir(exist_ok=True)
    (tmp_path / "config/workflow_schedule.yaml").write_text(
        (REPO / "config/workflow_schedule.yaml").read_text(encoding="utf-8"))
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    monkeypatch.setenv("GITHUB_TOKEN", "t")
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setattr(wd, "artifact_ok", lambda spec, day: False)

    class _DT(datetime):
        @classmethod
        def now(cls, tz=None):
            return now
    monkeypatch.setattr(wd, "datetime", _DT)
    return wd, summary


def test_api_fehler_beim_lesen_nie_blind_dispatchen(tmp_path, monkeypatch, capsys):
    wd, summary = _watchdog_env(tmp_path, monkeypatch, utc("2026-10-19T18:00"))

    def boom(*a, **k):
        raise urllib.error.URLError("503")
    monkeypatch.setattr(wd, "_api", boom)
    wd.main([])
    out = capsys.readouterr().out
    st = json.loads((tmp_path / "outputs/state/workflow_status.json").read_text())["days"]["2026-10-19"]
    assert st["scanner"]["state"] == st["source_health"]["state"] == "UNKNOWN"
    assert all(r["action"] is None for r in st.values())
    assert "::warning::UNKNOWN source_health" in out and "UNKNOWN" in summary.read_text()


def test_api_fehler_beim_dispatch_kein_zweiter_versuch(tmp_path, monkeypatch, capsys):
    wd, summary = _watchdog_env(tmp_path, monkeypatch, utc("2026-10-19T16:00"))
    monkeypatch.setattr(wd, "list_runs", lambda repo, f, day: [])
    calls = []

    def fail(method, path, body=None):
        calls.append(path)
        raise urllib.error.HTTPError(path, 500, "boom", None, None)
    monkeypatch.setattr(wd, "_api", fail)
    wd.main([])
    wd.main([])
    st = json.loads((tmp_path / "outputs/state/workflow_status.json").read_text())
    assert [c for c in calls if "source_health" in c] == ["/repos/o/r/actions/workflows/source_health.yml/dispatches"]
    assert st["recoveries"]["2026-10-19"]["source_health"] == 1
    assert st["days"]["2026-10-19"]["source_health"]["state"] == "RECOVERY_EXHAUSTED"
    assert "::warning::RECOVERY_EXHAUSTED source_health" in capsys.readouterr().out


# ── 9. Wochenende / NYSE-Feiertag / verkürzte Sitzung ───────────────────────────────────────────────────
@pytest.mark.parametrize("now, guard_status", [("2026-10-24T17:00", "SKIP_NOT_TRADING_DAY"),      # Samstag
                                               ("2026-11-26T17:00", "SKIP_MARKET_CLOSED")])     # Thanksgiving
def test_wochenende_und_feiertag_kein_scanner_keine_recovery(tmp_path, now, guard_status):
    assert guard(tmp_path, now, snap=now + ":00+00:00")["status"] == guard_status
    p = plan(now, {"source_health": [run(now[:10] + "T16:00:00Z")]}, {"source_health": True, "scanner": False})
    assert p["scanner"]["state"] == "NOT_SCHEDULED" and dispatches(p) == []


def test_verkuerzte_sitzung_recovery_rechtzeitig_vor_13_uhr(tmp_path):
    """27.11.2026: Schluss 13:00 ET = 18:00 UTC -> spätester Start 17:00 UTC, Recovery-Frist 16:00 UTC."""
    sh = {"source_health": [run("2026-11-27T14:00:00Z")]}
    p = plan("2026-11-27T16:05", sh, {"source_health": True, "scanner": False})
    assert p["scanner"]["deadline"] == "2026-11-27T16:00+00:00" and dispatches(p) == ["scanner"]
    assert guard(tmp_path, "2026-11-27T16:06", snap="2026-11-27T14:00:00+00:00")["status"] == "RUN"
    assert guard(tmp_path, "2026-11-27T17:01", snap="2026-11-27T14:00:00+00:00")["status"] == "MISSED_WINDOW_CLOSED"


# ── 10. Recovery-Fenster bereits überschritten ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("day, cutoff_utc", [("2026-10-19", "19:00"),     # Sommer: 15:00 ET
                                             ("2026-11-04", "20:00")])    # Winter: 15:00 ET
def test_fenster_ueberschritten_missed_ohne_dispatch_mit_warnung(tmp_path, day, cutoff_utc):
    sh = {"source_health": [run(day + "T14:00:00Z")]}
    cut = utc(f"{day}T{cutoff_utc}")
    vorher = plan((cut - timedelta(minutes=1)).isoformat()[:16], sh, {"source_health": True, "scanner": False})
    assert dispatches(vorher) == ["scanner"]
    p = plan(cut.isoformat()[:16], sh, {"source_health": True, "scanner": False})
    assert p["scanner"]["state"] == "MISSED" and p["scanner"]["window_closed"] is True and dispatches(p) == []
    import scripts.workflow_watchdog as wd
    assert any(m.startswith("MISSED scanner: spätester Start") for m in wd._alerts(p))
    assert guard(tmp_path, cut.isoformat()[:16], snap=day + "T14:00:00+00:00")["status"] == "MISSED_WINDOW_CLOSED"


@pytest.mark.parametrize("day", ["2026-10-19", "2026-11-04", "2026-11-27", "2026-12-24", "2027-03-15"])
def test_recovery_frist_liegt_immer_vor_dem_spaetesten_start(day):
    d = datetime.fromisoformat(day).date()
    due, cut = wh.deadline(CFG["workflows"]["scanner"], CFG["gates"]["scanner"], d)
    sh_due, _ = wh.deadline(CFG["workflows"]["source_health"], {}, d)
    open_utc = wh._at(d, "13:30") if wh._ny(wh._at(d, "13:30")).hour == 9 else wh._at(d, "14:30")
    assert open_utc < sh_due < due <= cut - wh.RECOVERY_LEAD


# ── Watchdog-Auslösung und keine Retry-Schleifen ────────────────────────────────────────────────────────
def test_watchdog_redundant_ausgeloest_ohne_retry_kreis():
    wd = yaml.safe_load((WF_DIR / "watchdog.yml").read_text(encoding="utf-8"))
    on = wd[True]
    assert len(on["schedule"]) >= 2 and "workflow_dispatch" in on
    names = {f.name: yaml.safe_load(f.read_text(encoding="utf-8"))["name"] for f in WF_DIR.glob("*.yml")}
    triggers = on["workflow_run"]["workflows"]
    assert set(triggers) <= set(names.values())                       # exakte Namen, sonst feuert nichts
    assert names["options_capture.yml"] in triggers                   # 6 Crons je Werktag im Handelsfenster
    # Was der Watchdog täglich mehrfach anstoßen könnte, darf ihn nicht wieder auslösen
    for job in ("source_health", "scanner", "feedback"):
        assert names[CFG["workflows"][job]["file"]] not in triggers
    assert names["watchdog.yml"] not in triggers
    assert CFG["max_recoveries_per_day"] == 1


def test_watchdog_zweimal_ausgeloest_dispatcht_einmal(tmp_path, monkeypatch, capsys):
    wd, summary = _watchdog_env(tmp_path, monkeypatch, utc("2026-10-19T18:29"))
    monkeypatch.setattr(wd, "list_runs", lambda repo, f, day: [])
    calls = []
    monkeypatch.setattr(wd, "dispatch", lambda repo, f, ref: calls.append(f) or True)
    wd.main([])
    wd.main([])
    assert sorted(calls) == ["external_data.yml", "source_health.yml"]  # je einmal; Scanner wartet auf den Upstream
    text = summary.read_text()
    assert "Workflow-Watchdog 2026-10-19" in text and "RECOVERY_TRIGGERED" in text
    out = capsys.readouterr().out
    assert "::warning::MISSED source_health" in out and "::warning::MISSED scanner: Pflicht-Upstream" in out


def test_recovered_source_health_dispatches_scanner_directly_but_guarded():
    """GitHub-token-initiated Source Health completion may not emit workflow_run.

    Use explicit workflow_dispatch, NEVER the 'force' override, while preserving
    all source-health snapshot and market-day guards.
    """
    source = yaml.safe_load((WF_DIR / "source_health.yml").read_text(encoding="utf-8"))
    scanner = yaml.safe_load((WF_DIR / "scanner.yml").read_text(encoding="utf-8"))
    assert source["permissions"]["actions"] == "write"
    assert source["permissions"]["contents"] == "write"
    steps = source["jobs"]["check"]["steps"]
    downstream = next(x for x in steps if "Direct scanner dispatch" in x["name"])
    assert "health_check.outcome == 'success'" in downstream["if"]
    assert "snapshot_commit.outcome == 'success'" in downstream["if"]
    assert "gh workflow run scanner.yml" in downstream["run"]
    assert "force" not in downstream["run"].lower()
    assert "date -u +%u" in downstream["run"]
    # Every dispatch, even repeated recovery dispatches, still runs the same guard.
    assert "workflow_dispatch" in scanner["on"]
    assert scanner["jobs"]["scan"]["needs"] == "guard"
    assert "needs.guard.outputs.run == 'true'" in scanner["jobs"]["scan"]["if"]


def test_source_health_failure_never_dispatches_scanner():
    wf = yaml.safe_load((WF_DIR / "source_health.yml").read_text(encoding="utf-8"))
    steps = wf["jobs"]["check"]["steps"]
    assert any(s.get("id") == "health_check" and "source_health check" in s.get("run", "") for s in steps)
    assert any(s.get("id") == "snapshot_commit" and s.get("if") == "always()" for s in steps)
    direct = next(s for s in steps if "Direct scanner dispatch" in s["name"])
    assert "success" in direct["if"] and "snapshot_commit" in direct["if"]

"""Self-Improvement Notifications (modules/research_director.py --notify): technische Verbesserungsaufgaben nur aus
belegten Befunden, feste Vorlagen ohne LLM, Wochen-Digest mit Dedupe, sofortige Warnung bei Widerruf."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from modules import research_director as rd
from modules import research_memory as rm
from modules.atomic_io import read_jsonl

NOW = datetime(2026, 10, 31, 9, tzinfo=timezone.utc)
ROOT = Path(__file__).resolve().parent.parent


def _gap(name, h=5, **kw):
    return {"gap_id": f"{kw.pop('kind', 'data')}:{name}:{h}d", "kind": kw.pop("gkind", "data"), "horizon": h, **kw}


def _empty_feature(cov=0.0, h=5):
    return _gap("sparse_feature:news_novelty", h, value=cov, threshold=0.5, severity=1 - cov, feature="news_novelty",
                finding=f"Merkmal news_novelty nur in {cov:.0%} der 359 geprüften Labels befüllt")


def _ledger(n=40, as_dict=True):
    """Ledger wie im echten Bestand: news_novelty unter features als Objekt statt Zahl."""
    rows = []
    for i in range(n):
        v = ({"novelty_max": 0.4 if i % 4 else None, "novelty_mean": 0.3 if i % 4 else None, "n_today": 5,
              "n_history": i, "novelty_version": "news-novelty-v1", "novelty_status": "ok" if i % 4 else "no_history"}
             if as_dict else 0.4)
        rows.append({"date": "2026-10-20", "ticker": f"T{i}", "features": {"news_novelty": v, "impact": 0.1}})
    return rows


def _inputs(gaps=(), status=None, ledger=None, **kw):
    return {"improvement": {"generated": "2026-10-31T09:00:00+00:00", "gaps": list(gaps)},
            "status": status if status is not None else {5: {"as_of": "2026-10-31", "verified_labels": 359}},
            "ledger": ledger, "source_history": kw.get("history", []), "source_snapshot": kw.get("snapshot", {}),
            "workflow_status": kw.get("workflow", {})}


class Outbox:
    """Ersetzt mailer.send_mail: zählt Mails, liefert den vorgegebenen Status."""

    def __init__(self, status="sent"):
        self.status, self.mails = status, []

    def __call__(self, subject, html, text):
        self.mails.append({"subject": subject, "html": html, "text": text})
        return {"status": self.status, "error": None if self.status == "sent" else "SMTPException: ***"}


def _notify(tmp_path, tasks, now=NOW, send=None, **kw):
    send = send if send is not None else Outbox()
    rep = rd.notify_tasks(tasks, now, path=tmp_path / "tasks.jsonl", state_path=tmp_path / "tasks.json", send=send,
                          decisions=kw.pop("decisions", {}), **kw)
    return rep, send


def _variant(status="PROPOSE_ADOPTION", state="VALID", contradicts=False, gap_ids=("model:overfit_risk:5d",), **kw):
    return {"variant_id": kw.get("vid", "OLV-abc"), "overrides": {"ridge_alpha": 30.0}, "status": status,
            "look": kw.get("look", 1), "looks": 3, "relative_verdict": kw.get("relative", "BETTER_THAN_BASE"),
            "economic_verdict": kw.get("economic", "POSITIVE_EXECUTABLE_ALPHA"), "gap_ids": list(gap_ids),
            "closed_at": kw.get("closed_at"), "look_report": {"report_id": "OLV-abc_look1_v1"},
            "comparison": {"coverage": kw.get("coverage", 0.95), "excluded": kw.get("excluded", {})},
            "vs_base": {"learning_minus_reference": {"mean": 0.012, "ci95": [0.004, 0.02]},
                        "learning_minus_no_trade": {"ci95": [0.002, 0.03]}},
            "current_evidence": {"status": "FALSIFIED_NO_EFFECT" if contradicts else status,
                                 "contradicts_decision": contradicts, "relative_verdict": "NO_SIGNIFICANT_DIFFERENCE",
                                 "economic_verdict": "NOT_PROVEN"},
            "change_proposal": {"state": state} if state else None}


def _status_with(*variants, budget=None, **st):
    return {5: {"as_of": "2026-10-31", "verified_labels": 359,
                "variants": {"variants": {v["variant_id"]: v for v in variants},
                             "test_budget": budget or {"exhausted": False, "used": 1, "max_tests": 20}}, **st}}


# ── Pflichttest mit dem echten, dokumentierten Defekt ────────────────────────────────────────────────────────────

def test_real_documented_defect_yields_a_sensible_copy_ready_email(tmp_path, monkeypatch):
    """Wie im echten Bestand (31.10.2026): news_novelty in 0 % der 359 geprüften Labels, im Ledger aber als Objekt
    {novelty_max, …} statt als Zahl gespeichert -> belegte P1-Aufgabe mit kopierfertiger Claude-Code-Aufgabe."""
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    tasks, issues = rd.improvement_tasks(_inputs([_empty_feature()], ledger=_ledger()), NOW)
    assert issues == [] and [t["kind"] for t in tasks] == ["empty_feature"]
    t = tasks[0]
    assert t["priority"] == "P1" and t["task_id"].startswith("IMP-") and t["data_as_of"] == "2026-10-31"
    assert "als dict statt als Zahl" in t["cause"]
    assert any("{'dict': 40}" in e and "'novelty_max': 30" in e for e in t["evidence"])
    assert "modules/option_learner.py (LEDGER_FEATURES, features)" in t["files"]
    assert any(f.startswith(("pipeline.py:", "modules/news_novelty.py")) for f in t["files"])
    rep, out = _notify(tmp_path, tasks)
    assert rep["mail_status"] == "sent" and len(out.mails) == 1
    mail = out.mails[0]
    assert mail["subject"] == ("[Scanner] Technischer Verbesserungsvorschlag – Merkmal news_novelty im "
                               "Options-Lernmodell leer")
    for section in ("1. Was wurde erkannt?", "2. Warum ist das problematisch?", "3. Wie schwerwiegend ist das Problem?",
                    "4. Welche Verbesserung wird vorgeschlagen?", "5. Welche Daten belegen das?",
                    "6. Entstehen zusätzliche Infrastrukturkosten?", "7. Aufgabe für Claude Code (zum Kopieren)"):
        assert section in mail["text"]
    task = mail["text"].split("----- BEGIN CLAUDE CODE TASK -----")[1].split("----- END CLAUDE CODE TASK -----")[0]
    assert "Repository: owner/repo" in task and "HYPOTHESE, noch nicht bestätigt" in task
    assert "1. Ursache zuerst verifizieren" in task and "Regressionstests ergänzen" in task
    assert "KEIN automatischer Merge" in task and "Positionsgrößen, Risikolimits" in task
    assert "LLM-/Claude-Aufrufe: 0" in mail["text"] and "kostenpflichtige Daten: nein" in mail["text"]
    assert "Renditeversprechen" in mail["text"]                       # kein erfundener Ertrag
    assert "&lt;" not in mail["text"] and mail["html"].startswith("<pre")
    state = json.loads((tmp_path / "tasks.json").read_text())
    assert state["tasks"][t["task_id"]]["status"] == "NOTIFIED"
    assert state["tasks"][t["task_id"]]["notifications"][0]["channel"] == "digest"


# ── Pflichttests der Benachrichtigungsregeln ─────────────────────────────────────────────────────────────────────

def test_no_improvement_need_sends_no_email(tmp_path):
    """NEED_MORE_DATA, normale Wartezeit, reine Heuristiken und laufende Varianten erzeugen keine Aufgabe."""
    gaps = [_gap("insufficient", kind="evidence", gkind="evidence", value="NEED_MORE_DATA",
                 finding="NEED_MORE_DATA – walk_forward: 0/20 Testtage"),
            _gap("overfit_risk", kind="model", gkind="model", value=11.0, finding="33 Merkmale auf 359 Labels"),
            _gap("untested_alternatives", kind="model", gkind="model", value=0, finding="noch keine Variante"),
            _gap("selection_no_value", kind="model", gkind="model", value=-0.01, finding="Punktschätzung"),
            _gap("representativeness:abc", kind="representativeness", gkind="representativeness",
                 finding="Walk-Forward-Evidenz: nur 8 Kalendertage: höchstens eine Marktphase")]
    running = _variant(status="RUNNING", state=None, look=0, relative="AWAITING_LOOK", economic="AWAITING_LOOK")
    tasks, _issues = rd.improvement_tasks(_inputs(gaps, status=_status_with(running)), NOW)
    assert tasks == []
    rep, out = _notify(tmp_path, tasks)
    assert rep["mail_status"] == "nothing_to_send" and out.mails == []
    assert not (tmp_path / "tasks.jsonl").exists() or read_jsonl(tmp_path / "tasks.jsonl") == []


def test_known_unchanged_task_is_never_sent_twice(tmp_path):
    tasks, _ = rd.improvement_tasks(_inputs([_empty_feature()], ledger=_ledger()), NOW)
    _rep, out = _notify(tmp_path, tasks)
    for later in (NOW + timedelta(days=1), NOW + timedelta(days=7), NOW + timedelta(days=30)):
        again, _ = rd.improvement_tasks(_inputs([_empty_feature()], ledger=_ledger(60)), later)  # mehr Zeilen
        rep, out2 = _notify(tmp_path, again, now=later)
        assert rep["mail_status"] == "nothing_to_send" and out2.mails == []
    events = read_jsonl(tmp_path / "tasks.jsonl")
    assert [e["event"] for e in events] == ["detected", "notified"]       # idempotent, keine Wiederholung
    assert len(out.mails) == 1


def test_substantially_changed_finding_sends_an_update(tmp_path):
    first, _ = rd.improvement_tasks(_inputs([_empty_feature(0.0)], ledger=_ledger()), NOW)
    _notify(tmp_path, first)
    later = NOW + timedelta(days=7)
    changed, _ = rd.improvement_tasks(_inputs([_empty_feature(0.3)], ledger=_ledger()), later)  # 0 % -> 30 %
    assert changed[0]["task_id"] == first[0]["task_id"] and changed[0]["fingerprint"] != first[0]["fingerprint"]
    rep, out = _notify(tmp_path, changed, now=later)
    assert rep["mail_status"] == "sent" and len(out.mails) == 1
    assert "Befund wesentlich geändert" in out.mails[0]["text"] and "kaum befüllt" in out.mails[0]["subject"]
    assert changed[0]["priority"] == "P2"                                 # nicht mehr leer
    assert [e["event"] for e in read_jsonl(tmp_path / "tasks.jsonl")] == ["detected", "notified", "changed",
                                                                          "notified"]


def test_missing_mail_configuration_is_a_visible_error_and_never_counts_as_sent(tmp_path, monkeypatch):
    for k in ("GMAIL_SENDER", "GMAIL_APP_PW", "NOTIFY_EMAIL", "MAIL_TO", "SMTP_HOST", "MAIL_PROVIDER"):
        monkeypatch.delenv(k, raising=False)
    from modules import mailer
    tasks, _ = rd.improvement_tasks(_inputs([_empty_feature()], ledger=_ledger()), NOW)
    rep, _ = _notify(tmp_path, tasks, send=mailer.send_mail)
    assert rep["mail_status"] == "not_configured" and rep["mails"][0]["status"] == "not_configured"
    state = json.loads((tmp_path / "tasks.json").read_text())
    tid = tasks[0]["task_id"]
    assert state["last_run"]["mail_status"] == "not_configured"
    assert state["tasks"][tid]["status"] == "OPEN" and state["tasks"][tid]["notifications"] == []
    assert state["tasks"][tid]["send_failures"] == 1
    rep2, out = _notify(tmp_path, tasks, now=NOW + timedelta(hours=1))   # Konfiguration repariert: erneuter Versuch
    assert rep2["mail_status"] == "sent" and "Versand stand noch aus" in out.mails[0]["text"]
    failed, out3 = _notify(tmp_path / "f", tasks, send=Outbox("failed"))
    assert failed["mail_status"] == "failed" and len(out3.mails) == 1
    assert json.loads((tmp_path / "f" / "tasks.json").read_text())["tasks"][tid]["notifications"] == []

    def boom(subject, html, text):
        raise ConnectionError("smtp down")
    crashed, _ = _notify(tmp_path / "c", tasks, send=boom)
    assert crashed["mail_status"] == "failed" and crashed["mails"][0]["error"] == "ConnectionError"
    assert json.loads((tmp_path / "c" / "tasks.json").read_text())["tasks"][tid]["notifications"] == []


def test_invalid_or_incomplete_data_never_invents_a_task():
    broken = [_gap("sparse_feature:x", value=None, feature="x"), _gap("sparse_feature:y", value=1.7, feature="y"),
              _gap("sparse_feature:z", value=0.0),                       # ohne Merkmalsname
              _gap("outcome_coverage:forward", value="viel", reasons={"EXIT_NOT_ARCHIVED": 3}),
              {"gap_id": None, "finding": "kaputt"}, "kein dict"]
    bad_variant = {"variant_id": "OLV-x", "status": "PROPOSE_ADOPTION"}           # ohne Änderung/Evidenz
    status = {5: {"as_of": "2026-10-31", "variants": {"variants": {"OLV-x": bad_variant, "OLV-y": "kaputt"}}},
              20: None, 99: "kaputt", "abc": {"as_of": "x"}}
    snapshot = {"sources": {"q": {"status": "BROKEN"}, "r": {"status": "BROKEN", "criticality": "IMPORTANT"}}}
    history = [{"source_id": "r", "status": "BROKEN"}, "kaputt", {"source_id": "r", "checked_at": "kein Datum",
                                                                  "status": "BROKEN"}]
    workflow = {"recoveries": {"2026-10-30": "kaputt", "kein datum": {"x": 1}}}
    tasks, issues = rd.improvement_tasks(_inputs(broken, status=status, history=history, snapshot=snapshot,
                                                 workflow=workflow), NOW)
    assert tasks == []
    assert issues and all("ignoriert" in i for i in issues)
    assert rd.improvement_tasks({}, NOW) == ([], [])


def test_corrected_evidence_that_refutes_an_earlier_success_sends_an_immediate_warning(tmp_path):
    adopted = _variant()
    tasks, _ = rd.improvement_tasks(_inputs(status=_status_with(adopted)), NOW)
    assert [(t["kind"], t["priority"]) for t in tasks] == [("adoption", "P1")]
    _notify(tmp_path, tasks)
    later = NOW + timedelta(days=1)                                       # gleiche Woche: Digest-Kontingent
    refuted = _variant(status="CONFIRMED_ADOPTION", state="REVIEW_REQUIRED", contradicts=True,
                       closed_at="2026-10-30T09:00:00+00:00")
    tasks2, _ = rd.improvement_tasks(_inputs(status=_status_with(refuted)), later)
    assert [(t["kind"], t["priority"], t["category"]) for t in tasks2] == [("withdrawn_success", "P0", "critical")]
    rep, out = _notify(tmp_path, tasks2, now=later)
    assert rep["mail_status"] == "sent" and len(out.mails) == 1
    assert out.mails[0]["subject"].startswith("[Scanner] Technischer Verbesserungsvorschlag – WARNUNG: belegte "
                                              "Verbesserung ridge_alpha=30.0 widerrufen")
    assert "REVIEW_REQUIRED" in out.mails[0]["text"] and "Rücknahme" in out.mails[0]["text"]
    channels = [e.get("channel") for e in read_jsonl(tmp_path / "tasks.jsonl") if e["event"] == "notified"]
    assert channels == ["digest", "critical"]
    events = {e["event"] for e in read_jsonl(tmp_path / "tasks.jsonl")}
    assert "resolved" in events                                           # Übernahme-Aufgabe ist erledigt


# ── weitere Regeln ───────────────────────────────────────────────────────────────────────────────────────────────

def _five_tasks():
    gaps = [_empty_feature()] + [_gap(f"sparse_feature:f{i}", value=0.1 * i, feature=f"f{i}") for i in range(1, 5)]
    return rd.improvement_tasks(_inputs(gaps), NOW)[0]


def test_weekly_digest_holds_at_most_three_tasks_and_defers_the_rest(tmp_path):
    tasks = _five_tasks()
    assert len(tasks) == 5
    rep, out = _notify(tmp_path, tasks)
    assert len(out.mails) == 1 and "(+2 weitere)" in out.mails[0]["subject"]
    assert out.mails[0]["text"].count("----- BEGIN CLAUDE CODE TASK -----") == 3
    assert len(rep["deferred_by_weekly_limit"]) == 2
    assert tasks[0]["task_id"] in out.mails[0]["text"]                    # P1 zuerst
    rep2, out2 = _notify(tmp_path, tasks, now=NOW + timedelta(days=1))    # gleiche Kalenderwoche: Limit erreicht
    assert out2.mails == [] and len(rep2["deferred_by_weekly_limit"]) == 2
    rep3, out3 = _notify(tmp_path, tasks, now=NOW + timedelta(days=7))    # nächste Woche: der Rest
    assert len(out3.mails) == 1 and out3.mails[0]["text"].count("BEGIN CLAUDE CODE TASK") == 2
    assert rep3["deferred_by_weekly_limit"] == []


def test_dry_run_sends_nothing_and_stores_nothing(tmp_path):
    tasks = _five_tasks()
    rep, out = _notify(tmp_path, tasks, dry_run=True)
    assert out.mails == [] and rep["mail_status"] == "dry_run" and rep["mails"][0]["text"]
    assert not (tmp_path / "tasks.jsonl").exists() and not (tmp_path / "tasks.json").exists()


def test_human_decision_stops_further_mails(tmp_path):
    tasks, _ = rd.improvement_tasks(_inputs([_empty_feature()], ledger=_ledger()), NOW)
    tid = tasks[0]["task_id"]
    rep, out = _notify(tmp_path, tasks, decisions={tid: {"status": "rejected", "note": "bewusst"}})
    assert out.mails == [] and rep["mail_status"] == "nothing_to_send"
    state = json.loads((tmp_path / "tasks.json").read_text())["tasks"][tid]
    assert state["status"] == "REJECTED" and state["decision_note"] == "bewusst"
    cfg = yaml.safe_load((ROOT / "config" / "improvement_task_decisions.yaml").read_text(encoding="utf-8"))
    assert cfg == {"tasks": {}}


def test_resolved_task_is_silent_and_a_reappearance_is_reported(tmp_path):
    tasks, _ = rd.improvement_tasks(_inputs([_empty_feature()], ledger=_ledger()), NOW)
    _notify(tmp_path, tasks)
    _rep, out = _notify(tmp_path, [], now=NOW + timedelta(days=7))        # Befund behoben
    state = json.loads((tmp_path / "tasks.json").read_text())
    assert out.mails == [] and state["tasks"][tasks[0]["task_id"]]["status"] == "RESOLVED"
    _rep2, out2 = _notify(tmp_path, tasks, now=NOW + timedelta(days=14))  # wieder da
    assert len(out2.mails) == 1 and "erneut aufgetreten" in out2.mails[0]["text"]


def test_no_credentials_reach_mail_state_or_report(tmp_path, monkeypatch):
    monkeypatch.setenv("GMAIL_APP_PW", "geheimes-passwort-123")
    monkeypatch.setenv("GMAIL_SENDER", "absender@example.com")
    monkeypatch.setenv("NOTIFY_EMAIL", "empfaenger@example.com")
    tasks, _ = rd.improvement_tasks(_inputs([_empty_feature()], ledger=_ledger()), NOW)
    rep, out = _notify(tmp_path, tasks, send=Outbox("failed"))
    blob = json.dumps(rep) + json.dumps(out.mails) + (tmp_path / "tasks.json").read_text() + \
        (tmp_path / "tasks.jsonl").read_text()
    for secret in ("geheimes-passwort-123", "absender@example.com", "empfaenger@example.com"):
        assert secret not in blob


def test_running_automatic_remedy_is_normal_waiting_but_an_exhausted_one_is_a_task():
    gap = _gap("no_skill", kind="model", gkind="model", value="NOT_VALIDATED", finding="Walk-Forward NOT_VALIDATED")
    running = _variant(status="RUNNING", state=None, gap_ids=("model:no_skill:5d",), look=1,
                       relative="NO_SIGNIFICANT_DIFFERENCE")
    assert rd.improvement_tasks(_inputs([gap], status=_status_with(running)), NOW)[0] == []
    assert rd.improvement_tasks(_inputs([gap], status=_status_with()), NOW)[0] == []     # noch nicht registriert
    falsified = _variant(status="FALSIFIED_NO_EFFECT", state=None, gap_ids=("model:no_skill:5d",), look=3,
                         closed_at="2026-10-20", relative="NO_SIGNIFICANT_DIFFERENCE")
    tasks, _ = rd.improvement_tasks(_inputs([gap], status=_status_with(falsified)), NOW)
    assert [(t["kind"], t["priority"]) for t in tasks] == [("proven_weakness", "P2")]
    assert "OLV-abc FALSIFIED_NO_EFFECT" in tasks[0]["finding"]


def test_untestable_variant_budget_and_release_findings():
    untestable = _variant(status="RUNNING", state=None, relative="INSUFFICIENT_COVERAGE", look=1, coverage=0.4,
                          excluded={"NO_VARIANT_DECISION": 30})
    seq = {"status": "NOT_CONFIRMED", "plan_id": "FWD-5d-x", "binding": {"look": 3, "verdict": "INCONCLUSIVE",
                                                                        "selected_ci95": [-0.01, 0.05]}}
    status = _status_with(untestable, budget={"exhausted": True, "used": 20, "max_tests": 20, "total_alpha": 0.05},
                          forward_sequential=seq)
    kinds = {t["kind"]: t["priority"] for t in rd.improvement_tasks(_inputs(status=status), NOW)[0]}
    assert kinds == {"variant_untestable": "P2", "budget_exhausted": "P3", "base_not_confirmed": "P2"}
    broken = _status_with(forward_sequential={"status": "PLAN_INVALID", "plan_id": "FWD-5d-x"})
    assert [(t["kind"], t["priority"]) for t in rd.improvement_tasks(_inputs(status=broken), NOW)[0]] == [
        ("release_integrity", "P0")]
    missing = _status_with(forward_sequential={"status": "RUNNING", "binding": {
        "look": 1, "verdict": "INSUFFICIENT_COVERAGE", "decision_coverage": 0.5, "excluded": {"NO_DECISION": 9}}})
    assert [t["kind"] for t in rd.improvement_tasks(_inputs(status=missing), NOW)[0]] == ["decision_capture"]
    waiting = _status_with(forward_sequential={"status": "AWAITING_LOOK", "binding": None})
    assert rd.improvement_tasks(_inputs(status=waiting), NOW)[0] == []


def test_persistent_infrastructure_problems_only():
    snap = {"sources": {"port": {"status": "BROKEN", "criticality": "IMPORTANT", "last_success": "2026-10-20"},
                        "minor": {"status": "BROKEN", "criticality": "NON_CRITICAL"},
                        "fresh": {"status": "BROKEN", "criticality": "CRITICAL"}}}
    hist = ([{"source_id": "port", "checked_at": f"2026-10-{d:02d}T18:00:00+00:00", "status": "HEALTHY"}
             for d in (18, 19)]
            + [{"source_id": s, "checked_at": f"2026-10-{d:02d}T18:00:00+00:00", "status": "BROKEN",
                "error": "HTTP 404"} for s in ("port", "minor") for d in range(20, 31)]
            + [{"source_id": "fresh", "checked_at": "2026-10-29T18:00:00+00:00", "status": "BROKEN"}])
    rec = {f"2026-10-{d:02d}": {"source_health": 1} for d in range(20, 31)} | {"2026-10-28": {"source_health": 1,
                                                                                            "scanner": 1}}
    days = {"2026-10-30": {"source_health": {"file": "source_health.yml", "expected": "2026-10-30T12:41+00:00",
                                             "actual_last_run": "2026-10-30T18:10:19Z", "state": "OK"}}}
    tasks, _ = rd.improvement_tasks(_inputs(history=hist, snapshot=snap, workflow={"recoveries": rec, "days": days}),
                                    NOW)
    by = {(t["kind"], t["subject"]): t for t in tasks}
    # nicht: minor (unkritisch), fresh (zu kurz), scanner (nur ein Nachstart)
    assert set(by) == {("source_failure", "port"), ("workflow_recovery", "source_health")}
    assert "seit 2026-10-20 (10 Tage, 11 Prüfungen)" in by[("source_failure", "port")]["finding"]
    wf = by[("workflow_recovery", "source_health")]
    assert "an 6 der letzten 7 Tage" in wf["finding"] and ".github/workflows/source_health.yml" in wf["files"]
    assert any("letzter Lauf 2026-10-30T18:10:19Z" in e for e in wf["evidence"])


def test_research_memory_and_meta_cognition_show_the_tasks(tmp_path):
    tasks, _ = rd.improvement_tasks(_inputs([_empty_feature()], ledger=_ledger()), NOW)
    _notify(tmp_path, tasks)
    state = json.loads((tmp_path / "tasks.json").read_text())
    entries = rm.task_entries(state)
    assert [(e["kind"], e["status"], e["hypothesis_id"]) for e in entries] == [("dev_task", "NOTIFIED",
                                                                                tasks[0]["task_id"])]
    assert entries[0]["evidence"]["fingerprint"] == tasks[0]["fingerprint"]
    assert rm.SOURCES["improvement_tasks"] == Path("outputs/research/improvement_tasks.json")
    from modules import meta_cognition as mc
    st = mc.machine_state({"improvement_tasks": state}, {}, {}, {}, {"active": False, "reasons": []})
    assert st["open_improvement_tasks"] == [f"[P1] {tasks[0]['task_id']} {tasks[0]['short']} (NOTIFIED)"]
    assert st["improvement_mail_status"] == "sent"
    assert "Offene technische Verbesserungsaufgaben" in mc.render_md(st)


def test_cli_returns_error_status_when_mail_is_not_configured(tmp_path, monkeypatch):
    for k in ("GMAIL_SENDER", "GMAIL_APP_PW", "NOTIFY_EMAIL", "MAIL_TO", "SMTP_HOST", "MAIL_PROVIDER"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(rd, "TASKS", tmp_path / "tasks.jsonl")
    monkeypatch.setattr(rd, "TASKS_STATE", tmp_path / "tasks.json")
    monkeypatch.setattr(rd, "TASK_DECISIONS", tmp_path / "none.yaml")
    monkeypatch.setattr(rd, "load_task_inputs", lambda paths=None, now=None: _inputs([_empty_feature()],
                                                                                     ledger=_ledger()))
    monkeypatch.setattr(rm, "sync", lambda *a, **k: 0)
    assert rd.main(["--notify", "--dry-run"]) == 0 and not (tmp_path / "tasks.jsonl").exists()
    assert rd.main(["--notify"]) == 2                                     # sichtbar rot (DEGRADED im Workflow)
    assert json.loads((tmp_path / "tasks.json").read_text())["last_run"]["mail_status"] == "not_configured"


def test_workflows_wire_the_weekly_run_and_a_manual_dry_run_trigger():
    ml = yaml.safe_load((ROOT / ".github/workflows/ml_research.yml").read_text(encoding="utf-8"))
    steps = ml["jobs"]["ml"]["steps"]
    ids = [s.get("id") for s in steps]
    assert ids.index("improvement_notify") == ids.index("research_improvement") + 1
    step = steps[ids.index("improvement_notify")]
    assert step["run"] == "python -m modules.research_director --notify" and step["continue-on-error"] is True
    assert set(step["env"]) == {"GMAIL_SENDER", "GMAIL_APP_PW", "NOTIFY_EMAIL"}
    assert all(v.startswith("${{ secrets.") for v in step["env"].values())
    check = next(s for s in steps if s.get("name", "").startswith("Stufen-Bilanz"))
    assert "improvement_notify=${{ steps.improvement_notify.outcome }}" in check["env"]["DEGRADED"]
    manual = yaml.safe_load((ROOT / ".github/workflows/improvement_notify.yml").read_text(encoding="utf-8"))
    trigger = manual[True]["workflow_dispatch"]["inputs"]["dry_run"]        # YAML: "on" -> True
    assert trigger["default"] == "true" and manual["concurrency"]["group"] == "research-write"
    run = manual["jobs"]["notify"]["steps"][3]["run"]
    assert "--notify --dry-run" in run and "${{" not in run               # Eingabe nur über env (keine Injektion)
    src = (ROOT / "modules" / "research_director.py").read_text(encoding="utf-8")
    assert "anthropic" not in src.lower() and "openai" not in src.lower()   # kein LLM-Aufruf


@pytest.mark.parametrize("missing", ["improvement", "status", "ledger"])
def test_missing_inputs_are_tolerated(missing):
    inp = _inputs([_empty_feature()], ledger=_ledger())
    inp[missing] = None
    tasks, _ = rd.improvement_tasks(inp, NOW)
    assert all(t["kind"] == "empty_feature" for t in tasks)
    if missing == "improvement":
        assert tasks == []

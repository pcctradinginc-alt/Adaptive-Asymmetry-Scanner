"""Self-Improvement Notifications (modules/research_director.py --notify): technische Verbesserungsaufgaben nur aus
belegten, aktuellen Befunden; feste Vorlagen ohne LLM; täglich nur P0 (--critical), samstags Digest P1–P3;
gemeinsame Historie mit Dedupe; OPEN / RESOLVED (nur mit Nachweis) / UNKNOWN (Daten fehlen oder veraltet)."""
from __future__ import annotations

import json
import shutil
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


def _status(at=NOW, **kw):
    return {5: {"as_of": at.date().isoformat(), "verified_labels": 359, **kw}}


def _inputs(gaps=(), status=None, ledger=None, at=NOW, **kw):
    return {"improvement": {"generated": at.isoformat(timespec="seconds"), "gaps": list(gaps)},
            "status": status if status is not None else _status(at),
            "ledger": ledger, "source_history": kw.get("history", []), "source_snapshot": kw.get("snapshot", {}),
            "workflow_status": kw.get("workflow", {}), "daily_seq": kw.get("daily_seq", {}),
            "problems": kw.get("problems", {})}


class Outbox:
    """Ersetzt mailer.send_mail: zählt Mails, liefert den vorgegebenen Status."""

    def __init__(self, status="sent"):
        self.status, self.mails = status, []

    def __call__(self, subject, html, text):
        self.mails.append({"subject": subject, "html": html, "text": text})
        return {"status": self.status, "error": None if self.status == "sent" else "SMTPException: ***"}


def _notify(tmp_path, tasks, now=NOW, send=None, inputs=None, **kw):
    send = send if send is not None else Outbox()
    resolve = kw.pop("resolve", None) or ((lambda t: rd.resolution(t, inputs, now)) if inputs is not None else None)
    rep = rd.notify_tasks(tasks, now, path=tmp_path / "tasks.jsonl", state_path=tmp_path / "tasks.json", send=send,
                          decisions=kw.pop("decisions", {}), resolve=resolve, **kw)
    return rep, send


def _run(tmp_path, inputs, now=NOW, mode="weekly", send=None, **kw):
    """Ein kompletter Lauf wie im Workflow: Erkennung aus den Eingaben, Auflösung mit denselben Eingaben."""
    tasks, issues = rd.improvement_tasks(inputs, now)
    return _notify(tmp_path, tasks, now=now, send=send, inputs=inputs, mode=mode, issues=issues, **kw)


def _events(tmp_path):
    return [e["event"] for e in read_jsonl(tmp_path / "tasks.jsonl")] if (tmp_path / "tasks.jsonl").exists() else []


def _state(tmp_path):
    return json.loads((tmp_path / "tasks.json").read_text())


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


def _status_with(*variants, budget=None, at=NOW, **st):
    return _status(at, variants={"variants": {v["variant_id"]: v for v in variants},
                                 "test_budget": budget or {"exhausted": False, "used": 1, "max_tests": 20}}, **st)


def _plan_invalid(at=NOW):
    return {20: {"as_of": at.date().isoformat(), "forward_sequential": {"status": "PLAN_INVALID",
                                                                        "plan_id": "FWD-20d-x"}}}


# ── Pflichttest mit dem echten, dokumentierten Defekt ────────────────────────────────────────────────────────────

def test_real_documented_defect_yields_a_sensible_copy_ready_email(tmp_path, monkeypatch):
    """Wie im echten Bestand (31.10.2026): news_novelty in 0 % der 359 geprüften Labels, im Ledger aber als Objekt
    {novelty_max, …} statt als Zahl gespeichert -> belegte P1-Aufgabe mit kopierfertiger Claude-Code-Aufgabe."""
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    tasks, issues = rd.improvement_tasks(_inputs([_empty_feature()], ledger=_ledger()), NOW)
    assert issues == [] and [t["kind"] for t in tasks] == ["empty_feature"]
    t = tasks[0]
    assert t["priority"] == "P1" and t["task_id"].startswith("IMP-") and t["data_as_of"] == "2026-10-31"
    assert t["scope"] == {"feature": "news_novelty", "horizons": [5]}
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
    assert mail["html"].startswith("<pre")
    state = _state(tmp_path)
    assert state["tasks"][t["task_id"]]["status"] == "NOTIFIED"
    assert state["tasks"][t["task_id"]]["notifications"][0]["channel"] == "digest"


# ── tägliche P0-Prüfung und Wochen-Digest ────────────────────────────────────────────────────────────────────────

def test_p0_warning_is_sent_by_the_daily_run(tmp_path):
    inputs = _inputs(status=_plan_invalid())
    rep, out = _run(tmp_path, inputs, mode="critical")
    assert rep["mode"] == "critical" and rep["mail_status"] == "sent" and len(out.mails) == 1
    assert out.mails[0]["subject"] == ("[Scanner] Technischer Verbesserungsvorschlag – WARNUNG: Freigabeprüfung 20 T "
                                       "defekt (PLAN_INVALID)")
    assert [n["channel"] for t in _state(tmp_path)["tasks"].values() for n in t["notifications"]] == ["critical"]
    weekly, out2 = _run(tmp_path / "w", inputs, mode="weekly")             # der Wochenlauf meldet P0 nie
    assert out2.mails == [] and weekly["tasks_detected"] == 0


def test_daily_run_reads_the_latest_release_check_from_the_scanner(tmp_path):
    """Die tägliche Prüfung nutzt den Freigabestand der heutigen Scanner-Empfehlung, nicht nur den Wochenstatus."""
    root = tmp_path / "ol"
    (root / "recommendations").mkdir(parents=True)
    (root / "status_20d.json").write_text(json.dumps({"as_of": "2026-10-24", "forward_sequential": {
        "status": "AWAITING_LOOK"}}))
    (root / "recommendations" / "2026-10-30_20d.json").write_text(json.dumps({
        "date": "2026-10-30", "forward_sequential": {"status": "ERROR", "error": "kaputt"}}))
    paths = {"option_root": root, "source_history": tmp_path / "x.jsonl", "source_snapshot": tmp_path / "x.json",
             "workflow_status": tmp_path / "y.json", "ledger": tmp_path / "none"}
    inputs = rd.load_task_inputs(paths, NOW)
    tasks, _ = rd.improvement_tasks(inputs, NOW)
    assert [(t["kind"], t["subject"]) for t in tasks] == [("release_integrity", "20:ERROR")]


def test_unchanged_p0_finding_is_never_mailed_twice(tmp_path):
    inputs = _inputs(status=_plan_invalid())
    _run(tmp_path, inputs, mode="critical")
    for later in (NOW + timedelta(hours=12), NOW + timedelta(days=1), NOW + timedelta(days=3)):
        rep, out = _run(tmp_path, _inputs(status=_plan_invalid(later), at=later), now=later, mode="critical")
        assert out.mails == [] and rep["mail_status"] == "nothing_to_send"
    assert _events(tmp_path) == ["detected", "notified"]


def test_p1_to_p3_only_in_the_weekly_digest(tmp_path):
    inputs = _inputs([_empty_feature()], status=_status_with(_variant()), ledger=_ledger())
    rep, out = _run(tmp_path, inputs, mode="critical")
    assert out.mails == [] and rep["tasks_detected"] == 0 and _events(tmp_path) == []   # nicht angefasst
    rep, out = _run(tmp_path, inputs, mode="weekly")
    assert len(out.mails) == 1 and out.mails[0]["text"].count("BEGIN CLAUDE CODE TASK") == 2
    assert {n["channel"] for t in _state(tmp_path)["tasks"].values() for n in t["notifications"]} == {"digest"}


def _five_tasks(at=NOW):
    gaps = [_empty_feature()] + [_gap(f"sparse_feature:f{i}", value=0.1 * i, feature=f"f{i}") for i in range(1, 5)]
    return _inputs(gaps, at=at)


def test_weekly_digest_holds_at_most_three_tasks_and_defers_the_rest(tmp_path):
    rep, out = _run(tmp_path, _five_tasks())
    assert len(out.mails) == 1 and "(+2 weitere)" in out.mails[0]["subject"]
    assert out.mails[0]["text"].count("----- BEGIN CLAUDE CODE TASK -----") == 3
    assert len(rep["deferred_by_weekly_limit"]) == 2 and rep["pending"][0] in out.mails[0]["text"]   # P1 zuerst
    later = NOW + timedelta(days=1)                                       # gleiche Kalenderwoche: Limit erreicht
    rep2, out2 = _run(tmp_path, _five_tasks(later), now=later)
    assert out2.mails == [] and len(rep2["deferred_by_weekly_limit"]) == 2
    nxt = NOW + timedelta(days=7)                                         # nächste Woche: der Rest
    rep3, out3 = _run(tmp_path, _five_tasks(nxt), now=nxt)
    assert len(out3.mails) == 1 and out3.mails[0]["text"].count("BEGIN CLAUDE CODE TASK") == 2
    assert rep3["deferred_by_weekly_limit"] == []


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
    waiting = {"status": "AWAITING_LOOK", "plan_id": "FWD-20d-x", "binding": None}
    inputs = _inputs(gaps, status={**_status_with(running), 20: {"as_of": "2026-10-31",
                                                                  "forward_sequential": waiting}})
    assert rd.improvement_tasks(inputs, NOW)[0] == []
    for mode in rd.MODES:
        rep, out = _run(tmp_path, inputs, mode=mode)
        assert rep["mail_status"] == "nothing_to_send" and out.mails == []
    assert _events(tmp_path) == []


def test_known_unchanged_task_is_never_sent_twice(tmp_path):
    _rep, out = _run(tmp_path, _inputs([_empty_feature()], ledger=_ledger()))
    for later in (NOW + timedelta(days=1), NOW + timedelta(days=7), NOW + timedelta(days=30)):
        rep, out2 = _run(tmp_path, _inputs([_empty_feature()], ledger=_ledger(60), at=later), now=later)
        assert rep["mail_status"] == "nothing_to_send" and out2.mails == []
    assert _events(tmp_path) == ["detected", "notified"]                  # idempotent, keine Wiederholung
    assert len(out.mails) == 1


def test_substantially_changed_finding_sends_an_update(tmp_path):
    first, _ = rd.improvement_tasks(_inputs([_empty_feature(0.0)], ledger=_ledger()), NOW)
    _run(tmp_path, _inputs([_empty_feature(0.0)], ledger=_ledger()))
    later = NOW + timedelta(days=7)
    changed_inputs = _inputs([_empty_feature(0.3)], ledger=_ledger(), at=later)            # 0 % -> 30 %
    changed, _ = rd.improvement_tasks(changed_inputs, later)
    assert changed[0]["task_id"] == first[0]["task_id"] and changed[0]["fingerprint"] != first[0]["fingerprint"]
    rep, out = _run(tmp_path, changed_inputs, now=later)
    assert rep["mail_status"] == "sent" and len(out.mails) == 1
    assert "Befund wesentlich geändert" in out.mails[0]["text"] and "kaum befüllt" in out.mails[0]["subject"]
    assert changed[0]["priority"] == "P2"
    assert _events(tmp_path) == ["detected", "notified", "changed", "notified"]


def test_missing_mail_configuration_is_a_visible_error_and_never_counts_as_sent(tmp_path, monkeypatch):
    for k in ("GMAIL_SENDER", "GMAIL_APP_PW", "NOTIFY_EMAIL", "MAIL_TO", "SMTP_HOST", "MAIL_PROVIDER"):
        monkeypatch.delenv(k, raising=False)
    from modules import mailer
    inputs = _inputs([_empty_feature()], ledger=_ledger())
    rep, _ = _run(tmp_path, inputs, send=mailer.send_mail)
    assert rep["mail_status"] == "not_configured" and rep["mails"][0]["status"] == "not_configured"
    state = _state(tmp_path)
    tid = next(iter(state["tasks"]))
    assert state["last_runs"]["weekly"]["mail_status"] == "not_configured"
    assert state["tasks"][tid]["status"] == "OPEN" and state["tasks"][tid]["notifications"] == []
    assert state["tasks"][tid]["send_failures"] == 1


@pytest.mark.parametrize("mode,inputs", [("weekly", _inputs([_empty_feature()], ledger=_ledger())),
                                         ("critical", _inputs(status=_plan_invalid()))], ids=["weekly", "critical"])
def test_smtp_failure_is_retried_on_the_next_run(tmp_path, mode, inputs):
    rep, out = _run(tmp_path, inputs, mode=mode, send=Outbox("failed"))
    assert rep["mail_status"] == "failed" and len(out.mails) == 1
    assert _events(tmp_path) == ["detected", "send_failed"]

    def boom(subject, html, text):
        raise ConnectionError("smtp down")
    crashed, _ = _run(tmp_path, inputs, mode=mode, send=boom, now=NOW + timedelta(hours=1))
    assert crashed["mail_status"] == "failed" and crashed["mails"][0]["error"] == "ConnectionError"
    rep2, out2 = _run(tmp_path, inputs, mode=mode, now=NOW + timedelta(hours=2))   # SMTP wieder erreichbar
    assert rep2["mail_status"] == "sent" and "Versand stand noch aus" in out2.mails[0]["text"]
    assert _events(tmp_path) == ["detected", "send_failed", "send_failed", "notified"]
    _rep3, out3 = _run(tmp_path, inputs, mode=mode, now=NOW + timedelta(hours=3))
    assert out3.mails == []                                               # danach nie wieder


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


def test_stale_inputs_are_not_evaluated():
    old = NOW - timedelta(days=20)
    tasks, issues = rd.improvement_tasks(_inputs([_empty_feature()], ledger=_ledger(), at=old), NOW)
    assert tasks == [] and any("veraltet" in i for i in issues)


# ── OPEN / RESOLVED / UNKNOWN ───────────────────────────────────────────────────────────────────────────────────

def _write_option_root(root: Path, improvement=None, status5=None, corrupt=()):
    root.mkdir(parents=True, exist_ok=True)
    for name, data in (("improvement.json", improvement), ("status_5d.json", status5)):
        path = root / name
        if name in corrupt:
            path.write_text("{kaputt")
        elif data is not None:
            path.write_text(json.dumps(data))
        elif path.exists():
            path.unlink()
    return {"option_root": root, "source_history": root / "none.jsonl", "source_snapshot": root / "none.json",
            "workflow_status": root / "none2.json", "ledger": root / "no_ledger"}


@pytest.mark.parametrize("damage", ["missing", "corrupt", "stale"])
def test_missing_damaged_or_stale_data_gives_unknown_and_keeps_the_finding(tmp_path, damage):
    root = tmp_path / "ol"
    gap = _empty_feature()
    imp, st5 = {"generated": NOW.isoformat(), "gaps": [gap]}, {"as_of": "2026-10-31", "verified_labels": 359,
                                                               "feature_coverage": {"news_novelty": 0.0}}
    inputs = rd.load_task_inputs(_write_option_root(root, imp, st5), NOW)
    rep, _out = _run(tmp_path, inputs)
    tid = rep["pending"][0]
    later = NOW + timedelta(days=12 if damage == "stale" else 1)
    if damage == "missing":
        paths = _write_option_root(root, None, None)
    elif damage == "corrupt":
        paths = _write_option_root(root, imp, st5, corrupt=("improvement.json", "status_5d.json"))
    else:
        paths = _write_option_root(root, imp, st5)                         # nicht erneuert: 12 Tage alt
    damaged = rd.load_task_inputs(paths, later)
    _rep2, out2 = _run(tmp_path, damaged, now=later)
    state = _state(tmp_path)["tasks"][tid]
    assert out2.mails == [] and state["status"] == "UNKNOWN"
    assert state["finding"] and state["notifications"] and state["unknown_reason"]
    expected = {"missing": "fehlt", "corrupt": "beschädigt", "stale": "veraltet"}[damage]
    assert expected in state["unknown_reason"]
    assert _events(tmp_path) == ["detected", "notified", "unknown"]
    _run(tmp_path, damaged, now=later + timedelta(days=1))                # kein wiederholter Statuswechsel
    assert _events(tmp_path) == ["detected", "notified", "unknown"]
    back = rd.load_task_inputs(_write_option_root(root, {**imp, "generated": later.isoformat()},
                                                  {**st5, "as_of": later.date().isoformat()}), later)
    _rep3, out3 = _run(tmp_path, back, now=later)                         # Daten wieder da, Befund unverändert
    assert out3.mails == [] and _state(tmp_path)["tasks"][tid]["status"] == "NOTIFIED"
    assert _events(tmp_path)[-1] == "confirmed"


def test_proven_fix_with_current_data_is_resolved_and_a_reappearance_reported(tmp_path):
    _run(tmp_path, _inputs([_empty_feature()], ledger=_ledger()))
    later = NOW + timedelta(days=7)
    fixed = _inputs([], status=_status(later, feature_coverage={"news_novelty": 0.82}), at=later)   # Lücke weg
    rep, out = _run(tmp_path, fixed, now=later)
    task = next(iter(_state(tmp_path)["tasks"].values()))
    assert out.mails == [] and task["status"] == "RESOLVED" and "Abdeckung" in task["resolved_reason"]
    assert rep["status_changes"][0]["event"] == "resolved"
    nxt = NOW + timedelta(days=14)
    _rep2, out2 = _run(tmp_path, _inputs([_empty_feature()], ledger=_ledger(), at=nxt), now=nxt)
    assert len(out2.mails) == 1 and "erneut aufgetreten" in out2.mails[0]["text"]


def test_gap_gone_without_positive_evidence_is_unknown_not_resolved(tmp_path):
    """Lücke verschwindet, weil zu wenige Labels für die Abdeckungsprüfung vorliegen -> nicht behoben."""
    _run(tmp_path, _inputs([_empty_feature()], ledger=_ledger()))
    later = NOW + timedelta(days=7)
    few = _inputs([], status=_status(later, verified_labels=12, feature_coverage={}), at=later)
    _run(tmp_path, few, now=later)
    task = next(iter(_state(tmp_path)["tasks"].values()))
    assert task["status"] == "UNKNOWN" and "weniger als 50" in task["unknown_reason"]


def test_feature_removed_from_the_specification_counts_as_resolved(tmp_path):
    _run(tmp_path, _inputs([_empty_feature()], ledger=_ledger()))
    later = NOW + timedelta(days=7)
    removed = _inputs([], status=_status(later, experiment={"features": {"ledger": ["impact"]}}), at=later)
    _run(tmp_path, removed, now=later)
    assert next(iter(_state(tmp_path)["tasks"].values()))["status"] == "RESOLVED"


def test_corrected_evidence_that_refutes_an_earlier_success_sends_an_immediate_warning(tmp_path):
    _rep, out = _run(tmp_path, _inputs(status=_status_with(_variant())))                   # Wochenlauf: Übernahme
    assert len(out.mails) == 1 and "Belegte Verbesserung ridge_alpha=30.0" in out.mails[0]["subject"]
    later = NOW + timedelta(days=1)
    refuted = _inputs(status=_status_with(_variant(status="CONFIRMED_ADOPTION", state="REVIEW_REQUIRED",
                                                   contradicts=True, closed_at="2026-10-30T09:00:00+00:00"),
                                          at=later), at=later)
    rep2, out2 = _run(tmp_path, refuted, now=later, mode="critical")     # nächster täglicher Lauf
    assert rep2["mail_status"] == "sent" and len(out2.mails) == 1
    assert out2.mails[0]["subject"].startswith("[Scanner] Technischer Verbesserungsvorschlag – WARNUNG: belegte "
                                               "Verbesserung ridge_alpha=30.0 widerrufen")
    assert "REVIEW_REQUIRED" in out2.mails[0]["text"] and "Rücknahme" in out2.mails[0]["text"]
    _rep3, out3 = _run(tmp_path, refuted, now=later + timedelta(days=1), mode="weekly")
    assert out3.mails == []
    states = {t["kind"]: t["status"] for t in _state(tmp_path)["tasks"].values()}
    assert states == {"adoption": "RESOLVED", "withdrawn_success": "NOTIFIED"}  # Übernahme belegt erledigt


# ── Koordination paralleler Läufe ───────────────────────────────────────────────────────────────────────────────

def test_parallel_daily_and_weekly_runs_never_send_the_same_task_twice(tmp_path):
    """Beide Läufe starten vom selben Stand (parallele Workflows); ihre Ereignisse werden per Union-Merge in
    beliebiger Reihenfolge zusammengeführt. Jede Aufgabe gehört genau einer Laufart -> keine doppelte Mail."""
    inputs = _inputs([_empty_feature()], status={**_status(), **_plan_invalid()}, ledger=_ledger())
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _rep_w, weekly = _run(a, inputs, mode="weekly")
    _rep_c, critical = _run(b, inputs, mode="critical")
    sent = [m["subject"] for m in weekly.mails + critical.mails]
    assert len(sent) == 2 and len(set(sent)) == 2                         # P1 einmal im Digest, P0 einmal sofort
    merged = tmp_path / "merged"
    merged.mkdir()
    lines_a, lines_b = (p.read_text().splitlines() for p in (a / "tasks.jsonl", b / "tasks.jsonl"))
    (merged / "tasks.jsonl").write_text("\n".join(lines_b[:1] + lines_a + lines_b[1:]) + "\n")   # verschränkt
    for mode in rd.MODES:
        later = NOW + timedelta(hours=6)
        rep, out = _run(merged, _inputs([_empty_feature()], status={**_status(later), **_plan_invalid(later)},
                                        ledger=_ledger(), at=later), now=later, mode=mode)
        assert out.mails == [] and rep["mail_status"] == "nothing_to_send"
    assert {t["status"] for t in _state(merged)["tasks"].values()} == {"NOTIFIED"}


def test_serialized_daily_runs_share_the_history(tmp_path):
    inputs = _inputs(status=_plan_invalid())
    _run(tmp_path, inputs, mode="critical")
    rep, out = _run(tmp_path, inputs, mode="critical", now=NOW + timedelta(minutes=5))      # manuell kurz danach
    assert out.mails == [] and rep["mail_status"] == "nothing_to_send"


def test_dry_run_sends_nothing_and_changes_no_persistent_state(tmp_path):
    _run(tmp_path, _five_tasks())                                         # vorhandene Historie
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    for mode, inputs in (("weekly", _five_tasks(NOW + timedelta(days=7))),
                         ("critical", _inputs(status=_plan_invalid(NOW + timedelta(days=7)),
                                              at=NOW + timedelta(days=7)))):
        rep, out = _run(tmp_path, inputs, now=NOW + timedelta(days=7), mode=mode, dry_run=True)
        assert out.mails == [] and rep["mail_status"] == "dry_run" and rep["mails"][0]["text"]
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


# ── weitere Regeln ───────────────────────────────────────────────────────────────────────────────────────────────

def test_human_decision_stops_further_mails(tmp_path):
    tasks, _ = rd.improvement_tasks(_inputs([_empty_feature()], ledger=_ledger()), NOW)
    tid = tasks[0]["task_id"]
    rep, out = _notify(tmp_path, tasks, decisions={tid: {"status": "rejected", "note": "bewusst"}})
    assert out.mails == [] and rep["mail_status"] == "nothing_to_send"
    state = _state(tmp_path)["tasks"][tid]
    assert state["status"] == "REJECTED" and state["decision_note"] == "bewusst"
    cfg = yaml.safe_load((ROOT / "config" / "improvement_task_decisions.yaml").read_text(encoding="utf-8"))
    assert cfg == {"tasks": {}}


def test_no_credentials_reach_mail_state_or_report(tmp_path, monkeypatch):
    monkeypatch.setenv("GMAIL_APP_PW", "geheimes-passwort-123")
    monkeypatch.setenv("GMAIL_SENDER", "absender@example.com")
    monkeypatch.setenv("NOTIFY_EMAIL", "empfaenger@example.com")
    rep, out = _run(tmp_path, _inputs([_empty_feature()], ledger=_ledger()), send=Outbox("failed"))
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
    missing = _status_with(forward_sequential={"status": "RUNNING", "binding": {
        "look": 1, "verdict": "INSUFFICIENT_COVERAGE", "decision_coverage": 0.5, "excluded": {"NO_DECISION": 9}}})
    assert [t["kind"] for t in rd.improvement_tasks(_inputs(status=missing), NOW)[0]] == ["decision_capture"]


def _infra():
    snap = {"generated": "2026-10-30T18:11:25+00:00",
            "sources": {"port": {"status": "BROKEN", "criticality": "IMPORTANT", "last_success": "2026-10-20"},
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
    return snap, hist, {"recoveries": rec, "days": days}


def test_persistent_infrastructure_problems_only():
    snap, hist, wf = _infra()
    tasks, _ = rd.improvement_tasks(_inputs(history=hist, snapshot=snap, workflow=wf), NOW)
    by = {(t["kind"], t["subject"]): t for t in tasks}
    # nicht: minor (unkritisch), fresh (zu kurz), scanner (nur ein Nachstart)
    assert set(by) == {("source_failure", "port"), ("workflow_recovery", "source_health")}
    assert "seit 2026-10-20 (10 Tage, 11 Prüfungen)" in by[("source_failure", "port")]["finding"]
    wf_task = by[("workflow_recovery", "source_health")]
    assert "an 6 der letzten 7 Tage" in wf_task["finding"] and ".github/workflows/source_health.yml" in wf_task["files"]
    assert any("letzter Lauf 2026-10-30T18:10:19Z" in e for e in wf_task["evidence"])


def test_infrastructure_resolution_needs_fresh_health_data(tmp_path):
    snap, hist, wf = _infra()
    _run(tmp_path, _inputs(history=hist, snapshot=snap, workflow=wf))
    later = NOW + timedelta(days=8)
    healthy = {"generated": later.isoformat(), "sources": {"port": {"status": "HEALTHY", "criticality": "IMPORTANT"}}}
    quiet = {"recoveries": {}, "days": {later.date().isoformat(): {}}}
    _run(tmp_path, _inputs(snapshot=healthy, workflow=quiet, at=later), now=later)
    assert {t["kind"]: t["status"] for t in _state(tmp_path)["tasks"].values()} == {
        "source_failure": "RESOLVED", "workflow_recovery": "RESOLVED"}
    other = tmp_path / "stale"
    other.mkdir()
    _run(other, _inputs(history=hist, snapshot=snap, workflow=wf))
    _run(other, _inputs(snapshot=snap, workflow=wf, at=later), now=later)  # Gesundheitsdaten 9 Tage alt
    assert {t["status"] for t in _state(other)["tasks"].values()} == {"UNKNOWN"}


def test_research_memory_and_meta_cognition_show_the_tasks(tmp_path):
    _run(tmp_path, _inputs([_empty_feature()], ledger=_ledger()))
    state = _state(tmp_path)
    tid, task = next(iter(state["tasks"].items()))
    entries = rm.task_entries(state)
    assert [(e["kind"], e["status"], e["hypothesis_id"]) for e in entries] == [("dev_task", "NOTIFIED", tid)]
    assert entries[0]["evidence"]["fingerprint"] == task["fingerprint"]
    assert rm.SOURCES["improvement_tasks"] == Path("outputs/research/improvement_tasks.json")
    from modules import meta_cognition as mc
    st = mc.machine_state({"improvement_tasks": state}, {}, {}, {}, {"active": False, "reasons": []})
    assert st["open_improvement_tasks"] == [f"[P1] {tid} {task['short']} (NOTIFIED)"]
    assert st["improvement_mail_status"] == {"weekly": "sent"}
    assert "Offene technische Verbesserungsaufgaben" in mc.render_md(st)


def test_cli_modes_and_error_status(tmp_path, monkeypatch):
    for k in ("GMAIL_SENDER", "GMAIL_APP_PW", "NOTIFY_EMAIL", "MAIL_TO", "SMTP_HOST", "MAIL_PROVIDER"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(rd, "TASKS", tmp_path / "tasks.jsonl")
    monkeypatch.setattr(rd, "TASKS_STATE", tmp_path / "tasks.json")
    monkeypatch.setattr(rd, "TASK_DECISIONS", tmp_path / "none.yaml")
    inputs = _inputs([_empty_feature()], status={**_status(), **_plan_invalid()}, ledger=_ledger())
    monkeypatch.setattr(rd, "load_task_inputs", lambda paths=None, now=None: inputs)
    monkeypatch.setattr(rd, "datetime", type("D", (datetime,), {"now": classmethod(lambda cls, tz=None: NOW)}))
    synced = []
    monkeypatch.setattr(rm, "sync", lambda *a, **k: synced.append(1) or 0)
    assert rd.main(["--notify", "--dry-run"]) == 0 and rd.main(["--notify", "--critical", "--dry-run"]) == 0
    assert not (tmp_path / "tasks.jsonl").exists() and synced == []
    assert rd.main(["--notify", "--critical"]) == 2                      # sichtbar rot (Konfiguration fehlt)
    assert rd.main(["--notify"]) == 2 and synced == [1]                   # Research Memory nur im Wochenlauf
    runs = _state(tmp_path)["last_runs"]
    assert {m: r["mail_status"] for m, r in runs.items()} == {"critical": "not_configured", "weekly": "not_configured"}


def test_workflows_wire_daily_p0_weekly_digest_and_safe_concurrency():
    ml = yaml.safe_load((ROOT / ".github/workflows/ml_research.yml").read_text(encoding="utf-8"))
    assert ml["concurrency"]["group"] == "research-write" and "23 8 * * 6" in [c["cron"] for c in ml[True]["schedule"]]
    steps = ml["jobs"]["ml"]["steps"]
    ids = [s.get("id") for s in steps]
    assert ids.index("improvement_notify") == ids.index("research_improvement") + 1
    step = steps[ids.index("improvement_notify")]
    assert step["run"] == "python -m modules.research_director --notify" and step["continue-on-error"] is True
    assert set(step["env"]) == {"GMAIL_SENDER", "GMAIL_APP_PW", "NOTIFY_EMAIL"}
    assert all(v.startswith("${{ secrets.") for v in step["env"].values())
    check = next(s for s in steps if s.get("name", "").startswith("Stufen-Bilanz"))
    assert "improvement_notify=${{ steps.improvement_notify.outcome }}" in check["env"]["DEGRADED"]
    daily = yaml.safe_load((ROOT / ".github/workflows/improvement_notify.yml").read_text(encoding="utf-8"))
    assert [c["cron"] for c in daily[True]["schedule"]] == ["37 21 * * *"]
    inputs = daily[True]["workflow_dispatch"]["inputs"]
    assert inputs["dry_run"]["default"] == "true" and inputs["mode"]["default"] == "critical"
    group = daily["concurrency"]["group"]
    assert "improvement-notify" in group and "research-write" in group and "improvement-preview-" in group
    assert daily["concurrency"]["cancel-in-progress"] is False
    notify = next(s for s in daily["jobs"]["notify"]["steps"] if s.get("id") == "notify")
    assert "--critical" in notify["run"] and "--dry-run" in notify["run"] and "${{" not in notify["run"]
    assert notify["env"]["DRY_RUN"] == "${{ github.event_name == 'schedule' && 'false' || inputs.dry_run }}"
    push = (ROOT / "scripts" / "ci_push.sh").read_text(encoding="utf-8")
    assert "outputs/research/improvement_tasks.json)" in push            # abgeleitet: Konflikt -> dieser Lauf
    assert "outputs/**/*.jsonl merge=union" in (ROOT / ".gitattributes").read_text(encoding="utf-8")
    src = (ROOT / "modules" / "research_director.py").read_text(encoding="utf-8")
    assert "anthropic" not in src.lower() and "openai" not in src.lower()   # kein LLM-Aufruf


@pytest.mark.parametrize("missing", ["improvement", "status", "ledger"])
def test_missing_inputs_are_tolerated(missing):
    inp = _inputs([_empty_feature()], ledger=_ledger())
    inp[missing] = None
    tasks, _ = rd.improvement_tasks(inp, NOW)
    assert all(t["kind"] == "empty_feature" for t in tasks)
    if missing in ("improvement", "status"):
        assert tasks == []


def test_real_research_snapshot_round_trip(tmp_path):
    """Kopie des echten Wochenstatus (falls im Checkout vorhanden) lässt sich laden und auswerten, ohne Fehler."""
    src = ROOT / "outputs" / "research" / "option_learner"
    if not (src / "status_5d.json").exists():
        pytest.skip("kein Wochenstatus im Checkout")
    shutil.copytree(src, tmp_path / "ol", ignore=shutil.ignore_patterns("predictions.jsonl", "variants"))
    inputs = rd.load_task_inputs({"option_root": tmp_path / "ol"}, NOW)
    tasks, issues = rd.improvement_tasks(inputs, NOW)
    assert isinstance(tasks, list) and isinstance(issues, list)

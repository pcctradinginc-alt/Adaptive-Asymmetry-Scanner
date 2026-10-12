"""
Tests für modules/ops_report.py – Bericht, Historie, Deltas, dedupe-fähige Warnungen, Mail, CLI.

ALLE Eingaben sind SYNTHETISCH (keine echten Ledger-Daten, kein Netzwerk, kein SMTP): `make_status()` baut ein
gesundes ops-status-v1-Dict; Tests verschlechtern gezielt einzelne Felder. `ops_monitor.collect` und
`modules.mailer.send_mail` werden gemockt.
"""
from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from modules import ops_monitor, ops_report

AS_OF = "2026-10-13"                      # Dienstag (Handelstag)


def make_status(as_of: str = AS_OF) -> dict:
    """SYNTHETISCHER, gesunder Status (Schema ops-status-v1)."""
    entry = {"denominator": 10, "observed": 9, "ok": 9, "suspect": 0, "unusable": 0, "missing": 1, "coverage": 0.9,
             "by_status": {"ENTRY_OK": 9, "ENTRY_NOT_CAPTURED": 1}, "excluded_no_contract": 0}
    ex_c = {"denominator": 4, "observed": 4, "coverage": 1.0, "by_status": {"EXIT_OBSERVED_ARCHIVED": 4}}
    comp = {"denominator": 4, "evaluable": 4, "coverage": 1.0, "by_status": {"OK": 4}, "by_tier": {}, "reasons": {}}
    win = {"entry": entry, "exit_closed": ex_c, "complete": comp}
    mark = {"due": 10, "marked": 10, "pending": 0, "late_marks": 0, "oldest_pending_days": None}
    obs = {"n": 3, "n_independent_days": 3, "sum_net": -0.5, "mean_net": -0.1667, "hit_rate": 0.33,
           "worst": -0.4, "max_drawdown_cum": -0.5, "verdict": "NEED_MORE_DATA", "ci": None}
    chal = {"state": "PROSPECTIVE_CHALLENGER", "population": "CHAMPION_TRADE", "evaluable_resolved": 1,
            "would_act_on": 1, "avoided_losses": 0.5, "foregone_gains": 0, "incremental_vs_champion_sum": 0.5,
            "verdict": "NEED_MORE_DATA"}
    return {
        "schema": "ops-status-v1", "generated": f"{as_of}T16:00:00+00:00", "as_of": as_of, "read_only": True,
        "run": {"workflows": {j: {"state": "OK", "conclusion": "success", "last_run": f"{as_of}T13:40:00Z",
                                  "delay_hours": 0.2} for j in ("scanner", "feedback", "external_data", "source_health")},
                "report_date": as_of, "report_is_today": True,
                "scanner": {"universe": 300, "analyzed": 90, "champion_candidates": 12, "contracts_designed": 8,
                            "trades": 1, "stop_reason": None, "session": "regular"}},
        "quote_coverage": {"windows": {"today": copy.deepcopy(win), "7d": copy.deepcopy(win), "30d": copy.deepcopy(win)},
                           "exit_active": {"denominator": 4, "by_status": {}, "mark_today": 4, "coverage": 1.0,
                                           "oldest_mark_age_days": 0.2, "trading_day": True},
                           "candidate_marks": {"real_opt_ret_5d": dict(mark), "real_strat_ret_5d": dict(mark)}},
        "lifecycle": {"champion": {"by_state": {"SIGNAL": 0, "CONTRACT_SELECTED": 0, "ENTRY_OBSERVED": 0, "EXIT_PENDING": 4,
                                                "OUTCOME_FINAL": 10, "LEARNING_ELIGIBLE": 0, "LEARNED": 6},
                                   "issues": {}, "blocked": [], "legacy": {"note": "Altbestand"},
                                   "exactly_once": {"learned_keys": 6, "duplicate_learnable_trades": 0,
                                                    "learned_not_eligible": 0, "eligible_not_learned": 0, "ok": True}},
                      "candidates": {"SIGNAL": 50, "EXIT_PENDING": 5, "OUTCOME_FINAL": 3}},
        "forward": {"champion": {"observed": obs, "not_evaluable": {}}, "challengers": {"H-1@v1": chal},
                    "scorecard": {"CHAMPION_TRADE|ALL": {"n": 0, "verdict": "NO_DATA"}}},
        "training": {"status": "NEVER_RUN", "n_labels": None, "n_label_dates": None, "required": None, "versions": 0,
                     "active": None, "hold": None, "active_artifact_ok": None, "manifest_violations": []},
        "promotion": {"max_automatic_influence": "ABSTENTION_ONLY", "kill_switch_active": False, "safe_mode": True,
                      "champion_version": "rules@x", "by_state": {"PROSPECTIVE_CHALLENGER": 1},
                      "active_influence": [], "unjustified_influence": [],
                      "hypotheses": {"H-1@v1": {"state": "PROSPECTIVE_CHALLENGER", "influence_level": "NONE",
                                                "decision": "KEEP_SHADOW", "blocking": "n 1/60"}}},
        "integrity": {"issues": [], "promotion_state_hash_ok": True},
        "data_health": {"source_health": {"generated": as_of, "by_status": {"HEALTHY": 40, "DEGRADED": 1}},
                        "external_data": {"by_status": {}}, "quote_archive_days": {as_of: {"n": 100}},
                        "quote_archive_median_14d": 100, "quote_archive_root_exists": True},
    }


def put(status: dict, path: str, value) -> dict:
    """Tiefe Kopie von `status` mit gesetztem Pfad 'a.b.c'."""
    s = copy.deepcopy(status)
    cur = s
    keys = path.split(".")
    for k in keys[:-1]:
        cur = cur[k]
    cur[keys[-1]] = value
    return s


def evaluate(status: dict, history: list[dict] | None = None, state: dict | None = None):
    row = ops_report.history_row(status)
    return ops_report.alerts(status, row, history or [], state or {}) + (row,)


def ids(al: list[dict]) -> dict:
    return {a["id"]: a for a in al}


# ── Historienzeile ──────────────────────────────────────────────────────────
def test_history_row_flat_metrics():
    row = ops_report.history_row(make_status())
    assert row["as_of"] == AS_OF and row["generated"].startswith(AS_OF)
    assert row["run.scanner.champion_candidates"] == 12 and row["run.scanner.contracts_designed"] == 8
    assert row["quote.entry.7d.denominator"] == 10 and row["quote.entry.30d.coverage"] == 0.9
    assert row["quote.exit_active.coverage"] == 1.0 and row["quote.complete.7d.evaluable"] == 4
    assert row["marks.real_opt_ret_5d.pending"] == 0                  # echter Nullwert bleibt 0
    assert row["lifecycle.champion.state.LEARNED"] == 6
    assert row["lifecycle.champion.issue.EXIT_OVERDUE"] == 0 and row["lifecycle.candidates.EXIT_OVERDUE"] == 0
    assert row["forward.champion.n"] == 3 and row["forward.champion.sum_net"] == -0.5
    assert row["training.status"] == "NEVER_RUN" and row["training.n_labels"] is None
    assert row["promotion.active_influence"] == 0 and row["promotion.kill_switch_active"] is False
    assert row["integrity.issues"] == 0
    assert row["data.source_health.HEALTHY"] == 40 and row["data.source_health.n_bad"] == 1
    assert row["data.quote_archive_median_14d"] == 100 and row["data.quote_archive_today"] == 100
    json.dumps(row)                                                   # JSON-tauglich


def test_history_row_none_stays_none_for_missing_and_errored_sections():
    s = make_status()
    s["integrity"] = {"error": "ValueError: kaputt"}
    s["lifecycle"] = {"error": "boom"}
    s["data_health"]["quote_archive_root_exists"] = False
    s["data_health"]["quote_archive_days"] = {}
    s["data_health"]["quote_archive_median_14d"] = None
    del s["promotion"]
    row = ops_report.history_row(s)
    for k in ("integrity.issues", "lifecycle.champion.state.LEARNED", "lifecycle.champion.issue.EXIT_OVERDUE",
              "lifecycle.candidates.EXIT_OVERDUE", "promotion.active_influence", "promotion.kill_switch_active",
              "data.quote_archive_today", "data.quote_archive_median_14d"):
        assert row[k] is None, k
    empty = ops_report.history_row({})
    assert all(v is None for k, v in empty.items() if k not in ("schema",))


# ── N/A-Rendering ───────────────────────────────────────────────────────────
def test_render_none_is_na_never_zero():
    s = make_status()
    e7 = s["quote_coverage"]["windows"]["7d"]
    e7["entry"].update(coverage=None, denominator=None, observed=None, by_status={})
    e7["exit_closed"].update(coverage=None, denominator=None, observed=None)
    s["quote_coverage"]["exit_active"].update(coverage=None, denominator=None, mark_today=None, oldest_mark_age_days=None)
    s["forward"]["champion"]["observed"] = {"n": None, "mean_net": None, "sum_net": None}
    s["training"]["status"] = None
    s["integrity"] = {"error": "x"}
    row = ops_report.history_row(s)
    md = ops_report.render_markdown(s, row, ops_report.deltas(row, []), [])
    line = next(ln for ln in md.splitlines() if ln.startswith("| Entry 7d"))
    assert "N/A" in line and "0 %" not in line and "0/" not in line
    for label in ("Exit geschlossen 7d", "Exit aktiv"):
        ln = next(ln for ln in md.splitlines() if ln.startswith(f"| {label}"))
        assert ln.count("N/A") >= 2 and "0 %" not in ln
    assert "**mean_net**: N/A" in md and "**sum_net**: N/A" in md
    assert "Abschnitt nicht berechenbar" in md and "N/A" in md
    page = ops_report.render_html(s, row, {}, [])
    assert "N/A" in page


def test_render_header_forward_disclaimer_and_sections():
    s = make_status()
    row = ops_report.history_row(s)
    md = ops_report.render_markdown(s, row, ops_report.deltas(row, []), [])
    assert "READ-ONLY Monitoring – keine Handelsempfehlung" in md
    assert "keine Alpha-Behauptung" in md
    for sec in ("## Lauf", "## Quote-Abdeckung", "## Lebenszyklus", "## Forward-Performance", "## Training",
                "## Promotion", "## Integrität", "## Daten", "## Warnungen", "## Trends"):
        assert sec in md
    assert "| Entry 7d | 90.0 % | 9/10 |" in md                      # Nenner als x/y
    assert "Keine aktiven Warnungen." in md
    html_page = ops_report.render_html(s, row, {}, [])
    assert "READ-ONLY Monitoring – keine Handelsempfehlung" in html_page and "<script" not in html_page


def test_render_html_escapes_everything():
    evil = "<script>alert(1)</script> & \"q\""
    s = put(make_status(), "run.scanner.stop_reason", evil)
    s["promotion"]["hypotheses"]["H-<b>"] = {"state": evil, "influence_level": "NONE", "decision": evil, "blocking": evil}
    s["forward"]["challengers"]["<i>x</i>"] = {"verdict": evil, "population": evil}
    al = [{"id": "X<1>", "severity": "WARNING", "component": evil, "cause": evil, "since": AS_OF, "status": "NEW"}]
    row = ops_report.history_row(s)
    page = ops_report.render_html(s, row, ops_report.deltas(row, []), al)
    assert "<script>" not in page and "<b>" not in page.replace("<b>READ-ONLY", "") and "<i>x</i>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page and "X&lt;1&gt;" in page


# ── Deltas ──────────────────────────────────────────────────────────────────
def _hist(day: str, **kw) -> dict:
    return {"as_of": day, "generated": f"{day}T10:00:00+00:00", **kw}


def test_deltas_prev_d7_d30_and_none_safety():
    row = {"as_of": "2026-10-31", "m": 10, "z": 0, "only_now": 4, "gone": None, "flag": True, "text": "x"}
    hist = [_hist("2026-09-30", m=1, z=0, gone=7),          # >= 30 Tage zurück
            _hist("2026-10-20", m=4, z=None, gone=5),       # 11 Tage zurück -> d7
            _hist("2026-10-25", m=6, z=0, gone=None),       # 6 Tage zurück: zu jung für d7, aber nächstliegendes für nichts
            _hist("2026-10-30", m=9, z=1, only_now=None, gone=3),   # prev
            _hist("2026-10-31", m=999),                     # gleicher Tag: ignoriert
            _hist("2026-11-02", m=998)]                     # Zukunft: ignoriert
    d = ops_report.deltas(row, hist)
    assert "flag" not in d and "text" not in d and "as_of" not in d       # nur numerisch (bool/str raus)
    assert d["m"] == {"value": 10, "prev": 9, "d7": 4, "d30": 1, "diff_prev": 1, "diff_d7": 6, "diff_d30": 9}
    assert d["z"]["prev"] == 1 and d["z"]["diff_prev"] == -1
    assert d["z"]["d7"] is None and d["z"]["diff_d7"] is None             # Referenz None -> Differenz None
    assert d["z"]["d30"] == 0 and d["z"]["diff_d30"] == 0                  # echte 0 bleibt 0
    assert d["only_now"]["prev"] is None and d["only_now"]["diff_prev"] is None
    assert d["gone"]["value"] is None and d["gone"]["diff_prev"] is None and d["gone"]["prev"] == 3  # Wert fehlt jetzt
    assert ops_report.reference_rows(row, hist)["d7"]["as_of"] == "2026-10-20"
    assert ops_report.reference_rows(row, hist)["d30"]["as_of"] == "2026-09-30"


def test_deltas_without_history_or_as_of():
    d = ops_report.deltas({"as_of": AS_OF, "a": 1}, [])
    assert d["a"] == {"value": 1, "prev": None, "d7": None, "d30": None, "diff_prev": None, "diff_d7": None,
                      "diff_d30": None}
    assert ops_report.deltas({"a": 1}, [_hist("2026-10-01", a=1)])["a"]["prev"] is None


# ── Historie ────────────────────────────────────────────────────────────────
def test_history_idempotent_per_date_first_row_wins(tmp_path):
    p = tmp_path / "h" / "ops_history.jsonl"
    r1 = {**ops_report.history_row(make_status()), "run.scanner.trades": 1}
    r2 = {**ops_report.history_row(make_status()), "run.scanner.trades": 99}
    assert ops_report.append_history(p, r1) is True
    assert ops_report.append_history(p, r2) is False                       # gleicher Tag -> nicht angehängt
    assert ops_report.append_history(p, ops_report.history_row(make_status("2026-10-14"))) is True
    rows = ops_report.load_history(p)
    assert [r["as_of"] for r in rows] == [AS_OF, "2026-10-14"]
    assert rows[0]["run.scanner.trades"] == 1                              # erste Zeile kanonisch
    assert ops_report.append_history(p, {"as_of": None}) is False


# ── Warnregeln ──────────────────────────────────────────────────────────────
def test_healthy_status_has_no_alerts():
    al, state, _ = evaluate(make_status())
    assert al == [] and state == {}


def _wf(job, state=None, conclusion=None):
    s = make_status()
    s["run"]["workflows"][job].update(state=state or "OK", conclusion=conclusion or "success")
    return s


@pytest.mark.parametrize("status, aid, sev", [
    (_wf("scanner", "MISSED"), "SCANNER_FAILED:scanner", "WARNING"),
    (_wf("scanner", "RECOVERY_EXHAUSTED"), "SCANNER_FAILED:scanner", "CRITICAL"),
    (_wf("scanner", "PENDING", "failure"), "SCANNER_FAILED:scanner", "CRITICAL"),
    (_wf("feedback", "OK", "failure"), "SCANNER_FAILED:feedback", "WARNING"),   # Tag durch früheren Lauf versorgt
    (_wf("external_data", "MISSED"), "ARCHIVE_FAILED:external_data", "WARNING"),
    (_wf("source_health", "RECOVERY_EXHAUSTED"), "ARCHIVE_FAILED:source_health", "WARNING"),
    # Qualitäts-/Integritätsregeln
    (put(make_status(), "integrity.issues", ["PROMOTION_STATE_HASH", "QUOTE_ARCHIVE"]), "LEDGER_INTEGRITY", "CRITICAL"),
    (put(make_status(), "lifecycle.champion.exactly_once.ok", False), "LEARNING_EXACTLY_ONCE", "CRITICAL"),
    (put(make_status(), "lifecycle.champion.exactly_once.learned_not_eligible", 2), "LEARNING_EXACTLY_ONCE", "CRITICAL"),
    (put(make_status(), "lifecycle.champion.exactly_once.eligible_not_learned", 1),
     "LEARNING_EXACTLY_ONCE:eligible_not_learned", "WARNING"),
    (put(make_status(), "training.manifest_violations", [{"version": "v1", "violations": ["LABEL_AFTER_TRAINING"]}]),
     "TRAINING_INADMISSIBLE:manifest", "CRITICAL"),
    (put(make_status(), "training.active_artifact_ok", False), "TRAINING_INADMISSIBLE:active_artifact", "CRITICAL"),
    (put(make_status(), "promotion.unjustified_influence", ["H-1@v1"]), "PROMOTION_UNJUSTIFIED", "CRITICAL"),
    (put(make_status(), "promotion.kill_switch_active", True), "PROMOTION_UNJUSTIFIED:kill_switch", "WARNING"),
    # Überfällige Outcomes
    (put(put(make_status(), "lifecycle.champion.issues", {"EXIT_OVERDUE": 1}), "lifecycle.champion.blocked",
         [{"ticker": "AAA", "entry_date": "2026-08-01", "state": "EXIT_PENDING", "issues": ["EXIT_OVERDUE"]}]),
     "OUTCOMES_OVERDUE:champion", "WARNING"),
    (put(make_status(), "lifecycle.candidates", {"EXIT_OVERDUE": 2}), "OUTCOMES_OVERDUE:candidates", "WARNING"),
    (put(make_status(), "quote_coverage.candidate_marks.real_opt_ret_5d",
         {"due": 10, "marked": 5, "pending": 5, "late_marks": 0, "oldest_pending_days": 4}),
     "OUTCOMES_OVERDUE:marks", "WARNING"),
])
def test_each_rule_triggers_on_its_condition(status, aid, sev):
    al, _, _ = evaluate(status)
    assert aid in ids(al), [a["id"] for a in al]
    assert ids(al)[aid]["severity"] == sev and ids(al)[aid]["status"] == "NEW"
    a = ids(al)[aid]
    assert set(a) == {"id", "severity", "component", "cause", "since", "status"} and a["since"] == AS_OF
    assert a["cause"] and a["component"]


@pytest.mark.parametrize("status", [
    _wf("scanner", "PENDING"), _wf("scanner", "NOT_SCHEDULED"), _wf("scanner", "RUNNING", "success"),
    put(make_status(), "integrity.issues", []),
    put(make_status(), "lifecycle.champion.exactly_once.ok", True),
    put(make_status(), "lifecycle.champion.exactly_once.eligible_not_learned", 0),
    put(make_status(), "training.manifest_violations", []),
    put(make_status(), "training.active_artifact_ok", True),
    put(make_status(), "promotion.unjustified_influence", []),
    put(make_status(), "promotion.kill_switch_active", False),
    put(make_status(), "lifecycle.champion.issues", {"ENTRY_NOT_CAPTURED_BEFORE_ARCHIVE": 124}),   # Altbestand: kein Alarm
    put(make_status(), "lifecycle.candidates", {"EXIT_OVERDUE": 0}),
    put(make_status(), "quote_coverage.candidate_marks.real_opt_ret_5d",
        {"due": 10, "marked": 5, "pending": 5, "late_marks": 0, "oldest_pending_days": 3}),            # Schwelle ist > 3
])
def test_rules_stay_silent_otherwise(status):
    assert evaluate(status)[0] == []


def _entry(den, obs, legacy=0):
    s = make_status()
    s["quote_coverage"]["windows"]["7d"]["entry"].update(
        denominator=den, observed=obs, coverage=obs / den if den else None,
        by_status={"ENTRY_NOT_CAPTURED_BEFORE_ARCHIVE": legacy, "ENTRY_OK": obs})
    return s


def test_entry_coverage_excludes_legacy_and_needs_denominator():
    al = ids(evaluate(_entry(10, 5))[0])                                       # 5/10 = 50 % < 80 %
    assert al["QUOTE_COVERAGE_LOW:entry_7d"]["severity"] == "WARNING" and "5/10" in al["QUOTE_COVERAGE_LOW:entry_7d"]["cause"]
    assert evaluate(_entry(2, 0, legacy=2))[0] == []                           # nur Altbestand -> kein Alarm
    assert evaluate(_entry(10, 0, legacy=10))[0] == []
    assert evaluate(_entry(12, 8, legacy=4))[0] == []                          # 8/8 ohne Altbestand = 100 %
    assert "QUOTE_COVERAGE_LOW:entry_7d" in ids(evaluate(_entry(12, 4, legacy=4))[0])    # 4/8 = 50 %
    assert evaluate(_entry(2, 0))[0] == []                                     # Nenner < 3
    assert evaluate(_entry(5, 4))[0] == []                                     # 80 % genau: kein Alarm


def test_exit_active_coverage_only_on_trading_days():
    def exit_active(cov, den, trading=True):
        s = make_status()
        s["quote_coverage"]["exit_active"].update(coverage=cov, denominator=den, mark_today=0, trading_day=trading)
        return s
    assert evaluate(exit_active(0.5, 4))[0][0]["id"] == "QUOTE_COVERAGE_LOW:exit_active"
    assert evaluate(exit_active(0.5, 4, trading=False))[0] == []
    assert evaluate(exit_active(0.0, 0))[0] == [] and evaluate(exit_active(None, 0))[0] == []
    assert evaluate(exit_active(0.8, 5))[0] == []
    assert evaluate(exit_active(1.0, 1))[0] == []


def test_complete_coverage_30d_is_info_only():
    def complete(cov, den):
        s = make_status()
        s["quote_coverage"]["windows"]["30d"]["complete"].update(coverage=cov, denominator=den, evaluable=int(cov * den))
        return s
    a = evaluate(complete(0.4, 5))[0]
    assert [(x["id"], x["severity"]) for x in a] == [("QUOTE_COVERAGE_LOW:complete_30d", "INFO")]
    assert evaluate(complete(0.4, 4))[0] == [] and evaluate(complete(0.5, 10))[0] == []


def test_data_degradation_source_health_increase_vs_previous_row():
    s = put(make_status(), "data_health.source_health.by_status", {"HEALTHY": 38, "DEGRADED": 2, "STALE": 1})
    prev = lambda n: [_hist("2026-10-12", **{"data.source_health.n_bad": n})]      # noqa: E731
    assert "DATA_DEGRADATION:source_health" in ids(evaluate(s, prev(1))[0])         # 3 vs 1 = +2
    assert evaluate(s, prev(2))[0] == []                                            # +1
    assert evaluate(s, [])[0] == []                                                 # kein Vorlauf -> unbekannt
    assert evaluate(s, [_hist("2026-10-12")])[0] == []                              # Vorlauf ohne Kennzahl (None)
    fail = put(make_status(), "data_health.source_health.by_status", {"HEALTHY": 10, "FAIL": 2})
    assert "DATA_DEGRADATION:source_health" in ids(evaluate(fail, prev(0))[0])


def test_data_degradation_quote_archive_drop():
    def drop(today, median=100, as_of=AS_OF, scanner_state="OK"):
        s = make_status(as_of)
        s["data_health"].update(quote_archive_median_14d=median, quote_archive_days={as_of: {"n": today}})
        s["run"]["workflows"]["scanner"]["state"] = scanner_state
        return s
    a = ids(evaluate(drop(40))[0])["DATA_DEGRADATION:quote_archive"]
    assert a["severity"] == "WARNING" and "40" in a["cause"]
    assert evaluate(drop(50))[0] == [] and evaluate(drop(100))[0] == []             # genau 50 %: kein Alarm
    assert evaluate(drop(0, median=19))[0] == []                                    # Median < 20: zu wenig Basis
    assert evaluate(drop(10, as_of="2026-10-11"))[0] == []                          # Sonntag
    assert evaluate(drop(10, scanner_state="PENDING"))[0] == []                     # Scanner läuft noch
    s = drop(0)
    s["data_health"]["quote_archive_days"] = {}
    assert "DATA_DEGRADATION:quote_archive" in ids(evaluate(s)[0])                  # Archiv-Verzeichnis da, heute 0 Quotes
    s["data_health"]["quote_archive_root_exists"] = False
    assert evaluate(s)[0] == []                                                     # unbekannt (None), nicht 0


def test_section_error_alert_and_dependent_rules_skipped():
    s = make_status()
    s["integrity"] = {"error": "ValueError: kaputt"}
    s["promotion"] = {"error": "KeyError: x"}
    al, _, _ = evaluate(s)
    d = ids(al)
    assert set(d) == {"MONITOR_SECTION_ERROR:integrity", "MONITOR_SECTION_ERROR:promotion"}
    assert all(a["severity"] == "WARNING" for a in al)
    assert "Integrität" in d["MONITOR_SECTION_ERROR:integrity"]["cause"] and "integrity" in d["MONITOR_SECTION_ERROR:integrity"]["cause"]


def test_section_error_keeps_existing_alert_state_instead_of_resolving():
    bad = put(make_status(), "integrity.issues", ["TRANSITION_CHAIN"])
    _, st, _ = evaluate(bad)
    assert "LEDGER_INTEGRITY" in st
    err = put(make_status(), "integrity", {"error": "boom"})
    al, st2, _ = evaluate(err, state=st)
    assert "LEDGER_INTEGRITY" not in ids(al) and st2["LEDGER_INTEGRITY"] == st["LEDGER_INTEGRITY"]
    assert ids(al)["MONITOR_SECTION_ERROR:integrity"]["status"] == "NEW"
    al3, st3, _ = evaluate(make_status(), state=st2)
    assert ids(al3)["LEDGER_INTEGRITY"]["status"] == "RESOLVED" and "LEDGER_INTEGRITY" not in st3


def _overdue(names: list[str], day: str = AS_OF) -> dict:
    s = make_status(day)
    s["lifecycle"]["champion"]["issues"] = {"EXIT_OVERDUE": len(names)}
    s["lifecycle"]["champion"]["blocked"] = [{"ticker": n, "entry_date": "2026-08-01", "state": "EXIT_PENDING",
                                              "issues": ["EXIT_OVERDUE"]} for n in names]
    return s


def test_alert_lifecycle_new_ongoing_changed_resolved():
    aid = "OUTCOMES_OVERDUE:champion"
    al, st, _ = evaluate(_overdue(["AAA"], "2026-10-13"))
    assert ids(al)[aid]["status"] == "NEW" and ids(al)[aid]["since"] == "2026-10-13"
    assert st[aid]["first_seen"] == "2026-10-13" and st[aid]["last_seen"] == "2026-10-13" and st[aid]["fingerprint"]
    al, st2, _ = evaluate(_overdue(["AAA"], "2026-10-14"), state=st)               # unveränderte Ursache
    assert ids(al)[aid]["status"] == "ONGOING" and ids(al)[aid]["since"] == "2026-10-13"
    assert st2[aid]["fingerprint"] == st[aid]["fingerprint"] and st2[aid]["last_seen"] == "2026-10-14"
    assert st2[aid]["first_seen"] == "2026-10-13"
    al, st3, _ = evaluate(_overdue(["AAA", "BBB"], "2026-10-15"), state=st2)       # neuer überfälliger Trade
    assert ids(al)[aid]["status"] == "CHANGED" and ids(al)[aid]["since"] == "2026-10-13"
    assert st3[aid]["fingerprint"] != st2[aid]["fingerprint"] and st3[aid]["first_seen"] == "2026-10-13"
    al, st4, _ = evaluate(make_status("2026-10-16"), state=st3)                    # behoben
    assert [(a["id"], a["status"], a["since"]) for a in al] == [(aid, "RESOLVED", "2026-10-13")]
    assert st4 == {}
    al, st5, _ = evaluate(make_status("2026-10-17"), state=st4)                    # RESOLVED nur einmal
    assert al == [] and st5 == {}


def test_sliding_window_numbers_do_not_change_fingerprint():
    a, st, _ = evaluate(_entry(10, 5))
    b, _, _ = evaluate(_entry(11, 5), state=st)
    assert ids(b)["QUOTE_COVERAGE_LOW:entry_7d"]["status"] == "ONGOING"
    assert "5/11" in ids(b)["QUOTE_COVERAGE_LOW:entry_7d"]["cause"]                 # Ursache zeigt aktuelle Werte


def test_alerts_sorted_new_changed_first_and_state_not_mutated():
    s = put(put(make_status(), "integrity.issues", ["QUOTE_ARCHIVE"]), "promotion.kill_switch_active", True)
    _, st, _ = evaluate(put(make_status(), "promotion.kill_switch_active", True))
    snapshot = copy.deepcopy(st)
    al, _, _ = evaluate(s, state=st)
    assert [a["status"] for a in al] == ["NEW", "ONGOING"] and al[0]["id"] == "LEDGER_INTEGRITY"
    assert st == snapshot


# ── run(): Dateien, Historie, Mail ──────────────────────────────────────────
class Mailbox:
    def __init__(self, result=None):
        self.calls, self.result = [], result or {"status": "sent", "attempts": 1, "error": None}

    def __call__(self, subject, html, text, *, dry_run=False, env=None, **kw):
        self.calls.append({"subject": subject, "html": html, "text": text, "dry_run": dry_run, "env": env})
        return dict(self.result)


@pytest.fixture
def box(monkeypatch):
    b = Mailbox()
    monkeypatch.setattr("modules.mailer.send_mail", b)
    return b


def patch_collect(monkeypatch, status: dict | None = None, exc: Exception | None = None):
    def fake(now=None, paths=None):
        if exc:
            raise exc
        return copy.deepcopy(status)
    monkeypatch.setattr(ops_monitor, "collect", fake)


NOW = datetime(2026, 10, 13, 16, 0, tzinfo=timezone.utc)


def test_run_writes_only_inside_out_dir(tmp_path, monkeypatch):
    cwd, out = tmp_path / "cwd", tmp_path / "out" / "mon"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    patch_collect(monkeypatch, put(make_status(), "integrity.issues", ["QUOTE_ARCHIVE"]))
    res = ops_report.run(NOW, out_dir=out)
    assert res["ok"] is True and res["errors"] == []
    written = {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()}
    assert written == {f"out/mon/{n}" for n in ("ops_status.json", "ops_status.md", "ops_status.html",
                                                  "ops_history.jsonl", "alert_state.json")}
    assert not list(cwd.iterdir())
    doc = json.loads((out / "ops_status.json").read_text(encoding="utf-8"))
    assert doc["status"]["as_of"] == AS_OF and doc["row"]["as_of"] == AS_OF and doc["alerts"][0]["id"] == "LEDGER_INTEGRITY"
    assert "deltas" in doc and doc["history"]["appended"] is True
    assert "READ-ONLY Monitoring" in (out / "ops_status.md").read_text(encoding="utf-8")
    assert (out / "ops_status.html").read_text(encoding="utf-8").startswith("<!doctype html>")


def test_run_history_one_row_per_date(tmp_path, monkeypatch):
    out = tmp_path / "o"
    patch_collect(monkeypatch, make_status())
    r1 = ops_report.run(NOW, out_dir=out)
    r2 = ops_report.run(NOW, out_dir=out)                                           # Wiederholung am selben Tag
    assert r1["history"]["appended"] is True and r2["history"]["appended"] is False
    assert r2["history"]["reason"] == "as_of_exists"
    assert len(ops_report.load_history(out / "ops_history.jsonl")) == 1
    patch_collect(monkeypatch, make_status("2026-10-14"))
    r3 = ops_report.run(NOW.replace(day=14), out_dir=out)
    assert r3["history"]["appended"] is True and len(ops_report.load_history(out / "ops_history.jsonl")) == 2
    doc = json.loads((out / "ops_status.json").read_text(encoding="utf-8"))
    assert doc["delta_refs"]["prev"] == AS_OF and doc["deltas"]["run.scanner.trades"]["prev"] == 1


def test_run_mail_only_for_new_changed_resolved(tmp_path, monkeypatch, box):
    out = tmp_path / "o"
    warn = put(make_status(), "promotion.kill_switch_active", True)               # nur WARNING
    patch_collect(monkeypatch, make_status())
    assert ops_report.run(NOW, out_dir=out, mail=True)["mail"]["status"] == "skipped"
    assert box.calls == []                                                          # keine Warnung -> keine Mail
    patch_collect(monkeypatch, warn)
    r = ops_report.run(NOW, out_dir=out, mail=True)                                 # NEW
    assert len(box.calls) == 1 and r["mail"]["status"] == "sent"
    assert box.calls[0]["subject"] == "[Scanner-Monitor] 2026-10-13: 1 neue Warnung(en)" and box.calls[0]["dry_run"] is False
    assert "READ-ONLY Monitoring" in box.calls[0]["text"] and "<table" in box.calls[0]["html"]
    r = ops_report.run(NOW, out_dir=out, mail=True)                                 # ONGOING -> keine erneute Mail
    assert len(box.calls) == 1 and r["mail"]["status"] == "skipped"
    patch_collect(monkeypatch, put(make_status(), "integrity.issues", ["QUOTE_ARCHIVE"]))
    ops_report.run(NOW, out_dir=out, mail=True)                                     # RESOLVED + NEW CRITICAL
    assert len(box.calls) == 2 and "1 neue Warnung(en), 1 behoben, 1 KRITISCH" in box.calls[1]["subject"]
    r = ops_report.run(NOW, out_dir=out, mail=True)                                 # CRITICAL unverändert -> KEINE Mail
    assert len(box.calls) == 2 and r["mail"]["status"] == "skipped"                # (keine täglich identischen Alarme)
    assert any(a["severity"] == "CRITICAL" and a["status"] == "ONGOING" for a in r["alerts"])   # bleibt im Bericht
    patch_collect(monkeypatch, make_status())
    ops_report.run(NOW, out_dir=out, mail=True)                                     # RESOLVED -> Mail
    assert len(box.calls) == 3
    ops_report.run(NOW, out_dir=out, mail=True)                                     # nichts mehr -> still
    assert len(box.calls) == 3


def test_run_mail_changed_triggers_mail(tmp_path, monkeypatch, box):
    out = tmp_path / "o"
    patch_collect(monkeypatch, _overdue(["AAA"]))
    ops_report.run(NOW, out_dir=out, mail=True)
    ops_report.run(NOW, out_dir=out, mail=True)
    assert len(box.calls) == 1
    patch_collect(monkeypatch, _overdue(["AAA", "BBB"]))
    ops_report.run(NOW, out_dir=out, mail=True)
    assert len(box.calls) == 2 and "1 neue Warnung(en)" in box.calls[1]["subject"]


def test_run_mail_false_never_calls_mailer(tmp_path, monkeypatch, box):
    patch_collect(monkeypatch, put(make_status(), "integrity.issues", ["QUOTE_ARCHIVE"]))
    res = ops_report.run(NOW, out_dir=tmp_path / "o")
    assert box.calls == [] and res["mail"]["status"] == "skipped"


def test_run_dry_run_mail_passes_dry_run_and_keeps_alert_state(tmp_path, monkeypatch, box):
    out = tmp_path / "o"
    box.result = {"status": "dry_run", "attempts": 0, "error": None, "preview": {"subject": "x"}}
    patch_collect(monkeypatch, put(make_status(), "integrity.issues", ["QUOTE_ARCHIVE"]))
    res = ops_report.run(NOW, out_dir=out, dry_run_mail=True, env={"MAIL_TO": "a@example.com"})
    assert len(box.calls) == 1 and box.calls[0]["dry_run"] is True and box.calls[0]["env"] == {"MAIL_TO": "a@example.com"}
    assert res["mail"]["status"] == "dry_run"
    assert not (out / "alert_state.json").exists()                                  # Dry-Run verbraucht NEW nicht
    assert (out / "ops_status.json").exists() and res["history"]["appended"] is True
    res = ops_report.run(NOW, out_dir=out, mail=True, dry_run_mail=True)           # beides -> Dry-Run gewinnt
    assert box.calls[1]["dry_run"] is True and not (out / "alert_state.json").exists()
    ops_report.run(NOW, out_dir=out, mail=True)                                     # echter Lauf sieht weiterhin NEW
    assert box.calls[2]["dry_run"] is False and (out / "alert_state.json").exists()


def test_run_failed_mail_keeps_alert_state_for_retry(tmp_path, monkeypatch, box):
    out = tmp_path / "o"
    box.result = {"status": "failed", "attempts": 3, "error": "SMTPException"}
    patch_collect(monkeypatch, put(make_status(), "integrity.issues", ["QUOTE_ARCHIVE"]))
    r = ops_report.run(NOW, out_dir=out, mail=True)
    assert r["mail"]["status"] == "failed" and not (out / "alert_state.json").exists()
    box.result = {"status": "sent"}
    ops_report.run(NOW, out_dir=out, mail=True)                                     # erneut NEW -> erneut versucht
    assert len(box.calls) == 2 and (out / "alert_state.json").exists()


def test_run_not_configured_mail_is_recorded_not_fatal(tmp_path, monkeypatch, box):
    box.result = {"status": "not_configured", "attempts": 0, "error": None}
    patch_collect(monkeypatch, put(make_status(), "integrity.issues", ["QUOTE_ARCHIVE"]))
    r = ops_report.run(NOW, out_dir=tmp_path / "o", mail=True)
    assert r["ok"] is True and r["mail"]["status"] == "not_configured"
    doc = json.loads((tmp_path / "o" / "ops_status.json").read_text(encoding="utf-8"))
    assert doc["mail"]["status"] == "not_configured"


def test_run_never_raises_when_collect_raises(tmp_path, monkeypatch, box):
    out = tmp_path / "o"
    patch_collect(monkeypatch, exc=RuntimeError("collect kaputt"))
    res = ops_report.run(NOW, out_dir=out, mail=True)
    assert res["ok"] is False and any("collect kaputt" in e for e in res["errors"])
    assert ids(res["alerts"])["MONITOR_COLLECT_FAILED"]["severity"] == "WARNING"
    assert res["history"]["appended"] is False and res["history"]["reason"] == "collect_failed"
    assert not (out / "ops_history.jsonl").exists()                                 # leere Zeile würde den Tag blockieren
    assert (out / "ops_status.json").exists() and (out / "ops_status.md").exists() and len(box.calls) == 1
    patch_collect(monkeypatch, make_status())                                       # später am Tag: normaler Lauf
    assert ops_report.run(NOW, out_dir=out)["history"]["appended"] is True


def test_run_never_raises_when_mailer_or_state_or_history_break(tmp_path, monkeypatch):
    out = tmp_path / "o"
    patch_collect(monkeypatch, put(make_status(), "integrity.issues", ["QUOTE_ARCHIVE"]))

    def boom(*a, **k):
        raise RuntimeError("smtp down")
    monkeypatch.setattr("modules.mailer.send_mail", boom)
    res = ops_report.run(NOW, out_dir=out, mail=True)
    assert res["mail"]["status"] == "failed" and "smtp down" in res["mail"]["error"]
    out.mkdir(exist_ok=True)
    (out / "ops_history.jsonl").write_text('{"as_of": "2026-10-01"}\nkaputt\n{"as_of": "2026-10-02"}\n', encoding="utf-8")
    (out / "alert_state.json").write_text("{kein json", encoding="utf-8")
    res = ops_report.run(NOW, out_dir=out)                                          # defekte Historie + defekter Zustand
    assert res["ok"] is False and any(e.startswith("history") for e in res["errors"])
    assert res["history"]["appended"] is False and (out / "ops_status.json").exists()
    assert "LEDGER_INTEGRITY" in ids(res["alerts"])                                 # defekter Zustand -> wieder NEW


def test_run_survives_unwritable_out_dir(tmp_path, monkeypatch):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    patch_collect(monkeypatch, make_status())
    res = ops_report.run(NOW, out_dir=blocker / "sub")                              # Verzeichnis nicht anlegbar
    assert res["ok"] is False and res["errors"]


# ── CLI ─────────────────────────────────────────────────────────────────────
def test_cli_exit_zero_and_warning_lines_for_critical(tmp_path, monkeypatch, capsys, box):
    patch_collect(monkeypatch, put(make_status(), "integrity.issues", ["PROMOTION_STATE_HASH"]))
    assert ops_report.main(["--out", str(tmp_path / "o"), "--dry-run-mail"]) == 0
    out = capsys.readouterr().out
    assert "READ-ONLY Monitoring – keine Handelsempfehlung" in out
    assert "::warning::[CRITICAL] LEDGER_INTEGRITY" in out and "1 NEW" in out
    assert box.calls and box.calls[0]["dry_run"] is True


def test_cli_exit_zero_on_failures(tmp_path, monkeypatch, capsys):
    patch_collect(monkeypatch, exc=RuntimeError("x"))
    assert ops_report.main(["--out", str(tmp_path / "o")]) == 0
    assert "::warning::" in capsys.readouterr().out
    assert ops_report.main(["--gibt-es-nicht"]) == 0
    assert "::warning::" in capsys.readouterr().out


def test_cli_without_critical_prints_no_warning_annotation(tmp_path, monkeypatch, capsys):
    patch_collect(monkeypatch, put(make_status(), "promotion.kill_switch_active", True))
    assert ops_report.main(["--out", str(tmp_path / "o")]) == 0
    out = capsys.readouterr().out
    assert "PROMOTION_UNJUSTIFIED:kill_switch" in out and "::warning::" not in out


def test_fixture_status_is_not_mutated_by_report_functions():
    s = make_status()
    snap = copy.deepcopy(s)
    row = ops_report.history_row(s)
    ops_report.alerts(s, row, [], {})
    ops_report.render_markdown(s, row, ops_report.deltas(row, []), [])
    ops_report.render_html(s, row, {}, [])
    assert s == snap
    assert Path(ops_report.__file__).name == "ops_report.py"



def test_current_job_step_failure_is_critical(tmp_path, monkeypatch, box):
    """Synthetisch: OPS_STEP_OUTCOMES aus dem Scanner-Workflow – ein fehlgeschlagener Schritt im aktuellen Job ist
    CRITICAL, auch bevor der Watchdog den Tag bewertet; Erfolg/übersprungen löst nichts aus."""
    assert ops_report.step_outcomes({"OPS_STEP_OUTCOMES": "rth_marks=success scanner=failure feedback=skipped"}) == \
        {"rth_marks": "success", "scanner": "failure", "feedback": "skipped"}
    assert ops_report.step_outcomes({}) == {}
    patch_collect(monkeypatch, make_status())
    r = ops_report.run(NOW, out_dir=tmp_path / "a", env={"OPS_STEP_OUTCOMES": "scanner=failure feedback=skipped"})
    a = [x for x in r["alerts"] if x["id"] == "SCANNER_FAILED:current_job"]
    assert a and a[0]["severity"] == "CRITICAL" and "scanner" in a[0]["cause"]
    patch_collect(monkeypatch, make_status())
    ok = ops_report.run(NOW, out_dir=tmp_path / "b", env={"OPS_STEP_OUTCOMES": "scanner=success feedback=success"})
    assert not [x for x in ok["alerts"] if x["id"].startswith("SCANNER_FAILED")]

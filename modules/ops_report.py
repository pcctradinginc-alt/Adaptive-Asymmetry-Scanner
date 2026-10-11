"""modules/ops_report.py – Bericht, Trends, dedupe-fähige Warnungen und Mail zum read-only Monitoring (2026-10-11).

Baut auf `ops_monitor.collect()` (Schema ops-status-v1) auf und schreibt AUSSCHLIESSLICH in das Ausgabeverzeichnis
(Default outputs/monitoring): ops_status.json / .md / .html, ops_history.jsonl (append-only), alert_state.json.
Keine Handelsentscheidung, keine Änderung an Trades, Modellen, Verträgen oder Konfiguration.

Historie    Eine Zeile je Kalendertag (`as_of`). Die ERSTE Zeile eines Tages ist kanonisch; spätere Läufe desselben
            Tages werden berichtet, aber nicht angehängt (append-only, nie überschreiben).
Warnungen   Zustand `alert_state.json`: id -> {fingerprint, first_seen, last_seen, severity, component}.
            Fingerabdruck = Hash aus (id, ursachenrelevante Werte). Gleich -> ONGOING (keine erneute Mail), anders ->
            CHANGED, neu -> NEW, vorher da und jetzt weg -> RESOLVED (einmalig, dann aus dem Zustand entfernt).
            Gleitende Fenster-Quoten gehen NICHT in den Fingerabdruck ein (sonst täglich CHANGED); diskrete Zähler
            (überfällige Trades, Integritätsprobleme, Hypothesen-IDs) schon.
Mail        Nur bei NEW/CHANGED/RESOLVED – auch CRITICAL wird nicht täglich wiederholt, solange sich nichts ändert
            (sie bleibt im Bericht als ONGOING sichtbar und färbt den Job-Log mit ::warning::).
Aktueller   OPS_STEP_OUTCOMES ("schritt=outcome …", vom Scanner-Workflow gesetzt) -> run.current_job; ein Schritt mit
Job         failure ergibt SCANNER_FAILED:current_job (CRITICAL), auch bevor der Watchdog den Tag bewertet.
Fehler      `run()` und `main()` werfen nie; Fehler werden im Ergebnis vermerkt (Monitoring darf den Scanner-Job nie
            scheitern lassen).
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from modules import ops_monitor
from modules.atomic_io import append_jsonl, atomic_write_json, atomic_write_text, read_jsonl

log = logging.getLogger(__name__)

REPORT_SCHEMA, HISTORY_SCHEMA = "ops-report-v1", "ops-history-v1"
DEFAULT_OUT = Path("outputs/monitoring")
HISTORY_FILE, STATE_FILE = "ops_history.jsonl", "alert_state.json"
NA = "N/A"
BANNER = "READ-ONLY Monitoring – keine Handelsempfehlung"

SECTIONS = {"run": "Lauf", "quote_coverage": "Quote-Abdeckung", "lifecycle": "Lebenszyklus",
            "forward": "Forward-Performance", "training": "Training", "promotion": "Promotion",
            "integrity": "Integrität", "data_health": "Daten", "timing": "Timing-Policy"}
WORKFLOW_KIND = {"scanner": "SCANNER_FAILED", "feedback": "SCANNER_FAILED",
                 "external_data": "ARCHIVE_FAILED", "source_health": "ARCHIVE_FAILED"}
BAD_SOURCE = ("FAIL", "FAILED", "BROKEN", "STALE", "DEGRADED")
LEGACY_ENTRY = "ENTRY_NOT_CAPTURED_BEFORE_ARCHIVE"
# Schwellen
ENTRY_COV_MIN, ENTRY_DEN_MIN = 0.8, 3
EXIT_ACTIVE_COV_MIN = 0.8
TIMING_PENDING_MAX_DAYS = 3                 # timing-v2-Exit länger ausstehend -> Capture/Archiv prüfen
COMPLETE_COV_MIN, COMPLETE_DEN_MIN = 0.5, 5
MARK_OVERDUE_DAYS = 3
SOURCE_BAD_INCREASE = 2
ARCHIVE_DROP_RATIO, ARCHIVE_MEDIAN_MIN = 0.5, 20
SEV_ORDER = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
STATUS_ORDER = {"NEW": 0, "CHANGED": 1, "RESOLVED": 2, "ONGOING": 3}
META_KEYS = ("schema", "as_of", "generated")


# ── Hilfen ──────────────────────────────────────────────────────────────────
def _g(d, *keys):
    """Verschachtelter Zugriff; fehlender Pfad oder Nicht-Dict -> None (nie 0)."""
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _isnum(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _d(x) -> date | None:
    try:
        return date.fromisoformat(str(x)[:10]) if x else None
    except ValueError:
        return None


def _diff(a, b):
    return round(a - b, 6) if _isnum(a) and _isnum(b) else None


def _ok(status: dict, name: str) -> dict | None:
    sec = (status or {}).get(name)
    return sec if isinstance(sec, dict) and "error" not in sec else None


# ── 1. Historienzeile ───────────────────────────────────────────────────────
def history_row(status: dict) -> dict:
    """Flache Kennzahlen für Trends. Fehlendes bleibt None (nie 0)."""
    s = status or {}
    row: dict = {"schema": HISTORY_SCHEMA, "as_of": s.get("as_of"), "generated": s.get("generated")}
    for k in ("universe", "analyzed", "champion_candidates", "contracts_designed", "trades"):
        row[f"run.scanner.{k}"] = _g(s, "run", "scanner", k)
    row["run.report_is_today"] = _g(s, "run", "report_is_today")
    qc = s.get("quote_coverage")
    for w in ("7d", "30d"):
        for part, fields in (("entry", ("coverage", "denominator", "observed")),
                             ("exit_closed", ("coverage", "denominator", "observed")),
                             ("complete", ("coverage", "denominator", "evaluable"))):
            for f in fields:
                row[f"quote.{part}.{w}.{f}"] = _g(qc, "windows", w, part, f)
        by = _g(qc, "windows", w, "entry", "by_status")
        row[f"quote.entry.{w}.legacy"] = by.get(LEGACY_ENTRY, 0) if isinstance(by, dict) else None
    for f in ("coverage", "denominator", "mark_today"):
        row[f"quote.exit_active.{f}"] = _g(qc, "exit_active", f)
    for m in ("real_opt_ret_5d", "real_strat_ret_5d"):
        for f in ("pending", "oldest_pending_days"):
            row[f"marks.{m}.{f}"] = _g(qc, "candidate_marks", m, f)
    champ = _g(s, "lifecycle", "champion")
    by_state, issues = _g(champ, "by_state"), _g(champ, "issues")
    for st in ops_monitor.STATES:
        row[f"lifecycle.champion.state.{st}"] = _g(by_state, st)
    for iss in ("EXIT_OVERDUE", "ELIGIBLE_NOT_LEARNED", "LEARNED_NOT_ELIGIBLE"):
        row[f"lifecycle.champion.issue.{iss}"] = issues.get(iss, 0) if isinstance(issues, dict) else None
    row["lifecycle.champion.exactly_once_ok"] = _g(champ, "exactly_once", "ok")
    cand = _g(s, "lifecycle", "candidates")
    for k in ("EXIT_OVERDUE", "EXIT_PENDING", "OUTCOME_FINAL"):
        row[f"lifecycle.candidates.{k}"] = cand.get(k, 0) if isinstance(cand, dict) else None
    for k in ("n", "n_independent_days", "mean_net", "sum_net", "verdict"):
        row[f"forward.champion.{k}"] = _g(s, "forward", "champion", "observed", k)
    row["training.status"] = _g(s, "training", "status")
    row["training.n_labels"] = _g(s, "training", "n_labels")
    for k in ("active_influence", "unjustified_influence"):
        v = _g(s, "promotion", k)
        row[f"promotion.{k}"] = len(v) if isinstance(v, list) else None
    row["promotion.kill_switch_active"] = _g(s, "promotion", "kill_switch_active")
    iss = _g(s, "integrity", "issues")
    row["integrity.issues"] = len(iss) if isinstance(iss, list) else None
    by = _g(s, "data_health", "source_health", "by_status")
    ok_by = isinstance(by, dict) and bool(by)
    for st, n in sorted(by.items()) if ok_by else []:
        row[f"data.source_health.{st}"] = n
    row["data.source_health.n_bad"] = sum(by.get(x, 0) for x in BAD_SOURCE) if ok_by else None
    row["data.quote_archive_median_14d"] = _g(s, "data_health", "quote_archive_median_14d")
    days, exists = _g(s, "data_health", "quote_archive_days"), _g(s, "data_health", "quote_archive_root_exists")
    row["data.quote_archive_today"] = ((_g(days, s.get("as_of"), "n") or 0) if exists and isinstance(days, dict)
                                       else _g(days, s.get("as_of"), "n"))
    return row


# ── 2. Historie (append-only JSONL, ein Eintrag je Tag) ─────────────────────
def load_history(path) -> list[dict]:
    return [r for r in read_jsonl(path) if isinstance(r, dict)]


def append_history(path, row: dict, history: list[dict] | None = None) -> bool:
    """Hängt `row` an, falls für `row['as_of']` noch keine Zeile existiert (erste Zeile des Tages bleibt kanonisch).
    -> True wenn angehängt."""
    if not row.get("as_of"):
        return False
    history = load_history(path) if history is None else history
    if any(r.get("as_of") == row["as_of"] for r in history):
        return False
    append_jsonl(path, [row])
    return True


def reference_rows(row: dict, history: list[dict]) -> dict:
    """prev = jüngste Zeile vor as_of; d7/d30 = nächstliegende Zeile mit as_of <= as_of-7d bzw. -30d."""
    a = _d(row.get("as_of"))
    by_day: dict[date, dict] = {}
    for r in history or []:
        dd = _d(r.get("as_of"))
        if dd is not None and a is not None and dd < a:
            by_day.setdefault(dd, r)                             # erste Zeile je Tag
    def pick(limit):
        c = [dd for dd in by_day if dd <= limit]
        return by_day[max(c)] if c else None
    if a is None:
        return {"prev": None, "d7": None, "d30": None}
    return {"prev": pick(a - timedelta(days=1)), "d7": pick(a - timedelta(days=7)), "d30": pick(a - timedelta(days=30))}


def deltas(row: dict, history: list[dict]) -> dict:
    """je numerischer Kennzahl {value, prev, d7, d30, diff_prev, diff_d7, diff_d30}; jede Differenz mit None -> None."""
    ref = reference_rows(row, history)
    out = {}
    keys = dict.fromkeys(k for r in [row, *ref.values()] if r for k in r if k not in META_KEYS)
    for k in keys:
        vals = {n: (r.get(k) if r and _isnum(r.get(k)) else None) for n, r in ref.items()}
        if not (_isnum(row.get(k)) or any(v is not None for v in vals.values())):
            continue
        v = row.get(k) if _isnum(row.get(k)) else None
        out[k] = {"value": v, **vals, **{f"diff_{n}": _diff(v, x) for n, x in vals.items()}}
    return out


# ── 3. Warnungen ────────────────────────────────────────────────────────────
def _pct(v) -> str:
    return NA if not _isnum(v) else f"{v * 100:.1f} %"


def _frac(n, d) -> str:
    return f"{n}/{d}" if n is not None and d is not None else NA


def _rules(status: dict, row: dict, prev: dict | None) -> list[tuple]:
    """-> [(id, severity, component, cause, key)]; key = ursachenrelevante Werte für den Fingerabdruck."""
    out: list[tuple] = []

    def add(id_, sev, comp, cause, key=()):
        out.append((id_, sev, comp, cause, key))

    if status.get("collect_error"):
        add("MONITOR_COLLECT_FAILED", "WARNING", "Monitor",
            f"Datensammlung fehlgeschlagen ({status['collect_error']}); keine Regeln geprüft.", ("collect",))
    for name, label in SECTIONS.items():
        sec = status.get(name)
        if isinstance(sec, dict) and "error" in sec:
            add(f"MONITOR_SECTION_ERROR:{name}", "WARNING", "Monitor",
                f"Abschnitt „{label}“ ({name}) nicht berechenbar: {str(sec['error'])[:160]}; zugehörige Regeln übersprungen.",
                (name, str(sec["error"]).split(":")[0]))
    # Läufe
    run = _ok(status, "run")
    for job, w in sorted(((run or {}).get("workflows") or {}).items()):
        kind = WORKFLOW_KIND.get(job)
        if not kind or not isinstance(w, dict):
            continue
        state, concl = w.get("state"), w.get("conclusion")
        if state not in ("MISSED", "RECOVERY_EXHAUSTED") and concl != "failure":
            continue
        why = {"MISSED": "heute nicht ausgeführt (verpasst)",
               "RECOVERY_EXHAUSTED": "verpasst, Wiederholungsversuche ausgeschöpft"}.get(state, "fehlgeschlagen")
        hard = state == "RECOVERY_EXHAUSTED" or (concl == "failure" and state != "OK")
        add(f"{kind}:{job}", "CRITICAL" if kind == "SCANNER_FAILED" and hard else "WARNING", "Lauf",
            f"Workflow „{job}“ {why}: Status {state or NA}, Ergebnis {concl or NA}, letzter Lauf "
            f"{w.get('last_run') or NA}" + (" (Tag durch früheren Lauf trotzdem versorgt)" if state == "OK" else ""),
            (state, concl))
    failed = sorted(k for k, v in ((run or {}).get("current_job") or {}).items() if v == "failure")
    if failed:
        add("SCANNER_FAILED:current_job", "CRITICAL", "Lauf",
            f"Im aktuellen Scanner-Job fehlgeschlagen: {', '.join(failed)} (Schritt-Outcomes: "
            f"{(run or {}).get('current_job')}).", tuple(failed))
    # Quote-Abdeckung
    qc = _ok(status, "quote_coverage")
    e7 = _g(qc, "windows", "7d", "entry")
    if isinstance(e7, dict) and isinstance(e7.get("denominator"), int) and isinstance(e7.get("observed"), int):
        legacy = (e7.get("by_status") or {}).get(LEGACY_ENTRY, 0)
        eff = e7["denominator"] - legacy                          # Altbestand vor Archivstart ist nicht erfassbar
        if eff >= ENTRY_DEN_MIN and e7["observed"] / eff < ENTRY_COV_MIN:
            add("QUOTE_COVERAGE_LOW:entry_7d", "WARNING", "Quote-Abdeckung",
                f"Entry-Quoten (7 Tage): nur {_frac(e7['observed'], eff)} ({_pct(e7['observed'] / eff)}) mit "
                f"archivierter Quote, Schwelle {ENTRY_COV_MIN:.0%} (ohne {legacy} Altbestand vor Archivstart).")
    ea = _g(qc, "exit_active")
    if (isinstance(ea, dict) and ea.get("trading_day") and isinstance(ea.get("denominator"), int)
            and ea["denominator"] >= 1 and _isnum(ea.get("coverage")) and ea["coverage"] < EXIT_ACTIVE_COV_MIN):
        add("QUOTE_COVERAGE_LOW:exit_active", "WARNING", "Quote-Abdeckung",
            f"Exit-Marks aktiver Trades: nur {_frac(ea.get('mark_today'), ea['denominator'])} ({_pct(ea['coverage'])}) "
            f"mit Mark von heute, Schwelle {EXIT_ACTIVE_COV_MIN:.0%}; ältester Mark {ea.get('oldest_mark_age_days') or NA} Tage.")
    c30 = _g(qc, "windows", "30d", "complete")
    if (isinstance(c30, dict) and isinstance(c30.get("denominator"), int) and c30["denominator"] >= COMPLETE_DEN_MIN
            and _isnum(c30.get("coverage")) and c30["coverage"] < COMPLETE_COV_MIN):
        add("QUOTE_COVERAGE_LOW:complete_30d", "INFO", "Quote-Abdeckung",
            f"Vollständig bewertbare Outcomes (30 Tage): {_frac(c30.get('evaluable'), c30['denominator'])} "
            f"({_pct(c30['coverage'])}), Schwelle {COMPLETE_COV_MIN:.0%}. Enthält Altbestand vor Archivstart "
            "(aus Aggregaten nicht abtrennbar) – nur Hinweis.")
    # Outcomes überfällig
    lc = _ok(status, "lifecycle")
    ch = _g(lc, "champion")
    n_over = (_g(ch, "issues") or {}).get("EXIT_OVERDUE") or 0
    if n_over > 0:
        who = sorted(f"{b.get('ticker')} ({b.get('entry_date')})" for b in _g(ch, "blocked") or []
                     if "EXIT_OVERDUE" in (b.get("issues") or []))
        add("OUTCOMES_OVERDUE:champion", "WARNING", "Lebenszyklus",
            f"{n_over} aktive Champion-Trade(s) länger als Haltedauer + Karenz offen: {', '.join(who) or NA}.", (n_over, who))
    n_cand = (_g(lc, "candidates") or {}).get("EXIT_OVERDUE") or 0
    if n_cand > 0:
        add("OUTCOMES_OVERDUE:candidates", "WARNING", "Lebenszyklus",
            f"{n_cand} Kandidat(en) mit überfälligem 5-Tage-Real-Option-Mark "
            f"(> {ops_monitor.MARK_OVERDUE_DAYS} Tage nach Fälligkeit).", (n_cand,))
    late = [(m, v) for m, v in sorted((_g(qc, "candidate_marks") or {}).items())
            if isinstance(v, dict) and _isnum(v.get("oldest_pending_days")) and v["oldest_pending_days"] > MARK_OVERDUE_DAYS]
    if late:
        add("OUTCOMES_OVERDUE:marks", "WARNING", "Quote-Abdeckung",
            "Fällige Marks offen: " + "; ".join(f"{m} {v.get('pending')} von {v.get('due')} offen, ältester "
                                                f"{v['oldest_pending_days']} Tage" for m, v in late)
            + f" (Schwelle > {MARK_OVERDUE_DAYS} Tage).", [m for m, _ in late])
    # Lernen exactly-once
    eo = _g(ch, "exactly_once")
    if isinstance(eo, dict):
        lne = eo.get("learned_not_eligible") or 0
        if eo.get("ok") is False or lne > 0:
            add("LEARNING_EXACTLY_ONCE", "CRITICAL", "Lebenszyklus",
                f"Training trotz unzulässiger Daten bzw. Exactly-once verletzt: {lne} gelernte, nicht lernbare Trade(s), "
                f"{eo.get('duplicate_learnable_trades') or 0} doppelte lernbare Trade(s), ok={eo.get('ok')}.",
                (eo.get("ok"), lne, eo.get("duplicate_learnable_trades") or 0))
        if (eo.get("eligible_not_learned") or 0) > 0:
            add("LEARNING_EXACTLY_ONCE:eligible_not_learned", "WARNING", "Lebenszyklus",
                f"{eo['eligible_not_learned']} lernbare, geschlossene Trade(s) wurden nicht gelernt.",
                (eo["eligible_not_learned"],))
    # Integrität
    ig = _ok(status, "integrity")
    if ig and ig.get("issues"):
        add("LEDGER_INTEGRITY", "CRITICAL", "Integrität",
            "Ledger-/Hash-Integrität verletzt: " + "; ".join(str(i) for i in ig["issues"]) + ".", sorted(map(str, ig["issues"])))
    # Training
    tr = _ok(status, "training")
    if tr and tr.get("manifest_violations"):
        mv = [f"{m.get('version')} ({', '.join(m.get('violations') or [])})" for m in tr["manifest_violations"]]
        add("TRAINING_INADMISSIBLE:manifest", "CRITICAL", "Training",
            "Modellversion(en) mit unzulässigem Manifest (Label-Leakage oder unter Readiness trainiert): " + "; ".join(mv),
            mv)
    if tr and tr.get("active_artifact_ok") is False:
        add("TRAINING_INADMISSIBLE:active_artifact", "CRITICAL", "Training",
            f"Aktives Modell-Artefakt {tr.get('active') or NA} nicht ladbar bzw. Prüfung fehlgeschlagen.", (tr.get("active"),))
    # Promotion
    pr = _ok(status, "promotion")
    if pr and pr.get("unjustified_influence"):
        add("PROMOTION_UNJUSTIFIED", "CRITICAL", "Promotion",
            "Hypothese(n) mit Einfluss ohne Rechtfertigung (Stufe/Integrität): " + ", ".join(pr["unjustified_influence"]),
            sorted(pr["unjustified_influence"]))
    if pr and pr.get("kill_switch_active") is True:
        add("PROMOTION_UNJUSTIFIED:kill_switch", "WARNING", "Promotion",
            "Kill-Switch aktiv – automatischer Einfluss gestoppt (informativ).")
    # Datenqualität
    dh = _ok(status, "data_health")
    if dh:
        nb, pb = row.get("data.source_health.n_bad"), (prev or {}).get("data.source_health.n_bad")
        if _isnum(nb) and _isnum(pb) and nb - pb >= SOURCE_BAD_INCREASE:
            add("DATA_DEGRADATION:source_health", "WARNING", "Daten",
                f"Quellen mit Status {'/'.join(BAD_SOURCE[1:])}: {nb} (Vorlauf {pb}, +{nb - pb}).", (nb,))
        med, today = dh.get("quote_archive_median_14d"), row.get("data.quote_archive_today")
        a = _d(status.get("as_of"))
        sc_state = _g(run, "workflows", "scanner", "state")
        if (_isnum(med) and med >= ARCHIVE_MEDIAN_MIN and _isnum(today) and today < ARCHIVE_DROP_RATIO * med
                and a is not None and a.weekday() < 5 and sc_state not in ("PENDING", "RUNNING", "NOT_SCHEDULED")):
            add("DATA_DEGRADATION:quote_archive", "WARNING", "Daten",
                f"Quote-Archiv heute {today} Quotes, unter {ARCHIVE_DROP_RATIO:.0%} des 14-Tage-Medians ({med:g}).")
    # Timing-Policy (timing-v2): ausstehende Exits ohne zulässige Quote, endgültig unbrauchbare Outcomes
    tm = _ok(status, "timing")
    old_days = _g(tm, "v2_exit_pending_oldest_days")
    if _isnum(old_days) and old_days > TIMING_PENDING_MAX_DAYS:
        add("TIMING_EXIT_PENDING_OVERDUE", "WARNING", "Timing",
            f"{_g(tm, 'v2_exit_pending')} timing-v2-Exit(s) ohne zulässige Optionsquote, ältester seit {old_days:g} Tagen "
            f"(Gründe {_f(_g(tm, 'v2_exit_pending_reasons'))}); Capture/Archiv prüfen.", (_g(tm, "v2_exit_pending"),))
    nu = _g(tm, "v2_unusable")
    if _isnum(nu) and nu > 0:
        add("TIMING_UNUSABLE", "WARNING", "Timing",
            f"{nu} timing-v2-Trade(s) ohne zulässigen Preis endgültig UNUSABLE (Gründe "
            f"{_f(_g(tm, 'v2_unusable_reasons'))}); kein Outcome, nicht gelernt.", (nu,))
    return out


def _fingerprint(id_: str, key) -> str:
    return hashlib.sha256(json.dumps([id_, key], sort_keys=True, default=str, ensure_ascii=False).encode()).hexdigest()[:16]


_OWNER = {"SCANNER_FAILED": "run", "ARCHIVE_FAILED": "run", "QUOTE_COVERAGE_LOW": "quote_coverage",
          "OUTCOMES_OVERDUE": "lifecycle", "LEARNING_EXACTLY_ONCE": "lifecycle", "LEDGER_INTEGRITY": "integrity",
          "TRAINING_INADMISSIBLE": "training", "PROMOTION_UNJUSTIFIED": "promotion", "DATA_DEGRADATION": "data_health",
          "TIMING_EXIT_PENDING_OVERDUE": "timing", "TIMING_UNUSABLE": "timing"}


def _owner(id_: str) -> str | None:
    """Abschnitt, aus dem eine Regel ihre Daten liest (bei Abschnittsfehler bleibt ihr Zustand unverändert)."""
    return "quote_coverage" if id_ == "OUTCOMES_OVERDUE:marks" else _OWNER.get(id_.split(":")[0])


def alerts(status: dict, row: dict, history: list[dict], state: dict | None) -> tuple[list[dict], dict]:
    """-> (Warnungen mit Status NEW|CHANGED|ONGOING|RESOLVED, neuer Zustand). `state` wird nicht verändert."""
    state, as_of = dict(state or {}), str(status.get("as_of") or row.get("as_of") or "")
    prev = reference_rows(row, history)["prev"]
    out, new_state, seen = [], {}, set()
    for id_, sev, comp, cause, key in _rules(status, row, prev):
        fp, old = _fingerprint(id_, key), state.get(id_)
        st = "NEW" if not old else "ONGOING" if old.get("fingerprint") == fp else "CHANGED"
        first = (old or {}).get("first_seen") or as_of
        new_state[id_] = {"fingerprint": fp, "first_seen": first, "last_seen": as_of, "severity": sev, "component": comp}
        out.append({"id": id_, "severity": sev, "component": comp, "cause": cause, "since": first, "status": st})
        seen.add(id_)
    blind = set(SECTIONS) if status.get("collect_error") else {n for n in SECTIONS if not _ok(status, n)}
    for id_, old in sorted(state.items()):
        if id_ not in seen:
            if _owner(id_) in blind:                       # Abschnitt nicht berechenbar -> weder behoben noch neu
                new_state[id_] = old
                continue
            out.append({"id": id_, "severity": old.get("severity") or "WARNING", "component": old.get("component") or NA,
                        "cause": "Nicht mehr aktiv – Bedingung behoben.", "since": old.get("first_seen") or as_of,
                        "status": "RESOLVED"})
    out.sort(key=lambda a: (STATUS_ORDER[a["status"]], SEV_ORDER.get(a["severity"], 9), a["id"]))
    return out, new_state


# ── 4. Rendering ────────────────────────────────────────────────────────────
def _f(v, nd: int = 4) -> str:
    if v is None:
        return NA
    if isinstance(v, bool):
        return "ja" if v else "nein"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        s = f"{v:.{nd}f}".rstrip("0").rstrip(".")
        return "0" if s in ("", "-0") else s
    if isinstance(v, (list, tuple)):
        return ", ".join(_f(x) for x in v) if v else "keine"
    if isinstance(v, dict):
        return ", ".join(f"{k}: {_f(x)}" for k, x in v.items()) if v else "keine"
    return str(v)


def _sf(v) -> str:
    return NA if v is None else ("+" + _f(v) if v > 0 else _f(v))


def _counts(a: list[dict]) -> dict:
    c = {s: sum(1 for x in a if x["status"] == s) for s in STATUS_ORDER}
    c["CRITICAL"] = sum(1 for x in a if x["severity"] == "CRITICAL" and x["status"] != "RESOLVED")
    return c


def _document(status: dict, row: dict, d: dict, al: list[dict]) -> tuple[str, list[str], list[tuple]]:
    s = status or {}
    c = _counts(al)
    head = [BANNER, f"Stand {s.get('as_of') or NA} · erzeugt {s.get('generated') or NA} · Schema {s.get('schema') or NA}",
            f"Warnungen: {c['NEW']} neu, {c['CHANGED']} geändert, {c['ONGOING']} laufend, {c['RESOLVED']} behoben; "
            f"davon aktiv CRITICAL: {c['CRITICAL']}"]
    secs: list[tuple] = []

    def sec(name, fn):
        d_ = s.get(name)
        if not isinstance(d_, dict):
            return [("p", f"{NA} – Abschnitt fehlt.")]
        if "error" in d_:
            return [("p", f"{NA} – Abschnitt nicht berechenbar: {d_['error']}")]
        return fn(d_)

    def run_blocks(r):
        sc = r.get("scanner") or {}
        wf = [[j, w.get("state"), w.get("conclusion"), w.get("last_run"), w.get("delay_hours")]
              for j, w in sorted((r.get("workflows") or {}).items())]
        b = [("table", ["Workflow", "Status", "Ergebnis", "Letzter Lauf", "Verzug (h)"], wf)] if wf else \
            [("p", f"Workflow-Status für heute: {NA}")]
        return b + [("kv", [("Tagesreport", f"{r.get('report_date') or NA} (von heute: {_f(r.get('report_is_today'))})"),
                            ("Universum / analysiert", f"{_f(sc.get('universe'))} / {_f(sc.get('analyzed'))}"),
                            ("Champion-Kandidaten (Final-MC)", _f(sc.get("champion_candidates"))),
                            ("Kontrakte entworfen", _f(sc.get("contracts_designed"))), ("Trades", _f(sc.get("trades"))),
                            ("Session", _f(sc.get("session"))), ("Stop-Grund", _f(sc.get("stop_reason")))])]

    def quote_blocks(q):
        rows = []
        for w in ("7d", "30d"):
            e = _g(q, "windows", w, "entry") or {}
            lg = (e.get("by_status") or {}).get(LEGACY_ENTRY, 0) if e else None
            eff = e["denominator"] - lg if lg is not None and e.get("denominator") is not None else None
            adj = e["observed"] / eff if eff and e.get("observed") is not None else None
            rows.append([f"Entry {w}", _pct(e.get("coverage")), _frac(e.get("observed"), e.get("denominator")),
                         f"Altbestand vor Archiv: {_f(lg)}; ohne Altbestand {_pct(adj)} ({_frac(e.get('observed'), eff)})"])
        for w in ("7d", "30d"):
            x = _g(q, "windows", w, "exit_closed") or {}
            rows.append([f"Exit geschlossen {w}", _pct(x.get("coverage")), _frac(x.get("observed"), x.get("denominator")),
                         _f(x.get("by_status"))])
        for w in ("7d", "30d"):
            x = _g(q, "windows", w, "complete") or {}
            rows.append([f"Vollständig bewertbar {w}", _pct(x.get("coverage")), _frac(x.get("evaluable"), x.get("denominator")),
                         _f(x.get("reasons"))])
        ea = q.get("exit_active") or {}
        rows.append(["Exit aktiv (Mark heute)", _pct(ea.get("coverage")), _frac(ea.get("mark_today"), ea.get("denominator")),
                     f"Handelstag: {_f(ea.get('trading_day'))}; ältester Mark: {_f(ea.get('oldest_mark_age_days'))} Tage"])
        cm = [[m, v.get("due"), v.get("marked"), v.get("pending"), v.get("late_marks"), v.get("oldest_pending_days")]
              for m, v in sorted((q.get("candidate_marks") or {}).items()) if isinstance(v, dict)]
        return [("table", ["Kennzahl", "Abdeckung", "Beobachtet/Nenner", "Hinweis"], rows),
                ("table", ["Kandidaten-Mark", "fällig", "gemarkt", "offen", "spät", "ältester offener (Tage)"], cm)]

    def lifecycle_blocks(lc):
        ch = lc.get("champion") or {}
        eo = ch.get("exactly_once") or {}
        return [("table", ["Champion-Zustand", "Anzahl"], [[k, v] for k, v in (ch.get("by_state") or {}).items()]),
                ("kv", [("Probleme (Champion)", _f(ch.get("issues"))),
                        ("Exactly-once ok", _f(eo.get("ok"))),
                        ("gelernt/doppelt/nicht lernbar gelernt/lernbar ungelernt",
                         f"{_f(eo.get('learned_keys'))}/{_f(eo.get('duplicate_learnable_trades'))}/"
                         f"{_f(eo.get('learned_not_eligible'))}/{_f(eo.get('eligible_not_learned'))}"),
                        ("Altbestand", _f((ch.get("legacy") or {}).get("note"))),
                        ("Kandidaten (Real-Option)", _f(lc.get("candidates")))])]

    def forward_blocks(fw):
        o = _g(fw, "champion", "observed") or {}
        v = o.get("verdict")
        claim = (f"Verdikt {_f(v)}: keine Alpha-Behauptung (n={_f(o.get('n'))}, unabhängige Tage={_f(o.get('n_independent_days'))}; "
                 f"Mindestwerte n≥{ops_monitor.MIN_N}, Tage≥{ops_monitor.MIN_DAYS})." if v in (None, "NEED_MORE_DATA", "NO_DATA")
                 else f"Verdikt {v} (Bootstrap-Intervall über unabhängige Tage, nur beobachtete Outcomes) – keine Handelsempfehlung.")
        ch = [[k, x.get("avoided_losses"), x.get("foregone_gains"), x.get("incremental_vs_champion_sum"),
               f"{_f(x.get('would_act_on'))}/{_f(x.get('evaluable_resolved'))}", x.get("population"), x.get("verdict")]
              for k, x in sorted((fw.get("challengers") or {}).items())]
        nmd = sum(1 for r in ch if r[-1] == "NEED_MORE_DATA")
        sc = [[k, x.get("n"), x.get("mean_net"), x.get("hit_rate"), x.get("verdict")]
              for k, x in sorted((fw.get("scorecard") or {}).items()) if isinstance(x, dict)]
        return [("kv", [("Champion realisiert (nur beobachtet): n", _f(o.get("n"))), ("mean_net", _f(o.get("mean_net"))),
                        ("sum_net", _f(o.get("sum_net"))), ("Trefferquote", _f(o.get("hit_rate"))),
                        ("schlechtester", _f(o.get("worst"))), ("max. Drawdown (kum.)", _f(o.get("max_drawdown_cum"))),
                        ("nicht bewertbar", _f(_g(fw, "champion", "not_evaluable")))]), ("p", claim),
                ("table", ["Challenger (Shadow)", "vermiedene Verluste", "entgangene Gewinne", "Δ vs. Champion (Σ)",
                           "greift/bewertbar", "Population", "Verdikt"], ch),
                ("p", f"{nmd} von {len(ch)} Challenger-Verdikten NEED_MORE_DATA – dort keine Alpha-Behauptung."),
                ("table", ["Scorecard", "n", "mean_net", "Trefferquote", "Verdikt"], sc)]

    def training_blocks(t):
        return [("kv", [("Status", _f(t.get("status"))), ("Labels / Tage (benötigt)",
                                                          f"{_f(t.get('n_labels'))} / {_f(t.get('n_label_dates'))} ({_f(t.get('required'))})"),
                        ("Versionen / aktiv / Hold", f"{_f(t.get('versions'))} / {_f(t.get('active'))} / {_f(t.get('hold'))}"),
                        ("Aktives Artefakt ok", _f(t.get("active_artifact_ok"))),
                        ("Manifest-Verstöße", _f(t.get("manifest_violations")))])]

    def promotion_blocks(p):
        hy = [[k, h.get("state"), h.get("influence_level"), h.get("decision"), (h.get("blocking") or NA)[:160]]
              for k, h in sorted((p.get("hypotheses") or {}).items())]
        return [("kv", [("Max. automatischer Einfluss", _f(p.get("max_automatic_influence"))),
                        ("Kill-Switch / Safe-Mode", f"{_f(p.get('kill_switch_active'))} / {_f(p.get('safe_mode'))}"),
                        ("Champion-Version", _f(p.get("champion_version"))), ("Zustände", _f(p.get("by_state"))),
                        ("Aktiver Einfluss", _f(p.get("active_influence"))),
                        ("Ungerechtfertigter Einfluss", _f(p.get("unjustified_influence")))]),
                ("table", ["Hypothese", "Zustand", "Einfluss", "Entscheidung", "Blockierungsgründe"], hy)]

    def integrity_blocks(i):
        return [("kv", [("Probleme", _f(i.get("issues"))), ("Promotion-Hash ok", _f(i.get("promotion_state_hash_ok"))),
                        ("Kette/Registry", f"{_f(i.get('transition_chain_problems'))} / {_f(i.get('registry_problems'))}"),
                        ("Duplikate Decision-Ledger / Trade-Outcomes",
                         f"{_f(i.get('decision_ledger_duplicates'))} / {_f(i.get('trade_outcome_duplicates'))}"),
                        ("Quote-Archiv 30 T: Records / doppelte IDs / Fremdpreise",
                         f"{_f(i.get('quote_records_30d'))} / {_f(i.get('quote_duplicate_ids'))} / {_f(i.get('quote_non_observed_prices'))}"),
                        ("Quarantäne geschlossener Trades", _f(i.get("closed_trade_quarantine"))),
                        ("Externes Archiv", _f(i.get("external_archive")))])]

    def data_blocks(h):
        return [("kv", [("Quellenstatus", _f(_g(h, "source_health", "by_status"))),
                        ("Stand Quellenstatus", _f(_g(h, "source_health", "generated"))),
                        ("Externe Daten", _f(_g(h, "external_data", "by_status"))),
                        ("Quote-Archiv heute / Median 14 T", f"{_f(row.get('data.quote_archive_today'))} / {_f(h.get('quote_archive_median_14d'))}"),
                        ("Quote-Archiv-Verzeichnis vorhanden", _f(h.get("quote_archive_root_exists")))])]

    def timing_blocks(t):
        return [("kv", [("Policy neuer Entscheidungen", _f(t.get("current_policy"))),
                        ("Bestand je Policy", _f(t.get("by_policy"))),
                        ("timing-v2 Entry offener Trades", _f(t.get("v2_active_entry"))),
                        ("timing-v2 Exit ausstehend (ältester, Tage)",
                         f"{_f(t.get('v2_exit_pending'))} ({_f(t.get('v2_exit_pending_oldest_days'))})"),
                        ("Gründe ausstehend", _f(t.get("v2_exit_pending_reasons"))),
                        ("timing-v2 ausgeführt / UNUSABLE", f"{_f(t.get('v2_closed'))} / {_f(t.get('v2_unusable'))}"),
                        ("Gründe UNUSABLE", _f(t.get("v2_unusable_reasons"))),
                        ("Abweichung Ausführung vs. Entscheid", _f(t.get("v2_deviation_vs_decision")))])]

    builders = {"run": run_blocks, "quote_coverage": quote_blocks, "lifecycle": lifecycle_blocks,
                "forward": forward_blocks, "training": training_blocks, "promotion": promotion_blocks,
                "integrity": integrity_blocks, "data_health": data_blocks, "timing": timing_blocks}
    for name, label in SECTIONS.items():
        secs.append((label, sec(name, builders[name])))
    ar = [[a["status"], a["severity"], a["component"], a["id"], a["cause"], a["since"]] for a in al]
    secs.append(("Warnungen", [("table", ["Status", "Schwere", "Komponente", "ID", "Ursache", "seit"], ar,
                                [a["severity"] for a in al])] if al else [("p", "Keine aktiven Warnungen.")]))
    def cmp(v, ref):
        return NA if v[ref] is None else f"{_f(v[ref])} ({_sf(v['diff_' + ref])})"

    tr = [[k, v["value"], cmp(v, "prev"), cmp(v, "d7"), cmp(v, "d30")] for k, v in sorted((d or {}).items())]
    miss = sorted(k for k, v in (row or {}).items() if v is None and k not in (d or {}))
    secs.append(("Trends", ([("table", ["Kennzahl", "Wert", "Vorlauf (Δ)", "≥7 T zurück (Δ)", "≥30 T zurück (Δ)"], tr)]
                             if tr else [("p", f"Keine numerischen Kennzahlen: {NA}")])
                 + ([("p", f"{len(miss)} Kennzahl(en) ohne Wert (N/A, nie 0): {', '.join(miss)}")] if miss else [])))
    return "Scanner-Monitor", head, secs


def _cell(v) -> str:
    return _f(v).replace("|", "\\|").replace("\n", " ")


def render_markdown(status: dict, row: dict, d: dict, alerts: list[dict]) -> str:
    title, head, secs = _document(status, row, d, alerts)
    out = [f"# {title}", "", f"**{head[0]}**", ""] + [f"{h}  " for h in head[1:]] + [""]
    for name, blocks in secs:
        out += [f"## {name}", ""]
        for b in blocks:
            if b[0] == "p":
                out += [b[1], ""]
            elif b[0] == "kv":
                out += [f"- **{k}**: {_f(v)}" for k, v in b[1]] + [""]
            elif b[0] == "table":
                out += ["| " + " | ".join(b[1]) + " |", "|" + "---|" * len(b[1])]
                out += ["| " + " | ".join(_cell(x) for x in r) + " |" for r in b[2]] + [""]
    return "\n".join(out).rstrip() + "\n"


_TONE = {"CRITICAL": "#b00020", "WARNING": "#9a5b00", "INFO": "#555555"}
_TABLE = '<table border="1" cellpadding="3" cellspacing="0" style="border-collapse:collapse;border-color:#cccccc;margin:4px 0">'


def render_html(status: dict, row: dict, d: dict, alerts: list[dict]) -> str:
    e = html.escape
    title, head, secs = _document(status, row, d, alerts)
    p = ['<div style="font-family:Arial,Helvetica,sans-serif;font-size:13px;color:#111111;background:#ffffff;max-width:1000px">',
         f'<h1 style="font-size:20px;margin:0 0 4px">{e(title)}</h1>',
         f'<p style="margin:0 0 4px;color:#b00020"><b>{e(head[0])}</b></p>'] + \
        [f'<p style="margin:0 0 4px;color:#444444">{e(h)}</p>' for h in head[1:]]
    for name, blocks in secs:
        p.append(f'<h2 style="font-size:16px;margin:16px 0 6px;border-bottom:1px solid #cccccc">{e(name)}</h2>')
        for b in blocks:
            if b[0] == "p":
                p.append(f'<p style="margin:4px 0">{e(b[1])}</p>')
            elif b[0] == "kv":
                p.append(_TABLE + "".join(f'<tr><th align="left" valign="top">{e(k)}</th><td valign="top">{e(_f(v))}</td></tr>'
                                          for k, v in b[1]) + "</table>")
            elif b[0] == "table":
                tones = b[3] if len(b) > 3 else [None] * len(b[2])
                rows = "".join(f'<tr style="color:{_TONE.get(t, "#111111")}">' + "".join(
                    f'<td valign="top">{e(_f(x))}</td>' for x in r) + "</tr>" for r, t in zip(b[2], tones))
                p.append(_TABLE + '<tr bgcolor="#eeeeee">' + "".join(f'<th align="left">{e(h)}</th>' for h in b[1])
                         + f"</tr>{rows}</table>")
    p.append("</div>")
    return ('<!doctype html><html lang="de"><head><meta charset="utf-8"><title>Scanner-Monitor</title></head><body>'
            + "\n".join(p) + "</body></html>")


# ── 5. Lauf ─────────────────────────────────────────────────────────────────
def mail_subject(as_of: str, al: list[dict]) -> str:
    c = _counts(al)
    s = f"[Scanner-Monitor] {as_of}: {c['NEW'] + c['CHANGED']} neue Warnung(en)"
    return s + (f", {c['RESOLVED']} behoben" if c["RESOLVED"] else "") + (f", {c['CRITICAL']} KRITISCH" if c["CRITICAL"] else "")


def _should_mail(al: list[dict]) -> bool:
    """Nur Statusänderungen werden gemailt – keine täglich wiederholten identischen Alarme (auch nicht CRITICAL)."""
    return any(a["status"] in ("NEW", "CHANGED", "RESOLVED") for a in al)


def step_outcomes(env: dict | None = None) -> dict:
    """OPS_STEP_OUTCOMES="rth_marks=success scanner=failure …" (GitHub steps.<id>.outcome) -> {schritt: outcome}."""
    import os
    raw = (env if env is not None else os.environ).get("OPS_STEP_OUTCOMES") or ""
    return {k: v for k, _, v in (x.partition("=") for x in raw.split()) if k and v}


def _load_state(path: Path) -> dict:
    try:
        v = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        return v if isinstance(v, dict) else {}
    except (OSError, ValueError):
        return {}


def run(now: datetime | None = None, *, paths: dict | None = None, out_dir: Path | str = DEFAULT_OUT,
        mail: bool = False, dry_run_mail: bool = False, env: dict | None = None,
        history_path: Path | str | None = None, state_path: Path | str | None = None) -> dict:
    """collect -> Zeile -> Deltas -> Warnungen -> Dateien -> Historie -> (Mail). Wirft nie; Fehler stehen in `errors`.
    Schreibt nur nach out_dir (Historie/Zustand dort, sofern keine Pfade übergeben). Dry-Run-Mail ändert den
    Warnungszustand nicht."""
    errors: list[str] = []
    out_dir = Path(out_dir)
    hist_p, state_p = Path(history_path or out_dir / HISTORY_FILE), Path(state_path or out_dir / STATE_FILE)
    now = now or datetime.now(timezone.utc)
    res: dict = {"ok": True, "as_of": now.date().isoformat(), "out_dir": str(out_dir), "files": [], "errors": errors,
                 "history": {"appended": False, "reason": None}, "mail": {"status": "skipped"}, "alerts": []}
    try:
        status = ops_monitor.collect(now=now, paths=paths)
    except Exception as e:  # noqa: BLE001 – Monitoring darf den Job nie scheitern lassen
        errors.append(f"collect: {type(e).__name__}: {e}")
        status = {"schema": ops_monitor.SCHEMA, "generated": now.isoformat(timespec="seconds"),
                  "as_of": now.date().isoformat(), "read_only": True, "collect_error": f"{type(e).__name__}: {e}"[:200]}
    res["as_of"] = status.get("as_of") or res["as_of"]
    so = step_outcomes(env)
    if so and isinstance(status.get("run"), dict) and "error" not in status["run"]:
        status["run"]["current_job"] = so
    try:
        row = history_row(status)
        try:
            history = load_history(hist_p)
        except Exception as e:  # noqa: BLE001 – defekte Historie: ohne Trends weiter, nichts anhängen
            errors.append(f"history: {type(e).__name__}: {e}")
            history, res["history"]["reason"] = [], "history_unreadable"
        d = deltas(row, history)
        al, new_state = alerts(status, row, history, _load_state(state_p))
        res["alerts"] = al
        subject, md, page = mail_subject(res["as_of"], al), render_markdown(status, row, d, al), render_html(status, row, d, al)
        if (mail or dry_run_mail) and _should_mail(al):
            try:
                from modules import mailer
                res["mail"] = dict(mailer.send_mail(subject, page, md, dry_run=bool(dry_run_mail), env=env))
            except Exception as e:  # noqa: BLE001 – send_mail wirft nie; Absicherung
                res["mail"] = {"status": "failed", "error": f"{type(e).__name__}: {e}"[:200]}
        elif mail or dry_run_mail:
            res["mail"] = {"status": "skipped", "reason": "keine neuen/geänderten/behobenen oder kritischen Warnungen"}
        res["mail"]["subject"] = subject
        if status.get("collect_error"):
            res["history"]["reason"] = "collect_failed"
        elif res["history"]["reason"] is None:
            try:
                res["history"]["appended"] = append_history(hist_p, row, history)
                res["history"]["reason"] = None if res["history"]["appended"] else "as_of_exists"
            except Exception as e:  # noqa: BLE001
                errors.append(f"history append: {type(e).__name__}: {e}")
        doc = {"schema": REPORT_SCHEMA, "as_of": res["as_of"], "generated": status.get("generated"), "read_only": True,
               "status": status, "row": row, "deltas": d, "delta_refs": {k: (r or {}).get("as_of")
                                                                         for k, r in reference_rows(row, history).items()},
               "alerts": al, "mail": res["mail"], "history": res["history"], "errors": errors}
        for name, write, payload in (("ops_status.json", atomic_write_json, doc), ("ops_status.md", atomic_write_text, md),
                                     ("ops_status.html", atomic_write_text, page)):
            try:
                kw = {"indent": 1, "ensure_ascii": False} if name.endswith(".json") else {}
                write(out_dir / name, payload, **kw)
                res["files"].append(str(out_dir / name))
            except Exception as e:  # noqa: BLE001
                errors.append(f"write {name}: {type(e).__name__}: {e}")
        if not dry_run_mail and not (mail and res["mail"].get("status") == "failed"):
            try:                                   # bei fehlgeschlagener Mail Zustand zurückhalten -> nächster Lauf meldet erneut
                atomic_write_json(state_p, new_state, indent=1)
            except Exception as e:  # noqa: BLE001
                errors.append(f"state: {type(e).__name__}: {e}")
    except Exception as e:  # noqa: BLE001
        errors.append(f"report: {type(e).__name__}: {e}")
    res["ok"] = not errors
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m modules.ops_report", description=f"Ops-Bericht – {BANNER}")
    ap.add_argument("--mail", action="store_true", help="Mail senden (nur bei neuen/geänderten/behobenen Warnungen; laufende nie erneut)")
    ap.add_argument("--dry-run-mail", action="store_true", help="Mail nur vorbereiten (kein Versand, Warnungszustand unverändert)")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="Ausgabeverzeichnis (Default outputs/monitoring)")
    try:
        args = ap.parse_args(argv)
        res = run(out_dir=Path(args.out), mail=args.mail, dry_run_mail=args.dry_run_mail)
    except SystemExit:
        print("::warning::ops_report: ungültige Argumente – kein Lauf")
        return 0
    except Exception as e:  # noqa: BLE001 – Exit-Code bleibt 0
        print(f"::warning::ops_report: {type(e).__name__}: {e}")
        return 0
    al, c = res["alerts"], _counts(res["alerts"])
    print(f"Scanner-Monitor {res['as_of']} – {BANNER}")
    print(f"Warnungen: {c['NEW']} NEW, {c['CHANGED']} CHANGED, {c['ONGOING']} ONGOING, {c['RESOLVED']} RESOLVED "
          f"(aktiv CRITICAL: {c['CRITICAL']})")
    for a in al:
        print(f"  [{a['severity']}] {a['status']} {a['id']}: {a['cause']}")
    print(f"Historie: {'angehängt' if res['history']['appended'] else 'nicht angehängt (' + str(res['history']['reason']) + ')'}"
          f" · Mail: {res['mail'].get('status')} · Dateien: {len(res['files'])} in {res['out_dir']}")
    for a in al:
        if a["severity"] == "CRITICAL" and a["status"] != "RESOLVED":
            print(f"::warning::[CRITICAL] {a['id']}: {a['cause']}".replace("\n", " "))
    for err in res["errors"]:
        print(f"::warning::ops_report Fehler: {err}".replace("\n", " "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

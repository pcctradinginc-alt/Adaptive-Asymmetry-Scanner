"""modules/ops_monitor.py – read-only Betriebs- und Forward-Alpha-Monitoring (2026-10-11).

Liest AUSSCHLIESSLICH vorhandene Ledger, Archive und Zustandsdateien und schreibt nichts außer seinem eigenen
Bericht (Bericht, Warnungen und Mail: modules/ops_report.py). Es verändert keine Trades, Modelle, Verträge oder
Konfigurationen.

`collect()` liefert ein Dict (Schema `ops-status-v1`) mit diesen Abschnitten:
  run              Scanner-/Archiv-Läufe (workflow_status.json) und Tagesreport-Kennzahlen (daily_reports)
  quote_coverage   Entry-, Exit- und vollständige Outcome-Abdeckung der Champion-Trades je Zeitraum, Provider und
                   Strategie, mit Nennern und Ausschlussgründen; Real-Options-Marks der Kandidaten
  lifecycle        SIGNAL → CONTRACT_SELECTED → ENTRY_OBSERVED → EXIT_PENDING → OUTCOME_FINAL → LEARNING_ELIGIBLE
                   → LEARNED, je Trade mit Problemen (Issues) und Exactly-once-Prüfung des Lernens
  forward          Champion (realisiert, nur beobachtete Options-Outcomes) gegen Shadow-Challenger auf derselben
                   eingefrorenen Population (Decision-Ledger), inkl. Wert vermiedener Trades und entgangener Gewinne
  training         Meta-Label-Lernstand (Readiness, Versionen, Manifest-Prüfungen)
  promotion        Stufen, Blockierungsgründe, Kill-Switch, automatische Obergrenze
  integrity        Hashes, Ketten, Duplikate, unzulässiges Lernen
  data_health      Quellenstatus und Quote-Archiv-Volumen

Fehlende Kennzahlen sind None (Bericht: N/A), nie 0. Ein Abschnittsfehler wird als {"error": …} gemeldet und bricht
die übrigen Abschnitte nicht ab.
"""
from __future__ import annotations

import json
import logging
import math
import statistics
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from modules.atomic_io import read_jsonl

log = logging.getLogger(__name__)

SCHEMA = "ops-status-v1"
ARCHIVE_START = date(2026, 10, 10)          # Quote-Archiv-Hooks (option-quote-v1) existieren ab diesem Tag
CONSUME_ONCE_START = date(2026, 10, 10)     # learning_ledger (Consume-once-Schlüssel) ab diesem Tag (#108)
EXIT_OVERDUE_GRACE_DAYS = 2                 # aktiver Trade länger als close_after_days + Karenz -> überfällig
MARK_OVERDUE_DAYS = 2                       # fälliger Real-Options-Mark länger offen -> überfällig
MIN_N, MIN_DAYS = 30, 20                    # darunter keine Aussage (NEED_MORE_DATA), wie Scorecard
WINDOWS = {"today": 0, "7d": 7, "30d": 30}

DEFAULT_PATHS = {
    "history": Path("outputs/history.json"),
    "daily_reports": Path("outputs/daily_reports"),
    "decision_ledger": Path("outputs/intelligence/decision_ledger"),
    "decision_outcomes": Path("outputs/intelligence/decision_outcomes.jsonl"),
    "promotion_state": Path("outputs/intelligence/promotion_state.json"),
    "promotion_transitions": Path("outputs/intelligence/promotion_transitions.jsonl"),
    "contract_registry": Path("outputs/intelligence/contract_registry.jsonl"),
    "workflow_status": Path("outputs/state/workflow_status.json"),
    "system_state": Path("outputs/state/system_state.json"),
    "source_health": Path("outputs/health/source_health_snapshot.json"),
    "external_health": Path("outputs/external_data/health/source_health.json"),
    "trade_outcomes": None,                 # None -> neben dem Quote-Archiv (feedback.TRADE_OUTCOMES_PATH)
    "quotes": None,                         # None -> option_quotes.ROOT
    "candidate_ledger": None,               # None -> candidate_ledger.LEDGER_ROOT
    "meta_label": None,                     # None -> meta_label.ROOT
    "kill_switch": None,                    # None -> production_intelligence_adapter.KILL_SWITCH
}


# ── Hilfen ──────────────────────────────────────────────────────────────────
def _json(p) -> dict | None:
    try:
        p = Path(p)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except (OSError, ValueError) as e:
        log.warning(f"ops_monitor: {p} unlesbar ({e})")
        return None


def _day(x) -> date | None:
    try:
        return date.fromisoformat(str(x)[:10]) if x else None
    except ValueError:
        return None


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def _paths(paths: dict | None) -> dict:
    from modules import candidate_ledger as cl
    from modules import meta_label as ml
    from modules import option_quotes as oq
    from modules import production_intelligence_adapter as pia
    P = {**DEFAULT_PATHS, **(paths or {})}
    P["quotes"] = Path(P["quotes"] or oq.ROOT)
    # Trade-Outcomes liegen neben dem (ggf. überschriebenen) Quote-Archiv, wie in feedback.record_trade_outcome
    P["trade_outcomes"] = Path(P["trade_outcomes"] or (P["quotes"].parent / "trade_outcomes.jsonl"))
    P["candidate_ledger"] = Path(P["candidate_ledger"] or cl.LEDGER_ROOT)
    P["meta_label"] = Path(P["meta_label"] or ml.ROOT)
    P["kill_switch"] = Path(P["kill_switch"] or pia.KILL_SWITCH)
    return {k: Path(v) for k, v in P.items()}                 # str-Pfade überall akzeptiert


def _in_window(d: date | None, today: date, days: int) -> bool:
    return d is not None and (today - timedelta(days=days)) <= d <= today


def _legs(trade: dict) -> list[str | None]:
    """OCC-Symbole je Leg (fehlendes Symbol aus Strike/Verfall rekonstruiert; None = Kontrakt unbekannt)."""
    from modules import option_quotes as oq
    opt = trade.get("option") or {}
    strat = str(trade.get("strategy", "")).upper()
    right = "put" if ("PUT" in strat or "BEAR" in strat) else "call"
    out = []
    for leg in [opt] + ([opt["spread_leg"]] if opt.get("spread_leg") else []):
        sym = leg.get("option_symbol") or leg.get("symbol")
        if not sym:
            try:
                sym = oq.occ_symbol(trade.get("ticker"), leg.get("expiry") or opt.get("expiry"), right,
                                    float(leg.get("strike")))
            except (TypeError, ValueError):
                sym = None
        out.append(sym)
    return out


def _evaluation(trade: dict) -> dict:
    """Gespeicherte Bewertung (ab 2026-10-11) oder abgeleitet (Altbestand, nie zurückgeschrieben)."""
    if trade.get("outcome_evaluation"):
        return trade["outcome_evaluation"]
    from modules.evidence_ledger import from_champion_trades
    recs = from_champion_trades({"closed_trades": [trade]})
    return ((recs[0].get("meta") or {}).get("evaluation") or {}) if recs else {}


def _safe(name: str, fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception as e:  # noqa: BLE001 – ein Abschnitt darf die anderen nie verhindern
        log.warning(f"ops_monitor.{name}: {type(e).__name__}: {e}")
        return {"error": f"{type(e).__name__}: {e}"}


# ── 1. Läufe ────────────────────────────────────────────────────────────────
def run_status(P: dict, today: date) -> dict:
    ws = _json(P["workflow_status"]) or {}
    day = (ws.get("days") or {}).get(today.isoformat()) or {}
    jobs = {job: {"state": r.get("state"), "conclusion": r.get("conclusion"), "last_run": r.get("actual_last_run"),
                  "delay_hours": r.get("delay_hours")} for job, r in day.items()}
    reports = sorted(Path(P["daily_reports"]).glob("*.json")) if Path(P["daily_reports"]).exists() else []
    rep_path = next((p for p in reversed(reports) if p.stem <= today.isoformat()), None)
    st = ((_json(rep_path) or {}).get("stats") or {}) if rep_path else {}
    roi_rej = (st.get("roi_rejects") or {}).get("n")
    contracts = (st.get("roi_ok") + roi_rej) if isinstance(st.get("roi_ok"), int) and isinstance(roi_rej, int) else None
    return {"workflows": jobs, "report_date": rep_path.stem if rep_path else None,
            "report_is_today": bool(rep_path and rep_path.stem == today.isoformat()),
            "scanner": {"universe": st.get("universe"), "analyzed": st.get("analyzed"),
                        "champion_candidates": st.get("final_mc"),       # Final-MC bestanden = Champion-Kandidaten
                        "contracts_designed": contracts,              # Options-Design ergab Kontrakt (ROI ok + ROI-Reject)
                        "trades": st.get("trades"), "stop_reason": st.get("stop_reason"),
                        "session": st.get("session"),
                        # Haiku/Sonnet-Richtungswiderspruch (pipeline stats.direction_conflicts, ab 2026-10-15)
                        "direction_conflicts": st.get("direction_conflicts"),
                        "direction_conflict_share": (round(st["direction_conflicts"] / st["analyzed"], 4)
                                                     if isinstance(st.get("direction_conflicts"), int)
                                                     and st.get("analyzed") else None)},
            "definitions": {"champion_candidates": "stats.final_mc (Final-MC bestanden)",
                            "contracts_designed": "stats.roi_ok + stats.roi_rejects.n (Kontrakt gefunden, vor ROI-Gate)"}}


# ── 2. Quote-Abdeckung ──────────────────────────────────────────────────────
def _entry_status(trade: dict, P: dict) -> tuple[str, str | None, list]:
    """-> (Status, Provider, Records). ENTRY_OK | ENTRY_SUSPECT | ENTRY_UNUSABLE | ENTRY_NOT_IN_ARCHIVE |
    ENTRY_NOT_CAPTURED_BEFORE_ARCHIVE | ENTRY_NOT_CAPTURED | NO_CONTRACT."""
    from modules import option_quotes as oq
    if not any(_legs(trade)):
        return "NO_CONTRACT", None, []
    ids = list(trade.get("entry_quote_record_ids") or [])
    d0 = _day(trade.get("entry_date"))
    if not ids:
        return ("ENTRY_NOT_CAPTURED_BEFORE_ARCHIVE" if d0 is not None and d0 < ARCHIVE_START
                else "ENTRY_NOT_CAPTURED"), None, []
    found = oq.by_ids(ids, (d0 - timedelta(days=1)).isoformat(), (d0 + timedelta(days=1)).isoformat(),
                      root=P["quotes"]) if d0 else {}
    recs = [found.get(i) for i in ids]
    if not all(recs):
        return "ENTRY_NOT_IN_ARCHIVE", None, [r for r in recs if r]
    worst = ("UNUSABLE" if any(r.get("quality_status") == "UNUSABLE" for r in recs) else
             "SUSPECT" if any(r.get("quality_status") == "SUSPECT" for r in recs) else "OK")
    return f"ENTRY_{worst}", recs[0].get("source"), recs


def _is_trading_day(now: datetime) -> bool:
    """NYSE-Handelstag am New-York-Datum von now (Feiertage, Sonderschließungen; Kalender modules.market_snapshot)."""
    from modules import market_snapshot as ms
    return ms.is_trading_day(now.astimezone(ms.NY_TZ).date())


def _exit_status_active(trade: dict, P: dict, now: datetime) -> tuple[str, float | None]:
    """Aktiver Trade: jüngste archivierte Quote je Leg (PIT <= now). EXIT_MARK_TODAY | EXIT_MARK_STALE | NO_EXIT_MARK."""
    from modules import option_quotes as oq
    legs = _legs(trade)
    if not any(legs):
        return "NO_CONTRACT", None
    ages, days = [], []
    for sym in legs:
        q = oq.latest_quote(sym, now, root=P["quotes"], lookback_days=10) if sym else None
        if q is None:
            return "NO_EXIT_MARK", None
        t = datetime.fromisoformat(q["retrieved_at"])
        t = t if t.tzinfo else t.replace(tzinfo=timezone.utc)
        ages.append((now - t).total_seconds() / 86400)
        days.append(t.astimezone(timezone.utc).date())
    # "heute" = jedes Leg am selben UTC-Kalendertag wie der Lauf beobachtet (nicht "jünger als 24 h")
    today_ok = all(d == now.astimezone(timezone.utc).date() for d in days)
    return ("EXIT_MARK_TODAY" if today_ok else "EXIT_MARK_STALE"), round(max(ages), 2)


def _exit_class(e: dict) -> str:
    """Exit-Beobachtung eines geschlossenen Trades: nicht realisierbar (Modell/Proxy/unbrauchbare Exit-Quote),
    unbekannt (DATA_BAD bricht vor der Exit-Prüfung ab), Settlement, beobachtet archiviert bzw. nicht archiviert."""
    if any(str(r).startswith("EXIT_") for r in e.get("reasons") or []):
        return "EXIT_NOT_REALIZABLE"
    if e.get("status") != "OK":
        return "EXIT_UNKNOWN_DATA_BAD" if e.get("status") == "DATA_BAD" else "EXIT_NOT_REALIZABLE"
    if e.get("exit_session") == "SETTLEMENT":
        return "EXIT_SETTLEMENT"
    return "EXIT_OBSERVED_NOT_ARCHIVED" if "EXIT_QUOTE_NOT_ARCHIVED" in (e.get("flags") or []) \
        else "EXIT_OBSERVED_ARCHIVED"


def quote_coverage(hist: dict, P: dict, now: datetime) -> dict:
    """Champion-Trades: Entry (alle mit Kontrakt), Exit (aktive: Mark heute; geschlossene: Bewertung), vollständig
    bewertbar (geschlossen, Status OK). Nenner je Kennzahl ausgewiesen; Ausschlüsse mit Grund."""
    from modules import evidence_ledger as el
    today = now.date()
    active = [t for t in hist.get("active_trades") or [] if isinstance(t, dict)]
    closed = [t for t in hist.get("closed_trades") or [] if isinstance(t, dict) and t.get("outcome") is not None]
    out: dict = {"windows": {}, "definitions": {
        "entry.denominator": "Champion-Trades mit Entry im Zeitraum und bekanntem Kontrakt",
        "exit_active.denominator": "aktive Champion-Trades mit Kontrakt (Mark heute erwartet an Handelstagen)",
        "exit_closed.denominator": "im Zeitraum geschlossene Champion-Trades mit Kontrakt",
        "entry.categories": "observed (OK+SUSPECT) + unusable + missing = denominator",
        "complete.denominator": "im Zeitraum geschlossene Champion-Trades (Bewertung OK = vollständig bewertbar)",
        "excluded": "NO_CONTRACT (Kontrakt unbekannt) zählt in keinen Entry-/Exit-Nenner, wird separat ausgewiesen"}}
    entry_rows = []
    for t in active + closed:
        st, prov, _ = _entry_status(t, P)
        entry_rows.append({"t": t, "status": st, "provider": prov or "unknown", "d": _day(t.get("entry_date")),
                           "strategy": t.get("strategy") or "unknown"})
    for name, days in WINDOWS.items():
        ew = [r for r in entry_rows if _in_window(r["d"], today, days)]
        den = [r for r in ew if r["status"] != "NO_CONTRACT"]
        c = Counter(r["status"] for r in den)
        observed = c["ENTRY_OK"] + c["ENTRY_SUSPECT"]
        cw_all = [t for t in closed if _in_window(_day(t.get("close_date")), today, days)]
        cw = [t for t in cw_all if any(_legs(t))]                  # NO_CONTRACT zählt in keinen Nenner
        evs = [_evaluation(t) for t in cw]
        exit_c = Counter(_exit_class(e) for e in evs)
        out["windows"][name] = {
            "entry": {"denominator": len(den), "observed": observed, "ok": c["ENTRY_OK"],
                      "suspect": c["ENTRY_SUSPECT"], "unusable": c["ENTRY_UNUSABLE"],
                      "missing": len(den) - observed - c["ENTRY_UNUSABLE"], "coverage": _ratio(observed, len(den)),
                      "by_status": dict(c), "excluded_no_contract": len(ew) - len(den)},
            "exit_closed": {"denominator": len(cw), "by_status": dict(exit_c),
                            "excluded_no_contract": len(cw_all) - len(cw),
                            "observed": exit_c["EXIT_OBSERVED_ARCHIVED"] + exit_c["EXIT_OBSERVED_NOT_ARCHIVED"]
                            + exit_c["EXIT_SETTLEMENT"],
                            "coverage": _ratio(exit_c["EXIT_OBSERVED_ARCHIVED"] + exit_c["EXIT_OBSERVED_NOT_ARCHIVED"]
                                               + exit_c["EXIT_SETTLEMENT"], len(cw))},
            "complete": {"denominator": len(cw), "evaluable": sum(1 for e in evs if e.get("status") == "OK"),
                         "coverage": _ratio(sum(1 for e in evs if e.get("status") == "OK"), len(cw)),
                         "by_status": dict(Counter(e.get("status") or "UNKNOWN" for e in evs)),
                         "by_tier": dict(Counter(e.get("tier") or "UNKNOWN" for e in evs)),
                         "reasons": dict(Counter(x for e in evs for x in e.get("reasons") or []))}}
    ex = [_exit_status_active(t, P, now) for t in active]
    ec = Counter(s for s, _ in ex)
    den_a = len(active) - ec["NO_CONTRACT"]
    out["exit_active"] = {"denominator": den_a, "by_status": dict(ec), "mark_today": ec["EXIT_MARK_TODAY"],
                          "coverage": _ratio(ec["EXIT_MARK_TODAY"], den_a),
                          "oldest_mark_age_days": max((a for _, a in ex if a is not None), default=None),
                          "trading_day": _is_trading_day(now)}
    w30 = [r for r in entry_rows if _in_window(r["d"], today, 30) and r["status"] != "NO_CONTRACT"]
    for dim in ("provider", "strategy"):
        g: dict = {}
        for r in w30:
            s = g.setdefault(r[dim], {"denominator": 0, "observed": 0})
            s["denominator"] += 1
            s["observed"] += r["status"] in ("ENTRY_OK", "ENTRY_SUSPECT")
        out[f"entry_by_{dim}_30d"] = {k: {**v, "coverage": _ratio(v["observed"], v["denominator"])}
                                      for k, v in sorted(g.items())}
    out["candidate_marks"] = el.option_mark_coverage(P["candidate_ledger"], as_of=today)
    return out


# ── 3. Lebenszyklus ─────────────────────────────────────────────────────────
STATES = ("SIGNAL", "CONTRACT_SELECTED", "ENTRY_OBSERVED", "EXIT_PENDING", "OUTCOME_FINAL", "LEARNING_ELIGIBLE",
          "LEARNED")


def trade_state(trade: dict, *, active: bool, processed: dict, P: dict, today: date,
                close_after_days: int) -> dict:
    """Weitester erreichter Zustand + Probleme (ein Trade kann einen Zustand trotz Problem erreichen, z. B.
    EXIT_PENDING ohne beobachteten Entry)."""
    from modules import learning_ledger as ll
    from modules.outcomes import is_reliable_outcome
    issues = []
    if not any(_legs(trade)):
        return {"state": "SIGNAL", "issues": ["NO_CONTRACT"]}
    entry, _, _ = _entry_status(trade, P)
    if entry not in ("ENTRY_OK", "ENTRY_SUSPECT"):
        issues.append(entry)
    if active:
        age = (today - _day(trade.get("entry_date"))).days if _day(trade.get("entry_date")) else None
        if age is not None and age > close_after_days + EXIT_OVERDUE_GRACE_DAYS:
            issues.append("EXIT_OVERDUE")
        return {"state": "EXIT_PENDING", "issues": issues}
    ev = _evaluation(trade)
    if not trade.get("outcome_evaluation"):
        issues.append("LEGACY_NO_STORED_EVALUATION")
    if ev.get("status") != "OK":
        issues.append(f"OUTCOME_{ev.get('status') or 'UNKNOWN'}")
    eligible = ev.get("status") == "OK" and is_reliable_outcome(trade)
    tid = ll.trade_id(trade)
    learned = sorted(k.split(":")[-1] for k in processed if k.startswith(f"{tid}:close:"))
    if learned and not eligible:
        issues.append("LEARNED_NOT_ELIGIBLE")              # Training trotz unzulässiger Daten
    if eligible and not learned:
        closed = _day(trade.get("close_date"))
        # vor dem Consume-once-Ledger geschlossen: mit dem Altmechanismus gelernt, aber ohne Schlüssel
        issues.append("LEGACY_LEARNED_UNTRACKED" if closed is None or closed < CONSUME_ONCE_START
                      else "ELIGIBLE_NOT_LEARNED")                # ohne close_date: Altschema, nicht datierbar
    state = "LEARNED" if learned else "LEARNING_ELIGIBLE" if eligible else "OUTCOME_FINAL"
    return {"state": state, "issues": issues, "learned_targets": learned, "trade_id": tid}


def lifecycle(hist: dict, P: dict, now: datetime) -> dict:
    """Champion-Trades je Zustand und Problem; Kandidaten (Real-Option) je Zustand.
    exactly_once.ok = kein nicht lernbarer Trade gelernt, kein lernbarer Trade doppelt, kein Ergebnis je Ziel mehrfach
    gelernt. eligible_not_learned (blockiert) und keys_without_trade (z. B. quarantänierte Altduplikate) werden separat
    gemeldet, verletzen Exactly-once aber nicht. Champion-Trades sind ab Buchung mindestens EXIT_PENDING; die Vorstufen
    SIGNAL/CONTRACT_SELECTED/ENTRY_OBSERVED bildet der Kandidaten-Lebenszyklus ab."""
    from modules import learning_ledger as ll
    today = now.date()
    processed = hist.get(ll.FIELD) or {}
    cad = _close_after_days()
    rows = [{"active": True, **trade_state(t, active=True, processed=processed, P=P, today=today,
                                           close_after_days=cad), "ticker": t.get("ticker"),
             "entry_date": t.get("entry_date")} for t in hist.get("active_trades") or [] if isinstance(t, dict)]
    rows += [{"active": False, **trade_state(t, active=False, processed=processed, P=P, today=today,
                                             close_after_days=cad), "ticker": t.get("ticker"),
              "entry_date": t.get("entry_date"), "close_date": t.get("close_date")}
             for t in hist.get("closed_trades") or [] if isinstance(t, dict) and t.get("outcome") is not None]
    issues = Counter(i for r in rows for i in r["issues"])
    tids = Counter(r.get("trade_id") for r in rows if r.get("trade_id"))
    learnable_tids = {r["trade_id"] for r in rows if r.get("trade_id") and r["state"] in ("LEARNING_ELIGIBLE", "LEARNED")}
    dup_learnable = sum(1 for t, v in tids.items() if v > 1 and t in learnable_tids)
    keys_by_tid = Counter(k.split(":close:")[0] for k in processed if ":close:" in k)
    unknown_keys = sorted(t for t in keys_by_tid if t not in tids and t.startswith("T"))
    champion = {"by_state": {s: sum(1 for r in rows if r["state"] == s) for s in STATES},
                "issues": dict(issues),
                "blocked": [{k: r.get(k) for k in ("ticker", "entry_date", "state", "issues")} for r in rows
                            if set(r["issues"]) & {"EXIT_OVERDUE", "ELIGIBLE_NOT_LEARNED", "LEARNED_NOT_ELIGIBLE"}],
                "legacy": {"duplicate_unlearnable_trades": sum(1 for t, v in tids.items()
                                                               if v > 1 and t not in learnable_tids),
                           "learned_untracked": issues["LEGACY_LEARNED_UNTRACKED"],
                           "note": "Altbestand vor Quote-Archiv/Consume-once (2026-10-10); nie lernbar bzw. ohne "
                                   "Schlüssel gelernt – keine Exactly-once-Verletzung"},
                "exactly_once": {"learned_keys": len(processed),
                                 "duplicate_closed_trades": sum(1 for v in tids.values() if v > 1),
                                 "duplicate_learnable_trades": dup_learnable,
                                 "learned_not_eligible": issues["LEARNED_NOT_ELIGIBLE"],
                                 "eligible_not_learned": issues["ELIGIBLE_NOT_LEARNED"],
                                 "keys_without_trade": len(unknown_keys),
                                 "duplicate_skips_total": (hist.get(ll.META) or {}).get("duplicate_skips"),
                                 "ok": not (issues["LEARNED_NOT_ELIGIBLE"] or dup_learnable or keys_by_tid_dupes(processed))}}
    # Kandidaten (Real-Option, 5-Tage-Horizont als frühester Outcome)
    cand = Counter()
    root = P["candidate_ledger"]
    for f in sorted(root.glob("*.jsonl")) if root.exists() else []:
        for r in read_jsonl(f):
            if r.get("direction") not in ("BULLISH", "BEARISH"):
                continue
            cand["SIGNAL"] += 1
            ro = r.get("real_option") or {}
            if not ro:
                cand["EXCLUDED_" + str(r.get("real_option_skip_reason") or "NO_CONTRACT").upper()] += 1
                continue
            cand["CONTRACT_SELECTED"] += 1
            if ro.get("entry_pending") or not ro.get("ask"):
                cand["ENTRY_PENDING"] += 1
                continue
            cand["ENTRY_OBSERVED"] += 1
            oc = r.get("outcomes") or {}
            ent = _day(ro.get("entry_date"))
            if "real_opt_ret_5d" in oc:
                cand["OUTCOME_FINAL"] += 1
                cand["LATE_MARK"] += bool(oc.get("real_opt_late_5d"))
            elif oc.get("real_opt_mark_status_5d"):          # Endzustand ohne Preis (candidate_ledger._give_up_marks)
                cand["MARK_UNAVAILABLE"] += 1
            elif ent and (today - ent).days >= 5 + MARK_OVERDUE_DAYS:
                cand["EXIT_OVERDUE"] += 1
            else:
                cand["EXIT_PENDING"] += 1
    return {"champion": champion, "candidates": dict(cand),
            "definitions": {"LEARNED": "Consume-once-Schlüssel <trade_id>:close:<Ziel> vorhanden (feedback.learn_on_close)",
                            "LEARNING_ELIGIBLE": "geschlossen, Bewertung OK und verlässliche Preismethode",
                            "candidate EXIT_OVERDUE": f"5-Tage-Mark > {MARK_OVERDUE_DAYS} Tage überfällig"}}


def keys_by_tid_dupes(processed: dict) -> int:
    """Schlüssel = <observation_id>:<horizont>:<ziel> (learning_ledger.key; die observation_id kann selbst ':'
    enthalten, z. B. 'shadow:<id>'). Gezählt werden (observation_id, ziel), die unter mehreren Horizont-Schreibweisen
    gelernt wurden – ein Ergebnis darf je Ziel nur einmal ins Lernen."""
    seen = Counter((k.rsplit(":", 2)[0], k.rsplit(":", 2)[2]) for k in processed if k.count(":") >= 2)
    return sum(1 for v in seen.values() if v > 1)


# ── 4. Forward-Performance ──────────────────────────────────────────────────
def _metrics(vals: list[float], days: list, *, alpha: float = 0.05) -> dict:
    """Netto-Kennzahlen; CI per Bootstrap über unabhängige Tage; Drawdown der kumulierten Summe (Tagesreihenfolge)."""
    from modules import evidence_core as ec
    out = {"n": len(vals), "n_independent_days": len(set(days))}
    if not vals:
        return {**out, "verdict": "NO_DATA"}
    order = sorted(range(len(vals)), key=lambda i: str(days[i]))
    cum, peak, mdd = 0.0, 0.0, 0.0
    for i in order:
        cum += vals[i]
        peak = max(peak, cum)
        mdd = min(mdd, cum - peak)
    losses = sorted(v for v in vals if v < 0)
    out.update(sum_net=round(sum(vals), 6), mean_net=round(statistics.fmean(vals), 6),
               median_net=round(statistics.median(vals), 6), hit_rate=round(sum(v > 0 for v in vals) / len(vals), 4),
               worst=round(min(vals), 6), max_drawdown_cum=round(mdd, 6),
               loss_distribution={"n_losses": len(losses), "p50_loss": round(losses[len(losses) // 2], 6)
                                  if losses else None, "share_total_loss": round(sum(v <= -0.99 for v in vals)
                                                                                 / len(vals), 4)},
               stdev=round(statistics.stdev(vals), 6) if len(vals) > 1 else None)
    if len(vals) < MIN_N or len(set(days)) < MIN_DAYS:
        return {**out, "verdict": "NEED_MORE_DATA", "ci": None}
    lo, hi = ec.group_bootstrap_ci(vals, days, alpha=alpha)
    return {**out, "ci": [lo, hi], "verdict": "POSITIVE" if lo is not None and lo > 0 else
            "NEGATIVE" if hi is not None and hi < 0 else "INCONCLUSIVE"}


def forward_performance(hist: dict, P: dict, now: datetime) -> dict:
    """Champion (realisiert): nur Bewertung OK (beobachtete Quotes/Settlement) zählt; modelliert/unbekannt separat.
    Challenger: alle Verträge auf DERSELBEN eingefrorenen Population (Decision-Ledger = Champion-Trades zum
    Entscheidungszeitpunkt) mit Outcomes aus decision_outcomes.jsonl – auch abgelehnte Trades laufen als
    Counterfactual weiter, daher sind vermiedene Verluste und entgangene Gewinne messbar."""
    from modules import evidence_ledger as el
    from modules.outcomes import RELIABLE_OUTCOME_METHODS
    champ = el.from_champion_trades(hist)
    ok = [r for r in champ if ((r.get("meta") or {}).get("evaluation") or {}).get("status") == "OK"
          and r.get("outcome") is not None]
    vals = [float(r["outcome_net"] if r.get("outcome_net") is not None else r["outcome"]) for r in ok]
    days = [r.get("label_available_at") for r in ok]
    champion = {"observed": _metrics(vals, days),
                "net_definition": "Brutto (Bid/Ask gekreuzt) minus Kommission 0.65 $/Kontrakt/Seite; "
                                  "gespeichertes outcome_net hat Vorrang",
                "not_evaluable": dict(Counter(((r.get("meta") or {}).get("evaluation") or {}).get("status")
                                              for r in champ if r not in ok)),
                "by_strategy": {}, "by_regime": {}}
    for dim, keyf in (("by_strategy", lambda r: (r.get("meta") or {}).get("strategy") or "unknown"),
                      ("by_regime", lambda r: str((r.get("features") or {}).get("vix_structure")
                                                  or (r.get("features") or {}).get("regime") or "unknown"))):
        g: dict = {}
        for r, v, d in zip(ok, vals, days):
            g.setdefault(keyf(r), ([], []))
            g[keyf(r)][0].append(v)
            g[keyf(r)][1].append(d)
        champion[dim] = {k: _metrics(a, b) for k, (a, b) in sorted(g.items())}
    # Challenger auf identischer Population
    rows = []
    d = Path(P["decision_ledger"])
    for f in sorted(d.glob("*.jsonl")) if d.exists() else []:
        rows += read_jsonl(f)
    outs = {o.get("decision_id"): o for o in read_jsonl(Path(P["decision_outcomes"]))}
    ps = _json(P["promotion_state"]) or {}
    hyps = ps.get("hypotheses") or {}
    keys = sorted({k for r in rows for k in (r.get("intelligence") or {})} | set(hyps))
    base = [(r, outs.get(r.get("decision_id"))) for r in rows]
    resolved_all = [(r, o) for r, o in base if o and o.get("outcome") is not None]
    # wie beim Champion: nur beobachtete (verlässliche) Outcomes zählen; modellierte separat ausgewiesen
    resolved = [(r, o) for r, o in resolved_all if o.get("outcome_method") in RELIABLE_OUTCOME_METHODS]
    challengers = {}
    for k in keys:
        fired, kept, ev_n = [], [], 0
        for r, o in resolved:
            h = (r.get("intelligence") or {}).get(k) or {}
            if not h.get("evaluable"):
                continue
            ev_n += 1
            (fired if h.get("fired") else kept).append(float(o["outcome"]))
        hs = hyps.get(k) or {}
        observed_share = _ratio(len(resolved), len(resolved_all))
        challengers[k] = {
            "state": hs.get("state"), "influence_level": hs.get("influence_level"),
            "production_class": hs.get("production_class"), "eligible_stage": hs.get("eligible_stage"),
            "population": "champion_decisions" if not hs.get("eligible_stage") else hs.get("eligible_stage"),
            "evaluable_resolved": ev_n, "would_act_on": len(fired),
            "avoided_losses": round(sum(-v for v in fired if v < 0), 6) if fired else None,
            "foregone_gains": round(sum(v for v in fired if v > 0), 6) if fired else None,
            "incremental_vs_champion_sum": round(-sum(fired), 6) if fired else None,
            "incremental_vs_champion_mean": round(-sum(fired) / ev_n, 6) if fired and ev_n else None,
            "kept_mean": round(statistics.fmean(kept), 6) if kept else None,
            "fired_mean": round(statistics.fmean(fired), 6) if fired else None,
            "controller_evidence": {x: (hs.get("evidence") or {}).get(x) for x in
                                    ("n_observations", "n_independent_dates", "n_fired", "delta_expectancy", "ci")},
            "verdict": "NEED_MORE_DATA" if ev_n < MIN_N or len(fired) < 10 else "SEE_CONTROLLER",
            "observed_outcome_share": observed_share}
    from modules import meta_label as ml
    sc_recs = el.from_candidate_ledger(P["candidate_ledger"]) + champ          # dieselben (überschreibbaren) Quellen
    return {"champion": champion,
            "population": {"decisions": len(rows), "resolved": len(resolved_all), "resolved_observed": len(resolved),
                           "excluded_not_observed": len(resolved_all) - len(resolved),
                           "outcome_methods": dict(Counter(o.get("outcome_method") for _, o in resolved_all)),
                           "definition": "Decision-Ledger (Champion-Trades zum Entscheidungszeitpunkt, eingefroren); "
                                         "Abstinenz-Wert in Rendite-Einheiten je Trade (Summe/Mittel), netto wie "
                                         "decision_outcomes (Bid/Ask-Kreuzung, ohne Kommission)"},
            "challengers": challengers,
            "meta_label": _safe("meta_label_status", ml.status, P["meta_label"]),
            "scorecard": _safe("scorecard", lambda: {k: v for k, v in el.scorecard(sc_recs, as_of=now.date())["cells"].items()
                                                     if k.endswith("|ALL") or k.endswith("|LATE_STAGE")
                                                     or k.endswith("|CHAMPION")})}


# ── 5. Training, Promotion, Integrität, Daten ───────────────────────────────
def training_status(P: dict) -> dict:
    from modules import meta_label as ml
    root = P["meta_label"]
    st, reg = ml.status(root), ml._registry(root)
    checks = []
    for v in reg.get("versions") or []:
        m = _json(root / "versions" / v["version"] / "manifest.json") or {}
        if not m:
            continue
        bad = []
        if m.get("label_cutoff") and m.get("trained_at") and m["label_cutoff"] > m["trained_at"][:10]:
            bad.append("LABEL_AFTER_TRAINING")              # Label nach Trainingszeit = Leakage
        need = (m.get("spec") or {}).get("readiness") or {}
        if need and (m.get("n_train", 0) < need.get("min_labels", 0)
                     or m.get("n_train_dates", 0) < need.get("min_dates", 0)):
            bad.append("TRAINED_BELOW_READINESS")
        if bad:
            checks.append({"version": v["version"], "violations": bad})
    active_ok = None
    if reg.get("active"):
        active_ok = ml.active_model(root)[0] is not None
    return {"status": st.get("status"), "n_labels": st.get("n_labels"), "n_label_dates": st.get("n_label_dates"),
            "required": st.get("required"), "as_of": st.get("as_of"), "versions": len(reg.get("versions") or []),
            "active": reg.get("active"), "hold": reg.get("hold"), "active_artifact_ok": active_ok,
            "manifest_violations": checks}


def promotion_status(P: dict) -> dict:
    from modules import production_intelligence_adapter as pia
    ps = _json(P["promotion_state"]) or {}
    hyps = ps.get("hypotheses") or {}
    rows = {k: {"state": h.get("state"), "influence_level": h.get("influence_level"), "decision": h.get("decision"),
                "blocking": h.get("next_requirement") or ("; ".join(h.get("reasons") or [])[:300] or None),
                "recommendation": h.get("recommendation"), "integrity_ok": h.get("integrity_ok")}
            for k, h in sorted(hyps.items())}
    from modules.promotion_controller import ACTIVE_STATES
    unjustified = [k for k, h in hyps.items() if (h.get("influence_level") or "NONE") != "NONE"
                   and (not h.get("integrity_ok") or h.get("state") not in ACTIVE_STATES)]
    sysst = _json(P["system_state"]) or {}
    return {"generated": ps.get("generated"), "max_automatic_influence": ps.get("max_automatic_influence"),
            "kill_switch_active": pia.kill_switch_active(P["kill_switch"]),
            "safe_mode": (ps.get("safe_mode") or {}).get("active"),          # Promotion pausiert (PromotionController)
            "safe_mode_reasons": (ps.get("safe_mode") or {}).get("reasons") or [],
            "champion_version": (sysst.get("champion_version") or {}).get("version"),
            "by_state": dict(Counter(h.get("state") for h in hyps.values())),
            "by_influence": dict(Counter(h.get("influence_level") for h in hyps.values())),
            "active_influence": sorted(k for k, h in hyps.items() if (h.get("influence_level") or "NONE") != "NONE"),
            "unjustified_influence": sorted(unjustified), "hypotheses": rows}


def integrity(hist: dict, P: dict, now: datetime) -> dict:
    from modules import hypothesis_contract as hc
    from modules import promotion_controller as pc
    out: dict = {}
    ps = _json(P["promotion_state"])
    out["promotion_state_hash_ok"] = None if ps is None else pc.state_digest(ps) == ps.get("state_hash")
    out["promotion_state_unreadable"] = ps is None and Path(P["promotion_state"]).exists()
    try:
        _, tr_problems = pc.read_transitions(P["promotion_transitions"])
    except Exception as e:  # noqa: BLE001 – unlesbare Kette ist selbst der Befund
        tr_problems = [f"Transition-Kette unlesbar: {type(e).__name__}: {e}"]
    try:
        _, reg_problems = hc.read_registry(P["contract_registry"])
    except Exception as e:  # noqa: BLE001
        reg_problems = [f"Registry unlesbar: {type(e).__name__}: {e}"]
    out["transition_chain_problems"] = tr_problems
    out["registry_problems"] = reg_problems
    rows = []
    d = Path(P["decision_ledger"])
    for f in sorted(d.glob("*.jsonl")) if d.exists() else []:
        rows += read_jsonl(f)
    dup = Counter((r.get("date"), r.get("ticker")) for r in rows)
    out["decision_ledger_duplicates"] = sum(1 for v in dup.values() if v > 1)
    to = read_jsonl(P["trade_outcomes"])
    out["trade_outcome_duplicates"] = sum(1 for v in Counter(r.get("trade_id") for r in to).values() if v > 1)
    qdup, foreign, n_quotes = 0, 0, 0
    lo = (now.date() - timedelta(days=30)).isoformat()
    from modules import option_quotes as oq
    recs = oq.read(lo, None, root=P["quotes"])
    n_quotes = len(recs)
    groups: dict = {}
    for r in recs:
        groups.setdefault(r.get("record_id"), []).append(r)
    qdup = reobs = 0
    for g in groups.values():
        if len(g) < 2:
            continue
        # Dieselbe Börsenquote (gleicher Inhalt) zu verschiedenen Zeitpunkten erneut abgerufen = Mehrfachbeobachtung
        # (vor dem tagesübergreifenden Archiv-Dedupe 2026-10-14 möglich; option_quotes.by_ids nimmt die erste).
        # Integritätsproblem nur bei abweichendem Inhalt oder identischem Abrufzeitpunkt (doppelt geschrieben).
        same_quote = len({tuple(x.get(k) for k in ("occ_symbol", "quote_time", "bid", "ask")) for x in g}) == 1
        ras = [x.get("retrieved_at") for x in g]
        if same_quote and len(set(ras)) == len(ras):
            reobs += 1
        else:
            qdup += 1
    foreign = sum(1 for r in recs if r.get("price_type") != oq.PRICE_TYPE)
    out.update(quote_records_30d=n_quotes, quote_duplicate_ids=qdup, quote_reobserved_ids=reobs,
               quote_non_observed_prices=foreign)
    out["closed_trade_quarantine"] = len(hist.get("closed_trades_quarantine") or [])
    out["external_archive"] = _safe("external_archive", _external_rows_check, now)
    issues = []
    if out["promotion_state_hash_ok"] is False:
        issues.append("PROMOTION_STATE_HASH")
    if out["promotion_state_unreadable"]:
        issues.append("PROMOTION_STATE_UNREADABLE")
    if tr_problems:
        issues.append("TRANSITION_CHAIN")
    if reg_problems:
        issues.append("CONTRACT_REGISTRY")
    if out["decision_ledger_duplicates"]:
        issues.append("DECISION_LEDGER_DUPLICATES")
    if out["trade_outcome_duplicates"]:
        issues.append("TRADE_OUTCOME_DUPLICATES")
    if qdup or foreign:
        issues.append("QUOTE_ARCHIVE")
    if (out["external_archive"] or {}).get("failed"):
        issues.append("EXTERNAL_ARCHIVE_HASH")
    out["issues"] = issues
    return out


def _external_rows_check(now: datetime, root: Path = Path("outputs/external_data/normalized"),
                         max_rows_per_file: int = 300) -> dict:
    """verify_row für die jüngsten Zeilen der in den letzten 3 Tagen geschriebenen Normalized-Dateien."""
    from modules.external import archive as ea
    res = Counter()
    cut = now.timestamp() - 3 * 86400
    for f in sorted(root.rglob("*.jsonl")) if root.exists() else []:
        try:
            if f.stat().st_mtime < cut:
                continue
            lines = f.read_text(encoding="utf-8").splitlines()[-max_rows_per_file:]
        except OSError:
            continue
        for ln in lines:
            try:
                v = ea.verify_row(json.loads(ln))
            except ValueError:
                res["unparsable"] += 1
                continue
            res["legacy" if v is None else "ok" if v else "failed"] += 1
    return dict(res)


def data_health(P: dict, now: datetime) -> dict:
    sh = _json(P["source_health"]) or {}
    srcs = sh.get("sources") or sh.get("results") or {}
    statuses = Counter((v or {}).get("status") for v in (srcs.values() if isinstance(srcs, dict) else srcs))
    eh = _json(P["external_health"]) or {}
    esrc = eh.get("sources") or {}
    estat = Counter((v or {}).get("status") or (v or {}).get("last_status") for v in esrc.values()) \
        if isinstance(esrc, dict) else Counter()
    days = {}
    for i in range(14):
        dd = (now.date() - timedelta(days=i)).isoformat()
        p = Path(P["quotes"]) / f"{dd}.jsonl"
        if p.exists():
            rs = read_jsonl(p)
            days[dd] = {"n": len(rs), "by_context": dict(Counter(r.get("context") for r in rs)),
                        "by_quality": dict(Counter(r.get("quality_status") for r in rs))}
    vals = [v["n"] for v in days.values()]
    return {"source_health": {"generated": sh.get("generated"), "by_status": dict(statuses)},
            "external_data": {"by_status": dict(estat)},
            "quote_archive_days": days,
            "quote_archive_median_14d": statistics.median(vals) if vals else None,
            "quote_archive_root_exists": Path(P["quotes"]).exists()}


# ── Gesamt ──────────────────────────────────────────────────────────────────
def timing_status(hist: dict, now: datetime) -> dict:
    """Timing-Policies getrennt (config/timing_policy.yaml): Bestand je Policy und Liste, timing-v2-Lebenszyklus
    (Entry offen, Exit ausstehend, ausgeführt, UNUSABLE mit Gründen) und Abweichung Entscheid vs. Ausführung."""
    from modules import timing_policy as tp
    lists = {"active": hist.get("active_trades") or [], "exit_pending": hist.get("exit_pending_trades") or [],
             "closed": hist.get("closed_trades") or [], "unusable": hist.get("closed_trades_unusable") or [],
             "counterfactual_exit_pending": hist.get("counterfactual_exit_pending") or [],
             "counterfactual_unusable": hist.get("counterfactual_unusable") or []}
    v2 = {k: [t for t in v if tp.policy_of(t) == tp.V2] for k, v in lists.items()}
    ages = []
    for t in v2["exit_pending"]:
        ct = tp._ts(t.get("close_time"))
        if ct is not None:
            ages.append(round((now - ct).total_seconds() / 86400, 2))
    dev = [float(t["outcome"]) - float(t["outcome_at_decision"]) for t in v2["closed"]
           if isinstance(t.get("outcome"), (int, float)) and isinstance(t.get("outcome_at_decision"), (int, float))]
    return {"current_policy": tp.current(),
            "by_policy": {k: dict(Counter(tp.policy_of(t) for t in v)) for k, v in lists.items()},
            "v2_active_entry": dict(Counter(((t.get("timing") or {}).get("entry") or {}).get("status") or "UNBEWERTET"
                                            for t in v2["active"])),
            "v2_exit_pending": len(v2["exit_pending"]), "v2_exit_pending_oldest_days": max(ages) if ages else None,
            "v2_exit_pending_reasons": dict(Counter(r for t in v2["exit_pending"]
                                                    for r in (t.get("timing") or {}).get("reasons") or [])),
            "v2_closed": len(v2["closed"]), "v2_unusable": len(v2["unusable"]),
            "v2_unusable_reasons": dict(Counter(r for t in v2["unusable"] for r in (t.get("timing") or {}).get("reasons") or [])),
            "v2_deviation_vs_decision": {"n": len(dev), "mean": round(statistics.fmean(dev), 6) if dev else None,
                                         "definition": "Outcome zur zulässigen Ausführung minus Return zum Entscheid"}}


# ── 9. Lern-Prognose (2026-10-15) ───────────────────────────────────────────
ETA_HORIZON_DAYS = 365                      # weiter wird nicht projiziert ("nicht absehbar")
ETA_RATE_DAYS = 14                          # Zulauf-Rate aus den letzten N Tagen


def _rl_cfg() -> tuple[int, str]:
    try:
        import yaml
        rl = (yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8")) or {}).get("rl", {}) or {}
        return int(rl.get("arm_threshold", 30)), str(rl.get("arm_since", "2026-06-11"))
    except Exception:  # noqa: BLE001
        return 30, "2026-06-11"


def _eta(have: int, need: int, per_day: float, today: date) -> str | None:
    """Lineare Projektion: Datum, an dem `need` bei `per_day` Zuwachs erreicht ist (None = nicht absehbar)."""
    if have >= need:
        return today.isoformat()
    if per_day <= 0:
        return None
    days = math.ceil((need - have) / per_day)
    return (today + timedelta(days=days)).isoformat() if days <= ETA_HORIZON_DAYS else None


def learning_eta(hist: dict, P: dict, now: datetime) -> dict:
    """Wann die Lernschwellen bei aktuellem Zulauf voraussichtlich erreicht sind (nur Projektion, keine Aussage über
    Qualität): Meta-Label (Labels + unabhängige Tage, Reifung der offenen Labels exakt nach label_available_at),
    RL-Arming (abgeschlossene Trades seit arm_since) und Champion-Evidenz (bewertbare Trades/Tage, MIN_N/MIN_DAYS)."""
    from modules import evidence_ledger as el
    from modules import meta_label as ml
    today = now.date()
    out: dict = {}

    # Meta-Label: Zielpopulation, Horizont der Spec; offene Labels reifen an label_available_at
    spec = ml.SPEC
    h, stages = spec["horizon"], set(spec.get("train_stages") or [])
    need_n, need_d = spec["readiness"]["min_labels"], spec["readiness"]["min_dates"]
    pop = spec["population"]
    rows = [r for r in el.from_candidate_ledger(P["candidate_ledger"]) if r.get("population") == pop
            and r.get("horizon") == h and r.get("direction") in (1, -1)
            and (not stages or (r.get("meta") or {}).get("stage") in stages)]
    have = el.labeled(rows, as_of=today, population=pop, horizon=h)        # exakt wie meta_label.run
    # Reifung: vorhandene Labels + noch offene (label_available_at in der Zukunft); bereits fällige Zeilen ohne
    # Outcome (z. B. NO_PRICE_DATA) zählen nicht als künftiges Label
    pending = [r for r in rows if (_day(r.get("label_available_at")) or today) > today]
    events = sorted((_day(r.get("label_available_at")), _day(r.get("decision_date"))) for r in have + pending
                    if _day(r.get("label_available_at")) and _day(r.get("decision_date")))
    recent = [d for _, d in events if 0 <= (today - d).days < ETA_RATE_DAYS]
    per_day = len(recent) / ETA_RATE_DAYS                      # neue Zielzeilen je Kalendertag
    dates_per_day = len(set(recent)) / ETA_RATE_DAYS            # neue Entscheidungstage je Kalendertag
    eta_ml, i, n, dates, acc_n, acc_d = None, 0, 0, set(), 0.0, 0.0
    for k in range(ETA_HORIZON_DAYS + 1):
        d = today + timedelta(days=k)
        while i < len(events) and events[i][0] <= d:         # vorhandene Zeilen reifen exakt an label_available_at
            n, i = n + 1, i + 1
            dates.add(events[i - 1][1])
        if k >= h + 2:                                        # ab morgen entschiedene Zeilen reifen nach h+1 Tagen
            acc_n, acc_d = acc_n + per_day, acc_d + dates_per_day
        if n + int(acc_n) >= need_n and len(dates) + int(acc_d) >= need_d:
            eta_ml = d.isoformat()
            break
    out["meta_label"] = {"have": len(have), "need": need_n, "have_dates": len({_day(r.get("decision_date")) for r in have}),
                         "need_dates": need_d, "pending_maturing": len(pending),
                         "new_rows_per_day": round(per_day, 2), "eta": eta_ml,
                         "basis": f"Zielpopulation {sorted(stages)}, Horizont {h} T; offene Labels reifen an "
                                  f"label_available_at, Zulauf aus den letzten {ETA_RATE_DAYS} T"}

    # RL-Arming
    thr, since = _rl_cfg()
    closed = hist.get("closed_trades") or []
    rl = [t for t in closed if t.get("outcome") is not None and str(t.get("entry_date", ""))[:10] >= since]
    last30 = [t for t in rl if (_day(t.get("close_date")) or date.min) >= today - timedelta(days=30)]
    out["rl_arming"] = {"have": len(rl), "need": thr, "closes_per_day_30d": round(len(last30) / 30, 3),
                        "eta": _eta(len(rl), thr, len(last30) / 30, today), "since": since}

    # Champion-Evidenz (wie forward_performance: nur Bewertung OK)
    champ = [r for r in el.from_champion_trades(hist)
             if ((r.get("meta") or {}).get("evaluation") or {}).get("status") == "OK" and r.get("outcome") is not None]
    ok30 = [r for r in champ if (_day(r.get("label_available_at")) or date.min) >= today - timedelta(days=30)]
    n_days = len({r.get("label_available_at") for r in champ})
    eta_n = _eta(len(champ), MIN_N, len(ok30) / 30, today)
    eta_d = _eta(n_days, MIN_DAYS, len({r.get("label_available_at") for r in ok30}) / 30, today)
    out["champion_evidence"] = {"have": len(champ), "need": MIN_N, "have_days": n_days, "need_days": MIN_DAYS,
                                "evaluable_per_day_30d": round(len(ok30) / 30, 3),
                                "eta": max(eta_n, eta_d) if eta_n and eta_d else None}
    out["note"] = "Projektion bei unverändertem Zulauf; frühestens-Datum, keine Qualitätsaussage"
    return out


def collect(now: datetime | None = None, paths: dict | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:                                       # naive Zeit = UTC (wie option_quotes)
        now = now.replace(tzinfo=timezone.utc)
    P = _paths(paths)
    hist = _json(P["history"]) or {}
    return {"schema": SCHEMA, "generated": now.isoformat(timespec="seconds"), "as_of": now.date().isoformat(),
            "read_only": True,
            "run": _safe("run", run_status, P, now.date()),
            "quote_coverage": _safe("quote_coverage", quote_coverage, hist, P, now),
            "lifecycle": _safe("lifecycle", lifecycle, hist, P, now),
            "forward": _safe("forward", forward_performance, hist, P, now),
            "training": _safe("training", training_status, P),
            "promotion": _safe("promotion", promotion_status, P),
            "integrity": _safe("integrity", integrity, hist, P, now),
            "data_health": _safe("data_health", data_health, P, now),
            "timing": _safe("timing", timing_status, hist, now),
            "learning_eta": _safe("learning_eta", learning_eta, hist, P, now),
            "workflows": _safe("workflows", _workflows, now)}


def _workflows(now: datetime) -> dict:
    from modules import ops_workflows
    return ops_workflows.collect(now)


def _close_after_days() -> int:
    try:
        import yaml
        return int((yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8")) or {})
                   .get("learning", {}).get("close_after_days", 45))
    except Exception:  # noqa: BLE001
        return 45

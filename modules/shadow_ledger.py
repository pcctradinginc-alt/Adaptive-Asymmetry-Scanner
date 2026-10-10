"""modules/shadow_ledger.py – append-only Shadow-Outcome-Ledger mit status-/horizontbasierter Retention.

Root Cause (Audit 2026-10-09): feedback.evaluate_shadow_trades bewertete Schatten-Trades erst nach
learning.close_after_days (45 T), behielt aber nur die letzten 300 in history.json. Bei 30–47 neuen
Schatten-Trades je Handelstag wurden sie nach ~8–9 Handelstagen verdrängt; das Archiv
(outputs/shadow_trades_archive.jsonl) las niemand -> 41 Gate-Rejects ohne Outcome archiviert.

Jetzt:
  records/<YYYY-MM>.jsonl   unveränderlicher Original-Snapshot je Schatten-Kandidat (append-only)
  events/<YYYY-MM>.jsonl    Horizont-Ereignisse EVALUATED | RETRY | UNAVAILABLE | ARCHIVED (append-only)
Der Lifecycle wird aus Record + Ereignissen abgeleitet (nie überschrieben):
  PENDING -> MATURED[_RETRY_REQUIRED] -> PARTIALLY_EVALUATED -> EVALUATED | OUTCOME_UNAVAILABLE -> ARCHIVED
Invariante: ein ungeklärter Record (nicht alle Horizonte aufgelöst) darf weder archiviert noch gelöscht
werden -> ShadowIntegrityError (nie stille Bereinigung).

PIT: Der Snapshot bleibt unverändert. Underlying-Outcomes nutzen nur Schlusskurse bis einschließlich
Fälligkeitstag (split-, nicht dividendenbereinigt). Options-/Spread-Outcomes nur aus Live-Quotes, wenn
die Bewertung höchstens LIVE_QUOTE_MAX_LAG_DAYS nach Fälligkeit erfolgt – sonst None (keine erfundenen
Werte, keine nachträglichen Quotes). Kein Einfluss auf Gates, Policy oder Produktion.
"""
from __future__ import annotations

import hashlib
import json
import logging
import statistics
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable

from modules.atomic_io import append_jsonl, read_jsonl

log = logging.getLogger(__name__)

LEDGER_DIR = Path("outputs/intelligence/shadow_ledger")
HORIZONS = (20, 45, 60)                 # Kalendertage; 45 = learning.close_after_days (Legacy-Outcome)
LIVE_QUOTE_MAX_LAG_DAYS = 4             # Options-Outcome nur aus Quotes nahe der Fälligkeit
MAX_RETRIES = 7                         # danach (frühestens 14 T nach Fälligkeit) OUTCOME_UNAVAILABLE
MIN_DAYS_BEFORE_UNAVAILABLE = 14
GATE_MIN_N = 30                         # darunter NEED_MORE_DATA im Gate-Learning-Report

ROI_GATES = {"roi_initial", "edge", "mc_pnl", "theta", "vega", "roi", "roi_net"}
LIQUIDITY_GATES = {"liquidity", "open_interest", "oi", "spread", "spread_ratio", "volume", "no_chain"}
EXECUTION_GATES = {"execution", "spread_execution", "immediate_liquidation_loss", "quote_quality", "slippage"}

RESOLVED = {"EVALUATED", "UNAVAILABLE"}


class ShadowIntegrityError(RuntimeError):
    """Ungeklärter Schatten-Record sollte archiviert/gelöscht werden – Abbruch, nichts wird verworfen."""


# ── Identität / Snapshot ─────────────────────────────────────────────────────
def _d(x) -> str:
    return str(x or "")[:10]


def key_of(snapshot: dict) -> tuple[str, str, str]:
    return (str(snapshot.get("ticker") or ""), _d(snapshot.get("entry_date")), str(snapshot.get("reject_reason") or ""))


def shadow_id(snapshot: dict) -> str:
    return hashlib.sha1("|".join(key_of(snapshot)).encode()).hexdigest()[:16]


def gate_group(snapshot: dict) -> str:
    """Gate-Attribution aus den UNVERÄNDERTEN Reject-Daten (Mehrfach-Gates bleiben Mehrfach-Gates)."""
    reason = str(snapshot.get("reject_reason") or "")
    if reason == "final_mc_survivor":
        return "FINAL_MC_SURVIVOR"
    if reason.startswith("score_"):
        return "SCORE_REJECT"
    fg = snapshot.get("fail_gates")
    names = set((fg.values() if isinstance(fg, dict) else fg) or []) if fg else set()
    names = {str(n) for n in names if n}
    if not names:
        return "ROI_GATE_UNSPECIFIED" if reason == "roi_gate" else "OTHER"
    groups = {("ROI" if n in ROI_GATES else "LIQUIDITY" if n in LIQUIDITY_GATES
               else "EXECUTION" if n in EXECUTION_GATES else "OTHER") for n in names}
    if len(groups) > 1:
        return "MULTI_GATE"
    return {"ROI": "ROI_ONLY", "LIQUIDITY": "LIQUIDITY_ONLY", "EXECUTION": "EXECUTION_ONLY"}.get(groups.pop(), "OTHER")


def make_record(snapshot: dict, source: str, now: datetime | date) -> dict:
    entry = _d(snapshot.get("entry_date"))
    sid = shadow_id(snapshot)
    maturity = {}
    try:
        e = date.fromisoformat(entry)
        maturity = {str(h): (e + timedelta(days=h)).isoformat() for h in HORIZONS}
    except ValueError:
        pass
    da = snapshot.get("deep_analysis") or {}
    ts = (now.isoformat(timespec="seconds") if isinstance(now, datetime) else now.isoformat())
    return {"shadow_id": sid, "candidate_id": f"{entry}:{snapshot.get('ticker')}",
            "ticker": snapshot.get("ticker"), "entry_date": entry,
            "candidate_snapshot_ref": f"outputs/candidate_ledger (date={entry}, ticker={snapshot.get('ticker')})",
            "rejection_reason": snapshot.get("reject_reason"), "fail_gates": snapshot.get("fail_gates"),
            "gate_group": gate_group(snapshot),
            "score": {"trade_score": snapshot.get("trade_score"), "roi_net": snapshot.get("roi_net"),
                      "roi_hurdle": snapshot.get("roi_hurdle"),
                      "target_hit_score": (snapshot.get("simulation") or {}).get("hit_rate")},
            "direction": da.get("direction") or "BULLISH", "strategy": snapshot.get("strategy") or "",
            "required_horizons": list(HORIZONS), "maturity": maturity,
            "source": source, "created_at": ts, "snapshot": snapshot}


# ── Persistenz ───────────────────────────────────────────────────────────────
def _records(ledger_dir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for f in sorted((ledger_dir / "records").glob("*.jsonl")):
        for r in read_jsonl(f):
            out.setdefault(r["shadow_id"], r)            # erster Snapshot gilt (unveränderlich)
    return out


def _events(ledger_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for f in sorted((ledger_dir / "events").glob("*.jsonl")):
        rows += read_jsonl(f)
    return rows


def register(snapshots: list[dict], source: str, now: datetime | date, ledger_dir: Path | None = None,
             known: dict | None = None) -> int:
    """Hängt neue Schatten-Kandidaten an (idempotent je shadow_id). Existierende Records bleiben unverändert."""
    ledger_dir = Path(ledger_dir or LEDGER_DIR)
    known = known if known is not None else _records(ledger_dir)
    by_month: dict[str, list[dict]] = {}
    for s in snapshots or []:
        if not isinstance(s, dict) or not s.get("ticker"):
            continue
        sid = shadow_id(s)
        if sid in known:
            continue
        rec = make_record(s, source, now)
        known[sid] = rec
        by_month.setdefault((rec["entry_date"] or "unknown")[:7], []).append(rec)
    for m, rows in by_month.items():
        append_jsonl(ledger_dir / "records" / f"{m}.jsonl", rows, ensure_ascii=False, default=str)
    return sum(len(v) for v in by_month.values())


def _append_events(ledger_dir: Path, rows: list[dict], today: date) -> None:
    if rows:
        append_jsonl(ledger_dir / "events" / f"{today.isoformat()[:7]}.jsonl", rows, ensure_ascii=False, default=str)


# ── Lifecycle ────────────────────────────────────────────────────────────────
def _index(events: list[dict]) -> dict[str, dict]:
    """shadow_id -> {horizon: {"final": event|None, "retries": [..]}, "archived": bool}"""
    idx: dict[str, dict] = {}
    for e in events:
        x = idx.setdefault(e["shadow_id"], {"h": {}, "archived": False})
        if e.get("kind") == "ARCHIVED":
            x["archived"] = True
            continue
        h = x["h"].setdefault(str(e.get("horizon")), {"final": None, "retries": [], "legacy": None})
        if e.get("kind") == "LEGACY_OUTCOME":
            h["legacy"] = h.get("legacy") or e
        elif e.get("kind") in RESOLVED and h["final"] is None:
            h["final"] = e
        elif e.get("kind") == "RETRY":
            h["retries"].append(e)
    return idx


def horizon_state(rec: dict, ix: dict | None, h: int, today: date) -> str:
    hx = ((ix or {}).get("h") or {}).get(str(h)) or {}
    if hx.get("final"):
        return hx["final"]["kind"]
    mat = (rec.get("maturity") or {}).get(str(h))
    if not mat:
        return "DUE"                                      # unvollständiger Snapshot -> Worker klassifiziert
    if today < date.fromisoformat(mat):
        return "PENDING"
    return "RETRY" if hx.get("retries") else "DUE"


def record_status(rec: dict, ix: dict | None, today: date) -> str:
    states = [horizon_state(rec, ix, h, today) for h in rec.get("required_horizons") or HORIZONS]
    if all(s in RESOLVED for s in states):
        if (ix or {}).get("archived"):
            return "ARCHIVED"
        return "OUTCOME_UNAVAILABLE" if all(s == "UNAVAILABLE" for s in states) else "EVALUATED"
    if "RETRY" in states:
        return "MATURED_RETRY_REQUIRED"
    if "DUE" in states:
        return "MATURED"
    if any(s in RESOLVED for s in states):
        return "PARTIALLY_EVALUATED"
    return "PENDING"


def is_resolved(snapshot_or_id, ledger_dir: Path | None = None, today: date | None = None,
                horizons: tuple[int, ...] | None = None) -> bool:
    ledger_dir = Path(ledger_dir or LEDGER_DIR)
    sid = snapshot_or_id if isinstance(snapshot_or_id, str) else shadow_id(snapshot_or_id)
    recs = _records(ledger_dir)
    if sid not in recs:
        return False
    ix = _index(_events(ledger_dir)).get(sid)
    t = today or date.today()
    return all(horizon_state(recs[sid], ix, h, t) in RESOLVED for h in (horizons or HORIZONS))


def assert_registered(snapshots: list[dict], ledger_dir: Path | None = None) -> None:
    recs = _records(Path(ledger_dir or LEDGER_DIR))
    missing = [key_of(s) for s in snapshots if shadow_id(s) not in recs]
    if missing:
        raise ShadowIntegrityError(f"{len(missing)} Schatten-Trade(s) nicht im Ledger registriert: {missing[:5]}")


def archive(snapshots: list[dict], today: date, ledger_dir: Path | None = None) -> int:
    """ARCHIVED nur für vollständig aufgelöste Records; sonst ShadowIntegrityError (nichts wird verändert)."""
    ledger_dir = Path(ledger_dir or LEDGER_DIR)
    recs, idx = _records(ledger_dir), _index(_events(ledger_dir))
    bad = []
    for s in snapshots:
        sid = shadow_id(s)
        if sid not in recs or not all(horizon_state(recs[sid], idx.get(sid), h, today) in RESOLVED
                                      for h in recs[sid].get("required_horizons") or HORIZONS):
            bad.append(key_of(s))
    if bad:
        raise ShadowIntegrityError(f"Archivierung ungeklärter Schatten-Records verweigert: {bad[:5]} "
                                   f"(n={len(bad)}) – No unresolved shadow record may be deleted or archived "
                                   f"before all required horizons are resolved.")
    rows = [{"shadow_id": shadow_id(s), "kind": "ARCHIVED", "at": today.isoformat()} for s in snapshots
            if not (idx.get(shadow_id(s)) or {}).get("archived")]
    _append_events(ledger_dir, rows, today)
    return len(rows)


# ── Outcome-Worker ───────────────────────────────────────────────────────────
PriceHistoryFn = Callable[[str, str, str], "list[tuple[str, float]] | None"]
OptionOutcomeFn = Callable[[dict], "tuple[float | None, str]"]


def _underlying_outcome(rec: dict, closes: list[tuple[str, float]], maturity: str) -> dict | None:
    path = [(d, float(c)) for d, c in closes if rec["entry_date"] <= d <= maturity and c and c > 0]
    if not path:
        return None
    ref = (rec["snapshot"].get("simulation") or {}).get("current_price")
    src = "snapshot.simulation.current_price"
    if not (isinstance(ref, (int, float)) and ref > 0):
        ref, src = path[0][1], "entry_day_close"
    after = [c for d, c in path if d > rec["entry_date"]] or [path[-1][1]]
    sign = -1.0 if (str(rec.get("direction")).upper() == "BEARISH" or "PUT" in str(rec.get("strategy")).upper()) else 1.0
    rets = [sign * (c / ref - 1) for c in after]
    return {"underlying_ref_price": round(float(ref), 4), "ref_price_source": src,
            "underlying_close_at_maturity": round(path[-1][1], 4), "last_close_date": path[-1][0],
            "underlying_return": round(rets[-1], 6), "mfe": round(max(rets), 6), "mae": round(min(rets), 6)}


def run_worker(today: date, price_history_fn: PriceHistoryFn, option_outcome_fn: OptionOutcomeFn | None = None,
               ledger_dir: Path | None = None, legacy_horizon: int = 45, now: datetime | None = None) -> dict:
    """Bewertet alle fälligen, ungeklärten Horizonte (idempotent, atomar je Batch)."""
    ledger_dir = Path(ledger_dir or LEDGER_DIR)
    now = now or datetime.combine(today, datetime.min.time())
    recs, idx = _records(ledger_dir), _index(_events(ledger_dir))
    new: list[dict] = []
    summary = {"evaluated": 0, "retry": 0, "unavailable": 0, "legacy": {}}
    price_cache: dict[tuple[str, str, str], object] = {}
    for sid, rec in recs.items():
        ix = idx.get(sid)
        for h in rec.get("required_horizons") or HORIZONS:
            st = horizon_state(rec, ix, h, today)
            if st not in ("DUE", "RETRY"):
                continue
            mat = (rec.get("maturity") or {}).get(str(h))
            retries = (((ix or {}).get("h") or {}).get(str(h)) or {}).get("retries") or []
            base = {"shadow_id": sid, "horizon": h, "maturity_date": mat, "evaluated_at": now.isoformat(timespec="seconds")}
            if not mat or not rec.get("ticker"):
                new.append({**base, "kind": "UNAVAILABLE", "reason": "incomplete_snapshot (ticker/entry_date fehlt)"})
                summary["unavailable"] += 1
                continue
            ck = (rec["ticker"], rec["entry_date"], mat)
            try:
                if ck not in price_cache:
                    price_cache[ck] = price_history_fn(rec["ticker"], rec["entry_date"], mat)
                closes = price_cache[ck]
                und = _underlying_outcome(rec, closes or [], mat) if closes else None
                if und is None:
                    raise LookupError("keine Schlusskurse bis Fälligkeit")
            except Exception as e:  # noqa: BLE001 – temporär: Retry, nie archivieren
                n = len(retries) + 1
                lag = (today - date.fromisoformat(mat)).days
                if n >= MAX_RETRIES and lag >= MIN_DAYS_BEFORE_UNAVAILABLE:
                    new.append({**base, "kind": "UNAVAILABLE", "reason": f"Kursdaten nach {n} Versuchen nicht verfügbar: {e}"})
                    summary["unavailable"] += 1
                else:
                    new.append({**base, "kind": "RETRY", "retry_count": n, "last_error": str(e)[:200],
                                "last_attempt": today.isoformat(), "next_attempt": (today + timedelta(days=1)).isoformat()})
                    summary["retry"] += 1
                continue
            lag = (today - date.fromisoformat(mat)).days
            opt_ret, opt_method, opt_note = None, None, None
            legacy_ev = (((ix or {}).get("h") or {}).get(str(h)) or {}).get("legacy")
            if legacy_ev is not None:                     # vor dem Archivieren bereits bewertet (Original)
                opt_ret, opt_method = legacy_ev.get("option_return"), legacy_ev.get("option_method")
                opt_note = f"Legacy-Outcome vom {legacy_ev.get('evaluated_at')} (zurückgeführt, unverändert)"
            elif option_outcome_fn is not None and lag <= LIVE_QUOTE_MAX_LAG_DAYS:
                try:
                    opt_ret, opt_method = option_outcome_fn(rec["snapshot"])
                except Exception as e:  # noqa: BLE001
                    opt_note = f"Quote-Fehler: {e}"[:200]
            elif lag > LIVE_QUOTE_MAX_LAG_DAYS:
                opt_note = f"Bewertung {lag} T nach Fälligkeit: keine historischen Quotes -> nur Underlying"
            from modules.outcomes import RELIABLE_OUTCOME_METHODS
            ev = {**base, "kind": "EVALUATED", **und, "eval_lag_days": lag,
                  "option_return": None if opt_ret is None else round(float(opt_ret), 6),
                  "option_method": opt_method, "option_note": opt_note,
                  "option_reliable": bool(opt_method in RELIABLE_OUTCOME_METHODS) if opt_ret is not None else False,
                  "outcome_quality": ("LEGACY_RECOVERED" if legacy_ev is not None else
                                      "LIVE_QUOTE" if opt_ret is not None else "UNDERLYING_ONLY")}
            new.append(ev)
            summary["evaluated"] += 1
            if h == legacy_horizon and opt_ret is not None and legacy_ev is None:
                summary["legacy"][sid] = {"outcome": round(float(opt_ret), 4), "outcome_method": opt_method or "unknown",
                                          "close_date": today.isoformat()}
    _append_events(ledger_dir, new, today)
    return summary


# ── Recovery ─────────────────────────────────────────────────────────────────
def recover(snapshots: list[dict], source: str, today: date, ledger_dir: Path | None = None,
            legacy_horizon: int = 45) -> dict:
    """Archiv-/Git-Records unverändert zurückführen. Bereits vorhandene Legacy-Outcomes (45-T-Bewertung vor dem
    Archivieren) werden als LEGACY_OUTCOME mitgeführt – der Horizont wird trotzdem regulär (Underlying, PIT)
    bewertet und übernimmt das Legacy-Options-Outcome unverändert. Nie geschätzt."""
    ledger_dir = Path(ledger_dir or LEDGER_DIR)
    known = _records(ledger_dir)
    before = set(known)
    n_new = register(snapshots, source, today, ledger_dir, known=known)
    idx = _index(_events(ledger_dir))
    rows = []
    for s in snapshots:
        sid = shadow_id(s)
        if s.get("outcome") is None or sid in before:
            continue
        hx = ((idx.get(sid) or {}).get("h") or {}).get(str(legacy_horizon)) or {}
        if hx.get("final") or hx.get("legacy"):
            continue
        rows.append({"shadow_id": sid, "horizon": legacy_horizon, "kind": "LEGACY_OUTCOME",
                     "evaluated_at": str(s.get("close_date") or "")[:10], "option_return": s.get("outcome"),
                     "option_method": s.get("outcome_method") or "unknown", "source": source})
    _append_events(ledger_dir, rows, today)
    return {"registered": n_new, "legacy_outcomes": len(rows)}


# ── Health / Reporting ───────────────────────────────────────────────────────
def health(today: date, ledger_dir: Path | None = None, archive_path: Path | None = None,
           view: list[dict] | None = None) -> dict:
    ledger_dir = Path(ledger_dir or LEDGER_DIR)
    recs, events = _records(ledger_dir), _events(ledger_dir)
    idx = _index(events)
    counts: dict[str, int] = {k: 0 for k in ("PENDING", "MATURED", "MATURED_RETRY_REQUIRED", "PARTIALLY_EVALUATED",
                                             "EVALUATED", "OUTCOME_UNAVAILABLE", "ARCHIVED")}
    oldest, due, overdue = None, 0, 0
    for sid, rec in recs.items():
        st = record_status(rec, idx.get(sid), today)
        counts[st] = counts.get(st, 0) + 1
        if st not in ("EVALUATED", "OUTCOME_UNAVAILABLE", "ARCHIVED"):
            age = (today - date.fromisoformat(rec["entry_date"])).days if rec.get("entry_date") else None
            oldest = age if oldest is None or (age is not None and age > oldest) else oldest
        for h in rec.get("required_horizons") or HORIZONS:
            hs = horizon_state(rec, idx.get(sid), h, today)
            if hs in ("DUE", "RETRY"):
                due += 1
                mat = (rec.get("maturity") or {}).get(str(h))
                if mat and (today - date.fromisoformat(mat)).days > LIVE_QUOTE_MAX_LAG_DAYS:
                    overdue += 1
    # LOST_BEFORE_EVALUATION: aus der Ansicht verdrängte (archivierte) Schatten-Trades ohne Ledger-Record.
    # Einträge, die noch in der history.json-Ansicht stehen, sind nicht verloren (nur noch nicht registriert).
    lost = 0
    if archive_path and Path(archive_path).exists():
        lost = sum(1 for s in read_jsonl(archive_path) if isinstance(s, dict) and s.get("ticker") and shadow_id(s) not in recs)
    unregistered_view = sum(1 for s in view or [] if isinstance(s, dict) and s.get("ticker") and shadow_id(s) not in recs)
    week_ago = (today - timedelta(days=7)).isoformat()
    return {"records": len(recs), "status": counts, "lost_before_evaluation": lost,
            "unregistered_in_view": unregistered_view,
            "oldest_unresolved_age_days": oldest, "due_outcomes": due, "overdue_outcomes": overdue,
            "evaluations_this_week": sum(1 for e in events if e.get("kind") == "EVALUATED"
                                         and str(e.get("evaluated_at", ""))[:10] >= week_ago),
            "recovered_records": sum(1 for r in recs.values() if str(r.get("source", "")).startswith("recovery"))}


def gate_learning(today: date, ledger_dir: Path | None = None, horizon: int = 45, min_n: int = GATE_MIN_N) -> dict:
    """Outcome-Statistik je Gate-Gruppe (Underlying, richtungsbereinigt). Keine Schwellenempfehlung."""
    ledger_dir = Path(ledger_dir or LEDGER_DIR)
    recs, idx = _records(ledger_dir), _index(_events(ledger_dir))
    groups: dict[str, list[dict]] = {}
    for sid, rec in recs.items():
        ev = ((((idx.get(sid) or {}).get("h") or {}).get(str(horizon))) or {}).get("final")
        groups.setdefault(rec.get("gate_group", "OTHER"), [])
        if ev and ev.get("kind") == "EVALUATED" and isinstance(ev.get("underlying_return"), (int, float)):
            groups[rec.get("gate_group", "OTHER")].append(ev)
    out = {}
    for g, evs in sorted(groups.items()):
        n = len(evs)
        if n < min_n:
            out[g] = {"n": n, "status": "NEED_MORE_DATA"}
            continue
        r = [e["underlying_return"] for e in evs]
        out[g] = {"n": n, "status": "OK", "win_rate": round(sum(x > 0 for x in r) / n, 3),
                  "expectancy": round(statistics.mean(r), 4),
                  "mfe": round(statistics.mean(e.get("mfe") or 0 for e in evs), 4),
                  "mae": round(statistics.mean(e.get("mae") or 0 for e in evs), 4)}
    return {"horizon_days": horizon, "basis": "underlying_return (richtungsbereinigt)", "min_n": min_n, "groups": out}


def resolved_ids(today: date, horizons: tuple[int, ...] = HORIZONS, ledger_dir: Path | None = None) -> set[str]:
    """shadow_ids, deren angegebene Horizonte alle aufgelöst sind (EVALUATED/UNAVAILABLE)."""
    ledger_dir = Path(ledger_dir or LEDGER_DIR)
    recs, idx = _records(ledger_dir), _index(_events(ledger_dir))
    return {sid for sid, rec in recs.items()
            if all(horizon_state(rec, idx.get(sid), h, today) in RESOLVED for h in horizons)}

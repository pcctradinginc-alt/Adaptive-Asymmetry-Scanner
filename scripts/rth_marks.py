"""scripts/rth_marks.py – Options-Marks/Quotes zu einem gültigen Handelszeitpunkt sichern (Outcome-Kreislauf).

Hintergrund: GitHub-Crons starten hier 4–7 h verspätet (Scanner real ~17–21 UTC, Feedback ~20–24 UTC) und die
US-Sitzung verschiebt sich mit der Zeitumstellung (Sommer 13:30–20:00 UTC, Winter 14:30–21:00 UTC). Ein fester
UTC-Zeitpunkt trifft die Sitzung daher nicht verlässlich. Lösung (2026-10-11):
  - Zeitfenster aus dem NYSE-Kalender (modules.market_snapshot: Feiertage, Sonderschließungen, verkürzte Sitzungen,
    DST über zoneinfo): frühestens MIN_AFTER_OPEN nach Eröffnung (Eröffnungsphase mit breitesten Spreads meiden),
    spätestens MIN_BEFORE_CLOSE vor dem Schluss.
  - Erfassung und Mark-Berechnung sind entkoppelt: Quotes landen im append-only Archiv; candidate_ledger nutzt
    außerhalb der Sitzung die archivierten RTH-Quotes (frühester Handelstag ab Fälligkeit).

Modi (beide idempotent, nie entscheidungswirksam, history.json bleibt unverändert – Exits entscheidet feedback.py):
  voll (Scanner-Job, history-write):   candidate_ledger.update_outcomes(heute) + Quotes aktiver Champion-Kontrakte.
  --capture-only (options_capture.yml): NUR Quotes archivieren – aktive Champion-Legs und fällige, offene
                                       Kandidaten-Marks, die heute noch keine verwertbare RTH-Quote haben.
                                       Schreibt ausschließlich outputs/options_archive/ (Union-Merge, eigene
                                       Concurrency-Gruppe, kein Konflikt mit Scanner/Feedback).
Außerhalb des Fensters: nichts tun (Exit 0) – nie ein Modellpreis als Ersatz.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules import option_quotes as _oq  # noqa: E402

log = logging.getLogger("rth_marks")
HISTORY = Path("outputs/history.json")
MIN_AFTER_OPEN = _oq.MARK_MIN_AFTER_OPEN          # eine Definition: option_quotes.mark_window
MIN_BEFORE_CLOSE = _oq.MARK_MIN_BEFORE_CLOSE


def capture_window(now: datetime) -> tuple[bool, str]:
    """(im Fenster?, Status): MARKET_CLOSED | BEFORE_WINDOW | AFTER_WINDOW | IN_WINDOW (NYSE-Kalender, DST-fest;
    dasselbe Fenster wie Live-Marks und verwertbare Archiv-Quotes)."""
    now = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    status = _oq.mark_window(now)
    return status == "IN_WINDOW", status


def _read_history(path: Path) -> dict:
    """history.json lesen; defekt/fehlend -> {} mit Warnung (Capture der Kandidaten-Marks läuft trotzdem)."""
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, ValueError) as e:
        print(f"::warning::rth_marks: {path} nicht lesbar ({type(e).__name__}) – aktive Trades ohne Capture")
        return {}


def active_trade_symbols(history: dict) -> list[str]:
    """OCC-Symbole aller Legs aktiver Champion-Trades (fehlendes Symbol -> aus Strike/Verfall rekonstruiert)."""
    from modules import timing_policy as tp
    return sorted({g["symbol"] for t in history.get("active_trades") or [] for g in tp.trade_legs(t) if g["symbol"]})


def _day_start(now: datetime) -> datetime:
    from modules import market_snapshot as ms
    d = now.astimezone(ms.NY_TZ).date()
    return datetime.combine(d, datetime.min.time(), tzinfo=ms.NY_TZ).astimezone(timezone.utc)


def trade_targets(history: dict, now: datetime) -> dict[str, datetime]:
    """Symbol -> frühester verwertbarer Abrufzeitpunkt für Champion-/Counterfactual-Legs: Tagesbeginn (NY); bei
    timing-v2 nie vor dem Entscheid eines noch offenen Entries bzw. dem Exit-Entscheid eines ausstehenden Exits
    (eine frühere Quote wäre keine zulässige Ausführung)."""
    from modules import timing_policy as tp
    day0, out = _day_start(now), {}

    def add(t: dict, after: datetime) -> None:
        for g in tp.trade_legs(t):
            if g["symbol"]:
                out[g["symbol"]] = max(out.get(g["symbol"], day0), after)
    for key in ("active_trades", "counterfactual_trades"):
        for t in history.get(key) or []:
            pending_entry = tp.policy_of(t) == tp.V2 and ((t.get("timing") or {}).get("entry") or {}).get("status") \
                != "FILLED"
            add(t, max(day0, tp._ts(t.get("decision_time")) or day0) if pending_entry else day0)
    for key in ("exit_pending_trades", "counterfactual_exit_pending"):
        for t in history.get(key) or []:
            add(t, max(day0, tp._ts(t.get("close_time")) or day0))
    return out


def _satisfied(targets: dict[str, datetime], now: datetime) -> set[str]:
    """Symbole, die am heutigen NY-Handelstag schon eine verwertbare RTH-Quote mit Abruf >= ihrem Mindestzeitpunkt
    im Archiv haben."""
    from modules import option_quotes as oq
    today = oq.ny_date(now)
    hist = oq.rth_quote_history(list(targets), now, lookback_days=1)
    return {s for s, rows in hist.items()
            if any(day == today and oq._parse_ts(r.get("retrieved_at")) >= targets[s] for day, r in rows)}


def _captured_today(symbols: list[str], now: datetime) -> set[str]:
    """Symbole mit verwertbarer RTH-Quote am heutigen NY-Handelstag (ohne Mindestzeitpunkt)."""
    day0 = _day_start(now)
    return _satisfied({s: day0 for s in symbols}, now)


def run(now: datetime | None = None, history_path: Path = HISTORY, *, capture_only: bool = False) -> dict:
    from modules import candidate_ledger as cl
    from modules import market_snapshot as ms
    fixed_clock = now is not None
    now = now or datetime.now(timezone.utc)
    ok, window = capture_window(now)
    res = {"mode": "capture" if capture_only else "full", "window": window,
           "session": ms.us_market_session(now), "skipped": not ok}
    if not ok:
        return res
    history = _read_history(history_path)
    targets = trade_targets(history, now)
    trade_syms = sorted(targets)
    if capture_only:
        day0 = _day_start(now)
        cand_t = {c: day0 for c in cl.due_mark_symbols(now.strftime("%Y-%m-%d"))}
        for c, sig in cl.pending_entry_targets(now.strftime("%Y-%m-%d")).items():   # timing-v2: nie vor dem Signal
            cand_t[c] = max(cand_t.get(c, day0), sig)
        # Research-only shadow contracts: no new LLM or scanner decisions.
        # Capped in roi_learning_shadow at 30 extra OCC symbols per capture.
        try:
            from modules import roi_learning_shadow as rls
            for sym, earliest in rls.due_capture_targets(today=now.astimezone(ms.NY_TZ).date()).items():
                cand_t[sym] = max(cand_t.get(sym, day0), earliest)
        except Exception as exc:
            log.warning("roi_learning_shadow capture targets unavailable: %s", exc)
        # Alpha-to-Options link (research/paper only, config/alpha_option_link.yaml): binding RTH decisions for
        # frozen hypothesis signals (archives the observed chain quotes itself, capped per run) and exit-quote
        # targets of open paper positions (capped). Never an order; never breaks the capture.
        try:
            from modules.autonomous_alpha import option_link as al
            res["alpha_link"] = al.decide(now=now if fixed_clock else None)
            for sym, earliest in al.due_capture_targets(today=now.astimezone(ms.NY_TZ).date()).items():
                cand_t[sym] = max(cand_t.get(sym, day0), earliest)
        except Exception as exc:  # noqa: BLE001 – research add-on, never a capture condition
            log.warning("alpha option link unavailable: %s", exc)
            res["alpha_link"] = {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"[:300]}
        cand_syms = sorted(set(cand_t) - set(trade_syms))
        done = _satisfied({**{c: cand_t[c] for c in cand_syms}, **targets}, now)
        todo_t = [s for s in trade_syms if s not in done]
        todo_c = [s for s in cand_syms if s not in done]
        got = {}
        if todo_t:
            got.update(ms.fetch_option_quotes(todo_t, context="trade_mark_rth"))
        if todo_c:
            got.update(ms.fetch_option_quotes(todo_c, context="candidate_mark_rth"))
        res.update(targets=len(trade_syms) + len(cand_syms), already_captured=len(done),
                   requested=len(todo_t) + len(todo_c), quotes=len(got),
                   missing=sorted(set(todo_t + todo_c) - set(got)))
        return res
    cl.update_outcomes(now.strftime("%Y-%m-%d"))
    quotes = ms.fetch_option_quotes(trade_syms, context="trade_mark_rth") if trade_syms else {}
    res.update(active_contracts=len(trade_syms), quotes=len(quotes), missing=sorted(set(trade_syms) - set(quotes)))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-only", action="store_true",
                    help="nur Quotes archivieren (kein Ledger-Schreiben; options_capture.yml)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    try:
        res = run(capture_only=a.capture_only)
    except Exception as e:  # noqa: BLE001 – sichtbar rot; im Scanner-Job continue-on-error (nie Scanner-Bedingung)
        print(f"::error::rth_marks fehlgeschlagen: {type(e).__name__}: {e}")
        return 1
    print(json.dumps(res, ensure_ascii=False))
    if not res.get("skipped") and res.get("missing"):
        print(f"::warning::{len(res['missing'])} Kontrakt(e) ohne Quote: {res['missing'][:10]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

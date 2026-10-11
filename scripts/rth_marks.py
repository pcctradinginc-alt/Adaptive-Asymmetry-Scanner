"""scripts/rth_marks.py – Options-Marks im Fenster der regulären US-Session sichern (Outcome-Kreislauf, 2026-10-11).

Problem: Real-Options-Marks (candidate_ledger real_opt_ret_*/real_strat_ret_*) werden nur in der regulären Session
gesetzt. Die GitHub-Crons starten 4–7 h verspätet; der Feedback-Schritt läuft dadurch meist NACH Handelsschluss ->
fällige Marks blieben liegen (beobachtet: Marks nur am 2026-10-06/07). Dieser Schritt läuft am ANFANG des
Scanner-Jobs (Start ~19 UTC, noch RTH) vor der langen Pipeline.

Er ist idempotent und ändert keine Entscheidung:
  1. candidate_ledger.update_outcomes(heute): fällige Horizonte und Real-Options-Marks (je Horizont genau einmal).
  2. Quotes aller AKTIVEN Champion-Kontrakte beobachten und archivieren (Kontext trade_mark_rth); history.json bleibt
     unverändert (Exits entscheidet weiterhin feedback.py).
Außerhalb der regulären Session: nichts tun (Exit 0) – nie ein Modellpreis als Ersatz.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

log = logging.getLogger("rth_marks")
HISTORY = Path("outputs/history.json")


def active_trade_symbols(history: dict) -> list[str]:
    """OCC-Symbole aller Legs aktiver Champion-Trades (fehlendes Symbol -> aus Strike/Verfall rekonstruiert)."""
    from modules import option_quotes as oq
    out = []
    for t in history.get("active_trades") or []:
        opt = t.get("option") or {}
        right = "put" if "PUT" in str(t.get("strategy", "")).upper() or "BEAR" in str(t.get("strategy", "")).upper() \
            else "call"
        for leg in [opt] + ([opt["spread_leg"]] if opt.get("spread_leg") else []):
            sym = leg.get("option_symbol") or leg.get("symbol")
            if not sym:
                try:
                    sym = oq.occ_symbol(t.get("ticker"), leg.get("expiry") or opt.get("expiry"), right,
                                        float(leg.get("strike")))
                except (TypeError, ValueError):
                    sym = None
            if sym:
                out.append(sym)
    return sorted(set(out))


def run(now: datetime | None = None, history_path: Path = HISTORY) -> dict:
    from modules import candidate_ledger as cl
    from modules import market_snapshot as ms
    now = now or datetime.now(timezone.utc)
    session = ms.us_market_session(now)
    if session != "regular":
        return {"session": session, "skipped": True}
    cl.update_outcomes(now.strftime("%Y-%m-%d"))
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else {}
    syms = active_trade_symbols(history)
    quotes = ms.fetch_option_quotes(syms, context="trade_mark_rth") if syms else {}
    return {"session": session, "skipped": False, "active_contracts": len(syms), "quotes": len(quotes),
            "missing": sorted(set(syms) - set(quotes))}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    try:
        res = run()
    except Exception as e:  # noqa: BLE001 – Marks sind Beobachtung, nie Scanner-Bedingung
        print(f"::warning::rth_marks fehlgeschlagen: {type(e).__name__}: {e}")
        return 0
    print(json.dumps(res, ensure_ascii=False))
    if not res.get("skipped") and res.get("missing"):
        print(f"::warning::{len(res['missing'])} aktive Kontrakt(e) ohne Quote: {res['missing'][:10]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""modules/booking_labels.py – Buchungsstatus eines Trade-Vorschlags für Reports/Mail (Maintenance 2026-10-09).

Bestehende Policy (unverändert): Paper-Trade, wenn trade_score >= gates.trade_score_min (config.yaml: 55;
"Stufe 10 Trade-Score-Gate (Pipeline UND Mail)"); 30-Tage-Cooldown je Ticker. Das Grade-Label
(trade_scorer: >=75 STRONG BUY, 60–74 BUY, 45–59 WATCH) ist nur eine Bezeichnung – ein WATCH mit Score
55–59 besteht das Gate und wird gemäß Policy als Paper-Trade gebucht.
"""
from __future__ import annotations


def booking_text(p: dict) -> str:
    b = p.get("booking") or {}
    ts = p.get("trade_score") or {}
    grade = str(ts.get("grade", "") if isinstance(ts, dict) else "")
    tsm = b.get("trade_score_min")
    gate = f"Trade-Gate Score ≥ {tsm:.0f} bestanden" if isinstance(tsm, (int, float)) else "Trade-Gate bestanden"
    watch = (f" – Label WATCH (Score 45–59), {gate} → Paper-Trade gemäß Policy" if grade.startswith("WATCH") else "")
    st = b.get("status")
    if st == "BOOKED":
        return "Paper-Trade: neu gebucht" + watch
    if st == "NOT_BOOKED":
        return (f"Paper-Trade: NICHT neu gebucht – offene Position seit {b.get('open_since') or '?'} "
                f"(30-Tage-Cooldown); kein neuer, unabhängiger Trade")
    if st == "ALREADY_BOOKED_TODAY":
        return "Paper-Trade: heute bereits gebucht (kein Duplikat)"
    return "Paper-Trade: Status unbekannt"

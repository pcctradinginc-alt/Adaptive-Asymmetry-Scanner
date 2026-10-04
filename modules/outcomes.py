"""modules/outcomes.py – einzige Definition, welche Trade-Outcomes als Lern-/Evidenz-
grundlage zählen (Audit 2026-10-03).

Zuverlässig = realisierbarer Preis aus echten Quotes (Option oder Spread, inkl. Bid 0 =
-100 %). Näherungen (Delta-Approx, gekappt bei +500 %, Aktien-Fallback, unbekannt) werden
weiter geschlossen und berichtet, fließen aber in kein Lernen (Bins, Gewichte, RL) und in
keine Promotion-Evidenz ein.
"""
from __future__ import annotations

RELIABLE_OUTCOME_METHODS = frozenset({"option_quote", "option_quote_bid_zero", "spread_quote"})


def is_reliable_outcome(t: dict) -> bool:
    """Neue Trades tragen outcome_reliable/outcome_method; Altbestand gilt als zuverlässig,
    außer er wurde nachträglich rekonstruiert (Delta-Approx/Fallback, Audit 2026-09)."""
    if "outcome_reliable" in t:
        return bool(t["outcome_reliable"])
    if t.get("outcome_method"):
        return t["outcome_method"] in RELIABLE_OUTCOME_METHODS
    return not t.get("outcome_method_reconstructed")


# Vier Klassen (Audit 2026-10-04). Nur RELIABLE darf produktives Lernen beeinflussen; der Altbestand
# OHNE dokumentierte Methode wurde am 2026-10-03 als zuverlässig eingestuft (Owner-Entscheidung,
# docs/FINAL_AUDIT_2026-10-04.md) – strikt wäre er UNKNOWN. outcome_class(strict=True) zeigt das.
APPROX_METHODS = frozenset({"delta_approx", "stock_fallback", "capped", "underlying_close_daily"})


def outcome_class(t: dict, strict: bool = False) -> str:
    """RELIABLE | RECONSTRUCTED | APPROXIMATED | UNKNOWN."""
    if t.get("outcome") is None:
        return "UNKNOWN"
    m = t.get("outcome_method")
    if t.get("outcome_method_reconstructed"):
        return "RECONSTRUCTED"
    if m in RELIABLE_OUTCOME_METHODS:
        return "RELIABLE"
    if m in APPROX_METHODS or (m and "approx" in str(m)):
        return "APPROXIMATED"
    if m:
        return "UNKNOWN"
    if "outcome_reliable" in t:
        return "RELIABLE" if t["outcome_reliable"] else "APPROXIMATED"
    return "UNKNOWN" if strict else "RELIABLE"


def class_counts(trades: list[dict], strict: bool = False) -> dict:
    out: dict = {}
    for t in trades:
        k = outcome_class(t, strict)
        out[k] = out.get(k, 0) + 1
    return out

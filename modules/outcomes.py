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

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
    """Neue Trades tragen outcome_reliable/outcome_method. Altbestand OHNE dokumentierte
    Preismethode ist UNKNOWN und NICHT zuverlässig (Owner-Entscheidung 2026-10-04): er bleibt
    in history.json und in explorativen Reports, fließt aber in kein produktives Lernen,
    keine Calibration, keine Promotion-/Alpha-Evidenz und keine Meta-Learning-Evidenz ein."""
    if t.get("outcome_method_reconstructed"):
        return False                      # nachträglich geschätzt (Delta-Approx), nie Lernbasis
    if "outcome_reliable" in t:
        return bool(t["outcome_reliable"])
    if t.get("outcome_method"):
        return t["outcome_method"] in RELIABLE_OUTCOME_METHODS
    return False


# Vier Klassen (Audit 2026-10-04). Nur RELIABLE darf produktives Lernen beeinflussen. Altbestand
# OHNE dokumentierte Methode ist UNKNOWN (Owner-Entscheidung 2026-10-04, ersetzt die Einstufung
# vom 2026-10-03). `strict` bleibt nur aus Kompatibilitätsgründen und ändert nichts mehr.
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
    return "UNKNOWN"


def class_counts(trades: list[dict], strict: bool = False) -> dict:
    out: dict = {}
    for t in trades:
        k = outcome_class(t, strict)
        out[k] = out.get(k, 0) + 1
    return out

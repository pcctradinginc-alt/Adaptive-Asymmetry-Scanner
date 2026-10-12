"""modules/outcomes.py – einzige Definition, welche Trade-Outcomes als Lern-/Evidenz-
grundlage zählen (Audit 2026-10-03).

Zuverlässig = realisierbarer Preis aus echten Quotes (Option oder Spread, inkl. Bid 0 =
-100 %). Näherungen (Delta-Approx, gekappt bei +500 %, Aktien-Fallback, unbekannt) werden
weiter geschlossen und berichtet, fließen aber in kein Lernen (Bins, Gewichte, RL) und in
keine Promotion-Evidenz ein.
"""
from __future__ import annotations

# expiry_intrinsic: verfallene Option/Spread = vertraglicher Settlement-Wert aus dem offiziellen, UNbereinigten
# Schlusskurs exakt am Verfallstag (kein Modellpreis; seit 2026-10-10 eigene Methode statt "spread_quote").
RELIABLE_OUTCOME_METHODS = frozenset({"option_quote", "option_quote_bid_zero", "spread_quote", "expiry_intrinsic"})
# Preistyp je Outcome (meta["price_type"]): OBSERVED_QUOTE | SETTLEMENT_INTRINSIC | MODELED | UNDERLYING_PROXY.
# Nur OBSERVED_QUOTE und SETTLEMENT_INTRINSIC sind realisierbare Preise.
REALIZABLE_PRICE_TYPES = frozenset({"OBSERVED_QUOTE", "SETTLEMENT_INTRINSIC"})

# Produktives Gewichts-Update (feedback.compute_pearson_weights) erst ab so vielen RELIABLE-Outcomes.
MIN_RELIABLE_FOR_WEIGHT_UPDATE = 5


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


# Artefakte, die "reliable"-Kennzahlen enthalten, tragen die Definition, unter der sie gerechnet wurden.
# Reports zeigen eine Zahl nur dann als RELIABLE, wenn Definition passt und sie den aktuellen
# RELIABLE-Bestand nicht übersteigt (sonst: EXPLORATIV / veraltet, nie "zuverlässig").
RELIABILITY_DEFINITION = "outcome-classes-v3-2026-10-10"   # v3: + expiry_intrinsic (unbereinigter Settlement-Schluss)


def reliability_stamp(trades: list[dict]) -> dict:
    closed = [t for t in trades or [] if isinstance(t, dict) and t.get("outcome") is not None]
    return {"reliability_definition": RELIABILITY_DEFINITION, "outcome_classes": class_counts(closed)}


def artifact_is_current(art, n_claimed=None, current_reliable=None) -> bool:
    """True nur, wenn das Artefakt mit der gültigen Definition gerechnet wurde und – falls angegeben –
    seine RELIABLE-Population den aktuellen RELIABLE-Bestand nicht übersteigt."""
    if not isinstance(art, dict) or art.get("reliability_definition") != RELIABILITY_DEFINITION:
        return False
    if n_claimed is not None and current_reliable is not None and n_claimed > current_reliable:
        return False
    return True


# ── Bewertungsstatus geschlossener Options-Trades (Outcome-Kreislauf, 2026-10-11) ─────────────
# OK        realisierbarer Preis auf beiden Seiten (beobachtete Quote oder vertragliches Settlement)
# UNUSABLE  Outcome existiert, ist aber kein realisierbarer Preis (Modell/Proxy/unbrauchbare Quote) -> nie Evidenz
# DATA_BAD  Eingangsdaten widersprüchlich/ungültig (Entry-Debit, nicht-endlicher Outcome, unbrauchbare Entry-Quote)
# Offene Trades sind PENDING (kein Record). Ein Verlust mit Status OK ist ein echter Verlust; UNUSABLE/DATA_BAD
# sind "nicht bewertbar" – nie als Verlust oder Gewinn zu zählen.
TRADE_EVAL_VERSION = "trade-eval-v1"
_EXIT_REASON_BY_PRICE_TYPE = {"MODELED": "EXIT_PRICE_MODELED", "UNDERLYING_PROXY": "EXIT_PRICE_UNDERLYING_PROXY"}


def effective_entry_debit(trade: dict) -> float:
    """Entry-Debit je Kontrakt wie im Outcome-Pfad (einzige Definition, von feedback.compute_outcome genutzt):
    gespeichertes entry_debit > 0, sonst Spread: net_debit bzw. Long-Ask − Short-Bid, Long: Ask bzw. Last. 0 = unbekannt."""
    option = trade.get("option") or {}
    try:
        debit = float(trade.get("entry_debit") or 0)
        if debit > 0:
            return debit
        if "SPREAD" in str(trade.get("strategy", "")):
            sl, nd = option.get("spread_leg") or {}, option.get("net_debit")
            if nd:
                return float(nd)
            la, sb = float(option.get("ask", 0) or 0), float(sl.get("bid", 0) or 0)
            return round(la - sb, 2) if la > 0 and sb > 0 else la
        return float(option.get("ask", 0) or 0) or float(option.get("last", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def trade_evaluation(trade: dict, *, price_type: str | None, entry_quotes: list[dict | None] | None = None,
                     exit_quotes: list[dict | None] | None = None) -> dict:
    """Reine Funktion. entry_quotes/exit_quotes: archivierte Records je Leg (None = nicht archiviert).
    -> {version, status, reasons[], flags[], entry_session, exit_session, reproducible, rth_executable, tier}.
    tier: REALIZED_RTH_REPRODUCIBLE (OK, beide Seiten archiviert, Entry in RTH, Exit in RTH oder Settlement)
          | OBSERVED (OK, aber nicht voll reproduzierbar/RTH) | NOT_EVALUABLE."""
    import math

    from modules import option_quotes as oq
    reasons, flags = [], []
    out = trade.get("outcome")
    if effective_entry_debit(trade) <= 0:
        reasons.append("ENTRY_DEBIT_INVALID")
    try:
        out_ok = out is not None and math.isfinite(float(out))
    except (TypeError, ValueError):
        out_ok = False
    if not out_ok:
        reasons.append("OUTCOME_NOT_FINITE")
    eq = [q for q in (entry_quotes or []) if q]
    xq = [q for q in (exit_quotes or []) if q]
    if any(q.get("quality_status") == "UNUSABLE" for q in eq):
        reasons.append("ENTRY_QUOTE_UNUSABLE")
    status = "DATA_BAD" if reasons else None
    settlement = price_type == "SETTLEMENT_INTRINSIC"
    if status is None:
        if price_type not in REALIZABLE_PRICE_TYPES:
            reasons.append(_EXIT_REASON_BY_PRICE_TYPE.get(price_type, "EXIT_PRICE_UNKNOWN"))
        elif not settlement and any(q.get("quality_status") == "UNUSABLE" for q in xq):
            reasons.append("EXIT_QUOTE_UNUSABLE")
        status = "UNUSABLE" if reasons else "OK"
    n_legs = max(len(entry_quotes or []), len(exit_quotes or []), 1)
    entry_arch = bool(entry_quotes) and len(eq) == len(entry_quotes)
    exit_arch = settlement or (bool(exit_quotes) and len(xq) == len(exit_quotes))
    if not entry_arch:
        flags.append("ENTRY_QUOTE_NOT_ARCHIVED")
    if not exit_arch:
        flags.append("EXIT_QUOTE_NOT_ARCHIVED")
    if any(q.get("quality_status") == "SUSPECT" for q in eq + xq):
        flags.append("QUOTE_SUSPECT")
    es = sorted({oq.quote_session(q) for q in eq}) or ["UNKNOWN"]
    xs = ["SETTLEMENT"] if settlement else (sorted({oq.quote_session(q) for q in xq}) or ["UNKNOWN"])
    entry_session = es[0] if len(es) == 1 else "MIXED"
    exit_session = xs[0] if len(xs) == 1 else "MIXED"
    rth = entry_session == "RTH" and exit_session in ("RTH", "SETTLEMENT")
    if entry_session not in ("RTH", "UNKNOWN"):
        flags.append("ENTRY_OUTSIDE_RTH")
    if exit_session not in ("RTH", "SETTLEMENT", "UNKNOWN"):
        flags.append("EXIT_OUTSIDE_RTH")             # z. B. Schluss-Quote nach Handelsende beobachtet
    reproducible = entry_arch and exit_arch
    tier = ("NOT_EVALUABLE" if status != "OK" else
            "REALIZED_RTH_REPRODUCIBLE" if reproducible and rth else "OBSERVED")
    return {"version": TRADE_EVAL_VERSION, "status": status, "reasons": reasons, "flags": sorted(set(flags)),
            "entry_session": entry_session, "exit_session": exit_session, "legs": n_legs,
            "reproducible": reproducible, "rth_executable": rth, "tier": tier}

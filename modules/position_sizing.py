"""
Positionsgrößen-Empfehlung via Fractional Kelly.

V6 (2026-10-18): Kelly aus der simulierten Netto-P&L-Verteilung (roi_analysis.kelly_inputs):
  f* = p / L − (1 − p) / W
  p = Anteil der Pfade mit Netto-Gewinn, W = mittlerer Netto-Gewinn, L = mittlerer Netto-Verlust (<= 100 %).
Vorher f* = p − (1 − p) / b mit p = Berührwahrscheinlichkeit und b = roi_net / 0,80: roi_net ist selbst schon ein
mit p gewichteter Wert (p doppelt gezählt), p war keine Gewinnwahrscheinlichkeit. Ohne Verteilung (MC-P&L fehlte)
gibt es keine Grundlage -> Kelly 0. Maximales Risiko einer Debit-Position = 100 % des Einsatzes.

Voll-Kelly ist für Options viel zu aggressiv (Schätzfehler in p und b
multiplizieren sich) → ¼-Kelly als Default, zusätzlich Hard-Cap pro Trade.
"""

import logging
import math

from modules.config import cfg

log = logging.getLogger(__name__)


def _portfolio_cfg() -> dict:
    p = getattr(cfg, "portfolio", None)
    return {
        "size_usd":         float(getattr(p, "size_usd", 5000)),
        "kelly_fraction":   float(getattr(p, "kelly_fraction", 0.25)),
        "max_position_pct": float(getattr(p, "max_position_pct", 0.10)),
        "assumed_loss_pct": float(getattr(p, "assumed_loss_pct", 0.45)),
    }


def compute_position_size(proposal: dict) -> dict:
    """
    Berechnet die empfohlene Positionsgröße für einen Trade-Vorschlag.

    Returns dict mit:
        kelly_raw        – Voll-Kelly-Anteil (informativ)
        position_pct     – empfohlener Depot-Anteil nach ¼-Kelly + Cap
        position_usd     – Dollar-Budget
        contracts        – ganze Kontrakte, die ins Budget passen (kann 0 sein)
        cost_per_contract– Entry-Kosten × 100
        note             – Hinweis, z.B. wenn 1 Kontrakt das Budget sprengt
    """
    pc      = _portfolio_cfg()
    option  = proposal.get("option") or {}
    roi     = proposal.get("roi_analysis") or {}
    strategy = proposal.get("strategy", "")

    # Entry-Kosten pro Kontrakt
    if "SPREAD" in strategy and option.get("net_debit"):
        entry = float(option.get("net_debit") or 0)
    else:
        entry = float(option.get("ask") or 0)
    cost_per_contract = round(entry * 100, 2)

    ki = roi.get("kelly_inputs") or {}
    p_win, avg_win, avg_loss = (float(ki.get(k) or 0) for k in ("p_win", "avg_win", "avg_loss"))
    if not ki:
        kelly_raw, basis = 0.0, "NO_PNL_DISTRIBUTION"
    elif avg_win <= 0 or p_win <= 0:
        kelly_raw, basis = 0.0, "MC_PNL_DISTRIBUTION"
    elif avg_loss <= 0:                                        # kein Verlustpfad: Hard-Cap begrenzt
        kelly_raw, basis = 1.0, "MC_PNL_DISTRIBUTION"
    else:
        kelly_raw, basis = p_win / avg_loss - (1 - p_win) / avg_win, "MC_PNL_DISTRIBUTION"
    if not math.isfinite(kelly_raw):
        kelly_raw = 0.0

    position_pct = max(0.0, kelly_raw) * pc["kelly_fraction"]
    position_pct = min(position_pct, pc["max_position_pct"])
    position_usd = round(pc["size_usd"] * position_pct, 2)

    contracts = int(position_usd // cost_per_contract) if cost_per_contract > 0 else 0

    note = ""
    if basis == "NO_PNL_DISTRIBUTION":
        note = "Keine P&L-Verteilung (MC-P&L fehlt) — keine Kelly-Grundlage, nur beobachten."
    elif kelly_raw <= 0:
        note = "Kelly ≤ 0: Edge zu klein für eine Position — nur beobachten."
    elif contracts == 0 and cost_per_contract > 0:
        note = (
            f"1 Kontrakt (${cost_per_contract:.0f}) übersteigt das Kelly-Budget "
            f"(${position_usd:.0f}) — Trade auslassen oder bewusst Risiko erhöhen."
        )

    sizing = {
        "portfolio_usd":     pc["size_usd"],
        "kelly_raw":         round(kelly_raw, 4),
        "kelly_fraction":    pc["kelly_fraction"],
        "position_pct":      round(position_pct, 4),
        "position_usd":      position_usd,
        "cost_per_contract": cost_per_contract,
        "contracts":         contracts,
        "max_risk_usd":      round(contracts * cost_per_contract, 2),   # Debit-Position: bis 100 % Verlust
        "kelly_basis":       basis,
        "p_win":             round(p_win, 4) if ki else None,
        "note":              note,
    }
    log.info(
        f"  [{proposal.get('ticker','?')}] Sizing: Kelly={kelly_raw:.2f} → "
        f"{position_pct:.1%} (${position_usd:.0f}) → {contracts} Kontrakt(e) "
        f"à ${cost_per_contract:.0f}{' | ' + note if note else ''}"
    )
    return sizing


def apply_size_multiplier(sizing: dict, multiplier, applied_by=None) -> dict:
    """Validierte Sizing-Stufe (Adapter, SIZE_DOWN_ONLY): Position NUR verkleinern. Faktor wird auf (0, 1]
    begrenzt – ein Wert > 1 kann die Position nie vergrößern. Original bleibt nachvollziehbar erhalten."""
    try:
        m = float(multiplier)
    except (TypeError, ValueError):
        return sizing
    if not (0.0 < m < 1.0) or not sizing:
        return sizing
    cost = float(sizing.get("cost_per_contract") or 0)
    usd0, pct0 = float(sizing.get("position_usd") or 0), float(sizing.get("position_pct") or 0)
    pos_usd = min(usd0, round(usd0 * m, 2))                 # Rundung darf nie über das Original heben
    orig = int(sizing.get("contracts") or 0)
    contracts = min(int(pos_usd // cost) if cost > 0 else 0, orig)
    risk0 = float(sizing.get("max_risk_usd") or 0)
    note = sizing.get("note") or ""
    if orig > 0 and contracts == 0:                  # Abrundung: Budget reicht nach Verkleinerung für keinen Kontrakt
        note = (note + " | " if note else "") + f"Sizing-Stufe ×{m:.2f}: Budget unter 1 Kontrakt – nur beobachten."
    return {**sizing, "position_pct": min(pct0, round(pct0 * m, 4)), "position_usd": pos_usd,
            "contracts": contracts, "max_risk_usd": round(risk0 * contracts / orig, 2) if orig else 0.0, "note": note,
            "size_adjustment": {"multiplier": round(m, 4), "applied_by": applied_by or [],
                                "original_contracts": sizing.get("contracts"),
                                "original_position_usd": sizing.get("position_usd")}}


def enrich_with_sizing(proposals: list[dict]) -> list[dict]:
    for p in proposals:
        try:
            p["position_sizing"] = compute_position_size(p)
            if p.get("intelligence_size_multiplier") is not None:
                p["position_sizing"] = apply_size_multiplier(p["position_sizing"], p["intelligence_size_multiplier"],
                                                             p.get("intelligence_size_applied_by"))
        except Exception as e:
            log.warning(f"  [{p.get('ticker','?')}] Sizing-Fehler: {e}")
    return proposals

"""
Positionsgrößen-Empfehlung via Fractional Kelly.

Kelly für binäres Outcome: f* = p − (1−p)/b
  p = Gewinnwahrscheinlichkeit (MC Hit-Rate, gecappt)
  b = Gewinn/Verlust-Verhältnis (roi_net / assumed_loss bei Stop-Loss)

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

    # p: MC Hit-Rate (konservativ gecappt — Modell-Cap liegt bei 75%)
    p_win = float(
        proposal.get("mc_hit_rate")
        or (proposal.get("simulation") or {}).get("hit_rate")
        or 0.5
    )
    p_win = max(0.05, min(p_win, 0.75))

    # b: Gewinn/Verlust-Verhältnis
    roi_net = float(roi.get("roi_net") or 0)
    loss    = pc["assumed_loss_pct"]
    b       = max(roi_net, 0.01) / loss

    kelly_raw = p_win - (1 - p_win) / b
    if not math.isfinite(kelly_raw):
        kelly_raw = 0.0

    position_pct = max(0.0, kelly_raw) * pc["kelly_fraction"]
    position_pct = min(position_pct, pc["max_position_pct"])
    position_usd = round(pc["size_usd"] * position_pct, 2)

    contracts = int(position_usd // cost_per_contract) if cost_per_contract > 0 else 0

    note = ""
    if kelly_raw <= 0:
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
        "max_risk_usd":      round(contracts * cost_per_contract * loss, 2),
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
    pos_usd = round(float(sizing.get("position_usd") or 0) * m, 2)
    orig = int(sizing.get("contracts") or 0)
    contracts = min(int(pos_usd // cost) if cost > 0 else 0, orig)
    loss = _portfolio_cfg()["assumed_loss_pct"]
    note = sizing.get("note") or ""
    if orig > 0 and contracts == 0:                  # Abrundung: Budget reicht nach Verkleinerung für keinen Kontrakt
        note = (note + " | " if note else "") + f"Sizing-Stufe ×{m:.2f}: Budget unter 1 Kontrakt – nur beobachten."
    return {**sizing, "position_pct": round(float(sizing.get("position_pct") or 0) * m, 4), "position_usd": pos_usd,
            "contracts": contracts, "max_risk_usd": round(contracts * cost * loss, 2), "note": note,
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

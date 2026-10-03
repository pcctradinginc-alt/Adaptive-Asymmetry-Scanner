"""
modules/abstention_intelligence.py – Risikovektor je Champion-Trade (Abstention Intelligence, SHADOW).

Neben P(Trade gelingt) schätzt das System, WIE SEHR es dem eigenen Urteil im konkreten Fall
trauen darf. Jede Komponente liegt in [0, 1] (1 = maximales Risiko) oder ist None (unbekannt,
nie 0):

  p_model_wrong            1 − realisierte Win Rate des MC-Hit-Rate-Bands auf echten Paper-Trades
                           (paper_performance_analysis; erst ab n >= MIN_BAND_N)
  data_quality             1 − Data Quality des kanonischen SystemState
  regime_mismatch          Drift-Stufe (NORMAL 0 · MILD 1/3 · MODERATE 2/3 · SEVERE 1)
  unknown_risk             Titel liegt in einem gemessenen Blind-Spot-Sektor (0/1)
  model_disagreement       Perzentil der ML-Modelluneinigkeit des Titels unter allen ML-Karten
  counterfactual_fragility Anteil der Champion-Gates, die nur knapp (< FRAGILE_MARGIN relativ)
                           bestanden wurden – kleine Änderungen hätten die Entscheidung gekippt
  alpha_decay              Anteil der Research-Modelle mit signifikant fallender Prognosekraft

abstain_score = Mittel der bekannten Komponenten (gleich gewichtet, bewusst NICHT trainiert).

Wirkung: keine. Die Komponenten werden als `risk_*`-Merkmale in die Trade-Features und das
Decision-Ledger geschrieben. Dadurch kann `abstention_proposals` sie auf echten Outcomes per
Walk-Forward prüfen; Einfluss entsteht ausschließlich über einen registrierten Vertrag ->
PromotionController -> ProductionIntelligenceAdapter.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

PAPER_ANALYSIS = Path("outputs/research/paper_performance_analysis.json")
MIN_BAND_N = 10
FRAGILE_MARGIN = 0.10
DRIFT_RISK = {"NORMAL": 0.0, "MILD": 1 / 3, "MODERATE": 2 / 3, "SEVERE": 1.0}
COMPONENTS = ("p_model_wrong", "data_quality", "regime_mismatch", "unknown_risk", "model_disagreement",
              "counterfactual_fragility", "alpha_decay")


def _num(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def load_context_extras(ctx: dict) -> dict:
    """Ergänzt den Adapter-Kontext einmal je Lauf um Kalibrierung, SystemState und Disagreement-Verteilung."""
    extras = {"mc_calibration": None, "data_quality": None, "drift_level": ctx.get("drift_level"),
              "share_deteriorating": None, "disagreement_dist": []}
    try:
        extras["mc_calibration"] = json.loads(PAPER_ANALYSIS.read_text()).get("mc_hit_rate_calibration")
    except (OSError, ValueError) as e:
        log.debug(f"abstention_intelligence: Kalibrierung nicht lesbar ({e})")
    try:
        from modules import system_state as ss
        st = json.loads(Path(ss.STATE).read_text())
        extras["data_quality"] = _num((st.get("data_health") or {}).get("data_quality"))
        extras["share_deteriorating"] = _num((((st.get("drift_state") or {}).get("components") or {})
                                              .get("model") or {}).get("share_deteriorating"))
        extras["drift_level"] = extras["drift_level"] or (st.get("drift_state") or {}).get("level")
    except (OSError, ValueError, ImportError) as e:
        log.debug(f"abstention_intelligence: SystemState nicht lesbar ({e})")
    extras["disagreement_dist"] = sorted(v for v in (_num((c or {}).get("model_disagreement_sd"))
                                                    for c in (ctx.get("ml_cards") or {}).values()) if v is not None)
    return extras


def _band(mc_hit: float | None) -> str | None:
    if mc_hit is None:
        return None
    return "<0.55" if mc_hit < 0.55 else "0.55-0.65" if mc_hit < 0.65 else "0.65-0.75" if mc_hit < 0.75 else ">=0.75"


def _rel_margin(value, threshold) -> float | None:
    v, t = _num(value), _num(threshold)
    if v is None or t is None or t == 0:
        return None
    return (v - t) / abs(t)


def fragility(p: dict, trade_score_min: float | None, mc_threshold: float | None) -> float | None:
    """Anteil knapp bestandener Gates (bestanden, aber relativer Abstand < FRAGILE_MARGIN)."""
    roi = p.get("roi_analysis") or {}
    margins = [
        _rel_margin((p.get("trade_score") or {}).get("total"), trade_score_min),
        _rel_margin(p.get("mc_hit_rate") or (p.get("simulation") or {}).get("hit_rate"), mc_threshold),
        _rel_margin(roi.get("roi_net"), roi.get("min_roi_threshold")),
    ]
    known = [m for m in margins if m is not None and m >= 0]
    if not known:
        return None
    return round(sum(1 for m in known if m < FRAGILE_MARGIN) / len(known), 3)


def risk_vector(p: dict, env: dict, ctx: dict, extras: dict, *, trade_score_min: float | None = None,
                mc_threshold: float | None = None) -> dict:
    mc = _num(env.get("champion_probability"))
    band = ((extras.get("mc_calibration") or {}).get(_band(mc)) or {}) if mc is not None else {}
    p_wrong = (round(1 - float(band["win_rate"]), 3)
               if band and (band.get("n") or 0) >= MIN_BAND_N and _num(band.get("win_rate")) is not None else None)
    dq = extras.get("data_quality")
    dist = extras.get("disagreement_dist") or []
    sd = _num(env.get("ml_disagreement_sd"))
    disagree = round(sum(1 for x in dist if x <= sd) / len(dist), 3) if sd is not None and len(dist) >= 20 else None
    bs = env.get("blind_spot_sector_match")
    vec = {
        "p_model_wrong": p_wrong,
        "data_quality": round(1 - dq, 3) if dq is not None else None,
        "regime_mismatch": (round(DRIFT_RISK[extras["drift_level"]], 3)
                            if extras.get("drift_level") in DRIFT_RISK else None),
        "unknown_risk": float(bs) if bs is not None else None,
        "model_disagreement": disagree,
        "counterfactual_fragility": fragility(p, trade_score_min, mc_threshold),
        "alpha_decay": round(extras["share_deteriorating"], 3) if extras.get("share_deteriorating") is not None else None,
    }
    known = [v for v in vec.values() if v is not None]
    vec["n_known"] = len(known)
    vec["abstain_score"] = round(sum(known) / len(known), 3) if known else None
    return vec


def as_features(vec: dict) -> dict:
    """risk_*-Merkmale für Trade-Features/Verträge (None bleibt None)."""
    return {f"risk_{k}": vec.get(k) for k in COMPONENTS + ("abstain_score",)}

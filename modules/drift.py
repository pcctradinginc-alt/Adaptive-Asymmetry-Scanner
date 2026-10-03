"""modules/drift.py – abgestufte Drift (NORMAL | MILD | MODERATE | SEVERE).

Drei messbare Komponenten (Schwellen: config/drift_policy.yaml):
  feature  Regime-Merkmale gegen ihren Trainingsbereich [p01, p99] (meta_learning.json "drift"):
           Überschreitung relativ zur Spannweite; ein einzelnes leicht überschrittenes Merkmal
           ist MILD/MODERATE, nie allein SEVERE.
  model    Anteil Basismodelle mit Trend 'deteriorating' (meta_learning.json model_intelligence).
  data     gewichtete Data Quality des täglichen Source Health Checks.
Gesamt = schlechteste Komponente. Konsequenzen je Stufe (confidence_multiplier,
positive_boost_cap, allow_weight_increase, safe_mode) kommen aus derselben Datei – Verbraucher
lesen sie über den SystemState, nie direkt.
"""
from __future__ import annotations

from pathlib import Path

import yaml

POLICY = Path("config/drift_policy.yaml")
LEVELS = ("NORMAL", "MILD", "MODERATE", "SEVERE")


def load_policy(path: Path | None = None) -> dict:
    return yaml.safe_load((path or POLICY).read_text(encoding="utf-8"))


def _max(*lv: str) -> str:
    return max(lv, key=LEVELS.index) if lv else "NORMAL"


def feature_level(feature_drift: dict | None, pol: dict) -> dict:
    """feature_drift: {name: {value, p01, p99, out_of_range}} -> Stufe je Merkmal + Aggregat."""
    f = pol["feature"]
    per, out = {}, []
    for k, v in (feature_drift if isinstance(feature_drift, dict) else {}).items():
        if not isinstance(v, dict):
            continue
        val, lo, hi = v.get("value"), v.get("p01"), v.get("p99")
        if val is None or lo is None or hi is None or hi <= lo:
            per[k] = {"level": "NORMAL", "excess": None, "note": "nicht bewertbar"}
            continue
        ex = max(lo - val, val - hi, 0.0) / (hi - lo)
        lv = "NORMAL" if ex <= 0 else "MILD" if ex <= f["excess_mild"] else "MODERATE" if ex <= f["excess_moderate"] \
            else "SEVERE"
        per[k] = {"level": lv, "excess": round(ex, 4), "value": val, "p01": lo, "p99": hi}
        if ex > 0:
            out.append(k)
    worst = _max(*(p["level"] for p in per.values())) if per else "NORMAL"
    share = len(out) / len(per) if per else 0.0
    if worst == "SEVERE" and len(out) < f["severe_min_features"] and share < f["severe_share"]:
        worst = "MODERATE"                     # ein einzelner Ausreißer blockiert nie das System
    if share >= f["severe_share"] and len(out) >= f["severe_min_features"]:
        worst = "SEVERE"
    return {"level": worst, "features_out_of_range": out, "share_out": round(share, 3), "per_feature": per}


def model_level(model_intelligence: dict | None, pol: dict) -> dict:
    m = pol["model"]
    mi = {k: v for k, v in (model_intelligence or {}).items() if isinstance(v, dict)} \
        if isinstance(model_intelligence, dict) else {}
    if not mi:
        return {"level": "NORMAL", "share_deteriorating": None, "note": "keine Modelltrends"}
    det = sorted(k for k, v in mi.items() if (v or {}).get("trend") == "deteriorating")
    share = len(det) / len(mi)
    lv = "SEVERE" if share >= m["severe_share"] else "MODERATE" if share >= m["moderate_share"] \
        else "MILD" if share >= m["mild_share"] else "NORMAL"
    return {"level": lv, "share_deteriorating": round(share, 3), "deteriorating": det, "n_models": len(mi)}


def data_level(data_quality: float | None, pol: dict) -> dict:
    d = pol["data"]
    if data_quality is None:
        return {"level": "SEVERE", "data_quality": None, "note": "Data Health unbekannt"}
    lv = "SEVERE" if data_quality < d["severe_below"] else "MODERATE" if data_quality < d["moderate_below"] \
        else "MILD" if data_quality < d["mild_below"] else "NORMAL"
    return {"level": lv, "data_quality": data_quality}


def assess(meta_learning: dict | None, data_quality: float | None, pol: dict | None = None) -> dict:
    pol = pol or load_policy()
    drift = (meta_learning or {}).get("drift") if isinstance(meta_learning, dict) else {}
    drift = drift if isinstance(drift, dict) else {}
    comp = {"feature": feature_level(drift.get("feature_drift"), pol),
            "model": model_level((meta_learning or {}).get("model_intelligence") if isinstance(meta_learning, dict)
                                 else None, pol),
            "data": data_level(data_quality, pol)}
    overall = _max(*(c["level"] for c in comp.values()))
    reasons = [f"{k.upper()} DRIFT {c['level']}" + (
        f": {c.get('features_out_of_range')}" if k == "feature" else
        f": {c.get('share_deteriorating')} der Modelle verschlechtern sich" if k == "model" else
        f": Data Quality {c.get('data_quality')}") for k, c in comp.items() if c["level"] != "NORMAL"]
    return {"level": overall, "components": comp, "reasons": reasons, "policy_version": pol["version"],
            "consequences": pol["consequences"][overall]}

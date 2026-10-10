"""modules/expectation_alpha/future_state.py – Rate-of-Change-Satz je Indikator (nur Vergangenheit).

Wochenraster; 1M = `delta_1m_weeks`, 3M = `delta_3m_weeks` Zeilen. Definitionen (fest, dokumentiert):
  level            x_t
  delta_1m         x_t - x_{t-1M}
  delta_3m         x_t - x_{t-3M}
  velocity         delta_3m × (1M/3M)      mittlere Änderung je 1M-Fenster über 3 Monate (gleiche Einheit wie delta_1m)
  acceleration     delta_1m - velocity     letzter Monat schneller (+) / langsamer (-) als das 3M-Tempo
                                           (linear -> 0, konvex -> > 0)
  change_of_change acceleration_t - acceleration_{t-1M}
  z                (x_t - Mittel_{<t}) / Std_{<t}, expandierend, nur Werte VOR t, Mindesthistorie
  percentile       Anteil der Werte VOR t, die <= x_t sind (Mindesthistorie)
  regime_state     high/neutral/low aus z (Schwelle des World Models)
  regime_transition_probability  regime_change.transition_probability (nur bekannte Anker)
  uncertainty      1 - min(1, ||z| - Schwelle| / Schwelle)   (Konvention des World Models)
Modell-Zukunftszustand je Domäne = naive Persistenzprognose (das aktuelle Tempo der harten Daten hält an,
z. B. CPI-3M-Rate annualisiert; expectation_gap). Eine zusätzliche Extrapolation (velocity × Horizont) wird
bewusst NICHT gebildet (ohne Validierung keine Prognosemaschine); die Dynamik wird als eigene, testbare
Merkmale mitgeführt (EA003).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from modules.expectation_alpha.schemas import INSUFFICIENT_DATA, OK, UNAVAILABLE, rnd

ROC_FIELDS = ("level", "delta_1m", "delta_3m", "velocity", "acceleration", "change_of_change", "z",
              "percentile", "regime_state", "regime_transition_probability", "uncertainty")


def expanding_z(s: pd.Series, min_hist: int) -> pd.Series:
    """z nur aus Werten VOR t. (Nahezu) konstante Historie -> z fehlend (relative Toleranz gegen
    Gleitkommarauschen; nie Division durch ~0, die als ±4 erscheinen würde)."""
    mu = s.expanding(min_periods=min_hist).mean().shift(1)
    sd = s.expanding(min_periods=min_hist).std().shift(1)
    sd = sd.where(sd > 1e-9 * (mu.abs() + 1.0))
    return ((s - mu) / sd).clip(-4, 4)


def expanding_percentile(s: pd.Series, min_hist: int) -> pd.Series:
    vals = s.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    hist: list[float] = []
    for i, v in enumerate(vals):
        if not np.isnan(v) and len(hist) >= min_hist:
            out[i] = float(np.mean(np.asarray(hist) <= v))
        if not np.isnan(v):
            hist.append(v)                     # erst NACH der Bewertung: t zählt nie für sich selbst
    return pd.Series(out, index=s.index)


def roc_frame(s: pd.Series, d1: int = 4, d3: int = 13) -> pd.DataFrame:
    s = s.astype(float)
    f = pd.DataFrame(index=s.index)
    f["level"] = s
    f["delta_1m"] = s - s.shift(d1)
    f["delta_3m"] = s - s.shift(d3)
    f["velocity"] = f["delta_3m"] * (d1 / d3)       # mittlere Änderung je 1M-Fenster über die letzten 3M
    f["acceleration"] = f["delta_1m"] - f["velocity"]
    f["change_of_change"] = f["acceleration"] - f["acceleration"].shift(d1)
    return f


def roc_set(s: pd.Series, cfg: dict, *, unit: str, min_hist: int, thr: float | None = None) -> dict:
    """Vollständiger RoC-Satz am letzten Stichtag (+ Historienfenster). Fehlt x_t -> UNAVAILABLE."""
    from modules import world_model as wm
    from modules.expectation_alpha.regime_change import state_of, transition_probability
    fs = cfg.get("future_state") or {}
    thr = float(wm.WP["state_threshold"]) if thr is None else thr
    s = s.astype(float)
    hist = s.dropna()
    base = {"unit": unit, "n_history": int(len(hist)),
            "window_start": hist.index[0].date().isoformat() if len(hist) else None,
            "window_end": hist.index[-1].date().isoformat() if len(hist) else None}
    if s.empty or np.isnan(s.iloc[-1]):
        return {**base, **{k: None for k in ROC_FIELDS}, "status": UNAVAILABLE}
    rf = roc_frame(s, int(fs.get("delta_1m_weeks", 4)), int(fs.get("delta_3m_weeks", 13))).iloc[-1]
    z = expanding_z(s, min_hist)
    pct = expanding_percentile(s, min_hist)
    zl = z.iloc[-1]
    out = {**base, **{k: rnd(rf[k]) for k in ("level", "delta_1m", "delta_3m", "velocity", "acceleration",
                                               "change_of_change")}}
    out["z"] = rnd(zl, 4)
    out["percentile"] = rnd(pct.iloc[-1], 4)
    out["regime_state"] = state_of(zl, thr)
    tp = transition_probability(z, thr, int(fs.get("transition_lookahead_weeks", 4)),
                                int(fs.get("transition_min_history", 104)))
    out["regime_transition_probability"] = tp["value"]
    out["transition_detail"] = tp
    out["uncertainty"] = rnd(1.0 - min(1.0, abs(abs(zl) - thr) / thr), 3) if not np.isnan(zl) else None
    out["status"] = OK if out["z"] is not None else INSUFFICIENT_DATA
    return out

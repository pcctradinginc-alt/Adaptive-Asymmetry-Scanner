"""modules/expectation_alpha/regime_change.py – Regime aus dem bestehenden World Model + Wechselrisiko.

Kein zweites World Model: Dimensionen, z-Scores, Zustände und Unsicherheit stammen aus
`world_model.build_world`. Neu sind nur:
* der Zustandswechsel gegenüber der Vorwoche;
* die empirische Wahrscheinlichkeit eines Zustandswechsels in `lookahead` Wochen, bedingt auf den Abstand
  zur Schwelle. Sie beruht nur auf Ankerwochen, deren Ergebnis zum Stichtag bereits bekannt war;
* Velocity/Acceleration des Dimensions-Scores (future_state.roc_frame).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from modules.expectation_alpha.schemas import INSUFFICIENT_DATA, OK, UNAVAILABLE, rnd

MARGIN_BUCKETS = (0.25, 0.5, 1.0)      # |(|z| - Schwelle)| in z-Einheiten -> 4 Klassen (fest, nicht gefittet)
MIN_BUCKET_N = 20


def state_of(z, thr: float) -> str | None:
    if z is None or (isinstance(z, float) and np.isnan(z)):
        return None
    return "high" if z > thr else "low" if z < -thr else "neutral"


def _bucket(margin: float) -> int:
    for i, b in enumerate(MARGIN_BUCKETS):
        if margin < b:
            return i
    return len(MARGIN_BUCKETS)


def transition_probability(z: pd.Series, thr: float, lookahead: int, min_history: int,
                           at: int | None = None) -> dict:
    """P(Zustand in `lookahead` Wochen != heutiger Zustand | Abstandsklasse) zum Zeitpunkt `at`
    (Position; Default letzte). Anker i zählen nur, wenn i + lookahead <= at. Ihr Ergebnis war am
    Stichtag also bekannt; es gibt keinen Look-ahead."""
    vals = z.to_numpy(dtype=float)
    n = len(vals)
    at = n - 1 if at is None else at
    if at < 0 or np.isnan(vals[at]):
        return {"value": None, "status": UNAVAILABLE, "n_anchors": 0}
    cur_margin = abs(abs(vals[at]) - thr)
    cb = _bucket(cur_margin)
    anchors, same = 0, []
    for i in range(0, at - lookahead + 1):
        a, b = vals[i], vals[i + lookahead]
        if np.isnan(a) or np.isnan(b):
            continue
        anchors += 1
        if _bucket(abs(abs(a) - thr)) == cb:
            same.append(state_of(a, thr) != state_of(b, thr))
    if anchors < min_history or len(same) < MIN_BUCKET_N:
        return {"value": None, "status": INSUFFICIENT_DATA, "n_anchors": anchors, "n_bucket": len(same)}
    return {"value": rnd(float(np.mean(same)), 4), "status": OK, "n_anchors": anchors, "n_bucket": len(same),
            "margin_bucket": cb}


def build_regime(px: pd.DataFrame, archive_obs: dict, dates: list[pd.Timestamp]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """-> (Indikatoren, Zustände) des bestehenden World Models auf dem EA-Wochenraster (ohne Panel:
    Breite bleibt fehlend statt Survivorship-verzerrt)."""
    from modules import world_model as wm
    return wm.build_world(px, archive_obs, None, dates=dates)


def regime_snapshot(states: pd.DataFrame, cfg: dict) -> dict:
    """Zustand je World-Model-Dimension am letzten Stichtag + Wechselrisiko + Dynamik."""
    from modules import world_model as wm
    from modules.expectation_alpha.future_state import roc_frame
    if states is None or states.empty:
        return {"status": UNAVAILABLE, "dimensions": {}, "regime_uncertainty": None}
    fs = cfg.get("future_state") or {}
    thr = float(wm.WP["state_threshold"])
    look, mh = int(fs.get("transition_lookahead_weeks", 4)), int(fs.get("transition_min_history", 104))
    last = states.iloc[-1]
    dims = {}
    for dim in wm.DIMENSIONS:
        st = last.get(f"{dim}_state")
        if st in ("unavailable", "no_data") or st is None:
            dims[dim] = {"state": st or "no_data", "status": UNAVAILABLE,
                         "reason": wm.UNAVAILABLE_REASON.get(dim)}
            continue
        score = states[f"{dim}_score"].astype(float)
        prev = states[f"{dim}_state"].iloc[-2] if len(states) > 1 else None
        roc = roc_frame(score, int(fs.get("delta_1m_weeks", 4)), int(fs.get("delta_3m_weeks", 13))).iloc[-1]
        dims[dim] = {"state": st, "previous_state": prev, "changed": bool(prev is not None and prev != st),
                     "score": rnd(last.get(f"{dim}_score"), 4), "uncertainty": rnd(last.get(f"{dim}_uncertainty"), 3),
                     "velocity": rnd(roc["velocity"], 4), "acceleration": rnd(roc["acceleration"], 4),
                     "transition": transition_probability(score, thr, look, mh), "status": OK}
    return {"status": OK, "date": states.index[-1].date().isoformat(),
            "regime_uncertainty": rnd(last.get("uncertainty"), 3),
            "n_dims_available": int(last.get("n_dims_available") or 0), "dimensions": dims}

"""
modules/external/features.py – deterministische, ausschließlich rückwärtsgerichtete
Feature-Mathematik über (time, value)-Serien.

Grundregeln:
  - Serien sind aufsteigend nach observation_time sortiert; an Position i darf
    NUR verwendet werden, was zu Zeitpunkt time[i] bereits bekannt war (also
    Werte an Positionen <= i bzw. innerhalb eines Zeitfensters davor).
  - Klassifikationen/Scores sind methodologische Defaults (siehe config.yaml
    external_context.states) — "nicht gegen Returns optimiert".
  - Eine Formeländerung erhöht die zugehörige FEATURE_VERSION-Konstante.
"""

from __future__ import annotations

import math
import statistics
from datetime import datetime, timedelta
from typing import Iterable, Literal, Sequence

Point = tuple[datetime, float]

FEATURE_VERSION = "v1"
FEATURE_VERSIONS = {
    "rolling": "v1",
    "changes": "v1",
    "zscore": "v1",
    "state": "v1",
    "breadth": "v1",
}


def _values_up_to(series: Sequence[Point], i: int, window_n: int | None,
                   window_days: float | None) -> list[float]:
    """Werte bis inkl. Index i, begrenzt auf die letzten window_n Beobachtungen
    und/oder window_days Kalendertage (past-only)."""
    t_i = series[i][0]
    lo = 0
    if window_n is not None:
        lo = max(0, i - window_n + 1)
    vals = []
    for j in range(lo, i + 1):
        tj, vj = series[j]
        if window_days is not None and (t_i - tj) > timedelta(days=window_days):
            continue
        if vj is None:
            continue
        vals.append(vj)
    return vals


def rolling_stat(series: Sequence[Point], window_n: int | None = None,
                  window_days: float | None = None,
                  stat: Literal["mean", "median", "std", "quantile"] = "mean",
                  quantile: float = 0.5, min_periods: int = 2) -> list[float | None]:
    """Rolling mean/median/std/quantile über N Beobachtungen oder N Tage."""
    if window_n is None and window_days is None:
        raise ValueError("window_n oder window_days angeben")
    out: list[float | None] = []
    for i in range(len(series)):
        vals = _values_up_to(series, i, window_n, window_days)
        if len(vals) < max(min_periods, 1):
            out.append(None)
            continue
        if stat == "mean":
            out.append(statistics.fmean(vals))
        elif stat == "median":
            out.append(statistics.median(vals))
        elif stat == "std":
            out.append(statistics.pstdev(vals) if len(vals) >= 2 else None)
        elif stat == "quantile":
            out.append(_quantile(vals, quantile))
        else:
            raise ValueError(f"Unbekannte stat: {stat}")
    return out


def _quantile(vals: list[float], q: float) -> float:
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return s[lo]
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def pct_change(series: Sequence[Point], periods: int = 1) -> list[float | None]:
    """Prozentänderung gegenüber `periods` Beobachtungen zuvor (Index-basiert)."""
    out: list[float | None] = []
    for i in range(len(series)):
        j = i - periods
        if j < 0:
            out.append(None)
            continue
        prev = series[j][1]
        cur = series[i][1]
        if prev in (None, 0) or cur is None:
            out.append(None)
            continue
        out.append((cur - prev) / abs(prev))
    return out


def _value_near(series: Sequence[Point], i: int, delta: timedelta,
                 tolerance_days: float) -> float | None:
    """Wert mit time ~ time[i] - delta (nächstgelegener Punkt innerhalb
    tolerance_days), NUR aus Positionen <= i (past-only). Für unregelmäßige
    Serien, bei denen kein Punkt exakt auf dem Zieldatum liegt."""
    target = series[i][0] - delta
    best = None
    best_diff = None
    for j in range(i + 1):
        tj, vj = series[j]
        if vj is None:
            continue
        diff_days = abs((tj - target).total_seconds()) / 86400.0
        if diff_days <= tolerance_days and (best_diff is None or diff_days < best_diff):
            best, best_diff = vj, diff_days
    return best


def wow(series: Sequence[Point], tolerance_days: float = 2) -> list[float | None]:
    """Week-over-week: Vergleich mit Wert ~7 Tage zuvor (Datums-basiert)."""
    out = []
    for i in range(len(series)):
        prev = _value_near(series, i, timedelta(days=7), tolerance_days)
        cur = series[i][1]
        out.append(None if prev in (None, 0) or cur is None else (cur - prev) / abs(prev))
    return out


def mom(series: Sequence[Point], tolerance_days: float = 6) -> list[float | None]:
    """Month-over-month: Vergleich mit Wert ~30.4 Tage zuvor (Datums-basiert)."""
    out = []
    for i in range(len(series)):
        prev = _value_near(series, i, timedelta(days=30.4), tolerance_days)
        cur = series[i][1]
        out.append(None if prev in (None, 0) or cur is None else (cur - prev) / abs(prev))
    return out


def yoy(series: Sequence[Point], tolerance_days: float = 10) -> list[float | None]:
    """Year-over-year: NUR wenn ein Wert ~365 Tage zuvor existiert (sonst None,
    NIE über unvollständige Vorjahresdaten interpolieren)."""
    out = []
    for i in range(len(series)):
        prev = _value_near(series, i, timedelta(days=365.25), tolerance_days)
        cur = series[i][1]
        out.append(None if prev in (None, 0) or cur is None else (cur - prev) / abs(prev))
    return out


def rolling_zscore(series: Sequence[Point], window_n: int | None = None,
                    window_days: float | None = None, min_periods: int = 5) -> list[float | None]:
    """z = (x - rolling_mean) / rolling_std, nur aus VERGANGENEN Werten (Fenster
    exklusive current? Nein: inklusive, wie üblich für 'wo steht x relativ zu
    seiner jüngeren Historie')."""
    out: list[float | None] = []
    for i in range(len(series)):
        vals = _values_up_to(series, i, window_n, window_days)
        cur = series[i][1]
        if cur is None or len(vals) < min_periods:
            out.append(None)
            continue
        mean = statistics.fmean(vals)
        std = statistics.pstdev(vals) if len(vals) >= 2 else 0.0
        out.append(0.0 if std == 0 else (cur - mean) / std)
    return out


def acceleration(short_z: Sequence[float | None], medium_z: Sequence[float | None]) -> list[float | None]:
    """acceleration = z(short growth) - z(medium growth); beide Serien müssen
    gleich lang / gleich indiziert sein (z.B. rolling_zscore der WoW- bzw.
    der MoM-Wachstumsrate)."""
    out = []
    for s, m in zip(short_z, medium_z):
        out.append(None if s is None or m is None else s - m)
    return out


def drop_incomplete_last(series: Sequence[Point], period_end_rule) -> list[Point]:
    """Entfernt den letzten Punkt, wenn seine Periode laut period_end_rule(time)
    (Callable: time -> Ende-Datum der Periode) noch nicht abgeschlossen ist.
    Nie einen unvollständigen aktuellen Zeitraum mit vollständigen vergleichen."""
    if not series:
        return list(series)
    last_time = series[-1][0]
    period_end = period_end_rule(last_time)
    now = datetime.now(last_time.tzinfo) if last_time.tzinfo else datetime.now()
    if period_end > now:
        return list(series[:-1])
    return list(series)


def _states_config() -> dict:
    try:
        from modules.config import cfg
        st = getattr(getattr(cfg, "external_context", None), "states", None)
        if st:
            return dict(st)
    except Exception:
        pass
    return {}


def classify_state(z: float | None, strong: float | None = None, normal: float | None = None) -> str:
    """z >= strong → STRONG_EXPANSION; >= normal → EXPANSION; > -normal → NEUTRAL;
    > -strong → CONTRACTION; sonst STRONG_CONTRACTION."""
    if z is None:
        return "UNKNOWN"
    cfg = _states_config()
    strong = strong if strong is not None else float(cfg.get("strong", 1.5))
    normal = normal if normal is not None else float(cfg.get("normal", 0.5))
    if z >= strong:
        return "STRONG_EXPANSION"
    if z >= normal:
        return "EXPANSION"
    if z > -normal:
        return "NEUTRAL"
    if z > -strong:
        return "CONTRACTION"
    return "STRONG_CONTRACTION"


def breadth(values_z: dict[str, float | None], threshold: float, min_valid: int) -> dict:
    """Anteil negativer/positiver z-Werte über/unter threshold. breadth_valid=False
    wenn weniger als min_valid gültige (nicht-None) Werte vorliegen."""
    valid = {k: v for k, v in values_z.items() if v is not None}
    valid_count = len(valid)
    if valid_count == 0:
        return {"negative_breadth": None, "positive_breadth": None,
                "valid_count": 0, "breadth_valid": False}
    positive = sum(1 for v in valid.values() if v >= threshold)
    negative = sum(1 for v in valid.values() if v <= -threshold)
    return {
        "negative_breadth": negative / valid_count,
        "positive_breadth": positive / valid_count,
        "valid_count": valid_count,
        "breadth_valid": valid_count >= min_valid,
    }


def combine_states(entries: list[dict]) -> dict:
    """entries: [{source_id, z, state?, is_fresh (bool), age_days}, ...].
    confidence <= 0.3 wenn <=1 frische Quelle ODER alle Quellen 'stale' sind."""
    if not entries:
        return {"state": "UNKNOWN", "confidence": 0.0, "source_count": 0,
                "fresh_source_count": 0, "agreement_ratio": 0.0,
                "breadth": None, "data_age_days": None}

    zs = [e["z"] for e in entries if e.get("z") is not None]
    fresh_count = sum(1 for e in entries if e.get("is_fresh"))
    states = [classify_state(e.get("z")) for e in entries]
    non_unknown = [s for s in states if s != "UNKNOWN"]
    if non_unknown:
        from collections import Counter
        combined_state, top_n = Counter(non_unknown).most_common(1)[0]
        agreement_ratio = top_n / len(non_unknown)
    else:
        combined_state, agreement_ratio = "UNKNOWN", 0.0

    ages = [e.get("age_days") for e in entries if e.get("age_days") is not None]
    data_age_days = max(ages) if ages else None

    mean_abs_z = statistics.fmean(abs(z) for z in zs) if zs else 0.0
    base_confidence = min(1.0, agreement_ratio * min(1.0, mean_abs_z / 2.0))
    all_stale = fresh_count == 0 and len(entries) > 0
    confidence = base_confidence
    if fresh_count <= 1 or all_stale:
        confidence = min(confidence, 0.3)

    breadth_info = breadth({e.get("source_id", str(i)): e.get("z")
                             for i, e in enumerate(entries)}, threshold=0.5, min_valid=1)

    return {
        "state": combined_state,
        "confidence": round(confidence, 4),
        "source_count": len(entries),
        "fresh_source_count": fresh_count,
        "agreement_ratio": round(agreement_ratio, 4),
        "breadth": breadth_info,
        "data_age_days": data_age_days,
    }


def divergence(z_a: float | None, z_b: float | None, threshold: float = 0.5) -> dict:
    """Divergenz zwischen zwei standardisierten Serien. Reines Richtungs-/
    Stärke-Signal — KEINE Kauf/Verkauf-Semantik."""
    if z_a is None or z_b is None:
        return {"divergence_z": None, "agreement": "UNKNOWN"}
    div = z_a - z_b
    a_exp, b_exp = z_a >= threshold, z_b >= threshold
    a_con, b_con = z_a <= -threshold, z_b <= -threshold
    if a_exp and b_exp:
        agreement = "BOTH_EXPANDING"
    elif a_con and b_con:
        agreement = "BOTH_CONTRACTING"
    elif a_exp and not b_exp and not b_con:
        agreement = "A_STRONGER"
    elif b_exp and not a_exp and not a_con:
        agreement = "B_STRONGER"
    elif a_con and not b_con and not b_exp:
        agreement = "A_STRONGER" if abs(z_a) > abs(z_b) else "B_STRONGER"
    elif b_con and not a_con and not a_exp:
        agreement = "B_STRONGER" if abs(z_b) > abs(z_a) else "A_STRONGER"
    else:
        agreement = "MIXED"
    return {"divergence_z": div, "agreement": agreement}

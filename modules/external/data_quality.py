"""
modules/external/data_quality.py – Data-Quality-Gates je Abruf.

Eine Quelle gilt NICHT als erfolgreich, nur weil keine Exception auftrat
(Audit 2026-09-29: nws_alerts meldete wochenlang PASS mit 0 Zeilen). assess()
prüft jeden Abruf-Batch und liefert Kennzahlen + Befunde; der Orchestrator
stuft PASS bei schweren Befunden auf WARN herab und persistiert `dq` in der
Source-Health.

Prüfungen (Registry-Overrides in Klammern):
  EMPTY_RESULT        0 Beobachtungen (may_be_empty: true für Ereignisquellen)
  BELOW_MIN           weniger als min_observations
  HIGH_NULL_RATE      Anteil value=None > max_null_rate (Standard 0.5)
  NON_FINITE          NaN/Inf-Werte
  DUPLICATE_CONFLICT  gleiche Identität im selben Abruf mit verschiedenen Werten
  FUTURE_OBSERVATION  observation_time > jetzt + 2 Tage (außer Prognosen)
  AVAILABLE_BEFORE_PERIOD  available_at < observation_time bei Periodendaten
                      (eine Monatsstatistik kann nicht vor Monatsbeginn vorliegen)
  OUT_OF_RANGE        Wert außerhalb plausible_ranges[metric]
  SCALE_SHIFT         Median je Kennzahl weicht > 20x vom archivierten Median ab
                      (Einheitenwechsel, z.B. Tonnen -> Tausend Tonnen)
"""

from __future__ import annotations

import math
import statistics
from datetime import datetime, timedelta

from modules.external.pit import ensure_utc

SEVERE = {"EMPTY_RESULT", "BELOW_MIN", "HIGH_NULL_RATE", "NON_FINITE", "DUPLICATE_CONFLICT",
          "FUTURE_OBSERVATION", "AVAILABLE_BEFORE_PERIOD", "OUT_OF_RANGE", "SCALE_SHIFT"}
SCALE_SHIFT_FACTOR = 20.0
FUTURE_TOLERANCE = timedelta(days=2)


def _median_abs(values: list[float]) -> float | None:
    vals = [abs(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v) and v != 0]
    return statistics.median(vals) if vals else None


def assess(observations: list, source_cfg: dict | None, now: datetime,
           history: list | None = None) -> dict:
    cfg = source_cfg or {}
    now = ensure_utc(now)
    obs = list(observations or [])
    n = len(obs)
    issues: list[str] = []
    details: dict = {}

    if n == 0:
        if not cfg.get("may_be_empty"):
            issues.append("EMPTY_RESULT")
        return {"n_observations": 0, "issues": issues, "details": details,
                "severe": bool(set(issues) & SEVERE)}

    min_obs = cfg.get("min_observations")
    if min_obs and n < int(min_obs):
        issues.append("BELOW_MIN")
        details["min_observations"] = int(min_obs)

    nulls = sum(1 for o in obs if o.value is None)
    null_rate = nulls / n
    if null_rate > float(cfg.get("max_null_rate", 0.5)):
        issues.append("HIGH_NULL_RATE")

    non_finite = sum(1 for o in obs if isinstance(o.value, float) and not math.isfinite(o.value))
    if non_finite:
        issues.append("NON_FINITE")

    seen: dict = {}
    conflicts = 0
    for o in obs:
        # Vintage-Quellen (ALFRED) liefern je Periode MEHRERE Vintages mit
        # legitim verschiedenen Werten (Revisionen) -> Schlüssel inkl. vintage_time.
        k = (o.identity_key(), o.vintage_time)
        if k in seen and seen[k] != o.value:
            conflicts += 1
        seen.setdefault(k, o.value)
    if conflicts:
        issues.append("DUPLICATE_CONFLICT")

    future = [o for o in obs if o.forecast_valid_time is None and o.forecast_issue_time is None
              and ensure_utc(o.observation_time) > now + FUTURE_TOLERANCE]
    if future and not cfg.get("allow_future_observations"):
        issues.append("FUTURE_OBSERVATION")
        details["future_example"] = future[0].observation_time.isoformat()

    if cfg.get("frequency") in ("monthly", "quarterly", "annual"):
        early = [o for o in obs if o.available_at is not None
                 and ensure_utc(o.available_at) < ensure_utc(o.observation_time)]
        if early:
            issues.append("AVAILABLE_BEFORE_PERIOD")
            details["available_before_period_example"] = early[0].identity_key()

    ranges = cfg.get("plausible_ranges") or {}
    out_of_range = 0
    for o in obs:
        r = ranges.get(o.metric)
        if r and isinstance(o.value, (int, float)) and not (r[0] <= o.value <= r[1]):
            out_of_range += 1
    if out_of_range:
        issues.append("OUT_OF_RANGE")

    shifts = {}
    if history:
        by_metric_new: dict = {}
        by_metric_old: dict = {}
        for o in obs:
            by_metric_new.setdefault(o.metric, []).append(o.value)
        for o in history:
            by_metric_old.setdefault(o.metric, []).append(o.value)
        for m, vals in by_metric_new.items():
            a, b = _median_abs(vals), _median_abs(by_metric_old.get(m, []))
            if a and b and (a / b > SCALE_SHIFT_FACTOR or b / a > SCALE_SHIFT_FACTOR):
                shifts[m] = round(a / b, 3)
        if shifts:
            issues.append("SCALE_SHIFT")
            details["scale_shift_ratio"] = shifts

    return {
        "n_observations": n,
        "null_rate": round(null_rate, 4),
        "non_finite": non_finite,
        "duplicate_conflicts": conflicts,
        "future_observations": len(future),
        "out_of_range": out_of_range,
        "observation_range": [min(o.observation_time for o in obs).isoformat(),
                              max(o.observation_time for o in obs).isoformat()],
        "issues": issues,
        "details": details,
        "severe": bool(set(issues) & SEVERE),
    }

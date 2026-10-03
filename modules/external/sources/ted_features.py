"""TED-Features je (Stichtag, Emittent) – streng point-in-time.

* Nur Zuschläge mit available_at <= Stichtag 21:00 UTC.
* Echte Null nur innerhalb belegter Abdeckung: alle Kalenderjahre des Fensters wurden für
  diese Entität vollständig abgefragt (nicht abgeschnitten), liegen ab dem Jahr, ab dem TED
  Gewinnernamen zuverlässig liefert (Probe), und – im laufenden Jahr – vor dem Abrufzeitpunkt.
  Sonst NaN (+ alt_ted_available = 0).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

FEATURE_VERSION = "ted-f1"
FEATURES = {
    "ted_awards_90d": "Anzahl zugeordneter EU-Zuschläge (TED), 90 T",
    "ted_any_award_365d": "1 wenn mind. ein Zuschlag in 365 T",
    "ted_award_value_365d": "log1p(Summe EUR-Zuschlagswerte, 365 T); NaN wenn Zuschläge ohne EUR-Wert",
    "ted_awards_z": "Zuschläge 90 T gegen die vier vorangehenden 90-T-Fenster, z",
}
DAY = np.int64(86_400 * 10**9)


def _cutoff_ns(d) -> int:
    t = pd.Timestamp(d)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return (t.normalize() + pd.Timedelta(hours=21)).value


def covered(t_ns: int, days: int, years: dict, field_start_year: int | None) -> bool:
    """Fenster (t-days, t] vollständig abgedeckt?"""
    if field_start_year is None:
        return False
    t = pd.Timestamp(t_ns, tz="UTC")
    start = t - pd.Timedelta(days=days)
    for y in range(start.year, t.year + 1):
        info = years.get(str(y))
        if y < field_start_year or not info or not info.get("complete"):
            return False
        if y == t.year and pd.Timestamp(info["fetched_at"]).tz_convert("UTC") < t:
            return False
    return True


def features_cik(dates, awards: pd.DataFrame, years: dict, field_start_year: int | None) -> pd.DataFrame:
    """awards: DataFrame[avail (datetime UTC), value (float|NaN)] einer Entität."""
    T = np.array([_cutoff_ns(d) for d in dates], dtype=np.int64)
    out = {k: np.full(len(T), np.nan) for k in FEATURES}
    a = awards.sort_values("avail", kind="mergesort")
    av = pd.DatetimeIndex(pd.to_datetime(a["avail"], utc=True)).as_unit("ns").asi8.astype(np.int64) \
        if len(a) else np.array([], dtype=np.int64)
    vals = a["value"].to_numpy(dtype=float) if len(a) else np.array([], dtype=float)
    hi = np.searchsorted(av, T, "right")
    for i, t in enumerate(T):
        if covered(t, 365, years, field_start_year):
            l365 = np.searchsorted(av, t - 365 * DAY, "right")
            l90 = np.searchsorted(av, t - 90 * DAY, "right")
            out["ted_awards_90d"][i] = float(hi[i] - l90)
            n365 = hi[i] - l365
            out["ted_any_award_365d"][i] = 1.0 if n365 > 0 else 0.0
            v = vals[l365:hi[i]]
            if n365 == 0:
                out["ted_award_value_365d"][i] = 0.0
            elif np.isfinite(v).any():
                out["ted_award_value_365d"][i] = math.log1p(float(np.nansum(v)))
        if covered(t, 450, years, field_start_year):
            b = np.searchsorted(av, t - np.arange(6) * 90 * DAY, "right")
            cnt = (b[:-1] - b[1:]).astype(float)          # Fenster j: (t-90(j+1), t-90j]
            cur, hist = cnt[0], cnt[1:5]
            sd = float(hist.std())
            out["ted_awards_z"][i] = (cur - float(hist.mean())) / sd if sd > 0 else 0.0
    return pd.DataFrame(out)


def build_feature_table(obs_rows: pd.DataFrame, dates, cik_by_ticker: dict[str, str], entity_years: dict,
                        field_start_year: int | None) -> pd.DataFrame:
    """obs_rows: Event-Store-Zeilen (entity_id, available_at, value). -> [date, ticker, cik, FEATURES, alt_ted_available]."""
    dates = [pd.Timestamp(d) for d in dates]
    if len(obs_rows):
        aw = pd.DataFrame({"cik": obs_rows["entity_id"].astype(str),
                           "avail": pd.to_datetime(obs_rows["available_at"], utc=True),
                           "value": pd.to_numeric(obs_rows["value"], errors="coerce")})
    else:
        aw = pd.DataFrame(columns=["cik", "avail", "value"])
    by = dict(tuple(aw.groupby("cik"))) if len(aw) else {}
    parts = []
    for tk, cik in cik_by_ticker.items():
        f = features_cik(dates, by.get(cik, aw.iloc[:0]), entity_years.get(cik, {}), field_start_year)
        f.insert(0, "cik", cik)
        f.insert(0, "ticker", tk)
        f.insert(0, "date", dates)
        f["alt_ted_available"] = f[list(FEATURES)].notna().any(axis=1).astype(float)
        f["feature_version"] = FEATURE_VERSION
        parts.append(f)
    cols = ["date", "ticker", "cik", *FEATURES, "alt_ted_available", "feature_version"]
    return pd.concat(parts, ignore_index=True)[cols] if parts else pd.DataFrame(columns=cols)

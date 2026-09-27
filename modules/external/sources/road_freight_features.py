"""
modules/external/sources/road_freight_features.py – reine Feature-Funktionen
auf Basis von road_freight-Observations.

Alle Funktionen sind rein (keine I/O) und PAST-ONLY: es werden nie
unvollständige laufende Perioden mit vollständigen Vorperioden verglichen
(z.B. "diese Woche" gegen "letzte volle Woche"), sondern stets vollständige,
bereits abgeschlossene Fenster relativ zum letzten verfügbaren Datenpunkt.

Hinweis: modules/external/features.py (geteilte rolling/z/acceleration-
Helfer) wird parallel von einem anderen Agenten gebaut. Falls es zum
Zeitpunkt der Nutzung existiert, kann es diese privaten Helfer ersetzen;
bis dahin sind hier minimale lokale Implementierungen enthalten, damit
dieses Modul unabhängig lauffähig und testbar ist.
"""

from __future__ import annotations

import statistics
from datetime import datetime, timedelta, timezone
from typing import Iterable

from modules.external.pit import Observation

# ---------------------------------------------------------------------------
# private Basis-Helfer (bewusst nicht exportiert - siehe Docstring oben)
# ---------------------------------------------------------------------------


def _series(observations: Iterable[Observation], *, source_id: str | None = None,
            metric: str | None = None, entity_id: str | None = None
            ) -> list[tuple[datetime, float]]:
    """Sortierte (observation_time, value)-Liste, None-Werte ausgeschlossen."""
    pts = []
    for o in observations:
        if source_id is not None and o.source_id != source_id:
            continue
        if metric is not None and o.metric != metric:
            continue
        if entity_id is not None and o.entity_id != entity_id:
            continue
        if o.value is None:
            continue
        pts.append((o.observation_time, o.value))
    pts.sort(key=lambda p: p[0])
    # bei Duplikaten (mehrere Vintages je observation_time): letzten Wert behalten
    dedup: dict[datetime, float] = {}
    for t, v in pts:
        dedup[t] = v
    return sorted(dedup.items(), key=lambda p: p[0])


def _last_value(series: list[tuple[datetime, float]]) -> float | None:
    return series[-1][1] if series else None


def _value_n_periods_back(series: list[tuple[datetime, float]], n: int) -> float | None:
    """Wert n vollständige Beobachtungen vor der letzten (Index -1-n)."""
    if len(series) <= n:
        return None
    return series[-1 - n][1]


def _value_at_or_before(series: list[tuple[datetime, float]], cutoff: datetime) -> float | None:
    """Letzter Wert mit observation_time <= cutoff (past-only Fenstersuche)."""
    candidates = [v for t, v in series if t <= cutoff]
    return candidates[-1] if candidates else None


def _mean_last_n(series: list[tuple[datetime, float]], n: int) -> float | None:
    if len(series) < n:
        return None
    vals = [v for _, v in series[-n:]]
    return sum(vals) / len(vals)


def _pct_change(curr: float | None, prev: float | None) -> float | None:
    if curr is None or prev is None or prev == 0:
        return None
    return (curr - prev) / abs(prev)


def _zscore(series: list[tuple[datetime, float]], window_days: int) -> float | None:
    """Past-only Z-Score des letzten Werts gegen die Verteilung der Werte in
    [letzter_zeitpunkt - window_days, letzter_zeitpunkt) (Vorperiode
    ausgeschlossen, damit der aktuelle Wert die eigene Baseline nicht
    verzerrt)."""
    if not series:
        return None
    last_t, last_v = series[-1]
    window_start = last_t - timedelta(days=window_days)
    baseline = [v for t, v in series[:-1] if window_start <= t < last_t]
    if len(baseline) < 5:
        return None
    mean = statistics.fmean(baseline)
    stdev = statistics.pstdev(baseline)
    if stdev == 0:
        return None
    return (last_v - mean) / stdev


def _acceleration(series: list[tuple[datetime, float]], step: int) -> float | None:
    """Änderung der `step`-Perioden-Änderungsrate: (chg[t] - chg[t-step])."""
    if len(series) < 2 * step + 1:
        return None
    curr = _pct_change(series[-1][1], series[-1 - step][1])
    prev = _pct_change(series[-1 - step][1], series[-1 - 2 * step][1])
    if curr is None or prev is None:
        return None
    return curr - prev


# ---------------------------------------------------------------------------
# destatis_truck_toll (Deutschland, täglicher Fahrleistungsindex)
# ---------------------------------------------------------------------------

def de_truck_level(observations: Iterable[Observation], entity_id: str = "",
                    metric: str = "index_sa") -> float | None:
    s = _series(observations, source_id="destatis_truck_toll", metric=metric, entity_id=entity_id)
    return _last_value(s)


def de_truck_7d_mean(observations: Iterable[Observation], entity_id: str = "",
                      metric: str = "index_sa") -> float | None:
    s = _series(observations, source_id="destatis_truck_toll", metric=metric, entity_id=entity_id)
    return _mean_last_n(s, 7)


def de_truck_28d_mean(observations: Iterable[Observation], entity_id: str = "",
                       metric: str = "index_sa") -> float | None:
    s = _series(observations, source_id="destatis_truck_toll", metric=metric, entity_id=entity_id)
    return _mean_last_n(s, 28)


def de_truck_90d_mean(observations: Iterable[Observation], entity_id: str = "",
                       metric: str = "index_sa") -> float | None:
    s = _series(observations, source_id="destatis_truck_toll", metric=metric, entity_id=entity_id)
    return _mean_last_n(s, 90)


def de_truck_wow(observations: Iterable[Observation], entity_id: str = "",
                  metric: str = "index_sa") -> float | None:
    """Week-over-week: letzter Wert vs. Wert 7 Beobachtungstage zuvor."""
    s = _series(observations, source_id="destatis_truck_toll", metric=metric, entity_id=entity_id)
    return _pct_change(_last_value(s), _value_n_periods_back(s, 7))


def de_truck_28d_change(observations: Iterable[Observation], entity_id: str = "",
                         metric: str = "index_sa") -> float | None:
    s = _series(observations, source_id="destatis_truck_toll", metric=metric, entity_id=entity_id)
    return _pct_change(_last_value(s), _value_n_periods_back(s, 28))


def de_truck_yoy(observations: Iterable[Observation], entity_id: str = "",
                  metric: str = "index_sa") -> float | None:
    s = _series(observations, source_id="destatis_truck_toll", metric=metric, entity_id=entity_id)
    if not s:
        return None
    last_t, last_v = s[-1]
    cutoff = last_t.replace(year=last_t.year - 1) if not (last_t.month == 2 and last_t.day == 29) \
        else last_t.replace(year=last_t.year - 1, day=28)
    prev = _value_at_or_before(s, cutoff)
    return _pct_change(last_v, prev)


def de_truck_z_1y(observations: Iterable[Observation], entity_id: str = "",
                   metric: str = "index_sa") -> float | None:
    s = _series(observations, source_id="destatis_truck_toll", metric=metric, entity_id=entity_id)
    return _zscore(s, 365)


def de_truck_z_3y(observations: Iterable[Observation], entity_id: str = "",
                   metric: str = "index_sa") -> float | None:
    s = _series(observations, source_id="destatis_truck_toll", metric=metric, entity_id=entity_id)
    return _zscore(s, 3 * 365)


def de_truck_acceleration(observations: Iterable[Observation], entity_id: str = "",
                           metric: str = "index_sa") -> float | None:
    s = _series(observations, source_id="destatis_truck_toll", metric=metric, entity_id=entity_id)
    return _acceleration(s, 7)


# ---------------------------------------------------------------------------
# bts_freight_tsi (USA, monatlich)
# ---------------------------------------------------------------------------

def us_freight_tsi_level(observations: Iterable[Observation]) -> float | None:
    s = _series(observations, source_id="bts_freight_tsi", metric="us_freight_tsi")
    return _last_value(s)


def us_freight_tsi_mom(observations: Iterable[Observation]) -> float | None:
    s = _series(observations, source_id="bts_freight_tsi", metric="us_freight_tsi")
    return _pct_change(_last_value(s), _value_n_periods_back(s, 1))


def us_freight_tsi_yoy(observations: Iterable[Observation]) -> float | None:
    s = _series(observations, source_id="bts_freight_tsi", metric="us_freight_tsi")
    return _pct_change(_last_value(s), _value_n_periods_back(s, 12))


def us_freight_tsi_3m_momentum(observations: Iterable[Observation]) -> float | None:
    s = _series(observations, source_id="bts_freight_tsi", metric="us_freight_tsi")
    return _pct_change(_last_value(s), _value_n_periods_back(s, 3))


def us_freight_tsi_6m_momentum(observations: Iterable[Observation]) -> float | None:
    s = _series(observations, source_id="bts_freight_tsi", metric="us_freight_tsi")
    return _pct_change(_last_value(s), _value_n_periods_back(s, 6))


def us_freight_tsi_z(observations: Iterable[Observation], window_days: int = 365 * 3) -> float | None:
    s = _series(observations, source_id="bts_freight_tsi", metric="us_freight_tsi")
    return _zscore(s, window_days)


def us_freight_tsi_acceleration(observations: Iterable[Observation]) -> float | None:
    s = _series(observations, source_id="bts_freight_tsi", metric="us_freight_tsi")
    return _acceleration(s, 1)


def us_trucking_level(observations: Iterable[Observation], metric: str = "us_trucking") -> float | None:
    """Nur befüllt, falls ein truckingspezifisches TSI-Komponentenmetric
    tatsächlich vorhanden ist (kein erfundener Ersatz)."""
    s = _series(observations, source_id="bts_freight_tsi", metric=metric)
    return _last_value(s)


def us_trucking_yoy(observations: Iterable[Observation], metric: str = "us_trucking") -> float | None:
    s = _series(observations, source_id="bts_freight_tsi", metric=metric)
    return _pct_change(_last_value(s), _value_n_periods_back(s, 12))


# ---------------------------------------------------------------------------
# eurostat_road_freight (EU, wöchentlicher Check / typ. jährliche-quartalsweise Daten)
#
# Eine EU-Länderserie ist erst über (metric, entity_id) UND die übrigen
# JSON-stat-Dimensionen (tra_type/carriage/nst07/...) eindeutig -- siehe
# EurostatRoadFreightConnector._parse_jsonstat (series_id kodiert das).
# Für die Z-Scores/Features hier wird bewusst GENAU EINE gut-definierte
# Serie je Land verwendet: die kuratierte "total transport"-Serie in
# Tausend Tonnen (unit=THS_T, siehe config/external_sources/road_freight.yaml
# dimension_filters), NIE ein aus dem Unit-Label geratener "tonnes"-Sammel-
# metric mehrerer verschiedener Einheiten/Kategorien.
# ---------------------------------------------------------------------------

def eu_road_freight_level(observations: Iterable[Observation], entity_id: str = "DE",
                           metric: str = "road_freight_ths_t") -> float | None:
    s = _series(observations, source_id="eurostat_road_freight", metric=metric, entity_id=entity_id)
    return _last_value(s)


def eu_road_freight_yoy(observations: Iterable[Observation], entity_id: str = "DE",
                         metric: str = "road_freight_ths_t") -> float | None:
    s = _series(observations, source_id="eurostat_road_freight", metric=metric, entity_id=entity_id)
    return _pct_change(_last_value(s), _value_n_periods_back(s, 1))


def eu_road_freight_z(observations: Iterable[Observation], entity_id: str = "DE",
                       metric: str = "road_freight_ths_t", window_days: int = 365 * 3) -> float | None:
    s = _series(observations, source_id="eurostat_road_freight", metric=metric, entity_id=entity_id)
    return _zscore(s, window_days)


def eu_road_freight_acceleration(observations: Iterable[Observation], entity_id: str = "DE",
                                  metric: str = "road_freight_ths_t") -> float | None:
    s = _series(observations, source_id="eurostat_road_freight", metric=metric, entity_id=entity_id)
    return _acceleration(s, 1)


# ---------------------------------------------------------------------------
# estat_jp_truck (Japan)
# ---------------------------------------------------------------------------

def jp_truck_level(observations: Iterable[Observation], metric: str) -> float | None:
    s = _series(observations, source_id="estat_jp_truck", metric=metric)
    return _last_value(s)


def jp_truck_mom(observations: Iterable[Observation], metric: str) -> float | None:
    s = _series(observations, source_id="estat_jp_truck", metric=metric)
    return _pct_change(_last_value(s), _value_n_periods_back(s, 1))


def jp_truck_yoy(observations: Iterable[Observation], metric: str) -> float | None:
    s = _series(observations, source_id="estat_jp_truck", metric=metric)
    return _pct_change(_last_value(s), _value_n_periods_back(s, 12))


def jp_truck_z(observations: Iterable[Observation], metric: str, window_days: int = 365 * 3) -> float | None:
    s = _series(observations, source_id="estat_jp_truck", metric=metric)
    return _zscore(s, window_days)

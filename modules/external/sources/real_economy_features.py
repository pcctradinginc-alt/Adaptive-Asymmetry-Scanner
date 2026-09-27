"""
modules/external/sources/real_economy_features.py – reine Feature-Funktionen
auf Basis von real_economy-Observations (eurostat_sentiment,
eurostat_industrial_production, fred_us_macro).

Alle Funktionen sind rein (keine I/O) und PAST-ONLY: der Z-Score eines
Monats wird ausschließlich gegen die ihm vorausgehenden ~5 Jahre berechnet
(modules.external.features.rolling_zscore, inklusive des aktuellen Werts
selbst -- wie bei den übrigen Feature-Modulen dieser Codebase, siehe
road_freight_features.py). Methodologische Defaults, nicht gegen Returns
optimiert.
"""

from __future__ import annotations

import statistics
from typing import Iterable

from modules.external import features as feat
from modules.external.pit import Observation

Point = tuple

WINDOW_5Y_DAYS = 365 * 5


def _series(observations: Iterable[Observation], *, source_id: str | None = None,
            metric: str | None = None, entity_id: str | None = None) -> list:
    """Sortierte (observation_time, value)-Liste, None-Werte ausgeschlossen,
    bei Duplikaten (mehrere Vintages je observation_time) wird der letzte
    Wert behalten."""
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
    dedup: dict = {}
    for t, v in pts:
        dedup[t] = v
    return sorted(dedup.items(), key=lambda p: p[0])


def _z_latest(series: list, window_days: int) -> float | None:
    z = feat.rolling_zscore(series, window_days=window_days, min_periods=5)
    return z[-1] if z else None


def _yoy_series(series: list) -> list:
    """Vorjahresänderung (%) je Monat, datumsbasiert (gleicher Monat des
    Vorjahres; fehlt er, entfällt der Punkt). Ein Produktionsindex hat einen
    Trend -- sein Niveau-Z misst vor allem den Trend, nicht die Konjunktur.
    Umfragesalden (ESI, Konsumklima) sind dagegen um ihr Mittel stationär.
    Vergleichbar werden beide erst, wenn Hard Data als Wachstumsrate
    eingeht."""
    by_month = {(t.year, t.month): v for t, v in series}
    out = []
    for t, v in series:
        prev = by_month.get((t.year - 1, t.month))
        if prev in (None, 0):
            continue
        out.append((t, (v - prev) / abs(prev)))
    return out


def _momentum_3m(series: list) -> float | None:
    """3-Monats-Momentum: prozentuale Änderung ggü. dem Wert 3 vollständige
    Beobachtungen (Monate) zuvor (indexbasiert -- monatliche Serien sind
    bereits regelmäßig getaktet)."""
    pc = feat.pct_change(series, periods=3)
    return pc[-1] if pc else None


# ---------------------------------------------------------------------------
# eurostat_sentiment: Economic Sentiment Indicator (ESI) + Industrial
# confidence indicator (ICI)
# ---------------------------------------------------------------------------

def esi_z(observations: Iterable[Observation], entity_id: str = "EU27_2020",
          window_days: int = WINDOW_5Y_DAYS) -> float | None:
    s = _series(observations, source_id="eurostat_sentiment", metric="esi", entity_id=entity_id)
    return _z_latest(s, window_days)


def esi_3m_momentum(observations: Iterable[Observation], entity_id: str = "EU27_2020") -> float | None:
    s = _series(observations, source_id="eurostat_sentiment", metric="esi", entity_id=entity_id)
    return _momentum_3m(s)


def industrial_confidence_z(observations: Iterable[Observation], entity_id: str = "EU27_2020",
                             window_days: int = WINDOW_5Y_DAYS) -> float | None:
    s = _series(observations, source_id="eurostat_sentiment", metric="industrial_confidence",
                entity_id=entity_id)
    return _z_latest(s, window_days)


def industrial_confidence_3m_momentum(observations: Iterable[Observation],
                                       entity_id: str = "EU27_2020") -> float | None:
    s = _series(observations, source_id="eurostat_sentiment", metric="industrial_confidence",
                entity_id=entity_id)
    return _momentum_3m(s)


def eu_survey_z(observations: Iterable[Observation], entity_id: str = "EU27_2020",
                 window_days: int = WINDOW_5Y_DAYS) -> float | None:
    """Kombinierter EU-Umfrage-Z: Mittel aus ESI-Z und Industrial-Confidence-Z
    (nur tatsächlich vorhandene Komponenten; fehlen beide -> None, nie 0)."""
    obs = list(observations)
    vals = [v for v in (esi_z(obs, entity_id, window_days),
                        industrial_confidence_z(obs, entity_id, window_days)) if v is not None]
    return statistics.fmean(vals) if vals else None


# ---------------------------------------------------------------------------
# eurostat_industrial_production: Volume index of production in industry
# (total industry excluding construction, calendar & seasonally adjusted)
# ---------------------------------------------------------------------------

def industrial_production_z(observations: Iterable[Observation], entity_id: str = "EU27_2020",
                             window_days: int = WINDOW_5Y_DAYS) -> float | None:
    """Z der Vorjahresrate (nicht des Indexniveaus, siehe _yoy_series)."""
    s = _series(observations, source_id="eurostat_industrial_production",
                metric="production_volume_index", entity_id=entity_id)
    return _z_latest(_yoy_series(s), window_days)


def industrial_production_3m_momentum(observations: Iterable[Observation],
                                       entity_id: str = "EU27_2020") -> float | None:
    s = _series(observations, source_id="eurostat_industrial_production",
                metric="production_volume_index", entity_id=entity_id)
    return _momentum_3m(s)


# ---------------------------------------------------------------------------
# fred_us_macro: UMCSENT (Umfrage) + INDPRO (Hard Data) -- nur mit
# FRED_API_KEY archiviert (siehe real_economy.FredUsMacroConnector).
# ---------------------------------------------------------------------------

def us_survey_z(observations: Iterable[Observation], window_days: int = WINDOW_5Y_DAYS) -> float | None:
    s = _series(observations, source_id="fred_us_macro", metric="us_umcsent")
    return _z_latest(s, window_days)


def us_survey_3m_momentum(observations: Iterable[Observation]) -> float | None:
    s = _series(observations, source_id="fred_us_macro", metric="us_umcsent")
    return _momentum_3m(s)


def us_hard_z(observations: Iterable[Observation], window_days: int = WINDOW_5Y_DAYS) -> float | None:
    """Z der Vorjahresrate (nicht des Indexniveaus, siehe _yoy_series)."""
    s = _series(observations, source_id="fred_us_macro", metric="us_indpro")
    return _z_latest(_yoy_series(s), window_days)


def us_hard_3m_momentum(observations: Iterable[Observation]) -> float | None:
    s = _series(observations, source_id="fred_us_macro", metric="us_indpro")
    return _momentum_3m(s)

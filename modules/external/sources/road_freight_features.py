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
# destatis_truck_toll (GENESIS-Online, bevorzugt) bzw. destatis_truck_toll_
# download (EXDAT-Direktdownload ohne Login, Fallback) -- Deutschland,
# täglicher Fahrleistungsindex. Beide Konnektoren teilen sich exakt die
# gleichen Metrik-/Entity-Konventionen (index_sa/index_unadjusted,
# entity_id="" = Deutschland gesamt), _de_truck_series() wählt pro Aufruf
# genau EINE Quelle (GENESIS zuerst, nie vermischt).
# ---------------------------------------------------------------------------

def _de_truck_series(observations: Iterable[Observation], entity_id: str, metric: str):
    obs = list(observations)
    primary = _series(obs, source_id="destatis_truck_toll", metric=metric, entity_id=entity_id)
    if primary:
        return primary
    return _series(obs, source_id="destatis_truck_toll_download", metric=metric, entity_id=entity_id)


def de_truck_level(observations: Iterable[Observation], entity_id: str = "",
                    metric: str = "index_sa") -> float | None:
    s = _de_truck_series(observations, entity_id, metric)
    return _last_value(s)


def de_truck_7d_mean(observations: Iterable[Observation], entity_id: str = "",
                      metric: str = "index_sa") -> float | None:
    s = _de_truck_series(observations, entity_id, metric)
    return _mean_last_n(s, 7)


def de_truck_28d_mean(observations: Iterable[Observation], entity_id: str = "",
                       metric: str = "index_sa") -> float | None:
    s = _de_truck_series(observations, entity_id, metric)
    return _mean_last_n(s, 28)


def de_truck_90d_mean(observations: Iterable[Observation], entity_id: str = "",
                       metric: str = "index_sa") -> float | None:
    s = _de_truck_series(observations, entity_id, metric)
    return _mean_last_n(s, 90)


def de_truck_wow(observations: Iterable[Observation], entity_id: str = "",
                  metric: str = "index_sa") -> float | None:
    """Week-over-week: letzter Wert vs. Wert 7 Beobachtungstage zuvor."""
    s = _de_truck_series(observations, entity_id, metric)
    return _pct_change(_last_value(s), _value_n_periods_back(s, 7))


def de_truck_28d_change(observations: Iterable[Observation], entity_id: str = "",
                         metric: str = "index_sa") -> float | None:
    s = _de_truck_series(observations, entity_id, metric)
    return _pct_change(_last_value(s), _value_n_periods_back(s, 28))


def de_truck_yoy(observations: Iterable[Observation], entity_id: str = "",
                  metric: str = "index_sa") -> float | None:
    s = _de_truck_series(observations, entity_id, metric)
    if not s:
        return None
    last_t, last_v = s[-1]
    cutoff = last_t.replace(year=last_t.year - 1) if not (last_t.month == 2 and last_t.day == 29) \
        else last_t.replace(year=last_t.year - 1, day=28)
    prev = _value_at_or_before(s, cutoff)
    return _pct_change(last_v, prev)


def de_truck_z_1y(observations: Iterable[Observation], entity_id: str = "",
                   metric: str = "index_sa") -> float | None:
    s = _de_truck_series(observations, entity_id, metric)
    return _zscore(s, 365)


def de_truck_z_3y(observations: Iterable[Observation], entity_id: str = "",
                   metric: str = "index_sa") -> float | None:
    s = _de_truck_series(observations, entity_id, metric)
    return _zscore(s, 3 * 365)


def de_truck_acceleration(observations: Iterable[Observation], entity_id: str = "",
                           metric: str = "index_sa") -> float | None:
    s = _de_truck_series(observations, entity_id, metric)
    return _acceleration(s, 7)


# ---------------------------------------------------------------------------
# bts_freight_tsi (USA, monatlich, FRED/ALFRED bei FRED_API_KEY) bzw.
# bts_open_data_tsi (USA, monatlich, Socrata Open Data ohne API-Key) --
# bts_freight_tsi wird IMMER bevorzugt (ALFRED-Vintages, siehe
# BtsFreightTsiConnector), bts_open_data_tsi ist der Fallback ohne Key.
# _us_tsi_source() wählt pro Aufruf genau EINE Quelle (nie vermischt).
# ---------------------------------------------------------------------------

_US_TSI_PREFERRED_SOURCE = "bts_freight_tsi"
_US_TSI_FALLBACK_SOURCE = "bts_open_data_tsi"


def _us_tsi_source(observations: Iterable[Observation]) -> str:
    """Welche Quelle für us_freight_tsi_*-Features verwendet wird: bevorzugt
    bts_freight_tsi (ALFRED-Vintages bei FRED_API_KEY), sonst
    bts_open_data_tsi, falls dafür Observations vorliegen."""
    has_preferred = any(o.source_id == _US_TSI_PREFERRED_SOURCE and o.metric == "us_freight_tsi"
                         for o in observations)
    return _US_TSI_PREFERRED_SOURCE if has_preferred else _US_TSI_FALLBACK_SOURCE


def us_freight_tsi_level(observations: Iterable[Observation]) -> float | None:
    obs = list(observations)
    s = _series(obs, source_id=_us_tsi_source(obs), metric="us_freight_tsi")
    return _last_value(s)


def us_freight_tsi_mom(observations: Iterable[Observation]) -> float | None:
    obs = list(observations)
    s = _series(obs, source_id=_us_tsi_source(obs), metric="us_freight_tsi")
    return _pct_change(_last_value(s), _value_n_periods_back(s, 1))


def us_freight_tsi_yoy(observations: Iterable[Observation]) -> float | None:
    obs = list(observations)
    s = _series(obs, source_id=_us_tsi_source(obs), metric="us_freight_tsi")
    return _pct_change(_last_value(s), _value_n_periods_back(s, 12))


def us_freight_tsi_3m_momentum(observations: Iterable[Observation]) -> float | None:
    obs = list(observations)
    s = _series(obs, source_id=_us_tsi_source(obs), metric="us_freight_tsi")
    return _pct_change(_last_value(s), _value_n_periods_back(s, 3))


def us_freight_tsi_6m_momentum(observations: Iterable[Observation]) -> float | None:
    obs = list(observations)
    s = _series(obs, source_id=_us_tsi_source(obs), metric="us_freight_tsi")
    return _pct_change(_last_value(s), _value_n_periods_back(s, 6))


def us_freight_tsi_z(observations: Iterable[Observation], window_days: int = 365 * 3) -> float | None:
    obs = list(observations)
    s = _series(obs, source_id=_us_tsi_source(obs), metric="us_freight_tsi")
    return _zscore(s, window_days)


def us_freight_tsi_acceleration(observations: Iterable[Observation]) -> float | None:
    obs = list(observations)
    s = _series(obs, source_id=_us_tsi_source(obs), metric="us_freight_tsi")
    return _acceleration(s, 1)


def us_trucking_level(observations: Iterable[Observation], metric: str = "us_trucking") -> float | None:
    """Nur befüllt, falls ein truckingspezifisches TSI-Komponentenmetric
    tatsächlich vorhanden ist (kein erfundener Ersatz). Sucht in der
    bevorzugten Quelle (bts_freight_tsi) zuerst, sonst im Socrata-Fallback
    bts_open_data_tsi (dort heißt die Metrik "us_trucking_index")."""
    obs = list(observations)
    s = _series(obs, source_id="bts_freight_tsi", metric=metric)
    if s:
        return _last_value(s)
    s = _series(obs, source_id="bts_open_data_tsi", metric="us_trucking_index")
    return _last_value(s)


def us_trucking_yoy(observations: Iterable[Observation], metric: str = "us_trucking") -> float | None:
    obs = list(observations)
    s = _series(obs, source_id="bts_freight_tsi", metric=metric)
    if not s:
        s = _series(obs, source_id="bts_open_data_tsi", metric="us_trucking_index")
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


_TOTAL_CODES = {"TOTAL", "TOT", "T"}


def select_eurostat_total_series(observations: Iterable[Observation], entity_id: str,
                                 metric: str, source_id: str = "eurostat_road_freight") -> str | None:
    """Wählt deterministisch GENAU EINE Eurostat-Reihe (series_id) je Land und
    Einheit, damit Varianten (z.B. gewerblich/Werkverkehr) nie vermischt
    werden: bevorzugt die Reihe, deren Nicht-geo/time/unit-Dimensionen alle
    Gesamt-Codes (TOTAL/TOT/T) tragen; sonst die mit dem größten mittleren
    Niveau (Gesamt >= Komponenten); Gleichstand -> lexikografisch kleinste ID.

    `source_id` wählt die Quelle (Default: die jährliche/gemischte
    eurostat_road_freight-Reihe); eurostat_road_freight_quarterly übergibt
    hier explizit ihre eigene source_id, siehe _eu_quarterly_series()."""
    by_series: dict[str, list] = {}
    attrs_of: dict[str, dict] = {}
    for o in observations:
        if (o.source_id != source_id or o.metric != metric
                or o.entity_id != entity_id or o.value is None):
            continue
        by_series.setdefault(o.series_id, []).append(o.value)
        attrs_of.setdefault(o.series_id, o.attrs or {})
    if not by_series:
        return None

    def is_total(sid: str) -> bool:
        dims = {k: v for k, v in attrs_of[sid].items() if k not in ("geo", "time", "unit", "freq")}
        return bool(dims) and all(str(v).upper() in _TOTAL_CODES for v in dims.values())

    totals = sorted(sid for sid in by_series if is_total(sid))
    if totals:
        return totals[0]
    return sorted(by_series, key=lambda sid: (-(sum(by_series[sid]) / len(by_series[sid])), sid))[0]


def _eu_series(observations, entity_id: str, metric: str):
    obs = list(observations)
    sid = select_eurostat_total_series(obs, entity_id, metric)
    if sid is None:
        return []
    return _series([o for o in obs if o.series_id == sid], source_id="eurostat_road_freight",
                   metric=metric, entity_id=entity_id)


def eu_road_freight_z(observations: Iterable[Observation], entity_id: str = "DE",
                       metric: str = "road_freight_ths_t", window_days: int = 365 * 3) -> float | None:
    return _zscore(_eu_series(observations, entity_id, metric), window_days)


def eu_road_freight_acceleration(observations: Iterable[Observation], entity_id: str = "DE",
                                  metric: str = "road_freight_ths_t") -> float | None:
    return _acceleration(_eu_series(observations, entity_id, metric), 1)


# ---------------------------------------------------------------------------
# eurostat_road_freight_quarterly – bevorzugte, feinere Zeitauflösung für
# z-Score/Beschleunigung; Fallback auf die jährliche eurostat_road_freight-
# Reihe, wenn (noch) zu wenige Quartalspunkte vorliegen (siehe
# eu_road_freight_z_preferred/_acceleration_preferred unten).
# ---------------------------------------------------------------------------

def _eu_quarterly_series(observations, entity_id: str, metric: str):
    obs = list(observations)
    sid = select_eurostat_total_series(
        [o for o in obs if o.source_id == "eurostat_road_freight_quarterly"], entity_id, metric,
        source_id="eurostat_road_freight_quarterly")
    if sid is None:
        return []
    return _series([o for o in obs if o.series_id == sid], source_id="eurostat_road_freight_quarterly",
                   metric=metric, entity_id=entity_id)


_QUARTERLY_MIN_POINTS = 5
_QUARTERLY_WINDOW_QUARTERS = 12   # ~3 Jahre


def eu_road_freight_z_preferred(observations: Iterable[Observation], entity_id: str = "DE",
                                 metric: str = "road_freight_ths_t") -> tuple[float | None, str]:
    """Bevorzugt die Eurostat-QUARTERLY-Reihe (Fenster ~12 Quartale = 3
    Jahre), sofern mindestens _QUARTERLY_MIN_POINTS Punkte vorliegen;
    andernfalls Fallback auf die jährliche eurostat_road_freight-Reihe
    (3-Jahres-Fenster in Tagen, wie eu_road_freight_z). Rückgabe:
    (z_score, frequency_used) mit frequency_used in {"quarterly", "annual",
    "none"} -- das Aufrufer-/Reporting-Layer kann so dokumentieren, welche
    Auflösung tatsächlich verwendet wurde."""
    obs = list(observations)
    q_series = _eu_quarterly_series(obs, entity_id, metric)
    if len(q_series) >= _QUARTERLY_MIN_POINTS:
        window_days = int(_QUARTERLY_WINDOW_QUARTERS * 91.3)
        z = _zscore(q_series, window_days)
        if z is not None:
            return z, "quarterly"
    # Jahreswerte: 12-Jahres-Fenster (>= 5 Basispunkte), wie im Kontext.
    # Das 3-Jahres-Default-Fenster hätte nur 3 Jahrespunkte -> immer None.
    z = eu_road_freight_z(obs, entity_id, metric, window_days=365 * 12)
    return z, ("annual" if z is not None else "none")


def eu_road_freight_acceleration_preferred(observations: Iterable[Observation], entity_id: str = "DE",
                                            metric: str = "road_freight_ths_t") -> tuple[float | None, str]:
    """Wie eu_road_freight_z_preferred(), aber für die Beschleunigungsrate
    (1-Perioden-Schritt in der jeweils gewählten Frequenz)."""
    obs = list(observations)
    q_series = _eu_quarterly_series(obs, entity_id, metric)
    if len(q_series) >= _QUARTERLY_MIN_POINTS:
        acc = _acceleration(q_series, 1)
        if acc is not None:
            return acc, "quarterly"
    acc = eu_road_freight_acceleration(obs, entity_id, metric)
    return acc, ("annual" if acc is not None else "none")


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
    """z der Vorjahresrate: e-Stat-Tonnage ist NICHT saisonbereinigt, ein
    Niveau-z würde vor allem das Saisonmuster messen (Audit 2026-09-27)."""
    obs = [o for o in observations if o.source_id == "estat_jp_truck" and o.metric == metric
           and o.value is not None]
    if not obs:
        return None
    # nur die aktuelle Tabelle (series_id der jüngsten Beobachtung) -- nie eine
    # abgeschlossene Altserie mit gleichem Metriknamen untermischen
    current_sid = max(obs, key=lambda o: o.observation_time).series_id
    s = _series([o for o in obs if o.series_id == current_sid], source_id="estat_jp_truck", metric=metric)
    by_month = {(t.year, t.month): v for t, v in s}
    yoy = [(t, (v - by_month[(t.year - 1, t.month)]) / abs(by_month[(t.year - 1, t.month)]))
           for t, v in s if by_month.get((t.year - 1, t.month)) not in (None, 0)]
    return _zscore(yoy, window_days)

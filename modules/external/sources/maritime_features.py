"""
modules/external/sources/maritime_features.py – reine Feature-Berechnung auf
IMF-PortWatch-Observations (Ports + Chokepoints).

Alle Funktionen sind rein (keine I/O, kein Zufall) und PAST-ONLY: jede
Berechnung für einen Stichtag `asof_date` filtert zuerst auf Zeilen mit
`activity_date <= asof_date` -- Daten NACH `asof_date` dürfen das Ergebnis
niemals verändern (Test: past-only invariance).

`modules/external/features.py` existiert (noch) nicht in dieser Codebase,
daher lebt die Logik hier vollständig als private Hilfsfunktionen + eine
öffentliche Fassade `compute_all_features`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable

import numpy as np
import pandas as pd

from modules.external.pit import Observation

DEFAULT_MIN_VALID_FOR_BREADTH = 8
DEFAULT_INCOMPLETE_DAY_COVERAGE = 0.90
DEFAULT_BREADTH_Z_THRESHOLD = 1.0
SEASONAL_WINDOW_DAYS = 10  # +/- Fenster um "1 Jahr zuvor" für YoY/Saison-Z


# --------------------------------------------------------------------------
# Observation -> DataFrame
# --------------------------------------------------------------------------

def observations_to_frame(observations: Iterable[Observation]) -> pd.DataFrame:
    rows = []
    for o in observations:
        rows.append({
            "activity_date": o.observation_time.date(),
            "entity_id": o.entity_id,
            "metric": o.metric,
            "value": o.value,
            "is_backfill": bool(o.attrs.get("historical_backfill")),
            "port_name": o.attrs.get("port_name"),
            "country": o.attrs.get("country"),
            "vessel_type": o.attrs.get("vessel_type"),
        })
    if not rows:
        return pd.DataFrame(columns=[
            "activity_date", "entity_id", "metric", "value", "is_backfill",
            "port_name", "country", "vessel_type",
        ])
    return pd.DataFrame(rows)


def _past_only(df: pd.DataFrame, asof_date: date) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df["activity_date"] <= asof_date]


# --------------------------------------------------------------------------
# Incomplete-latest-day drop
# --------------------------------------------------------------------------

@dataclass
class CoverageDecision:
    kept_through: date | None
    dropped_dates: list[date]
    coverage_by_date: dict


def drop_incomplete_latest_day(
    df: pd.DataFrame,
    monitored_entity_ids: list[str],
    min_coverage: float = DEFAULT_INCOMPLETE_DAY_COVERAGE,
) -> tuple[pd.DataFrame, CoverageDecision]:
    """Entfernt die jüngsten Aktivitätstage, solange weniger als
    `min_coverage` (Default 90%) der überwachten Ports/Chokepoints an diesem
    Tag mindestens einen Datenpunkt haben. Grund: PortWatch füllt Tage
    verzögert nach; ein "letzter Tag" mit z.B. 20% Coverage würde
    Global-/Breadth-Features künstlich verzerren (sieht aus wie ein
    Aktivitätseinbruch, ist aber nur unvollständige Datenlieferung)."""
    if df.empty or not monitored_entity_ids:
        return df, CoverageDecision(None, [], {})
    n_monitored = len(set(monitored_entity_ids))
    sub = df[df["entity_id"].isin(monitored_entity_ids)]
    coverage = (
        sub.groupby("activity_date")["entity_id"].nunique() / n_monitored
    ).to_dict()
    dropped = []
    dates_sorted = sorted(coverage.keys())
    kept_through = None
    for d in dates_sorted:
        if coverage[d] >= min_coverage:
            kept_through = d
        else:
            # nur droppen, falls es sich um die/​die letzten Tage handelt
            dropped.append(d)
    # nur Tage NACH dem letzten ausreichend abgedeckten Tag droppen (nicht
    # zufällige Lücken mitten in der Historie)
    if kept_through is not None:
        dropped = [d for d in dropped if d > kept_through]
    else:
        dropped = dates_sorted  # nichts erreicht die Schwelle
    clean = df[~df["activity_date"].isin(dropped)]
    return clean, CoverageDecision(kept_through, dropped, coverage)


# --------------------------------------------------------------------------
# Rolling stats je Entity/Metrik
# --------------------------------------------------------------------------

def _window_mean(series: pd.Series, dates: pd.Series, asof: date, days: int) -> float | None:
    start = asof - timedelta(days=days - 1)
    mask = (dates >= start) & (dates <= asof)
    vals = series[mask]
    return float(vals.mean()) if len(vals) else None


def _seasonal_window_values(series: pd.Series, dates: pd.Series, center: date, half_window: int) -> np.ndarray:
    start = center - timedelta(days=half_window)
    end = center + timedelta(days=half_window)
    mask = (dates >= start) & (dates <= end)
    return series[mask].to_numpy(dtype=float)


def entity_rolling_stats(df: pd.DataFrame, entity_id: str, metric: str, asof_date: date) -> dict:
    """7d/28d/90d Mittelwerte, 7d_vs_28d, 28d_vs_90d, YoY (nur falls ein
    Vorjahres-Fenster existiert) und ein saisonaler Z-Score (aktueller 28d-
    Mittelwert vs. Verteilung des gleichen Kalenderfensters vor 1 Jahr)."""
    past = _past_only(df, asof_date)
    sub = past[(past["entity_id"] == entity_id) & (past["metric"] == metric)].sort_values("activity_date")
    if sub.empty:
        return {"mean_7d": None, "mean_28d": None, "mean_90d": None, "ratio_7d_vs_28d": None,
                "ratio_28d_vs_90d": None, "yoy": None, "seasonal_z": None}

    dates = sub["activity_date"]
    values = sub["value"]

    m7 = _window_mean(values, dates, asof_date, 7)
    m28 = _window_mean(values, dates, asof_date, 28)
    m90 = _window_mean(values, dates, asof_date, 90)

    ratio_7_28 = (m7 / m28) if (m7 is not None and m28 not in (None, 0)) else None
    ratio_28_90 = (m28 / m90) if (m28 is not None and m90 not in (None, 0)) else None

    one_year_ago = asof_date - timedelta(days=365)
    prior_year_vals = _seasonal_window_values(values, dates, one_year_ago, SEASONAL_WINDOW_DAYS)
    yoy = None
    seasonal_z = None
    if m28 is not None and len(prior_year_vals) > 0:
        baseline_mean = float(np.mean(prior_year_vals))
        yoy = (m28 / baseline_mean - 1.0) if baseline_mean else None
        if len(prior_year_vals) >= 2:
            baseline_std = float(np.std(prior_year_vals, ddof=1))
            if baseline_std > 0:
                seasonal_z = (m28 - baseline_mean) / baseline_std

    return {
        "mean_7d": m7, "mean_28d": m28, "mean_90d": m90,
        "ratio_7d_vs_28d": ratio_7_28, "ratio_28d_vs_90d": ratio_28_90,
        "yoy": yoy, "seasonal_z": seasonal_z,
    }


# --------------------------------------------------------------------------
# Globale / regionale Aggregate
# --------------------------------------------------------------------------

GLOBAL_METRIC_GROUPS = {
    "global_port_calls": ("portcalls_total",),
    "global_import_activity": ("import_total",),
    "global_export_activity": ("export_total",),
    "global_container_activity": ("portcalls_container",),
    "global_drybulk_activity": ("portcalls_dry_bulk",),
    "global_tanker_activity": ("portcalls_tanker",),
    "global_general_cargo_activity": ("portcalls_general_cargo",),
    "global_roro_activity": ("portcalls_roro",),
}


def _aggregate_across_entities(df: pd.DataFrame, entity_ids: list[str], metric: str,
                                asof_date: date, agg: str = "sum") -> pd.DataFrame:
    past = _past_only(df, asof_date)
    sub = past[(past["entity_id"].isin(entity_ids)) & (past["metric"] == metric)]
    if sub.empty:
        return pd.DataFrame(columns=["activity_date", "value"])
    grouped = sub.groupby("activity_date")["value"].agg(agg).reset_index()
    return grouped.rename(columns={grouped.columns[1]: "value"})


def global_feature(df: pd.DataFrame, entity_ids: list[str], metric: str, asof_date: date) -> dict:
    agg = _aggregate_across_entities(df, entity_ids, metric, asof_date)
    if agg.empty:
        return {"level": None, "mean_28d": None, "yoy": None, "z": None}
    agg = agg.rename(columns={"value": "metric_value"})
    fake_entity = "__global__"
    tmp = agg.rename(columns={"metric_value": "value"}).assign(entity_id=fake_entity, metric=metric)
    stats = entity_rolling_stats(tmp, fake_entity, metric, asof_date)
    latest_row = agg[agg["activity_date"] <= asof_date].sort_values("activity_date").tail(1)
    level = float(latest_row["metric_value"].iloc[0]) if len(latest_row) else None
    return {"level": level, "mean_28d": stats["mean_28d"], "yoy": stats["yoy"], "z": stats["seasonal_z"]}


def regional_aggregate(df: pd.DataFrame, group_entity_ids: list[str], metric: str, asof_date: date) -> dict:
    return global_feature(df, group_entity_ids, metric, asof_date)


# --------------------------------------------------------------------------
# Breadth
# --------------------------------------------------------------------------

def breadth_features(
    df: pd.DataFrame,
    monitored_entity_ids: list[str],
    metric: str,
    asof_date: date,
    z_threshold: float = DEFAULT_BREADTH_Z_THRESHOLD,
    min_valid: int = DEFAULT_MIN_VALID_FOR_BREADTH,
) -> dict:
    """Anteil überwachter Ports mit z < -threshold ("negative breadth") bzw.
    z > +threshold ("positive breadth"). `breadth_valid` ist nur True, wenn
    mindestens `min_valid` Ports einen berechenbaren Z-Score haben --
    verhindert eine falsch präzise Breadth-Zahl aus z.B. 2 von 30 Ports."""
    zs = []
    for entity_id in monitored_entity_ids:
        stats = entity_rolling_stats(df, entity_id, metric, asof_date)
        if stats["seasonal_z"] is not None:
            zs.append(stats["seasonal_z"])
    valid_port_count = len(zs)
    breadth_valid = valid_port_count >= min_valid
    if valid_port_count == 0:
        return {"negative_breadth": None, "positive_breadth": None,
                "valid_port_count": 0, "breadth_valid": False}
    neg = sum(1 for z in zs if z < -z_threshold) / valid_port_count
    pos = sum(1 for z in zs if z > z_threshold) / valid_port_count
    return {
        "negative_breadth": neg if breadth_valid else None,
        "positive_breadth": pos if breadth_valid else None,
        "valid_port_count": valid_port_count,
        "breadth_valid": breadth_valid,
    }


# --------------------------------------------------------------------------
# Chokepoints
# --------------------------------------------------------------------------

def chokepoint_features(df: pd.DataFrame, chokepoint_ids: dict[str, str], metric: str, asof_date: date) -> dict:
    """chokepoint_ids: {"suez": entity_id, "panama": entity_id, ...} ->
    {"suez_z": ..., "panama_z": ..., ...} (nur Slugs, für die eine Entity-ID
    aufgelöst werden konnte)."""
    out = {}
    for slug, entity_id in chokepoint_ids.items():
        if entity_id is None:
            out[f"{slug}_z"] = None
            continue
        stats = entity_rolling_stats(df, entity_id, metric, asof_date)
        out[f"{slug}_z"] = stats["seasonal_z"]
    return out


# --------------------------------------------------------------------------
# Fassade
# --------------------------------------------------------------------------

def compute_all_features(
    df: pd.DataFrame,
    port_universe_ids: dict,
    chokepoint_ids: dict,
    asof_date: date,
    min_coverage: float = DEFAULT_INCOMPLETE_DAY_COVERAGE,
    min_valid_breadth: int = DEFAULT_MIN_VALID_FOR_BREADTH,
) -> dict:
    monitored_ids = [v for v in port_universe_ids.get("flat", {}).values() if v]
    clean_df, coverage = drop_incomplete_latest_day(df, monitored_ids, min_coverage)

    features: dict = {"asof_date": asof_date.isoformat(),
                       "incomplete_day_drop": {
                           "kept_through": coverage.kept_through.isoformat() if coverage.kept_through else None,
                           "dropped_dates": [d.isoformat() for d in coverage.dropped_dates],
                       }}

    for feat_name, (metric,) in GLOBAL_METRIC_GROUPS.items():
        g = global_feature(clean_df, monitored_ids, metric, asof_date)
        features[feat_name] = g["level"]
        features[f"{feat_name}_yoy"] = g["yoy"]
        features[f"{feat_name}_z"] = g["z"]

    for group_name, ids in (port_universe_ids.get("groups") or {}).items():
        g = regional_aggregate(clean_df, [i for i in ids if i], "portcalls_total", asof_date)
        features[f"regional_{group_name}_port_calls"] = g["level"]
        features[f"regional_{group_name}_port_calls_z"] = g["z"]

    breadth = breadth_features(clean_df, monitored_ids, "portcalls_total", asof_date,
                                min_valid=min_valid_breadth)
    features.update({f"breadth_{k}": v for k, v in breadth.items()})

    features.update(chokepoint_features(clean_df, chokepoint_ids, "n_total", asof_date))

    return features

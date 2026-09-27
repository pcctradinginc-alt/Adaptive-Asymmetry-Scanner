"""
Tests für modules/external/sources/maritime_features.py.

Deckt ab: incomplete-latest-day-drop, breadth mit min_valid-Schwelle,
past-only-Invarianz (keine Lookahead-Nutzung zukünftiger Zeilen), separate
Cargo-Typ-Metriken.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from modules.external.sources import maritime_features as mf


def _rows(entity_id, start, end, value, metric="portcalls_total", wiggle=0.0):
    rows = []
    d = start
    i = 0
    while d <= end:
        v = value + (wiggle if i % 2 == 0 else -wiggle)
        rows.append({"activity_date": d, "entity_id": entity_id, "metric": metric,
                     "value": v, "is_backfill": False, "port_name": entity_id,
                     "country": "X", "vessel_type": "all"})
        d += timedelta(days=1)
        i += 1
    return rows


def _build_baseline_df(entities: list[str], asof: date, shock: dict[str, float] | None = None) -> pd.DataFrame:
    """Baut für jede Entity durchgängige Historie von 400 Tagen vor `asof`
    bis `asof`, konstanter Wert 100, außer für die letzten 28 Tage von
    Entities in `shock` (dort verschobener Wert)."""
    shock = shock or {}
    rows = []
    history_start = asof - timedelta(days=400)
    shock_start = asof - timedelta(days=27)
    for e in entities:
        rows.extend(_rows(e, history_start, shock_start - timedelta(days=1), 100.0, wiggle=3.0))
        shocked_value = shock.get(e, 100.0)
        rows.extend(_rows(e, shock_start, asof, shocked_value, wiggle=3.0))
    return pd.DataFrame(rows)


ASOF = date(2024, 1, 31)


# --------------------------------------------------------------------------
# Incomplete latest day
# --------------------------------------------------------------------------

def test_drop_incomplete_latest_day_removes_undercovered_trailing_dates():
    monitored = [f"p{i}" for i in range(10)]
    df = _build_baseline_df(monitored, ASOF)
    # letzter Tag: nur 2/10 Ports haben Daten -> unter 90% -> muss gedroppt werden
    last_day = ASOF
    df = df[~((df["activity_date"] == last_day) & (~df["entity_id"].isin(monitored[:2])))]

    clean, decision = mf.drop_incomplete_latest_day(df, monitored, min_coverage=0.9)
    assert last_day in decision.dropped_dates
    assert decision.kept_through == last_day - timedelta(days=1)
    assert clean[clean["activity_date"] == last_day].empty


def test_drop_incomplete_latest_day_keeps_well_covered_days():
    monitored = [f"p{i}" for i in range(10)]
    df = _build_baseline_df(monitored, ASOF)
    clean, decision = mf.drop_incomplete_latest_day(df, monitored, min_coverage=0.9)
    assert decision.dropped_dates == []
    assert decision.kept_through == ASOF
    assert not clean[clean["activity_date"] == ASOF].empty


# --------------------------------------------------------------------------
# Breadth min-valid
# --------------------------------------------------------------------------

def test_breadth_below_min_valid_returns_none_but_reports_count():
    # nur 5 Ports mit ausreichender Historie fürs Vorjahresfenster (< min_valid=8)
    monitored = [f"p{i}" for i in range(5)]
    df = _build_baseline_df(monitored, ASOF, shock={"p0": 200.0, "p1": 200.0})
    result = mf.breadth_features(df, monitored, "portcalls_total", ASOF, min_valid=8)
    assert result["valid_port_count"] == 5
    assert result["breadth_valid"] is False
    assert result["negative_breadth"] is None
    assert result["positive_breadth"] is None


def test_breadth_at_or_above_min_valid_reports_ratios():
    monitored = [f"p{i}" for i in range(10)]
    # 3 Ports mit klarem positiven Schock (z hoch), Rest neutral
    df = _build_baseline_df(monitored, ASOF, shock={"p0": 300.0, "p1": 300.0, "p2": 300.0})
    result = mf.breadth_features(df, monitored, "portcalls_total", ASOF, z_threshold=1.0, min_valid=8)
    assert result["valid_port_count"] == 10
    assert result["breadth_valid"] is True
    assert result["positive_breadth"] == pytest.approx(0.3)
    assert result["negative_breadth"] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Past-only Invarianz
# --------------------------------------------------------------------------

def test_entity_rolling_stats_never_uses_future_rows():
    df = _build_baseline_df(["p0"], ASOF, shock={"p0": 150.0})
    stats_now = mf.entity_rolling_stats(df, "p0", "portcalls_total", ASOF)

    future_rows = _rows("p0", ASOF + timedelta(days=1), ASOF + timedelta(days=60), 999999.0)
    df_with_future = pd.concat([df, pd.DataFrame(future_rows)], ignore_index=True)
    stats_with_future = mf.entity_rolling_stats(df_with_future, "p0", "portcalls_total", ASOF)

    assert stats_now == stats_with_future


def test_drop_incomplete_latest_day_never_uses_future_rows_either():
    monitored = [f"p{i}" for i in range(6)]
    df = _build_baseline_df(monitored, ASOF)
    clean_a, decision_a = mf.drop_incomplete_latest_day(df, monitored, min_coverage=0.9)

    future_rows = _rows("p0", ASOF + timedelta(days=1), ASOF + timedelta(days=5), 1.0)
    df_with_future = pd.concat([df, pd.DataFrame(future_rows)], ignore_index=True)
    clean_b, decision_b = mf.drop_incomplete_latest_day(
        mf._past_only(df_with_future, ASOF), monitored, min_coverage=0.9,
    )
    assert decision_a.kept_through == decision_b.kept_through
    assert decision_a.dropped_dates == decision_b.dropped_dates


# --------------------------------------------------------------------------
# Cargo-Typen bleiben getrennt
# --------------------------------------------------------------------------

def test_metrics_stay_separate_per_cargo_type():
    rows = []
    rows.extend(_rows("p0", ASOF - timedelta(days=5), ASOF, 10.0, metric="portcalls_container"))
    rows.extend(_rows("p0", ASOF - timedelta(days=5), ASOF, 40.0, metric="portcalls_total"))
    df = pd.DataFrame(rows)
    container_stats = mf.entity_rolling_stats(df, "p0", "portcalls_container", ASOF)
    total_stats = mf.entity_rolling_stats(df, "p0", "portcalls_total", ASOF)
    assert container_stats["mean_7d"] == pytest.approx(10.0)
    assert total_stats["mean_7d"] == pytest.approx(40.0)


# --------------------------------------------------------------------------
# YoY / seasonal z Grundverhalten
# --------------------------------------------------------------------------

def test_yoy_none_when_no_prior_year_window_exists():
    rows = _rows("p0", ASOF - timedelta(days=30), ASOF, 100.0)
    df = pd.DataFrame(rows)
    stats = mf.entity_rolling_stats(df, "p0", "portcalls_total", ASOF)
    assert stats["yoy"] is None
    assert stats["seasonal_z"] is None
    assert stats["mean_28d"] == pytest.approx(100.0)


def test_seasonal_z_reflects_positive_shock():
    df = _build_baseline_df(["p0"], ASOF, shock={"p0": 500.0})
    stats = mf.entity_rolling_stats(df, "p0", "portcalls_total", ASOF)
    assert stats["seasonal_z"] is not None
    assert stats["seasonal_z"] > 0
    assert stats["yoy"] > 0

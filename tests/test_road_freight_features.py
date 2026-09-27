"""
tests/test_road_freight_features.py – Feature-Mathematik auf synthetischen
Observation-Reihen (keine Fixtures/Netzwerk nötig, reine Funktionen).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from modules.external.pit import AvailabilityPrecision, Observation
from modules.external.sources import road_freight_features as feat

BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _obs(source_id, metric, day_offset, value, entity_id="") -> Observation:
    t = BASE + timedelta(days=day_offset)
    return Observation(
        source_id=source_id, dataset="test", series_id="s1", entity_id=entity_id,
        metric=metric, value=value, unit="index_points",
        observation_time=t, available_at=t, retrieved_at=t,
        availability_precision=AvailabilityPrecision.CONSERVATIVE_DATE,
        parser_version="1",
    )


def test_de_truck_level_and_means():
    obs = [_obs("destatis_truck_toll", "index_sa", i, 100 + i) for i in range(30)]
    assert feat.de_truck_level(obs) == pytest.approx(129)
    assert feat.de_truck_7d_mean(obs) == pytest.approx(sum(range(23, 30)) / 7 + 100)
    assert feat.de_truck_28d_mean(obs) == pytest.approx(sum(100 + i for i in range(2, 30)) / 28)


def test_de_truck_wow_and_28d_change():
    obs = [_obs("destatis_truck_toll", "index_sa", i, 100 + i) for i in range(30)]
    wow = feat.de_truck_wow(obs)
    # letzter Wert 129 (Tag29) vs Wert 7 Beobachtungen zuvor (Tag22=122)
    assert wow == pytest.approx((129 - 122) / 122)
    chg28 = feat.de_truck_28d_change(obs)
    assert chg28 == pytest.approx((129 - 101) / 101)


def test_de_truck_yoy_uses_past_only_cutoff():
    obs = [_obs("destatis_truck_toll", "index_sa", i, 100 + i) for i in range(0, 400, 1)]
    yoy = feat.de_truck_yoy(obs)
    assert yoy is not None
    assert yoy > 0  # steigende Reihe -> positives YoY


def test_de_truck_acceleration_positive_when_growth_speeds_up():
    # 7-Tage-Wachstumsrate beschleunigt sich zwischen Tag15-22 (+10%) und
    # Tag22-29 (+20%). Frühere Werte (Tag0-14) sind reine Füllwerte.
    values = [100.0] * 15
    for i in range(15, 23):
        values.append(100 + (10 / 7) * (i - 15))   # Tag15..22: 100 -> 110
    for i in range(23, 30):
        values.append(110 + (22 / 7) * (i - 22))   # Tag22..29: 110 -> 132
    obs = [_obs("destatis_truck_toll", "index_sa", i, v) for i, v in enumerate(values)]
    acc = feat.de_truck_acceleration(obs)
    assert acc is not None
    assert acc > 0


def test_metric_isolation_index_sa_vs_unadjusted():
    obs = [
        _obs("destatis_truck_toll", "index_sa", 0, 100),
        _obs("destatis_truck_toll", "index_unadjusted", 0, 95),
        _obs("destatis_truck_toll", "index_sa", 1, 101),
        _obs("destatis_truck_toll", "index_unadjusted", 1, 96),
    ]
    assert feat.de_truck_level(obs, metric="index_sa") == 101
    assert feat.de_truck_level(obs, metric="index_unadjusted") == 96


def test_us_freight_tsi_mom_yoy_and_momentum():
    obs = [_obs("bts_freight_tsi", "us_freight_tsi", i * 30, 100 + i) for i in range(15)]
    assert feat.us_freight_tsi_level(obs) == 114
    assert feat.us_freight_tsi_mom(obs) == pytest.approx((114 - 113) / 113)
    assert feat.us_freight_tsi_yoy(obs) == pytest.approx((114 - 102) / 102)
    assert feat.us_freight_tsi_3m_momentum(obs) == pytest.approx((114 - 111) / 111)
    assert feat.us_freight_tsi_6m_momentum(obs) == pytest.approx((114 - 108) / 108)


def test_us_trucking_component_absent_returns_none():
    obs = [_obs("bts_freight_tsi", "us_freight_tsi", 0, 100)]
    assert feat.us_trucking_level(obs) is None
    assert feat.us_trucking_yoy(obs) is None


def test_eu_road_freight_level_yoy_and_z():
    obs = [_obs("eurostat_road_freight", "road_freight_tonnes", i * 365, 100 + i * 5, entity_id="DE")
           for i in range(6)]
    assert feat.eu_road_freight_level(obs) == 125
    assert feat.eu_road_freight_yoy(obs) == pytest.approx((125 - 120) / 120)
    z = feat.eu_road_freight_z(obs)
    assert z is None or isinstance(z, float)


def test_jp_truck_functions_use_given_metric():
    obs = [_obs("estat_jp_truck", "jp_truck_100", i * 30, 100 + i) for i in range(15)]
    assert feat.jp_truck_level(obs, "jp_truck_100") == 114
    assert feat.jp_truck_mom(obs, "jp_truck_100") == pytest.approx((114 - 113) / 113)
    assert feat.jp_truck_yoy(obs, "jp_truck_100") == pytest.approx((114 - 102) / 102)


def test_features_ignore_none_values():
    obs = [
        _obs("destatis_truck_toll", "index_sa", 0, 100),
        _obs("destatis_truck_toll", "index_sa", 1, None),
        _obs("destatis_truck_toll", "index_sa", 2, 102),
    ]
    assert feat.de_truck_level(obs) == 102


def test_zscore_excludes_current_value_from_baseline():
    # Baseline schwankt leicht um 100 (Varianz != 0), letzter Wert weicht
    # stark ab -> |z| deutlich > 0. Ein rein konstanter Baseline-Wert (stdev
    # 0) würde absichtlich None liefern (kein sinnvoller Z-Score).
    obs = [_obs("destatis_truck_toll", "index_sa", i, 100 + (1 if i % 2 == 0 else -1))
           for i in range(20)]
    obs.append(_obs("destatis_truck_toll", "index_sa", 20, 200))
    z = feat.de_truck_z_1y(obs)
    assert z is not None
    assert z > 5

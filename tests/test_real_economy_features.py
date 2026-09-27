"""
tests/test_real_economy_features.py

Reine Feature-Tests für modules/external/sources/real_economy_features.py
(kein I/O, keine Fixtures nötig) + modules/external/features.divergence
angewendet auf reale/synthetische real_economy-Z-Scores.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from modules.external import features as feat
from modules.external.pit import AvailabilityPrecision, Observation
from modules.external.sources import real_economy_features as ref

UTC = timezone.utc


def mk_obs(source_id, metric, value, obs_time, entity_id="", **kw):
    return Observation(
        source_id=source_id, dataset="d", series_id="s", entity_id=entity_id,
        metric=metric, value=value, unit="idx",
        observation_time=obs_time, available_at=obs_time, retrieved_at=obs_time,
        availability_precision=AvailabilityPrecision.EXACT_TIMESTAMP, parser_version="1", **kw,
    )


def _monthly_series(source_id, metric, values, entity_id="EU27_2020", start=datetime(2021, 1, 1, tzinfo=UTC)):
    obs = []
    for i, v in enumerate(values):
        year = start.year + (start.month - 1 + i) // 12
        month = (start.month - 1 + i) % 12 + 1
        t = datetime(year, month, 1, tzinfo=UTC)
        obs.append(mk_obs(source_id, metric, v, t, entity_id=entity_id))
    return obs


# ---------------------------------------------------------------------------
# esi_z / industrial_confidence_z / eu_survey_z
# ---------------------------------------------------------------------------

def test_esi_z_none_with_too_few_observations():
    obs = _monthly_series("eurostat_sentiment", "esi", [100.0, 101.0, 99.0])
    assert ref.esi_z(obs) is None  # min_periods=5 in rolling_zscore


def test_esi_z_positive_when_latest_is_high_outlier():
    values = [100.0, 100.5, 99.5, 100.2, 99.8, 100.1, 99.9, 100.0, 100.3, 108.0]
    obs = _monthly_series("eurostat_sentiment", "esi", values)
    z = ref.esi_z(obs)
    assert z is not None and z > 1.5


def test_esi_z_filters_by_entity_id():
    de = _monthly_series("eurostat_sentiment", "esi", [100.0] * 9 + [110.0], entity_id="DE")
    fr = _monthly_series("eurostat_sentiment", "esi", [100.0] * 10, entity_id="FR")
    z_de = ref.esi_z(de + fr, entity_id="DE")
    z_fr = ref.esi_z(de + fr, entity_id="FR")
    assert z_de is not None and z_de > 0
    assert z_fr == 0.0  # konstante Serie -> keine Abweichung


def test_industrial_confidence_z_ignores_other_metrics():
    esi = _monthly_series("eurostat_sentiment", "esi", [100.0] * 10)
    ici = _monthly_series("eurostat_sentiment", "industrial_confidence",
                           [0.0] * 9 + [10.0])
    z = ref.industrial_confidence_z(esi + ici)
    assert z is not None and z > 0


def test_eu_survey_z_combines_esi_and_ici_means():
    esi = _monthly_series("eurostat_sentiment", "esi", [100.0] * 9 + [110.0])
    ici = _monthly_series("eurostat_sentiment", "industrial_confidence", [0.0] * 9 + [10.0])
    combined = ref.eu_survey_z(esi + ici)
    esi_only = ref.esi_z(esi)
    ici_only = ref.industrial_confidence_z(ici)
    assert combined == pytest.approx((esi_only + ici_only) / 2)


def test_eu_survey_z_none_when_both_components_missing():
    assert ref.eu_survey_z([]) is None


def test_eu_survey_z_uses_only_available_component():
    esi = _monthly_series("eurostat_sentiment", "esi", [100.0] * 9 + [110.0])
    combined = ref.eu_survey_z(esi)  # keine industrial_confidence-Beobachtungen
    assert combined == pytest.approx(ref.esi_z(esi))


# ---------------------------------------------------------------------------
# industrial_production_z / momentum
# ---------------------------------------------------------------------------

def test_industrial_production_z_and_3m_momentum():
    # stetiger Trend (+0.5/Monat) und zuletzt ein Sprung: die Vorjahresrate
    # springt, obwohl ein Niveau-Z schon wegen des Trends hoch wäre.
    values = [100.0 + 0.5 * i for i in range(21)] + [120.0]
    obs = _monthly_series("eurostat_industrial_production", "production_volume_index", values)
    z = ref.industrial_production_z(obs)
    mom = ref.industrial_production_3m_momentum(obs)
    assert z is not None and z > 1.0
    assert mom == pytest.approx((120.0 - values[18]) / values[18])


def test_industrial_production_3m_momentum_none_with_short_history():
    obs = _monthly_series("eurostat_industrial_production", "production_volume_index", [100.0, 101.0])
    assert ref.industrial_production_3m_momentum(obs) is None


# ---------------------------------------------------------------------------
# fred_us_macro (survey/hard)
# ---------------------------------------------------------------------------

def test_us_survey_and_hard_z_and_momentum():
    umcsent = _monthly_series("fred_us_macro", "us_umcsent",
                               [70.0] * 9 + [80.0], entity_id="US")
    indpro = _monthly_series("fred_us_macro", "us_indpro",
                              [100.0] * 21 + [90.0], entity_id="US")
    obs = umcsent + indpro
    survey_z = ref.us_survey_z(obs)
    hard_z = ref.us_hard_z(obs)
    assert survey_z is not None and survey_z > 0
    assert hard_z is not None and hard_z < 0
    assert ref.us_survey_3m_momentum(obs) is not None
    assert ref.us_hard_3m_momentum(obs) is not None


def test_us_survey_z_none_when_no_fred_observations():
    assert ref.us_survey_z([]) is None
    assert ref.us_hard_z([]) is None


# ---------------------------------------------------------------------------
# feat.divergence: hard_data_vs_survey_divergence = hard_z - survey_z
# ---------------------------------------------------------------------------

def test_divergence_hard_minus_survey_positive_when_hard_stronger():
    out = feat.divergence(z_a=2.0, z_b=0.5)  # z_a=hard, z_b=survey
    assert out["divergence_z"] == pytest.approx(1.5)
    assert out["agreement"] in ("A_STRONGER", "BOTH_EXPANDING")


def test_divergence_none_when_either_side_missing():
    assert feat.divergence(None, 1.0) == {"divergence_z": None, "agreement": "UNKNOWN"}
    assert feat.divergence(1.0, None) == {"divergence_z": None, "agreement": "UNKNOWN"}
    assert feat.divergence(None, None) == {"divergence_z": None, "agreement": "UNKNOWN"}


def test_divergence_both_contracting_agreement():
    out = feat.divergence(z_a=-1.0, z_b=-1.2)
    assert out["agreement"] == "BOTH_CONTRACTING"
    assert out["divergence_z"] == pytest.approx(0.2)

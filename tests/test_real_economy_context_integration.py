"""
tests/test_real_economy_context_integration.py

Integrationstests: modules/external/context.py._build_real_economy() +
das primitives/supply_chain/quality-Wiring von hard_data_vs_survey_divergence
über ein echtes ExternalArchive (fixture-artige, synthetische Observations --
kein Netzwerk, keine echten Konnektoren nötig).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from modules.external.archive import ExternalArchive
from modules.external.pit import AvailabilityPrecision, Observation
from modules.external import context as ctxmod

UTC = timezone.utc


def mk_obs(source_id, metric, value, obs_time, entity_id="", **kw):
    return Observation(
        source_id=source_id, dataset="d", series_id="s", entity_id=entity_id,
        metric=metric, value=value, unit="idx",
        observation_time=obs_time, available_at=obs_time, retrieved_at=obs_time,
        availability_precision=AvailabilityPrecision.EXACT_TIMESTAMP, parser_version="1", **kw,
    )


def _monthly(source_id, metric, values, entity_id, start_year=2025, start_month=11):
    obs = []
    for i, v in enumerate(values):
        year = start_year + (start_month - 1 + i) // 12
        month = (start_month - 1 + i) % 12 + 1
        t = datetime(year, month, 1, tzinfo=UTC)
        obs.append(mk_obs(source_id, metric, v, t, entity_id=entity_id))
    return obs


NOW = datetime(2026, 8, 1, tzinfo=UTC)

# 10 Monate ESI/ICI/Produktion, letzter Monat jeweils ein positiver
# ESI/ICI-Ausreißer (Survey optimistischer) und ein negativer Produktions-
# Ausreißer (Hard Data schwächer) -> hard_z < survey_z -> divergence_z < 0.
ESI_VALUES = [100.0] * 9 + [110.0]
ICI_VALUES = [0.0] * 9 + [8.0]
PRODUCTION_VALUES = [100.0] * 9 + [90.0]


def _seed_eu_real_economy(archive: ExternalArchive):
    obs = []
    obs += _monthly("eurostat_sentiment", "esi", ESI_VALUES, "EU27_2020")
    obs += _monthly("eurostat_sentiment", "industrial_confidence", ICI_VALUES, "EU27_2020")
    obs += _monthly("eurostat_industrial_production", "production_volume_index",
                     PRODUCTION_VALUES, "EU27_2020")
    archive.store_observations(obs)


def test_real_economy_populated_from_archive(tmp_path):
    archive = ExternalArchive(root=tmp_path)
    _seed_eu_real_economy(archive)

    snap = ctxmod.build_external_context(NOW, archive=archive)
    eu = snap["real_economy"]["eu"]
    assert eu["survey_z"] is not None and eu["survey_z"] > 0
    assert eu["hard_z"] is not None and eu["hard_z"] < 0
    assert eu["divergence_z"] is not None and eu["divergence_z"] < 0
    assert set(eu["sources"]) == {"eurostat_sentiment", "eurostat_industrial_production"}
    assert snap["real_economy"]["us"] == {
        "survey_z": None, "hard_z": None, "divergence_z": None,
        "agreement": "UNKNOWN", "sources": [],
    }


def test_hard_data_vs_survey_divergence_primitive_no_longer_null(tmp_path):
    archive = ExternalArchive(root=tmp_path)
    _seed_eu_real_economy(archive)

    snap = ctxmod.build_external_context(NOW, archive=archive)
    assert snap["primitives"]["hard_data_vs_survey_divergence"] is not None
    assert snap["supply_chain"]["hard_data_vs_survey_divergence"] == (
        snap["primitives"]["hard_data_vs_survey_divergence"])
    assert snap["quality"]["hard_data_vs_survey_divergence_reason"] is None
    assert "real_economy" in snap["quality"]["families_with_data"]
    assert snap["divergences"]["hard_data_vs_survey_divergence_z"] == pytest.approx(
        snap["primitives"]["hard_data_vs_survey_divergence"])


def test_hard_data_vs_survey_divergence_none_with_reason_when_empty_archive(tmp_path):
    archive = ExternalArchive(root=tmp_path)  # komplett leer

    snap = ctxmod.build_external_context(NOW, archive=archive)
    assert snap["primitives"]["hard_data_vs_survey_divergence"] is None
    assert snap["quality"]["hard_data_vs_survey_divergence_reason"] == "missing_survey_and_hard_data"
    assert "real_economy" not in snap["quality"]["families_with_data"]
    assert snap["real_economy"]["eu"]["agreement"] == "UNKNOWN"


def test_hard_data_vs_survey_divergence_reason_missing_hard_data_only(tmp_path):
    archive = ExternalArchive(root=tmp_path)
    archive.store_observations(_monthly("eurostat_sentiment", "esi", ESI_VALUES, "EU27_2020"))

    snap = ctxmod.build_external_context(NOW, archive=archive)
    assert snap["primitives"]["hard_data_vs_survey_divergence"] is None
    assert snap["quality"]["hard_data_vs_survey_divergence_reason"] == "missing_hard_data"
    assert snap["real_economy"]["eu"]["survey_z"] is not None
    assert snap["real_economy"]["eu"]["hard_z"] is None


def test_hard_data_vs_survey_divergence_reason_missing_survey_data_only(tmp_path):
    archive = ExternalArchive(root=tmp_path)
    archive.store_observations(_monthly(
        "eurostat_industrial_production", "production_volume_index", PRODUCTION_VALUES, "EU27_2020"))

    snap = ctxmod.build_external_context(NOW, archive=archive)
    assert snap["primitives"]["hard_data_vs_survey_divergence"] is None
    assert snap["quality"]["hard_data_vs_survey_divergence_reason"] == "missing_survey_data"
    assert snap["real_economy"]["eu"]["hard_z"] is not None
    assert snap["real_economy"]["eu"]["survey_z"] is None


def test_eu_hard_z_blends_industrial_production_with_road_freight_when_present(tmp_path):
    """Straßengüterverkehr (EU27_2020) fließt NUR ein, wenn der road_freight-
    Konnektor diese Entity tatsächlich liefert -- sonst bleibt hard_z rein
    industrieproduktionsbasiert (siehe vorherige Tests)."""
    archive = ExternalArchive(root=tmp_path)
    obs = []
    obs += _monthly("eurostat_sentiment", "esi", ESI_VALUES, "EU27_2020")
    obs += _monthly("eurostat_industrial_production", "production_volume_index",
                     PRODUCTION_VALUES, "EU27_2020")
    # Jahresreihe (12 Jahre) für eurostat_road_freight/EU27_2020, wie vom
    # bestehenden road_freight-Feature-Fenster (365*12 Tage) erwartet.
    road_values = [1000.0, 1010.0, 995.0, 1005.0, 1002.0, 998.0,
                   1003.0, 997.0, 1001.0, 999.0, 1004.0, 1300.0]
    for i, v in enumerate(road_values):
        t = datetime(2015 + i, 1, 1, tzinfo=UTC)
        obs.append(mk_obs("eurostat_road_freight", "road_freight_ths_t", v, t, entity_id="EU27_2020"))
    archive.store_observations(obs)

    snap = ctxmod.build_external_context(datetime(2026, 1, 1, tzinfo=UTC), archive=archive)
    eu = snap["real_economy"]["eu"]
    assert "eurostat_road_freight" in eu["sources"]
    assert eu["hard_z"] is not None


def test_real_economy_us_block_from_fred_observations(tmp_path):
    archive = ExternalArchive(root=tmp_path)
    obs = []
    obs += _monthly("fred_us_macro", "us_umcsent", [70.0] * 9 + [80.0], "US")
    obs += _monthly("fred_us_macro", "us_indpro", [100.0] * 9 + [92.0], "US")
    archive.store_observations(obs)

    snap = ctxmod.build_external_context(NOW, archive=archive)
    us = snap["real_economy"]["us"]
    assert us["survey_z"] is not None and us["survey_z"] > 0
    assert us["hard_z"] is not None and us["hard_z"] < 0
    assert us["divergence_z"] is not None
    assert us["sources"] == ["fred_us_macro"]
    # Der globale hard_data_vs_survey_divergence-Primitive bleibt EU-basiert
    # -- ein US-Datensatz allein füllt ihn NICHT.
    assert snap["primitives"]["hard_data_vs_survey_divergence"] is None


def test_real_economy_never_raises_on_broken_archive():
    class BrokenArchive:
        def as_of(self, source_id, t, filters=None):
            raise RuntimeError("boom")

    snap = ctxmod.build_external_context(NOW, archive=BrokenArchive())
    assert snap["real_economy"]["eu"]["survey_z"] is None
    assert snap["real_economy"]["us"]["hard_z"] is None
    assert snap["primitives"]["hard_data_vs_survey_divergence"] is None

"""
tests/test_weather_exposure.py

Exposure-Hierarchie (ticker override > industry > sector > unknown), HQ wird
nie verwendet, "unknown" liefert None statt "NONE"/0.
"""

from pathlib import Path

import yaml

from modules.external.sources import weather as w

REPO_ROOT = Path(__file__).resolve().parent.parent


def _configs():
    exposures = yaml.safe_load((REPO_ROOT / "config" / "weather_exposures.yaml").read_text())
    industry = yaml.safe_load((REPO_ROOT / "config" / "industry_exposure.yaml").read_text())
    return exposures, industry


def test_ticker_override_wins_over_industry_and_sector():
    exposures, industry = _configs()
    result = w.resolve_weather_relevance(
        ticker="DAL", yf_sector="Industrials", yf_industry="Trucking",
        exposures_cfg=exposures, industry_cfg=industry,
    )
    assert result["level"] == "ticker"
    assert result["weather_relevance"] == "HIGH"
    assert "ATL" in result["hub_codes"]


def test_industry_wins_over_sector_when_no_ticker_override():
    exposures, industry = _configs()
    result = w.resolve_weather_relevance(
        ticker="SOME_UNMAPPED_TICKER", yf_sector="Industrials", yf_industry="Trucking",
        exposures_cfg=exposures, industry_cfg=industry,
    )
    assert result["level"] == "industry"
    assert result["industry"] == "Road Freight / Trucking"
    assert result["weather_relevance"] == "HIGH"


def test_sector_fallback_when_industry_unmapped():
    exposures, industry = _configs()
    result = w.resolve_weather_relevance(
        ticker="SOME_UNMAPPED_TICKER", yf_sector="Utilities", yf_industry="Some Unmapped Industry String",
        exposures_cfg=exposures, industry_cfg=industry,
    )
    assert result["level"] == "sector"
    assert result["industry"] == "Utilities"
    assert result["weather_relevance"] == "HIGH"


def test_unknown_when_nothing_matches_returns_none_not_zero_or_none_string():
    exposures, industry = _configs()
    result = w.resolve_weather_relevance(
        ticker="SOME_UNMAPPED_TICKER", yf_sector="Not A Real Sector", yf_industry="Not A Real Industry",
        exposures_cfg=exposures, industry_cfg=industry,
    )
    assert result["level"] == "unknown"
    assert result["weather_relevance"] is None
    assert result["hub_codes"] is None


def test_ticker_override_absent_falls_through_hierarchy():
    exposures, industry = _configs()
    result = w.resolve_weather_relevance(
        ticker=None, yf_sector="Utilities", yf_industry="Utilities—Regulated Electric",
        exposures_cfg=exposures, industry_cfg=industry,
    )
    assert result["level"] == "industry"
    assert result["industry"] == "Utilities"


def test_resolve_exposure_hub_codes_none_for_unknown_ticker():
    exposures, _ = _configs()
    assert w.resolve_exposure_hub_codes("NOT_A_TICKER", exposures_cfg=exposures) is None


def test_resolve_exposure_hub_codes_for_known_ticker():
    exposures, _ = _configs()
    codes = w.resolve_exposure_hub_codes("FDX", exposures_cfg=exposures)
    assert codes == ["MEM_FDX"]


def test_ticker_overrides_never_carry_an_hq_field():
    """HQ darf nie als Exposure-Region verwendet werden – config-seitig
    stellen wir sicher, dass gar kein HQ-Feld existiert, das versehentlich
    als Fallback genutzt werden könnte."""
    exposures, _ = _configs()
    for ticker, override in exposures["ticker_overrides"].items():
        assert "hq" not in {k.lower() for k in override}
        assert "headquarters" not in {k.lower() for k in override}


def test_resolve_weather_relevance_signature_has_no_hq_parameter():
    import inspect
    sig = inspect.signature(w.resolve_weather_relevance)
    assert "hq" not in sig.parameters
    assert "headquarters" not in sig.parameters


def test_all_ticker_overrides_have_source_and_rationale():
    exposures, _ = _configs()
    for ticker, override in exposures["ticker_overrides"].items():
        assert override.get("source"), f"{ticker} fehlt 'source'"
        assert override.get("rationale"), f"{ticker} fehlt 'rationale'"


def test_industry_relevance_values_are_valid_categories_no_direction():
    _, industry = _configs()
    valid = {"NONE", "LOW", "MEDIUM", "HIGH"}
    for name, spec in industry["industries"].items():
        assert spec["weather_relevance"] in valid
        assert spec["road_freight_relevance"] in valid
        assert spec["maritime_relevance"] in valid

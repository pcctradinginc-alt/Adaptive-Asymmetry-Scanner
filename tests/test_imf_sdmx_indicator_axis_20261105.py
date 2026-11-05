"""Regression tests for IMF SDMX indicator-axis discovery; offline only."""
import pytest
from modules.external.sources.imf_sdmx import _select, IMFSDMXError

def test_nonstandard_axis_from_unique_official_label():
    dims = [("COUNTRY", "CL_COUNTRY"), ("MEASURE", "CL_MEASURE"),
            ("FREQUENCY", "CL_FREQ")]
    lists = {
        "CL_COUNTRY": {"DE": "Germany", "US": "United States"},
        "CL_MEASURE": {"BROAD_MONEY_OFFICIAL": "Broad money",
                       "UNRELATED": "Other"},
        "CL_FREQ": {"A": "Annual"},
    }
    key, selected = _select(dims, lists, "MFS_MA", ("broad money",))
    assert key == "DE+US.BROAD_MONEY_OFFICIAL.A"
    assert selected["indicator_dimension"] == "MEASURE"
    assert selected["indicator_codes"] == ["BROAD_MONEY_OFFICIAL"]

def test_multiple_matching_axes_fail_closed():
    dims = [("COUNTRY", "CL_COUNTRY"), ("MEASURE", "CL_MEASURE"),
            ("OTHER", "CL_OTHER")]
    lists = {"CL_COUNTRY": {"DE": "Germany", "US": "United States"},
             "CL_MEASURE": {"X": "Broad money"},
             "CL_OTHER": {"Y": "Broad money"}}
    with pytest.raises(IMFSDMXError, match="INDICATOR_AXIS_NOT_UNAMBIGUOUS"):
        _select(dims, lists, "MFS_MA", ("broad money",))

def test_ambiguous_label_does_not_guess_code():
    dims = [("COUNTRY", "CL_COUNTRY"), ("INDICATOR", "CL_IND")]
    lists = {"CL_COUNTRY": {"DE": "Germany", "US": "United States"},
             "CL_IND": {"A": "Broad money", "B": "Broad money"}}
    with pytest.raises(IMFSDMXError, match="INDICATOR_AXIS_NOT_UNAMBIGUOUS"):
        _select(dims, lists, "MFS_MA", ("broad money",))

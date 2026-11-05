"""Official IMF SDMX3 WEO/FM research adapter: strict JSON 2.0 PIT tests."""
from datetime import datetime, timezone
import json

import pytest

from modules.external import http
from modules.external.sources.base import SourceStatus
from modules.external.sources.imf_sdmx3 import (
    FLOW_SPECS, SDMX3Error, fetch_research, parse_data
)

NOW = datetime(2026, 11, 5, 5, tzinfo=timezone.utc)


def result(payload, *, url="https://api.imf.org/external/sdmx/3.0/data/dataflow/IMF.RES/WEO/+/DEU.NGDP_RPCH.A"):
    if isinstance(payload, dict):
        payload = json.dumps(payload).encode()
    return http.FetchResult(url=url, status=200, content=payload,
                            content_type="application/vnd.sdmx.data+json;version=2.0",
                            retrieved_at=NOW, content_hash="sha", fingerprint="fp",
                            bytes=len(payload))


def sample(country="DEU", indicator="NGDP_RPCH", freq="A", *,
           years=("2025", "2026", "2027"), values=(2.1, 1.8, 2.4),
           key=None, sector=None):
    if key is None:
        key = "0:0:0:0" if sector is not None else "0:0:0"
    return {
        "meta": {"id": "provider-test"},
        "data": {
            "structures": [{
                "dimensions": {
                    "series": [
                        {"id": "COUNTRY", "values": [{"id": country, "name": country}]},
                        *([{"id": "SECTOR", "values": [{"id": sector}]}]
                          if sector is not None else []),
                        {"id": "INDICATOR", "values": [{"id": indicator}]},
                        {"id": "FREQUENCY", "values": [{"id": freq}]},
                    ],
                    "observation": [
                        {"id": "TIME_PERIOD", "values": [{"name": d} for d in years]},
                    ],
                },
                "measures": {"observation": [{"id": "OBS_VALUE"}]},
            }],
            "dataSets": [{
                "structure": 0, "series": {
                    key: {"observations": {
                        str(i): [v] for i, v in enumerate(values)
                    }}
                },
            }],
        },
    }


def test_valid_weo_json2_decodes_series_and_stamps_prospective_pit():
    x = parse_data(result(sample()), source_id="imf_weo",
                   country="DEU", indicator="NGDP_RPCH",
                   spec=FLOW_SPECS["imf_weo"])
    assert len(x) == 3
    assert {o.observation_time.year for o in x} == {2025, 2026, 2027}
    assert all(o.available_at == NOW for o in x)
    assert all(o.attrs["historical_pit_verified"] is False for o in x)
    future = [o for o in x if o.observation_time.year == 2027]
    assert len(future) == 1 and future[0].forecast_issue_time == NOW
    assert future[0].forecast_valid_time.year == 2027


@pytest.mark.parametrize("changes,reason", [
    ({"country": "USA"}, "UNREQUESTED_COUNTRY"),
    ({"indicator": "WRONG"}, "UNREQUESTED_COUNTRY"),
    ({"freq": "Q"}, "UNREQUESTED_COUNTRY"),
    ({"key": "2:0:0"}, "SERIES_CODE_INDEX"),
    ({"values": ("NaN", 2, 3)}, "NON_FINITE"),
    ({"values": (True, 2, 3)}, "BOOLEAN"),
])
def test_malformed_or_spoofed_values_never_pass(changes, reason):
    with pytest.raises(SDMX3Error, match=reason):
        parse_data(result(sample(**changes)), source_id="imf_weo",
                   country="DEU", indicator="NGDP_RPCH",
                   spec=FLOW_SPECS["imf_weo"])


@pytest.mark.parametrize("raw", [b"", b"<html>error</html>", b"{}", b"{invalid"])
def test_zero_bytes_and_invalid_html_json_fail(raw):
    with pytest.raises(SDMX3Error):
        parse_data(result(raw), source_id="imf_weo", country="DEU",
                   indicator="NGDP_RPCH", spec=FLOW_SPECS["imf_weo"])


def test_missing_values_are_skipped_not_zero_filled():
    x = parse_data(result(sample(values=(None, 0, 2.4))),
                   source_id="imf_weo", country="DEU",
                   indicator="NGDP_RPCH", spec=FLOW_SPECS["imf_weo"])
    assert len(x) == 2 and [o.value for o in x] == [0.0, 2.4]


def test_wrong_dimension_order_blocked():
    body = sample()
    dims = body["data"]["structures"][0]["dimensions"]["series"]
    dims[0], dims[1] = dims[1], dims[0]
    with pytest.raises(SDMX3Error, match="DIMENSION_ORDER"):
        parse_data(result(body), source_id="imf_weo", country="DEU",
                   indicator="NGDP_RPCH", spec=FLOW_SPECS["imf_weo"])


def test_well_scoped_weo_and_fm_all_countries_and_indicators():
    from modules.external.pit import available_as_of
    for source_id, spec in FLOW_SPECS.items():
        def fake(url, **kw):
            segments = url.rsplit("/", 1)[-1].split(".")
            if spec.get("sector"):
                country, sector, indicator, freq = segments
                assert sector == spec["sector"]
            else:
                country, indicator, freq = segments
            assert country in spec["countries"]
            assert indicator in spec["indicators"]
            assert freq == spec["frequency"]
            assert kw["params"]["format"] == "jsondata"
            assert "startPeriod" in kw["params"] and "endPeriod" in kw["params"]
            return result(sample(country=country, indicator=indicator,
                                 years=("2025", "2026"), values=(1.0, 2.0),
                                 sector=spec.get("sector")), url=url)
        x = fetch_research(source_id, NOW, fetch_fn=fake)
        assert x.status is SourceStatus.PASS, x.message
        assert len(x.observations) == len(spec["indicators"]) * len(spec["countries"]) * 2
        assert len(x.raw) == len(spec["indicators"]) * len(spec["countries"])
        assert all(o.attrs["research_only"] is True for o in x.observations)
        assert available_as_of(x.observations,
                               datetime(2025, 1, 1, tzinfo=timezone.utc)) == []


def test_any_failed_subrequest_returns_zero_partial_observations():
    spec = FLOW_SPECS["imf_weo"]
    calls = [0]
    def flaky(url, **kw):
        calls[0] += 1
        if calls[0] == 2:
            raise http.FetchError("HTTP 503 official upstream")
        country, indicator, freq = url.rsplit("/", 1)[-1].split(".")
        return result(sample(country=country, indicator=indicator,
                             years=("2025",), values=(1.0,)), url=url)
    r = fetch_research("imf_weo", NOW, fetch_fn=flaky)
    assert r.status is SourceStatus.FAIL
    assert not r.observations
    assert calls[0] == 2

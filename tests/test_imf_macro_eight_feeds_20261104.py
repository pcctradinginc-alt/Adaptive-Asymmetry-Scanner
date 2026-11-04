"""Offline acceptance tests for 8 official IMF research feeds and PIT integrity."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import pytest

from modules.external import http
from modules.external.pit import available_as_of
from modules.external.sources import imf_datamapper as dm
from modules.external.sources import imf_sdmx as sd
from modules.external.sources.base import SourceStatus

NOW = datetime(2026, 11, 4, 22, 0, tzinfo=timezone.utc)


def response(body: bytes, url="https://www.imf.org", *, when=NOW, type_="application/json"):
    return http.FetchResult(url=url, status=200, content=body, content_type=type_,
                            retrieved_at=when, content_hash="hash", fingerprint="fingerprint",
                            bytes=len(body))


def catalog():
    d = {}
    for c in dm.CONNECTORS.values():
        for indicator in c.INDICATORS:
            d[indicator] = {"dataset": c.DATASET, "label": indicator, "unit": "percent",
                            **({"projection-year": 2026} if c.DATASET == "WEO" else {})}
    return {"indicators": d}


class DataMapperFake:
    def __init__(self, *, bad_catalog=False, empty_code=None, invalid_value=None):
        self.bad_catalog = bad_catalog
        self.empty_code = empty_code
        self.invalid_value = invalid_value
        self.calls = []
    def __call__(self, url, **kw):
        self.calls.append(url)
        if url.endswith("/indicators"):
            data = catalog()
            if self.bad_catalog:
                data["indicators"]["NGDP_RPCH"]["dataset"] = "GDD"
            return response(json.dumps(data).encode(), url)
        code = url.rsplit("/", 1)[-1]
        if code == self.empty_code:
            return response(json.dumps({"values": {code: {}}}).encode(), url)
        series = {"USA": {"2023": 1.5, "2024": 2.0, "2026": 2.25, "2027": 2.5},
                  "DEU": {"2023": 3.0, "2024": 4.0, "2026": 3.5}}
        if self.invalid_value is not None:
            series["USA"]["2024"] = self.invalid_value
        return response(json.dumps({"values": {code: series}}).encode(), url)


@pytest.mark.parametrize("cls", list(dm.CONNECTORS.values()))
def test_four_datamapper_feeds_ingest_only_approved_prospective_vintages(cls):
    fake = DataMapperFake()
    x = cls({}, fetch_fn=fake).fetch(NOW)
    assert x.status is SourceStatus.PASS
    assert x.observations and x.raw and len(x.raw) == len(cls.INDICATORS) + 1
    assert {o.entity_id for o in x.observations} == {"USA", "DEU"}
    assert {o.series_id for o in x.observations} == {
        f"{cls.DATASET}:{k}" for k in cls.INDICATORS}
    assert {o.available_at for o in x.observations} == {NOW}
    assert all(o.attrs["historical_pit_verified"] is False for o in x.observations)
    assert available_as_of(x.observations, datetime(2025, 1, 1, tzinfo=timezone.utc)) == []
    assert x.discovered_ids["historical_pit_verified"] is False
    assert all(o.unit == "percent" for o in x.observations)
    if cls.DATASET == "WEO":
        future = [o for o in x.observations if o.observation_time.year == 2027]
        assert future and all(o.forecast_issue_time == NOW and o.forecast_valid_time for o in future)
        current_projection = [o for o in x.observations if o.observation_time.year == 2026]
        assert current_projection and all(o.forecast_issue_time == NOW for o in current_projection)
    else:
        assert all(o.forecast_issue_time is None for o in x.observations)


def test_datamapper_rejects_invalid_metadata_without_writing_observations():
    x = dm.IMFWEOConnector({}, fetch_fn=DataMapperFake(bad_catalog=True)).fetch(NOW)
    assert x.status is SourceStatus.SCHEMA_CHANGED
    assert not x.observations


def test_datamapper_rejects_missing_series_without_partial_ingestion():
    x = dm.IMFWEOConnector({}, fetch_fn=DataMapperFake(empty_code="PCPIPCH")).fetch(NOW)
    assert x.status is SourceStatus.SCHEMA_CHANGED  # selected indicator entirely absent is not a valid success
    assert not x.observations


@pytest.mark.parametrize("invalid", ["NaN", "Infinity", "not-a-number", True])
def test_datamapper_rejects_bad_numeric_values(invalid):
    x = dm.IMFGDDConnector({}, fetch_fn=DataMapperFake(invalid_value=invalid)).fetch(NOW)
    assert x.status is SourceStatus.SCHEMA_CHANGED
    assert not x.observations


def test_datamapper_missing_values_are_not_fabricated_as_zero():
    c = dm.IMFWEOConnector({}, fetch_fn=DataMapperFake())
    raw = response(json.dumps({"values": {"NGDP_RPCH": {
        "USA": {"2025": None}, "DEU": {"2025": 3.0}}}}).encode())
    vals = dm.parse_indicator(raw, "NGDP_RPCH", "WEO",
                              {"source_id": c.source_id, "_validated_units": {"NGDP_RPCH": "percent"}},
                              NOW)
    assert len(vals) == 1 and vals[0].entity_id == "DEU"


def synthetic_structure(flow, label1, label2):
    return f"""<?xml version="1.0"?>
<mes:Structure xmlns:mes="urn:mes" xmlns:str="urn:struct" xmlns:com="urn:common"
xmlns:xml="http://www.w3.org/XML/1998/namespace">
<str:Dataflows><str:Dataflow id="{flow}" agencyID="IMF.STA"><str:Structure><Ref id="DSD_{flow}"/></str:Structure></str:Dataflow></str:Dataflows>
<str:DataStructures><str:DataStructure id="DSD_{flow}" agencyID="IMF.STA">
<str:DataStructureComponents><str:DimensionList>
<str:Dimension id="COUNTRY"><str:LocalRepresentation><str:Enumeration><Ref id="COUNTRIES"/></str:Enumeration></str:LocalRepresentation></str:Dimension>
<str:Dimension id="INDICATOR"><str:LocalRepresentation><str:Enumeration><Ref id="INDICATORS"/></str:Enumeration></str:LocalRepresentation></str:Dimension>
<str:Dimension id="FREQUENCY"><str:LocalRepresentation><str:Enumeration><Ref id="FREQUENCIES"/></str:Enumeration></str:LocalRepresentation></str:Dimension>
<str:TimeDimension id="TIME_PERIOD"/>
</str:DimensionList></str:DataStructureComponents></str:DataStructure></str:DataStructures>
<str:Codelists><str:Codelist id="COUNTRIES">
<str:Code id="USA"><com:Name xml:lang="en">United States</com:Name></str:Code>
<str:Code id="DEU"><com:Name xml:lang="en">Germany</com:Name></str:Code>
</str:Codelist><str:Codelist id="INDICATORS">
<str:Code id="C1"><com:Name xml:lang="en">{label1}</com:Name></str:Code>
<str:Code id="C2"><com:Name xml:lang="en">{label2}</com:Name></str:Code>
</str:Codelist><str:Codelist id="FREQUENCIES">
<str:Code id="Q"><com:Name>Quarterly</com:Name></str:Code>
<str:Code id="A"><com:Name>Annual</com:Name></str:Code>
</str:Codelist></str:Codelists>
</mes:Structure>""".encode()


def synthetic_data():
    return b"""<message:StructureSpecificData xmlns:message="urn:sdmx">
<message:DataSet>
<Series COUNTRY="USA" INDICATOR="C1" FREQUENCY="Q"><Obs TIME_PERIOD="2026-Q2" OBS_VALUE="6.2"/><Obs TIME_PERIOD="2026-Q3" OBS_VALUE="6.8"/></Series>
<Series COUNTRY="DEU" INDICATOR="C2" FREQUENCY="Q"><Obs TIME_PERIOD="2026-Q2" OBS_VALUE="2.1"/></Series>
</message:DataSet></message:StructureSpecificData>"""


class SDMXFake:
    def __init__(self, flow="FSIC", phrases=None, *, fail_mode="", body=None):
        self.flow = flow
        self.phrases = phrases or sd.IMFFSIConnector.PHRASES
        self.fail_mode = fail_mode
        self.body = synthetic_data() if body is None else body
        self.urls = []
    def __call__(self, url, **kwargs):
        self.urls.append(url)
        if "/dataflow/" in url:
            text = synthetic_structure(self.flow, *self.phrases[:2])
            if self.fail_mode == "wrong_publisher":
                text = text.replace(b'agencyID="IMF.STA"', b'agencyID="OTHER"', 1)
            if self.fail_mode == "unknown_indicators":
                text = text.replace(self.phrases[0].encode(), b"Unrelated series")
                text = text.replace(self.phrases[1].encode(), b"Another series")
            return response(text, url, type_="application/xml")
        return response(self.body, url, type_="application/xml")


@pytest.mark.parametrize("cls", list(sd.CONNECTORS.values()))
def test_four_sdmx_feeds_validate_dsd_key_and_archive_forward_only(cls):
    fake = SDMXFake(cls.FLOW, cls.PHRASES)
    x = cls({}, fetch_fn=fake).fetch(NOW)
    assert x.status is SourceStatus.PASS, x.message
    assert len(x.observations) == 3
    assert x.discovered_ids["countries"] == ["USA", "DEU"]
    assert x.discovered_ids["indicator_codes"] == ["C1", "C2"]
    assert x.discovered_ids["flow"] == cls.FLOW
    assert all(o.available_at == NOW for o in x.observations)
    assert all(o.attrs["historical_pit_verified"] is False for o in x.observations)
    assert not available_as_of(x.observations, datetime(2025, 1, 1, tzinfo=timezone.utc))
    assert "lastNObservations" not in fake.urls[-1]  # bounded parameter in kwargs


@pytest.mark.parametrize("error", ["wrong_publisher", "unknown_indicators"])
def test_sdmx_unverified_metadata_fails_closed(error):
    x = sd.IMFFSIConnector({}, fetch_fn=SDMXFake(fail_mode=error)).fetch(NOW)
    assert x.status is SourceStatus.SCHEMA_CHANGED
    assert not x.observations


@pytest.mark.parametrize("bad", [
    b"", b"<html>Invalid URL</html>",
    b"<StructureSpecificData><DataSet><Series COUNTRY='USA' INDICATOR='C1'><Obs TIME_PERIOD='2026-Q1'/></Series></DataSet></StructureSpecificData>",
    b"<StructureSpecificData><DataSet><Series COUNTRY='USA' INDICATOR='C1'><Obs TIME_PERIOD='2026-Q1' OBS_VALUE='NaN'/></Series></DataSet></StructureSpecificData>",
])
def test_sdmx_bad_or_empty_data_never_creates_observations(bad):
    x = sd.IMFFSIConnector({}, fetch_fn=SDMXFake(body=bad)).fetch(NOW)
    assert x.status is SourceStatus.SCHEMA_CHANGED
    assert not x.observations


def test_reject_sdmx_duplicate_observation_identities():
    data = synthetic_data().replace(b'<Obs TIME_PERIOD="2026-Q3" OBS_VALUE="6.8"/>',
                                    b'<Obs TIME_PERIOD="2026-Q2" OBS_VALUE="6.8"/>')
    x = sd.IMFFSIConnector({}, fetch_fn=SDMXFake(body=data)).fetch(NOW)
    assert x.status is SourceStatus.SCHEMA_CHANGED
    assert "DUPLICATE_SERIES" in x.message


def test_all_eight_feeds_in_repo_registry_are_research_only():
    from modules.external.registry import SourceRegistry
    reg = SourceRegistry()
    ids = set(dm.CONNECTORS) | set(sd.CONNECTORS)
    assert len(ids) == 8
    for sid in ids:
        cfg = reg.sources[sid]
        assert cfg["family"] == "imf_macro"
        assert cfg["enabled"] is True
        assert cfg["research_only"] is True
        assert cfg["production_promotion_allowed"] is False
        assert cfg["pit_quality"] == "CONSERVATIVE_DATE"
        assert cfg["min_refetch_days"] >= 14
        assert reg.connector_for(sid) is not None


def test_imf_indicator_series_are_not_a_historical_release_vintage():
    fake = DataMapperFake()
    x = dm.IMFWEOConnector({}, fetch_fn=fake).fetch(NOW)
    assert x.status == SourceStatus.PASS
    # Even 2015/2023 IMF observations are NOT available to a 2025 test.
    older = [o for o in x.observations if o.observation_time.year == 2023]
    assert older
    assert not available_as_of(older, datetime(2025, 12, 31, tzinfo=timezone.utc))

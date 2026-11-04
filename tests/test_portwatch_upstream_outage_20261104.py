"""IMF PortWatch outage/recovery regression tests (2026-11-04).

All responses are in-memory fakes. Passing these tests is NOT proof that IMF's
live ArcGIS publisher has restored its data services.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from modules.external.sources import maritime as mw
from modules.external.sources.base import SourceStatus
from tests.test_maritime_source import (
    NOW, SERVICE_URLS, FakeResponse, _load, make_fake_fetch,
)


def _upstream_400(message: str) -> dict:
    return {"error": {"code": 400, "message": message,
                      "details": ["Object reference not set to an instance of an object."]}}


def test_arcgis_400_json_is_upstream_unavailable_not_schema_drift():
    url = SERVICE_URLS["ports_daily"]
    bad = lambda _url, params=None, **kw: FakeResponse(
        _upstream_400("Layer 'Daily_Ports_Data' does not exist or is inaccessible."))
    with pytest.raises(mw.PortWatchUpstreamUnavailable) as exc:
        mw.fetch_layer_metadata(url, 0, bad)
    assert "UPSTREAM_UNAVAILABLE" in str(exc.value)
    assert "arcgis_code=400" in str(exc.value)
    assert url in str(exc.value)

    # A successful JSON response without fields is still a *real* schema issue.
    with pytest.raises(mw.SchemaError):
        mw.fetch_layer_metadata(url, 0, lambda *_a, **_kw: FakeResponse({"id": 0}))


def test_daily_ports_upstream_400_fails_closed_and_can_recover(tmp_path):
    fake_fetch, _ = make_fake_fetch()
    outage = {"down": True}

    def fetch(url, params=None, **kwargs):
        if outage["down"] and url == SERVICE_URLS["ports_daily"] + "/0":
            return FakeResponse(_upstream_400("Layer 'Daily_Ports_Data' does not exist or is inaccessible."))
        return fake_fetch(url, params=params, **kwargs)

    connector = mw.PortWatchPortsConnector(http_fetch=fetch,
                                           cache_path=str(tmp_path / "endpoints.json"))
    failed = connector.fetch(NOW)
    assert failed.status == SourceStatus.FAIL
    assert "UPSTREAM_UNAVAILABLE" in failed.message
    assert failed.observations == [] and failed.raw == []

    outage["down"] = False
    recovered = connector.fetch(NOW)
    assert recovered.status in (SourceStatus.PASS, SourceStatus.WARN)
    assert recovered.observations and recovered.raw
    # A new *real* provider response, never an invented stale fallback.


def test_arcgis_root_error_is_never_silently_assigned_layer_zero():
    url = SERVICE_URLS["chokepoints_daily"]
    with pytest.raises(mw.PortWatchUpstreamUnavailable):
        mw._resolve_layer(url, lambda *_a, **_kw:
                          FakeResponse(_upstream_400("Invalid URL")))
    # PortWatch daily services may publish a TABLE instead of a MAP layer.
    layer = mw._resolve_layer(url, lambda *_a, **_kw:
                              FakeResponse({"layers": [], "tables": [{"id": 7, "name": "daily"}]}))
    assert layer == {"layer_id": 7}
    with pytest.raises(mw.DiscoveryError):
        mw._resolve_layer(url, lambda *_a, **_kw: FakeResponse({"layers": [], "tables": []}))


def test_chokepoints_minimal_probe_confirms_invalid_url_without_extra_queries():
    url = SERVICE_URLS["chokepoints_daily"]
    seen = []

    def fetch(_url, params=None, **kwargs):
        seen.append(dict(params or {}))
        return FakeResponse({"error": {"code": 400, "message": "Invalid URL",
                                       "details": ["Invalid URL"]}})

    iso = mw.isolate_query_error(url, 0, fetch,
                                 where="date >= DATE '2026-11-01'",
                                 out_fields="date,portid,n_total",
                                 order_by="date ASC", offset=0,
                                 base_field="ObjectId", page_size=1000)
    assert len(seen) == 1
    assert seen[0]["where"] == "1=1" and seen[0]["outFields"] == "ObjectId"
    assert iso["basis"]["result"] == "FAIL(400)"
    assert all(v["result"] == "SKIPPED(UPSTREAM_DOWN)"
               for k, v in iso.items() if k != "basis")


def test_arcgis_chokepoints_query_fail_is_explicit_and_no_partial_data():
    url = SERVICE_URLS["chokepoints_daily"]
    called = []

    def fetch(_url, params=None, **kwargs):
        called.append(dict(params or {}))
        return FakeResponse({"error": {"code": 400, "message": "Invalid URL",
                                       "details": ["Invalid URL"]}})

    with pytest.raises(mw.ArcGISQueryError) as exc:
        mw.query_features(url, 0, "date >= DATE '2026-11-01'", ["date", "portid", "n_total"],
                          fetch, order_by="date ASC", max_record_count=1000,
                          isolation_field="ObjectId")
    assert "[UPSTREAM_UNAVAILABLE]" in str(exc.value)
    assert "[Isolation basis=FAIL(400)" in str(exc.value)
    # Main query + one minimal probe; no synthetic records, no five expensive probes.
    assert len(called) == 2


def test_isolation_still_tests_filter_when_minimal_endpoint_works():
    url = SERVICE_URLS["ports_daily"]
    def fetch(_url, params=None, **kwargs):
        if params and "date >=" in params.get("where", ""):
            return FakeResponse({"error": {"code": 400, "message": "Invalid query parameters",
                                           "details": ["bad where"]}})
        return FakeResponse({"features": [], "exceededTransferLimit": False})
    iso = mw.isolate_query_error(url, 0, fetch,
                                 where="date >= DATE '2026-11-01'",
                                 out_fields="date,portid,portcalls",
                                 order_by="date ASC", offset=0,
                                 base_field="ObjectId", page_size=1000)
    assert iso["basis"]["result"] == "OK"
    assert iso["where"]["result"] == "FAIL(400)"
    assert iso["outFields"]["result"] == "OK"



def test_reference_service_outage_falls_back_to_verified_daily_layer(tmp_path):
    base, _ = make_fake_fetch()

    def fetch(url, params=None, **kwargs):
        if url == SERVICE_URLS["ports_reference"]:
            return FakeResponse(_upstream_400("Invalid URL"))
        return base(url, params=params, **kwargs)

    endpoints = mw.discover_endpoints(http_fetch=fetch,
                                      cache_path=str(tmp_path / "endpoints.json"), now=NOW)
    assert endpoints["ports_reference"]["fallback_to_daily_layer"] is True
    assert endpoints["ports_reference"]["service_url"] == endpoints["ports_daily"]["service_url"]
    assert endpoints["ports_reference"]["layer_id"] == endpoints["ports_daily"]["layer_id"]
    # The fallback uses the same verified publisher, not another ArcGIS owner.



def test_failed_forced_discovery_cannot_poison_last_known_endpoint_cache(tmp_path):
    original_fetch, _ = make_fake_fetch()
    cache_path = tmp_path / "endpoints.json"
    mw.discover_endpoints(http_fetch=original_fetch, cache_path=str(cache_path), now=NOW)
    before = cache_path.read_bytes()

    def outage_fetch(url, params=None, **kwargs):
        if url == SERVICE_URLS["ports_daily"]:
            return FakeResponse(_upstream_400("Invalid URL"))
        return original_fetch(url, params=params, **kwargs)

    with pytest.raises(mw.PortWatchUpstreamUnavailable):
        mw.discover_endpoints(http_fetch=outage_fetch, cache_path=str(cache_path),
                              now=NOW, force_refresh=True)
    assert cache_path.read_bytes() == before

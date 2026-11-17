"""Provider-verified ArcGIS ObjectID recovery, fail-closed even under partial reads."""
import pytest

from modules.external.sources import maritime as m
from modules.external.http import FetchError


class Reply:
    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


def _arcgis_with_broken_offset(url, *, params, **kwargs):
    # Layer is healthy for ordinary reads but the ArcGIS table rejects
    # both orderByFields and resultOffset (observed 2026-11-17).
    if params.get("returnIdsOnly") == "true":
        return Reply({"objectIdFieldName": "ObjectId", "objectIds": [3, 1, 2]})
    if "objectIds" in params:
        ids = [int(x) for x in params["objectIds"].split(",")]
        return Reply({"features": [
            {"attributes": {"ObjectId": n, "date": "2026-11-16", "portid": f"port{n}"}}
            for n in reversed(ids)
        ]})
    if "resultOffset" in params or "orderByFields" in params:
        return Reply({"error": {"code": 400, "message": "Invalid query parameters."}})
    return Reply({"features": [{"attributes": {"ObjectId": 3}}]})


def test_broken_arcgis_offset_and_sort_can_recover_every_verified_row():
    m.reset_rate_limit_breaker()
    result = m.query_features(
        "https://services9.arcgis.com/example/FeatureServer",
        0, "date >= DATE '2026-11-01'", ["date", "portid"],
        _arcgis_with_broken_offset, order_by="date ASC",
        isolation_field="ObjectId", max_record_count=2,
    )
    assert len(result) == 3
    assert {row["attributes"]["ObjectId"] for row in result} == {1, 2, 3}


def test_objectid_batch_missing_one_row_fails_closed():
    m.reset_rate_limit_breaker()
    def fetch(url, *, params, **kwargs):
        if params.get("returnIdsOnly") == "true":
            return Reply({"objectIdFieldName": "ObjectId", "objectIds": [1, 2]})
        return Reply({"features": [{"attributes": {"ObjectId": 1}}]})

    with pytest.raises(FetchError, match="incomplete"):
        m.query_features_by_verified_ids(
            "https://services9.arcgis.com/example/FeatureServer",
            0, "1=1", "portid", fetch, "ObjectId", max_record_count=2,
        )


def test_objectid_enumeration_duplicates_and_excessive_counts_fail_closed(monkeypatch):
    m.reset_rate_limit_breaker()
    def fetch(url, *, params, **kwargs):
        return Reply({"objectIdFieldName": "ObjectId", "objectIds": [1, 1]})
    with pytest.raises(FetchError, match="duplicate"):
        m.query_features_by_verified_ids("https://services9.arcgis.com/example/FeatureServer",
                                         0, "1=1", "portid", fetch, "ObjectId")
    monkeypatch.setattr(m, "OBJECT_ID_FALLBACK_MAX_IDS", 1)
    def long_fetch(url, *, params, **kwargs):
        return Reply({"objectIdFieldName": "ObjectId", "objectIds": [1, 2]})
    with pytest.raises(FetchError, match="bounded limit"):
        m.query_features_by_verified_ids("https://services9.arcgis.com/example/FeatureServer",
                                         0, "1=1", "portid", long_fetch, "ObjectId")


def test_unverified_objectid_recovery_cannot_fabricate_complete_rows():
    m.reset_rate_limit_breaker()
    def fetch(url, *, params, **kwargs):
        if params.get("returnIdsOnly") == "true":
            return Reply({"error": {"code": 400, "message": "Invalid query parameters."}})
        return _arcgis_with_broken_offset(url, params=params, **kwargs)
    with pytest.raises(m.ArcGISQueryError, match="fallback failed closed"):
        m.query_features("https://services9.arcgis.com/example/FeatureServer",
                         0, "date >= DATE '2026-11-01'", ["date", "portid"], fetch,
                         order_by="date ASC", isolation_field="ObjectId",
                         max_record_count=2)


def test_no_objectid_fallback_when_filtered_where_is_rejected():
    m.reset_rate_limit_breaker()
    def fetch(url, *, params, **kwargs):
        if params.get("returnIdsOnly") == "true":
            pytest.fail("Must not fetch ID inventory when where clause is invalid")
        if "date" in params.get("where", ""):
            return Reply({"error": {"code": 400, "message": "Invalid where clause."}})
        return Reply({"features": [{"attributes": {"ObjectId": 1}}]})
    with pytest.raises(m.ArcGISQueryError, match="ArcGIS-Query-Fehler"):
        m.query_features("https://services9.arcgis.com/example/FeatureServer",
                         0, "date INVALID", ["date", "portid"], fetch,
                         order_by="date ASC", isolation_field="ObjectId",
                         max_record_count=2)


def test_objectid_enumeration_rate_limit_blocks_shared_host(monkeypatch):
    m.reset_rate_limit_breaker()
    monkeypatch.setattr(m, "RATE_LIMIT_RETRIES", 0)
    def limit_fetch(url, *, params, **kwargs):
        return Reply({"error": {"code": 429, "message": "Too many requests"}})

    with pytest.raises(FetchError, match="enumeration rejected"):
        m.query_features_by_verified_ids(
            "https://services9.arcgis.com/example/FeatureServer",
            0, "1=1", "portid", limit_fetch, "ObjectId",
        )
    with pytest.raises(m.PortWatchUpstreamUnavailable, match="UPSTREAM_RATE_LIMITED"):
        m.query_features_by_verified_ids(
            "https://services9.arcgis.com/example/FeatureServer",
            0, "1=1", "portid", limit_fetch, "ObjectId",
        )
    m.reset_rate_limit_breaker()


def test_objectid_batch_rate_limit_blocks_shared_host(monkeypatch):
    m.reset_rate_limit_breaker()
    monkeypatch.setattr(m, "RATE_LIMIT_RETRIES", 0)
    def limit_fetch(url, *, params, **kwargs):
        if params.get("returnIdsOnly") == "true":
            return Reply({"objectIdFieldName": "ObjectId", "objectIds": [1]})
        return Reply({"error": {"code": 429, "message": "Too many requests"}})

    with pytest.raises(FetchError, match="batch rejected"):
        m.query_features_by_verified_ids(
            "https://services9.arcgis.com/example/FeatureServer",
            0, "1=1", "portid", limit_fetch, "ObjectId",
        )
    with pytest.raises(m.PortWatchUpstreamUnavailable, match="UPSTREAM_RATE_LIMITED"):
        m.query_features_by_verified_ids(
            "https://services9.arcgis.com/example/FeatureServer",
            0, "1=1", "portid", limit_fetch, "ObjectId",
        )
    m.reset_rate_limit_breaker()

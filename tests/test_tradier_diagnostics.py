"""Regression tests for fail-closed Tradier provider taxonomy (no network/API)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import requests

from modules import options_designer as od
from modules import tradier_diagnostics as td


class Response:
    def __init__(self, status=200, text="", body=None):
        self.status_code = status
        self.text = text
        self.body = body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def json(self):
        return self.body


@pytest.mark.parametrize(
    "status,body,expected",
    [
        (400, "Quota Violation: 6001 of 6000 calls", "QUOTA_EXHAUSTED"),
        (400, "invalid symbol", "HTTP_ERROR"),
        (403, "rate limit exceeded", "QUOTA_EXHAUSTED"),
        (429, "", "QUOTA_EXHAUSTED"),
        (401, "not authorized", "AUTH_OR_PERMISSION"),
        (503, "provider outage", "PROVIDER_DOWN"),
    ],
)
def test_provider_status_classification(status, body, expected):
    response = Response(status, body)
    assert td.status_category(response) == expected
    with pytest.raises(td.ProviderDataError) as ex:
        td.check_response(response, "chains")
    assert ex.value.category == expected
    assert body not in str(ex.value)  # no raw provider contents logged


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"expirations": None}, []),
        ({"expirations": {"date": None}}, []),
        ({"expirations": {"date": "2027-01-15"}}, ["2027-01-15"]),
        ({"expirations": {"date": ["2027-06-18", "2027-01-15"]}},
         ["2027-01-15", "2027-06-18"]),
    ],
)
def test_valid_expirations(payload, expected):
    assert td.expirations(payload) == expected


@pytest.mark.parametrize("data", [None, [], {}, {"expirations": []},
                                   {"expirations": {"date": [None]}},
                                   {"options": {"option": []}}])
def test_bad_expiration_schema_cannot_be_no_contract(data):
    with pytest.raises(td.ProviderDataError, match="SCHEMA_CHANGED"):
        td.expirations(data)


def test_valid_chain_empty_and_single_contract():
    assert td.chain({"options": None}) == []
    assert td.chain({"options": {"option": None}}) == []
    quote = {"symbol": "AAPL270115C00200000", "bid": 4.2, "ask": 4.5}
    assert td.chain({"options": {"option": quote}}) == [quote]
    assert td.chain({"options": {"option": [quote]}}) == [quote]


@pytest.mark.parametrize("bad", [None, [], {}, {"options": []},
                                 {"options": {"option": "not-a-chain"}},
                                 {"options": {"option": [None]}}])
def test_bad_chain_schema_rejected(bad):
    with pytest.raises(td.ProviderDataError, match="SCHEMA_CHANGED"):
        td.chain(bad)


def test_http_200_embedded_quota_error_is_not_empty():
    with pytest.raises(td.ProviderDataError, match="QUOTA_EXHAUSTED"):
        td.expirations({"error": "Quota Violation: calls exceeded"})


def test_timeout_and_connection_failure_classified():
    assert td.error_category(requests.Timeout()) == "TIMEOUT"
    assert td.error_category(requests.ConnectionError()) == "PROVIDER_DOWN"


def test_designer_chain_provider_failure_fails_closed_and_records(monkeypatch):
    monkeypatch.setattr(od.requests, "get", lambda *args, **kwargs:
                        Response(400, "Quota Violation", body={"options": None}))
    before = td.counters().get("chains:QUOTA_EXHAUSTED", 0)
    assert od._tradier_chain("AAPL", "2027-01-15") == []
    assert td.counters()["chains:QUOTA_EXHAUSTED"] == before + 1


def test_designer_chain_valid_quotes_never_modified(monkeypatch):
    quote = {"option_type": "call", "strike": 200, "bid": 1, "ask": 1.2}
    monkeypatch.setattr(od.requests, "get", lambda *args, **kwargs:
                        Response(200, body={"options": {"option": [quote]}}))
    assert od._tradier_chain("AAPL", "2027-01-15") == [quote]


def test_designer_expiration_uses_existing_cache_only(monkeypatch):
    monkeypatch.setattr(od.request_cache, "cached",
                        lambda provider, name, params, loader: loader())
    monkeypatch.setattr(od.requests, "get", lambda *args, **kwargs:
                        Response(400, "Quota Violation", body={}))
    assert od._tradier_expirations("AAPL") == []
    monkeypatch.setattr(od.requests, "get", lambda *args, **kwargs:
                        Response(200, body={"expirations": {"date": "2027-01-15"}}))
    assert od._tradier_expirations("AAPL") == ["2027-01-15"]


def test_schema_and_no_contract_have_different_codes(monkeypatch):
    monkeypatch.setattr(od.requests, "get", lambda *args, **kwargs:
                        Response(200, body={"options": {"bad_key": []}}))
    before = td.counters().get("chains:SCHEMA_CHANGED", 0)
    assert od._tradier_chain("AAPL", "2027-01-15") == []
    assert td.counters()["chains:SCHEMA_CHANGED"] == before + 1
    monkeypatch.setattr(od.requests, "get", lambda *args, **kwargs:
                        Response(200, body={"options": {"option": []}}))
    before_empty = td.counters().get("chains:NO_CONTRACT", 0)
    assert od._tradier_chain("AAPL", "2027-01-15") == []
    assert td.counters()["chains:NO_CONTRACT"] == before_empty + 1


def test_report_is_serializable_classified_without_secrets():
    import json
    before = td.report()["total_provider_failures"]
    td.record("chains", "TIMEOUT")
    result = td.report()
    assert result["state"] == "DEGRADED"
    assert result["total_provider_failures"] >= before + 1
    assert result["additional_api_requests"] == 0
    assert "token" not in json.dumps(result).lower()
    assert "chains:TIMEOUT" in result["by_endpoint_category"]

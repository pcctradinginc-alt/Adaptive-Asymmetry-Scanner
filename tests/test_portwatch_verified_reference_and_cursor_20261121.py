"""PortWatch recovery never trusts a historical ID without a fresh official lookup."""
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from modules.external.sources import maritime as m


DAILY = {"service_url": "https://services9.arcgis.com/verified/ports_daily/FeatureServer",
         "layer_id": 0, "owner": "IMF-portwatch_imf_dataviz"}
REFERENCE = {"service_url": "https://services9.arcgis.com/verified/ports_database/FeatureServer",
             "layer_id": 0, "owner": "IMF-portwatch_imf_dataviz", "fallback_to_daily_layer": False}
SCHEMA = {"id_field": "portid", "name_field": "portname", "country_field": "country"}
WANTED = [{"name": "Houston", "country": "United States"},
          {"name": "Shenzhen", "country": "China"}]
ALIASES = {"Shenzhen": ["Shekou"]}


class Reply:
    def __init__(self, body):
        self.body = body

    def json(self):
        return self.body


def _preflight(path, endpoint=DAILY, status="PASS"):
    path.write_text(json.dumps([{
        "source_id": "imf_portwatch_ports",
        "status": status,
        "endpoints": [endpoint["service_url"] + "/0/query"],
        "discovered_ids": {"ports": {"Houston": "port481", "Shenzhen": "port1189"},
                           "unresolved_ports": {}},
    }]), encoding="utf-8")


def _recover(tmp_path, responder, *, reference=None, missing=None):
    path = tmp_path / "preflight.json"
    _preflight(path)
    calls = []

    def fetch(url, *, params, **kwargs):
        calls.append((url, dict(params)))
        return Reply(responder(params))

    result, diag = m.recover_verified_port_reference(
        daily_endpoint=DAILY, reference_endpoint=reference or REFERENCE,
        ref_schema=SCHEMA, missing_names=missing or {"Houston", "Shenzhen"},
        wanted=WANTED, alias_map=ALIASES, http_fetch=fetch,
        prior_path=str(path),
    )
    return result, diag, calls


def test_recovery_queries_exact_historical_ids_but_accepts_only_live_official_rows(tmp_path):
    def responder(params):
        assert params["where"] == "portid IN ('port1189','port481')"
        return {"features": [
            {"attributes": {"portid": "port481", "portname": "Houston",
                            "country": "United States"}},
            {"attributes": {"portid": "port1189", "portname": "Shekou",
                            "country": "China"}},
        ]}
    result, diag, calls = _recover(tmp_path, responder)
    assert {r["attributes"]["portid"] for r in result} == {"port481", "port1189"}
    assert diag["status"] == "LIVE_VERIFIED"
    assert len(calls) == 1 and calls[0][0].startswith(REFERENCE["service_url"])


def test_partial_reference_query_cannot_fake_complete_recovery(tmp_path):
    result, diag, calls = _recover(tmp_path, lambda p: {
        "features": [{"attributes": {"portid": "port481", "portname": "Houston",
                                      "country": "United States"}}],
    })
    assert len(result) == 1 and diag["status"] == "PARTIAL_LIVE_VERIFICATION"
    assert diag["rejected"] == ["Shenzhen"]
    assert len(calls) == 1


def test_changed_name_or_country_is_rejected_even_if_cached_id_matches(tmp_path):
    result, diag, _ = _recover(tmp_path, lambda p: {
        "features": [
            {"attributes": {"portid": "port481", "portname": "Houston", "country": "Canada"}},
            {"attributes": {"portid": "port1189", "portname": "Yantian", "country": "China"}},
        ]})
    assert result == [] and diag["status"] == "PARTIAL_LIVE_VERIFICATION"
    assert set(diag["rejected"]) == {"Houston", "Shenzhen"}


def test_wrong_publisher_and_unverified_provenance_do_not_query(tmp_path):
    def should_not_call(_params):
        pytest.fail("Historical reference must not be used against unknown publisher")
    result, diag, calls = _recover(
        tmp_path, should_not_call, reference={**REFERENCE, "owner": "another-publisher"})
    assert result == [] and calls == [] and diag["status"] == "PUBLISHER_NOT_VERIFIED"
    path = tmp_path / "wrong-preflight.json"
    _preflight(path, endpoint={"service_url": "https://other.example/FeatureServer"})
    result, diag = m.recover_verified_port_reference(
        daily_endpoint=DAILY, reference_endpoint=REFERENCE, ref_schema=SCHEMA,
        missing_names={"Houston"}, wanted=WANTED, alias_map=ALIASES,
        http_fetch=lambda *a, **kw: pytest.fail("Wrong endpoint provenance"),
        prior_path=str(path))
    assert result == [] and diag["status"] == "INVALID_OR_MISSING_HINTS"


def test_provider_error_does_not_become_verified_data(tmp_path):
    m.reset_rate_limit_breaker()
    try:
        result, diag, _ = _recover(tmp_path, lambda p: {
            "error": {"code": 400, "message": "Invalid query parameters."}
        })
        pytest.fail(f"Unexpected success: {result}, {diag}")
    except m.ArcGISQueryError:
        pass


def test_archived_cursor_reduces_overfetch_without_skipping_gaps(tmp_path):
    path = tmp_path / "health.json"
    now = datetime(2026, 11, 21, 12, tzinfo=timezone.utc)
    path.write_text(json.dumps({"imf_portwatch_ports": {
        "latest_observation": "2026-11-13T00:00:00+00:00",
        "last_success": "2026-11-13T12:00:00+00:00",
    }}), encoding="utf-8")
    assert m.archived_incremental_window(now, str(path)) == (
        datetime(2026, 11, 8).date(), datetime(2026, 11, 21).date())
    # A long upstream outage must be filled from its real archive cursor.
    state = json.loads(path.read_text())
    state["imf_portwatch_ports"]["latest_observation"] = "2026-10-01T00:00:00+00:00"
    path.write_text(json.dumps(state))
    assert m.archived_incremental_window(now, str(path))[0] == datetime(2026, 9, 26).date()


def test_cursor_missing_or_future_state_uses_conservative_window(tmp_path):
    now = datetime(2026, 11, 21, 12, tzinfo=timezone.utc)
    path = tmp_path / "health.json"
    baseline = m.incremental_window(now)
    assert m.archived_incremental_window(now, str(path)) == baseline
    path.write_text(json.dumps({"imf_portwatch_ports": {
        "latest_observation": "2026-11-22T00:00:00+00:00",
        "last_success": "2026-11-20T12:00:00+00:00",
    }}))
    assert m.archived_incremental_window(now, str(path)) == baseline

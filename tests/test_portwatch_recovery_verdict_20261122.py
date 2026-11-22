"""Strict recovery evidence must not confuse official publication lag with ingestion faults."""
import json
from datetime import datetime, timezone, timedelta

import pytest

from scripts import verify_portwatch_recovery as vr


NOW = datetime(2026, 11, 22, 9, 0, tzinfo=timezone.utc)


class Response:
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error

    def raise_for_status(self):
        if self.error:
            raise self.error

    def json(self):
        return self.payload


def evidence(tmp_path, port_day="2026-11-13", choke_day="2026-11-15"):
    summary = {"sources": {name: {
        "status": "PASS", "pit_errors": 0, "archive_error": "", "dq_issues": [],
    } for name in vr.REQUIRED}}
    health = {name: {
        "status": "PASS",
        "last_success": (NOW - timedelta(minutes=2)).isoformat(),
        "latest_observation": (port_day if name.endswith("_ports") else choke_day) + "T00:00:00+00:00",
        "message": "4578 observations; 0 unresolved ports: []" if name.endswith("_ports") else "5880 observations",
        "pit_integrity_failures": 0,
        "archive_error": "",
        "dq": {"severe": False},
    } for name in vr.REQUIRED}
    normalized = tmp_path / "normalized"
    normalized.mkdir()
    with (normalized / "2026-11.jsonl").open("w", encoding="utf-8") as out:
        for name, pid in vr.EXPECTED_PORTS.items():
            out.write(json.dumps({
                "entity_id": pid, "metric": "portcalls_total",
                "value": 12.0, "observation_time": port_day + "T00:00:00+00:00",
            }) + "\n")
    return summary, health, normalized


def test_verified_official_publication_lag_is_nonfatal_but_explicit(tmp_path):
    summary, health, normalized = evidence(tmp_path)
    requests = []

    def provider(url, params, timeout):
        requests.append((url, params))
        return Response({"objectIds": [], "objectIdFieldName": "ObjectId"})

    verdict, latest, archived = vr.assess(summary, health, normalized, NOW, get=provider)
    assert verdict == "PUBLISHER_LAG_VERIFIED"
    assert latest["imf_portwatch_ports"].isoformat() == "2026-11-13"
    assert len(archived) == len(vr.EXPECTED_PORTS)
    assert requests == [(vr.OFFICIAL_DAILY_PORTS_QUERY, {
        "where": "date > DATE '2026-11-13'",
        "returnIdsOnly": "true", "f": "json",
    })]


def test_current_archive_needs_no_publisher_query(tmp_path):
    summary, health, normalized = evidence(
        tmp_path, port_day="2026-11-17", choke_day="2026-11-17"
    )
    verdict, _, _ = vr.assess(
        summary, health, normalized, NOW,
        get=lambda *_args, **_kwargs: pytest.fail("unexpected provider call"),
    )
    assert verdict == "CURRENT_ARCHIVED"


def test_publisher_has_newer_rows_requires_failure(tmp_path):
    summary, health, normalized = evidence(tmp_path)
    with pytest.raises(vr.VerificationError, match="publisher HAS newer"):
        vr.assess(
            summary, health, normalized, NOW,
            get=lambda *_args, **_kwargs: Response({"objectIds": [8831, 8832]}),
        )


@pytest.mark.parametrize("payload", [
    {"error": {"code": 429, "message": "rate limited"}},
    {"objectIds": None},
    {},
    {"objectIds": [1, 1]},
    {"objectIds": ["not an integer"]},
])
def test_incomplete_or_faulty_official_provider_response_fails(tmp_path, payload):
    summary, health, normalized = evidence(tmp_path)
    with pytest.raises(vr.VerificationError, match="official PortWatch"):
        vr.assess(
            summary, health, normalized, NOW,
            get=lambda *_args, **_kwargs: Response(payload),
        )


def test_unavailable_official_provider_fails_instead_of_guessing(tmp_path):
    import requests
    summary, health, normalized = evidence(tmp_path)
    with pytest.raises(vr.VerificationError, match="probe unavailable"):
        vr.assess(
            summary, health, normalized, NOW,
            get=lambda *_args, **_kwargs: Response(
                error=requests.exceptions.Timeout("rate limited")
            ),
        )


@pytest.mark.parametrize("bad", ["WARN", "FAIL", "SCHEMA_CHANGED"])
def test_ingestion_not_pass_fails(tmp_path, bad):
    summary, health, normalized = evidence(tmp_path)
    summary["sources"]["imf_portwatch_ports"]["status"] = bad
    with pytest.raises(vr.VerificationError, match="not PASS"):
        vr.assess(summary, health, normalized, NOW)


def test_missing_one_curated_numeric_series_fails(tmp_path):
    summary, health, normalized = evidence(tmp_path)
    file = normalized / "2026-11.jsonl"
    rows = [json.loads(x) for x in file.read_text().splitlines()]
    rows = [x for x in rows if x["entity_id"] != "port481"]
    file.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    with pytest.raises(vr.VerificationError, match="Houston"):
        vr.assess(summary, health, normalized, NOW)


def test_null_valued_portcalls_are_not_coverage(tmp_path):
    summary, health, normalized = evidence(tmp_path)
    file = normalized / "2026-11.jsonl"
    rows = [json.loads(x) for x in file.read_text().splitlines()]
    for item in rows:
        if item["entity_id"] == "port1069":
            item["value"] = None
    file.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    with pytest.raises(vr.VerificationError, match="Qingdao"):
        vr.assess(summary, health, normalized, NOW)


def test_old_curated_archive_cannot_hide_behind_recent_source_status(tmp_path):
    summary, health, normalized = evidence(tmp_path)
    file = normalized / "2026-11.jsonl"
    rows = [json.loads(x) for x in file.read_text().splitlines()]
    for item in rows:
        if item["entity_id"] == "port1189":
            item["observation_time"] = "2026-11-02T00:00:00+00:00"
    file.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    with pytest.raises(vr.VerificationError, match="Shenzhen"):
        vr.assess(summary, health, normalized, NOW)


def test_unresolved_curated_name_fails(tmp_path):
    summary, health, normalized = evidence(tmp_path)
    health["imf_portwatch_ports"]["message"] = "4 unresolved ports"
    with pytest.raises(vr.VerificationError, match="identities"):
        vr.assess(summary, health, normalized, NOW)


def test_outdated_retrieval_timestamp_fails(tmp_path):
    summary, health, normalized = evidence(tmp_path)
    health["imf_portwatch_ports"]["last_success"] = (NOW - timedelta(days=1)).isoformat()
    with pytest.raises(vr.VerificationError, match="successful archive write"):
        vr.assess(summary, health, normalized, NOW)


def test_chokepoints_stale_cannot_be_excused_by_ports_publisher(tmp_path):
    summary, health, normalized = evidence(tmp_path, choke_day="2026-11-10")
    with pytest.raises(vr.VerificationError, match="chokepoints too old"):
        vr.assess(summary, health, normalized, NOW)


def test_pit_or_archival_errors_fail(tmp_path):
    summary, health, normalized = evidence(tmp_path)
    summary["sources"]["imf_portwatch_ports"]["pit_errors"] = 1
    with pytest.raises(vr.VerificationError, match="point-in-time"):
        vr.assess(summary, health, normalized, NOW)

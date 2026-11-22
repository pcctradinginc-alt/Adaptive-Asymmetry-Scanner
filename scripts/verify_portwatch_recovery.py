"""Verify PortWatch ingestion separately from the official publisher's publication lag.

An ingestion error, missing/incorrect curated harbour, unarchived row or unverified
provider response FAILS. A lagged *official* Daily Ports publication is a WARNING
only if the live publisher confirms there are zero newer official daily rows.
No stale observations are re-labelled as fresh; this check never changes health
or trading data and does not promote any signals.
"""
from __future__ import annotations

import json
import math
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import requests


OFFICIAL_DAILY_PORTS_QUERY = (
    "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/"
    "Daily_Ports_Data/FeatureServer/0/query"
)
EXPECTED_PORTS = {
    "Houston": "port481",
    "Los Angeles": "port664",
    "Qingdao": "port1069",
    "Shenzhen (Shekou)": "port1189",
}
REQUIRED = ("imf_portwatch_chokepoints", "imf_portwatch_ports")
MAX_OBSERVATION_LAG_DAYS = 8
MAX_SUCCESS_AGE_MINUTES = 40


class VerificationError(RuntimeError):
    """No supported basis for claiming that the data import works."""


def _timestamp(value: object, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise VerificationError(f"{label}: timestamp missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise VerificationError(f"{label}: invalid timestamp") from exc
    if parsed.tzinfo is None:
        raise VerificationError(f"{label}: timezone missing")
    return parsed.astimezone(timezone.utc)


def _archived_port_dates(directory: Path) -> dict[str, date]:
    """Read REAL numeric portcalls from the recent normalized archive partitions."""
    if not directory.is_dir():
        raise VerificationError(f"normalized ports directory absent: {directory}")
    latest: dict[str, date] = {}
    files = sorted(directory.glob("????-??.jsonl"))[-2:]
    if not files:
        raise VerificationError("normalized ports archive contains no recent partitions")
    for filename in files:
        with filename.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise VerificationError(f"malformed archive row in {filename}") from exc
                if not isinstance(row, dict):
                    raise VerificationError(f"non-object archive row in {filename}")
                pid = row.get("entity_id")
                if pid not in EXPECTED_PORTS.values() or row.get("metric") != "portcalls_total":
                    continue
                value = row.get("value")
                if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                    continue
                observed = _timestamp(row.get("observation_time"), f"{pid} observation").date()
                if pid not in latest or observed > latest[pid]:
                    latest[pid] = observed
    return latest


def verify_local_ingest(
    summary: dict, health: dict, archive_dir: Path, now: datetime
) -> tuple[dict[str, date], dict[str, date]]:
    """Fail-closed on missing/failing source, archival and PIT evidence."""
    if not isinstance(summary, dict) or not isinstance(health, dict):
        raise VerificationError("invalid ingestion summary or source health")
    source_summaries = summary.get("sources")
    if not isinstance(source_summaries, dict):
        raise VerificationError("ingestion summary has no sources")
    now = now.astimezone(timezone.utc)
    latest_source: dict[str, date] = {}
    for name in REQUIRED:
        result = source_summaries.get(name)
        record = health.get(name)
        if not isinstance(result, dict) or not isinstance(record, dict):
            raise VerificationError(f"{name}: missing run or archive health evidence")
        if result.get("status") != "PASS" or record.get("status") != "PASS" or result.get("skipped"):
            raise VerificationError(f"{name}: ingest/archival status is not PASS")
        if (result.get("archive_error") or result.get("pit_errors") or
                result.get("dq_issues") or record.get("archive_error") or
                record.get("pit_integrity_failures") or
                (record.get("dq") or {}).get("severe")):
            raise VerificationError(f"{name}: archive, point-in-time or DQ errors")
        success = _timestamp(record.get("last_success"), f"{name} last_success")
        age = now - success
        if not (-timedelta(minutes=5) <= age <= timedelta(minutes=MAX_SUCCESS_AGE_MINUTES)):
            raise VerificationError(f"{name}: no successful archive write in this run")
        observed = _timestamp(record.get("latest_observation"), f"{name} latest_observation").date()
        if observed > now.date():
            raise VerificationError(f"{name}: future-dated source observation")
        latest_source[name] = observed
    port_message = health["imf_portwatch_ports"].get("message") or ""
    if "0 unresolved ports" not in port_message:
        raise VerificationError("curated port identities were not completely resolved")

    latest_port = latest_source["imf_portwatch_ports"]
    archived = _archived_port_dates(archive_dir)
    for display_name, pid in EXPECTED_PORTS.items():
        if pid not in archived:
            raise VerificationError(f"{display_name} ({pid}): no numeric archived portcalls")
        if archived[pid] != latest_port:
            raise VerificationError(
                f"{display_name} ({pid}): latest archived numeric portcalls "
                f"{archived[pid]} != official ingested latest {latest_port}"
            )
    return latest_source, archived


def publisher_has_newer_rows(
    last_observation: date, get: Callable = requests.get,
) -> bool:
    """Bounded, read-only LIVE publisher query; an unavailable publisher is NOT proof."""
    params = {
        "where": f"date > DATE '{last_observation.isoformat()}'",
        "returnIdsOnly": "true",
        "f": "json",
    }
    try:
        response = get(OFFICIAL_DAILY_PORTS_QUERY, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError, AttributeError) as exc:
        raise VerificationError(f"official PortWatch freshness probe unavailable: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("error"):
        raise VerificationError("official PortWatch freshness probe returned an ArcGIS error")
    ids = payload.get("objectIds")
    if not isinstance(ids, list) or len(ids) > 100_000 or len(set(map(str, ids))) != len(ids):
        raise VerificationError("official PortWatch freshness probe returned invalid ID inventory")
    if any(not isinstance(x, int) or isinstance(x, bool) for x in ids):
        raise VerificationError("official PortWatch freshness probe returned nonnumeric IDs")
    return bool(ids)


def assess(
    summary: dict,
    health: dict,
    archive_dir: Path,
    now: datetime,
    get: Callable = requests.get,
) -> tuple[str, dict[str, date], dict[str, date]]:
    latest, archived = verify_local_ingest(summary, health, archive_dir, now)
    today = now.astimezone(timezone.utc).date()
    chokepoint_day = latest["imf_portwatch_chokepoints"]
    if (today - chokepoint_day).days > MAX_OBSERVATION_LAG_DAYS:
        raise VerificationError(
            f"chokepoints too old ({chokepoint_day}) without independent publisher verification"
        )
    last_port_day = latest["imf_portwatch_ports"]
    if (today - last_port_day).days <= MAX_OBSERVATION_LAG_DAYS:
        return "CURRENT_ARCHIVED", latest, archived
    if publisher_has_newer_rows(last_port_day, get=get):
        raise VerificationError(
            f"official publisher HAS newer Daily Ports rows after {last_port_day}; "
            "import missed them, investigate connector"
        )
    return "PUBLISHER_LAG_VERIFIED", latest, archived


def main() -> int:
    try:
        summary = json.loads(Path("/tmp/portwatch_recovery_summary.json").read_text(encoding="utf-8"))
        health = json.loads(Path("outputs/external_data/health/source_health.json").read_text(encoding="utf-8"))
        verdict, latest, archived = assess(
            summary, health,
            Path("outputs/external_data/normalized/imf_portwatch_ports"),
            datetime.now(timezone.utc),
        )
    except (VerificationError, OSError, ValueError, TypeError) as exc:
        message = f"PortWatch recovery not verified: {exc}"
        print(f"::error::{message}")
        summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary_path:
            with open(summary_path, "a", encoding="utf-8") as output:
                output.write(f"## PortWatch recovery: FAILED\n\n{message}\n")
        return 1

    lines = [
        "## PortWatch ingestion / publication verdict",
        "",
        f"**Verdict: {verdict}**",
        "",
        f"- Ports ingested/latest official observed date: {latest['imf_portwatch_ports']}",
        f"- Chokepoints ingested/latest observed date: {latest['imf_portwatch_chokepoints']}",
        "- Curated identities: all resolved; PIT, DQ and archival gates passed",
    ]
    for name, pid in EXPECTED_PORTS.items():
        lines.append(f"- {name} ({pid}): archived numeric portcalls through {archived[pid]}")
    if verdict == "PUBLISHER_LAG_VERIFIED":
        lines.extend([
            "",
            "**Publisher delay:** live official Daily_Ports_Data confirms **zero newer rows**.",
            "The archive is NOT current, and this result does NOT certify fresh signals or alpha.",
            "The daily recovery continues; any provider/API, PIT, DQ, ID or archive failure remains fatal.",
        ])
        print(
            f"::warning::PortWatch PUBLISHER_LAG_VERIFIED: official Daily Ports has "
            f"no observations after {latest['imf_portwatch_ports']}; ingestion PASS, data not current"
        )
    else:
        lines.append("\nPublisher observations and curated archived portcall series are current.")
        print("PortWatch ingestion and provider freshness both verified")
    output_text = "\n".join(lines) + "\n"
    print(output_text)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as output:
            output.write(output_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

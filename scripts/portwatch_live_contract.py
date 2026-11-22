"""Bounded, read-only LIVE ArcGIS contract check (not an ingest or alpha test)."""
from __future__ import annotations

from datetime import date, timedelta
import requests

from modules.external.sources import maritime as mw

ROOT = "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services"
SERVICES = ("Daily_Chokepoints_Data", "Daily_Ports_Data")


def query(url: str, params: dict) -> dict:
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict) or "error" in data:
        raise RuntimeError(f"ArcGIS contract request failed: {str(data)[:250]}")
    return data


def verify_curated_port_identities() -> None:
    """Live verify the official four identities, not merely historical IDs."""
    universe = mw.load_port_universe()
    hints = universe["verified_port_id_hints"]
    wanted = [p for p in mw.flatten_port_universe(universe) if p["name"] in hints]
    url = f"{ROOT}/PortWatch_ports_database/FeatureServer/0/query"
    result = query(url, {
        "where": "portid IN (" + ",".join(f"'{pid}'" for pid in sorted(hints.values())) + ")",
        "outFields": "portid,portname,country",
        "returnGeometry": "false",
        "resultRecordCount": 16,
        "f": "json",
    })
    features = result.get("features")
    if not isinstance(features, list) or result.get("exceededTransferLimit"):
        raise RuntimeError("Official PortWatch reference response incomplete")
    got = mw.resolve_entities_by_name(
        wanted, features, "portid", "portname", "country",
        alias_map=mw.load_port_aliases(universe),
        verified_ids=hints,
    )
    failures = [f"{r.name}: {r.status}" for r in got
                if r.status != "resolved" or r.entity_id != hints[r.name]]
    if failures:
        observed = [(f.get("attributes") or {}).get("portid") for f in features]
        raise RuntimeError(f"Four curated PortWatch IDs not all verified live: {failures}; "
                           f"reference_ids={observed[:16]}")
    for row in got:
        print(f"::notice::Verified curated official port: {row.name} = {row.entity_id}")
    print("Four curated PortWatch IDs verified against the LIVE official reference")


def check_official_ports_after_archive_date() -> None:
    """Read-only provider count beyond the actual archived Ports cursor.

    If the publisher has newer rows, our ingestion is missing data.
    If it has none, the source is publishing late; retries cannot invent
    fresh official observations.
    """
    import json
    from pathlib import Path

    health_path = Path("outputs/external_data/health/source_health.json")
    health = json.loads(health_path.read_text(encoding="utf-8"))
    last_observed = health["imf_portwatch_ports"]["latest_observation"][:10]
    url = f"{ROOT}/Daily_Ports_Data/FeatureServer/0/query"
    result = query(url, {
        "where": f"date > DATE '{last_observed}'",
        "returnIdsOnly": "true",
        "f": "json",
    })
    newer = result.get("objectIds")
    if not isinstance(newer, list):
        raise RuntimeError("Official PortWatch newer-date ID inventory unavailable")
    if len(newer) > 100_000 or len(newer) != len(set(newer)):
        raise RuntimeError("Official PortWatch newer-date ID inventory invalid")
    print(f"::notice::LIVE PortWatch Daily_Ports_Data rows after archived "
          f"{last_observed}: {len(newer)}")
    if newer:
        samples = query(url, {
            "where": "1=1", "objectIds": ",".join(map(str, newer[:3])),
            "outFields": "date,portid,ObjectId", "returnGeometry": "false", "f": "json",
        })
        print(f"::warning::Official publisher has newer port rows than archive: "
              f"{str(samples.get('features', []))[:400]}")
    else:
        print(f"::warning::Official PortWatch publisher has NO daily port rows "
              f"after {last_observed}; ingestion cannot update observations yet")


def main() -> None:
    # A bounded 21d population, never an unrestricted historical download.
    start = date.today() - timedelta(days=21)
    where = f"date >= DATE '{start.isoformat()}'"
    for name in SERVICES:
        url = f"{ROOT}/{name}/FeatureServer/0/query"
        ids_result = query(url, {
            "where": where, "returnIdsOnly": "true", "f": "json",
        })
        field = ids_result.get("objectIdFieldName")
        ids = ids_result.get("objectIds")
        if field not in (None, "ObjectId") or not isinstance(ids, list):
            raise RuntimeError(f"{name}: no verified ObjectID inventory")
        if not 0 < len(ids) <= 100_000 or len(set(ids)) != len(ids):
            raise RuntimeError(f"{name}: empty, duplicate or excessive ID inventory")
        sample = ids[:2]
        observed = query(url, {
            "where": "1=1",
            "objectIds": ",".join(map(str, sample)),
            "outFields": "ObjectId,date,portid",
            "returnGeometry": "false",
            "f": "json",
        })
        features = observed.get("features")
        if not isinstance(features, list) or observed.get("exceededTransferLimit"):
            raise RuntimeError(f"{name}: objectIds response truncated or missing")
        result_ids = [r.get("attributes", {}).get("ObjectId") for r in features]
        if len(result_ids) != len(sample) or set(result_ids) != set(sample):
            raise RuntimeError(f"{name}: provider ID reconciliation failed")
        # Exercise the production read function against one *observed* provider
        # date. The date comes from a live ArcGIS feature, never a guessed day.
        observed_day = features[0]["attributes"].get("date")
        if not isinstance(observed_day, str):
            raise RuntimeError(f"{name}: unrecognized provider DateOnly representation: {observed_day!r}")
        day = date.fromisoformat(observed_day)
        query_where = mw.build_where_clause(day, day, date_field="date",
                                            date_field_type="esriFieldTypeDateOnly")
        def live_fetch(url, *, params, timeout=30, **kwargs):
            return requests.get(url, params=params, timeout=timeout)
        mw.reset_rate_limit_breaker()
        rows = mw.query_features(
            f"{ROOT}/{name}/FeatureServer", 0, query_where,
            ["date", "portid", "ObjectId"], live_fetch,
            order_by="date ASC", isolation_field="ObjectId",
            max_record_count=1000 if name == "Daily_Ports_Data" else 10000,
            supports_pagination=True, objectid_recovery=True,
        )
        if not rows:
            raise RuntimeError(f"{name}: production query returned no rows for observed day")
        print(f"::notice::{name}: LIVE provider ObjectID contract passed; "
              f"population={len(ids)}; sample={len(sample)}; "
              f"production_query_rows={len(rows)}; observed_day={day}")
    verify_curated_port_identities()
    check_official_ports_after_archive_date()
    print("LIVE ArcGIS ObjectID contract passed for BOTH important PortWatch datasets")


if __name__ == "__main__":
    main()

"""Bounded, read-only LIVE ArcGIS contract check (not an ingest or alpha test)."""
from __future__ import annotations

from datetime import date, timedelta
import re
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
        if not isinstance(observed_day, str) or not re.fullmatch(r"\\d{4}-\\d{2}-\\d{2}", observed_day):
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
            supports_pagination=True,
        )
        if not rows:
            raise RuntimeError(f"{name}: production query returned no rows for observed day")
        print(f"::notice::{name}: LIVE provider ObjectID contract passed; "
              f"population={len(ids)}; sample={len(sample)}; "
              f"production_query_rows={len(rows)}; observed_day={day}")
    print("LIVE ArcGIS ObjectID contract passed for BOTH important PortWatch datasets")


if __name__ == "__main__":
    main()

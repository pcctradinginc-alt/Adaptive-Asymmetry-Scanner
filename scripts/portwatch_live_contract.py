"""Bounded, read-only LIVE ArcGIS contract check (not an ingest or alpha test)."""
from __future__ import annotations

from datetime import date, timedelta
import requests

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
        print(f"::notice::{name}: LIVE provider ObjectID contract passed; "
              f"population={len(ids)}; sample={len(sample)}")
    print("LIVE ArcGIS ObjectID contract passed for BOTH important PortWatch datasets")


if __name__ == "__main__":
    main()

"""Read-only NCEI climate-normal missingness breakdown (station/metric).

Reads the existing PIT archive; does not fetch, impute, alter ingestion, model
features, risk rules or normalize missing values into invented observations.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import UTC, date, datetime, time
from pathlib import Path

from modules.external.archive import ExternalArchive

SOURCE = "ncei_normals"
OUTPUT = Path("outputs/research/data_quality")


def _valid(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def diagnose(rows, as_of: date) -> dict:
    """Compare missingness across actual archived entity_id and metric values."""
    if isinstance(as_of, str):
        as_of = date.fromisoformat(as_of)
    by_station = defaultdict(lambda: [0, 0])
    by_metric = defaultdict(lambda: [0, 0])
    by_pair = defaultdict(lambda: [0, 0])
    valid = 0
    total = 0
    for r in rows:
        if r.source_id != SOURCE:
            continue
        total += 1
        station = str(r.entity_id)
        metric = str(r.metric)
        ok = _valid(r.value)
        valid += int(ok)
        for group, key in ((by_station, station), (by_metric, metric), (by_pair, (station, metric))):
            group[key][0] += 1
            group[key][1] += int(not ok)

    def describe(group, limit=30):
        records = [
            {"key": str(k), "total": v[0], "missing": v[1],
             "missing_fraction": round(v[1] / v[0], 4)}
            for k, v in group.items() if v[0]
        ]
        records.sort(key=lambda x: (-x["missing"], -x["total"], x["key"]))
        return records[:limit]

    return {
        "schema": "ncei-missingness-audit-v1",
        "as_of": as_of.isoformat(),
        "source_id": SOURCE,
        "status": "MEASURED" if total else "NO_ARCHIVED_OBSERVATIONS",
        "observations": total,
        "numeric": valid,
        "missing": total - valid,
        "missing_fraction": round((total - valid) / total, 4) if total else None,
        "unique_stations": len(by_station),
        "unique_metrics": len(by_metric),
        "stations_by_missing_count": describe(by_station),
        "metrics_by_missing_count": describe(by_metric),
        "station_metrics_by_missing_count": describe(by_pair, 50),
        "note": "Missingness uses PIT-visible archive observations; no values are imputed. "
                "The global health store may count a different denominator, e.g. historical vintages.",
    }


def run(as_of: date, archive: ExternalArchive | None = None, out: Path = OUTPUT) -> tuple[Path, dict]:
    if isinstance(as_of, str):
        as_of = date.fromisoformat(as_of)
    archive = archive or ExternalArchive()
    end = datetime.combine(as_of, time.max, UTC)
    report = diagnose(archive.as_of(SOURCE, end), as_of)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"ncei_missingness_{as_of.isoformat()}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return path, report


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--as-of", type=date.fromisoformat, required=True)
    p.add_argument("--archive", type=Path, default=ExternalArchive().root)
    p.add_argument("--output", type=Path, default=OUTPUT)
    args = p.parse_args(argv)
    path, report = run(args.as_of, archive=ExternalArchive(args.archive), out=args.output)
    print(f"NCEI audit: {path} {report['status']} missing={report['missing_fraction']}")
    return 0 if report["status"] == "MEASURED" else 2


if __name__ == "__main__":
    raise SystemExit(main())

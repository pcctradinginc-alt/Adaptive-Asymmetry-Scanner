"""Audit eight IMF feeds; compute bounded point-in-time feature *candidates*.

No portfolio trade, model fit, score promotion or alpha claim. Candidate features
are snapshots of verified normalized series at first-observed retrieval. This
does not replace a future embargoed Walk-Forward vs base-model comparison.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from modules.external.archive import ExternalArchive
from modules.external.pit import ensure_utc, utc_now
from modules.external.registry import SourceRegistry

IMF_IDS = ("imf_fsi", "imf_weo", "imf_mfs", "imf_gdd",
           "imf_fiscal_monitor", "imf_climate", "imf_fas", "imf_aipi")
MAX_CANDIDATES = 100
MAX_SOURCE_OBS = 40_000


def summarize(now: datetime | None = None, *, registry=None, archive=None) -> dict:
    now = ensure_utc(now) or utc_now()
    registry = registry or SourceRegistry()
    archive = archive or ExternalArchive(registry.archive_root)
    health = registry.load_health()
    result = {"generated_at": now.isoformat(), "family": "imf_macro",
              "mode": "RESEARCH_ONLY", "model_input_enabled": False,
              "safe_mode_override": False, "production_promotion_allowed": False,
              "alpha_validation": "NOT_EVALUATED",
              "reason": "PIT snapshots and DQ are NOT proof of predictive alpha",
              "sources": {}, "candidate_features": []}
    for sid in IMF_IDS:
        cfg = registry.sources.get(sid)
        if not cfg:
            result["sources"][sid] = {"ready": False, "status": "NOT_REGISTERED"}
            continue
        h = health.get(sid) or {}
        try:
            observations = archive.as_of(sid, now)
        except (OSError, ValueError) as e:
            result["sources"][sid] = {"ready": False, "status": "ARCHIVE_ERROR",
                                      "error": type(e).__name__}
            continue
        valid = [o for o in observations if o.value is not None and
                 isinstance(o.value, (float, int)) and math.isfinite(o.value)
                 and o.available_at <= now and o.retrieved_at <= now]
        statuses = (h.get("status"), h.get("staleness"))
        countries = sorted({x.entity_id for x in valid})
        ready = (h.get("status") in ("PASS", "WARN")
                 and h.get("staleness") != "STALE"
                 and not (h.get("dq") or {}).get("severe")
                 and h.get("pit_integrity_failures", 0) == 0
                 and len(valid) >= 2 and len(countries) >= 2
                 and len(observations) <= MAX_SOURCE_OBS)
        by_series = defaultdict(list)
        for o in valid:
            by_series[(o.metric, o.entity_id, o.series_id)].append(o)
        candidates = []
        if ready:
            for (metric, country, series), rows in sorted(by_series.items()):
                rows.sort(key=lambda o: o.observation_time)
                last = rows[-1]
                if last.value is None:
                    continue
                feature = {"source_id": sid, "metric": metric, "country": country,
                           "series_id": series, "as_of": now.isoformat(),
                           "observation_time": last.observation_time.isoformat(),
                           "available_at": last.available_at.isoformat(),
                           "value": last.value, "unit": last.unit,
                           "not_historical_pit_backfill": True}
                if len(rows) >= 2 and rows[-2].observation_time < last.observation_time:
                    feature["previous_period_delta"] = last.value - rows[-2].value
                    feature["previous_period"] = rows[-2].observation_time.isoformat()
                if len(candidates) < MAX_CANDIDATES:
                    candidates.append(feature)
        result["sources"][sid] = {
            "status": h.get("status", "NOT_COLLECTED"),
            "staleness": h.get("staleness", "UNKNOWN"),
            "n_observations_as_of": len(observations),
            "n_valid_observations": len(valid),
            "n_countries": len(countries),
            "candidate_features": len(candidates),
            "latest_observation": max((o.observation_time.isoformat() for o in valid), default=None),
            "last_success": h.get("last_success"),
            "ready": ready, "research_only": True,
        }
        result["candidate_features"].extend(candidates)
    result["ready_count"] = sum(bool(v.get("ready")) for v in result["sources"].values())
    result["all_sources_ready"] = result["ready_count"] == len(IMF_IDS)
    # Capturing historical IMF rows TODAY does not permit 2019 backtests.
    result["historical_walk_forward_eligible"] = False
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/research/imf_macro/status.json")
    parser.add_argument("--assert-ready", action="store_true")
    args = parser.parse_args(argv)
    report = summarize()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    print(json.dumps({"sources_ready": report["ready_count"],
                      "sources_expected": len(IMF_IDS),
                      "all_sources_ready": report["all_sources_ready"],
                      "alpha_validation": report["alpha_validation"]}))
    if args.assert_ready and not report["all_sources_ready"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

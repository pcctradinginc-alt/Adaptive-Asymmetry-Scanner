"""Audit the enabled IMF macro feeds; compute bounded point-in-time feature *candidates*.

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
from modules.external.pit import available_as_of, ensure_utc, utc_now
from modules.external.registry import SourceRegistry

FAMILY = "imf_macro"
MAX_CANDIDATES = 100
MAX_SOURCE_OBS = 40_000


def strict_backtest_eligible(o) -> bool:
    attrs = o.attrs or {}
    if attrs.get("edition", "current") == "current":
        return True
    return (attrs.get("strict_backtest_eligible") is True
            and attrs.get("availability_basis") == "IMF_LAST_UPDATED_AT"
            and o.available_at <= o.retrieved_at)


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
    family = sorted(sid for sid, cfg in registry.sources.items() if cfg.get("family") == FAMILY)
    # Only enabled feeds are expected; a disabled feed (e.g. no admissible
    # machine route) is reported, never silently counted as ready.
    expected = [sid for sid in family if registry.sources[sid].get("enabled")]
    result["expected_sources"] = expected
    for sid in family:
        cfg = registry.sources[sid]
        if sid not in expected:
            result["sources"][sid] = {"ready": False, "status": "DISABLED",
                                      "reason": str(cfg.get("notes", ""))[:300]}
            continue
        h = health.get(sid) or {}
        try:
            loaded = archive.load(sid)
            # Strict backtest gate: a historical IMF vintage counts only with an
            # availability date backed by official registry metadata. Legacy
            # rows (first ingest 2026-11-05, edition-month heuristic) and any
            # row without a recorded availability basis are excluded.
            history = [o for o in loaded if strict_backtest_eligible(o)]
            excluded = len(loaded) - len(history)
            observations = available_as_of(history, now)
        except (OSError, ValueError) as e:
            result["sources"][sid] = {"ready": False, "status": "ARCHIVE_ERROR",
                                      "error": type(e).__name__}
            continue
        valid = [o for o in observations if o.value is not None and
                 isinstance(o.value, (float, int)) and math.isfinite(o.value)
                 and o.available_at <= now and o.retrieved_at <= now
                 and getattr(o, "quality_status", "") == "OK"]
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
        # All archived versions (official IMF vintages + prospective weekly
        # snapshots) that were already available at `now`, per identity.
        # Keyed by target period, not identity: a projection's identity carries
        # its issue time, so editions of the same forecast differ only there.
        def target(o):
            return (o.series_id, o.entity_id, o.metric, o.observation_time)
        versions = defaultdict(list)
        for o in history:
            if o.available_at <= now and o.value is not None and getattr(o, "quality_status", "") == "OK":
                versions[target(o)].append(o)
        candidates = []
        if ready:
            for (metric, country, series), rows in sorted(by_series.items()):
                rows.sort(key=lambda o: (o.observation_time, o.available_at))
                last = rows[-1]
                feature = {"source_id": sid, "metric": metric, "country": country,
                           "series_id": series, "as_of": now.isoformat(),
                           "observation_time": last.observation_time.isoformat(),
                           "available_at": last.available_at.isoformat(),
                           "value": last.value, "unit": last.unit,
                           "not_historical_pit_backfill": True}
                prev_period = [o for o in rows if o.observation_time < last.observation_time]
                if prev_period:
                    p = max(prev_period, key=lambda o: (o.observation_time, o.available_at))
                    feature["previous_period_delta"] = last.value - p.value
                    feature["previous_period"] = p.observation_time.isoformat()
                # A forecast revision compares the *same* forecast target year
                # across distinct first-seen vintages; NEVER call it a period
                # change. Only later prospective retrievals can establish one.
                older_version = [o for o in versions.get(target(last), [])
                                 if o.available_at < last.available_at]
                if older_version:
                    prior = max(older_version, key=lambda o: o.available_at)
                    key = ("as_observed_forecast_revision" if last.attrs.get("is_projection")
                           else "as_observed_revision")
                    feature[key] = last.value - prior.value
                    feature["previous_vintage_available_at"] = prior.available_at.isoformat()
                    feature["previous_vintage"] = prior.attrs.get("edition")
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
            "excluded_unverified_vintage_rows": excluded,
            "verified_historical_vintage_rows": sum(
                1 for o in history if (o.attrs or {}).get("availability_basis") == "IMF_LAST_UPDATED_AT"),
        }
        result["candidate_features"].extend(candidates)
    result["ready_count"] = sum(bool(result["sources"][s].get("ready")) for s in expected)
    result["all_sources_ready"] = bool(expected) and result["ready_count"] == len(expected)
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
                      "sources_expected": len(report["expected_sources"]),
                      "all_sources_ready": report["all_sources_ready"],
                      "alpha_validation": report["alpha_validation"]}))
    if args.assert_ready and not report["all_sources_ready"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

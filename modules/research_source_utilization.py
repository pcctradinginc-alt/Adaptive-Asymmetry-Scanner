"""Read-only inventory of archived-but-not-evaluated research data sources.

This report is a cost/coverage and integration audit, NOT a new alpha test.
The explicit reader status comes from the reviewed code-linkage audit in
docs/DATA_LINK_AUDIT_2026-11-08.md. It is NOT inferred from PASS ingestion
health, and must be reviewed whenever a reader is added. No trading gates,
new features, source downloads, historical outcome fits or code generation.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import json
from pathlib import Path

HEALTH = Path("outputs/external_data/health/source_health.json")
STORAGE = Path("outputs/external_data/health/storage_telemetry.json")
OUT = Path("outputs/research/source_utilization.json")
VERSION = "source-utilization-v1"
# Explicit research-only audit scope, not all known sources.
UNUSED = {
    "fomc_calendar": ("EVENT_CALENDAR", "TIME_TO_EVENT", "PIT release announcement revisions"),
    "bls_release_calendar": ("EVENT_CALENDAR", "TIME_TO_EVENT", "PIT scheduled vs revised timestamps"),
    "bea_release_calendar": ("EVENT_CALENDAR", "TIME_TO_EVENT", "PIT scheduled vs revised timestamps"),
    "clinicaltrials_sponsors": ("COMPANY_EVENT", "SPONSOR_CHANGE", "historical sponsor-to-ticker PIT mapping"),
    "usaspending_contracts": ("COMPANY_EVENT", "AWARD_CHANGE", "award announcement time and entity PIT linkage"),
    "sec_13f_holdings": ("DELAYED_FILING", "HOLDINGS_CHANGE", "actual 13F publication timestamp, not quarter end"),
    "fred_liquidity": ("MACRO_LIQUIDITY", "LIQUIDITY_CHANGE", "ALFRED vintage / measured availability lag"),
    "noaa_storm_events": ("HISTORICAL_WEATHER", "EVENT_EXPOSURE", "regional corporate exposure and release lag"),
}
# Explicitly freeze this table to reported audit scope. Another PR is needed
# to reclassify a source as consumed or use its observations for alpha.
READERS = {name: "NO_ALPHA_READER_REVIEWED" for name in UNUSED}
READERS["noaa_storm_events"] = "INGESTION_ONLY_MONTHLY_BACKFILL"


def _load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Research source audit input missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Research source audit input must be a dict: {path}")
    return data


def build(health: dict, storage: dict, *, as_of: str) -> dict:
    date.fromisoformat(as_of)  # explicit UTC date, reproducible snapshot
    sizes = storage.get("per_source") or {}
    items = []
    for sid, (family, hypothesis, gate) in sorted(UNUSED.items()):
        h, s = health.get(sid) or {}, sizes.get(sid) or {}
        if not isinstance(h, dict) or not isinstance(s, dict):
            raise ValueError("malformed source record: " + sid)
        items.append({
            "source_id": sid, "category": family,
            "code_linkage_status": READERS[sid],
            "source_health": h.get("status", "UNOBSERVED"),
            "last_success": h.get("last_success"),
            "health_rows_ingested": h.get("rows_ingested"),
            "health_message": h.get("message"),
            "telemetry_retrieval_days": s.get("retrieval_days"),
            "telemetry_rows": s.get("rows"),
            "measured_raw_bytes_per_day": s.get("raw_bytes_per_day"),
            "measured_normalized_bytes_per_day": s.get("normalized_bytes_per_day"),
            "one_off_backfill_bytes": s.get("one_off_backfill_bytes"),
            "candidate_research_question": hypothesis,
            "preregistration_preconditions": gate,
            "alpha_test_status": "NOT_EVALUATED",
            "feature_status": "NOT_ENABLED",
            "candidate_priority": ("P1_REVIEW" if sid in
                                   ("fred_liquidity", "fomc_calendar", "bea_release_calendar", "bls_release_calendar")
                                   else "P2_REVIEW" if sid in ("clinicaltrials_sponsors", "usaspending_contracts")
                                   else "P3_REVIEW"),
            "recommended_action": ("PREPARE_NEW_FROZEN_PROTOCOL" if sid in
                                   ("fred_liquidity", "fomc_calendar", "bea_release_calendar", "bls_release_calendar")
                                   else "REVIEW_COST_AND_PIT_FEASIBILITY"),
        })
    total_raw = sum(x["measured_raw_bytes_per_day"] for x in items
                    if isinstance(x["measured_raw_bytes_per_day"], (int, float)))
    total_normalized = sum(x["measured_normalized_bytes_per_day"] for x in items
                           if isinstance(x["measured_normalized_bytes_per_day"], (int, float)))
    return {
        "version": VERSION, "as_of": as_of,
        "telemetry_measured_at": storage.get("measured_at"),
        "classification_source": "docs/DATA_LINK_AUDIT_2026-11-08.md; manual linkage audit, NOT runtime analysis",
        "scope": "8 reviewed unused-source IDs, not all external ingestion",
        "n_sources": len(items), "sources": items,
        "measured_daily_bytes_for_scope": {"raw": total_raw, "normalized": total_normalized},
        "unmeasured_cost_is_not_zero": True,
        "no_new_features": True, "no_alpha_test": True,
        "no_automatic_source_disable": True,
        "no_protocol_or_production_changes": True,
        "proposed_next_protocol": {
            "status": "DRAFT_ONLY_NOT_REGISTERED",
            "focus": ["fred_liquidity", "fomc_calendar", "bls_release_calendar", "bea_release_calendar"],
            "required": [
                "Separate preregistered version, finite fixed PIT signal expressions and explicit hypothesis family",
                "Historical as-of timestamp/revision audit and source-to-panel join coverage",
                "Selection/training/locked-test isolation, no reuse of contaminated holdout",
                "Frozen baseline, ablation, placebo, BH/FDR budget and prospective independent periods",
                "Existing option economics with real archived entry/exit RTH quotes only",
                "Human approval before any real production or code promotion"],
        },
    }


def run(*, health_path: Path = HEALTH, storage_path: Path = STORAGE,
        out: Path = OUT, as_of: str | None = None) -> Path:
    as_of = as_of or datetime.now(timezone.utc).date().isoformat()
    result = build(_load(health_path), _load(storage_path), as_of=as_of)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", help="UTC date; defaults to today")
    a = ap.parse_args(argv)
    print(run(as_of=a.as_of))


if __name__ == "__main__":
    main()

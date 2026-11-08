"""Weekly, evidence-only inventory of archived source data without analytic consumers.

This is an OBSERVABILITY report, not an alpha model, a live code dependency
graph, an ingestion change or permission to use a new feature in production.
The no-reader classification is pinned to the reviewed 2026-11-08 audit and
must be re-reviewed if the analysis pipeline changes.
"""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path

HEALTH = Path("outputs/external_data/health/source_health.json")
TELEMETRY = Path("outputs/external_data/health/storage_telemetry.json")
OUT = Path("outputs/research/source_utilization")
AUDIT_REVIEWED_ON = "2026-11-08"
MAX_TELEMETRY_AGE_DAYS = 14

# Manually reviewed in docs/DATA_LINK_AUDIT_2026-11-08.md.
# IDs are *not* inferred to have consumers simply because the ingestion
# orchestrator, registry or a source-health test mentions them.
REVIEWED_WITHOUT_ALPHA_CONSUMER = {
    "fred_liquidity": ("HIGH", "preregister new ALFRED-vintage liquidity study; no live feature"),
    "bea_release_calendar": ("HIGH", "preregister ex-ante event-distance/volatility study"),
    "bls_release_calendar": ("HIGH", "preregister ex-ante event-distance/volatility study"),
    "fomc_calendar": ("HIGH", "preregister ex-ante event-distance/volatility study"),
    "clinicaltrials_sponsors": ("MEDIUM", "check issuer matching and announcement timestamps first"),
    "usaspending_contracts": ("MEDIUM", "verify award/publication timestamps and issuer mapping first"),
    "sec_13f_holdings": ("LOW", "large historical backfill; validate filing/ZIP publication times before testing"),
    "noaa_storm_events": ("LOW", "backfill-only already throttled; no PIT issuer exposure available"),
}


def _read(path: Path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (FileNotFoundError, OSError, ValueError):
        return None


def inventory(health: dict | None, telemetry: dict | None, as_of: date) -> dict:
    """No internet, no additional source reads; missing != zero and PASS != alpha."""
    as_of = date.fromisoformat(as_of) if isinstance(as_of, str) else as_of
    measured = (telemetry or {}).get("measured_at")
    try:
        snapshot_date = date.fromisoformat(str(measured)[:10])
        age = (as_of - snapshot_date).days
        telemetry_ok = 0 <= age <= MAX_TELEMETRY_AGE_DAYS
    except ValueError:
        age, telemetry_ok = None, False
    result = []
    source_telemetry = (telemetry or {}).get("per_source") or {}
    for sid, (priority, proposal) in REVIEWED_WITHOUT_ALPHA_CONSUMER.items():
        t = source_telemetry.get(sid) or {}
        h = (health or {}).get(sid) or {}
        metrics_available = bool(t) and telemetry_ok
        result.append({
            "source_id": sid,
            "consumer_classification": "NO_RESEARCH_READER_VERIFIED_AT_CODE_AUDIT",
            "consumer_audit_reviewed_on": AUDIT_REVIEWED_ON,
            "consumer_audit_is_live_dependency_scan": False,
            "research_priority": priority,
            "proposed_next_action": proposal,
            "action_state": "RECOMMENDATION_ONLY_NOT_EXECUTED",
            "health_status": h.get("status") or "UNKNOWN",
            "last_success": h.get("last_success"),
            "last_observation": h.get("latest_observation"),
            "archive_rows_total_reported": t.get("rows") if metrics_available else None,
            "estimated_raw_bytes_per_day": t.get("raw_bytes_per_day") if metrics_available else None,
            "estimated_normalized_bytes_per_day": t.get("normalized_bytes_per_day") if metrics_available else None,
            "one_off_backfill_bytes": t.get("one_off_backfill_bytes") if metrics_available else None,
            "retrieval_days": t.get("retrieval_days") if metrics_available else None,
            "data_completeness": "MEASURED_TELEMETRY" if metrics_available else "NOT_MEASURED_OR_STALE",
            "alpha_tested": False,
            "automated_source_deactivation": False,
        })
    return {
        "as_of": as_of.isoformat(),
        "measured_at": measured,
        "telemetry_age_days": age,
        "telemetry_max_age_days": MAX_TELEMETRY_AGE_DAYS,
        "status": ("MEASURED" if telemetry_ok and all(
            row["data_completeness"] == "MEASURED_TELEMETRY" and row["health_status"] != "UNKNOWN"
            for row in result) else "DATA_GAP"),
        "source_count": len(result),
        "sources": result,
        "research_protocol_modified": False,
        "data_sources_enabled_or_disabled": False,
        "consumer_audit_note": "Code audit as of 2026-11-08; this report does not rescan source-code consumers.",
        "purpose": "resource accounting and prioritization only; no alpha attribution or trading decisions",
    }


def run(as_of: date, health_path: Path = HEALTH, telemetry_path: Path = TELEMETRY, out: Path = OUT):
    as_of = date.fromisoformat(as_of) if isinstance(as_of, str) else as_of
    rep = inventory(_read(health_path), _read(telemetry_path), as_of)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{as_of.isoformat()}.json"
    dest.write_text(json.dumps(rep, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return dest, rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", required=True)
    ap.add_argument("--health", type=Path, default=HEALTH)
    ap.add_argument("--telemetry", type=Path, default=TELEMETRY)
    ap.add_argument("--output", type=Path, default=OUT)
    args = ap.parse_args(argv)
    target, rep = run(args.as_of, args.health, args.telemetry, args.output)
    print(target, rep["status"], rep["source_count"])
    return 0 if rep["status"] == "MEASURED" else 2


if __name__ == "__main__":
    raise SystemExit(main())

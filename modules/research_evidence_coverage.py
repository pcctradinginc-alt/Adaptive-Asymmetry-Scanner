"""Read-only P0/P1 evidence coverage audit for all candidate-ledger rows.

No network calls, no production changes, no invented labels.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path

from modules.research_evidence_discovery import eligible, load, number


def audit(rows, as_of):
    as_of = date.fromisoformat(as_of) if isinstance(as_of, str) else as_of
    cohort = [r for r in rows if isinstance(r, dict) and r.get("date") and r.get("ticker")
              and str(r["date"]) <= as_of.isoformat()]
    dates = {r["date"] for r in cohort}
    counts = Counter()
    feature_presence = Counter()
    for row in cohort:
        counts["candidate_rows"] += 1
        if row.get("signal_timestamp"):
            counts["timestamped_rows"] += 1
        if row.get("signal_id") or row.get("event_id"):
            counts["identified_rows"] += 1
        if row.get("direction") in ("BULLISH", "BEARISH"):
            counts["direction_rows"] += 1
        features = row.get("features")
        if isinstance(features, dict):
            counts["rows_with_features"] += 1
            feature_presence.update(k for k, v in features.items() if number(v) is not None)
        outcomes = row.get("outcomes") or {}
        if isinstance(outcomes, dict):
            if any(number(outcomes.get("ret_" + str(h) + "d")) is not None for h in (5, 20, 45)):
                counts["underlying_observed"] += 1
            if any(number(outcomes.get("real_strat_ret_" + str(h) + "d")) is not None for h in (5, 20, 45)):
                counts["option_mark_present"] += 1
        if eligible(row, as_of):
            counts["mature_20d_option_label"] += 1
        else:
            counts["not_eligible_for_20d_option_learning"] += 1
    total = counts["candidate_rows"]
    return {"as_of": as_of.isoformat(), "days_with_candidates": len(dates),
            "counts": dict(sorted(counts.items())),
            "rates": {k: (round(v / total, 4) if total else None) for k, v in counts.items()},
            "feature_presence": dict(sorted(feature_presence.items())),
            "label_definition": "real_strat_ret_20d, >=21 calendar days since signal",
            "quote_provenance_verified": False,
            "closed_loop_training_verified": False,
            "production_influence": "NONE"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--ledger", type=Path, default=Path("outputs/candidate_ledger"))
    parser.add_argument("--output", type=Path, default=Path("outputs/research/evidence_coverage"))
    args = parser.parse_args()
    report = audit(load(args.ledger, date.fromisoformat(args.as_of)), args.as_of)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / (args.as_of + ".json")
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()

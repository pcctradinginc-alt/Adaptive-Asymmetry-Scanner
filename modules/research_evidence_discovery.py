"""Offline, no-LLM option-outcome discovery with strict evidence and forward holdout.

No network calls, trading decisions, production config changes or inferred option marks.
Run: python -m modules.research_evidence_discovery --as-of 2026-10-30
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import date, timedelta
from itertools import combinations
from pathlib import Path

LEDGER = Path("outputs/candidate_ledger")
OUT = Path("outputs/research/evidence_discovery")
HORIZON = 20
MIN_TRAIN = 60
MIN_TEST = 30
MIN_TEST_DATES = 10
MIN_TRAIN_DATES = 20
MAX_FEATURES = 24
MAX_PAIRS = 100
MIN_GROUP = 20


def number(value):
    if isinstance(value, bool):
        return None
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (ValueError, TypeError):
        return None


def eligible(row, as_of):
    """Only completed, strictly past real option marks. No stock/BS proxy."""
    try:
        signal = date.fromisoformat(row["date"])
    except (ValueError, TypeError, KeyError):
        return False
    if signal > as_of - timedelta(days=HORIZON + 1):
        return False
    return number((row.get("outcomes") or {}).get("real_strat_ret_20d")) is not None


def load(root, as_of):
    rows = []
    for path in sorted(Path(root).glob("????-??.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict) and row.get("ticker") and row.get("date"):
                rows.append(row)
    return rows


def mean(vals):
    return sum(vals) / len(vals) if vals else None


def discover(rows, as_of, min_train=MIN_TRAIN, min_test=MIN_TEST):
    """Explore frozen training data; assess preselected rules on later dates only.

    Candidate feature thresholds come from TRAIN ONLY. Holdout is never searched
    for a favorable threshold. No automatic promotion to production.
    """
    as_of = date.fromisoformat(as_of) if isinstance(as_of, str) else as_of
    all_rows = [r for r in rows if isinstance(r, dict) and r.get("ticker") and r.get("date")]
    valid = [r for r in all_rows if eligible(r, as_of)]
    unique = {}
    for r in valid:
        key = (r["date"], r["ticker"], str(r.get("signal_id") or r.get("event_id") or ""))
        unique.setdefault(key, r)
    valid = sorted(unique.values(), key=lambda r: (r["date"], r["ticker"]))
    dates = sorted({r["date"] for r in valid})
    base = {"as_of": as_of.isoformat(), "population_rows": len(all_rows),
            "genuine_option_labels": len(valid),
            "missing_or_immature_option_labels": len(all_rows) - len(valid),
            "no_option_proxy": True, "production_influence": "NONE",
            "status": "INSUFFICIENT_EVIDENCE", "discoveries": []}
    if len(valid) < min_train + min_test or len(dates) < MIN_TRAIN_DATES + MIN_TEST_DATES:
        return base
    # Strictly date-based holdout: no same-day candidates across partitions.
    split = max(MIN_TRAIN_DATES, int(len(dates) * 0.7))
    if len(dates) - split < MIN_TEST_DATES:
        split = len(dates) - MIN_TEST_DATES
    train_days = set(dates[:split])
    train = [r for r in valid if r["date"] in train_days]
    test = [r for r in valid if r["date"] not in train_days]
    base.update(train_rows=len(train), test_rows=len(test),
                train_dates=len(train_days), test_dates=len(dates)-len(train_days),
                train_end=max(train_days), test_start=min(d for d in dates if d not in train_days))
    if len(train) < min_train or len(test) < min_test:
        return base
    def outcome(r):
        return number(r["outcomes"]["real_strat_ret_20d"])
    def features(r):
        return r.get("features") if isinstance(r.get("features"), dict) else {}
    names = Counter(k for r in train for k, v in features(r).items() if number(v) is not None)
    names = [k for k, n in sorted(names.items(), key=lambda x: (-x[1], x[0]))
             if n >= max(MIN_GROUP * 2, len(train)//2)][:MAX_FEATURES]
    candidates = []
    for name in names:
        vals = sorted(number(features(r).get(name)) for r in train if number(features(r).get(name)) is not None)
        if len(vals) < 2 * MIN_GROUP:
            continue
        for quantile in (0.25, 0.5, 0.75):
            threshold = vals[min(len(vals)-1, int(quantile * (len(vals)-1)))]
            for op in ("ge", "lt"):
                def match(r, n=name, t=threshold, o=op):
                    v = number(features(r).get(n))
                    return v is not None and (v >= t if o == "ge" else v < t)
                chosen = [outcome(r) for r in train if match(r)]
                rest = [outcome(r) for r in train if not match(r)]
                if min(len(chosen), len(rest)) < MIN_GROUP:
                    continue
                delta = mean(chosen) - mean(rest)
                candidates.append((delta, name, op, threshold, match))
    # A single top rule is selected on training data; no multiple-testing
    # search on the holdout. Results are descriptive, not significance claims.
    candidates.sort(key=lambda c: (-c[0], c[1], c[2], c[3]))
    base["train_rules_tested"] = len(candidates)
    if not candidates:
        base["status"] = "NO_TESTABLE_RULE"
        return base
    delta, name, op, threshold, match = candidates[0]
    a = [outcome(r) for r in test if match(r)]
    b = [outcome(r) for r in test if not match(r)]
    independent = len({r["date"] for r in test if match(r)})
    result = {"feature": name, "operator": op, "threshold": threshold,
              "train_delta": round(delta, 6), "holdout_group_n": len(a),
              "holdout_control_n": len(b), "holdout_independent_days": independent,
              "holdout_mean_group": mean(a), "holdout_mean_control": mean(b),
              "holdout_delta": (mean(a)-mean(b)) if a and b else None,
              "promotion_eligible": False}
    base["discoveries"] = [result]
    base["status"] = "HOLDOUT_OBSERVED" if len(a) >= MIN_GROUP and len(b) >= MIN_GROUP and independent >= MIN_TEST_DATES else "HOLDOUT_INSUFFICIENT"
    return base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    day = date.fromisoformat(args.as_of)
    result = discover(load(args.ledger, day), day)
    args.output.mkdir(parents=True, exist_ok=True)
    dest = args.output / (args.as_of + ".json")
    dest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(dest, result["status"])


if __name__ == "__main__":
    main()

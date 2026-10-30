"""Offline, LLM-free census of candidate-ledger decisions and observed outcomes.

Read-only: never requests quotes, changes gates, or invents option P&L.
Run: python -m modules.research_candidate_census --date YYYY-MM-DD
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path("outputs/candidate_ledger")
OUT = Path("outputs/research/candidate_census")


def audit(rows, day, sample_size=40):
    cohort = [r for r in rows if r.get("date") == day and r.get("ticker")]
    reasons = Counter(str(r.get("reject_reason") or "none") for r in cohort)
    unique = sorted({str(r["ticker"]) for r in cohort})
    # Hash-based audit sample is reproducible and independent of outcomes.
    sample = sorted(unique, key=lambda t: hashlib.sha256(
        (day + ":research:" + t).encode()).hexdigest())[:max(0, sample_size)]
    actual_option_marks = 0
    underlying_marks = 0
    for r in cohort:
        outcomes = r.get("outcomes") or {}
        if any(outcomes.get("ret_" + str(h) + "d") is not None for h in (5, 20, 45)):
            underlying_marks += 1
        if any(outcomes.get("real_strat_ret_" + str(h) + "d") is not None
               for h in (5, 20, 45)):
            actual_option_marks += 1
    return {"date": day, "signals": len(cohort), "unique_tickers": len(unique),
            "reasons": dict(sorted(reasons.items())),
            "underlying_outcome_rows": underlying_marks,
            "actual_option_outcome_rows": actual_option_marks,
            "audit_sample": sample,
            "audit_sample_is_quote_request": False,
            "missing_option_marks_are_not_zero_returns": True}


def run(day, root=ROOT, out=OUT, sample_size=40):
    month = day[:7]
    source = root / (month + ".jsonl")
    if not source.is_file():
        raise FileNotFoundError("Missing ledger: " + str(source))
    rows = []
    with source.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    result = audit(rows, day, sample_size)
    out.mkdir(parents=True, exist_ok=True)
    target = out / (day + ".json")
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--sample-size", type=int, default=40)
    args = parser.parse_args()
    print(run(args.date, sample_size=args.sample_size))


if __name__ == "__main__":
    main()

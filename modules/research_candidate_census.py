"""Offline, LLM-free census of candidate-ledger decisions and observed outcomes.

Read-only: never requests quotes, changes gates, or invents option P&L.
Run: python -m modules.research_candidate_census --date YYYY-MM-DD
"""
import argparse
import hashlib
import json
from collections import Counter
from datetime import date, datetime, timezone
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
            "missing_option_marks_are_not_zero_returns": True,
            "option_outcome_provenance": "LEDGER_REPORTED_NOT_QUOTE_VERIFIED",
            "verification_required": "Compare each option mark with archived entry/exit RTH bid/ask through existing research_quote_reconciliation",
            "test_population": "all existing ledger candidates, including budget deferrals; no new quotes"}


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


def latest_completed_day(root=ROOT, as_of=None):
    """Latest ledger date strictly before as-of (UTC); never research same-day partial data.

    Read the existing monthly ledger, not a speculative exchange calendar.
    Explicit missing/corrupt archives fail rather than fabricate outcomes.
    """
    if as_of is None:
        as_of = datetime.now(timezone.utc).date()
    if isinstance(as_of, str):
        as_of = date.fromisoformat(as_of)
    if not isinstance(as_of, date):
        raise ValueError("as_of must be an ISO date")
    days = set()
    for path in sorted(root.glob("????-??.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                raw = row.get("date")
                if not isinstance(raw, str):
                    continue
                try:
                    day = date.fromisoformat(raw)
                except ValueError:
                    continue
                if day < as_of and row.get("ticker"):
                    days.add(day)
    if not days:
        raise FileNotFoundError("No completed candidate ledger date before " + as_of.isoformat())
    return max(days).isoformat()


def run_latest(root=ROOT, out=OUT, as_of=None, sample_size=40):
    """Run existing census automatically, with explicit provenance limitations."""
    day = latest_completed_day(root, as_of)
    target = run(day, root=root, out=out, sample_size=sample_size)
    return target


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--date")
    group.add_argument("--latest-completed", action="store_true")
    parser.add_argument("--as-of", help="UTC reference date; only with --latest-completed")
    parser.add_argument("--sample-size", type=int, default=40)
    args = parser.parse_args()
    if args.as_of and not args.latest_completed:
        parser.error("--as-of requires --latest-completed")
    target = (run_latest(as_of=args.as_of, sample_size=args.sample_size)
              if args.latest_completed else run(args.date, sample_size=args.sample_size))
    print(target)


if __name__ == "__main__":
    main()

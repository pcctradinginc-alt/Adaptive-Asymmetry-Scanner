"""Offline, LLM-free census of candidate-ledger decisions and observed outcomes.

Read-only: never requests quotes, changes gates, or invents option P&L.
Run: python -m modules.research_candidate_census --date YYYY-MM-DD
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from datetime import date, timedelta

from modules.research_evidence_discovery import number

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




EVIDENCE = Path("outputs/research/evidence_discovery")


def audit_window(rows, as_of, days=7, sample_size=40):
    """All candidates across a fixed window, including economics-first deferrals.

    Observed marks are *unverified* here: this function never accesses quotes,
    attributes the existing all-history verification to this weekly cohort,
    chooses trading actions or backfills missing outcomes.
    """
    as_of = date.fromisoformat(as_of) if isinstance(as_of, str) else as_of
    if not (1 <= days <= 31 and 0 <= sample_size <= 100):
        raise ValueError("bounded weekly census requires 1..31 days, sample <= 100")
    first = as_of - timedelta(days=days - 1)
    subset = [r for r in rows if isinstance(r, dict) and r.get("ticker")
              and isinstance(r.get("date"), str) and first.isoformat() <= r["date"] <= as_of.isoformat()]
    reasons = Counter(str(r.get("reject_reason") or "none") for r in subset)
    selections = Counter(str(r.get("option_sample") or "none") for r in subset)
    tickers = sorted({str(r["ticker"]) for r in subset})
    samples = sorted(tickers, key=lambda t: hashlib.sha256(
        (first.isoformat() + ":" + as_of.isoformat() + ":research:" + t).encode()).hexdigest())[:sample_size]
    under = opt = ident = 0
    for r in subset:
        if r.get("signal_id") or r.get("event_id"):
            ident += 1
        outcomes = r.get("outcomes") if isinstance(r.get("outcomes"), dict) else {}
        under += any(number(outcomes.get(f"ret_{h}d")) is not None for h in (5, 20, 45))
        opt += any(number(outcomes.get(f"real_strat_ret_{h}d")) is not None for h in (5, 20, 45))
    return {"window_start": first.isoformat(), "as_of": as_of.isoformat(),
            "window_days": days, "candidate_rows": len(subset),
            "days_with_candidate_rows": len({r["date"] for r in subset}),
            "unique_tickers": len(tickers), "signal_identified_rows": ident,
            "reject_reasons": dict(sorted(reasons.items())),
            "option_sampling_classes": dict(sorted(selections.items())),
            "underlying_mark_present_rows": under,
            "option_mark_present_unverified_rows": opt,
            "sampling_audit_tickers": samples,
            "sampling_audit_is_not_quote_request": True,
            "marked_option_returns_are_not_verified_by_this_count": True,
            "no_real_option_return_imputation": True}


def quote_evidence_summary(report, as_of):
    """Reuse the independently produced quote archive reconciliation in-place.

    That report covers ALL historical eligible ledger rows, not just the
    seven-day cohort. No inference of weekly verified rows from a global total.
    """
    if not isinstance(report, dict) or report.get("as_of") != as_of or not report.get("no_option_proxy") \
            or report.get("production_influence") != "NONE":
        return {"status": "MISSING_OR_INVALID_EXISTING_EVIDENCE", "global_verified_option_labels": None,
                "global_unverified_option_labels": None, "provenance_counts": None,
                "scope": "ALL_ELIGIBLE_HISTORICAL_CANDIDATES_NOT_THIS_WEEK"}
    valid = report.get("genuine_option_labels")
    invalid = report.get("unverified_option_labels")
    counters = report.get("provenance_counts")
    if not (type(valid) is int and valid >= 0 and type(invalid) is int and invalid >= 0
            and isinstance(counters, dict)
            and all(type(v) is int and v >= 0 for v in counters.values())
            and counters.get("VERIFIED", 0) == valid
            and sum(counters.values()) == valid + invalid):
        return {"status": "MISSING_OR_INVALID_EXISTING_EVIDENCE", "global_verified_option_labels": None,
                "global_unverified_option_labels": None, "provenance_counts": None,
                "scope": "ALL_ELIGIBLE_HISTORICAL_CANDIDATES_NOT_THIS_WEEK"}
    return {"status": "FROM_EXISTING_RTH_QUOTE_PROVENANCE_AUDIT",
            "global_verified_option_labels": valid,
            "global_unverified_option_labels": invalid,
            "provenance_counts": report["provenance_counts"],
            "evidence_class": report.get("evidence_class"),
            "scope": "ALL_ELIGIBLE_HISTORICAL_CANDIDATES_NOT_THIS_WEEK"}


def run_week(as_of, root=ROOT, out=OUT, evidence_dir=EVIDENCE, days=7, sample_size=40):
    """Run weekly from committed ledger plus the already-run evidence stage."""
    as_of = date.fromisoformat(as_of) if isinstance(as_of, str) else as_of
    if not (1 <= days <= 31):
        raise ValueError("1..31 days only")
    first = as_of - timedelta(days=days - 1)
    required_months = sorted({(first + timedelta(days=n)).strftime("%Y-%m") for n in range(days)})
    rows, missing = [], []
    for month in required_months:
        source = root / f"{month}.jsonl"
        if not source.is_file():
            missing.append(month)
            continue
        with source.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError("Invalid candidate ledger JSONL record")
                    rows.append(record)
    rep = audit_window(rows, as_of, days, sample_size)
    rep["missing_ledger_partitions"] = missing
    evidence_file = evidence_dir / (as_of.isoformat() + ".json")
    try:
        existing = json.loads(evidence_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        existing = None
    rep["quote_provenance"] = quote_evidence_summary(existing, as_of.isoformat())
    rep["status"] = ("AUDIT_COMPLETE" if not missing and
                     rep["quote_provenance"]["status"] == "FROM_EXISTING_RTH_QUOTE_PROVENANCE_AUDIT"
                     else "DATA_GAP")
    rep["production_influence"] = "NONE"
    rep["research_only_no_new_capture"] = True
    out.mkdir(parents=True, exist_ok=True)
    dest = out / (as_of.isoformat() + ".json")
    dest.write_text(json.dumps(rep, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return dest, rep


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--date")
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--as-of", help="Weekly research audit as-of date; mutually exclusive with --date")
    parser.add_argument("--days", type=int, default=7, help="Calendar days including as-of")
    parser.add_argument("--evidence-dir", type=Path, default=EVIDENCE,
                        help="Existing quote-provenance report directory, never fetch new quotes")
    parser.add_argument("--ledger", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args(argv)
    if bool(args.date) == bool(args.as_of):
        parser.error("provide exactly one of --date (legacy) or --as-of (weekly)")
    if args.date:
        print(run(args.date, root=args.ledger, out=args.output, sample_size=args.sample_size))
        return 0
    dest, rep = run_week(args.as_of, root=args.ledger, out=args.output,
                         evidence_dir=args.evidence_dir, days=args.days, sample_size=args.sample_size)
    print(dest, rep["status"], "candidate_rows=", rep["candidate_rows"],
          "global_verified_option_labels=", rep["quote_provenance"]["global_verified_option_labels"])
    return 0 if rep["status"] == "AUDIT_COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())

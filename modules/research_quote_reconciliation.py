"""P0: independently reconcile archived real option quotes with research marks.

Never infer missing prices or silently treat a claimed mark as training truth.
Only verified, matured rows may be exported for offline research.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from modules import option_quotes as oq
from modules import roi_learning_shadow as rl


def reconcile(root: Path, as_of: date) -> dict:
    cohorts = rl._journal_read(root / rl.COHORTS)
    marks = rl._journal_read(root / rl.MARKS)
    by_id = {c["cohort_id"]: c for c in cohorts}
    counts = Counter()
    verified = []
    for mark in marks:
        counts["marks"] += 1
        cohort = by_id.get(mark.get("cohort_id"))
        horizon = mark.get("horizon_days")
        if cohort is None or horizon not in (5, 20, 45):
            counts["orphan_or_invalid_mark"] += 1
            continue
        signal = date.fromisoformat(cohort["signal_date"])
        if signal + timedelta(days=horizon) > as_of:
            counts["immature"] += 1
            continue
        entry_ids = mark.get("entry_quote_ids") or []
        exit_ids = mark.get("mark_quote_ids") or []
        if len(entry_ids) != len(cohort["legs"]) or len(exit_ids) != len(cohort["legs"]):
            counts["missing_leg_references"] += 1
            continue
        quotes = oq.by_ids(
            list(dict.fromkeys(entry_ids + exit_ids)), cohort["signal_date"],
            as_of.isoformat(), root=root / "outputs/options_archive/quotes")
        entry = [quotes.get(i) for i in entry_ids]
        exit_ = [quotes.get(i) for i in exit_ids]
        symbols = [leg["occ_symbol"] for leg in cohort["legs"]]
        if any(q is None or not oq.usable_rth_quote(q) for q in entry + exit_):
            counts["missing_or_unusable_quotes"] += 1
            continue
        if [q["occ_symbol"] for q in entry] != symbols or [q["occ_symbol"] for q in exit_] != symbols:
            counts["wrong_contract"] += 1
            continue
        decision = rl._time(cohort.get("decision_at"))
        times_entry = [rl._time(q.get("retrieved_at")) for q in entry]
        times_exit = [rl._time(q.get("retrieved_at")) for q in exit_]
        if decision is None or any(t is None or t < decision for t in times_entry):
            counts["predecision_entry"] += 1
            continue
        if any(t is None or t < min(times_entry) + timedelta(days=horizon)
               for t in times_exit):
            counts["premature_exit"] += 1
            continue
        debit = rl._debit(entry)
        credit = rl._credit(exit_)
        fees = 2 * len(symbols) * float(cohort["commission_usd_per_leg_per_side"]) / 100.0
        expected = (credit - debit - fees) / debit if debit and credit is not None else None
        if expected is None or abs(expected - float(mark.get("net_return", "nan"))) > 0.000002:
            counts["economics_mismatch"] += 1
            continue
        counts["verified"] += 1
        verified.append({"cohort_id": cohort["cohort_id"], "horizon_days": horizon,
                         "signal_date": cohort["signal_date"], "ticker": cohort["ticker"],
                         "net_return": round(expected, 6),
                         "entry_quote_ids": entry_ids, "exit_quote_ids": exit_ids,
                         "evidence": "ARCHIVED_EXECUTABLE_RTH_QUOTES",
                         "production_eligible": False})
    return {"as_of": as_of.isoformat(), "counts": dict(sorted(counts.items())),
            "verified_research_outcomes": verified,
            "training_consumption_verified": False,
            "closed_loop_verified": False,
            "production_influence": "NONE"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("outputs/research/p0_quote_reconciliation"))
    args = parser.parse_args()
    report = reconcile(args.root, date.fromisoformat(args.as_of))
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / (args.as_of + ".json")
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()

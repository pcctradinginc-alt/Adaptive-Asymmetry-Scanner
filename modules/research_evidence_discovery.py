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
from pathlib import Path

from modules import option_quotes as oq

LEDGER = Path("outputs/candidate_ledger")
QUOTES = Path("outputs/options_archive/quotes")
OUT = Path("outputs/research/evidence_discovery")
HORIZON = 20
MIN_TRAIN = 60
MIN_TEST = 30
MIN_TEST_DATES = 10
MIN_TRAIN_DATES = 20
MAX_FEATURES = 24
MIN_GROUP = 20
PRICE_TOL = 0.005        # ledger leg price vs. archived quote (half a cent)
RET_TOL = 0.00006        # ledger rounds real_strat_ret to 4 decimals


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
    outcomes = row.get("outcomes") or {}
    if not isinstance(outcomes, dict):
        return False
    return number(outcomes.get("real_strat_ret_20d")) is not None


def quote_index(records):
    """Archived option quotes by record_id (first observation wins, as in option_quotes.by_ids)
    and usable RTH quotes by OCC symbol."""
    by_id, by_symbol = {}, {}
    for r in sorted((r for r in records or [] if isinstance(r, dict)),
                    key=lambda r: str(r.get("retrieved_at") or "")):
        if r.get("record_id"):
            by_id.setdefault(r["record_id"], r)
        if oq.usable_rth_quote(r):
            by_symbol.setdefault(r.get("occ_symbol"), []).append(r)
    return {"by_id": by_id, "by_symbol": by_symbol}


def _same(a, b, tol=PRICE_TOL):
    a, b = number(a), number(b)
    return a is not None and b is not None and abs(a - b) <= tol


def provenance(row, index, horizon=HORIZON):
    """VERIFIED only if entry and exit of the recorded strategy are reproduced from archived,
    usable RTH quotes; otherwise a reason code. Never a substitute price."""
    rs = row.get("real_strategy")
    if not isinstance(rs, dict) or rs.get("entry_pending") or rs.get("entry_status") == "UNUSABLE":
        return "NO_FILLED_STRATEGY"
    legs = sorted((g for g in rs.get("legs") or [] if isinstance(g, dict)),
                  key=lambda g: 0 if g.get("side") == "long" else 1)
    if not legs or len(legs) > 2 or legs[0].get("side") != "long" or not all(g.get("symbol") for g in legs):
        return "LEGS_INVALID"
    symbols = [g["symbol"] for g in legs]
    entry_day = str(rs.get("entry_date") or "")
    debit = number(rs.get("net_debit_entry"))
    if debit is None and len(legs) == 1:
        debit = number(legs[0].get("ask"))
    if not entry_day or debit is None or debit <= 0:
        return "ENTRY_INVALID"

    by_symbol = index["by_symbol"]
    ids = rs.get("entry_quote_record_ids")
    if ids:                                   # timing-v2 fill: exact archive records
        recs = [index["by_id"].get(i) for i in ids]
        if any(r is None for r in recs):
            return "ENTRY_NOT_ARCHIVED"
        if [r.get("occ_symbol") for r in recs] != symbols:
            return "ENTRY_CONTRACT_MISMATCH"
        if any(not oq.usable_rth_quote(r) or oq.quote_day(r) != entry_day for r in recs):
            return "ENTRY_QUOTE_UNUSABLE"
        price = number(recs[0].get("ask"))
        if len(recs) == 2 and price is not None:
            short_bid = number(recs[1].get("bid"))
            price = None if short_bid is None else price - short_bid
        if not _same(price, debit):
            return "ENTRY_PRICE_MISMATCH"
    else:                                     # snapshot entry: the recorded leg prices must be archived that day
        for g in legs:
            field = "ask" if g.get("side") == "long" else "bid"
            if not any(oq.quote_day(r) == entry_day and _same(r.get(field), g.get(field))
                       for r in by_symbol.get(g["symbol"], [])):
                return "ENTRY_NOT_ARCHIVED"

    outcomes = row.get("outcomes") or {}
    claimed = number(outcomes.get(f"real_strat_ret_{horizon}d"))
    mark_day = str(outcomes.get(f"real_strat_mark_date_{horizon}d") or "")
    if claimed is None:
        return "NO_OPTION_LABEL"
    if outcomes.get(f"real_strat_late_{horizon}d"):
        return "LATE_MARK"
    try:
        due = date.fromisoformat(entry_day) + timedelta(days=horizon)
        if date.fromisoformat(mark_day) < due:
            return "PREMATURE_MARK"
    except ValueError:
        return "MARK_DATE_MISSING"
    if rs.get("expiry") and mark_day > str(rs["expiry"]):
        return "SETTLEMENT_NOT_QUOTE"
    longs = [r for r in by_symbol.get(symbols[0], []) if oq.quote_day(r) == mark_day]
    shorts = ([r for r in by_symbol.get(symbols[1], []) if oq.quote_day(r) == mark_day]
              if len(symbols) == 2 else [None])
    if not longs or not shorts:
        return "EXIT_NOT_ARCHIVED"
    for lq in longs:
        for sq in shorts:
            bid = number(lq.get("bid"))
            ask = None if sq is None else number(sq.get("ask"))
            if bid is None or (sq is not None and ask is None):
                continue
            exit_value = bid if sq is None else max(bid - ask, 0.0)
            if abs(round(exit_value / debit - 1.0, 4) - claimed) <= RET_TOL:
                return "VERIFIED"
    return "EXIT_PRICE_MISMATCH"


def load_quotes(root, rows, as_of):
    """Archive records from the earliest labelled signal date up to as_of."""
    days = [r["date"] for r in rows if isinstance(r, dict) and r.get("date")]
    if not days:
        return []
    return oq.read(min(days), as_of.isoformat(), root=Path(root))


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


def discover(rows, as_of, min_train=MIN_TRAIN, min_test=MIN_TEST, quotes=None):
    """Explore frozen training data; assess preselected rules on later dates only.

    Only labels whose entry and exit are reproduced from archived quotes (`quotes`,
    see provenance) are used; without quotes nothing is verified (fail closed).
    Candidate feature thresholds come from TRAIN ONLY. Holdout is never searched
    for a favorable threshold. No automatic promotion to production.
    """
    as_of = date.fromisoformat(as_of) if isinstance(as_of, str) else as_of
    all_rows = [r for r in rows if isinstance(r, dict) and r.get("ticker") and r.get("date")]
    numeric = [r for r in all_rows if eligible(r, as_of)]
    unique = {}
    for r in numeric:
        key = (r["date"], r["ticker"], str(r.get("signal_id") or r.get("event_id") or ""))
        unique.setdefault(key, r)
    index = quote_index(quotes)
    checked = [(r, provenance(r, index)) for r in unique.values()]
    provenance_counts = Counter(status for _, status in checked)
    valid = sorted((r for r, status in checked if status == "VERIFIED"),
                   key=lambda r: (r["date"], r["ticker"]))
    dates = sorted({r["date"] for r in valid})
    base = {"as_of": as_of.isoformat(), "population_rows": len(all_rows),
            "genuine_option_labels": len(valid),
            "unverified_option_labels": len(unique) - len(valid),
            "missing_or_immature_option_labels": len(all_rows) - len(numeric),
            "provenance_counts": dict(sorted(provenance_counts.items())),
            "verified_label_dates": len(dates),
            # Der Datums-Holdout wird jede Woche erneut ausgewertet -> nur explorativ. Bindend ist der
            # Forward-Test vorab festgehaltener Prognosen (modules/option_learner, predictions.jsonl).
            "evidence_class": "EXPLORATORY_REPEATED_HOLDOUT",
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
    base["caveats"] = ["Single holdout is not a repeated purged walk-forward.",
                       "The holdout is re-read every week: exploratory only; binding evidence is the option_learner "
                       "forward test on pre-registered predictions.",
                       "Exploratory training search requires multiple-testing correction before promotion."]
    base["status"] = "HOLDOUT_OBSERVED" if len(a) >= MIN_GROUP and len(b) >= MIN_GROUP and independent >= MIN_TEST_DATES else "HOLDOUT_INSUFFICIENT"
    return base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--quotes", type=Path, default=QUOTES)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    day = date.fromisoformat(args.as_of)
    rows = load(args.ledger, day)
    labelled = [r for r in rows if isinstance(r, dict) and eligible(r, day)]
    result = discover(rows, day, quotes=load_quotes(args.quotes, labelled, day))
    args.output.mkdir(parents=True, exist_ok=True)
    dest = args.output / (args.as_of + ".json")
    dest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(dest, result["status"])


if __name__ == "__main__":
    main()

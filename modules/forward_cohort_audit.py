"""Independent, read-only audit of 20-session forward XS returns.

Never used as an options P&L, strategy gate, live alpha proof or permission
to promote models. Deduplicates prediction_id and aggregates by signal_date.
HAC variance is diagnostic; no statistical claims when effective independent
blocks are insufficient.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

HORIZON = 20
MIN_INDEPENDENT_BLOCKS = 5
METRIC = "fwd_xs_20 (cross-sectional stock return, NOT options net P&L)"


def audit(rows: list[dict], *, horizon: int = HORIZON) -> dict:
    if type(horizon) is not int or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    report = {"schema": "forward_cohort_audit_v1", "metric": METRIC,
              "horizon_sessions": horizon, "raw_rows": len(rows), "verified_rows": 0,
              "unique_signal_dates": 0, "independent_blocks_upper_bound": 0,
              "first_signal_date": None, "last_signal_date": None,
              "day_mean": None, "hac_t_stat": None, "hac_ci95": None,
              "alpha_after_costs": "NOT_EVALUATED",
              "production_influence": "NONE", "status": "INSUFFICIENT_EVIDENCE"}
    by_id = {}
    invalid = 0
    pending = 0
    for row in rows:
        if not isinstance(row, dict):
            invalid += 1
            continue
        pid = row.get("prediction_id")
        day = row.get("signal_date")
        value = row.get("actual_return")
        try:
            if not isinstance(pid, str) or not pid.strip() or not isinstance(day, str):
                raise ValueError("missing key/date")
            date.fromisoformat(day)
            if value is None:
                pending += 1
                continue
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("invalid return")
            if pid in by_id:
                if by_id[pid] != (day, float(value)):
                    invalid += 1
                continue
            by_id[pid] = (day, float(value))
        except (ValueError, TypeError):
            invalid += 1
    report["invalid_or_conflicting_rows"] = invalid
    report["pending_outcome_rows"] = pending
    report["duplicate_rows"] = len(rows) - len(by_id) - invalid - pending
    days = defaultdict(list)
    for day, value in by_id.values():
        days[day].append(value)
    values = [sum(days[d]) / len(days[d]) for d in sorted(days)]
    n = len(values)
    report["verified_rows"] = len(by_id)
    report["unique_signal_dates"] = n
    report["independent_blocks_upper_bound"] = n // horizon
    if n:
        report["first_signal_date"] = min(days)
        report["last_signal_date"] = max(days)
        report["day_mean"] = sum(values) / n
    # Strict gate: no pretending the cross-section provides independent days.
    # Minimum 5 horizon-length non-overlap blocks and no conflicting records.
    if invalid or n < MIN_INDEPENDENT_BLOCKS * horizon:
        report["reason"] = "too_few_independent_dates_or_invalid_provenance"
        return report

    mu = sum(values) / n
    # Guard floating-point cancellation: identical outcomes can otherwise
    # yield variance ~1e-36 and an absurdly large, spurious t-statistic.
    if max(values) - min(values) <= 1e-12:
        report["reason"] = "nonpositive_or_unreliable_HAC_variance"
        return report
    residual = [x - mu for x in values]
    # Bartlett/Newey-West HAC for daily cohort means; sample dates not guaranteed
    # consecutive market sessions, hence diagnostic only, not formal calendar HAC.
    lag = min(horizon - 1, n - 1)
    variance = sum(x * x for x in residual) / n
    for k in range(1, lag + 1):
        covariance = sum(residual[i] * residual[i-k] for i in range(k, n)) / n
        variance += 2 * (1 - k / (lag + 1)) * covariance
    if not math.isfinite(variance) or variance <= 0:
        report["reason"] = "nonpositive_or_unreliable_HAC_variance"
        return report
    se = math.sqrt(variance / n)
    report["hac_t_stat"] = mu / se
    report["hac_ci95"] = [mu - 1.96 * se, mu + 1.96 * se]
    report["status"] = "DESCRIPTIVE_HAC_ONLY"
    report["reason"] = "No matched net options benchmark, no trading-calendar HAC, no promotion"
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-dir", type=Path, default=Path("outputs/research/prediction_memory"))
    parser.add_argument("--output", type=Path, default=Path("outputs/research/forward_cohort_audit.json"))
    args = parser.parse_args(argv)
    from modules import prediction_memory
    try:
        records = prediction_memory.load_memory(args.memory_dir)
        # DataFrame conversion represents absent outcomes as NaN. Only rows
        # without an outcome event are pending; actual nonfinite outcomes stay invalid.
        rows = records.to_dict("records")
        for row in rows:
            recorded = row.get("recorded_at")
            if recorded is None or (isinstance(recorded, float) and math.isnan(recorded)):
                if isinstance(row.get("actual_return"), float) and math.isnan(row["actual_return"]):
                    row["actual_return"] = None
        payload = audit(rows)
    except ValueError as exc:
        payload = audit([])
        payload.update(status="INVALID_LEDGER", reason=str(exc))
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

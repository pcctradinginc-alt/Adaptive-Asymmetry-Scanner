"""P0 forward cohort audit: dates/IDs must never fake sample size."""
import math
from datetime import date, timedelta

from modules.forward_cohort_audit import audit


def row(pid, day, x):
    return {"prediction_id": pid, "signal_date": day, "actual_return": x}


def test_twelve_same_day_observations_are_not_twelve_independent_periods():
    rows = [row(str(i), "2026-11-02", -.01 - i*.001) for i in range(12)]
    result = audit(rows)
    assert result["raw_rows"] == 12
    assert result["unique_signal_dates"] == 1
    assert result["hac_t_stat"] is None
    assert result["status"] == "INSUFFICIENT_EVIDENCE"


def test_no_dates_or_nan_never_enters_inference():
    a = audit([row("a", "", -.05), row("b", "2026-11-02", float("nan"))])
    assert a["verified_rows"] == 0 and a["invalid_or_conflicting_rows"] == 2
    assert a["hac_t_stat"] is None


def test_duplicate_prediction_ids_dont_inflate_count():
    data = [row("a", "2026-11-02", -.01)] * 8
    x = audit(data)
    assert x["verified_rows"] == 1
    assert x["duplicate_rows"] == 7


def test_conflicting_duplicate_ids_block_inference():
    x = audit([row("a", "2026-11-02", -.01), row("a", "2026-11-03", -.02)])
    assert x["invalid_or_conflicting_rows"] == 1
    assert x["status"] == "INSUFFICIENT_EVIDENCE"


def test_overlapping_forward_horizon_requires_independent_blocks():
    rows = [row(str(i), (date(2026, 1, 1) + timedelta(days=i)).isoformat(), -.01) for i in range(40)]
    x = audit(rows)
    assert x["unique_signal_dates"] == 40
    assert x["independent_blocks_upper_bound"] == 2
    assert x["hac_t_stat"] is None


def test_zero_variance_cannot_produce_false_significance():
    rows = [row(str(i), (date(2026, 1, 1) + timedelta(days=i)).isoformat(), -.01) for i in range(105)]
    x = audit(rows)
    assert x["status"] == "INSUFFICIENT_EVIDENCE"
    assert x["hac_t_stat"] is None


def test_sufficient_days_are_still_descriptive_not_options_alpha():
    rows = [row(str(i), (date(2026, 1, 1) + timedelta(days=i)).isoformat(),
                (-1 if i % 7 < 3 else 1) * 0.01) for i in range(120)]
    x = audit(rows)
    assert x["status"] == "DESCRIPTIVE_HAC_ONLY"
    assert math.isfinite(x["hac_t_stat"])
    assert x["production_influence"] == "NONE"
    assert x["alpha_after_costs"] == "NOT_EVALUATED"

"""Read-only, reproducible attribution of verified option cohort losses.

This is a research diagnostic, not a trading rule, optimizer, or live-fill model.
It reads committed reports only, fails closed on missing/inconsistent fields,
and never writes outputs or alters production state.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


class AuditError(ValueError):
    """Required independent loss-attribution evidence is invalid."""


def _load(root: Path, relative: str) -> dict:
    path = root / relative
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AuditError(f"Cannot read {relative}: {type(exc).__name__}") from exc
    if not isinstance(obj, dict):
        raise AuditError(f"{relative}: expected a JSON object")
    return obj


def _metric(row: dict, key: str) -> tuple[int, float]:
    metric = row.get(key)
    if not isinstance(metric, dict):
        raise AuditError(f"{key}: missing metric")
    n, v = metric.get("n"), metric.get("mean_diff")
    if type(n) is not int or n <= 0 or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise AuditError(f"{key}: invalid count or mean")
    return n, float(v)


def audit_cohort(status: dict) -> dict:
    """Validate quote-based return identity before attributing losses."""
    native = status.get("options_native") or {}
    comp = native.get("components") or {}
    execution = native.get("execution") or {}
    overall = execution.get("overall") or {}
    n_mid, mid = _metric(comp, "mid_ret")
    n_cost, cost = _metric(comp, "cost")
    n_exec, executed = _metric(comp, "exec_ret")
    if n_mid != n_cost or n_mid != n_exec:
        raise AuditError("Mid, cost, and executable populations differ")
    if abs(mid + cost - executed) > 0.00005:
        raise AuditError("Executable return does not equal mid return plus cost")
    if cost > 0:
        raise AuditError("Unexpected positive crossed-quote execution cost")
    for name in ("cost", "exec_ret"):
        n, mean = _metric(overall, name)
        expected = cost if name == "cost" else executed
        if n != n_mid or abs(mean - expected) > 0.00005:
            raise AuditError(f"execution.overall.{name} inconsistent with components")
    buckets = execution.get("by_entry_spread")
    if not isinstance(buckets, dict) or not buckets:
        raise AuditError("Entry-spread bucket evidence missing")
    bucket_summary = {}
    population = 0
    for key, value in buckets.items():
        if not isinstance(value, dict):
            raise AuditError(f"{key}: invalid bucket")
        nb, v = _metric(value, "exec_ret")
        population += nb
        bucket_summary[key] = {"n": nb, "mean_executable_return": v}
    if population != n_mid:
        raise AuditError(f"Bucket total {population} != valid cohort {n_mid}")
    independent = (overall.get("cost") or {}).get("independent_periods")
    if type(independent) is not int or independent < 0:
        raise AuditError("Independent holding-period count missing")
    return {
        "as_of": status.get("as_of"),
        "horizon_sessions": status.get("horizon"),
        "n_verified_contract_labels_with_attribution": n_mid,
        "independent_holding_periods": independent,
        "mean_mid_return": mid,
        "mean_cost_effect": cost,
        "mean_executable_return": executed,
        "identity_residual": round(executed - mid - cost, 8),
        "spread_buckets": bucket_summary,
        "interpretation": "Descriptive NBBO-crossed quote scenario, not observed order fills",
        "research_only": True,
    }


def audit_repository(root: Path) -> dict:
    five = audit_cohort(_load(root, "outputs/research/option_learner/status_5d.json"))
    twenty = audit_cohort(_load(root, "outputs/research/option_learner/status_20d.json"))
    if five["horizon_sessions"] != 5 or twenty["horizon_sessions"] != 20:
        raise AuditError("5d/20d experiment horizons are swapped or inconsistent")
    paper = _load(root, "outputs/research/paper_performance_analysis.json")
    reliable = paper.get("n_reliable")
    closed = paper.get("n_closed")
    if type(reliable) is not int or type(closed) is not int or not 0 <= reliable <= closed:
        raise AuditError("Paper trade reliability denominator invalid")
    strategies = {}
    for name, row in (paper.get("by_strategy") or {}).items():
        if isinstance(row, dict):
            strategies[name] = {k: row.get(k) for k in ("n", "win_rate", "mean")}
    return {
        "schema": "option-loss-attribution-readonly-v1",
        "research_only": True,
        "production_influence": "NONE",
        "fill_quality": "NBBO hypothetical; fills not verified",
        "cohorts": {"5d": five, "20d": twenty},
        "champion": {
            "closed": closed, "reliable": reliable,
            "n_unreliable_or_unassessed": closed - reliable,
            "strategies": strategies,
        },
        "caveats": [
            "Contract labels are not independent trades or portfolio returns.",
            "Independent holding periods, not contract count, constrain inference.",
            "Mid is not executable; limit scenarios are not fill-verified.",
            "No threshold changes, backfill, promotion, or model training follow from this report.",
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    print(json.dumps(audit_repository(args.root), indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

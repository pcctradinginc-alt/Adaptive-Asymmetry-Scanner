"""Deterministic, point-in-time quote allocation for research-only shadow.

Budget-neutral ranking experiment: existing Champion quotes + two cheap ranking
arms + a reproducible random audit. No trading signal, model training, or order.
"""
from __future__ import annotations

import hashlib
import math

from modules import economics_first as ef


def _bounded(value, upper: float) -> float:
    try:
        n = float(value)
        return min(max(n, 0.0), upper) if math.isfinite(n) else 0.0
    except (TypeError, ValueError):
        return 0.0


def research_priority(candidate: dict) -> float:
    """Unvalidated acquisition heuristic using ONLY already-present PIT fields.

    Different coefficients from the fixed Champion ranking are intentional:
    comparison with the original priority is a prospective research question,
    not a claim of predictive alpha.
    """
    rv = _bounded(candidate.get("rel_volume"), 8.0)
    dv = _bounded(candidate.get("dollar_volume"), 1e12)
    novelty = _bounded((candidate.get("features") or {}).get("novelty_max"), 2.0)
    news = min(len(candidate.get("news") or []), 5)
    return round(
        1.4 * math.log1p(rv) + 0.55 * math.log1p(dv / 1e7)
        + 0.45 * novelty + 0.12 * news,
        6,
    )


def effective_quote_cap(policy: dict, budget: dict, population: int) -> int:
    """Cap optional quote work using actual headroom and a fixed policy ceiling.

    A malformed or unknown budget means 0 optional probes. No paid tiers are
    activated, and the Champion's separate 8-probe budget is untouched.
    """
    try:
        ceiling = int(policy["max_quote_probes"])
        limit = float(budget["monthly_api_budget_usd"])
        projected = float(budget["projected_month_usd"])
        spent = float(budget["month_usd"])
        if not all(math.isfinite(x) for x in (limit, projected, spent)):
            return 0
        headroom = limit - max(projected, spent)
        reserve = float(policy.get("min_monthly_headroom_usd", 3.0))
        if not math.isfinite(reserve) or reserve < 0 or headroom < reserve:
            return 0
        tier = 24 if headroom >= 6.0 else 16 if headroom >= 4.5 else 12
        return max(0, min(population, ceiling, tier))
    except (KeyError, TypeError, ValueError, OverflowError):
        return 0


def select_probe_cohort(candidates: list[dict], champion_decisions: dict,
                        *, day: str, cap: int, audit_cap: int) -> tuple[list[dict], dict]:
    """Choose <=cap unique quotes; reuse all observable Champion probes first.

    Rank-control and challenger allocations alternate deterministically, with
    disjoint candidate identities. Audit picks from the unselected remainder
    using a day-seeded SHA256 ordering (not an alpha or random-outcome claim).
    """
    cap = max(0, min(int(cap), len(candidates)))
    audit_cap = max(0, min(int(audit_cap), cap))
    by_ticker = {str(c.get("ticker")): c for c in candidates}
    observed = {
        t for t, d in champion_decisions.items()
        if t in by_ticker and d.get("status") in ("pass", "unknown", "reject")
        and d.get("reason") not in (
            "haiku_budget", "quote_probe_budget", "time_budget",
            "acquisition_gate_error"
        )
    }
    cached = [by_ticker[t] for t in sorted(observed)[:cap]]
    selected = list(cached)
    source = {str(c["ticker"]): "champion_cached" for c in selected}
    remaining = [c for c in candidates if str(c.get("ticker")) not in source]
    audit_n = min(audit_cap, max(0, cap - len(selected)),
                  max(0, cap // 4))
    ranking_slots = max(0, cap - len(selected) - audit_n)

    control = sorted(remaining, key=lambda c: (-ef.priority(c), str(c.get("ticker"))))
    challenger = sorted(remaining, key=lambda c: (-research_priority(c), str(c.get("ticker"))))
    arms = [("legacy_rank", control), ("cheap_rank_v1", challenger)]
    # Vary first arm by day, not by market outcome.
    first_arm = int(hashlib.sha256(day.encode()).hexdigest()[-1], 16) % 2
    for i in range(ranking_slots):
        label, ranked = arms[(i + first_arm) % 2]
        item = next((c for c in ranked if str(c.get("ticker")) not in source), None)
        if item is None:
            break
        t = str(item["ticker"])
        selected.append(item)
        source[t] = label

    left = [c for c in remaining if str(c.get("ticker")) not in source]
    random_order = sorted(
        left,
        key=lambda c: (
            hashlib.sha256((day + ":audit-v1:" + str(c.get("ticker"))).encode()).hexdigest(),
            str(c.get("ticker")),
        ),
    )
    for c in random_order[:audit_n]:
        t = str(c["ticker"])
        selected.append(c)
        source[t] = "random_audit"

    # Fill unused capacity only when a sparse cohort exhausts an arm.
    for c in control:
        if len(selected) >= cap:
            break
        t = str(c["ticker"])
        if t not in source:
            selected.append(c)
            source[t] = "legacy_backfill"
    return selected, source

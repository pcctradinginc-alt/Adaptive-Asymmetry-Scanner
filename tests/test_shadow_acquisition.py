"""Cost-aware shadow acquisition: deterministic, observational, no paid APIs."""
from __future__ import annotations

import copy

from modules import shadow_acquisition as sa
from modules import throughput_shadow as ts


def _c(i, *, rv=None):
    return {
        "ticker": f"T{i:03d}", "current_price": 100.0,
        "rel_volume": float(rv if rv is not None else 1 + i / 10),
        "dollar_volume": 1e8 * (1 + (i % 7)),
        "news": ["time-stamped snapshot"] * (i % 4),
        "features": {"novelty_max": 0.1 * (i % 9)},
    }


def _budget(projected=1.85):
    return {"level": "OK", "soft_pressure": False,
            "monthly_api_budget_usd": 10.0,
            "projected_month_usd": projected, "month_usd": 1.5}


def test_cheap_rank_finite_and_uses_no_future_labels():
    c = _c(2)
    assert sa.research_priority(c) > 0
    for bad in (float("nan"), float("inf"), "garbled", None):
        x = {**c, "rel_volume": bad, "dollar_volume": bad,
             "features": {"novelty_max": bad}}
        assert sa.research_priority(x) >= 0
    assert "forward_return" not in sa.research_priority.__code__.co_consts


def test_budget_tiers_never_raise_champion_limits():
    p = ts.settings()
    assert sa.effective_quote_cap(p, _budget(1.85), 200) == 24
    assert sa.effective_quote_cap(p, _budget(5.4), 200) == 16
    assert sa.effective_quote_cap(p, _budget(5.8), 200) == 12
    assert sa.effective_quote_cap(p, _budget(6.9), 200) == 12
    assert sa.effective_quote_cap(p, _budget(7.1), 200) == 0
    assert sa.effective_quote_cap(p, {}, 200) == 0
    assert sa.effective_quote_cap(p, _budget(1.85), 5) == 5
    assert sa.effective_quote_cap({**p, "max_quote_probes": 10},
                                  _budget(1.85), 200) == 10


def test_probe_selection_deterministic_diversified_and_champion_cached():
    candidates = [_c(i) for i in range(100)]
    immutable = copy.deepcopy(candidates)
    champion = {c["ticker"]: {"status": "pass", "reason": "quoted_economics"}
                for c in candidates[:8]}
    first, a = sa.select_probe_cohort(
        candidates, champion, day="2026-11-23", cap=24, audit_cap=6)
    reversed_pool, b = sa.select_probe_cohort(
        list(reversed(candidates)), champion, day="2026-11-23",
        cap=24, audit_cap=6)
    assert len(first) == 24 == len({c["ticker"] for c in first})
    assert {c["ticker"] for c in candidates[:8]} <= set(a)
    assert sum(v == "champion_cached" for v in a.values()) == 8
    assert sum(v == "random_audit" for v in a.values()) == 6
    assert sum(v == "legacy_rank" for v in a.values()) == 5
    assert sum(v == "cheap_rank_v1" for v in a.values()) == 5
    assert a == b
    assert {c["ticker"] for c in first} == {c["ticker"] for c in reversed_pool}
    assert candidates == immutable


def test_sparse_or_zero_pool_never_exceeds_cap():
    for n in (0, 1, 5, 10):
        cohort, origins = sa.select_probe_cohort(
            [_c(i) for i in range(n)], {}, day="2026-11-23",
            cap=24, audit_cap=6)
        assert len(cohort) == n
        assert len(origins) == n
    cohort, origins = sa.select_probe_cohort(
        [_c(i) for i in range(5)], {}, day="2026-11-23",
        cap=0, audit_cap=6)
    assert cohort == [] and origins == {}


class _Yes:
    failed_tickers = []

    def run(self, rows):
        return list(rows)


class _NeverAnalyze:
    def run(self, rows, deadline=None):
        raise AssertionError("Sonnet must not run when budget falls below reserve")


def test_budget_recheck_stops_sonnet_without_losing_ledger():
    candidates = [_c(i) for i in range(40)]
    before = copy.deepcopy(candidates)
    budget_checks = []
    notes = []

    def budget_fn():
        budget_checks.append(1)
        return _budget(1.85 if len(budget_checks) <= 2 else 9.0)

    result = ts.run(
        candidates, {}, day="2026-11-23",
        probe=lambda c: {"status": "pass", "reason": "quoted_economics",
                         "immediate_friction": 0.05},
        note=lambda ticker, **kw: notes.append((ticker, kw)),
        budget_fn=budget_fn, allow_llm=True,
        prescreener_factory=_Yes, analyzer_factory=_NeverAnalyze)
    assert len(budget_checks) >= 3
    assert result["status"] == "BUDGET_LIMIT"
    assert result["sonnet_inputs"] == 0
    assert result["ledger_noted"] == 40
    assert all(set(v) == {"throughput_shadow"} for _, v in notes)
    assert all("direction" not in v for _, v in notes)
    assert candidates == before

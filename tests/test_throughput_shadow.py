"""Throughput shadow: strict caps, no Champion influence, no live APIs."""
from __future__ import annotations

import copy
import inspect
import time
from pathlib import Path

from modules import economics_first as ef
from modules import throughput_shadow as ts


def _c(i):
    return {"ticker": "T%03d" % i, "current_price": 100.0,
            "rel_volume": 1 + i / 10, "dollar_volume": 1e8,
            "news": ["Catalyst %s" % i], "features": {"novelty_max": 0.2}}


def _budget():
    return {"level": "OK", "soft_pressure": False,
            "monthly_api_budget_usd": 10.0, "projected_month_usd": 2.0, "month_usd": 1.0}


class _Prescreener:
    called = []

    def __init__(self):
        self.failed_tickers = []

    def run(self, items):
        self.called.append([x["ticker"] for x in items])
        # Only research candidates are passed; mutate them to test isolation.
        for x in items:
            x["prescreen_direction"] = "BULLISH"
            x["prescreen_reason"] = "shadow-only"
        return list(items)


class _Analyzer:
    called = []

    def run(self, items, deadline=None):
        self.called.append([x["ticker"] for x in items])
        return [{**x, "deep_analysis": {"direction": "BULLISH", "impact": 6, "surprise": 5}}
                for x in items]


def test_policy_caps_and_fail_closed(tmp_path):
    p = ts.settings()
    assert p["enabled"] and p["max_quote_probes"] == 24
    f = tmp_path / "test.yaml"
    f.write_text("enabled: true\nmax_quote_probes: 25\n")
    assert ts.settings(f)["enabled"] is False
    f.write_text("enabled: yes\n")
    assert ts.settings(f)["enabled"] is True
    f.write_text("enabled: potato\n")
    assert ts.settings(f)["enabled"] is False
    f.write_text("invalid_key: true\n")
    assert ts.settings(f)["enabled"] is False


def test_budget_fail_closed():
    p = ts.settings()
    assert ts.budget_ok(_budget(), p)
    for bad in (
        {},
        {"level": "WARN", **_budget()},
        {**_budget(), "level": "WARN"},
        {**_budget(), "projected_month_usd": 9},
        {**_budget(), "soft_pressure": True},
        {**_budget(), "month_usd": float("nan")},
    ):
        if bad.get("level") == "OK" and bad.get("projected_month_usd") == 2.0 and bad.get("month_usd") == 1.0 and not bad.get("soft_pressure"):
            continue
        assert not ts.budget_ok(bad, p)
    now = time.monotonic()
    assert ts.deadline_for(now - 600, 600, p, now=now) <= now - p["min_reserve_s"]


def test_research_24_total_probes_only_new_llm_and_namespaced_notes():
    candidates = [_c(i) for i in range(50)]
    untouched = copy.deepcopy(candidates)
    calls = []
    # Actual Champion acquisition fixture; no network calls.
    _, champion_decisions, _ = ef.select_before_haiku(
        candidates, day="2026-11-10",
        probe=lambda c: {"status": "pass", "reason": "quoted_economics",
                         "immediate_friction": 0.05},
        policy={**ef.DEFAULTS, "mode": "active", "max_quote_probes": 8,
                "audit_quote_probes": 2, "max_haiku_candidates": 8})
    notes = []
    _Prescreener.called.clear()
    _Analyzer.called.clear()

    def probe(c):
        calls.append(c["ticker"])
        return {"status": "pass", "reason": "quoted_economics", "immediate_friction": 0.04}

    r = ts.run(candidates, champion_decisions, day="2026-11-10",
               probe=probe, note=lambda ticker, **kw: notes.append((ticker, kw)),
               policy=ts.settings(), economics_policy=ef.DEFAULTS,
               budget_fn=_budget, allow_llm=True,
               prescreener_factory=_Prescreener, analyzer_factory=_Analyzer)
    assert r["status"] == "OK"
    assert r["probed"] == 24
    assert r["cached_champion_probes"] > 0
    assert r["new_probes"] == len(calls)
    assert r["new_probes"] <= 24
    assert r["quota_selected"] <= 12 and r["sonnet_inputs"] <= 6
    assert r["additional_haiku_inputs"] > 0
    assert r["ledger_noted"] == 50 and len(notes) == 50
    assert candidates == untouched
    original_champion_names = {t for t, d in champion_decisions.items() if d.get("selected_for_haiku")}
    assert original_champion_names.isdisjoint(set(sum(_Prescreener.called, [])))
    assert original_champion_names.isdisjoint(set(sum(_Analyzer.called, [])))
    assert all(set(fields) == {"throughput_shadow"} for _, fields in notes)
    assert all("direction" not in fields and "stage" not in fields for _, fields in notes)
    assert all("sonnet_direction" not in fields or fields["sonnet_direction"] == "BULLISH"
               for _, kw in notes for fields in [kw["throughput_shadow"]])


def test_guards_never_start_api_calls():
    candidates = [_c(i) for i in range(10)]
    seen = []
    fail = lambda c: seen.append(c["ticker"])
    arguments = dict(day="2026-11-10", probe=fail, note=lambda *a, **k: seen.append("note"))
    no_key = ts.run(candidates, {}, allow_llm=False, budget_fn=_budget, **arguments)
    assert no_key["reason"] == "no_anthropic_key"
    no_money = ts.run(candidates, {}, allow_llm=True,
                      budget_fn=lambda: {**_budget(), "projected_month_usd": 10}, **arguments)
    assert no_money["reason"] == "budget_guard"
    no_time = ts.run(candidates, {}, allow_llm=True, budget_fn=_budget,
                     run_deadline=time.monotonic() - 120,
                     finalize_reserve_s=120, **arguments)
    assert no_time["reason"] == "time_guard"
    assert seen == []


def test_pipeline_runs_only_at_finalization_and_protocol_unmodified():
    src = (Path(__file__).resolve().parents[1] / "pipeline.py").read_text()
    assert src.index("_ef_start = len(candidates)") < src.index("select_before_haiku(")
    assert src.index("_throughput_ref[0] = (") > src.index("select_before_haiku(")
    assert src.index("_llm_census_finalize()\n        _throughput_finalize()") > src.index("def save_stats_snapshot()")
    assert 'note=candidate_ledger.note' in src
    from modules import champion_protocol as cp
    assert cp.decision_config_sha() == cp.load_protocol()["decision_config_sha256"]

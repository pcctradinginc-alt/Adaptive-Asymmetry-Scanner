"""Bounded, research-only acquisition challenger for quote/LLM coverage.

Runs exclusively AFTER the Champion has finished making its decisions.
No module in the production decision path imports the results. All observations
are nested under 'throughput_shadow' in the existing candidate ledger; an
underlying forward return is NOT an executable options outcome.

The expanded quote cohort contains at most 24 probes TOTAL, including cached
Champion probes and 6 deterministic audit draws. Only additional names receive
the research LLM calls; existing Champion prescreens are never re-run.
"""
from __future__ import annotations

import copy
import logging
import math
import os
import time
from pathlib import Path

import yaml

from modules import economics_first as ef

log = logging.getLogger(__name__)
CONFIG = Path(__file__).resolve().parent.parent / "config" / "throughput_shadow.yaml"
DEFAULTS = {
    "enabled": True,
    "max_quote_probes": 24,
    "audit_quote_probes": 6,
    "max_haiku_candidates": 12,
    "max_sonnet_candidates": 6,
    "sonnet_audit_candidates": 2,
    "budget_s": 240,
    "min_reserve_s": 120,
    "min_start_seconds": 40,
    "min_monthly_headroom_usd": 3.0,
}
# Hard ceilings even if YAML is mistakenly edited.
CEILINGS = {"max_quote_probes": 24, "audit_quote_probes": 6,
            "max_haiku_candidates": 12, "max_sonnet_candidates": 6,
            "sonnet_audit_candidates": 2, "budget_s": 240}


def settings(path: Path = CONFIG) -> dict:
    """Malformed or unavailable policy means NO research API calls."""
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Policy must be a mapping")
        extra = set(raw) - set(DEFAULTS)
        if extra:
            raise ValueError("Unknown policy keys: " + ", ".join(sorted(extra)))
        p = {**DEFAULTS, **raw}
        if not isinstance(p["enabled"], bool):
            raise ValueError("enabled must be boolean")
        for k, cap in CEILINGS.items():
            if type(p[k]) is not int or not 0 <= p[k] <= cap:
                raise ValueError("Invalid cap: " + k)
        for k in ("min_reserve_s", "min_start_seconds"):
            if type(p[k]) is not int or not 0 <= p[k] <= 600:
                raise ValueError("Invalid time: " + k)
        v = p["min_monthly_headroom_usd"]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            raise ValueError("Invalid monthly headroom")
        if p["audit_quote_probes"] > p["max_quote_probes"]:
            raise ValueError("Audit quote slots exceed quote cap")
        if p["sonnet_audit_candidates"] > p["max_sonnet_candidates"]:
            raise ValueError("Audit Sonnet slots exceed Sonnet cap")
        return p
    except (OSError, yaml.YAMLError, ValueError) as exc:
        log.error("throughput_shadow policy failed closed: %s", exc)
        return {**DEFAULTS, "enabled": False, "config_error": str(exc)[:160]}


def deadline_for(run_deadline: float | None, finalize_reserve_s: float, p: dict,
                 now: float | None = None) -> float:
    """Never consume the established scanner finalization reserve."""
    now = time.monotonic() if now is None else now
    hard = (run_deadline + finalize_reserve_s - p["min_reserve_s"]
            if run_deadline is not None else now + p["budget_s"])
    return min(now + p["budget_s"], hard)


def budget_ok(status: dict, p: dict) -> bool:
    """Fail closed on missing/unknown cost information or monthly pressure."""
    if not isinstance(status, dict) or status.get("level") != "OK" or status.get("soft_pressure"):
        return False
    try:
        budget = float(status["monthly_api_budget_usd"])
        projected = float(status["projected_month_usd"])
        spent = float(status["month_usd"])
        return (all(math.isfinite(x) for x in (budget, projected, spent))
                and budget - max(projected, spent) >= p["min_monthly_headroom_usd"])
    except (TypeError, ValueError, KeyError):
        return False


def run(raw_candidates: list[dict], champion_decisions: dict, *, day: str,
        probe, note, policy: dict | None = None, economics_policy: dict | None = None,
        run_deadline: float | None = None, finalize_reserve_s: float = 600,
        budget_fn=None, prescreener_factory=None, analyzer_factory=None,
        allow_llm: bool | None = None) -> dict:
    """Research only. Never returns a production candidate, trade or position.

    All passed candidate data are deep-copied. Runs only after the Champion
    decisions are complete. Failures/timeouts are explicit, never NO_SIGNAL.
    """
    p = settings() if policy is None else policy
    summary = {"status": "SKIPPED", "mode": "research_only", "day": day,
               "population": len(raw_candidates), "production_influence": "NONE"}
    if not p.get("enabled"):
        return {**summary, "reason": "disabled_or_invalid_config"}
    if not raw_candidates:
        return {**summary, "reason": "empty_cohort"}
    if allow_llm is None:
        allow_llm = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if not allow_llm:
        return {**summary, "reason": "no_anthropic_key"}
    if budget_fn is None:
        from modules import cost_telemetry
        budget_fn = cost_telemetry.budget_status
    try:
        budget = budget_fn()
    except Exception as exc:
        return {**summary, "reason": "budget_unavailable", "error": str(exc)[:160]}
    if not budget_ok(budget, p):
        return {**summary, "reason": "budget_guard"}
    stop = deadline_for(run_deadline, finalize_reserve_s, p)
    if stop - time.monotonic() < p["min_start_seconds"]:
        return {**summary, "reason": "time_guard"}

    # Isolate all LLM-added keys from original production candidate objects.
    pool = [copy.deepcopy(c) for c in raw_candidates]
    production_haiku = {t for t, d in champion_decisions.items()
                        if d.get("selected_for_haiku")}
    cached = 0
    new = 0

    def shadow_probe(candidate):
        nonlocal cached, new
        prev = champion_decisions.get(candidate["ticker"]) or {}
        if prev.get("status") in ("pass", "unknown", "reject") and prev.get("reason") not in (
                "haiku_budget", "quote_probe_budget", "time_budget", "acquisition_gate_error"):
            cached += 1
            return {k: v for k, v in prev.items() if k not in ("selected_for_haiku", "audit", "priority")}
        new += 1
        return probe(candidate)

    ep = {**ef.settings(), **(economics_policy or {})}
    ep.update(mode="active", max_quote_probes=p["max_quote_probes"],
              audit_quote_probes=p["audit_quote_probes"],
              max_haiku_candidates=p["max_haiku_candidates"],
              max_sonnet_candidates=p["max_sonnet_candidates"],
              sonnet_audit_candidates=p["sonnet_audit_candidates"])
    try:
        selected, decisions, counts = ef.select_before_haiku(
            pool, day=day, probe=shadow_probe, policy=ep, deadline=stop)
    except Exception as exc:
        log.warning("throughput_shadow quote selection failed: %s", exc)
        return {**summary, "status": "ERROR", "reason": "quote_selection", "error": str(exc)[:160]}

    rows = {t: {"policy": "throughput-shadow-v1", "day": day,
                "quote_status": d["status"], "quote_reason": d.get("reason"),
                "quote_audit": bool(d.get("audit")), "quoted_friction": d.get("immediate_friction"),
                "haiku": "not_selected", "sonnet": "not_selected"}
            for t, d in decisions.items()}
    extra = [c for c in selected if c["ticker"] not in production_haiku]
    for c in selected:
        if c["ticker"] in production_haiku:
            rows[c["ticker"]]["haiku"] = "champion_already_evaluated"

    summary.update(status="OK", probed=counts.get("probed", 0),
                   cached_champion_probes=cached, new_probes=new,
                   quoted_pass=counts.get("quoted_pass", 0),
                   quota_selected=len(selected), additional_haiku_inputs=len(extra),
                   haiku_yes=0, sonnet_inputs=0, sonnet_analyzed=0)

    try:
        if time.monotonic() >= stop:
            summary["status"] = "TIME_LIMIT"
        elif extra:
            if prescreener_factory is None:
                from modules.prescreener import Prescreener
                prescreener_factory = Prescreener
            ps = prescreener_factory()
            yes = ps.run(extra) or []
            passed = {c["ticker"] for c in yes}
            failed = set(getattr(ps, "failed_tickers", []) or [])
            for c in extra:
                t = c["ticker"]
                rows[t]["haiku"] = ("api_error" if t in failed else
                                    "yes" if t in passed else "no")
            summary["haiku_yes"] = len(passed)
            # Research Sonnet is NEVER run on production names or Haiku NO.
            if yes and time.monotonic() < stop:
                deep, sonnet_skips = ef.select_before_sonnet(yes, day=day, policy=ep)
                summary["sonnet_inputs"] = len(deep)
                for c in yes:
                    if c["ticker"] in sonnet_skips:
                        rows[c["ticker"]]["sonnet"] = "budget_deferred"
                if deep:
                    if analyzer_factory is None:
                        from modules.deep_analysis import DeepAnalysis
                        analyzer_factory = DeepAnalysis
                    da = analyzer_factory()
                    results = da.run(deep, deadline=stop) or []
                    analyzed = {a["ticker"] for a in results}
                    timed = set(getattr(da, "skipped_for_time", []) or [])
                    for c in deep:
                        t = c["ticker"]
                        rows[t]["sonnet"] = ("analyzed" if t in analyzed else
                                             "time_deferred" if t in timed else "no_valid_analysis")
                    for a in results:
                        v = a.get("deep_analysis") or {}
                        rows[a["ticker"]]["sonnet_direction"] = v.get("direction")
                        rows[a["ticker"]]["sonnet_impact"] = v.get("impact")
                        rows[a["ticker"]]["sonnet_surprise"] = v.get("surprise")
                    summary["sonnet_analyzed"] = len(analyzed)
            elif yes:
                for c in yes:
                    rows[c["ticker"]]["sonnet"] = "time_deferred"
    except Exception as exc:
        log.exception("throughput_shadow research stage failed")
        summary.update(status="PARTIAL_ERROR", error=f"{type(exc).__name__}: {str(exc)[:160]}")
    finally:
        # Only a namespaced observational feature, never a production direction,
        # trade status, rejection or recommendation. Continue on note failures.
        recorded = 0
        for ticker, observation in rows.items():
            try:
                note(ticker, throughput_shadow=observation)
                recorded += 1
            except Exception as exc:
                log.warning("throughput_shadow ledger note %s: %s", ticker, exc)
        summary["ledger_noted"] = recorded
    return summary

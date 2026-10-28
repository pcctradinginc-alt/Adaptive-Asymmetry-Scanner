"""Quote-based economics BEFORE both language-model stages.

No LLM. Only point-in-time candidate fields and currently executable option
quotes. Bounded probes and deterministic holdout sampling avoid an unbounded
chain/API workload. This is an acquisition policy, NOT a validated alpha signal.

Every input ticker receives a decision row; upstream Candidate Ledger continues
to collect underlying forward prices for ALL tickers, including budget deferrals.
Unprobed symbols do not have real-option outcomes: do not label those as
economically unviable or pretend they were priced.
"""
from __future__ import annotations

import hashlib
import logging
import math
import time
from pathlib import Path

log = logging.getLogger(__name__)
CFG_PATH = Path(__file__).resolve().parent.parent / "config" / "economics_first.yaml"

DEFAULTS = {
    "mode": "active", "max_quote_probes": 24, "audit_quote_probes": 2,
    "max_haiku_candidates": 12, "max_sonnet_candidates": 3,
    "sonnet_audit_candidates": 1, "max_immediate_friction": 0.30,
    "probe_dte_min": 150, "probe_dte_max": 365,
}


def settings(path: Path | None = None) -> dict:
    import yaml
    try:
        raw = yaml.safe_load((path or CFG_PATH).read_text(encoding="utf-8")) or {}
        return {**DEFAULTS, **(raw if isinstance(raw, dict) else {})}
    except Exception as exc:
        # Configuration failure is observable; don't allow silently unlimited calls.
        log.error("economics_first configuration unavailable: %s", exc)
        return dict(DEFAULTS)


def _num(value, fallback=0.0) -> float:
    try:
        n = float(value)
        return n if math.isfinite(n) else fallback
    except (TypeError, ValueError):
        return fallback


def priority(candidate: dict) -> float:
    """Cheap, pre-registered heuristic for probe allocation, NOT an alpha score."""
    rv = min(max(_num(candidate.get("rel_volume")), 0.0), 5.0)
    news = min(len(candidate.get("news") or []), 5)
    dv = max(_num(candidate.get("dollar_volume")), 0.0)
    novelty = _num((candidate.get("features") or {}).get("novelty_max"), 0.0)
    return round(2.0 * math.log1p(rv) + 0.25 * news +
                 0.10 * math.log1p(dv / 1e7) + novelty, 6)


def _order(candidates: list[dict]) -> list[dict]:
    return sorted(candidates, key=lambda c: (-priority(c), str(c.get("ticker", ""))))


def _sample(candidates: list[dict], day: str, count: int, purpose: str) -> list[dict]:
    """Reproducible unbiased audit slots; cannot depend on future outcomes."""
    return sorted(
        candidates,
        key=lambda c: hashlib.sha256(
            (day + ":" + purpose + ":" + str(c.get("ticker", ""))).encode()
        ).hexdigest(),
    )[:max(count, 0)]


def probe_candidate(candidate: dict, *, designer, gates, policy: dict) -> dict:
    """Bounded directional-neutral quote viability before the expensive LLMs.

    A bad bullish CALL is NOT proof a bearish PUT is untradeable. Check the
    other direction ONLY if the first direction is conclusively unviable.
    An unavailable bearish quote is UNKNOWN; never exclude an unpriced
    bearish signal as economically rejected. Max two directional checks
    (spread fallback can perform a further bounded chain lookup).
    """
    ticker = candidate.get("ticker")
    spot = _num(candidate.get("current_price"))
    if spot <= 0:
        return {"status": "unknown", "reason": "spot_unavailable"}
    try:
        if gates.has_upcoming_earnings(ticker):
            return {"status": "reject", "reason": "earnings_gate"}
    except Exception as exc:
        log.warning("economics_first %s earnings check unknown: %s", ticker, exc)

    def _direction_quote(direction: str, iv_rank: float) -> dict:
        strategy = designer._select_strategy(ticker, direction, iv_rank)
        option = designer._find_option_for_dte(
            ticker, strategy, spot, int(policy["probe_dte_min"]),
            int(policy["probe_dte_max"]),
        )
        if not option and strategy in ("BULL_CALL_SPREAD", "BEAR_PUT_SPREAD"):
            strategy = "LONG_CALL" if direction == "BULLISH" else "LONG_PUT"
            option = designer._find_option_for_dte(
                ticker, strategy, spot, int(policy["probe_dte_min"]),
                int(policy["probe_dte_max"]),
            )
        if not option:
            return {"status": "unknown", "reason": "option_quote_unavailable",
                    "direction_checked": direction}
        long_bid = _num(option.get("bid"), -1)
        long_ask = _num(option.get("ask"), -1)
        if long_ask <= 0 or long_bid < 0 or long_bid > long_ask:
            return {"status": "unknown", "reason": "invalid_quote",
                    "direction_checked": direction}
        short = option.get("spread_leg") or {}
        is_spread = "SPREAD" in strategy and bool(short)
        if is_spread:
            short_bid, short_ask = _num(short.get("bid"), -1), _num(short.get("ask"), -1)
            debit = _num(option.get("net_debit"))
            if short_bid <= 0 or short_ask <= 0 or short_bid > short_ask or debit <= 0:
                return {"status": "unknown", "reason": "invalid_combo_quote",
                        "direction_checked": direction}
            actual_debit = long_ask - short_bid
            if actual_debit <= 0 or abs(actual_debit-debit) > max(0.02, 0.05*debit):
                return {"status": "unknown", "reason": "inconsistent_combo_debit",
                        "direction_checked": direction}
            friction = max(0.0, (actual_debit-(long_bid-short_ask))/actual_debit)
        else:
            debit = long_ask
            friction = (long_ask-long_bid)/long_ask
        out = {"status": "pass", "reason": "quoted_economics",
               "direction_checked": direction,
               "strategy": strategy, "dte": option.get("dte"),
               "entry_debit": round(debit, 4),
               "immediate_friction": round(friction, 5),
               "iv_rank": round(_num(iv_rank), 2)}
        if friction > float(policy["max_immediate_friction"]):
            out.update(status="reject", reason="excessive_quote_friction")
        return out

    try:
        iv_rank = designer._get_iv_rank(ticker)
        if iv_rank is None:
            return {"status": "unknown", "reason": "iv_rank_unavailable"}
        bullish = _direction_quote("BULLISH", iv_rank)
        if bullish["status"] != "reject":
            return bullish
        # A bearish candidate must never be excluded solely on call costs.
        bearish = _direction_quote("BEARISH", iv_rank)
        if bearish["status"] == "reject":
            bullish["directions_checked"] = ["BULLISH", "BEARISH"]
            return bullish
        bearish["bullish_probe_status"] = bullish["status"]
        bearish["directions_checked"] = ["BULLISH", "BEARISH"]
        return bearish
    except Exception as exc:
        log.warning("economics_first %s probe unavailable: %s", ticker, exc)
        return {"status": "unknown", "reason": "probe_error"}

def select_before_haiku(candidates: list[dict], *, day: str, probe,
                        policy: dict | None = None, deadline: float | None = None) -> tuple[list, dict, dict]:
    """Return shortlist, per-ticker decisions and summary. Does NOT touch ledger."""
    pol = {**DEFAULTS, **(policy or {})}
    mode = str(pol.get("mode", "active"))
    if mode == "off" or not candidates:
        return list(candidates), {}, {"mode": mode, "input": len(candidates), "haiku": len(candidates)}
    ranked = _order(candidates)
    cap = max(0, int(pol["max_quote_probes"]))
    sample_n = min(cap, max(0, int(pol["audit_quote_probes"])))
    top = ranked[:max(cap - sample_n, 0)]
    outside = ranked[len(top):]
    audit = _sample(outside, day, min(sample_n, len(outside)), "probe")
    probes = top + [c for c in audit if c not in top]
    # If pool is smaller than cap, use spare slots without violating budget.
    remaining = [c for c in ranked if c not in probes]
    if len(probes) < cap:
        probes += remaining[:cap-len(probes)]
    decisions = {
        c["ticker"]: {"status": "deferred", "reason": "quote_probe_budget",
                      "priority": priority(c), "audit": False}
        for c in candidates
    }
    for c in probes:
        ticker = c["ticker"]
        if deadline is not None and time.monotonic() >= deadline:
            decisions[ticker] = {**decisions[ticker], "reason": "time_budget"}
            continue
        try:
            verdict = probe(c) or {"status": "unknown", "reason": "empty_probe"}
        except Exception as exc:
            log.warning("economics_first probe exception for %s: %s", ticker, exc)
            verdict = {"status": "unknown", "reason": "probe_error"}
        if verdict.get("status") not in ("pass", "unknown", "reject"):
            verdict = {"status": "unknown", "reason": "invalid_probe_status"}
        decisions[ticker] = {**decisions[ticker], **verdict,
                             "audit": c in audit}
    qualified = [c for c in ranked if decisions[c["ticker"]]["status"] in ("pass", "unknown")]
    # Audit slots must be included within the cap; NEVER add them on top.
    haiku_cap = max(0, int(pol["max_haiku_candidates"]))
    audit_qualified = [c for c in qualified if c in audit]
    audit_haiku = audit_qualified[:min(haiku_cap, len(audit_qualified))]
    normal = [c for c in qualified if c not in audit_haiku]
    # Quote-confirmed first, then cheap priority. Missing quotes are UNKNOWN.
    normal = sorted(normal, key=lambda c: (
        0 if decisions[c["ticker"]]["status"] == "pass" else 1,
        _num(decisions[c["ticker"]].get("immediate_friction"), 1.0),
        -priority(c), c["ticker"]))
    selected = normal[:max(0, haiku_cap-len(audit_haiku))] + audit_haiku
    selected_ids = {c["ticker"] for c in selected}
    for c in qualified:
        t = c["ticker"]
        if t not in selected_ids:
            decisions[t].update(status="deferred", reason="haiku_budget")
    for t in selected_ids:
        decisions[t]["selected_for_haiku"] = True
    if mode == "shadow":
        return list(candidates), decisions, {
            "mode": mode, "input": len(candidates), "probed": len(probes),
            "would_haiku": len(selected), "haiku": len(candidates)}
    summary = {"mode": mode, "input": len(candidates),
               "probed": sum(d.get("reason") not in ("time_budget", "quote_probe_budget")
                             for d in decisions.values()),
               "quoted_pass": sum(d["status"] == "pass" for d in decisions.values()),
               "unknown_selected": sum(decisions[c["ticker"]].get("status") == "unknown" for c in selected),
               "haiku": len(selected), "audit_haiku": len(audit_haiku),
               "deferred": len(candidates)-len(selected)}
    return selected, decisions, summary


def select_before_sonnet(candidates: list[dict], *, day: str,
                         policy: dict | None = None) -> tuple[list, dict]:
    pol = {**DEFAULTS, **(policy or {})}
    if pol.get("mode") != "active":
        return list(candidates), {}
    ranked = _order(candidates)
    cap = max(0, int(pol["max_sonnet_candidates"]))
    if len(ranked) <= cap:
        return ranked, {}
    audit_n = min(cap, int(pol.get("sonnet_audit_candidates", 0)))
    normal = ranked[:max(0, cap - audit_n)]
    audit = _sample(ranked[len(normal):], day, audit_n, "sonnet")
    selected = normal + audit
    selected_ids = {c["ticker"] for c in selected}
    skipped = {c["ticker"]: "sonnet_budget" for c in ranked if c["ticker"] not in selected_ids}
    return selected, skipped

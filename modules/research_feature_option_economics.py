"""FD3 Phase 4: quote-verified, strictly paired BASE vs CHALLENGER option economics.

The source of actual paper entry/exit bid/ask is the EXISTING
autonomous_alpha.option_link verified journal. No invented marks, no mid-price
returns, no real orders, no model training, no changed option rules/capture.
Pairs with one missing arm are DATA_GAP, NEVER treated as zero returns.
Only a separate owner-approved global-alpha-funded trial could confirm edge.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

from modules.autonomous_alpha import option_link as link
from modules import research_feature_discovery as fd

ROOT = Path("outputs/research/feature_discovery_v3")
STRUCTURES = ("LONG_CALL", "BULL_CALL_SPREAD")
MIN_INDEPENDENT_WINDOWS = 8


def _num(v):
    try:
        x = float(v)
    except (ValueError, TypeError):
        return None
    return x if math.isfinite(x) else None


def _time(v):
    try:
        t = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        return t.astimezone(timezone.utc) if t.tzinfo else None
    except (TypeError, ValueError):
        return None


def trade_return(decision: dict, outcome: dict, *, fee: float,
                 slippage_fraction: float = 0.0) -> dict | None:
    """1 paper contract: long ask/short bid in, long bid/short ask out.

    Adverse slippage is an additional part of each quoted leg spread for
    BOTH entry and exit. Quote IDs and RTH quality are mandatory at both ends.
    """
    if (decision.get("decision") != "TRADE" or
            decision.get("evidence_class_at_entry") != "QUOTE_VERIFIED_PAPER" or
            outcome.get("evidence_class") != "QUOTE_VERIFIED_PAPER" or
            outcome.get("status") != "EXIT_OBSERVED"):
        return None
    for k in ("decision_id", "ticker", "structure", "signal_id"):
        if decision.get(k) != outcome.get(k):
            raise ValueError("FD3 option economic mismatch on " + k)
    started, ended = _time(decision.get("decided_at")), _time(outcome.get("recorded_at"))
    if started is None or ended is None or started >= ended:
        return None
    legs = decision.get("legs") or []
    exits = {q.get("occ_symbol"): q for q in outcome.get("exit_legs") or []}
    if len(legs) not in (1, 2) or len(exits) != len(legs):
        return None
    if decision.get("structure") == "LONG_CALL" and len(legs) != 1:
        return None
    if decision.get("structure") == "BULL_CALL_SPREAD" and len(legs) != 2:
        return None
    if not 0 <= slippage_fraction <= 1 or fee < 0:
        raise ValueError("Invalid cost policy")
    entry, liquidation, entry_worse, exit_worse = 0.0, 0.0, 0.0, 0.0
    audit = []
    for leg in legs:
        symbol, side = leg.get("occ_symbol"), leg.get("side")
        exit_leg = exits.get(symbol)
        if (not symbol or side not in ("long", "short") or exit_leg is None or
                not leg.get("quote_record_id") or not exit_leg.get("quote_record_id") or
                not leg.get("usable_rth") or
                leg["quote_record_id"] == exit_leg["quote_record_id"]):
            return None
        b, a, eb, ea = (_num(leg.get("bid")), _num(leg.get("ask")),
                        _num(exit_leg.get("bid")), _num(exit_leg.get("ask")))
        if any(x is None for x in (b, a, eb, ea)):
            return None
        if not (0 <= b <= a and 0 <= eb <= ea):
            return None
        qt_entry = _time(leg.get("quote_time"))
        qt_exit = _time(exit_leg.get("quote_time"))
        if qt_entry is None or qt_exit is None or qt_entry > started or qt_exit <= started:
            return None
        if date.fromisoformat(str(outcome.get("exit_day"))) < date.fromisoformat(str(decision.get("decision_day"))):
            return None
        spread_in, spread_out = a - b, ea - eb
        if side == "long":
            entry += a
            liquidation += eb
            entry_worse += a + spread_in * slippage_fraction
            exit_worse += max(0.0, eb - spread_out * slippage_fraction)
        else:
            entry -= b
            liquidation -= ea
            entry_worse -= max(0.0, b - spread_in * slippage_fraction)
            exit_worse -= ea + spread_out * slippage_fraction
        audit.append({"entry_record_id": leg["quote_record_id"],
                      "exit_record_id": exit_leg["quote_record_id"],
                      "occ_symbol": symbol, "side": side})
    if entry <= 0 or not math.isclose(entry, float(decision.get("net_debit_ask") or 0),
                                      abs_tol=1e-4, rel_tol=1e-5):
        return None
    n = len(legs)
    # Cash paid at entry includes entry fees; exit proceeds net of exit fees.
    cost = 100 * entry + n * fee
    pnl = (100 * liquidation - n * fee) - cost
    worse_cost = 100 * entry_worse + n * fee
    worse_pnl = (100 * exit_worse - n * fee) - worse_cost
    return {"net_roi": pnl / cost, "adverse_net_roi": worse_pnl / worse_cost,
            "net_pnl_usd": pnl, "max_loss_capital_usd": cost,
            "exit_day": outcome["exit_day"], "quoted_entry_debit": entry,
            "quoted_exit_liquidation": liquidation, "evidence": audit}


def _group_signals(signals: list[dict], digest: str, hid: str, *, today: date) -> dict:
    rows = defaultdict(dict)
    for s in signals:
        if (s.get("hypothesis_source") != "fd3_feature" or
                s.get("hypothesis_version") != digest or
                s.get("excluded_by_correction") or s.get("role") != "TOP"):
            continue
        ident = str(s.get("hypothesis_id") or "")
        for arm in ("BASE", "CHALLENGER"):
            if ident != f"FD3:{hid}:{arm}":
                continue
            if s.get("production_influence") != "NONE" or s.get("order_placed") is not False:
                raise ValueError("FD3 contaminated production signal")
            ref = s.get("prediction_ref") or {}
            if ref.get("fd3_arm") != arm or ref.get("fd3_hypothesis") != hid:
                raise ValueError("FD3 mismatched frozen prediction arm")
            t_reg, t_hyp = _time(s.get("registered_at")), _time(s.get("hypothesis_registered_at"))
            if t_reg is not None and t_reg.date() > today:
                continue
            if t_reg is None or t_hyp is None or t_hyp >= t_reg:
                raise ValueError("FD3 signal not registered prospectively")
            day = s["pit_data_date"]
            if arm in rows[day]:
                raise ValueError("Duplicate FD3 arm on same signal date")
            rows[day][arm] = s
    return rows


def evaluate(data: dict, cfg: dict, *, today: date, protocol_hash: str) -> dict:
    rule = cfg["rule"]
    if cfg.get("production_influence") != "NONE":
        raise ValueError("FD3 options require paper-only rule")
    decisions = {d["signal_id"] + "|" + d["structure"]: d
                 for d in data.get("decisions") or [] if not d.get("excluded_by_correction")}
    outcomes = {o["decision_id"]: o for o in data.get("outcomes") or []
                if not o.get("excluded_by_correction")}
    result = {"as_of": today.isoformat(), "status": "QUOTE_GAP",
              "protocol_sha256": protocol_hash,
              "method": "paired_alpha_option_link_bid_ask_one_contract_20d",
              "research_only": True, "orders_created": 0,
              "option_edge_confirmed": False, "global_alpha_ledger_slot_allocated": False,
              "hypotheses": {}}
    for h in fd.load_protocol()[0]["hypotheses"]:
        hid = h["id"]
        arms = _group_signals(data.get("signals") or [], protocol_hash, hid, today=today)
        by_structure = {}
        for st in STRUCTURES:
            matched, gaps = [], Counter()
            for day, pairs in sorted(arms.items()):
                if set(pairs) != {"BASE", "CHALLENGER"}:
                    gaps["MISSING_REGISTERED_ARM"] += 1
                    continue
                values = {}
                for arm, s in pairs.items():
                    d = decisions.get(s["signal_id"] + "|" + st)
                    if d is None:
                        gaps["DECISION_NOT_CAPTURED"] += 1
                        break
                    if (_time(d.get("decided_at")) is None or
                            _time(d["decided_at"]).date() > today or
                            _time(d["decided_at"]) <= _time(s["registered_at"])):
                        gaps["DECISION_NOT_AFTER_SIGNAL"] += 1
                        break
                    o = outcomes.get(d["decision_id"])
                    if o is None or _time(o.get("recorded_at")) is None or _time(o.get("recorded_at")).date() > today:
                        gaps["EXIT_PENDING_OR_MISSING"] += 1
                        break
                    x = trade_return(d, o,
                                     fee=float(rule["commission_usd_per_contract_per_side"]),
                                     slippage_fraction=float(rule["adverse_slippage_fraction_of_spread"]))
                    if x is None:
                        gaps["NO_BID_ASK_VERIFIED_EXIT"] += 1
                        break
                    values[arm] = x
                if len(values) != 2:
                    continue
                delta = values["CHALLENGER"]["net_roi"] - values["BASE"]["net_roi"]
                stress_delta = (values["CHALLENGER"]["adverse_net_roi"] -
                                values["BASE"]["adverse_net_roi"])
                matched.append({"date": day,
                                "window_end": max(v["exit_day"] for v in values.values()),
                                "base_net_roi": values["BASE"]["net_roi"],
                                "challenger_net_roi": values["CHALLENGER"]["net_roi"],
                                "net_roi_delta": delta, "adverse_roi_delta": stress_delta,
                                "quote_ids": {arm: x["evidence"] for arm, x in values.items()}})
            independent = fd.independent(matched, {"evaluation": {
                "min_calendar_days_between_forward_windows": 28}})
            n = len(independent)
            ds = np.array([x["net_roi_delta"] for x in independent], dtype=float)
            stressed = np.array([x["adverse_roi_delta"] for x in independent], dtype=float)
            ci = None
            if n:
                seed = int(protocol_hash[:8], 16) ^ (1 if st == "LONG_CALL" else 2)
                rng = np.random.default_rng(seed)
                bootstrap = rng.choice(ds, size=(2048, n), replace=True).mean(axis=1)
                ci = [float(q) for q in np.quantile(bootstrap, [.025, .975])]
            by_structure[st] = {
                "status": "NEED_MORE_PAIRED_RTH_QUOTES" if n < MIN_INDEPENDENT_WINDOWS
                          else "FORWARD_ECONOMIC_DIAGNOSTIC_REVIEW_ONLY",
                "all_registered_paired_dates": len(arms),
                "both_arms_quote_verified_dates": len(matched),
                "independent_quote_verified_windows": n,
                "required_independent_windows": MIN_INDEPENDENT_WINDOWS,
                "coverage_gaps": dict(gaps), "mean_net_roi_delta": float(ds.mean()) if n else None,
                "mean_adverse_net_roi_delta": float(stressed.mean()) if n else None,
                "bootstrap_95pct_ci": ci,
                "positive_delta_fraction": float((ds > 0).mean()) if n else None,
                "quote_provenance": [{"date": row["date"], "quote_ids": row["quote_ids"]}
                                     for row in independent[:25]],
                "paper_outcomes_only": True, "economically_positive_alpha_proven": False,
                "pre_registered_confirmatory_test_required": True,
            }
        result["hypotheses"][hid] = {"source": h["source"], "structures": by_structure}
    if any(s["independent_quote_verified_windows"] >= MIN_INDEPENDENT_WINDOWS
           for h in result["hypotheses"].values() for s in h["structures"].values()):
        result["status"] = "FORWARD_ECONOMIC_DIAGNOSTIC_REVIEW_ONLY"
    return result


def run(*, root: Path = ROOT, repository: Path = Path("."), now: datetime | None = None):
    now = now or datetime.now(timezone.utc)
    _, digest = fd.load_protocol()
    cfg = link.config(repository)
    all_data = link.read_all(repository)  # hash-checked, corrections applied
    report = evaluate(all_data, cfg, today=now.date(), protocol_hash=digest)
    root.mkdir(parents=True, exist_ok=True)
    (root / "options_economics.json").write_text(
        json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default=None)
    arg = ap.parse_args(argv)
    now = (datetime.fromisoformat(arg.as_of).replace(tzinfo=timezone.utc)
           if arg.as_of else datetime.now(timezone.utc))
    rep = run(now=now)
    print(json.dumps({"status": rep["status"], "n_hypotheses": len(rep["hypotheses"]),
                      "confirmed": rep["option_edge_confirmed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

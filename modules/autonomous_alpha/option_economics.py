"""Alpha-to-Options economics (Phase B): option strategy per hypothesis.

RESEARCH / PAPER ONLY. The result is a research decision, never an order, a
promotion or a production change. Inputs are exclusively the Phase A journals
(option_link) and the existing quote archive; nothing is re-quoted, re-selected
or optimized on known outcomes (the contract rule is frozen in Phase A).

* Execution: buy at ask, sell at bid (spreads: long leg ask / short leg bid on
  entry, long leg bid / short leg ask on exit), per-contract fee each side,
  plus an ADVERSE scenario that moves every fill against us by a fixed share
  of the leg's bid/ask spread. ADVERSE is the primary scenario.
* Evidence classes stay separate: EXECUTED (real trades attributed ex ante),
  QUOTE_VERIFIED_PAPER (entry and exit quotes re-verified in the archive) and
  UNVERIFIABLE (counted with reasons, never in a metric).
* Three benchmarks are kept strictly apart (BENCHMARKS):
  - RANDOM_CONTROL: identical rule on random names of the same universe and
    date. It is NOT the existing base strategy.
  - FROZEN_ML_BENCHMARK: the frozen registry benchmark model (role benchmark),
    identical rule, same dates.
  - EXISTING_BASE_STRATEGY: the existing scanner / option_learner base
    decisions on scanner candidates. Different population and holding horizon,
    therefore reported descriptively only and NEVER used for a superiority
    claim (``used_for_superiority: False``).
  Superiority comparisons use only contemporaneously observed, comparable
  results (same signal dates, all decisions of a date final, same rule,
  horizon and execution, same budget per signal; ABSTAIN = cash = 0), plus no
  trade (0). Missing comparison data never count as an advantage; missing
  results are never positive: primary view excludes them and reports
  coverage, the conservative view books them as a total loss.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

from . import option_link as al
from .registry import _atomic_json

REPORT = Path("outputs/research/autonomous_alpha/option_economics.json")
BENCHMARKS = {
    "random_control": "RANDOM_CONTROL – identical frozen rule on random names of the same PIT universe and signal "
                      "date (not the existing base strategy)",
    "frozen_ml_benchmark": "FROZEN_ML_BENCHMARK – frozen registry benchmark model (role benchmark), identical rule "
                           "and dates",
    "existing_base_strategy": "EXISTING_BASE_STRATEGY – existing scanner/option_learner base decisions on scanner "
                              "candidates (descriptive only: different population and holding horizon)",
}
HOLD_CALENDAR_DAYS = 28            # 20 NYSE trading days ~ 4 weeks: non-overlapping holding windows
SCENARIOS = ("BASE", "ADVERSE")
PRIMARY = "ADVERSE"
MIN_QUOTE_COVERAGE = 0.8
MULTIPLIER = 100


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# ── single position ──────────────────────────────────────────────────────────

def position_economics(entry_legs: list[dict], exit_legs: list[dict], rule: dict, scenario: str = PRIMARY) -> dict:
    """Net P&L of one paper position (USD) from observed bid/ask.

    Entry: long legs pay ask, short legs receive bid; exit: long legs receive
    bid, short legs pay ask. ADVERSE moves each fill by
    ``adverse_slippage_fraction_of_spread`` x (ask - bid) against us (bids are
    floored at 0). Fees: commission per contract and side for every leg.
    """
    if scenario not in SCENARIOS:
        raise ValueError(scenario)
    slip = float(rule["adverse_slippage_fraction_of_spread"]) if scenario == "ADVERSE" else 0.0
    contracts = int(rule["contracts_per_position"])
    fee = float(rule["commission_usd_per_contract_per_side"]) * contracts
    mult = MULTIPLIER * contracts
    exits = {x["occ_symbol"]: x for x in exit_legs}

    def buy(q):
        bid, ask = float(q["bid"]), float(q["ask"])
        return ask + slip * (ask - bid)

    def sell(q):
        bid, ask = float(q["bid"]), float(q["ask"])
        return max(0.0, bid - slip * (ask - bid))
    entry = exit_value = 0.0
    for leg in entry_legs:
        x = exits[leg["occ_symbol"]]
        if leg["side"] == "long":
            entry += buy(leg)
            exit_value += sell(x)
        else:
            entry -= sell(leg)
            exit_value -= buy(x)
    n = len(entry_legs)
    capital = entry * mult + fee * n
    proceeds = exit_value * mult - fee * n
    pnl = proceeds - capital
    return {"scenario": scenario, "capital_usd": round(capital, 4), "exit_value_usd": round(proceeds, 4),
            "pnl_usd": round(pnl, 4), "return": round(pnl / capital, 6) if capital > 0 else None,
            "max_loss_usd": round(capital + fee * n, 4)}


# ── provenance (quote archive) ───────────────────────────────────────────────

def _verify(d: dict, o: dict, archive: dict) -> list[str]:
    """Re-check every referenced quote in the archive; reasons if anything is off."""
    from modules import option_quotes as oq
    reasons = []
    pairs = [(leg, d["decision_day"], "ENTRY") for leg in d["legs"]]
    pairs += [(leg, o["exit_day"], "EXIT") for leg in o.get("exit_legs") or []]
    if len(o.get("exit_legs") or []) != len(d["legs"]):
        reasons.append("EXIT_LEGS_INCOMPLETE")
    for leg, day, tag in pairs:
        rec = archive.get(leg.get("quote_record_id"))
        if rec is None:
            reasons.append(f"{tag}_QUOTE_NOT_IN_ARCHIVE")
            continue
        if rec.get("occ_symbol") != leg["occ_symbol"]:
            reasons.append(f"{tag}_QUOTE_CONTRACT_MISMATCH")
        if _num(rec.get("bid")) != _num(leg.get("bid")) or _num(rec.get("ask")) != _num(leg.get("ask")):
            reasons.append(f"{tag}_QUOTE_PRICE_MISMATCH")
        if not oq.usable_rth_quote(rec):
            reasons.append(f"{tag}_QUOTE_NOT_USABLE_RTH")
        if oq.quote_day(rec) != day:
            reasons.append(f"{tag}_QUOTE_WRONG_DAY")
        if _num(rec.get("multiplier")) not in (None, float(MULTIPLIER)):
            reasons.append(f"{tag}_NONSTANDARD_MULTIPLIER")
    return sorted(set(reasons))


def _archive(root: Path, decisions: list[dict], outcomes: list[dict]) -> dict:
    from modules import option_quotes as oq
    ids, days = set(), set()
    for d in decisions:
        for leg in d["legs"]:
            ids.add(leg.get("quote_record_id"))
        days.add(d["decision_day"])
    for o in outcomes:
        for leg in o.get("exit_legs") or []:
            ids.add(leg.get("quote_record_id"))
        if o.get("exit_day"):
            days.add(o["exit_day"])
    if not days:
        return {}
    lo, hi = min(days), max(days)
    lo = date.fromordinal(date.fromisoformat(lo).toordinal() - 2).isoformat()
    hi = date.fromordinal(date.fromisoformat(hi).toordinal() + 2).isoformat()
    return oq.by_ids({i for i in ids if i}, lo, hi, root=Path(root) / oq.ROOT)


# ── per-trade table ──────────────────────────────────────────────────────────

def trades(root: Path, data: dict, cfg: dict) -> tuple[list[dict], dict]:
    """One row per (signal, structure) decision with its final economic status."""
    rule = cfg["rule"]
    sig_by_id = {s["signal_id"]: s for s in data["signals"]}
    out_by_dec = {o["decision_id"]: o for o in data["outcomes"]}
    archive = _archive(root, data["decisions"], data["outcomes"])
    rows, coverage = [], Counter()
    for d in data["decisions"]:
        s = sig_by_id.get(d["signal_id"])
        if s is None:
            coverage["DECISION_WITHOUT_SIGNAL"] += 1
            continue
        row = {"decision_id": d["decision_id"], "signal_id": d["signal_id"], "hypothesis_id": d["hypothesis_id"],
               "hypothesis_version": d["hypothesis_version"], "comparison_group": d["comparison_group"],
               "structure": d["structure"], "ticker": d["ticker"], "pit_data_date": d["pit_data_date"],
               "decision_day": d["decision_day"], "regime": (s.get("regime") or {}).get("label") or "UNKNOWN",
               "decision": d["decision"], "status": None, "evidence_class": None, "reasons": [],
               "exit_day": None, "economics": {}}
        if d.get("excluded_by_correction") or s.get("excluded_by_correction"):
            row.update(status="EXCLUDED_BY_CORRECTION", evidence_class="UNVERIFIABLE")
        elif d["decision"] == "ABSTAIN":
            row.update(status="ABSTAIN", reasons=list(d["abstain_reasons"]))
        else:
            o = out_by_dec.get(d["decision_id"])
            if o is None:
                row.update(status="PENDING")
            elif o.get("excluded_by_correction"):
                row.update(status="EXCLUDED_BY_CORRECTION", evidence_class="UNVERIFIABLE")
            elif o["status"] != "EXIT_OBSERVED":
                row.update(status="EXIT_MISSING", evidence_class="UNVERIFIABLE",
                           reasons=[o.get("missing_reason") or "EXIT_MISSING"])
            else:
                reasons = _verify(d, o, archive)
                if d.get("evidence_class_at_entry") != "QUOTE_VERIFIED_PAPER":
                    reasons.append("ENTRY_NOT_VERIFIED")
                if o["evidence_class"] != "QUOTE_VERIFIED_PAPER":
                    reasons.append("OUTCOME_NOT_VERIFIED")
                row["exit_day"] = o["exit_day"]
                if reasons:
                    row.update(status="UNVERIFIABLE", evidence_class="UNVERIFIABLE", reasons=sorted(set(reasons)))
                else:
                    row.update(status="SETTLED", evidence_class="QUOTE_VERIFIED_PAPER",
                               economics={sc: position_economics(d["legs"], o["exit_legs"], rule, sc)
                                          for sc in SCENARIOS})
        coverage[row["status"]] += 1
        rows.append(row)
    return rows, dict(coverage)


# ── metrics ──────────────────────────────────────────────────────────────────

def _drawdown(pnls: list[float]) -> float:
    peak = cum = dd = 0.0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        dd = min(dd, cum - peak)
    return round(dd, 4)


def _peak_capital(rows: list[dict], scenario: str) -> float:
    events = []
    for r in rows:
        cap = r["economics"][scenario]["capital_usd"]
        events += [(r["decision_day"], 1, cap), (r["exit_day"], 0, -cap)]
    cur = peak = 0.0
    for _, _, c in sorted(events):
        cur += c
        peak = max(peak, cur)
    return round(peak, 4)


def independent_periods(dates) -> int:
    """Non-overlapping holding windows (same blocking as option_learner._periods)."""
    from modules.option_learner import _periods
    return len(_periods({d: [d] for d in set(dates)}, HOLD_CALENDAR_DAYS))


def metrics(rows: list[dict], scenario: str = PRIMARY) -> dict:
    """Verified paper positions only (rows with status SETTLED)."""
    rows = sorted((r for r in rows if r["status"] == "SETTLED"), key=lambda r: (r["exit_day"], r["decision_id"]))
    if not rows:
        return {"n": 0, "status": "NO_VERIFIED_OUTCOMES"}
    ec = [r["economics"][scenario] for r in rows]
    rets = [e["return"] for e in ec]
    pnls = [e["pnl_usd"] for e in ec]
    wins, losses = [r for r in rets if r > 0], [r for r in rets if r <= 0]
    gain, loss = sum(p for p in pnls if p > 0), -sum(p for p in pnls if p < 0)
    by_ticker, by_date = Counter(), Counter()
    for r, p in zip(rows, pnls):
        by_ticker[r["ticker"]] += abs(p)
        by_date[r["pit_data_date"]] += abs(p)
    total_abs = sum(abs(p) for p in pnls) or 1.0
    return {
        "n": len(rows), "winrate": round(len(wins) / len(rows), 4),
        "avg_win_return": round(sum(wins) / len(wins), 6) if wins else None,
        "avg_loss_return": round(sum(losses) / len(losses), 6) if losses else None,
        "net_ev_return": round(sum(rets) / len(rets), 6),
        "net_ev_usd": round(sum(pnls) / len(pnls), 4),
        "profit_factor": round(gain / loss, 4) if loss > 0 else None,
        "capital_weighted_return": round(sum(pnls) / sum(e["capital_usd"] for e in ec), 6),
        "total_pnl_usd": round(sum(pnls), 4), "total_capital_usd": round(sum(e["capital_usd"] for e in ec), 4),
        "max_loss_usd": round(min(pnls), 4), "worst_return": round(min(rets), 6),
        "max_drawdown_usd": _drawdown(pnls), "peak_concurrent_capital_usd": _peak_capital(rows, scenario),
        "max_structural_loss_usd": round(max(e["max_loss_usd"] for e in ec), 4),
        "distinct_signal_dates": len({r["pit_data_date"] for r in rows}),
        "independent_periods": independent_periods(r["pit_data_date"] for r in rows),
        "concentration": {"top_ticker": by_ticker.most_common(1)[0][0],
                          "top_ticker_share_abs_pnl": round(by_ticker.most_common(1)[0][1] / total_abs, 4),
                          "top_date": by_date.most_common(1)[0][0],
                          "top_date_share_abs_pnl": round(by_date.most_common(1)[0][1] / total_abs, 4),
                          "distinct_tickers": len(by_ticker)},
    }


# ── same-population, same-budget comparisons ─────────────────────────────────

def _signal_return(r: dict, view: str) -> float | None:
    """Return on one equal budget unit per signal. ABSTAIN = cash = 0."""
    if r["status"] == "ABSTAIN":
        return 0.0
    if r["status"] == "SETTLED":
        return r["economics"][PRIMARY]["return"]
    if r["status"] == "PENDING":
        return None
    # EXIT_MISSING / UNVERIFIABLE / EXCLUDED: never positive
    return -1.0 if view == "CONSERVATIVE" else None


def _daily(rows: list[dict], view: str) -> dict[str, float]:
    """Per-date mean signal return over dates whose decisions are ALL final."""
    by_date = defaultdict(list)
    pending = set()
    for r in rows:
        if r["status"] == "PENDING":
            pending.add(r["pit_data_date"])
        by_date[r["pit_data_date"]].append(r)
    out = {}
    for d, rs in by_date.items():
        if d in pending:
            continue
        vals = [v for v in (_signal_return(r, view) for r in rs) if v is not None]
        if vals:
            out[d] = sum(vals) / len(vals)
    return out


def compare(rows_h: list[dict], rows_c: list[dict] | None, *, view: str = "PRIMARY", alpha: float = 0.05) -> dict:
    """Paired per-date difference H - C (C=None: no trade = 0)."""
    from modules.option_learner import _block_bootstrap_ci
    a = _daily(rows_h, view)
    b = {d: 0.0 for d in a} if rows_c is None else _daily(rows_c, view)
    common = sorted(set(a) & set(b))
    if not common:
        return {"n_dates": 0, "status": "NO_COMMON_POPULATION"}
    diffs = {d: [a[d] - b[d]] for d in common}
    mean = sum(v[0] for v in diffs.values()) / len(diffs)
    ci = _block_bootstrap_ci(diffs, HOLD_CALENDAR_DAYS, alpha=alpha, method="t")
    return {"n_dates": len(common), "independent_periods": independent_periods(common),
            "mean_difference": round(mean, 6), "ci": [round(ci[0], 6), round(ci[1], 6)] if ci else None,
            "ci_level": 1 - alpha, "ci_method": "block_bootstrap_t_non_overlapping_windows",
            "status": "DESCRIPTIVE_NOT_A_TEST"}


# ── research decision ────────────────────────────────────────────────────────

def research_decision(m: dict, cov: dict, vs: dict, cfg: dict) -> dict:
    """Research label only. Never TRADE, PROMOTE or an order."""
    reasons = []
    trade_n = cov["TRADE_DECIDED"]
    verified = m.get("n", 0)
    final = trade_n - cov.get("PENDING", 0)
    quote_cov = verified / final if final else None
    if verified < int(cfg["coverage"]["min_verified_outcomes"]):
        reasons.append("TOO_FEW_VERIFIED_OUTCOMES")
    if (m.get("independent_periods") or 0) < int(cfg["coverage"]["min_distinct_signal_dates"]):
        reasons.append("TOO_FEW_INDEPENDENT_PERIODS")
    if quote_cov is None or quote_cov < MIN_QUOTE_COVERAGE:
        reasons.append("QUOTE_COVERAGE_BELOW_MIN")
    if reasons:
        label = "NEED_MORE_DATA"
    elif m["net_ev_return"] <= 0 or (vs["no_trade_conservative"].get("mean_difference") or 0) <= 0:
        label = "NO_ECONOMIC_EDGE_AFTER_COSTS"
    elif vs["random_control"].get("status") != "DESCRIPTIVE_NOT_A_TEST":
        label = "COMPARISON_DATA_MISSING"            # missing comparison data never count as an advantage
        reasons.append("RANDOM_CONTROL_NOT_CONTEMPORANEOUSLY_OBSERVED")
    elif (vs["random_control"].get("mean_difference") or 0) <= 0:
        label = "NOT_BETTER_THAN_RANDOM_CONTROL"
    elif not vs["random_control"].get("ci") or vs["random_control"]["ci"][0] <= 0:
        label = "ADVANTAGE_NOT_ESTABLISHED"
    else:
        label = "CANDIDATE_FOR_PROSPECTIVE_VALIDATION"
    return {"label": label, "reasons": reasons, "quote_coverage": round(quote_cov, 4) if quote_cov is not None else None,
            "decision_type": "RESEARCH_ONLY", "order": None, "production_influence": "NONE",
            "next_step": ("Phase C prospective validation (preregistered)" if label != "NEED_MORE_DATA"
                          else "Keep collecting prospective, quote-verified outcomes")}


def propose_structure(per_structure: dict) -> dict:
    """B5: which allowed structure to propose for further PROSPECTIVE research.

    Only structures labelled CANDIDATE_FOR_PROSPECTIVE_VALIDATION qualify (a
    higher return with insufficient liquidity or data quality is not a valid
    success: those never reach that label). Among them the higher lower CI
    bound versus the random control wins; ties prefer the spread (smaller
    maximum loss). The selection uses exploratory data, so Phase C validates
    it only on NEW data registered after this proposal.
    """
    cands = []
    for st, e in per_structure.items():
        if e["research_decision"]["label"] == "CANDIDATE_FOR_PROSPECTIVE_VALIDATION":
            ci = e["comparisons"]["random_control"].get("ci") or [float("-inf")]
            cands.append((ci[0], st == "BULL_CALL_SPREAD", st))
    if not cands:
        return {"structure": None, "reason": "NO_STRUCTURE_QUALIFIES_FOR_PROSPECTIVE_RESEARCH",
                "decision_type": "RESEARCH_ONLY", "order": None}
    best = max(cands)
    return {"structure": best[2], "reason": "HIGHEST_LOWER_CI_VS_RANDOM_CONTROL_AMONG_CANDIDATES",
            "candidates": sorted(c[2] for c in cands), "selection_uses_exploratory_data": True,
            "requires": "Phase C preregistration and validation on new data only",
            "decision_type": "RESEARCH_ONLY", "order": None}


def evaluate(root: Path = Path("."), *, now: datetime | None = None, persist: bool = True) -> dict:
    root = Path(root)
    cfg = al.config(root)
    now = now or datetime.now(timezone.utc)
    data = al.read_all(root)
    al._check_rule(cfg, data["decisions"])
    rows, coverage_all = trades(root, data, cfg)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["hypothesis_id"], r["structure"])].append(r)
    control = {st: groups.get((al.CONTROL_ID, st), []) for st in al.STRUCTURES}
    existing = existing_base_strategy(root, {r["decision_day"] for r in rows if r["status"] != "PENDING"})
    reference_id = next((s["hypothesis_id"] for s in data["signals"] if s["comparison_group"] == "REFERENCE"), None)
    out = {}
    for (hid, st), rs in sorted(groups.items()):
        cov = Counter(r["status"] for r in rs)
        cov["TRADE_DECIDED"] = sum(r["decision"] == "TRADE" for r in rs)
        unver = Counter(x for r in rs if r["status"] in ("UNVERIFIABLE", "EXIT_MISSING") for x in r["reasons"])
        m = {sc: metrics(rs, sc) for sc in SCENARIOS}
        by_regime = {}
        for reg in sorted({r["regime"] for r in rs}):
            sub = metrics([r for r in rs if r["regime"] == reg], PRIMARY)
            by_regime[reg] = {k: sub.get(k) for k in ("n", "winrate", "net_ev_return", "capital_weighted_return")}
        vs = {"no_trade": compare(rs, None), "no_trade_conservative": compare(rs, None, view="CONSERVATIVE"),
              "random_control": ({**compare(rs, control[st]), "benchmark": BENCHMARKS["random_control"]}
                                 if hid != al.CONTROL_ID else {"status": "SELF"}),
              "frozen_ml_benchmark": (
                  {**compare(rs, groups.get((reference_id, st), [])), "benchmark": BENCHMARKS["frozen_ml_benchmark"]}
                  if reference_id and hid != reference_id
                  else {"status": "SELF" if hid == reference_id else "NO_FROZEN_BENCHMARK_SIGNALS"}),
              "existing_base_strategy": existing}
        entry = {"comparison_group": rs[0]["comparison_group"],
                 "versions": sorted({r["hypothesis_version"] for r in rs}),
                 "coverage": {"decisions": len(rs), "trade_decided": cov["TRADE_DECIDED"],
                              "abstain": cov.get("ABSTAIN", 0), "settled_verified": cov.get("SETTLED", 0),
                              "pending": cov.get("PENDING", 0), "exit_missing": cov.get("EXIT_MISSING", 0),
                              "unverifiable": cov.get("UNVERIFIABLE", 0),
                              "excluded_by_correction": cov.get("EXCLUDED_BY_CORRECTION", 0),
                              "failure_reasons": dict(sorted(unver.items())),
                              "abstain_reasons": dict(sorted(Counter(
                                  x for r in rs if r["status"] == "ABSTAIN" for x in r["reasons"]).items()))},
                 "metrics": m, "by_regime": by_regime, "comparisons": vs}
        entry["research_decision"] = (research_decision(m[PRIMARY], cov, vs, cfg) if hid != al.CONTROL_ID
                                      else {"label": "RANDOM_CONTROL_BENCHMARK", "decision_type": "RESEARCH_ONLY",
                                            "order": None, "production_influence": "NONE"})
        out.setdefault(hid, {})[st] = entry
    executed = _executed(root, data)
    report = {
        "schema": "alpha-option-economics-v1", "generated_at": now.isoformat(timespec="seconds"),
        "rule_version": cfg["rule"]["rule_version"], "rule_hash": al.rule_hash(cfg),
        "primary_scenario": PRIMARY, "scenarios": list(SCENARIOS),
        "execution_model": {"entry": "long@ask, short@bid", "exit": "long@bid, short@ask",
                            "fee_usd_per_contract_per_side": cfg["rule"]["commission_usd_per_contract_per_side"],
                            "adverse_fraction_of_spread": cfg["rule"]["adverse_slippage_fraction_of_spread"],
                            "contracts_per_position": cfg["rule"]["contracts_per_position"]},
        "evidence_classes": {"EXECUTED": executed, "QUOTE_VERIFIED_PAPER": coverage_all.get("SETTLED", 0),
                             "UNVERIFIABLE": sum(coverage_all.get(k, 0) for k in
                                                 ("UNVERIFIABLE", "EXIT_MISSING", "EXCLUDED_BY_CORRECTION"))},
        "coverage": coverage_all, "benchmarks": BENCHMARKS, "existing_base_strategy": existing,
        "hypotheses": out,
        "proposals": {hid: propose_structure(h) for hid, h in sorted(out.items()) if hid != al.CONTROL_ID},
        "status": ("NEED_MORE_DATA" if not any(e["research_decision"]["label"] not in
                                               ("NEED_MORE_DATA", "RANDOM_CONTROL_BENCHMARK")
                                               for h in out.values() for e in h.values()) else "EVALUATED"),
        "data_origin": "REPOSITORY_JOURNALS_ONLY", "decision_type": "RESEARCH_ONLY",
        "order_placed": False, "production_influence": "NONE", "llm_calls": 0,
    }
    if persist:
        _atomic_json(root / REPORT, report)
    return report


def _executed(root: Path, data: dict) -> dict:
    """EXECUTED = documented broker execution AND ex-ante alpha signal ID. Paper
    bookings and archived bid/ask valuations never count here (no upgrade)."""
    from . import forward_options
    if not (Path(root) / forward_options.POLICY).exists():
        return {"n": 0, "status": "NO_POLICY"}
    rep = forward_options.evaluate(root, persist=False)
    n = rep.get("n_executed_attributed_ex_ante", 0)
    return {"n": n, "attribution_checks": rep.get("alpha_attribution_checks", {}),
            "real_trade_evidence_classes": rep.get("evidence_classes", {}),
            "rule": "documented broker execution (entry and exit fills) AND ex-ante alpha_signal_id",
            "status": "PRESENT" if n else "NONE_DOCUMENTED"}


def existing_base_strategy(root: Path, decision_days: set[str], horizon: int = 20) -> dict:
    """Existing scanner/option_learner base strategy on the same decision days.

    Descriptive only: its population (scanner candidates) and holding horizon
    (calendar-day ledger marks) differ from the hypothesis signals, so it is
    NOT comparable and never used for a superiority claim. Missing data are
    reported as such, never as an advantage."""
    out = {"benchmark": BENCHMARKS["existing_base_strategy"], "used_for_superiority": False,
           "comparability": {"population": "DIFFERENT (scanner candidates vs hypothesis signals)",
                             "holding_horizon": f"{horizon} calendar days vs 20 trading days",
                             "execution": "ask/bid + commission (no adverse scenario)"}}
    from modules import option_learner as ol
    from modules import research_evidence_discovery as red
    from modules.atomic_io import read_jsonl
    jp = Path(root) / ol.OUT / ol.JOURNAL
    if not decision_days or not jp.exists():
        return {**out, "status": "NO_CONTEMPORANEOUS_DATA", "reason": "no base journal or no decision days"}
    decs = [j for j in read_jsonl(jp) if j.get("kind") == "decision" and j.get("role") == "learning"
            and j.get("horizon") == horizon and j.get("date") in decision_days]
    if not decs:
        return {**out, "status": "NO_CONTEMPORANEOUS_DATA", "reason": "no base decision on these decision days"}
    try:
        as_of = datetime.now(timezone.utc).date()
        _, cands, _ = ol._load(as_of, horizon, Path(root) / red.LEDGER, Path(root) / red.QUOTES)
    except Exception as exc:  # noqa: BLE001 – descriptive add-on, never blocks the evaluation
        return {**out, "status": "UNAVAILABLE", "reason": f"{type(exc).__name__}: {exc}"[:200]}
    by_key = {(c["date"], c["signal_id"]): c for c in cands}
    vals, unverified, trades = [], 0, 0
    for j in decs:
        if j.get("action") != "TRADE":
            vals.append(0.0)
            continue
        trades += 1
        e = ol.realized(by_key.get((j["date"], j["signal_id"])) or {"exprs": []}, j.get("contract"))
        if e and e.get("status") == "VERIFIED" and e.get("y") is not None:
            vals.append(float(e["y"]))
        else:
            unverified += 1
    return {**out, "status": "NOT_COMPARABLE", "common_decision_days": len({j["date"] for j in decs}),
            "base_decisions": len(decs), "base_trades": trades, "base_unverified_or_pending": unverified,
            "base_mean_per_decision_verified": round(sum(vals) / len(vals), 6) if vals else None,
            "note": "No superiority claim versus the existing base strategy is possible from these data."}

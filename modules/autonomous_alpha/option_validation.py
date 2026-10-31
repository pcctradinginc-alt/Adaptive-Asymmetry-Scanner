"""Alpha-to-Options prospective validation (Phase C) and learning.

RESEARCH ONLY. Reuses the sequential machinery of PR #145/#147 instead of a new
one: ``option_learner.test_budget`` (permanent Bonferroni error budget),
``look_days`` / ``alpha_spending`` (preregistered looks, O'Brien-Fleming),
``freeze_look_report`` (create-only, versioned look reports) and the block
bootstrap with t-quantiles over non-overlapping holding windows.

Flow (weekly research job, no network):
1. ``register_plans``: for every hypothesis structure Phase B proposes, a
   create-only, hashed plan fixes hypothesis/signal definition, contract rule,
   target, comparisons, population (signals dated AFTER registration only),
   minimum coverage, minimum independent observations, looks, stop and
   falsification rules and its error-budget slot. Same hypothesis version and
   structure are never registered twice (no re-testing of the same data).
2. ``evaluate_plans``: look k only at its preregistered cutoff and only when its
   population is final; its report is frozen. CONFIRMED / FALSIFIED stop the
   plan; after the last look anything else is NOT_CONFIRMED.
3. ``qualification``: all eight criteria of the research mandate are required
   for QUALIFIED_FOR_HUMAN_REVIEW. Too little data stays NEED_MORE_DATA.
4. ``learning``: what fails, where, which data are missing; bounded research
   suggestions only (registration stays in the explorer's own budget).

Never an order, a promotion, a champion or a risk-limit change.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import yaml

from . import option_economics as oe
from . import option_link as al
from .registry import EvidenceIntegrityError, _atomic_json, digest

CONFIG = Path("config/alpha_option_validation.yaml")
OUT = Path("outputs/research/autonomous_alpha/option_validation")
PLAN_DIR = "plans"
LOOK_DIR = "looks"
REPORT = Path("outputs/research/autonomous_alpha/option_validation.json")
LEARNING = Path("outputs/research/autonomous_alpha/option_learning.json")
FINAL = ("CONFIRMED", "FALSIFIED", "NOT_CONFIRMED")
CLASSES = ("RESEARCH", "NOT_CONFIRMED", "QUALIFIED_FOR_HUMAN_REVIEW")


def config(root: Path = Path(".")) -> dict:
    cfg = yaml.safe_load((Path(root) / CONFIG).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or cfg.get("version") != "alpha-option-validation-v1":
        raise EvidenceIntegrityError("alpha option validation policy invalid")
    if cfg.get("production_influence") != "NONE" or cfg.get("allow_llm") is not False:
        raise EvidenceIntegrityError("validation must have no production/LLM access")
    if cfg.get("target") != "PER_SIGNAL_NET_RETURN_EQUAL_BUDGET_ADVERSE":
        raise EvidenceIntegrityError("unknown validation target")
    if sorted(cfg.get("comparisons") or []) != ["BASE_STRATEGY", "NO_TRADE"]:
        raise EvidenceIntegrityError("both comparisons (no trade, base strategy) are mandatory")
    if not 180 <= int(cfg.get("max_days", 0)) <= 730:
        raise EvidenceIntegrityError("validation horizon invalid")
    if not 0.8 <= float(cfg.get("min_decision_coverage", 0)) <= 1 or not 0.8 <= float(cfg.get("min_quote_coverage", 0)) <= 1:
        raise EvidenceIntegrityError("coverage floors cannot be relaxed below 80 %")
    if int(cfg.get("min_verified_outcomes", 0)) < 30 or int(cfg.get("min_independent_periods", 0)) < 6:
        raise EvidenceIntegrityError("minimum sample cannot be relaxed")
    tb = cfg.get("test_budget") or {}
    if not 0 < float(tb.get("total_alpha", 0)) <= 0.05 or not 1 <= int(tb.get("max_tests", 0)) <= 100:
        raise EvidenceIntegrityError("error budget invalid")
    conc = cfg.get("concentration") or {}
    if not 0 < float(conc.get("max_top_ticker_share", 0)) <= 0.5 or not 0 < float(conc.get("max_top_date_share", 0)) <= 0.5:
        raise EvidenceIntegrityError("concentration limits invalid")
    if int((cfg.get("stability") or {}).get("min_bucket_n", 0)) < 3:
        raise EvidenceIntegrityError("stability bucket size invalid")
    if not 0 <= int((cfg.get("learning") or {}).get("max_suggested_hypotheses", -1)) <= 5:
        raise EvidenceIntegrityError("learning suggestion budget invalid")
    return cfg


def _out(root: Path) -> Path:
    return Path(root) / OUT


def _budget(root: Path, cfg: dict) -> dict:
    from modules import option_learner as ol
    return ol.test_budget(_out(root), {"test_budget": cfg["test_budget"]})


def load_plan(path: Path) -> dict:
    from modules import option_learner as ol
    plan = ol._read_json(path)
    core = {k: v for k, v in plan.items() if k != "plan_sha256"}
    if plan.get("plan_sha256") != ol._sha(core):
        return {"plan_id": plan.get("plan_id") or path.stem, "invalid": True}
    return plan


def plans(root: Path) -> list[dict]:
    d = _out(root) / PLAN_DIR
    return [load_plan(p) for p in sorted(d.glob("*.json"))] if d.exists() else []


# ── 1. preregistration ───────────────────────────────────────────────────────

def register_plans(root: Path, econ: dict, *, now: datetime | None = None) -> list[str]:
    """Create-only plans for Phase B proposals (never for CONTROL, never twice)."""
    from modules import option_learner as ol
    from modules.atomic_io import append_jsonl, atomic_write_json
    root, now = Path(root), now or datetime.now(timezone.utc)
    cfg, link = config(root), al.config(root)
    signals = al.read_all(root)["signals"]
    latest = {}
    for s in signals:
        if s["comparison_group"] != "CONTROL":
            latest[s["hypothesis_id"]] = s                 # journal is sorted by registration time
    created = []
    for hid, prop in sorted((econ.get("proposals") or {}).items()):
        st = prop.get("structure")
        sig = latest.get(hid)
        if not st or sig is None:
            continue
        key = {"hypothesis_id": hid, "hypothesis_version": sig["hypothesis_version"], "structure": st,
               "rule_version": link["rule"]["rule_version"], "rule_hash": al.rule_hash(link)}
        plan_id = "AOV-" + digest(key)[:16]
        path = _out(root) / PLAN_DIR / f"{plan_id}.json"
        if path.exists():
            continue
        budget = _budget(root, cfg)
        slot = None if budget["exhausted"] else budget["slot_alpha"]
        start = now.date() + timedelta(days=1)            # only signals AFTER registration count
        days = ol.look_days(int(cfg["max_days"]))
        src = (link["sources"].get(sig["hypothesis_source"]) or {}) if sig["hypothesis_source"] != "control" else {}
        plan = {
            "plan_id": plan_id, "schema": "alpha-option-plan-v1", **key,
            "hypothesis": {"id": hid, "version": sig["hypothesis_version"], "hash": sig["hypothesis_hash"],
                           "source": sig["hypothesis_source"], "comparison_group": sig["comparison_group"],
                           "registered_at": sig["hypothesis_registered_at"],
                           "experiment_version": sig["experiment_version"]},
            "signal_definition": {"selection": f"top {src.get('top_k')} of each prospective prediction",
                                  "direction": "BULLISH", "horizon_trading_days": sig["horizon_trading_days"],
                                  "link_version": link["version"]},
            "contract_rule": {"rule_version": link["rule"]["rule_version"], "rule_hash": al.rule_hash(link),
                              "structure": st},
            "target": cfg["target"],
            "return_calculation": {"execution": "long@ask/short@bid entry, long@bid/short@ask exit",
                                   "scenario": oe.PRIMARY, "abstain": "cash = 0",
                                   "missing": "excluded and counted; conservative view = total loss",
                                   "fees_usd_per_contract_per_side": link["rule"]["commission_usd_per_contract_per_side"]},
            "comparisons": {"NO_TRADE": "0 per signal", "BASE_STRATEGY": f"{al.CONTROL_ID}, same structure and dates",
                            "FROZEN_REFERENCE": "descriptive only"},
            "population": {"signals": "tradable signals of this hypothesis version, pit_data_date in "
                                       "[start_date, look cutoff]", "unit": "signal date (equal budget)"},
            "min_decision_coverage": cfg["min_decision_coverage"], "min_quote_coverage": cfg["min_quote_coverage"],
            "min_verified_outcomes": cfg["min_verified_outcomes"],
            "min_independent_periods": cfg["min_independent_periods"],
            "registered_at": now.isoformat(timespec="seconds"), "start_date": start.isoformat(),
            "max_days": int(cfg["max_days"]), "look_days": days,
            "look_cutoffs": [(start + timedelta(days=d)).isoformat() for d in days],
            "alpha_slot": slot, "look_alpha": ol.alpha_spending(slot) if slot else None,
            "test_budget": {k: budget[k] for k in ("total_alpha", "max_tests", "used")},
            "ci_method": "block_bootstrap_t_non_overlapping_windows",
            "stop_rules": {"CONFIRMED": "CI(H-no trade) and CI(H-base) both entirely > 0 at the look alpha, "
                                        "conservative mean > 0, minimum sample and coverage met",
                           "FALSIFIED": "CI(H-no trade) entirely < 0 at the look alpha",
                           "NOT_CONFIRMED": "last look without confirmation (plan ends for good)"},
            "concentration": cfg["concentration"], "stability": cfg["stability"],
            "selection_note": "structure selected by Phase B on exploratory data; validated on new data only",
            "production_influence": "NONE",
        }
        plan["plan_sha256"] = ol._sha(plan)
        atomic_write_json(path, plan, indent=1)
        append_jsonl(_out(root) / ol.VARIANTS, [{"event": "registered", "plan_id": plan_id,
                                                  "registered_at": plan["registered_at"],
                                                  "test_budget": cfg["test_budget"], "alpha_slot": slot}])
        created.append(plan_id)
    return created


# ── 2. preregistered looks ───────────────────────────────────────────────────

def _population(plan: dict, rows: list[dict], signals: list[dict], cutoff: str):
    def inside(d):
        return plan["start_date"] <= d <= cutoff
    h = [r for r in rows if r["hypothesis_id"] == plan["hypothesis_id"]
         and r["hypothesis_version"] == plan["hypothesis_version"] and r["structure"] == plan["structure"]
         and inside(r["pit_data_date"])]
    c = [r for r in rows if r["hypothesis_id"] == al.CONTROL_ID and r["structure"] == plan["structure"]
         and inside(r["pit_data_date"])]
    sig = [s for s in signals if s["hypothesis_id"] == plan["hypothesis_id"]
           and s["hypothesis_version"] == plan["hypothesis_version"] and s["tradable"]
           and not s.get("excluded_by_correction") and inside(s["pit_data_date"])]
    return h, c, sig


def _due(h: list[dict], sig: list[dict], decided: set, cutoff: str, today: date, lag: int) -> bool:
    if today <= date.fromisoformat(cutoff):
        return False
    if any(r["status"] == "PENDING" for r in h):
        return False
    for s in sig:                                   # undecided signal still inside its decision window
        if s["signal_id"] not in decided:
            reg = datetime.fromisoformat(s["registered_at"]).date()
            if today <= al.add_trading_days(reg, lag):
                return False
    return True


def _look(root: Path, plan: dict, k: int, h: list[dict], c: list[dict], sig: list[dict],
          now: datetime, cfg: dict) -> dict:
    from modules import option_learner as ol
    alpha_k = plan["look_alpha"][k - 1]
    final_trades = [r for r in h if r["decision"] == "TRADE"]
    settled = [r for r in h if r["status"] == "SETTLED"]
    dcov = round(len(h) / len(sig), 4) if sig else None
    qcov = round(len(settled) / len(final_trades), 4) if final_trades else None
    m = oe.metrics(h)
    vs_nt = oe.compare(h, None, alpha=alpha_k)
    vs_cons = oe.compare(h, None, view="CONSERVATIVE", alpha=alpha_k)
    vs_base = oe.compare(h, c, alpha=alpha_k)
    periods = m.get("independent_periods") or 0
    final_look = k == len(plan["look_cutoffs"])
    if dcov is None or dcov < plan["min_decision_coverage"] or (qcov is not None and qcov < plan["min_quote_coverage"]):
        verdict = "INSUFFICIENT_COVERAGE"
    elif m.get("n", 0) < plan["min_verified_outcomes"] or periods < plan["min_independent_periods"]:
        verdict = "NEED_MORE_DATA"
    elif vs_nt.get("ci") and vs_nt["ci"][1] < 0:
        verdict = "FALSIFIED"
    elif (vs_nt.get("ci") and vs_nt["ci"][0] > 0 and vs_base.get("ci") and vs_base["ci"][0] > 0
          and (vs_cons.get("mean_difference") or 0) > 0):
        verdict = "CONFIRMED"
    else:
        verdict = "CONTINUE"
    if final_look and verdict not in ("CONFIRMED", "FALSIFIED"):
        verdict = "NOT_CONFIRMED"
    report = {
        "plan_id": plan["plan_id"], "plan_sha256": plan["plan_sha256"], "look": k,
        "looks": len(plan["look_cutoffs"]), "data_cutoff": plan["look_cutoffs"][k - 1],
        "evaluated_as_of": now.date().isoformat(), "hypothesis_id": plan["hypothesis_id"],
        "structure": plan["structure"], "n_population": len(sig), "n_compared": len(h),
        "decision_coverage": dcov, "quote_coverage": qcov,
        "label_ids": sorted(r["decision_id"] + "|" + str(r["exit_day"]) for r in settled),
        "alpha": {"slot": plan["alpha_slot"], "look_alpha": alpha_k, "ci_level": round(1 - alpha_k, 8)},
        "metrics": m, "vs_no_trade": vs_nt, "vs_no_trade_conservative": vs_cons, "vs_base_strategy": vs_base,
        "verdict": verdict,
        "input_sha256": ol._sha({"h": [[r["decision_id"], r["status"],
                                        (r["economics"].get(oe.PRIMARY) or {}).get("return")] for r in h],
                                 "c": [[r["decision_id"], r["status"],
                                        (r["economics"].get(oe.PRIMARY) or {}).get("return")] for r in c],
                                 "n_signals": len(sig), "look_alpha": alpha_k, "plan_sha256": plan["plan_sha256"]}),
    }
    rec = ol.freeze_look_report(_out(root), report, now, LOOK_DIR)
    return {**report, "report_id": rec["report_id"], "report_version": rec["version"], "rows": h}


def evaluate_plans(root: Path, *, now: datetime | None = None) -> list[dict]:
    root, now = Path(root), now or datetime.now(timezone.utc)
    cfg, link = config(root), al.config(root)
    data = al.read_all(root)
    rows, _ = oe.trades(root, data, link)
    decided = {d["signal_id"] for d in data["decisions"]}
    lag = int(link["decision_max_lag_trading_days"])
    out = []
    for plan in plans(root):
        if plan.get("invalid"):
            out.append({"plan_id": plan["plan_id"], "status": "PLAN_INVALID", "classification": "RESEARCH"})
            continue
        res = {"plan_id": plan["plan_id"], "hypothesis_id": plan["hypothesis_id"],
               "hypothesis_version": plan["hypothesis_version"], "structure": plan["structure"],
               "registered_at": plan["registered_at"], "start_date": plan["start_date"],
               "look_cutoffs": plan["look_cutoffs"], "alpha_slot": plan["alpha_slot"], "looks": []}
        if plan["alpha_slot"] is None:
            res.update(status="NO_BUDGET", classification="RESEARCH",
                       reason="error budget exhausted at registration - never qualifies")
            out.append(res)
            continue
        status, binding = "AWAITING_LOOK", None
        for k, cutoff in enumerate(plan["look_cutoffs"], 1):
            h, c, sig = _population(plan, rows, data["signals"], cutoff)
            if not _due(h, sig, decided, cutoff, now.date(), lag):
                break
            lk = _look(root, plan, k, h, c, sig, now, cfg)
            res["looks"].append({x: lk[x] for x in ("look", "data_cutoff", "verdict", "report_id",
                                                     "report_version", "n_compared", "decision_coverage",
                                                     "quote_coverage")})
            binding = lk
            status = lk["verdict"] if lk["verdict"] in FINAL else "RUNNING"
            if status in FINAL:
                break
        res["status"] = status
        if binding is not None:
            res["binding"] = {k: v for k, v in binding.items() if k != "rows"}
        if status == "CONFIRMED":
            q = qualification(plan, binding, cfg, link)
            res["qualification"] = q
            res["classification"] = "QUALIFIED_FOR_HUMAN_REVIEW" if q["qualified"] else "NOT_CONFIRMED"
        else:
            res["classification"] = "NOT_CONFIRMED" if status in ("FALSIFIED", "NOT_CONFIRMED") else "RESEARCH"
        out.append(res)
    return out


# ── 3. eight qualification criteria ──────────────────────────────────────────

def _stability(rows: list[dict], n_min: int) -> dict:
    settled = sorted((r for r in rows if r["status"] == "SETTLED"), key=lambda r: (r["pit_data_date"], r["decision_id"]))

    def mean(rs):
        return sum(r["economics"][oe.PRIMARY]["return"] for r in rs) / len(rs) if rs else None
    regimes = defaultdict(list)
    for r in settled:
        regimes[r["regime"]].append(r)
    reg = {k: {"n": len(v), "mean": round(mean(v), 6)} for k, v in sorted(regimes.items())}
    half = len(settled) // 2
    cohorts = {"first_half": settled[:half], "second_half": settled[half:]}
    coh = {k: {"n": len(v), "mean": round(mean(v), 6) if v else None} for k, v in cohorts.items()}
    tested = [x for x in [*reg.values(), *coh.values()] if x["n"] >= n_min]
    ok = bool(tested) and all(x["mean"] > 0 for x in tested) and all(x["n"] >= n_min for x in coh.values())
    return {"ok": ok, "by_regime": reg, "time_cohorts": coh, "min_bucket_n": n_min}


def qualification(plan: dict, look: dict, cfg: dict, link: dict) -> dict:
    """All eight criteria are required; each is reported with its evidence."""
    m, rows = look["metrics"], look["rows"]
    nt, base, cons = look["vs_no_trade"], look["vs_base_strategy"], look["vs_no_trade_conservative"]
    conc = m.get("concentration") or {}
    settled = [r for r in rows if r["status"] == "SETTLED"]
    # entry capital is capped by the rule; ADVERSE fills may add at most the slippage share on top
    cap_limit = (float(link["rule"]["max_capital_usd"]) * (1 + float(link["rule"]["adverse_slippage_fraction_of_spread"]))
                 + 4 * float(link["rule"]["commission_usd_per_contract_per_side"]))
    crit = {
        "1_positive_net_ev_prospective": bool(nt.get("ci") and nt["ci"][0] > 0 and (cons.get("mean_difference") or 0) > 0),
        "2_advantage_vs_base_strategy": bool(base.get("ci") and base["ci"][0] > 0),
        "3_enough_independent_periods": (m.get("independent_periods") or 0) >= plan["min_independent_periods"],
        "4_coverage_meets_plan": ((look["decision_coverage"] or 0) >= plan["min_decision_coverage"]
                                  and (look["quote_coverage"] or 0) >= plan["min_quote_coverage"]),
        "5_concentration_disclosed_and_limited": bool(
            conc and conc["top_ticker_share_abs_pnl"] <= cfg["concentration"]["max_top_ticker_share"]
            and conc["top_date_share_abs_pnl"] <= cfg["concentration"]["max_top_date_share"]),
        "6_regime_and_cohort_stability": _stability(rows, int(cfg["stability"]["min_bucket_n"]))["ok"],
        "7_multiple_testing_and_budget": bool(plan["alpha_slot"] and look.get("report_id")),
        "8_max_risk_traceable": bool(settled) and all(
            r["economics"][oe.PRIMARY].get("max_loss_usd") is not None
            and r["economics"][oe.PRIMARY]["max_loss_usd"] <= cap_limit for r in settled),
    }
    return {"qualified": all(crit.values()), "criteria": crit,
            "failed": sorted(k for k, v in crit.items() if not v),
            "stability": _stability(rows, int(cfg["stability"]["min_bucket_n"])),
            "concentration": conc, "max_structural_loss_usd": m.get("max_structural_loss_usd"),
            "peak_concurrent_capital_usd": m.get("peak_concurrent_capital_usd"),
            "note": "Profit factor > 1 or winrate > 50 % alone never qualify; human decision required."}


# ── 4. learning ──────────────────────────────────────────────────────────────

NEGATIVE = ("NO_ECONOMIC_EDGE_AFTER_COSTS", "NOT_BETTER_THAN_BASE_STRATEGY")


def _family(hid: str, signals: list[dict], root: Path) -> str:
    sig = next((s for s in signals if s["hypothesis_id"] == hid), None)
    if sig is None:
        return "unknown"
    if sig["hypothesis_source"] == "ml_registry":
        spec = al._ml_registry(root).get(hid) or {}
        feats = spec.get("features") or spec.get("rule_feature") or "?"
        return f"ml:{spec.get('model')}:{feats if isinstance(feats, str) else '+'.join(map(str, feats))}"
    if sig["hypothesis_source"] == "autonomous_factor":
        return f"auto:{hid}"
    return sig["hypothesis_source"]


def learning(root: Path, econ: dict, results: list[dict], link_cov: dict, cfg: dict) -> dict:
    """Deterministic lessons from rejected and confirmed hypotheses (no LLM)."""
    root = Path(root)
    data = al.read_all(root)
    signals = data["signals"]
    fam_fail, fam_ok = Counter(), Counter()
    struct = defaultdict(Counter)
    for hid, per in (econ.get("hypotheses") or {}).items():
        if hid == al.CONTROL_ID:
            continue
        for st, e in per.items():
            label = e["research_decision"]["label"]
            struct[st][label] += 1
            if label in NEGATIVE:
                fam_fail[_family(hid, signals, root)] += 1
            elif label == "CANDIDATE_FOR_PROSPECTIVE_VALIDATION":
                fam_ok[_family(hid, signals, root)] += 1
    for r in results:
        fam = _family(r.get("hypothesis_id", ""), signals, root)
        if r.get("status") in ("FALSIFIED", "NOT_CONFIRMED"):
            fam_fail[fam] += 1
        elif r.get("classification") == "QUALIFIED_FOR_HUMAN_REVIEW":
            fam_ok[fam] += 1
    link = al.config(root)
    rows, _ = oe.trades(root, data, link)
    reg = defaultdict(list)
    for r in rows:
        if r["status"] == "SETTLED" and r["hypothesis_id"] != al.CONTROL_ID:
            reg[r["regime"]].append(r["economics"][oe.PRIMARY]["return"])
    n_min = int(cfg["learning"]["min_regime_n"])
    problematic = {k: {"n": len(v), "mean": round(sum(v) / len(v), 6)} for k, v in sorted(reg.items())
                   if len(v) >= n_min and sum(v) / len(v) < 0}
    unsuitable = {st: dict(c) for st, c in struct.items()
                  if sum(c[x] for x in NEGATIVE) > max(1, sum(v for k, v in c.items() if k != "NEED_MORE_DATA")) / 2}
    missing = Counter()
    for per in (econ.get("hypotheses") or {}).values():
        for e in per.values():
            missing.update(e["coverage"]["failure_reasons"])
            missing.update({f"ABSTAIN:{k}": v for k, v in e["coverage"]["abstain_reasons"].items()})
    for h in (link_cov.get("hypotheses") or {}).values():
        if h.get("overdue_decision"):
            missing["DECISION_WINDOW_OVERDUE"] += h["overdue_decision"]
    if (link_cov.get("pre_link") or {}).get("ml_prediction_rows"):
        missing["PRE_LINK_PREDICTIONS_WITHOUT_OPTION_DECISION"] = link_cov["pre_link"]["ml_prediction_rows"]
    if not (link_cov.get("executed") or {}).get("n_attributed_ex_ante"):
        missing["NO_EXECUTED_TRADES_WITH_EX_ANTE_SIGNAL_ID"] = 1
    feat_null, feat_n = Counter(), Counter()
    for s in signals:
        for k, v in (s.get("features") or {}).items():
            feat_n[k] += 1
            feat_null[k] += v is None
    unusable = sorted(k for k in feat_n if feat_n[k] >= 10 and feat_null[k] / feat_n[k] > 0.5)
    unavailable = sum(s.get("features_status") == "UNAVAILABLE" for s in signals)
    suggestions = []
    for hid, per in sorted((econ.get("hypotheses") or {}).items()):
        if hid == al.CONTROL_ID:
            continue
        for st, e in per.items():
            for regime, x in (e.get("by_regime") or {}).items():
                if (x.get("n") or 0) >= n_min and (x.get("net_ev_return") or 0) > 0 \
                        and e["research_decision"]["label"] in NEGATIVE:
                    suggestions.append({"idea": f"{hid} restricted to regime {regime} ({st})",
                                        "basis": f"positive only in {regime} (n={x['n']}) on exploratory data",
                                        "status": "SUGGESTED_NOT_REGISTERED",
                                        "rule": "needs a new preregistered hypothesis within the explorer budget"})
    return {
        "schema": "alpha-option-learning-v1",
        "repeatedly_failing_feature_combinations": {k: v for k, v in fam_fail.most_common() if v >= 2},
        "failing_feature_combinations": dict(fam_fail.most_common()),
        "promising_feature_combinations": dict(fam_ok.most_common()),
        "problematic_regimes": problematic,
        "economically_unsuitable_structures": unsuitable,
        "structure_labels": {st: dict(c) for st, c in struct.items()},
        "missing_data": dict(missing.most_common()),
        "existing_features_not_usable": unusable,
        "signals_without_pit_features": unavailable,
        "suggested_hypotheses": suggestions[: int(cfg["learning"]["max_suggested_hypotheses"])],
        "budget_rule": "Suggestions are never auto-registered; new hypotheses stay within the explorer budget.",
    }


# ── weekly entry point ───────────────────────────────────────────────────────

def run(root: Path = Path("."), *, econ: dict | None = None, link_cov: dict | None = None,
        now: datetime | None = None, persist: bool = True) -> dict:
    root, now = Path(root), now or datetime.now(timezone.utc)
    cfg = config(root)
    econ = econ if econ is not None else oe.evaluate(root, now=now, persist=False)
    link_cov = link_cov if link_cov is not None else al.coverage(root, now=now, persist=False)
    created = register_plans(root, econ, now=now) if cfg.get("enabled") and persist else []
    results = evaluate_plans(root, now=now) if persist else []
    learn = learning(root, econ, results, link_cov, cfg)
    budget = _budget(root, cfg)
    report = {
        "schema": "alpha-option-validation-v1", "generated_at": now.isoformat(timespec="seconds"),
        "status": ("QUALIFIED_FOR_HUMAN_REVIEW" if any(r.get("classification") == "QUALIFIED_FOR_HUMAN_REVIEW"
                                                       for r in results)
                   else "NEED_MORE_DATA" if not any(r.get("status") in FINAL for r in results) else "EVALUATED"),
        "n_plans": len(results), "new_plans": created,
        "plans": results, "test_budget": budget,
        "classification_counts": dict(Counter(r.get("classification") for r in results)),
        "link_status": link_cov.get("status"), "economics_status": econ.get("status"),
        "learning": learn, "decision_type": "RESEARCH_ONLY", "order_placed": False,
        "production_influence": "NONE", "llm_calls": 0,
        "disclaimer": "No return promises. No automatically released trades. Human decision required.",
    }
    if persist:
        _atomic_json(root / REPORT, report)
        _atomic_json(root / LEARNING, {"generated_at": report["generated_at"], **learn})
    return report

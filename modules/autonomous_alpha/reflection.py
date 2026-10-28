"""Deterministic research reflection; no agent calls or production authority.

Produces bounded research-only recommendations, not model promotions. Missing
evidence is explicit and cannot be interpreted as successful research.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from . import explorer
from . import forward_options
from . import factor_forward
from .registry import EvidenceIntegrityError, _atomic_json

HYPOTHESES = Path("outputs/research/hypothesis_db.json")
OUTPUT = Path("outputs/research/autonomous_alpha/reflection.json")


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("not an object")
        return payload
    except (ValueError, TypeError) as exc:
        raise EvidenceIntegrityError(f"corrupt reflection source {path}: {exc}") from exc


def reflect(root: Path = Path("."), *, today=None, persist=True) -> dict:
    root = Path(root)
    trades = forward_options.evaluate(root, today=today, persist=False)
    historical = _load(root / HYPOTHESES)
    records = historical.get("hypotheses") or {}
    if not isinstance(records, dict):
        raise EvidenceIntegrityError("malformed hypothesis database")
    frozen = explorer.proposals(root)
    own = {h["id"]: h for h in frozen}
    states = Counter()
    families = defaultdict(Counter)
    missing = []
    for identifier, h in own.items():
        rec = records.get(identifier)
        if not isinstance(rec, dict):
            states["NOT_EVALUATED"] += 1
            families[h["family"]]["NOT_EVALUATED"] += 1
            continue
        state = str(rec.get("canonical_status") or rec.get("status") or "UNKNOWN").upper()
        states[state] += 1
        families[h["family"]][state] += 1
    if not (root / HYPOTHESES).exists():
        missing.append(str(HYPOTHESES))
    if trades["source_state"] == "MISSING":
        missing.append(str(forward_options.SOURCE))
    if not own:
        missing.append("NO_REGISTERED_AUTONOMOUS_FACTORS")
    # The separate forward factor ledger contains actual *matured* underlying
    # stock gross cohort spreads. These are exploratory priors, never options
    # profit, and never independent proof after repeated research.
    factor_events = factor_forward._read(root / factor_forward.OUTCOMES)
    factor_labels = defaultdict(list)
    for event in factor_events:
        row = event["payload"]
        if row.get("status") != "MATURED_GROSS_STOCK_ONLY":
            continue
        hid = row.get("hypothesis_id")
        if hid not in own:
            continue
        value = row.get("top_minus_bottom_gross")
        if not isinstance(value, (int, float)):
            continue
        factor_labels[hid].append((row["prediction_date"], float(value)))
    forward_families = defaultdict(lambda: {"successes": 0, "failures": 0,
                                            "n_hypotheses": 0, "n_cohorts": 0})
    for hid, outcomes in factor_labels.items():
        independent_dates = len({day for day, _ in outcomes})
        if independent_dates < 6:
            continue
        family = own[hid]["family"]
        v = forward_families[family]
        v["n_hypotheses"] += 1
        v["n_cohorts"] += len(outcomes)
        if sum(value for _, value in outcomes)/len(outcomes) > 0:
            v["successes"] += 1
        else:
            v["failures"] += 1
    # From option-trade-v1 one cannot infer the genetic factors' trade
    # outcome. Keep separate diagnostics, not a false factor P&L conclusion.
    all_matured_negative = bool(forward_families) and all(v["successes"] == 0 for v in forward_families.values())
    tasks = [
        {"id": "PROSPECTIVE_FACTOR_LABELS", "priority": 1,
         "reason": "Need out-of-sample, time-stamped predictions and matured labels for every frozen AUTO-* factor."},
        {"id": "OPTIONS_ATTRIBUTION_LINK", "priority": 2,
         "reason": "Observed option P&L has no immutable pre-trade factor identifier; do not attribute performance."},
    ]
    if trades["n_verified_post_registration"] < 26:
        tasks.append({"id": "MORE_EXECUTABLE_QUOTES", "priority": 3,
                      "reason": "Too few quote-verifiable post-registration option outcomes."})
    if states.get("REJECTED", 0):
        tasks.append({"id": "DIVERSIFY_REJECTED_FAMILIES", "priority": 4,
                      "reason": "Historical rejects are exploratory feedback, not proof of inverse alpha."})
    report = {
        "schema": "alpha-reflection-v1",
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_frozen_hypotheses": len(own),
        "n_historical_recorded": len(records),
        "explorer_states": dict(sorted(states.items())),
        "family_states": {k: dict(v) for k,v in sorted(families.items())},
        "real_options_diagnostics": {
            "n_verified": trades["n_verified_post_registration"],
            "n_unverified": trades["n_unverified_or_pre_registration"],
            "sample_floor_reached": trades["sample_floor_reached"],
            "outcome_net_mean_observational": trades["net_mean_verified"],
            "alpha_attribution": trades["strategy_alpha_status"],
        },
        "source_gaps": missing,
        "forward_stock_family_exploratory_counts": {k: v for k, v in sorted(forward_families.items())},
        "n_matured_stock_factor_cohorts": sum(len(xs) for xs in factor_labels.values()),
        "next_research_tasks": tasks,
        # With no independent validated option alpha, always reserve all
        # exploration slots: don't exploit unreliable historical winners.
        "exploration_slots_recommended": 2 if all_matured_negative else 1,
        "alpha_validated": False,
        "promotion_authorized": False,
        "code_changes_authorized": False,
        "llm_calls": 0,
        "production_influence": "NONE",
    }
    if persist:
        _atomic_json(root / OUTPUT, report)
    return report

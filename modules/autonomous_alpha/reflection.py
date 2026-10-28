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
    # From option-trade-v1 one cannot infer the genetic factors' trade
    # outcome. Keep separate diagnostics, not a false factor P&L conclusion.
    no_confirmed_alpha = True  # Cannot change until an independent validator exists
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
        "next_research_tasks": tasks,
        # With no independent validated option alpha, always reserve all
        # exploration slots: don't exploit unreliable historical winners.
        "exploration_slots_recommended": 2 if no_confirmed_alpha else 1,
        "alpha_validated": False,
        "promotion_authorized": False,
        "code_changes_authorized": False,
        "llm_calls": 0,
        "production_influence": "NONE",
    }
    if persist:
        _atomic_json(root / OUTPUT, report)
    return report

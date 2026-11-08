"""Deterministic research-capacity allocator for the EXISTING hypothesis pipelines.

No prediction, model, learning labels, market data or orders. Consumes only
pre-ranked, already-eligible Research Director / Hypothesis Factory dossiers.
Selects marginally informative, non-redundant research questions under the
caller's UNCHANGED budget, holding reserve, frozen protocol and validation
pipeline. Never calls a locked holdout or labels a hypothesis as validated.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
import math

from modules import research_memory as memory

VERSION = "marginal-research-value-v1"
MAX_SAME_FAMILY_PER_BATCH = 1  # stricter than lifetime family cap; no protocol changes
EXACT_DUPLICATE_SIMILARITY = 0.94  # identical feature expressions under changed names
MAX_AUDIT_ROWS = 250


def memory_as_of(entries: list[dict], cutoff: datetime) -> list[dict]:
    """Use only research-memory facts known at selection time (replay-safe).

    Old synthetic/legacy entries may lack recorded_at; the source time is
    explicitly unknown, not proof of independent forward performance.
    """
    if cutoff.tzinfo is None:
        raise ValueError("selection cutoff must be timezone aware")
    out = []
    for entry in entries:
        raw = entry.get("recorded_at")
        if raw:
            try:
                ts = datetime.fromisoformat(raw)
            except (TypeError, ValueError):
                continue
            if ts.tzinfo is None or ts > cutoff:
                continue
        out.append(entry)
    return out


def _features(h: dict) -> set[str]:
    return {v for v in memory.tokens_of({"signal": h.get("signal") or ""}) if v.startswith("f:")}


def _overlap(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _category(h: dict) -> str:
    return str(h.get("idea_source") or h.get("kind") or h.get("domain") or "unknown")


def _family(h: dict) -> str:
    # Factory families are economic mechanisms; Director candidates with no
    # explicit family are grouped by measured source finding, not random ID.
    return str(h.get("family") or h.get("source_finding") or h.get("research_id") or h.get("id"))


def _id(h: dict) -> str:
    return str(h.get("id") or h.get("research_id") or "")


def _value(h: dict, picked: list[dict]) -> dict:
    base = float(h.get("priority") or 0)
    if not math.isfinite(base) or base <= 0:
        return {"marginal_score": 0.0, "reason": "NONPOSITIVE_PRIOR_SCORE"}
    comp = h.get("priority_components") or {}
    # Posterior uncertainty is existing Research Memory evidence, not a newly
    # fitted model or a historical realized return of this candidate. With no
    # reliable family history the uncertainty prior is explicit and neutral.
    sd = comp.get("posterior_sd")
    if isinstance(sd, (float, int)) and math.isfinite(sd) and sd >= 0:
        uncertainty = min(1.0, float(sd) / 0.17)
        uncertainty_source = "existing_research_memory"
    else:
        uncertainty = 1.0
        uncertainty_source = "unmeasured_default"
    information_multiplier = 0.80 + 0.20 * uncertainty
    feats = _features(h)
    feature_overlap = max((_overlap(feats, _features(p)) for p in picked), default=0.0)
    source_n = sum(_category(p) == _category(h) for p in picked)
    # A familiar feature/source can still have real value in a new population:
    # penalize duplication rather than declaring an untested effect disproven.
    diversity_multiplier = (1 - 0.30 * feature_overlap) / (1 + 0.15 * source_n)
    raw = base * information_multiplier * diversity_multiplier
    return {"base_priority": base, "posterior_uncertainty": round(uncertainty, 4),
            "uncertainty_source": uncertainty_source,
            "information_multiplier": round(information_multiplier, 4),
            "max_feature_overlap_with_batch": round(feature_overlap, 4),
            "previous_same_source_in_batch": source_n,
            "diversity_multiplier": round(diversity_multiplier, 4),
            "marginal_score": round(raw, 8), "reason": "ELIGIBLE",
            "interpretation": "expected research information proxy, not realized or proven alpha"}


def select(candidates: list[dict], *, slots: int, exploratory_slots: int = 0,
           max_per_family: int = MAX_SAME_FAMILY_PER_BATCH) -> tuple[list[dict], dict]:
    """Greedy marginal value under the pre-existing, frozen caller test budget.

    Stage 1: reserve up to exploratory_slots for eligible exploratory ideas.
    Stage 2: allocate remaining slots across ALL eligible candidates.
    Hard block duplicate research questions, same-family batch repeats and
    expressions that are indistinguishable within a batch. Never fill a
    reserved empty slot with an unqualified candidate merely to hit a target.
    All tie breaks stable by hypothesis ID, not iteration order or hash seed.
    """
    if not (0 <= int(exploratory_slots) <= int(slots) and 0 <= int(slots) <= 100):
        raise ValueError("invalid bounded research-selection budget")
    if not (1 <= int(max_per_family) <= 100):
        raise ValueError("invalid per-batch family limit")
    ids = [_id(h) for h in candidates]
    if any(not k for k in ids) or len(set(ids)) != len(ids):
        raise ValueError("missing or duplicated research candidate ID")
    picked: list[dict] = []
    rejections: dict[str, str] = {}
    trace: list[dict] = []

    def eligible(h: dict) -> str | None:
        if sum(_family(x) == _family(h) for x in picked) >= max_per_family:
            return "FAMILY_ALREADY_SELECTED_IN_BATCH"
        if any(h.get("signal") and p.get("signal") and
               memory.similarity({"signal": h["signal"], "direction": h.get("direction")},
                                 {"signal": p["signal"], "direction": p.get("direction")})
               >= EXACT_DUPLICATE_SIMILARITY for p in picked):
            return "NEAR_DUPLICATE_EXPRESSION_IN_BATCH"
        return None

    def choose(pool: list[dict], purpose: str, limit: int) -> None:
        while len(picked) < limit:
            available = []
            for h in pool:
                if h in picked:
                    continue
                reason = eligible(h)
                if reason:
                    rejections[_id(h)] = reason
                    continue
                score = _value(h, picked)
                if score["marginal_score"] > 0:
                    available.append((score["marginal_score"], _id(h), h, score))
            if not available:
                return
            _, _, best, score = sorted(available, key=lambda x: (-x[0], x[1]))[0]
            picked.append(best)
            rejections.pop(_id(best), None)
            best["selection_audit"] = {**score, "selection_version": VERSION,
                                       "decision": "SELECTED", "purpose": purpose,
                                       "rank": len(picked),
                                       "basis": "pre-test evidence and Research Memory only"}

    capacity = max(0, int(slots))
    reserve = min(capacity, int(exploratory_slots))
    choose([h for h in candidates if h.get("exploratory")], "EXPLORATORY_RESERVE", reserve)
    choose(candidates, "MARGINAL_RESEARCH_VALUE", capacity)
    selected = {_id(h) for h in picked}
    for h in candidates:
        if _id(h) in selected:
            continue
        reason = rejections.get(_id(h)) or ("LOWER_MARGINAL_INFORMATION_VALUE"
                                            if len(picked) >= capacity else "NO_VALID_SLOT")
        h["selection_audit"] = {**_value(h, picked), "selection_version": VERSION,
                                "decision": "NOT_SELECTED", "reason": reason,
                                "basis": "pre-test evidence and Research Memory only"}
    audit = {"version": VERSION, "slots": capacity, "exploratory_slots": reserve,
             "allocated": len(picked), "unfilled": capacity - len(picked),
             "exploratory_allocated": sum(bool(h.get("exploratory")) for h in picked),
             "distinct_families": len({_family(h) for h in picked}),
             "distinct_idea_sources": len({_category(h) for h in picked}),
             "no_locked_holdout_used": True, "no_new_model": True,
             "no_statistical_proof_from_selection": True,
             "selection_order": [_id(h) for h in picked],
             "deferred_reasons": dict(Counter(h["selection_audit"]["reason"]
                                             for h in candidates if _id(h) not in selected)),
             "constraints": {"max_same_family_per_batch": max_per_family,
                             "near_duplicate_similarity": EXACT_DUPLICATE_SIMILARITY}}
    return picked, audit

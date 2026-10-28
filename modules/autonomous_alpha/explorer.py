"""Bounded, cost-free genetic feature discovery on point-in-time research fields.

Research ONLY. Historical outcomes are used to allocate search effort; they
ARE NOT an independent validation set after adaptive selection. No LLM,
arbitrary source-code execution, option orders, risk/promotion modifications,
or historical re-labeling. Every hypothesis is preregistered with a content
hash BEFORE the lab evaluates it.

Only the new monthly batch is allowed to increase the multiplicity denominator.
Archived proposals are always replayed to the lab, so old negative tests are
never silently dropped from BH correction.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

from modules import research_lab as lab
from . import registry as reg

POLICY = Path("config/autonomous_alpha_exploration.yaml")
PROPOSALS = Path("outputs/research/autonomous_alpha/proposals.jsonl")
LATEST = Path("outputs/research/autonomous_alpha/exploration_plan.json")
HISTORICAL_DB = Path("outputs/research/hypothesis_db.json")
MAX_NEW_HARD = 3
MAX_LIFETIME_HARD = 40


def _cfg(root: Path) -> dict:
    x = yaml.safe_load((root / POLICY).read_text(encoding="utf-8"))
    if not isinstance(x, dict) or x.get("version") != "explore-v1":
        raise reg.EvidenceIntegrityError("exploration policy invalid")
    if int(x.get("monthly_new_hypotheses", 0)) not in range(0, MAX_NEW_HARD + 1):
        raise reg.EvidenceIntegrityError("exploration monthly hard cap exceeded")
    if int(x.get("lifetime_hypotheses", 0)) not in range(0, MAX_LIFETIME_HARD + 1):
        raise reg.EvidenceIntegrityError("exploration lifetime hard cap exceeded")
    if not 0 <= float(x.get("exploratory_fraction", 0.0)) <= 1:
        raise reg.EvidenceIntegrityError("invalid exploration fraction")
    parents = x.get("parents")
    if not isinstance(parents, list) or len(parents) < 2:
        raise reg.EvidenceIntegrityError("missing parent features")
    seen = set()
    for p in parents:
        if not isinstance(p, dict) or set(("name","feature","direction")) - set(p):
            raise reg.EvidenceIntegrityError("invalid parent")
        feature = p["feature"]
        if feature not in lab.ALLOWED_NAMES:
            raise reg.EvidenceIntegrityError("unsafe/unavailable PIT feature: " + str(feature))
        if p["direction"] not in (-1, 1) or p["name"] in seen:
            raise reg.EvidenceIntegrityError("parent direction/identity invalid")
        seen.add(p["name"])
    return x


def _spec_hash(spec: dict) -> str:
    return reg.digest(spec)


def _genome(p: dict, q: dict, operator: str) -> dict:
    a, b = sorted((p, q), key=lambda v: v["name"])
    a_expr = f"rank({a['feature']})"
    b_expr = f"rank({b['feature']})"
    if operator == "blend":
        # Add both features with their ex-ante sign, without fitting weights.
        signal = f"({a_expr} * {a['direction']}) + ({b_expr} * {b['direction']})"
    elif operator == "interaction":
        # Cross-sectional combination with prespecified economic sign.
        signal = f"({a_expr} * {a['direction']}) * ({b_expr} * {b['direction']})"
    elif operator == "regime":
        # Non-trainable regime threshold, known at decision time.
        signal = (f"(({a_expr} * {a['direction']}) + "
                  f"({b_expr} * {b['direction']})) * step(vix - 20)")
    else:
        raise ValueError(operator)
    lab.validate_expr(signal)
    family = "x_".join((a["name"], b["name"]))
    contract = {"signal": signal, "direction": 1, "family": family,
                "parents": [a["feature"], b["feature"]], "operator": operator,
                "horizon": 20, "population": "PIT_S&P500"}
    h = _spec_hash(contract)
    return {
        "id": "AUTO-" + h[:12].upper(),
        "title": f"Automated {operator}: {a['name']} × {b['name']}",
        "signal": signal, "direction": 1, "family": family,
        "domain": "point_in_time_factor_recombination",
        "parents": [a["feature"], b["feature"]], "operator": operator,
        "spec_hash": h, "contract": contract,
        "statement": "Exploratory combination of preregistered parent signs; not established alpha.",
        "rationale": "Test a cost-adjusted cross-sectional interaction without estimating new signs from historical results.",
        "source": "autonomous_explorer",
    }


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    found, by_id = [], {}
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            e = json.loads(line)
            s = e["hypothesis"]
            if not isinstance(s, dict) or e.get("spec_hash") != _spec_hash(s["contract"]):
                raise ValueError("content hash mismatch")
            if s.get("spec_hash") != e["spec_hash"] or not s.get("id"):
                raise ValueError("hypothesis identity mismatch")
            lab.validate_expr(s["signal"])
            if e.get("id") != s["id"]:
                raise ValueError("id mismatch")
            if s["id"] in by_id:
                if by_id[s["id"]] != e:
                    raise ValueError("conflicting proposal id")
                continue  # Git union can duplicate identical JSONL rows
            by_id[s["id"]] = e
            found.append(e)
        except (KeyError, ValueError, TypeError) as exc:
            raise reg.EvidenceIntegrityError(f"corrupt explorer proposal line {n}: {exc}") from exc
    return found


def proposals(root: Path = Path(".")) -> list[dict]:
    """Load all frozen proposals, never only the latest month's 'winners'."""
    return [e["hypothesis"] for e in _read(Path(root) / PROPOSALS)]


def _baseline_signals(root: Path) -> set[tuple[str, int]]:
    f = root / HISTORICAL_DB
    try:
        historical = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        return {(lab.signal_key(r["signal"]), int(r.get("direction", 1)))
                for r in (historical.get("hypotheses") or {}).values()
                if r.get("signal") and r.get("direction")}
    except (ValueError, KeyError, TypeError) as exc:
        raise reg.EvidenceIntegrityError("historical research DB malformed: " + str(exc)) from exc


def _outcomes(root: Path) -> dict:
    p = root / HISTORICAL_DB
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("hypotheses") or {}
    except (ValueError, TypeError) as exc:
        raise reg.EvidenceIntegrityError(f"invalid hypothesis DB: {exc}") from exc


def _evidence_counts(outcomes: dict, parents: list[dict]) -> tuple[int, int]:
    """Historical feedback can allocate EXPLORATION only; never prove an alpha."""
    keywords = {p["feature"] for p in parents}
    success = failed = 0
    for r in outcomes.values():
        signal = str(r.get("signal") or "")
        if not any(k in signal for k in keywords):
            continue
        status = str(r.get("canonical_status") or r.get("status") or "").upper()
        if status in ("ACCEPTED", "ROBUST"):
            success += 1
        elif status in ("REJECTED", "REJECTED_LOCKED", "NOT_SIGNIFICANT_AFTER_FDR"):
            failed += 1
    return min(success, 5), min(failed, 20)


def select(root: Path = Path("."), now: datetime | None = None) -> dict:
    root = Path(root)
    cfg = _cfg(root)
    now = now or datetime.now(timezone.utc)
    month = now.astimezone(timezone.utc).strftime("%Y-%m")
    existing = _read(root / PROPOSALS)
    ids = {e["id"] for e in existing}
    if not cfg.get("enabled", False):
        return {"month": month, "new": [], "reason": "disabled", "prior_total": len(existing)}
    if any(e.get("batch") == month for e in existing):
        return {"month": month, "new": [], "reason": "already_registered_this_month",
                "prior_total": len(existing)}
    max_new = min(int(cfg["monthly_new_hypotheses"]),
                  int(cfg["lifetime_hypotheses"])-len(existing))
    if max_new <= 0:
        return {"month": month, "new": [], "reason": "budget_exhausted",
                "prior_total": len(existing)}
    parents = cfg["parents"]
    old_signals = _baseline_signals(root)
    previous = Counter(e["hypothesis"]["family"] for e in existing)
    outcomes = _outcomes(root)
    candidates = []
    for a,b in itertools.combinations(sorted(parents,key=lambda x:x["name"]), 2):
        for op in ("blend", "interaction", "regime"):
            h = _genome(a,b,op)
            if h["id"] in ids or previous[h["family"]] >= int(cfg["max_variants_per_family"]):
                continue
            if (lab.signal_key(h["signal"]), 1) in old_signals:
                continue
            successes, failures = _evidence_counts(outcomes, (a,b))
            # Thompson-sampling using one historical, NON-CONFIRMATORY prior.
            seed = int(hashlib.sha256((str(cfg["seed"])+month+h["id"]).encode()).hexdigest()[:16],16)
            rng = random.Random(seed)
            score = rng.betavariate(1+successes, 4+failures)
            novelty = 1.0 / (1.0 + previous[h["family"]])
            candidates.append((score*novelty, seed, h))
    candidates.sort(key=lambda x: (-x[0], x[1]))
    selected = []
    exploration_slots = math.ceil(max_new * float(cfg["exploratory_fraction"]))
    # Unbiased reproducible candidate lottery reserves exploration capacity.
    lottery = sorted(candidates, key=lambda x:x[1])
    reserved = lottery[:exploration_slots]
    chosen = reserved + [c for c in candidates if c not in reserved]
    for _, _, h in chosen:
        if len(selected) >= max_new:
            break
        if previous[h["family"]] + sum(v["family"]==h["family"] for v in selected) >= int(cfg["max_variants_per_family"]):
            continue
        selected.append(h)
    new = [{"id": h["id"], "batch": month, "registered_at": now.astimezone(timezone.utc).isoformat(),
            "spec_hash": h["spec_hash"], "hypothesis": h} for h in selected]
    return {"month": month, "new": new, "reason": "registered" if new else "no_novel_variants",
            "prior_total": len(existing), "candidate_pool": len(candidates)}


def register(root: Path = Path("."), now: datetime | None = None) -> dict:
    root = Path(root)
    plan = select(root, now)
    path = root / PROPOSALS
    if plan["new"]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a",encoding="utf-8") as fh:
            for event in plan["new"]:
                fh.write(reg.canonical(event)+"\n")
            fh.flush()
            import os
            os.fsync(fh.fileno())
    # Verify append + source identity before publishing the plan for the lab.
    all_entries = _read(path)
    result = {"generated": (now or datetime.now(timezone.utc)).isoformat(),
              "month": plan["month"], "status": plan["reason"],
              "n_registered_new": len(plan["new"]), "total_proposals": len(all_entries),
              "new_ids": [h["id"] for h in plan["new"]],
              "research_only": True, "production_effect": "NONE",
              "historical_selection_is_adaptive": True,
              "requires_independent_forward_validation": True}
    reg._atomic_json(root / LATEST, result)
    return result

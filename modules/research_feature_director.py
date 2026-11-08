"""FD3 Phase 3: cost-aware autonomous *research planning*, no live-trading agent.

Consumes immutable FD3 feature coverage and Phase-2 prospective diagnostics
and existing Research Memory. Emits bounded, traceable hypotheses/actions for
human review. Does NOT rewrite a preregistered experiment or spend alpha,
download data, send orders, modify champion/LLM/RL, or declare alpha.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

from modules import research_feature_discovery as fd

ROOT = Path("outputs/research/feature_discovery_v3")
MEMORY = Path("outputs/research/research_memory.jsonl")
VERSION = "fd3-autonomous-director-v1"
MAX_ACTIONS = 2
COST = {"fred_liquidity": 1.0, "fomc_calendar": 0.5,
        "bls_release_calendar": 0.5, "bea_release_calendar": 0.5}
VALID_ACTIONS = {"COLLECT_PIT", "CONTINUE_FORWARD", "DESIGN_OPTION_TRIAL",
                 "AUDIT_STABILITY", "REVIEW_LOW_VALUE"}


def _digest(obj: dict) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _read(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _memory_for_feature(entries: list[dict], feature: str) -> dict:
    related = [e for e in entries if e.get("kind") in ("feature_discovery_v3", "feature_director_v1")
               and (e.get("spec") or {}).get("signal") == feature]
    return {"related_records": len(related),
            "previous_falsifications": sum(e.get("status") in ("REJECTED", "NOT_ROBUST")
                                           for e in related),
            "prior_forward_confirmed": False}  # FD3 cannot validate alpha


def _status_bucket(coverage: float, count: int, windows: int) -> str:
    return ("MISSING" if count == 0 else
            "SPARSE" if count < 26 or coverage < 0.5 else
            "FORWARD_0" if windows == 0 else
            "FORWARD_1_7" if windows < 8 else
            "FORWARD_8_15" if windows < 16 else
            "FORWARD_16_23" if windows < 24 else "FORWARD_24_PLUS")


def propose(status: dict, validation: dict, memory: list[dict]) -> dict:
    """Deterministically choose at most two *investigation actions*.

    All candidate keys are fixed in FD3; no ex-post feature invention.
    Coverage/maturity are measured, while priority is a transparent heuristic
    for research attention, NOT a fitted expected-alpha estimate.
    """
    p, digest = fd.load_protocol()
    if status.get("protocol_sha256") != digest:
        raise ValueError("FD3 Phase3: upstream protocol hash mismatch")
    if validation and validation.get("protocol_sha256") != digest:
        raise ValueError("FD3 Phase3: Phase2/FD3 protocol mismatch")
    if validation and validation.get("global_alpha_ledger_slot_allocated") is not False:
        raise ValueError("FD3 Phase3: unexpected confirmatory allocation")
    out = []
    features = status.get("features") or {}
    for h in p["hypotheses"]:
        feat, source, hid = h["feature"], h["source"], h["id"]
        cov = features.get(feat) or {}
        fraction = float(cov.get("coverage") or 0.0)
        count = int(cov.get("dates_available") or 0)
        if not (0 <= fraction <= 1) or count < 0:
            raise ValueError("FD3 Phase3: invalid measured source coverage")
        trial = (validation.get("hypotheses") or {}).get(hid) or {}
        windows = int(trial.get("independent_forward_windows") or 0)
        looks = trial.get("looks") or []
        last = looks[-1] if looks else {}
        paired = last.get("paired_stock_mean")
        p_local = last.get("local_bonferroni_diagnostic_p")
        ci = last.get("paired_stock_95pct_bootstrap") or []
        uncertainty = 1 / math.sqrt(max(1, windows))
        if count < 26 or fraction < 0.5:
            action, reason = "COLLECT_PIT", "Coverage insufficient for bounded challenger evaluation"
            severity = min(1.0, max((26 - count) / 26, 1 - fraction))
        elif windows < 8:
            action, reason = "CONTINUE_FORWARD", "Need non-overlapping frozen 20-trading-day outcomes"
            severity = 0.65 + 0.35 * (8 - windows) / 8
        elif (paired is not None and p_local is not None and ci and
              float(paired) > 0 and float(p_local) < 0.05 and float(ci[0]) > 0):
            action, reason = "DESIGN_OPTION_TRIAL", "Positive stock-only diagnostic; new global-alpha-funded options trial needs review"
            severity = 0.75
        elif (paired is not None and windows >= 16 and float(paired) <= 0):
            action, reason = "REVIEW_LOW_VALUE", "Paired forward stock diagnostic non-positive; do not automatically disable source"
            severity = 0.55
        else:
            action, reason = "AUDIT_STABILITY", "Diagnostic evidence not independently decisive"
            severity = 0.50 + 0.25 * uncertainty
        if action not in VALID_ACTIONS:
            raise ValueError("Unregistered research action")
        mem = _memory_for_feature(memory, feat)
        memory_multiplier = 1 / (1 + 0.5 * mem["previous_falsifications"])
        priority = round(severity * max(0.2, fraction) * memory_multiplier / (1 + COST[source]), 6)
        if action == "COLLECT_PIT":
            priority = round(severity / (1 + COST[source]), 6)
        bucket = _status_bucket(fraction, count, windows)
        dossier = {
            "hypothesis_id": hid, "feature": feat, "source": source,
            "action": action, "status_bucket": bucket, "status": "RESEARCH_TASK_ONLY",
            "reason": reason, "priority": priority, "priority_is_heuristic": True,
            "coverage": round(fraction, 4), "data_dates": count,
            "forward_independent_windows": windows,
            "stock_diagnostic_p_local": p_local,
            "stock_mean_paired_delta": paired,
            "memory": mem,
            "proposed_next_study": {
                "pre_register_separate_protocol": action == "DESIGN_OPTION_TRIAL",
                "require_owner_approval": True,
                "require_global_alpha_ledger_slot": action == "DESIGN_OPTION_TRIAL",
                "require_archived_rth_quotes": action == "DESIGN_OPTION_TRIAL",
                "production_influence": "NONE",
            },
        }
        dossier["decision_key"] = _digest({
            "v": VERSION, "protocol": digest, "h": hid,
            "action": action, "bucket": bucket,
        })[:20]
        out.append(dossier)
    ranked = sorted(out, key=lambda x: (-x["priority"], x["hypothesis_id"]))
    return {"version": VERSION, "protocol_sha256": digest,
            "research_only": True, "orders_created": 0,
            "global_alpha_ledger_slot_allocated": False,
            "all_candidates": ranked, "selected": ranked[:MAX_ACTIONS],
            "max_actions": MAX_ACTIONS,
            "decision_rule": "cost_aware_coverage_and_independent_forward_priority_v1",
            "new_hypotheses_registered": 0,
            "champion_or_trade_modified": False}


def read_journal(path: Path) -> tuple[list[dict], str]:
    prev, out = "GENESIS", []
    for i, obj in enumerate(fd._read_jsonl(path)):
        if obj.get("prev_hash") != prev or obj.get("hash") != _digest(obj.get("payload")):
            raise ValueError(f"FD3 Phase3 research audit chain broken: line {i+1}")
        prev = _digest(obj)
        out.append(obj["payload"])
    if len({r["decision_key"] for r in out}) != len(out):
        raise ValueError("FD3 Phase3 duplicate decision keys in immutable journal")
    return out, prev


def run(*, root: Path = ROOT, memory_path: Path = MEMORY,
        now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    source = root / fd.REPORT
    if not source.exists():
        return {"version": VERSION, "status": "MISSING_FD3", "selected": [], "new_decisions": 0}
    status = _read(source, {})
    validation = _read(root / "paired_validation.json", {})
    memory = fd._read_jsonl(memory_path)
    plan = propose(status, validation, memory)
    plan["as_of"] = now.isoformat()
    journal = root / "director_decisions.jsonl"
    past, prev = read_journal(journal)
    existing = {r["decision_key"] for r in past}
    fresh = 0
    for candidate in plan["selected"]:
        if candidate["decision_key"] in existing:
            continue
        payload = {**candidate, "protocol_sha256": plan["protocol_sha256"],
                   "recorded_at": now.isoformat(), "research_only": True,
                   "no_auto_registration": True}
        event = {"prev_hash": prev, "payload": payload, "hash": _digest(payload)}
        journal.parent.mkdir(parents=True, exist_ok=True)
        with journal.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(event, sort_keys=True) + "\n")
            fp.flush()
            os.fsync(fp.fileno())
        prev = _digest(event)
        existing.add(candidate["decision_key"])
        fresh += 1
    plan["new_decisions"] = fresh
    (root / "director_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return plan


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", default=None)
    args = parser.parse_args(argv)
    now = (datetime.fromisoformat(args.as_of).replace(tzinfo=timezone.utc)
           if args.as_of else datetime.now(timezone.utc))
    out = run(now=now)
    print(json.dumps({"status": out.get("status", "RESEARCH_ONLY"),
                      "selected": [r["hypothesis_id"] for r in out["selected"]],
                      "new_decisions": out["new_decisions"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

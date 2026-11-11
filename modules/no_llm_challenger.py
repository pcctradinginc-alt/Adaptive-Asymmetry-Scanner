"""Prospective no-LLM options challenger using ONLY a pre-existing acquisition model.

Research/paper ranking, no orders, no Champion changes. If the model is missing,
stale, fitted on the signal day or later, or the contract quote is unusable,
explicitly ABSTAIN. Never backfill a prediction after seeing its outcome.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from datetime import date, datetime, timezone
from pathlib import Path

from modules import option_learner as ol
from modules import research_evidence_discovery as red

OUT = Path("outputs/research/no_llm_challenger")
LEDGER = Path("outputs/candidate_ledger")
MODEL = Path("outputs/research/option_learner")
POLICY = "no-llm-acquisition-ridge-v1"
MAX_WATCH = 5


def predict(x: dict, model: dict) -> float | None:
    try:
        values = [float(x.get(n, model["median"][i])) for i, n in enumerate(model["names"])]
        score = float(model["intercept"]) + sum(
            ((v - float(model["mu"][i])) / float(model["sd"][i])) * float(model["coef"][i])
            for i, v in enumerate(values))
        return round(score, 6) if math.isfinite(score) else None
    except (TypeError, ValueError, IndexError, KeyError, ZeroDivisionError):
        return None


def decide(rows: list[dict], day: date, model: dict | None) -> dict:
    report = {"policy": POLICY, "day": day.isoformat(),
              "mode": "research_only", "production_influence": "NONE",
              "orders": 0, "model_id": None, "action": "ABSTAIN",
              "candidates_seen": 0, "eligible_contracts": 0, "watchlist": []}
    if not model:
        return {**report, "reason": "no_prior_model"}
    try:
        cutoff = date.fromisoformat(model["label_cutoff"])
        if cutoff >= day or (day - cutoff).days > ol.ACQ_MAX_AGE_DAYS:
            return {**report, "reason": "model_not_prior_or_stale"}
    except (KeyError, ValueError, TypeError):
        return {**report, "reason": "invalid_model"}
    report["model_id"] = model.get("model_id")
    candidates = [r for r in rows if r.get("date") == day.isoformat()]
    report["candidates_seen"] = len(candidates)
    scores = []
    for row in candidates:
        # Non-production research census is an existing, observed bullish call.
        # Do NOT choose the winner from several expiries after observing outcomes.
        option = row.get("real_option") or {}
        rs = row.get("real_strategy") or {}
        if not isinstance(option, dict) or not isinstance(rs, dict):
            continue
        if not option.get("symbol") or not rs.get("legs"):
            continue
        # Do not claim filled P&L from a quote. Entry requires positive net debit.
        debit = rs.get("net_debit_entry")
        try:
            if float(debit) <= 0 or not math.isfinite(float(debit)):
                continue
        except (ValueError, TypeError):
            continue
        score = predict(ol.acquisition_features(row), model)
        if score is None:
            continue
        report["eligible_contracts"] += 1
        scores.append({"ticker": row.get("ticker"), "signal_id": row.get("signal_id"),
                       "contract": option["symbol"], "model_net_estimate": score,
                       "entry_ask_debit": debit,
                       "sample": row.get("option_sample"),
                       "entry_status": option.get("entry_status"),
                       "snapshot_selected": row.get("snapshot_selected")})
    # Fully reproducible ranking; never use outcomes as a rank input.
    scores.sort(key=lambda x: (-x["model_net_estimate"], str(x["ticker"]), str(x["signal_id"])))
    report["watchlist"] = scores[:MAX_WATCH]
    report["action"] = "PAPER_ONLY" if any(x["model_net_estimate"] > 0 for x in scores[:MAX_WATCH]) else "ABSTAIN"
    report["reason"] = "research_only_unvalidated"
    report["prediction_id"] = hashlib.sha256(json.dumps({
        "day": report["day"], "model_id": report["model_id"], "watchlist": report["watchlist"],
    }, sort_keys=True).encode()).hexdigest()[:16]
    return report


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--day", type=date.fromisoformat, default=date.today())
    p.add_argument("--ledger", type=Path, default=LEDGER)
    p.add_argument("--model-dir", type=Path, default=MODEL)
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args(argv)
    target = args.out / f"{args.day.isoformat()}.json"
    # Immutable: an existing prediction cannot be recomputed after future labels arrive.
    if target.exists():
        print("Already registered; refusing to overwrite prospective prediction: " + str(target))
        return 0
    candidates = red.load(args.ledger, args.day)
    model = ol.load_acquisition_model(args.day, args.model_dir)
    report = decide(candidates, args.day, model)
    report["recorded_at"] = datetime.now(timezone.utc).isoformat()
    args.out.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "watchlist"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

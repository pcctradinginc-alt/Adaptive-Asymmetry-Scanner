"""Evaluate frozen no-LLM proposals ONLY against VERIFIED archived option outcomes.

Never re-rank on realized prices; missing labels remain PENDING / UNVERIFIED.
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

from modules import option_learner as ol

DECISIONS = Path("outputs/research/no_llm_challenger")
RESULTS = Path("outputs/research/no_llm_evaluation")


def evaluate(predictions: list[dict], candidates: list[dict]) -> dict:
    lookup = {}
    for c in candidates:
        key = (c.get("date"), str(c.get("signal_id")), c.get("ticker"))
        lookup[key] = c
    observed = []
    for p in predictions:
        day = p.get("day")
        for pick in p.get("watchlist", []):
            score = pick.get("model_net_estimate")
            if score is None or score <= 0:
                continue
            k = (day, str(pick.get("signal_id")), pick.get("ticker"))
            c = lookup.get(k)
            matches = ([e for e in c.get("exprs", []) if pick.get("contract") in (e.get("contract") or [])]
                       if c else [])
            verified = [e for e in matches if e.get("status") == "VERIFIED"]
            if verified:
                # Precommitted option identity; never choose the best outcome.
                e = verified[0]
                observed.append({"day": day, "ticker": pick.get("ticker"),
                                 "contract": pick.get("contract"), "status": "VERIFIED",
                                 "net_return": e["y"], "label_day": str(e["label_day"])})
            else:
                reasons = sorted({e.get("status") for e in matches if e.get("status")})
                observed.append({"day": day, "ticker": pick.get("ticker"),
                                 "contract": pick.get("contract"), "status": "UNVERIFIED",
                                 "reason": reasons or ["NO_MATCHING_EXPRESSION"]})
    groups = defaultdict(list)
    for row in observed:
        if row["status"] == "VERIFIED":
            groups[row["day"]].append(row["net_return"])
    daily_means = [sum(vals)/len(vals) for _day, vals in sorted(groups.items())]
    return {"schema": "no_llm_forward_v1", "mode": "research_only",
            "production_influence": "NONE",
            "predictions": len(predictions), "selected": len(observed),
            "verified": sum(r["status"] == "VERIFIED" for r in observed),
            "unverified": sum(r["status"] != "VERIFIED" for r in observed),
            "observed_entry_days": len(daily_means),
            "mean_of_day_means": sum(daily_means)/len(daily_means) if daily_means else None,
            "never_trade_baseline": 0.0,
            "claim_proven_alpha": False,
            "rows": observed}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--as-of", default=date.today().isoformat())
    p.add_argument("--horizon", type=int, choices=[5,20], default=5)
    p.add_argument("--decisions", type=Path, default=DECISIONS)
    p.add_argument("--out", type=Path, default=RESULTS)
    args = p.parse_args(argv)
    as_of = date.fromisoformat(args.as_of)
    reports = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(args.decisions.glob("????-??-??.json"))
               if f.stem <= args.as_of]
    _rows, cands, _verified = ol._load(as_of, args.horizon, ol.red.LEDGER, ol.red.QUOTES)
    result = evaluate(reports, cands)
    result.update(horizon=args.horizon, as_of=args.as_of,
                  generated_at=datetime.now(timezone.utc).isoformat())
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.as_of}_{args.horizon}d.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

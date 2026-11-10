"""LLM-free, prospectively frozen long-call selection research.

This deliberately does NOT use LLM-derived impact, direction, surprise, or
gate decisions. Picks from the representative research option sample only.
The fixed exploratory ranking is NOT a trained model or a validated signal.

A selection is made AFTER today's scanner and cannot buy at an older quote.
Paper entry requires a usable *later-session* archived ask; exit requires a
separately archived later bid. No extra market data requests or live trades.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from modules import option_quotes as oq
from modules import research_evidence_discovery as red

ROOT = Path("outputs/research/llm_free_challenger")
PREDS = ROOT / "predictions"
REPORT = ROOT / "evaluation.json"
POLICY = "llm-free-fixed-v1"
MAX_PICKS = 10
RANKED_PICKS = 6
COMMISSION_ROUNDTRIP_PER_CONTRACT = 1.30  # $0.65/leg/side, 100 shares


def _num(x, fallback=None):
    try:
        y = float(x)
        return y if math.isfinite(y) else fallback
    except (TypeError, ValueError):
        return fallback


def fixed_score(features: dict, contract: dict) -> float | None:
    """Predeclared simple ranking, not optimized from observed outcomes."""
    if not isinstance(features, dict) or not isinstance(contract, dict):
        return None
    rel = _num(features.get("rel_volume"))
    bid, ask = _num(contract.get("bid")), _num(contract.get("ask"))
    if rel is None or rel <= 0 or bid is None or ask is None or ask <= 0 or bid < 0 or bid > ask:
        return None
    fric = (ask - bid) / ask
    if fric > 0.30:  # predeclared transaction-friction research screen
        return None
    sentiment = max(-1., min(1., _num(features.get("sentiment_score"), 0.)))
    move = max(-.2, min(.2, _num(features.get("scan_day_ret"), 0.)))
    return round(math.log1p(min(rel, 10.0)) + 0.20 * sentiment - 1.5 * abs(move) - 2.0 * fric, 6)


def select(rows: list[dict], day: str, *, n: int = MAX_PICKS) -> dict:
    """One entry per ticker/day (even if multiple events); audit draws from remaining pool."""
    pool = []
    used = set()
    for row in sorted(rows, key=lambda r: str(r.get("signal_id") or "")):
        if row.get("date") != day or row.get("option_sample") != "research_census" or row.get("direction"):
            continue
        ticker = row.get("ticker")
        if not ticker or ticker in used:
            continue
        option = row.get("real_option") or {}
        sym = option.get("symbol")
        if not sym or not str(sym)[-9:-8] == "C":
            continue
        score = fixed_score(row.get("features") or {}, option)
        if score is None:
            continue
        expiry = option.get("expiry")
        try:
            if expiry and date.fromisoformat(str(expiry)[:10]) <= date.fromisoformat(day) + timedelta(days=12):
                continue
        except ValueError:
            continue
        used.add(ticker)
        pool.append({"day": day, "ticker": ticker, "signal_id": row.get("signal_id"),
                     "occ_symbol": sym, "score": score,
                     "selection_input": {"rel_volume": (row.get("features") or {}).get("rel_volume"),
                                         "sentiment_score": (row.get("features") or {}).get("sentiment_score"),
                                         "scan_day_ret": (row.get("features") or {}).get("scan_day_ret"),
                                         "bid": option.get("bid"), "ask": option.get("ask")},
                     "source_option_sample": "research_census"})
    max_n = max(0, min(int(n), MAX_PICKS))
    n_ranked = min(max_n, RANKED_PICKS)
    ranked = sorted(pool, key=lambda c: (-c["score"], c["ticker"]))[:n_ranked]
    remaining = [x for x in pool if x["ticker"] not in {v["ticker"] for v in ranked}]
    audits = sorted(remaining, key=lambda x: hashlib.sha256(
        f"{POLICY}:{day}:audit:{x['ticker']}".encode()).hexdigest())[:max_n - len(ranked)]
    selected = [{**c, "cohort": "fixed_rank"} for c in ranked] + [{**c, "cohort": "random_audit",
                  "audit_selection_probability": ((len(audits) / len(remaining)) if remaining else None)}
                  for c in audits]
    return {"policy": POLICY, "day": day, "pool_n": len(pool),
            "selected": selected, "n_ranked": len(ranked), "n_audit": len(audits),
            "executable_trades": 0, "production_influence": "NONE",
            "note": "Selection after signal day; future RTH ask/bid mandatory for paper evaluation."}


def _quote_timestamp(q: dict):
    try:
        v = q.get("retrieved_at")
        if not v:
            return None
        ts = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def settle(selection: dict, quotes: list[dict], as_of: date, *, horizon: int = 5) -> dict:
    """NO_LABEL unless next-session entry ask AND due exit bid are archived and valid."""
    entries = []
    for p in selection.get("selected") or []:
        sym, day = p["occ_symbol"], p["day"]
        created_at = _quote_timestamp({"retrieved_at": selection.get("created_at")})
        start = date.fromisoformat(day)
        arrivals = [q for q in quotes if q.get("occ_symbol") == sym and
                    oq.usable_rth_quote(q) and (d := oq.quote_day(q)) and
                    d > day and date.fromisoformat(d) <= start + timedelta(days=5)
                    and _quote_timestamp(q) is not None
                    and (created_at is None or _quote_timestamp(q) > created_at)]
        arrivals.sort(key=lambda q: _quote_timestamp(q))
        entry = arrivals[0] if arrivals else None
        record = {"ticker": p["ticker"], "day": day, "cohort": p["cohort"],
                  "occ_symbol": sym, "horizon_calendar_days": horizon,
                  "status": "NO_ENTRY_ARCHIVED" if entry is None else "OPEN",
                  "net_return": None}
        if entry is None:
            entries.append(record)
            continue
        entry_day = date.fromisoformat(oq.quote_day(entry))
        due = entry_day + timedelta(days=horizon)
        record.update(entry_record_id=entry.get("record_id"), entry_day=entry_day.isoformat())
        if as_of < due:
            record["status"] = "NOT_DUE"
            entries.append(record)
            continue
        outs = [q for q in quotes if q.get("occ_symbol") == sym and oq.usable_rth_exit_quote(q, "long")
                and (d := oq.quote_day(q)) and due.isoformat() <= d
                and date.fromisoformat(d) <= due + timedelta(days=7)
                and _quote_timestamp(q) is not None
                and _quote_timestamp(q) > _quote_timestamp(entry)]
        outs.sort(key=lambda q: _quote_timestamp(q))
        if not outs:
            record["status"] = "NO_EXIT_ARCHIVED"
            entries.append(record)
            continue
        exit_quote = outs[0]
        ask, bid = _num(entry.get("ask")), _num(exit_quote.get("bid"))
        if ask is None or ask <= 0 or bid is None or bid < 0:
            record["status"] = "UNUSABLE_QUOTE"
            entries.append(record)
            continue
        record.update(status="VERIFIED_PAPER", exit_record_id=exit_quote.get("record_id"),
                      exit_day=oq.quote_day(exit_quote),
                      net_return=round((bid - ask - COMMISSION_ROUNDTRIP_PER_CONTRACT / 100) / ask, 6))
        entries.append(record)
    verified = [e for e in entries if e["status"] == "VERIFIED_PAPER"]
    return {"policy": POLICY, "as_of": as_of.isoformat(), "selection_day": selection.get("day"),
            "horizon": horizon, "rows": entries, "n_selected": len(entries), "verified_n": len(verified),
            "mean_net_verified_only": (round(sum(x["net_return"] for x in verified) / len(verified), 6)
                                        if verified else None),
            "independent_days": len({x["day"] for x in verified}),
            "production_influence": "NONE", "verdict": "OBSERVATIONAL_UNVALIDATED"}


def propose(day: str, *, root: Path = PREDS) -> dict:
    if day != datetime.now(timezone.utc).date().isoformat():
        return {"status": "NOT_TODAY", "day": day, "production_influence": "NONE"}
    path = root / f"{day}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    rows = red.load(red.LEDGER, date.fromisoformat(day))
    result = select(rows, day)
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    root.mkdir(parents=True, exist_ok=True)
    # Append/create-only; no historic backfilling on known outcomes.
    with path.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    return result


def evaluate(as_of: date, *, root: Path = PREDS, quote_root: Path = red.QUOTES) -> dict:
    files = sorted(root.glob("????-??-??.json")) if root.exists() else []
    selections = [json.loads(f.read_text(encoding="utf-8")) for f in files
                  if f.stem <= as_of.isoformat()]
    if not selections:
        return {"as_of": as_of.isoformat(), "n_predictions": 0, "production_influence": "NONE"}
    quotes = oq.read(min(x["day"] for x in selections), as_of.isoformat(), root=quote_root)
    reports = [settle(s, quotes, as_of, horizon=h) for s in selections for h in (5, 20)]
    counts = {str(h): {"observed": sum(x["verified_n"] for x in reports if x["horizon"] == h),
                       "days": len({x["selection_day"] for x in reports if x["horizon"] == h
                                   and x["verified_n"]}),
                       "missing_entries": sum(sum(r["status"] == "NO_ENTRY_ARCHIVED" for r in x["rows"])
                                              for x in reports if x["horizon"] == h)}
              for h in (5, 20)}
    return {"as_of": as_of.isoformat(), "policy": POLICY, "n_predictions": sum(
                len(s.get("selected") or []) for s in selections), "coverage": counts,
            "reports": reports, "production_influence": "NONE", "verdict": "NEED_FORWARD_EVIDENCE"}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["propose", "evaluate"])
    p.add_argument("--date", default=datetime.now(timezone.utc).date().isoformat())
    args = p.parse_args(argv)
    if args.action == "propose":
        result = propose(args.date)
        print(json.dumps({k: result.get(k) for k in ("day", "pool_n", "n_ranked", "n_audit", "status")},
                         ensure_ascii=False))
    else:
        result = evaluate(date.fromisoformat(args.date))
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({k: result.get(k) for k in ("as_of", "n_predictions", "coverage", "verdict")},
                         ensure_ascii=False))


if __name__ == "__main__":
    main()

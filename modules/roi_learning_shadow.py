"""Research-only ROI learning: bounded rejected-contract shadow cohorts.

Use the options that OptionsDesigner already evaluated and rejected.
Never lower production thresholds, create an order, call an LLM, invent a
quote, or convert stock returns into option P&L. Entry and marks require
independent, archived RTH bid/ask quotes *after* registration.

Schedule: scanner registers at most 5 distinct rejects per NY day;
options-capture fetches the already-registered legs at entry and 5/20/45
calendar-day marks; feedback appends strictly quote-verifiable outcomes.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import yaml

from modules import option_quotes as oq

CONFIG = Path("config/roi_learning_shadow.yaml")
ROOT = Path("outputs/research/roi_learning")
COHORTS = ROOT / "cohorts"
MARKS = ROOT / "marks"
SUMMARY = ROOT / "summary.json"
MAX_DAY_HARD = 5
MAX_CAPTURE_HARD = 30
RESEARCH_THRESHOLDS = (0.0, 0.05, 0.07)
HORIZONS = (5, 20, 45)


class ResearchIntegrityError(ValueError):
    pass


def _canonical(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _hash(x):
    return hashlib.sha256(_canonical(x).encode("utf-8")).hexdigest()


def _config(root: Path = Path(".")) -> dict:
    p = Path(root) / CONFIG
    obj = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict) or obj.get("version") != "roi-learning-v1":
        raise ResearchIntegrityError("invalid ROI research configuration")
    if obj.get("production_influence") != "NONE" or obj.get("allow_llm") is not False:
        raise ResearchIntegrityError("production/LLM access forbidden")
    if obj.get("roi_research_thresholds") != list(RESEARCH_THRESHOLDS) or obj.get("horizons_calendar_days") != list(HORIZONS):
        raise ResearchIntegrityError("non-preregistered research thresholds or horizons")
    if not 1 <= int(obj.get("max_new_per_day", 0)) <= MAX_DAY_HARD:
        raise ResearchIntegrityError("daily research budget exceeds hard cap")
    if not 1 <= int(obj.get("max_quote_symbols_per_capture", 0)) <= MAX_CAPTURE_HARD:
        raise ResearchIntegrityError("capture exceeds research hard cap")
    if not 1 <= int(obj.get("max_entry_lag_days", 0)) <= 5:
        raise ResearchIntegrityError("entry freshness cannot be relaxed")
    if not 1 <= int(obj.get("max_exit_lag_days", 0)) <= 8:
        raise ResearchIntegrityError("mark freshness cannot be relaxed")
    if not 0 <= float(obj.get("commission_usd_per_leg_per_side", -1)) <= 3:
        raise ResearchIntegrityError("invalid research fee parameter")
    return obj


def _num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _journal_read(directory: Path, *, max_rows=10000) -> list[dict]:
    seen_ids, out = {}, []
    if not directory.exists():
        return []
    for path in sorted(directory.glob("????-??.jsonl")):
        with path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                if len(out) >= max_rows:
                    raise ResearchIntegrityError("research journal record budget exceeded")
                try:
                    event = json.loads(line)
                    rec = event["record"]
                    event_id = event["event_id"]
                    if not isinstance(rec, dict) or event_id != _hash(rec):
                        raise ValueError("content hash mismatch")
                    logical = rec["cohort_id"] if directory.name == "cohorts" else rec["cohort_id"] + ":" + str(rec["horizon_days"])
                    if logical in seen_ids:
                        if seen_ids[logical] != event_id:
                            raise ValueError("conflicting frozen record")
                        continue  # Git union merge can duplicate identical rows.
                    seen_ids[logical] = event_id
                    out.append(rec)
                except (ValueError, KeyError, TypeError) as exc:
                    raise ResearchIntegrityError(f"corrupt journal {path}:{lineno}: {exc}") from exc
    return out


def _journal_append(directory: Path, rows: list[dict], *, known: list[dict]) -> int:
    seen = {x["cohort_id"] if directory.name == "cohorts" else x["cohort_id"] + ":" + str(x["horizon_days"]) for x in known}
    to_write = []
    for r in rows:
        key = r["cohort_id"] if directory.name == "cohorts" else r["cohort_id"] + ":" + str(r["horizon_days"])
        if key in seen:
            continue
        seen.add(key)
        to_write.append(r)
    for r in to_write:
        d = r["signal_date"] if directory.name == "cohorts" else r["marked_at"][:10]
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / (d[:7] + ".jsonl")).open("a", encoding="utf-8") as fh:
            fh.write(_canonical({"event_id": _hash(r), "record": r}) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    return len(to_write)


def _symbols(row: dict) -> tuple[list[dict], str | None]:
    opt = row.get("option") or {}
    ticker = str(row.get("ticker") or "").strip().upper()
    right = "put" if "PUT" in str(row.get("strategy") or "").upper() else "call"
    is_spread = "SPREAD" in str(row.get("strategy") or "").upper()
    if not ticker or not isinstance(opt, dict) or not opt:
        return [], "no_option_contract"
    if is_spread and not isinstance(opt.get("spread_leg"), dict):
        return [], "spread_leg_missing"
    legs = [("long", opt)]
    if is_spread:
        legs.append(("short", opt["spread_leg"]))
    output = []
    for side, leg in legs:
        expiry = leg.get("expiry") or opt.get("expiry")
        sym = leg.get("option_symbol") or leg.get("symbol")
        if not sym:
            try:
                sym = oq.occ_symbol(ticker, expiry, right, float(leg.get("strike")))
            except (TypeError, ValueError):
                return [], "no_occ_identity"
        sym = str(sym).strip().upper().replace(" ", "")
        if not oq.option_root(sym):
            return [], "invalid_occ_identity"
        output.append({"side": side, "occ_symbol": sym})
    if len(set(x["occ_symbol"] for x in output)) != len(output):
        return [], "duplicate_legs"
    return output, None


def _rank(rejects: list[dict], existing: set[str], signal_day: str, max_new: int) -> list[dict]:
    # Stratification by actual failing gate; avoid selecting 5 near-identical
    # "barely failed" cases and misrepresenting the rejection population.
    groups = defaultdict(list)
    for row in rejects:
        ticker = str(row.get("ticker") or "").strip().upper()
        if not ticker or ticker in existing:
            continue
        legs, err = _symbols(row)
        if err:
            continue
        failed = sorted({v for v in (row.get("fail_gates") or {}).values() if v})
        group = failed[0] if failed else "unknown"
        key = _hash({"date": signal_day, "ticker": ticker, "strategy": row.get("strategy"),
                     "legs": legs})
        groups[group].append((key, row, legs))
    for g in groups.values():
        g.sort(key=lambda x: x[0])
    keys = sorted(groups)
    out, chosen_tickers = [], set()
    while len(out) < max_new and keys:
        for k in list(keys):
            if not groups[k]:
                keys.remove(k)
                continue
            _, row, legs = groups[k].pop(0)
            ticker = str(row.get("ticker") or "").strip().upper()
            if ticker in chosen_tickers:
                continue  # same ticker + multiple strategy candidates is 1 daily cohort
            chosen_tickers.add(ticker)
            out.append((row, legs))
            if len(out) >= max_new:
                break
    return out


def register(rejects: list[dict], signal_day: str, *,
             now: datetime | None = None, root: Path = Path(".")) -> dict:
    """Add research shadow contracts, never modify OptionsDesigner or trades."""
    root = Path(root)
    cfg = _config(root)
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ResearchIntegrityError("decision timestamp must be timezone-aware")
    if date.fromisoformat(signal_day) != now.date():
        # Scanner date usually NY date; allow NY day across UTC midnight only
        from modules import market_snapshot as ms
        if date.fromisoformat(signal_day) != now.astimezone(ms.NY_TZ).date():
            raise ResearchIntegrityError("cannot backdate research registration")
    old = _journal_read(root / COHORTS)
    day_old = [r for r in old if r["signal_date"] == signal_day]
    slots = max(0, int(cfg["max_new_per_day"]) - len(day_old))
    result = {"mode": "research_shadow_only", "input_rejects": len(rejects),
              "already_registered_today": len(day_old), "registered": 0,
              "production_impact": False, "llm_calls": 0}
    if not cfg.get("enabled") or slots <= 0:
        return result
    seen_tickers = {r["ticker"] for r in day_old}
    chosen = _rank(rejects, seen_tickers, signal_day, slots)
    rows = []
    for source, legs in chosen:
        ticker = str(source["ticker"]).strip().upper()
        roi = _num(source.get("roi_net"))
        hurdle = _num(source.get("roi_hurdle"))
        failures = dict(source.get("fail_gates") or {})
        cohort_id = _hash({"date": signal_day, "ticker": ticker, "strategy": source.get("strategy"),
                           "legs": legs})[:24]
        rows.append({
            "schema": "roi-learning-cohort-v1",
            "cohort_id": cohort_id, "signal_date": signal_day,
            "decision_at": now.astimezone(timezone.utc).isoformat(timespec="seconds"),
            "ticker": ticker, "strategy": source.get("strategy"),
            "legs": legs,
            "original_roi_net": roi, "original_production_hurdle": hurdle,
            "roi_only_ge": {f"{x:.2f}": roi >= x if roi is not None else None for x in RESEARCH_THRESHOLDS},
            "other_production_gates": failures, "selection": "deterministic_stratified_gate_failures",
            "evidence": "REJECTED_ONLY_NOT_TRADABLE",
            "production_eligible": False,
            "commission_usd_per_leg_per_side": float(cfg["commission_usd_per_leg_per_side"]),
        })
    result["registered"] = _journal_append(root / COHORTS, rows, known=old)
    result["cohort_ids"] = [r["cohort_id"] for r in rows]
    return result


def _time(value):
    return oq._parse_ts(value)


def _quote_pairs(legs: list[dict], quotes_by_symbol: dict[str, list[dict]],
                 *, after: datetime, before: datetime):
    """Earliest actual executable bid/ask pair at/beyond decision/mark time."""
    left = []
    for leg in legs:
        vals = [q for q in quotes_by_symbol.get(leg["occ_symbol"], [])
                if oq.usable_rth_quote(q) and (ts := _time(q.get("retrieved_at")))
                and after <= ts <= before]
        vals.sort(key=lambda q: str(q.get("retrieved_at")))
        if not vals:
            return None
        left.append(vals)
    if len(left) == 1:
        q = left[0][0]
        return [q], _time(q["retrieved_at"])
    for a in left[0]:
        ta = _time(a["retrieved_at"])
        for b in left[1]:
            tb = _time(b["retrieved_at"])
            if abs((tb-ta).total_seconds()) <= 900 and oq.quote_day(a) == oq.quote_day(b):
                return [a, b], max(ta,tb)
    return None


def _debit(entry: list[dict]) -> float | None:
    value = _num(entry[0].get("ask"))
    if len(entry) > 1 and value is not None:
        value -= _num(entry[1].get("bid")) or 0
    return value if value is not None and value > 0 else None


def _credit(exit: list[dict]) -> float | None:
    value = _num(exit[0].get("bid"))
    if len(exit) > 1 and value is not None:
        ask = _num(exit[1].get("ask"))
        value = value - ask if ask is not None else None
    return value


def _quotes_since(root: Path, cohorts: list[dict], *, today: date) -> dict[str, list[dict]]:
    if not cohorts:
        return {}
    start = min(r["signal_date"] for r in cohorts)
    rows = oq.read(start, today.isoformat(), root=root / "outputs/options_archive/quotes")
    want = {leg["occ_symbol"] for r in cohorts for leg in r["legs"]}
    result = defaultdict(list)
    for q in rows:
        sym = q.get("occ_symbol")
        if sym in want and (ts := _time(q.get("retrieved_at"))) and ts.date() <= today:
            result[sym].append(q)
    for q in result.values():
        q.sort(key=lambda x: str(x.get("retrieved_at")))
    return result


def due_capture_symbols(*, today: date | None = None, root: Path = Path(".")) -> list[str]:
    """For options-capture only: bounded extra symbols, no expensive LLM/market scan."""
    root = Path(root)
    cfg = _config(root)
    if not cfg.get("enabled"):
        return []
    today = today or datetime.now(timezone.utc).date()
    cohorts = _journal_read(root / COHORTS)
    marks = _journal_read(root / MARKS)
    done = {(x["cohort_id"], x["horizon_days"]) for x in marks}
    wanted = []
    for c in sorted(cohorts, key=lambda x: (x["signal_date"], x["cohort_id"])):
        d = date.fromisoformat(c["signal_date"])
        elapsed = (today-d).days
        horizons = (0, *HORIZONS)
        for h in horizons:
            max_lag = int(cfg["max_entry_lag_days"] if h == 0 else cfg["max_exit_lag_days"])
            if (h == 0 or (c["cohort_id"], h) not in done) and h <= elapsed <= h + max_lag:
                wanted.extend(x["occ_symbol"] for x in c["legs"])
                break
    return list(dict.fromkeys(wanted))[:int(cfg["max_quote_symbols_per_capture"])]


def evaluate(*, today: date | None = None, root: Path = Path(".")) -> dict:
    """Mature observed marks, no forward return or training input is fabricated."""
    root = Path(root)
    cfg = _config(root)
    today = today or datetime.now(timezone.utc).date()
    cohorts = _journal_read(root / COHORTS)
    old = _journal_read(root / MARKS)
    done = {(x["cohort_id"], x["horizon_days"]) for x in old}
    active = [c for c in cohorts if 0 <= (today - date.fromisoformat(c["signal_date"])).days <= max(HORIZONS) + int(cfg["max_exit_lag_days"])]
    by_symbol = _quotes_since(root, active, today=today)
    new, unknown = [], Counter()
    for c in active:
        decision = _time(c["decision_at"])
        if decision is None or decision.date() > today:
            raise ResearchIntegrityError("frozen decision timestamp invalid")
        entry = _quote_pairs(c["legs"], by_symbol,
                             after=decision,
                             before=min(datetime.combine(today, datetime.max.time(), tzinfo=timezone.utc),
                                        decision + timedelta(days=int(cfg["max_entry_lag_days"]))))
        if not entry:
            unknown["entry_quote_missing"] += 1
            continue
        entry_legs, entry_at = entry
        debit = _debit(entry_legs)
        if debit is None:
            unknown["entry_not_positive_debit"] += 1
            continue
        for horizon in HORIZONS:
            if (c["cohort_id"], horizon) in done or (today - entry_at.date()).days < horizon:
                continue
            mark_start = datetime.combine(entry_at.date() + timedelta(days=horizon),
                                          datetime.min.time(), tzinfo=timezone.utc)
            end = min(datetime.combine(today, datetime.max.time(), tzinfo=timezone.utc),
                      mark_start + timedelta(days=int(cfg["max_exit_lag_days"])))
            mark = _quote_pairs(c["legs"], by_symbol, after=mark_start, before=end)
            if not mark:
                unknown[f"mark_{horizon}_quote_missing"] += 1
                continue
            exit_legs, mark_at = mark
            credit = _credit(exit_legs)
            if credit is None:
                unknown[f"mark_{horizon}_nonexecutable"] += 1
                continue
            fees = (2 * len(c["legs"]) * float(c["commission_usd_per_leg_per_side"])) / 100.0
            net_return = (credit - debit - fees) / debit
            new.append({
                "schema": "roi-learning-mark-v1",
                "cohort_id": c["cohort_id"], "horizon_days": horizon,
                "marked_at": mark_at.isoformat(timespec="seconds"),
                "entry_at": entry_at.isoformat(timespec="seconds"),
                "entry_quote_ids": [q["record_id"] for q in entry_legs],
                "mark_quote_ids": [q["record_id"] for q in exit_legs],
                "entry_debit": round(debit, 6),
                "mark_liquidation_credit": round(credit, 6),
                "round_trip_commissions_per_share": round(fees, 6),
                "net_return": round(net_return, 6),
                "price_evidence": "OBSERVED_RTH_EXECUTABLE_BID_ASK",
                "strategy_alpha_attributable": False,
                "production_eligible": False,
            })
    added = _journal_append(root / MARKS, new, known=old)
    all_marks = [*old, *new]
    total_by_horizon = Counter(m["horizon_days"] for m in all_marks)
    # Research thresholds are labels on the ORIGINAL rejected model's
    # expected ROI, not permission to trade, and are not backtest successes.
    buckets = {f"{t:.2f}": {"eligible_counterfactual_cohorts": 0, "realized_mark_45": 0}
               for t in RESEARCH_THRESHOLDS}
    ids45 = {m["cohort_id"] for m in all_marks if m["horizon_days"] == 45}
    for c in cohorts:
        roi = c.get("original_roi_net")
        for t in RESEARCH_THRESHOLDS:
            if roi is not None and roi >= t:
                bucket = buckets[f"{t:.2f}"]
                bucket["eligible_counterfactual_cohorts"] += 1
                bucket["realized_mark_45"] += int(c["cohort_id"] in ids45)
    report = {
        "schema": "roi-learning-summary-v1", "as_of": today.isoformat(),
        "n_rejected_candidates_registered": len(cohorts),
        "n_new_marks": added, "n_all_marks": len(all_marks),
        "marks_by_horizon": dict(sorted(total_by_horizon.items())),
        "missing_quotes_by_reason": dict(sorted(unknown.items())),
        "research_roi_thresholds": buckets,
        "actual_production_roi_gate_changed": False,
        "research_only": True, "orders_created": 0, "llm_calls": 0,
        "verified_trades_do_not_imply_factor_alpha": True,
    }
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    temp = root / (str(SUMMARY) + ".tmp")
    output = root / SUMMARY
    output.parent.mkdir(parents=True, exist_ok=True)
    temp.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, output)
    return report


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["evaluate", "status", "due"])
    opts = parser.parse_args(argv)
    if opts.command == "evaluate":
        obj = evaluate()
    elif opts.command == "due":
        obj = {"symbols": due_capture_symbols()}
    else:
        obj = json.loads(SUMMARY.read_text(encoding="utf-8")) if SUMMARY.exists() else {"status": "NOT_STARTED"}
    print(json.dumps(obj, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

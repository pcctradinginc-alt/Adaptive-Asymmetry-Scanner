"""Real-options forward audit: ONLY archived fills and existing observations.

Research-only, no market calls, no LLM, no order placement. A closed trade is
not evidence for a generated factor unless its frozen hypothesis ID and
contemporaneous signal were registered BEFORE entry. Current trade-outcome
schema lacks that binding, so outputs remain explicitly UNATTRIBUTED.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import yaml

from .registry import EvidenceIntegrityError, _atomic_json, digest

POLICY = Path("config/autonomous_alpha_phase_c.yaml")
SOURCE = Path("outputs/options_archive/trade_outcomes.jsonl")
QUOTE_ROOT = Path("outputs/options_archive/quotes")
REPORT = Path("outputs/research/autonomous_alpha/options_forward.json")


def policy(root: Path) -> dict:
    p = root / POLICY
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or cfg.get("version") != "phase-c-v1":
        raise EvidenceIntegrityError("invalid phase C policy")
    if cfg.get("production_influence") != "NONE":
        raise EvidenceIntegrityError("Phase C must have zero production influence")
    start = date.fromisoformat(str(cfg["forward_start"]))
    if start < date(2026, 10, 29):
        raise EvidenceIntegrityError("prospective registration date moved backward")
    if not 26 <= int(cfg.get("minimum_verified_trades", 0)) <= 1000:
        raise EvidenceIntegrityError("unsafe minimum sample count")
    if not 20 <= int(cfg.get("minimum_distinct_entry_dates", 0)) <= 365:
        raise EvidenceIntegrityError("unsafe independent time coverage")
    if not 1 <= int(cfg.get("max_trade_records", 0)) <= 100000:
        raise EvidenceIntegrityError("unsafe record budget")
    if not 1 <= int(cfg.get("max_quote_records_per_file", 0)) <= 50000:
        raise EvidenceIntegrityError("unsafe quote file budget")
    return cfg


def _read_lines(path: Path, *, maximum: int) -> list[dict]:
    """Fail closed on corrupt data: never treat missing or truncated data as zero alpha."""
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            if not line.strip():
                continue
            if len(records) >= maximum:
                raise EvidenceIntegrityError(f"archive record budget exceeded: {path}")
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise TypeError("not a JSON object")
            except (ValueError, TypeError) as exc:
                raise EvidenceIntegrityError(f"malformed {path}:{i}: {exc}") from exc
            records.append(obj)
    return records


def _n(x):
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except (TypeError, ValueError):
        return None


def _date(x) -> date | None:
    try:
        return date.fromisoformat(str(x))
    except (TypeError, ValueError):
        return None


def _quote_ids_for_days(root: Path, days: set[str], cfg: dict) -> tuple[set[str], list[str]]:
    """Read ONLY archived quotes for actual trade entry/exit days.

    Do not fetch future market data or manufacture contemporaneous quotes.
    """
    ids, missing = set(), []
    for d in sorted(days):
        try:
            date.fromisoformat(d)
        except ValueError:
            raise EvidenceIntegrityError(f"invalid quote day: {d}")
        path = root / QUOTE_ROOT / f"{d}.jsonl"
        if not path.exists():
            missing.append(d)
            continue
        for row in _read_lines(path, maximum=int(cfg["max_quote_records_per_file"])):
            if row.get("price_type") != "OBSERVED_QUOTE":
                continue
            rid = row.get("record_id")
            if rid:
                ids.add(str(rid))
    return ids, missing


def _inspect(row: dict, *, start: date, today: date, ids: set[str]) -> dict:
    tid = row.get("trade_id")
    ed, xd = _date(row.get("entry_date")), _date(row.get("close_date"))
    reasons = []
    net = _n(row.get("outcome_net"))
    gross = _n(row.get("outcome"))
    ev = row.get("evaluation") or {}
    if not isinstance(ev, dict):
        reasons.append("evaluation_invalid")
        ev = {}
    if row.get("schema") != "trade-outcome-v1" or not tid:
        reasons.append("invalid_trade_schema")
    if ed is None or xd is None or xd < ed or xd > today:
        reasons.append("invalid_or_unmatured_dates")
    if ed is None or ed < start:
        reasons.append("pre_registration")
    if not isinstance(net, float):
        reasons.append("net_return_missing")
    if _n(row.get("entry_debit")) is None or _n(row.get("entry_debit")) <= 0:
        reasons.append("invalid_entry_debit")
    if row.get("price_type") != "OBSERVED_QUOTE":
        reasons.append("non_observed_price")
    if row.get("reliable") is not True:
        reasons.append("source_not_reliable")
    if ev.get("tier") != "REALIZED_RTH_REPRODUCIBLE" or ev.get("reproducible") is not True or ev.get("rth_executable") is not True:
        reasons.append("not_rth_reproducible")
    if ev.get("status") != "OK" or ev.get("flags"):
        reasons.append("quote_quality_flags")
    entry_ids, exit_ids = row.get("entry_quote_record_ids"), row.get("exit_quote_record_ids")
    if not (isinstance(entry_ids, list) and entry_ids and isinstance(exit_ids, list) and exit_ids):
        reasons.append("entry_or_exit_quote_missing")
    else:
        missing_ids = [x for x in entry_ids+exit_ids if x not in ids]
        if missing_ids:
            reasons.append("quote_ids_not_in_archive")
    # No evidence of ex-ante alpha assignment in trade-outcome-v1.
    return {
        "trade_id": tid, "ticker": row.get("ticker"),
        "entry_date": ed.isoformat() if ed else None,
        "close_date": xd.isoformat() if xd else None,
        "strategy": row.get("strategy"), "net_return": net,
        "gross_return": gross,
        "quote_verifiable": not reasons, "exclude_reasons": sorted(set(reasons)),
        "alpha_id": None, "alpha_attribution": "UNATTRIBUTED",
        "forward_evidence": "POST_REGISTRATION_REALIZED" if not reasons else "INELIGIBLE",
    }


def evaluate(root: Path = Path("."), *, today: date | None = None,
             persist: bool = True) -> dict:
    root = Path(root)
    cfg = policy(root)
    today = today or datetime.now(timezone.utc).date()
    start = date.fromisoformat(str(cfg["forward_start"]))
    rows = _read_lines(root / SOURCE, maximum=int(cfg["max_trade_records"]))
    # Always read all source rows, including historical/unverified examples.
    # No silent overwriting of differing versions of the same trade.
    unique = {}
    for row in rows:
        key = row.get("trade_id")
        if not key:
            raise EvidenceIntegrityError("trade missing identifier")
        if key in unique and digest(row) != digest(unique[key]):
            raise EvidenceIntegrityError(f"conflicting closed outcome: {key}")
        unique[key] = row
    dates = {str(row.get(f)) for row in unique.values()
             for f in ("entry_date", "close_date") if _date(row.get(f))}
    ids, missing_quote_days = _quote_ids_for_days(root, dates, cfg)
    inspected = [_inspect(row, start=start, today=today, ids=ids)
                 for _, row in sorted(unique.items())]
    verified = [x for x in inspected if x["quote_verifiable"]]
    unverified_counts = Counter(reason for x in inspected for reason in x["exclude_reasons"])
    by_strategy = {}
    for strategy in sorted({str(x["strategy"]) for x in verified}):
        arr = [x["net_return"] for x in verified if str(x["strategy"]) == strategy]
        by_strategy[strategy] = {"n": len(arr), "net_mean": round(sum(arr)/len(arr), 6) if arr else None}
    distinct_days = len({x["entry_date"] for x in verified})
    enough = len(verified) >= int(cfg["minimum_verified_trades"]) and distinct_days >= int(cfg["minimum_distinct_entry_dates"])
    # Cohort may still be affected by nonrandom selection, survivorship, option
    # quality filtering, and missing OCC/venue costs. NEVER 'validated alpha'.
    result = {
        "schema": "alpha-options-forward-v1",
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "forward_start": start.isoformat(), "as_of": today.isoformat(),
        "source": str(SOURCE), "source_state": "PRESENT" if (root / SOURCE).exists() else "MISSING",
        "n_closed_rows": len(rows), "n_distinct_trades": len(unique),
        "n_verified_post_registration": len(verified),
        "n_distinct_verified_entry_dates": distinct_days,
        "n_unverified_or_pre_registration": len(inspected)-len(verified),
        "excluded_by_reason": dict(sorted(unverified_counts.items())),
        "missing_quote_days": sorted(missing_quote_days),
        "net_mean_verified": round(sum(x["net_return"] for x in verified)/len(verified),6) if verified else None,
        "net_median_verified": sorted(x["net_return"] for x in verified)[len(verified)//2] if verified else None,
        "by_strategy": by_strategy,
        "sample_floor_reached": enough,
        "strategy_alpha_status": "NOT_ATTRIBUTABLE",
        "research_eligible": bool(enough),
        "prospective_alpha_validated": False,
        "reason": "Real outcomes have no frozen ex-ante alpha ID/signal link; cost model omits some fees.",
        "production_influence": "NONE",
    }
    if persist:
        _atomic_json(root / REPORT, result)
    return result

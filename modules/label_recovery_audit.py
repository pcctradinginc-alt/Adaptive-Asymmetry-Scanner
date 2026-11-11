"""Forensic audit of missing historical option entry evidence. NO price invention.

This process is offline/read-only over ledger and observed quote archive.
It does not reconstruct or overwrite an original trade from market models.
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

from modules import option_quotes as oq
from modules import research_evidence_discovery as red

OUT = Path("outputs/research/label_recovery_audit")
ROOT = Path("outputs/candidate_ledger")
ARCHIVE = Path("outputs/options_archive/quotes")


def diagnose(row: dict, index: dict, horizon: int = 5) -> dict:
    rs = row.get("real_strategy") or {}
    legs = rs.get("legs") or []
    day = str(rs.get("entry_date") or "")
    symbols = [str(leg.get("symbol") or "") for leg in legs]
    entry_ids = rs.get("entry_quote_record_ids") or []
    reason = red.provenance(row, index, horizon)
    result = {"date": row.get("date"), "ticker": row.get("ticker"),
              "signal_id": row.get("signal_id"), "entry_day": day,
              "symbols": symbols, "reason": reason,
              "entry_record_ids": entry_ids, "recovered": reason == "VERIFIED"}
    if reason != "ENTRY_NOT_ARCHIVED":
        return result
    by_id = index.get("by_id", {})
    by_symbol = index.get("by_symbol", {})
    missing_ids = [rid for rid in entry_ids if rid not in by_id]
    matching = []
    for leg in legs:
        sym = leg.get("symbol")
        valid = [q for q in by_symbol.get(sym, []) if oq.quote_day(q) == day]
        matching.append(len(valid))
    result["missing_record_ids"] = missing_ids
    result["same_day_usable_quotes_per_leg"] = matching
    result["possible_archive_research"] = bool(any(matching))
    return result


def audit(rows: list[dict], quotes: list[dict], *, horizon: int = 5) -> dict:
    index = red.quote_index(quotes)
    observed = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        oc = row.get("outcomes") or {}
        if red.number(oc.get(f"real_strat_ret_{horizon}d")) is None:
            continue
        if not row.get("real_strategy"):
            continue
        diag = diagnose(row, index, horizon)
        observed.append(diag)
    counts = Counter(item["reason"] for item in observed)
    missing = [d for d in observed if d["reason"] == "ENTRY_NOT_ARCHIVED"]
    return {
        "schema": "historical_entry_audit_v1", "horizon": horizon,
        "assessed_option_outcomes": len(observed), "counts": dict(sorted(counts.items())),
        "historical_missing_entry_count": len(missing), "missing": missing,
        "restored_labels": 0,  # nothing is changed by an audit
        "production_influence": "NONE", "no_synthetic_prices": True,
        "note": ("Possible archive evidence requires manual independent provenance review. "
                 "An absent contemporaneous quote cannot be reconstructed from Black-Scholes, "
                 "current quotes, stock closes, or unverified vendor history.")}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--as-of", default=date.today().isoformat())
    p.add_argument("--horizon", type=int, choices=[5, 20], default=5)
    p.add_argument("--ledger", type=Path, default=ROOT)
    p.add_argument("--quotes", type=Path, default=ARCHIVE)
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args(argv)
    day = date.fromisoformat(a.as_of)
    rows = red.load(a.ledger, day)
    relevant = [r for r in rows if (r.get("outcomes") or {}).get(f"real_strat_ret_{a.horizon}d") is not None]
    quotes = oq.read(min((str(r["date"])[:10] for r in relevant), default=day.isoformat()),
                     day.isoformat(), root=a.quotes)
    report = audit(relevant, quotes, horizon=a.horizon)
    report["as_of"] = day.isoformat()
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    a.out.mkdir(parents=True, exist_ok=True)
    dest = a.out / f"{day.isoformat()}_{a.horizon}d.json"
    dest.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "missing"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

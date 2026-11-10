"""Forensic, read-only audit of missing executable option labels.

No historical quote reconstruction, no interpolated asks/bids, no ledger edits.
Reports exactly which archived OCC/RTH facts are missing. A missing quote is
NEVER a zero return or a VERIFIED training observation.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

from modules import option_quotes as oq
from modules import option_learner as ol
from modules import research_evidence_discovery as red

OUT = Path("outputs/research/option_learner/entry_provenance_audit.json")


def audit(rows: list[dict], quotes: list[dict], as_of: date, horizons=(5, 20)) -> dict:
    index = red.quote_index(quotes)
    first_archive = min((oq.quote_day(q) for q in quotes if oq.quote_day(q)), default=None)
    missing, due, verified, details = Counter(), Counter(), Counter(), []
    for c in ol.candidates(rows, quotes, as_of, horizon=5):
        # A different horizon must recheck independently; options expressions
        # and timestamps are derived only from original immutable ledger rows.
        pass
    for h in horizons:
        cs = ol.candidates(rows, quotes, as_of, horizon=h)
        for c in cs:
            for e in c["exprs"]:
                status = e["status"]
                if status == "NO_LABEL":
                    continue
                due[h] += 1
                if status == "VERIFIED":
                    verified[h] += 1
                    continue
                missing[(h, status)] += 1
                if status != "ENTRY_NOT_ARCHIVED":
                    continue
                row = e["row"]
                rs = row.get("real_strategy") or {}
                entry_day = str(rs.get("entry_date") or "")
                legs = rs.get("legs") or []
                ids = rs.get("entry_quote_record_ids") or []
                unknown_ids = [i for i in ids if i not in index["by_id"]]
                symbols = [leg.get("symbol") for leg in legs]
                available_same_day = {
                    symbol: sum(1 for q in index["by_symbol"].get(symbol) or []
                                if oq.quote_day(q) == entry_day)
                    for symbol in symbols if symbol
                }
                reason = ("before_archive" if first_archive and entry_day < first_archive
                          else "missing_exact_record_id" if unknown_ids
                          else "no_matching_archived_entry_quote")
                details.append({"horizon": h, "date": c["date"], "ticker": c["ticker"],
                                "signal_id": c["signal_id"], "expression": e["expression"],
                                "entry_day": entry_day, "reason": reason,
                                "occ_symbols": symbols,
                                "missing_record_id_count": len(unknown_ids),
                                "same_day_usable_quotes": available_same_day,
                                "recoverable_from_supplied_archive": False})
    return {
        "as_of": as_of.isoformat(), "audit": "archived_quotes_only",
        "first_archive_day": first_archive,
        "due_by_horizon": {f"{h}d": due[h] for h in horizons},
        "verified_by_horizon": {f"{h}d": verified[h] for h in horizons},
        "missing_by_horizon_reason": {f"{h}d": {reason: n for (hh, reason), n in missing.items() if hh == h}
                                      for h in horizons},
        "entry_not_archived_rows": len(details),
        "entry_not_archived_unique_signals": len({(x["date"], x["signal_id"]) for x in details}),
        "entry_missing_details": details,
        "recovered_labels": 0,  # report ONLY. Never generate synthetic labels.
        "production_influence": "NONE",
        "next_action": "Obtain original timestamped RTH OCC records for missing entry dates; "
                       "if unavailable, keep labels invalid. Do not amend immutable ledger.",
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--as-of", default=date.today().isoformat())
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args(argv)
    as_of = date.fromisoformat(args.as_of)
    rows = red.load(ol.LEDGER, as_of)
    labelled = [r for r in rows if any(
        red.number((r.get("outcomes") or {}).get(f"{prefix}_ret_{h}d")) is not None
        for h in (5, 20) for _k, prefix, _name in ol.SOURCES)]
    quotes = red.load_quotes(ol.QUOTES, labelled, as_of)
    report = audit(rows, quotes, as_of)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("entry_missing_details",)},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Fail-closed lifecycle integrity audit for candidate-ledger research.

This verifies structural integrity only, not independent quote provenance or
whether a production training job consumed the labeled records.
"""
from __future__ import annotations

from collections import Counter
from datetime import date, timedelta

from modules.research_evidence_discovery import number


def inspect(rows, as_of):
    today = date.fromisoformat(as_of) if isinstance(as_of, str) else as_of
    seen = {}
    issues = Counter()
    states = Counter()
    for row in rows:
        if not isinstance(row, dict):
            issues["invalid_row"] += 1
            continue
        ticker = str(row.get("ticker") or "").strip()
        day = row.get("date")
        if not ticker or not day:
            issues["missing_identity"] += 1
            continue
        try:
            signal_day = date.fromisoformat(str(day))
        except ValueError:
            issues["invalid_signal_date"] += 1
            continue
        key = (str(day), ticker, str(row.get("signal_id") or row.get("event_id") or ""))
        if key in seen:
            issues["duplicate_signal"] += 1
        seen[key] = row
        if not row.get("signal_timestamp"):
            issues["missing_timestamp"] += 1
        if not isinstance(row.get("features"), dict):
            issues["missing_features"] += 1
        if not row.get("reject_reason") and row.get("status") == "rejected":
            issues["rejection_without_reason"] += 1
        outcomes = row.get("outcomes")
        outcomes = outcomes if isinstance(outcomes, dict) else {}
        for h in (5, 20, 45):
            value = number(outcomes.get(f"real_strat_ret_{h}d"))
            if value is None:
                if signal_day <= today - timedelta(days=h + 1):
                    states[f"mature_missing_{h}d"] += 1
                else:
                    states[f"pending_{h}d"] += 1
            elif signal_day > today - timedelta(days=h + 1):
                issues[f"premature_mark_{h}d"] += 1
            else:
                states[f"observed_{h}d"] += 1
    blockers = sum(issues.values())
    return {"as_of": today.isoformat(), "unique_signals": len(seen),
            "issues": dict(sorted(issues.items())),
            "lifecycle": dict(sorted(states.items())),
            "structural_integrity": "FAIL" if blockers else "PASS",
            "quote_provenance": "NOT_VERIFIED",
            "training_consumption": "NOT_VERIFIED",
            "production_learning_closed_loop": "NOT_VERIFIED",
            "production_influence": "NONE"}

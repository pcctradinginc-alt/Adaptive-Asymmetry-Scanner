"""PIT data-quality evidence for the existing autonomous alpha/option learning loop.

Reads only previously committed repository research outputs; never asks an
API, invents quotes, labels, extra trades or independent observations.
This is observability, not a second research engine or training population.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from .registry import _atomic_json

OUT = Path("outputs/research/autonomous_alpha/data_quality.json")
HISTORY = Path("outputs/research/autonomous_alpha/data_quality_history")
LINK = Path("outputs/research/autonomous_alpha/option_link.json")
ECON = Path("outputs/research/autonomous_alpha/option_economics.json")
VALIDATION = Path("outputs/research/autonomous_alpha/option_validation.json")
ERRORS = Path("outputs/research/error_research.json")
LEARNER = Path("outputs/research/option_learner")
EVIDENCE = Path("outputs/research/evidence_discovery")
MIN_QUOTE_COVERAGE = 0.80  # Existing alpha-to-options validation lower bound, never relaxed.


def _js(root: Path, path: Path) -> dict:
    p = Path(root) / path
    if not p.exists():
        return {}
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("data quality source must be an object: " + str(path))
    return obj


def _oldest_day(root: Path, gap_id: str, now_day: str) -> str:
    """Earliest *observed in future monitoring*, not guessed historical onset."""
    found = [now_day]
    for p in sorted((Path(root) / HISTORY).glob("*.json")):
        obj = json.loads(p.read_text(encoding="utf-8"))
        if gap_id in (obj.get("active_gap_ids") or []):
            found.append(p.stem)
    return min(found)


def _horizon(root: Path, n: int) -> dict:
    s = _js(root, LEARNER / f"status_{n}d.json")
    counts = s.get("provenance_counts") or {}
    verified = int(counts.get("VERIFIED") or 0)
    entry = int(counts.get("ENTRY_NOT_ARCHIVED") or 0)
    exit_ = int(counts.get("EXIT_NOT_ARCHIVED") or 0)
    observed_denominator = verified + entry + exit_
    feature = s.get("feature_coverage") or {}
    missing = {k: round(float(v), 5) for k, v in feature.items()
               if isinstance(v, (int, float)) and not isinstance(v, bool)
               and float(v) < MIN_QUOTE_COVERAGE}
    label_dates = s.get("label_dates") or []
    req = s.get("evidence_requirements") or {}
    return {
        "status": "NOT_REPORTED" if not s else "HISTORICAL_PROVENANCE_ONLY",
        "as_of": s.get("as_of"), "horizon_trading_days": n,
        "verified_quote_labels": verified,
        "entry_not_archived": entry, "exit_not_archived": exit_,
        "archive_sample_total": observed_denominator,
        "archive_sample_verified_fraction": round(verified / observed_denominator, 5)
        if observed_denominator else None,
        "fraction_scope": "HISTORICAL_ARCHIVE_SAMPLE_NOT_INDEPENDENT_FORWARD_COVERAGE",
        "verified_signal_dates": len(label_dates) if isinstance(label_dates, list) else None,
        "independent_periods": None,  # count only from a separately validated forward test
        "min_independent_periods": req.get("min_independent_periods"),
        "feature_coverage_below_80pct": missing,
        "candidate_ledger_rows_7d": s.get("ledger_rows_7d"),
        "option_snapshots_7d": s.get("option_snapshots_7d"),
        "model_status": (s.get("model") or {}).get("status"),
    }


def build(root: Path = Path("."), *, now: datetime | None = None) -> dict:
    root = Path(root)
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("data quality timestamp must be timezone aware")
    day = now.date().isoformat()
    link, val, econ = (_js(root, p) for p in (LINK, VALIDATION, ECON))
    errors = _js(root, ERRORS)
    horizons = {str(h): _horizon(root, h) for h in (5, 20)}
    reports = sorted((root / EVIDENCE).glob("*.json"))
    evidence = _js(root, EVIDENCE / reports[-1].name) if reports else {}
    provenance = {str(k): int(v) for k, v in (evidence.get("provenance_counts") or {}).items()}
    gaps = []
    if provenance.get("ENTRY_NOT_ARCHIVED"):
        gaps.append({"id": "ENTRY_NOT_ARCHIVED", "count": provenance["ENTRY_NOT_ARCHIVED"],
                     "cause": "No authentic archived quote at entry (historic rows may predate capture)",
                     "action": "Increase future RTH entry capture; never backfill historical execution quotes",
                     "recoverable_historically": False})
    if provenance.get("EXIT_NOT_ARCHIVED"):
        gaps.append({"id": "EXIT_NOT_ARCHIVED", "count": provenance["EXIT_NOT_ARCHIVED"],
                     "cause": "No verified exit quote at required valuation time",
                     "action": "Monitor and capture due exit legs at RTH; do not fabricate exit prices",
                     "recoverable_historically": False})
    for n, h in horizons.items():
        for feat, fraction in h["feature_coverage_below_80pct"].items():
            gaps.append({"id": f"FEATURE:{n}d:{feat}", "count": None,
                         "coverage_fraction": fraction,
                         "cause": "Missing or invalid point-in-time feature in observed label sample",
                         "action": "Inspect existing feature producer and timestamps; never fill gaps as zero",
                         "recoverable_historically": False})
    pending = (val.get("learning") or {}).get("structure_labels") or {}
    blocked = {structure: group.get("NEED_MORE_DATA", 0)
               for structure, group in pending.items() if isinstance(group, dict)}
    if not val.get("plans"):
        gaps.append({"id": "NO_PREREGISTERED_ALPHA_OPTION_FORWARD_PLANS", "count": 0,
                     "cause": "Economically and statistically qualified new alpha-option candidates not available",
                     "action": "Collect post-registration quote-verified exits and independent forward windows",
                     "recoverable_historically": False})
    egaps = errors.get("data_gaps") or []
    for e in egaps:
        if e.get("kind") == "independent_periods":
            gaps.append({"id": f"INDEPENDENT_PERIODS:{e.get('horizon')}d", "count": None,
                         "cause": e.get("finding"), "action": "Await new nonoverlapping prospective windows",
                         "recoverable_historically": False})
    gaps = [{**g, "first_observed_in_monitoring": _oldest_day(root, g["id"], day),
             "historical_start_date_known": False} for g in gaps]
    max_count = max([g["count"] for g in gaps if isinstance(g["count"], int)] or [0])
    critical = [g["id"] for g in gaps if g.get("count") == max_count and max_count > 0]
    if not critical:
        critical = [g["id"] for g in gaps if g["id"].startswith("INDEPENDENT_PERIODS:")]
    report = {
        "schema": "autonomous-alpha-data-quality-v1", "generated_at": now.isoformat(),
        "decision_type": "RESEARCH_ONLY", "production_influence": "NONE",
        "link_as_of": link.get("as_of"), "link_status": link.get("status"),
        "link_signals": link.get("n_signals"), "link_decisions": link.get("n_decisions"),
        "link_settled_outcomes": link.get("n_outcomes"),
        "link_outcomes_status": "PENDING_OR_MISSING_NOT_TREATED_AS_VERIFIED",
        "alpha_option_status": val.get("status") or "UNKNOWN",
        "alpha_option_forward_plans": val.get("n_plans"),
        "blocked_hypotheses_by_structure": blocked,
        "observed_option_economic_status": econ.get("status"),
        "existing_option_learner": horizons,
        "evidence_discovery_as_of": evidence.get("as_of"),
        "evidence_population_rows": evidence.get("population_rows"),
        "evidence_provenance_counts": provenance,
        "data_gaps": gaps, "active_gap_ids": [g["id"] for g in gaps],
        "largest_observed_count_gaps": critical,
        "priority_note": "Counts rank historical missing data, not expected alpha; actionable priority is future "
                         "entry/exit capture and independent forward evidence.",
        "no_reconstructed_quotes": True, "no_fake_alpha": True,
    }
    return report


def write(root: Path = Path("."), *, now: datetime | None = None) -> dict:
    root = Path(root)
    report = build(root, now=now)
    _atomic_json(root / OUT, report)
    day = report["generated_at"][:10]
    path = root / HISTORY / (day + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("x", encoding="utf-8") as f:
            json.dump({"schema": "autonomous-alpha-data-gap-history-v1",
                       "generated_at": report["generated_at"],
                       "active_gap_ids": report["active_gap_ids"]},
                      f, sort_keys=True)
            f.write("\n")
    return report

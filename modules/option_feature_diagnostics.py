"""Read-only diagnosis of option learner feature availability by verified-label cohort.

No model fitting, parameter changes, historical re-labeling, quote fabrication,
network calls, signal generation, trading action or alpha-budget allocation.
The same verified-label selection as option_learner._load is used for each
horizon, so denominators match the training status exactly.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

from modules import option_learner as ol
from modules.atomic_io import atomic_write_json, atomic_write_text

FEATURES = ("impact", "surprise", "options_skew_ratio", "options_skew_25d", "dealer_oi_ratio")
HORIZONS = (5, 20)
CENSUS_START = "2026-10-30"  # First research-only census sampling; NOT a causal breakpoint
OUT = ol.OUT
SCHEMA = "option-feature-cohort-diagnostics-v1"


def _key(r: dict) -> tuple[str, str, str]:
    return str(r.get("date") or ""), str(r.get("ticker") or ""), str(r.get("signal_id") or "")


def _bucket(items: list[dict], names=FEATURES) -> dict:
    n = len(items)
    return {
        "labels": n,
        "unique_candidates": len({r["candidate_key"] for r in items}),
        "feature_coverage": {
            f: {"numeric": sum(f in r["x"] for r in items),
                "fraction": round(sum(f in r["x"] for r in items) / n, 4) if n else None}
            for f in names
        },
    }


def _split(items: list[dict], field: str) -> dict:
    groups = defaultdict(list)
    for r in items:
        groups[str(r[field])].append(r)
    return {k: _bucket(v) for k, v in sorted(groups.items())}


def _source_audit(items: list[dict]) -> dict:
    """Differentiate missing provenance from a reader/path defect.

    All counts are LABEL-weighted. Multiplicity across option structures is
    intentional and exposed separately as unique_candidates.
    """
    result = {}
    for feature in FEATURES:
        counter = dict(numeric_in_source=0, valid_and_extracted=0, missing_key=0,
                       null_value=0, invalid_non_numeric=0, reader_mismatch=0,
                       reader_without_valid_source=0, numeric_top_level_only=0)
        for item in items:
            raw = item["source_features"]
            value = raw.get(feature)
            numeric = ol._num(value) if feature in raw else None
            extracted = item["x"].get(feature)
            if feature not in raw:
                counter["missing_key"] += 1
            elif value is None:
                counter["null_value"] += 1
            elif numeric is None:
                counter["invalid_non_numeric"] += 1
            else:
                counter["numeric_in_source"] += 1
            if numeric is not None and extracted is not None and numeric == extracted:
                counter["valid_and_extracted"] += 1
            elif numeric is not None:
                counter["reader_mismatch"] += 1
            elif extracted is not None:
                counter["reader_without_valid_source"] += 1
            if numeric is None and ol._num(item["top_level"].get(feature)) is not None:
                counter["numeric_top_level_only"] += 1
        result[feature] = counter
    return result


def _items(rows: list[dict], cands: list[dict]) -> tuple[list[dict], int]:
    # Match the FIRST ledger row kept by option_learner.candidates; old
    # duplicate signal IDs are never silently remapped to a later row.
    by_key = {}
    for row in rows:
        if isinstance(row, dict) and row.get("date") and row.get("ticker"):
            by_key.setdefault(_key(row), row)
    items = []
    unmatched = 0
    for c in cands:
        raw_row = by_key.get(_key(c))
        for e in c.get("exprs") or []:
            if e.get("status") != "VERIFIED":
                continue
            if raw_row is None:
                unmatched += 1
            source = (raw_row.get("features") or {}) if raw_row else {}
            if not isinstance(source, dict):
                source = {}
            # Never reconstruct past features using new feature engineering.
            items.append({
                "candidate_key": _key(c),
                "date": str(c["date"])[:10],
                "sample": c.get("sample") or "unknown",
                "gate": c.get("gate") or "unknown",
                "expression": e.get("expression") or "unknown",
                "era": "before_research_census" if str(c["date"])[:10] < CENSUS_START
                       else "from_research_census_start",
                "directional_decision_timestamp_present": bool(raw_row and raw_row.get("decision_frozen_at")),
                "source_features": source,
                "top_level": {f: raw_row.get(f) for f in FEATURES} if raw_row else {},
                "x": e.get("x") or {},
            })
    return items, unmatched


def horizon_diagnostic(rows: list[dict], cands: list[dict], horizon: int) -> dict:
    """Pure analysis of EXACT verified labels; never makes a new label."""
    if horizon not in HORIZONS:
        raise ValueError("Only the frozen 5- and 20-day cohorts are supported")
    items, unmatched = _items(rows, cands)
    all_row_count = sum(isinstance(r, dict) and bool(r.get("date")) and bool(r.get("ticker")) for r in rows)
    raw_by_key = {_key(r): r for r in reversed(rows)
                  if isinstance(r, dict) and r.get("date") and r.get("ticker")}
    all_coverage = {f: sum(
        isinstance(r.get("features"), dict) and ol._num(r["features"].get(f)) is not None
        for r in raw_by_key.values()) for f in FEATURES}
    audit = _source_audit(items)
    mismatch = {k: v for k, v in audit.items()
                if v["reader_mismatch"] or v["reader_without_valid_source"]}
    return {
        "horizon": horizon,
        "verified_only": True,
        "selection": "option_learner.candidates -> VERIFIED (archived entry AND exit)",
        "n_ledger_rows": all_row_count,
        "n_unique_ledger_keys": len(raw_by_key),
        "all_ledger_numeric_feature_counts": all_coverage,
        "total": _bucket(items),
        "by_sample": _split(items, "sample"),
        "by_era": _split(items, "era"),
        "by_expression": _split(items, "expression"),
        "by_gate": _split(items, "gate"),
        "by_signal_day": _split(items, "date"),
        "source_audit": audit,
        "source_rows_unmatched": unmatched,
        "reader_consistency": "NOT_VERIFIABLE" if unmatched else
                              "MISMATCH" if mismatch else "CONSISTENT",
        "reader_mismatches": mismatch,
        "frozen_directional_decision_timestamp_count": sum(
            i["directional_decision_timestamp_present"] for i in items),
        "per_feature_pit_provenance": "NOT_RECORDED",
        "interpretation_limitations": [
            "Coverage describes verified labels, not the full tradable population.",
            "One candidate may contribute multiple option expressions (non-independent labels).",
            "5d and 20d have different maturity windows; raw percentages are not a paired experiment.",
            "Existing ledger does not record a separate timestamp for each feature; its exact ex-ante "
            "availability cannot be independently certified from this report.",
            "Missing features are not imputed, backfilled, or set to zero.",
        ],
    }


def _check_status(out: Path, h: int, as_of: date, diagnostic: dict, manifest: dict) -> dict:
    path = out / f"status_{h}d.json"
    if not path.exists():
        return {"status": "NO_STORED_STATUS"}
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"status": "STATUS_UNREADABLE"}
    if old.get("as_of") != as_of.isoformat():
        return {"status": "DIFFERENT_AS_OF", "stored_as_of": old.get("as_of")}
    old_man = old.get("manifest") or {}
    observed = diagnostic["total"]["feature_coverage"]
    same = (old_man.get("labels_sha256") == manifest["labels_sha256"]
            and old.get("verified_labels") == diagnostic["total"]["labels"]
            and all((old.get("feature_coverage") or {}).get(f) == observed[f]["fraction"]
                    for f in FEATURES))
    return {"status": "REPRODUCED" if same else "NOT_REPRODUCED",
            "stored_labels": old.get("verified_labels"),
            "stored_manifest": old_man.get("labels_sha256"),
            "current_manifest": manifest["labels_sha256"]}


def diagnose(as_of: date, *, ledger: Path, quotes: Path, out: Path) -> dict:
    if not isinstance(as_of, date):
        raise TypeError("as_of must be a date")
    cohorts = {}
    for h in HORIZONS:
        rows, candidates, data = ol._load(as_of, h, ledger, quotes)
        cohort = horizon_diagnostic(rows, candidates, h)
        cohort["manifest"] = ol.manifest(data, as_of)
        cohort["stored_status_reconciliation"] = _check_status(out, h, as_of, cohort, cohort["manifest"])
        if len(data) != cohort["total"]["labels"]:
            raise ValueError("Verified-label count contradicts canonical option_learner data")
        cohorts[str(h)] = cohort
    diffs = {}
    for f in FEATURES:
        a = cohorts["5"]["total"]["feature_coverage"][f]["fraction"]
        b = cohorts["20"]["total"]["feature_coverage"][f]["fraction"]
        diffs[f] = None if a is None or b is None else round(b - a, 4)
    return {
        "schema": SCHEMA, "as_of": as_of.isoformat(), "research_only": True,
        "production_influence": "NONE", "real_orders": False, "api_calls": 0, "llm_calls": 0,
        "model_or_protocol_changed": False,
        "feature_set": list(FEATURES), "research_census_cutoff": CENSUS_START,
        "horizons": cohorts, "raw_20d_minus_5d_coverage": diffs,
        "root_cause_status": (
            "SOURCE_JOIN_UNVERIFIED" if any(c["source_rows_unmatched"] for c in cohorts.values()) else
            "READER_MISMATCH_FOUND" if any(c["reader_mismatches"] for c in cohorts.values()) else
            "READ_PATH_CONSISTENT__COHORT_AND_STAGE_COMPOSITION_TO_REVIEW"),
        "required_next_decision": "Keep experiment frozen. Inspect by_sample/by_era/by_gate/by_signal_day "
                                  "and feature-specific source missingness before any feature edit or ablation.",
        "not_alpha_evidence": True,
    }


def render(report: dict) -> str:
    lines = [f"# Options-Feature-Diagnose – {report['as_of']}", "",
             "Research- und Observability-Bericht; **keine Modelländerung, keine Alpha-Bestätigung**.", "",
             f"Diagnose: **{report['root_cause_status']}**", "",
             "| Merkmal | 5-T-Labels | 5-T-Abdeckung | 20-T-Labels | 20-T-Abdeckung |",
             "|---|---:|---:|---:|---:|"]
    for f in FEATURES:
        a, b = report["horizons"]["5"]["total"], report["horizons"]["20"]["total"]
        lines.append(f"| {f} | {a['feature_coverage'][f]['numeric']}/{a['labels']} | "
                     f"{a['feature_coverage'][f]['fraction']} | "
                     f"{b['feature_coverage'][f]['numeric']}/{b['labels']} | "
                     f"{b['feature_coverage'][f]['fraction']} |")
    for h in HORIZONS:
        v = report["horizons"][str(h)]
        lines += ["", f"## {h} Tage – {v['total']['labels']} archivverifizierte Labels",
                  f"Bestandsstatus: {v['stored_status_reconciliation']['status']}; "
                  f"Reader: {v['reader_consistency']}.", "",
                  "| Kandidatengruppe | Labels | surprise | skew_ratio | skew_25d |",
                  "|---|---:|---:|---:|---:|"]
        for group, stat in v["by_sample"].items():
            cov = stat["feature_coverage"]
            lines.append(f"| {group} | {stat['labels']} | {cov['surprise']['fraction']} | "
                         f"{cov['options_skew_ratio']['fraction']} | {cov['options_skew_25d']['fraction']} |")
        lines += ["", "| Zeitraum | Labels | surprise | skew_ratio | skew_25d |",
                  "|---|---:|---:|---:|---:|"]
        for era, stat in v["by_era"].items():
            cov = stat["feature_coverage"]
            lines.append(f"| {era} | {stat['labels']} | {cov['surprise']['fraction']} | "
                         f"{cov['options_skew_ratio']['fraction']} | {cov['options_skew_25d']['fraction']} |")
        lines += ["", "Quellen-Audit (pro archivverifiziertem Label):", "",
                  "| Merkmal | Kein Feld | Null | Ungültig | Reader-Abweichung |",
                  "|---|---:|---:|---:|---:|"]
        for feature, count in v["source_audit"].items():
            lines.append(f"| {feature} | {count['missing_key']} | {count['null_value']} | "
                         f"{count['invalid_non_numeric']} | "
                         f"{count['reader_mismatch'] + count['reader_without_valid_source']} |")
    lines += ["", "## Interpretation", "",
              "Genaue Aufschlüsselungen nach Signaltag, Ablehnungsgrund und Optionsstruktur stehen "
              "in der begleitenden JSON-Datei. Die Gruppen sind beschreibend und keine kausale Identifikation.",
              "Einzelne Features besitzen im Ledger keine separat belegten Erfassungszeitpunkte: "
              "die Point-in-Time-Korrektheit ist damit **nicht unabhängig nachgewiesen**.",
              "Historische Daten, Modelle und Optionen bleiben unverändert.", ""]
    return "\n".join(lines)


def run(as_of: date, *, ledger: Path = None, quotes: Path = None, out: Path = OUT) -> dict:
    out = Path(out)
    report = diagnose(as_of, ledger=Path(ledger or ol.red.LEDGER),
                      quotes=Path(quotes or ol.red.QUOTES), out=out)
    atomic_write_json(out / "feature_diagnostics.json", report, indent=2)
    atomic_write_text(out / "feature_diagnostics.md", render(report))
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", type=date.fromisoformat, required=True)
    parser.add_argument("--ledger", type=Path, default=None)
    parser.add_argument("--quotes", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    report = run(args.as_of, ledger=args.ledger, quotes=args.quotes, out=args.out)
    print(json.dumps({"as_of": report["as_of"], "status": report["root_cause_status"],
                      "labels_5d": report["horizons"]["5"]["total"]["labels"],
                      "labels_20d": report["horizons"]["20"]["total"]["labels"]}))


if __name__ == "__main__":
    main()

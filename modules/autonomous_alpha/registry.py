"""Content-addressed, append-only experiment evidence registry.

Existing research files are the authoritative *calculations*. This registry is
an immutable snapshot journal, not a replacement for research_memory or the
promotion controller. GitHub Actions persists the JSONL journal; SQLite is
an optional, fully rebuildable local query index (never committed).

Important: "accepted" historical research is NOT prospective validation.
No backtest result here can approve an order, change a risk gate or promote a
challenger. An explicitly contaminated holdout is never counted as success.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

OUTPUT = Path("outputs/research/autonomous_alpha")
JOURNAL = OUTPUT / "experiments.jsonl"
SUMMARY = OUTPUT / "summary.json"
SCHEMA = "alpha-experiment-v1"
INPUTS = {
    "research_lab": "outputs/research/hypothesis_db.json",
    "factory_plan": "outputs/research/factory_plan.json",
    "factory_battery": "outputs/research/factory_results.json",
    "ml_models": "outputs/research/ml_research.json",
    "ml_forward": "outputs/research/ml_forward.json",
}


class EvidenceIntegrityError(ValueError):
    """Do not append on truncated, forged or inconsistent evidence."""


def canonical(x) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(x) -> str:
    return hashlib.sha256(canonical(x).encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _d(x) -> dict:
    return x if isinstance(x, dict) else {}


def _norm(x):
    """Clean non-finite floats into None, without inventing metrics."""
    import math
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {str(k): _norm(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_norm(v) for v in x]
    return x


def _metrics(record: dict, *, kind: str) -> dict:
    if kind == "ml_models":
        wf = _d(record.get("wf"))
    elif kind == "ml_forward":
        wf = record
    else:
        wf = _d(record.get("walk_forward"))
    base = _d(wf.get("base"))
    stress = _d(wf.get("stress"))
    ic = _d(wf.get("ic"))
    locked = _d(record.get("locked"))
    # Preserve uncertainty and evidence provenance: never impute zeros.
    return _norm({
        "oos_net_mean": base.get("mean"),
        "oos_stress_net_mean": stress.get("mean"),
        "oos_sharpe_ann": base.get("sharpe_ann"),
        "oos_max_dd": base.get("max_dd"),
        "oos_t_months": base.get("t_months"),
        "oos_n_cohorts": base.get("n_cohorts"),
        "oos_n_months": base.get("n_months"),
        "oos_ic": ic.get("mean_ic"),
        "oos_p_value": record.get("p_value"),
        "locked_mean_information_only": _d(locked.get("base")).get("mean"),
        "forward": record.get("forward"),  # may be null / unripe
    })


def _spec(kind: str, identifier: str, rec: dict) -> dict:
    raw = _d(rec.get("spec"))
    if kind in ("ml_models", "ml_forward"):
        return {"id": identifier, "model": identifier}
    if kind == "factory_plan":
        raw = rec
    if kind == "research_lab":
        raw = rec
    return _norm({
        "id": identifier,
        "signal": raw.get("signal"),
        "direction": raw.get("direction"),
        "family": raw.get("family"),
        "domain": raw.get("domain"),
        "horizon": raw.get("horizon"),
        "source": raw.get("source"),
    })


def _snapshot(kind: str, identifier: str, rec: dict, generated: str | None,
              source_path: str, locked_status: str, code_sha: str | None = None,
              panel_hash: str | None = None) -> dict:
    spec = _spec(kind, identifier, rec)
    status = (
        rec.get("plan_status") if kind == "factory_plan" else
        _d(rec.get("decision")).get("verdict", rec.get("registry_status"))
        if kind == "ml_models" else
        "FORWARD_OBSERVED_UNVALIDATED" if kind == "ml_forward" else
        rec.get("canonical_status") or rec.get("status") or "UNKNOWN"
    )
    record = {
        "schema": SCHEMA, "experiment_key": kind + ":" + identifier,
        "kind": kind, "hypothesis_id": identifier, "spec": spec,
        "spec_hash": digest(spec),
        "upstream_spec_hash": _d(rec.get("spec")).get("spec_hash") or rec.get("spec_hash"),
        "status": str(status).upper(),
        "source_path": source_path, "source_generated_at": generated,
        "source_record_sha256": digest(_norm(rec)),
        "model_or_code_sha": _d(rec.get("memory")).get("model_version") or rec.get("code_sha") or code_sha,
        "source_panel_hash": panel_hash,
        "parameters": (_d(rec.get("memory")).get("parameters")
                       if kind == "research_lab" else rec.get("params_by_year")
                       if kind == "ml_models" else _d(rec.get("spec")).get("parameters")),
        "metrics": _metrics(rec, kind=kind),
        "reasons": rec.get("reasons") or _d(rec.get("decision")).get("reasons") or [],
        "evidence_class": "HISTORICAL_RESEARCH",
        "locked_holdout_status": locked_status,
        "promotion_eligible": False,
        "production_impact": False,
    }
    if kind == "factory_plan":
        record["metrics"] = {}
        record["evidence_class"] = "UNTESTED_IDEA"
    if kind == "factory_battery":
        record["evidence_class"] = "HISTORICAL_ROBUSTNESS_BATTERY"
    if kind == "ml_forward":
        record["evidence_class"] = "PROSPECTIVE_OBSERVATIONS_UNVALIDATED"
    return _norm(record)


def load_inputs(root: Path = Path("."), *, strict: bool = False) -> tuple[list[dict], dict]:
    """Only read known inputs, never execute generated code or rewrite research."""
    root = Path(root)
    records: list[dict] = []
    missing: list[str] = []
    sources: dict = {}
    protocol_file = root / "config/research_protocol.yaml"
    locked_status = "UNKNOWN"
    protocol_sha = None
    if protocol_file.exists():
        protocol_sha = hashlib.sha256(protocol_file.read_bytes()).hexdigest()
        # The protocol is protected/pinned by existing tests. Detect, don't edit.
        import yaml
        settings = _d(yaml.safe_load(protocol_file.read_text(encoding="utf-8")))
        locked_status = str(_d(settings.get("periods")).get("locked_status") or "UNKNOWN")
    elif strict:
        raise EvidenceIntegrityError("missing protected research protocol")
    for kind, relative in INPUTS.items():
        path = root / relative
        if not path.exists():
            missing.append(relative)
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("expected dict")
            if kind == "research_lab":
                entries = _d(data.get("hypotheses"))
            elif kind == "factory_plan":
                entries = {v["id"]: v for v in (data.get("ideas") or [])
                           if isinstance(v, dict) and v.get("id")}
            elif kind == "factory_battery":
                entries = _d(data.get("results"))
            elif kind == "ml_forward":
                entries = _d(data.get("forward"))
            else:
                entries = _d(data.get("models"))
            if not entries:
                # An empty, valid source is distinct from a missing/broken source.
                sources[kind] = {"path": relative, "records": 0,
                                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                continue
            for identifier, row in sorted(entries.items()):
                if not isinstance(row, dict):
                    raise ValueError("non-object experiment " + str(identifier))
                records.append(_snapshot(kind, str(identifier), row, data.get("generated"),
                                         relative, locked_status, data.get("code_sha"),
                                         data.get("panel_hash")))
            sources[kind] = {"path": relative, "records": len(entries),
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        except (ValueError, TypeError, KeyError) as exc:
            # A broken available source must not be interpreted as zero results.
            raise EvidenceIntegrityError("invalid input " + relative + ": " + str(exc)) from exc
    if strict and missing:
        raise EvidenceIntegrityError("missing required inputs: " + ", ".join(missing))
    return records, {"missing_sources": missing, "sources": sources,
                     "research_protocol_sha256": protocol_sha,
                     "locked_holdout_status": locked_status}


def event_id(record: dict) -> str:
    return digest(record)


def read_journal(path: Path = JOURNAL) -> list[dict]:
    """Verify all rows before any append; tolerate repeated identical rows from
    concurrent git union merges, but never conflicting identities."""
    path = Path(path)
    if not path.exists():
        return []
    out, seen = [], {}
    with path.open("r", encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                record = event["record"]
                if not isinstance(record, dict) or record.get("schema") != SCHEMA:
                    raise EvidenceIntegrityError("invalid schema")
                eid = event["event_id"]
                if event_id(record) != eid:
                    raise EvidenceIntegrityError("content-address mismatch")
                if eid in seen:
                    if seen[eid] != record:
                        raise EvidenceIntegrityError("conflicting duplicate")
                    continue
                seen[eid] = record
                out.append(event)
            except (ValueError, KeyError, TypeError) as exc:
                raise EvidenceIntegrityError(f"corrupt experiment journal line {n}: {exc}") from exc
    return out


def _assert_specs_immutable(old: list[dict], current: list[dict]) -> None:
    seen = {}
    for rec in [*(e["record"] for e in old), *current]:
        key = rec["experiment_key"]
        h = rec["spec_hash"]
        if key in seen and seen[key] != h:
            raise EvidenceIntegrityError(
                f"frozen experiment {key} changed spec ({seen[key][:12]} -> {h[:12]}); "
                "register a NEW experiment ID")
        seen[key] = h


def append_records(records: list[dict], journal: Path = JOURNAL,
                   *, now: str | None = None) -> dict:
    journal = Path(journal)
    old = read_journal(journal)
    _assert_specs_immutable(old, records)
    seen = {e["event_id"] for e in old}
    new = []
    for rec in records:
        eid = event_id(rec)
        if eid not in seen:
            new.append({"event_id": eid, "recorded_at": now or utc_now(), "record": rec})
            seen.add(eid)
    if new:
        journal.parent.mkdir(parents=True, exist_ok=True)
        with journal.open("a", encoding="utf-8") as fh:
            for item in new:
                fh.write(canonical(item) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    return {"appended": len(new), "total_unique": len(seen)}


def latest_records(events: list[dict]) -> dict[str, dict]:
    """Use source generation time, never git merge line ordering, to select latest."""
    most_recent = {}
    for e in events:
        r = e["record"]
        key = r["experiment_key"]
        order = (r.get("source_generated_at") or "", e.get("recorded_at") or "", e["event_id"])
        if key not in most_recent or order > most_recent[key][0]:
            most_recent[key] = (order, r)
    return {k: v[1] for k, v in most_recent.items()}


def summary(events: list[dict], source_meta: dict | None = None) -> dict:
    latest = latest_records(events)
    cnt = Counter(r["status"] for r in latest.values())
    by_kind = Counter(r["kind"] for r in latest.values())
    historically_tested = [r for r in latest.values()
                           if r["evidence_class"].startswith("HISTORICAL")]
    prospectively_confirmed = sum(
        r.get("evidence_class") == "PROSPECTIVE_VALIDATED" for r in latest.values()
    )
    # This is an *observation/diagnostic*, never automatic promotion.
    return {
        "schema": SCHEMA, "generated": utc_now(),
        "n_unique_events": len(events), "n_experiments": len(latest),
        "n_historically_tested": len(historically_tested),
        "n_prospectively_confirmed": prospectively_confirmed,
        "by_status": dict(sorted(cnt.items())), "by_kind": dict(sorted(by_kind.items())),
        "locked_holdout_policy": (source_meta or {}).get("locked_holdout_status"),
        "source_health": source_meta or {},
        "production_effect": "NONE",
        "next_action": "KEEP_SHADOW_AND_COLLECT_FORWARD",
    }


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=str(path.parent), prefix="._alpha_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(value, fh, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def ingest(root: Path = Path("."), *, strict: bool = False) -> dict:
    root = Path(root)
    records, meta = load_inputs(root, strict=strict)
    journal = root / JOURNAL
    update = append_records(records, journal)
    events = read_journal(journal)
    report = {**summary(events, meta), **update}
    _atomic_json(root / SUMMARY, report)
    return report


def build_sqlite(events: list[dict], path: Path | str = ":memory:") -> sqlite3.Connection:
    """Derived local index. Safe to throw away; source of truth is JSONL."""
    conn = sqlite3.connect(str(path))
    conn.execute("""CREATE TABLE IF NOT EXISTS experiments (
        event_id TEXT PRIMARY KEY, experiment_key TEXT NOT NULL,
        kind TEXT NOT NULL, status TEXT NOT NULL,
        spec_hash TEXT NOT NULL, source_generated_at TEXT,
        recorded_at TEXT NOT NULL, payload TEXT NOT NULL
    )""")
    for e in events:
        r = e["record"]
        conn.execute("""INSERT OR IGNORE INTO experiments
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (e["event_id"], r["experiment_key"], r["kind"], r["status"],
             r["spec_hash"], r.get("source_generated_at"),
             e.get("recorded_at") or "", canonical(r)))
    conn.commit()
    return conn


def leaderboard(events: list[dict], limit: int = 15) -> list[dict]:
    """Historical **research** ranking; never label it a live options strategy."""
    rows = []
    for rec in latest_records(events).values():
        m = _d(rec.get("metrics"))
        net = m.get("oos_net_mean")
        if not isinstance(net, (int, float)):
            continue
        rows.append({"experiment_key": rec["experiment_key"],
                     "status": rec["status"], "evidence_class": rec["evidence_class"],
                     "oos_net_mean": net, "oos_stress_net_mean": m.get("oos_stress_net_mean"),
                     "sharpe_ann": m.get("oos_sharpe_ann"),
                     "max_dd": m.get("oos_max_dd"), "n_cohorts": m.get("oos_n_cohorts"),
                     "promotion_eligible": False})
    return sorted(rows, key=lambda r: (-(r["oos_stress_net_mean"] if isinstance(
        r["oos_stress_net_mean"], (int, float)) else -1e99),
        -r["oos_net_mean"], r["experiment_key"]))[:max(0, limit)]

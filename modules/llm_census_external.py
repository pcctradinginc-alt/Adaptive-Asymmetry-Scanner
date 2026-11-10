"""Dedicated research-only LLM census, detached from Champion scanner runtime.

Only the deterministically selected, signal-time-frozen candidates may be
analyzed; no backfills of past days with later news/data are permitted.
Results are sidecar records, NEVER amendments to the immutable trade ledger.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path

from modules import llm_census as lc

INPUT = Path("outputs/research/llm_census_inputs")
OUTPUT = Path("outputs/research/llm_census_runs")
MIN_MONTHLY_HEADROOM_USD = 3.0
MAX_K = 10
MAX_SECONDS = 420


def freeze(picks: list[dict], design: dict, *, root: Path = INPUT, now=None) -> dict:
    """Create-only at signal time. Empty selection is an observable input."""
    now = now or datetime.now(timezone.utc)
    day = str(design["day"])
    if now.astimezone(timezone.utc).date().isoformat() != day:
        return {"status": "DATE_MISMATCH", "day": day}
    payload = {"schema": "llm-census-frozen-v1", "day": day, "frozen_at": now.isoformat(),
               "design": design, "picks": picks, "production_influence": "NONE"}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    payload["sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{day}.json"
    if path.exists():
        return {"status": "ALREADY_FROZEN", "day": day, "path": str(path)}
    try:
        with path.open("x", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, default=str, indent=1))
    except FileExistsError:
        return {"status": "ALREADY_FROZEN", "day": day}
    return {"status": "FROZEN", "day": day, "n": len(picks), "path": str(path)}


def read_frozen(path: Path, *, now=None) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    sha = data.pop("sha256", None)
    canonical = json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)
    if sha != hashlib.sha256(canonical.encode()).hexdigest():
        raise ValueError("Frozen cohort has wrong checksum")
    if data.get("schema") != "llm-census-frozen-v1":
        raise ValueError("Unknown snapshot schema")
    now = now or datetime.now(timezone.utc)
    day = now.astimezone(timezone.utc).date().isoformat()
    if data.get("day") != day or Path(path).stem != day:
        raise ValueError("STALE_COHORT: no retrospective LLM calls")
    if data.get("design", {}).get("day") != day or len(data.get("picks") or []) > MAX_K:
        raise ValueError("Invalid selection size or day")
    if len({x.get("ticker") for x in data.get("picks") or []}) != len(data.get("picks") or []):
        raise ValueError("Duplicate ticker selection")
    data["sha256"] = sha
    return data


def budget_ok(b: dict) -> bool:
    if not isinstance(b, dict) or b.get("level") != "OK" or b.get("soft_pressure"):
        return False
    try:
        budget = float(b["monthly_api_budget_usd"])
        projected = float(b["projected_month_usd"])
        spent = float(b["month_usd"])
        weekly = float(b["week_usd"])
        llm_week = float(b["weekly_budget_usd"])
        return (all(math.isfinite(v) for v in (budget, projected, spent, weekly, llm_week))
                and budget - max(projected, spent) >= MIN_MONTHLY_HEADROOM_USD
                and llm_week - weekly >= 1.0)
    except (TypeError, KeyError, ValueError):
        return False


class FrozenContextAnalyzer:
    """Reuse production's JSON output schema but forbid new 48h price and macro context."""
    def __init__(self):
        from modules.deep_analysis import DeepAnalysis
        from modules.config import cfg
        self.engine = DeepAnalysis()
        self.engine._macro = {"data_available": False, "macro_regime": "unknown"}
        self.engine._get_48h_move = lambda _ticker: None
        self.engine._active_model = cfg.models.deep_analysis
        self.engine._routing = lambda: {"active": cfg.models.deep_analysis}
        self.skipped_for_time = []

    def run(self, picks, deadline):
        result = self.engine.run(picks, deadline=deadline)
        self.skipped_for_time = self.engine.skipped_for_time
        return result


class NoLiveMismatch:
    """Never fetch post-signal market prices to infer a contemporaneous mismatch."""
    def __init__(self):
        self.shadow = {}

    def _score(self, a):
        return None


def process(day: str, *, inputs: Path = INPUT, outputs: Path = OUTPUT,
            now=None, budget_fn=None, analyzer_factory=FrozenContextAnalyzer,
            allow_llm=None) -> dict:
    now = now or datetime.now(timezone.utc)
    if date.fromisoformat(day) != now.astimezone(timezone.utc).date():
        return {"status": "STALE_DAY", "day": day, "production_influence": "NONE"}
    file = inputs / f"{day}.json"
    if not file.exists():
        return {"status": "INPUT_NOT_COMMITTED", "day": day, "production_influence": "NONE"}
    dst = outputs / f"{day}.json"
    if dst.exists():
        return {"status": "ALREADY_RECORDED", "day": day, "production_influence": "NONE"}
    frozen = read_frozen(file, now=now)
    if allow_llm is None:
        allow_llm = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if not allow_llm:
        return {"status": "NO_API_KEY", "day": day, "production_influence": "NONE"}
    if budget_fn is None:
        from modules.cost_telemetry import budget_status
        budget_fn = budget_status
    try:
        budget = budget_fn()
    except Exception:
        return {"status": "BUDGET_UNKNOWN", "day": day, "production_influence": "NONE"}
    if not budget_ok(budget):
        return {"status": "BUDGET_GUARD", "day": day, "production_influence": "NONE"}
    obs = []
    k = len(frozen["picks"])
    if k:
        result = lc.run(
            frozen["picks"], frozen["design"], deadline=time.monotonic() + MAX_SECONDS,
            analyzer=analyzer_factory(), scorer_cls=NoLiveMismatch,
            note=lambda ticker, **fields: obs.append({"ticker": ticker, "fields": fields}))
    else:
        result = {"status": "empty", **frozen["design"]}
    payload = {"schema": "llm-census-external-v1", "day": day, "source_sha256": frozen["sha256"],
               "executed_at": datetime.now(timezone.utc).isoformat(), "research_only": True,
               "production_influence": "NONE", "summary": result, "observations": obs,
               "note": "LLM context is frozen; 48h move and macro deliberately unavailable. "
                       "Descriptive, no verified executable option return."}
    outputs.mkdir(parents=True, exist_ok=True)
    try:
        with dst.open("x", encoding="utf-8") as f:
            json.dump(payload, f, indent=1, ensure_ascii=False, default=str)
    except FileExistsError:
        return {"status": "ALREADY_RECORDED", "day": day, "production_influence": "NONE"}
    return {"status": "RECORDED", "day": day, "selected": k,
            "analyzed": result.get("analyzed", 0), "production_influence": "NONE"}


def joined_ledger_rows(rows: list[dict], *, root: Path = OUTPUT) -> list[dict]:
    """Join sidecars for DESCRIPTIVE census report only, never mutate source rows."""
    import copy
    by_date = {}
    for file in sorted(root.glob("????-??-??.json")) if root.exists() else []:
        try:
            payload = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if payload.get("schema") != "llm-census-external-v1":
            continue
        day = payload.get("day")
        for observation in payload.get("observations") or []:
            if observation.get("ticker") and isinstance(observation.get("fields"), dict):
                by_date[(day, observation["ticker"])] = observation["fields"]
    out = []
    for row in rows:
        match = by_date.get((row.get("date"), row.get("ticker")))
        if not match:
            out.append(row)
            continue
        r = copy.deepcopy(row)
        r["features"] = {**(r.get("features") or {}), **match}
        out.append(r)
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["run", "report"])
    p.add_argument("--date", default=datetime.now(timezone.utc).date().isoformat())
    args = p.parse_args(argv)
    if args.action == "run":
        result = process(args.date)
        print(json.dumps(result, ensure_ascii=False))
    else:
        from modules.challenger import load_ledger_rows
        rows = joined_ledger_rows(load_ledger_rows())
        report = lc.report(rows, as_of=date.fromisoformat(args.date))
        lc.OUT.parent.mkdir(parents=True, exist_ok=True)
        lc.OUT.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"analyzed_rows": report["analyzed_rows"],
                          "days": report["days"], "production_influence": "NONE"}))


if __name__ == "__main__":
    main()

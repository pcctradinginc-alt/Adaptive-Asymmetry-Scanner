"""Independent, point-in-time LLM census. No production or Ledger writes.

Freeze random picks *during* scanner ingestion, before gates. Another workflow
analyzes only those immutable inputs, independently of the scanner runtime.
"""
from __future__ import annotations
import argparse
import json
import logging
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
import yaml

from modules import llm_census as lc

log = logging.getLogger(__name__)
CONFIG = Path("config/llm_census_offline.yaml")
INPUT = Path("outputs/research/llm_census_inputs")
OUT = Path("outputs/research/llm_census_offline")
MAX_K = 10
MAX_TIME_S = 600


def policy(path: Path = CONFIG) -> dict:
    try:
        p = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(p, dict) or set(p) != {
            "enabled", "k", "max_k", "max_runtime_s",
            "projected_monthly_cap_usd", "minimum_monthly_headroom_usd"}:
            raise ValueError("unexpected census policy keys")
        if type(p["enabled"]) is not bool or type(p["k"]) is not int or type(p["max_k"]) is not int:
            raise ValueError("invalid census counts")
        if not 0 <= p["k"] <= p["max_k"] <= MAX_K:
            raise ValueError("census cap exceeded")
        if type(p["max_runtime_s"]) is not int or not 1 <= p["max_runtime_s"] <= MAX_TIME_S:
            raise ValueError("invalid runtime")
        for k in ("projected_monthly_cap_usd", "minimum_monthly_headroom_usd"):
            x = p[k]
            if type(x) not in (float, int) or not math.isfinite(x) or x <= 0:
                raise ValueError("invalid budget")
        return p
    except (OSError, ValueError, yaml.YAMLError) as e:
        log.error("LLM census offline disabled: %s", e)
        return {"enabled": False, "reason": "invalid_config"}


def freeze(candidates: list[dict], day: str, *, cfg: dict | None = None,
           root: Path = INPUT) -> dict:
    p = policy() if cfg is None else cfg
    if not p.get("enabled"):
        return {"status": "DISABLED"}
    target = root / f"{day}.json"
    if target.exists():
        return {"status": "ALREADY_FROZEN"}
    picks, design = lc.select(candidates, day, min(p["k"], MAX_K))
    payload = {"schema": "offline_llm_census_v1",
               "frozen_at": datetime.now(timezone.utc).isoformat(),
               "design": design, "candidates": picks}
    root.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, default=str)
        fh.write("\n")
    return {"status": "FROZEN", **design}


def budget_ok(status: dict, cfg: dict) -> bool:
    if status.get("level") != "OK" or status.get("soft_pressure"):
        return False
    try:
        cap = float(cfg["projected_monthly_cap_usd"])
        budget_cap = float(status["monthly_api_budget_usd"])
        projection = float(status["projected_month_usd"])
        current = float(status["month_usd"])
        headroom = float(cfg["minimum_monthly_headroom_usd"])
        return (all(math.isfinite(x) for x in (cap, budget_cap, projection, current, headroom))
                and min(cap, budget_cap) - max(projection, current) >= headroom)
    except (KeyError, ValueError, TypeError):
        return False


def analyze_frozen(frozen: dict, *, cfg: dict, budget: dict,
                   analyzer=None, scorer_cls=None) -> dict:
    design = frozen.get("design") or {}
    picks = frozen.get("candidates") or []
    report = {"status": "SKIPPED", "day": design.get("day"),
              "policy": lc.POLICY, "production_influence": "NONE",
              "orders_created": 0, "n": design.get("n"), "k": design.get("k"),
              "analyzed": 0, "rows": []}
    if not cfg.get("enabled"):
        return {**report, "reason": "disabled"}
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return {**report, "reason": "missing_anthropic_key"}
    if not budget_ok(budget, cfg):
        return {**report, "reason": "budget_guard"}
    if len(picks) > cfg["k"] or len(picks) > MAX_K or not design.get("day"):
        return {**report, "status": "ERROR", "reason": "invalid_frozen_cohort"}
    if analyzer is None:
        from modules.deep_analysis import DeepAnalysis
        from modules.config import cfg as production_config

        class CensusAnalyzer(DeepAnalysis):
            def _routing(self):
                # Disable A/B duplicate LLM calls in the independent census.
                self._active_model = production_config.models.deep_analysis
                return {"active": self._active_model}
        analyzer = CensusAnalyzer()
    notes = []
    def note(ticker, **fields):
        # No candidate_ledger.note() here; preserve finalized production ledger.
        notes.append({"ticker": ticker, "fields": fields})
    deadline = time.monotonic() + min(cfg["max_runtime_s"], MAX_TIME_S)
    try:
        result = lc.run(picks, design, deadline, analyzer=analyzer, scorer_cls=scorer_cls, note=note)
        report.update(status="DONE", analyzed=result.get("analyzed", 0),
                      not_analyzed=result.get("not_analyzed", 0), rows=notes)
    except Exception as e:
        report.update(status="ERROR", reason=f"{type(e).__name__}: {str(e)[:160]}", rows=notes)
    return report


# Audit 2026-11-26: Der Workflow rief --day "$(date -u)" auf; der Cron (22:30 UTC) startet real 4–7 h später, also
# am NÄCHSTEN UTC-Tag -> es gab nie eine Eingabe für "heute", seit dem 11.11. wurde keine Kohorte analysiert.
# Jetzt: ohne --day alle eingefrorenen Kohorten ohne Ergebnis. Analysiert wird nur zeitnah zum Einfrieren
# (Deep Analysis/Mismatch lesen Live-Kurse; eine Tage später analysierte Kohorte wäre nicht mehr point-in-time);
# ältere Kohorten erhalten einmalig einen SKIPPED-Bericht (stale_cohort) – sichtbar statt still verloren.
MAX_COHORT_AGE_H = 36


def pending_cohorts(input_root: Path = INPUT, out_root: Path = OUT,
                    now: datetime | None = None) -> tuple[list[str], list[str]]:
    """(fresh, stale): Tage mit eingefrorener Kohorte und ohne Ergebnisdatei; fresh = frozen_at <= MAX_COHORT_AGE_H."""
    now = now or datetime.now(timezone.utc)
    fresh, stale = [], []
    for path in sorted(Path(input_root).glob("*.json")) if Path(input_root).exists() else []:
        day = path.stem
        if (Path(out_root) / f"{day}.json").exists():
            continue
        try:
            frozen_at = datetime.fromisoformat(json.loads(path.read_text(encoding="utf-8"))["frozen_at"])
            if frozen_at.tzinfo is None:
                frozen_at = frozen_at.replace(tzinfo=timezone.utc)
        except (OSError, ValueError, KeyError, TypeError):
            stale.append(day)
            continue
        age_h = (now - frozen_at).total_seconds() / 3600
        (fresh if 0 <= age_h <= MAX_COHORT_AGE_H else stale).append(day)
    return fresh, stale


def _write_once(target: Path, report: dict) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--day", default=None,
                   help="ein bestimmter Tag; ohne Angabe: alle zeitnah eingefrorenen Kohorten ohne Ergebnis")
    p.add_argument("--input", type=Path, default=INPUT)
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args(argv)
    if args.day:
        return _run_day(args.day, args.input, args.out)
    fresh, stale = pending_cohorts(args.input, args.out)
    for day in stale:
        _write_once(args.out / f"{day}.json", {
            "status": "SKIPPED", "reason": "stale_cohort", "day": day, "policy": lc.POLICY,
            "production_influence": "NONE", "orders_created": 0, "analyzed": 0, "rows": [],
            "max_cohort_age_h": MAX_COHORT_AGE_H, "completed_at": datetime.now(timezone.utc).isoformat()})
        print(f"Kohorte {day}: nicht mehr zeitnah (> {MAX_COHORT_AGE_H} h) – SKIPPED stale_cohort, keine LLM-Analyse")
    if not fresh:
        print("Keine zeitnah eingefrorene Kohorte ohne Ergebnis.")
        return 0
    return max(_run_day(day, args.input, args.out) for day in fresh)


def _run_day(day: str, input_root: Path, out_root: Path) -> int:
    cfg = policy()
    target = out_root / f"{day}.json"
    if target.exists():
        print("Already evaluated (append-only): " + str(target))
        return 0
    frozen_path = input_root / f"{day}.json"
    if not frozen_path.exists():
        print("No frozen, point-in-time cohort: " + str(frozen_path))
        return 0
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    from modules import cost_telemetry
    try:
        budget = cost_telemetry.budget_status()
    except Exception:
        budget = {}
    report = analyze_frozen(frozen, cfg=cfg, budget=budget)
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    _write_once(target, report)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return 0 if report["status"] in ("DONE", "SKIPPED") else 1


if __name__ == "__main__":
    raise SystemExit(main())

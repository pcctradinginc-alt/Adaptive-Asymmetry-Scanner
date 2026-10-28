"""Prospective factor prediction journal with strictly matured underlying labels.

No trade execution, no LLM and no synthetic option P&L. Signals originate
ONLY from frozen AUTO-* proposals. Predictions are registered AFTER factor
preregistration and BEFORE any forward outcome can be used. Only the most
recent PIT panel date can be snapshotted. Performance is stock cross-sectional
GROSS spread, not a tradable options return or promotion evidence.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from . import explorer
from .registry import EvidenceIntegrityError, canonical, digest, _atomic_json

PREDICTIONS = Path("outputs/research/autonomous_alpha/factor_predictions")
OUTCOMES = Path("outputs/research/autonomous_alpha/factor_outcomes")
REPORT = Path("outputs/research/autonomous_alpha/factor_forward.json")
MIN_NAMES = 35
EACH_TAIL = 10
MAX_NEW_PER_DATE = 40
MAX_ROWS_PER_FILE = 3000


def _read(directory: Path) -> list[dict]:
    events = {}
    seen_semantic_ids = {}
    if not directory.exists():
        return []
    for p in sorted(directory.glob("????-??.jsonl")):
        with p.open(encoding="utf-8") as fh:
            for n, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    e = json.loads(line)
                    if not isinstance(e, dict) or e.get("sha256") != digest(e.get("payload")):
                        raise ValueError("incorrect content hash")
                    key = e["sha256"]
                    payload = e["payload"]
                    kind = payload.get("schema")
                    if kind == "alpha-factor-pred-v1":
                        logical_id = (kind, payload["hypothesis_id"], payload["prediction_date"])
                    elif kind == "alpha-factor-label-v1":
                        logical_id = (kind, payload["prediction_id"])
                    else:
                        raise ValueError("unexpected factor journal record type")
                    if logical_id in seen_semantic_ids and seen_semantic_ids[logical_id] != key:
                        raise ValueError("conflicting immutable prediction/outcome identity")
                    seen_semantic_ids[logical_id] = key
                    if key not in events:
                        events[key] = e
                    elif events[key]["payload"] != e["payload"]:
                        raise ValueError("conflicting duplicate")
                except (TypeError, ValueError, KeyError) as exc:
                    raise EvidenceIntegrityError(f"broken factor journal {p}:{n}: {exc}") from exc
                if len(events) > MAX_ROWS_PER_FILE * max(1, len(list(directory.glob("*.jsonl")))):
                    raise EvidenceIntegrityError("factor archive over global size budget")
    return list(events.values())


def _append(directory: Path, records: list[dict], existing: list[dict]) -> int:
    known = {e["sha256"] for e in existing}
    count = 0
    for obj in records:
        h = digest(obj)
        if h in known:
            continue
        day = obj["prediction_date"]
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / (day[:7] + ".jsonl")).open("a", encoding="utf-8") as out:
            out.write(canonical({"sha256": h, "payload": obj}) + "\n")
            out.flush()
            os.fsync(out.fileno())
        known.add(h)
        count += 1
    return count


def _trusted_panel(path: Path):
    """Local CI-generated panel only, NEVER a model-provided path or network URL."""
    if not path.is_file():
        raise FileNotFoundError("CI research panel cache missing")
    with path.open("rb") as fh:
        panel = pickle.load(fh)  # noqa: S301  Trusted same-run ML_PANEL_CACHE only
    if not isinstance(panel, pd.DataFrame):
        raise EvidenceIntegrityError("trusted panel has invalid type")
    for column in ("date", "ticker", "fwd_xs_20", "label_end_20"):
        if column not in panel.columns:
            raise EvidenceIntegrityError("missing trusted panel field: " + column)
    return panel


def process(root: Path = Path("."), *, panel=None, cache: Path | None = None,
            now: datetime | None = None, persist: bool = True) -> dict:
    root = Path(root)
    now = now or datetime.now(timezone.utc)
    today = now.date()
    if panel is None:
        if cache is None:
            raise EvidenceIntegrityError("requires same-job trusted panel cache")
        panel = _trusted_panel(Path(cache))
    if not isinstance(panel, pd.DataFrame) or panel.empty:
        raise EvidenceIntegrityError("missing PIT panel")
    p = panel.copy()
    p["date"] = pd.to_datetime(p["date"])
    valid = p.loc[p["date"].dt.date <= today]
    if valid.empty:
        raise EvidenceIntegrityError("PIT panel contains no observations as of today")
    latest = valid["date"].max()
    day = latest.date()
    point = valid.loc[valid["date"] == latest].copy()
    if len(point) < MIN_NAMES:
        raise EvidenceIntegrityError("PIT cross-section too small")
    frozen = explorer._read(root / explorer.PROPOSALS)
    if len(frozen) > MAX_NEW_PER_DATE:
        raise EvidenceIntegrityError("unbounded prospective factor population")
    prior = _read(root / PREDICTIONS)
    prior_ids = {(x["payload"]["hypothesis_id"], x["payload"]["prediction_date"]) for x in prior}
    predictions = []
    from modules.research_lab import eval_signal, validate_expr
    for e in frozen:
        h = e["hypothesis"]
        registered = datetime.fromisoformat(e["registered_at"])
        if registered.tzinfo is None:
            raise EvidenceIntegrityError("proposal registration must be timezone aware")
        if day <= registered.date():
            continue  # never retroactively use a same-day or earlier price
        identity = (h["id"], day.isoformat())
        if identity in prior_ids:
            continue
        validate_expr(h["signal"])
        series = eval_signal(point, h["signal"])
        scored = pd.DataFrame({"ticker": point["ticker"].astype(str), "score": series})
        scored = scored.dropna(subset=["score"]).drop_duplicates("ticker", keep=False)
        if len(scored) < MIN_NAMES:
            continue
        scored = scored.sort_values(["score", "ticker"], kind="mergesort")
        # Avoid false top/bottom difference when a factor is constant.
        if float(scored["score"].max()) <= float(scored["score"].min()):
            continue
        bottom = scored.head(EACH_TAIL)["ticker"].tolist()
        top = scored.tail(EACH_TAIL)["ticker"].tolist()
        if set(top) & set(bottom):
            raise EvidenceIntegrityError("overlapping prospective factor tails")
        predictions.append({
            "schema": "alpha-factor-pred-v1", "hypothesis_id": h["id"],
            "spec_hash": h["spec_hash"], "registered_at": e["registered_at"],
            "prediction_date": day.isoformat(),
            "top": top, "bottom": bottom, "n_universe": len(scored),
            "horizon_trading_days": 20, "metric": "GROSS_TOP_MINUS_BOTTOM_STOCK_RETURN",
            "options_profit": None, "promotion_eligible": False,
        })
    outcomes_prior = _read(root / OUTCOMES)
    already = {e["payload"]["prediction_id"] for e in outcomes_prior}
    outcomes = []
    # Use exactly the same point-in-time cohort and labels that matured AFTER
    # entry, before evaluation date. Never train/re-evaluate historical tails.
    future = (pd.concat([valid[["date", "ticker", "fwd_xs_20", "label_end_20"]]], ignore_index=True)
              .drop_duplicates(["date","ticker"], keep=False))
    by_key = future.set_index(["date", "ticker"])
    pending, stale, eligible = 0, 0, 0
    for e in prior:
        pred = e["payload"]
        eid = e["sha256"]
        if eid in already:
            continue
        pdate = date.fromisoformat(pred["prediction_date"])
        if pdate >= today:
            pending += 1
            continue
        if today < pdate + timedelta(days=20):
            pending += 1
            continue
        vals = {}
        all_matured = True
        for ticker in pred["top"] + pred["bottom"]:
            key = (pd.Timestamp(pred["prediction_date"]), ticker)
            if key not in by_key.index:
                all_matured = False
                break
            row = by_key.loc[key]
            if isinstance(row, pd.DataFrame):  # duplicate ambiguity must not resolve arbitrarily
                all_matured = False
                break
            lab_date = pd.to_datetime(row["label_end_20"], errors="coerce")
            ret = float(row["fwd_xs_20"]) if pd.notna(row["fwd_xs_20"]) else None
            if pd.isna(lab_date) or lab_date.date() > today or ret is None or not math.isfinite(ret):
                all_matured = False
                break
            vals[ticker] = ret
        if not all_matured:
            pending += 1
            if today > pdate + timedelta(days=70):
                stale += 1
            continue
        top_return = sum(vals[t] for t in pred["top"])/len(pred["top"])
        bottom_return = sum(vals[t] for t in pred["bottom"])/len(pred["bottom"])
        outcomes.append({
            "schema": "alpha-factor-label-v1",
            "prediction_id": eid,
            "hypothesis_id": pred["hypothesis_id"],
            "prediction_date": pred["prediction_date"],
            "evaluated_on": today.isoformat(),
            "top_mean_gross": round(top_return, 9),
            "bottom_mean_gross": round(bottom_return, 9),
            "top_minus_bottom_gross": round(top_return-bottom_return, 9),
            "n_top": len(pred["top"]), "n_bottom": len(pred["bottom"]),
            "status": "MATURED_GROSS_STOCK_ONLY",
            "net_options_pnl": None, "promotion_eligible": False,
        })
        eligible += 1
    if persist:
        new_pred = _append(root / PREDICTIONS, predictions, prior)
        new_lab = _append(root / OUTCOMES, outcomes, outcomes_prior)
    else:
        new_pred, new_lab = len(predictions), len(outcomes)
    report = {
        "schema": "alpha-factor-forward-v1",
        "as_of": today.isoformat(), "latest_pit_date": day.isoformat(),
        "n_frozen_hypotheses": len(frozen),
        "n_new_predictions": new_pred, "n_new_matured_labels": new_lab,
        "n_pending_labels": pending, "n_stale_missing_labels": stale,
        "n_prior_predictions": len(prior), "n_prior_matured_labels": len(outcomes_prior),
        "stock_factor_alpha_proven": False, "options_alpha_proven": False,
        "outcome_type": "HISTORICALLY_UNSEEN_MATURED_STOCK_GROSS_ONLY",
        "production_influence": "NONE", "llm_calls": 0,
    }
    if persist:
        _atomic_json(root / REPORT, report)
    return report

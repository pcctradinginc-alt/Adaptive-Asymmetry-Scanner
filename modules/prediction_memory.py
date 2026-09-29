"""
modules/prediction_memory.py – Modell-Gedächtnis (append-only, versioniert)

Jede Research-Prognose (Top-Liste des aktiven Ensembles, High-Confidence-
Kandidaten) wird als unveränderliches Ereignis gespeichert; realisierte
Ergebnisse kommen später als EIGENES Ereignis dazu. Nichts wird überschrieben:

  outputs/research/prediction_memory/YYYY-MM.jsonl
    {"type": "prediction", "prediction_id", "timestamp", "signal_date", "ticker",
     "model_versions", "meta_model_version", "features_version", "regime", "sector",
     "raw_model_predictions", "model_weights", "final_prediction", "predicted_return",
     "predicted_drawdown", "predicted_probability", "uncertainty", "calibration_state",
     "data_quality", "historical_analogy_score", "signal_reason", "code_sha", "panel_hash"}
    {"type": "outcome", "prediction_id", "recorded_at", "actual_return", "actual_return_60",
     "actual_drawdown", "actual_MFE", "actual_MAE", "trade_outcome", "error_category"}

prediction_id = Hash aus (signal_date, ticker, Modell-/Meta-/Feature-Version):
dieselbe Prognose wird nie doppelt gespeichert, eine neue Version erzeugt eine
neue id. Reproduzierbarkeit: Code-SHA und Panel-Hash werden mitgeschrieben.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

MEMORY_DIR = Path("outputs/research/prediction_memory")
FEATURES_VERSION = "fs-v1"          # modules/ml_research.ALL_FEATURES, Stand 2026-09-29


def prediction_id(signal_date: str, ticker: str, model_versions: dict, meta_version: str) -> str:
    raw = json.dumps([signal_date, ticker, sorted(model_versions.items()), meta_version, FEATURES_VERSION])
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def _events(mem_dir: Path) -> list[dict]:
    out = []
    for p in sorted(mem_dir.glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                log.warning(f"prediction_memory: defekte Zeile in {p.name} übersprungen")
    return out


def _append(events: list[dict], mem_dir: Path) -> None:
    mem_dir.mkdir(parents=True, exist_ok=True)
    by_month: dict = {}
    for e in events:
        month = (e.get("signal_date") or e.get("recorded_at") or "")[:7] or "unknown"
        by_month.setdefault(month, []).append(e)
    for month, es in by_month.items():
        with open(mem_dir / f"{month}.jsonl", "a", encoding="utf-8") as fh:
            for e in es:
                fh.write(json.dumps(e, sort_keys=True, default=str) + "\n")


def record_predictions(rows: list[dict], mem_dir: Path = MEMORY_DIR) -> int:
    """Hängt neue Prognose-Ereignisse an (bereits vorhandene ids werden übersprungen)."""
    have = {e["prediction_id"] for e in _events(mem_dir) if e.get("type") == "prediction"}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    new = []
    for r in rows:
        pid = prediction_id(r["signal_date"], r["ticker"], r.get("model_versions") or {}, r.get("meta_model_version") or "")
        if pid in have:
            continue
        have.add(pid)
        new.append({**r, "type": "prediction", "prediction_id": pid, "timestamp": now,
                    "features_version": FEATURES_VERSION})
    if new:
        _append(new, mem_dir)
    return len(new)


def error_category(actual_xs: float, actual_ret_60: float | None, interval: list | None) -> str:
    if interval and actual_ret_60 is not None and pd.notna(actual_ret_60):
        lo, hi = interval[0], interval[1]
        if lo is not None and actual_ret_60 < lo:
            return "below_interval"
        if hi is not None and actual_ret_60 > hi:
            return "above_interval"
    return "correct_direction" if actual_xs > 0 else "wrong_direction"


def record_outcomes(panel: pd.DataFrame, mem_dir: Path = MEMORY_DIR) -> int:
    """Outcome-Ereignis für jede Prognose, deren 20-Tage-Label jetzt feststeht
    (60-Tage-Werte, sofern ebenfalls fertig; sonst None – kein zweites Ereignis
    nötig, die Auswertung liest die 60d-Werte bei Bedarf aus dem Panel)."""
    ev = _events(mem_dir)
    done = {e["prediction_id"] for e in ev if e.get("type") == "outcome"}
    preds = [e for e in ev if e.get("type") == "prediction" and e["prediction_id"] not in done]
    if not preds:
        return 0
    lab = panel.set_index(["date", "ticker"])
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    new = []
    for p in preds:
        key = (pd.Timestamp(p["signal_date"]), p["ticker"])
        if key not in lab.index:
            continue
        r = lab.loc[key]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        if pd.isna(r.get("fwd_xs_20")):
            continue
        xs = float(r["fwd_xs_20"])
        r60 = float(r["fwd_ret_60"]) if pd.notna(r.get("fwd_ret_60")) else None
        new.append({"type": "outcome", "prediction_id": p["prediction_id"], "signal_date": p["signal_date"],
                    "recorded_at": now, "actual_return": round(xs, 5), "actual_return_60": r60 if r60 is None else round(r60, 5),
                    "actual_drawdown": round(float(r["mae_20"]), 5) if pd.notna(r.get("mae_20")) else None,
                    "actual_MFE": round(float(r["mfe_20"]), 5) if pd.notna(r.get("mfe_20")) else None,
                    "actual_MAE": round(float(r["mae_20"]), 5) if pd.notna(r.get("mae_20")) else None,
                    "trade_outcome": "win" if xs > 0 else "loss",
                    "error_category": error_category(xs, r60, (p.get("uncertainty") or {}).get("interval_80"))})
    if new:
        _append(new, mem_dir)
    return len(new)


def load_memory(mem_dir: Path = MEMORY_DIR) -> pd.DataFrame:
    """Prognosen mit (falls vorhanden) jüngstem Outcome-Ereignis."""
    ev = _events(mem_dir)
    preds = {e["prediction_id"]: e for e in ev if e.get("type") == "prediction"}
    outs = {}
    for e in ev:
        if e.get("type") == "outcome":
            outs[e["prediction_id"]] = e
    rows = []
    for pid, p in preds.items():
        o = outs.get(pid, {})
        rows.append({**p, **{k: v for k, v in o.items() if k not in ("type", "prediction_id", "signal_date")}})
    return pd.DataFrame(rows)

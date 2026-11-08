"""Autonomous feature discovery v3: bounded PIT baseline/challenger shadow study.

No broker calls, LLMs, training on future outcomes, production hooks, or
promotion. Freezes prospective predictions before outcomes and later scores
the SAME ticker/date population with/without the candidate feature.
Retrospective scores are explicitly exploratory; forward scores cannot
grant approval without an independently registered confirmatory protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from modules.external.archive import ExternalArchive
from modules.research_feature_store import PITResearchStore

PROTOCOL = Path("config/research_feature_discovery_v3.yaml")
OUT = Path("outputs/research/feature_discovery_v3")
JOURNAL = "predictions.jsonl"
OUTCOMES = "outcomes.jsonl"
REPORT = "status.json"


def load_protocol(path: Path = PROTOCOL) -> tuple[dict, str]:
    raw = path.read_bytes()
    p = yaml.safe_load(raw)
    if not isinstance(p, dict) or p.get("research_only") is not True:
        raise ValueError("FD3 requires research_only=true")
    if p.get("target") != "fwd_xs_20" or p.get("label_end") != "label_end_20":
        raise ValueError("Changing the target requires a new review and implementation")
    hs = p.get("hypotheses") or []
    maximum = int(p["evaluation"]["max_hypotheses"])
    if not 1 <= len(hs) <= maximum or len({h["id"] for h in hs}) != len(hs):
        raise ValueError("FD3 experiment budget exceeded or duplicate hypothesis IDs")
    if len({h["feature"] for h in hs}) != len(hs):
        raise ValueError("FD3 duplicate features")
    return p, hashlib.sha256(raw).hexdigest()


def _rank_ic(y, p) -> float | None:
    if len(y) < 5 or pd.Series(p).nunique() < 2 or pd.Series(y).nunique() < 2:
        return None
    return float(pd.Series(y).rank().corr(pd.Series(p).rank()))


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], p: dict):
    x = train[cols].to_numpy(dtype=float)
    z = test[cols].to_numpy(dtype=float)
    mu, sd = x.mean(axis=0), x.std(axis=0)
    sd[sd < 1e-10] = 1.0
    x, z = (x - mu) / sd, (z - mu) / sd
    y = np.clip(train[p["target"]].to_numpy(dtype=float),
                -p["model"]["label_clip"], p["model"]["label_clip"])
    a = float(p["model"]["ridge_alpha"])
    weights = np.linalg.solve(x.T @ x + a * np.eye(len(cols)), x.T @ (y - y.mean()))
    return z @ weights + y.mean()


def _pair(train, test, p, feature: str):
    base = list(p["baseline_features"])
    common = base + [feature]
    tr = train.dropna(subset=common + [p["target"]])
    te = test.dropna(subset=common)
    if tr.empty or te.empty or tr["date"].nunique() < p["evaluation"]["min_train_dates"]:
        return None, None, te
    return (_fit_predict(tr, te, base, p),
            _fit_predict(tr, te, common, p), te)


def _date_metrics(rows, base, challenger, p):
    y = rows[p["target"]].to_numpy(dtype=float)
    ic0, ic1 = _rank_ic(y, base), _rank_ic(y, challenger)
    if ic0 is None or ic1 is None:
        return None
    top = max(1, int(np.ceil(len(y) * 0.10)))
    # Same trade count, same rows, same stock cost: exact paired uplift.
    # This is a STOCK diagnostic, not an executable option return.
    c = 2 * float(p["evaluation"]["stock_cost_per_side"])
    ret0 = float(y[np.argsort(base)[-top:]].mean() - c)
    ret1 = float(y[np.argsort(challenger)[-top:]].mean() - c)
    return {"n": len(y), "ic_base": ic0, "ic_challenger": ic1,
            "ic_delta": ic1 - ic0, "top_xs_base_net": ret0,
            "top_xs_challenger_net": ret1, "top_xs_delta": ret1 - ret0}


def independent(records: list[dict], p: dict) -> list[dict]:
    min_gap = timedelta(days=p["evaluation"]["min_calendar_days_between_forward_windows"])
    selected, last = [], None
    for row in sorted(records, key=lambda r: r["date"]):
        d = date.fromisoformat(str(row["date"])[:10])
        if last is None or d - last >= min_gap:
            selected.append(row)
            last = d
    return selected


def summarize(records: list[dict], p: dict, *, forward: bool) -> dict:
    selected = independent(records, p)
    n = len(selected)
    delta = [r["top_xs_delta"] for r in selected]
    ic = [r["ic_delta"] for r in selected]
    required = int(p["evaluation"]["min_independent_forward_dates"])
    return {
        "status": ("INSUFFICIENT_INDEPENDENT_DATES" if n < required else
                   "FORWARD_DIAGNOSTIC_REVIEW_REQUIRED" if forward else "RETROSPECTIVE_EXPLORATORY"),
        "dates_total": len(records), "independent_dates": n, "required_dates": required,
        "mean_paired_top_xs_delta": float(np.mean(delta)) if delta else None,
        "mean_paired_rank_ic_delta": float(np.mean(ic)) if ic else None,
        "note": "No trading approval; not a statistically corrected alpha claim",
    }


def retrospective(panel: pd.DataFrame, h: dict, p: dict, until: date) -> dict:
    feature = h["feature"]
    fields = list(p["baseline_features"]) + [feature]
    data = panel.dropna(subset=fields).copy()
    dates = sorted(set(data["date"]))
    reports = []
    for d in dates:
        day = pd.Timestamp(d)
        if day.date() >= until:
            continue
        test = data[data["date"] == day].dropna(subset=[p["target"]])
        test = test[pd.to_datetime(test[p["label_end"]], errors="coerce") < pd.Timestamp(until)]
        if len(test) < p["evaluation"]["min_cross_section"]:
            continue
        past = data[(data["date"] < day) &
                    (pd.to_datetime(data[p["label_end"]], errors="coerce") < day)]
        pred0, pred1, test = _pair(past, test, p, feature)
        if pred0 is None or len(test) < p["evaluation"]["min_cross_section"]:
            continue
        m = _date_metrics(test, pred0, pred1, p)
        if m:
            reports.append({"date": str(day.date()), **m})
    return summarize(reports, p, forward=False)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def freeze_latest(panel: pd.DataFrame, p: dict, digest: str, journal: Path, now: datetime):
    """Freeze once per (experiment, date, ticker), only BEFORE label maturation."""
    latest = pd.Timestamp(panel["date"].max())
    if not 0 <= (now.date() - latest.date()).days <= 10:
        return 0
    recorded = _read_jsonl(journal)
    keys = {(r["protocol_sha256"], r["hypothesis_id"], r["date"], r["ticker"]) for r in recorded}
    additions = []
    test = panel[panel["date"] == latest]
    for h in p["hypotheses"]:
        feature = h["feature"]
        past = panel[(panel["date"] < latest) &
                     (pd.to_datetime(panel[p["label_end"]], errors="coerce") < latest)]
        pred0, pred1, eligible = _pair(past, test, p, feature)
        if pred0 is None or len(eligible) < p["evaluation"]["min_cross_section"]:
            continue
        for row, b, c in zip(eligible.itertuples(index=False), pred0, pred1):
            ticker = str(getattr(row, "ticker"))
            key = (digest, h["id"], str(latest.date()), ticker)
            if key in keys:
                continue
            additions.append({
                "protocol_sha256": digest, "protocol_version": p["version"],
                "hypothesis_id": h["id"],
                "date": str(latest.date()), "ticker": ticker,
                "signal_asof": str(latest.date()) + "T23:59:59Z",
                "feature_value": float(getattr(row, feature)),
                "frozen_at": now.isoformat(), "baseline_pred": float(b),
                "challenger_pred": float(c), "feature": feature,
                "label": "FUTURE_UNOBSERVED", "research_only": True,
            })
    if additions:
        journal.parent.mkdir(parents=True, exist_ok=True)
        with journal.open("a", encoding="utf8") as f:
            for row in additions:
                f.write(json.dumps(row, sort_keys=True) + "\n")
    return len(additions)


def freeze_matured_outcomes(panel: pd.DataFrame, p: dict, digest: str,
                           journal: Path, out: Path, as_of: date) -> int:
    """Freeze first observed mature stock label, instead of rewriting past outcomes.

    The source is the existing yfinance-backed ML panel, NOT an option fill.
    Locked stock labels are reused by every challenger on the same cohort.
    """
    prior = _read_jsonl(out)
    seen = {(r["date"], r["ticker"]) for r in prior if r.get("protocol_sha256") == digest}
    source = panel.set_index(["date", "ticker"])
    additions = []
    for row in _read_jsonl(journal):
        if row.get("protocol_sha256") != digest:
            continue
        key = (row["date"], row["ticker"])
        if key in seen:
            continue
        idx = (pd.Timestamp(row["date"]), row["ticker"])
        if idx not in source.index:
            continue
        data = source.loc[idx]
        last = pd.to_datetime(data[p["label_end"]], errors="coerce")
        if pd.isna(last) or last.date() >= as_of:
            continue
        if datetime.fromisoformat(row["frozen_at"]).date() >= last.date():
            continue
        y = data[p["target"]]
        if pd.isna(y) or not np.isfinite(float(y)):
            continue
        additions.append({"protocol_sha256": digest, "date": key[0], "ticker": key[1],
                          "label_end": str(last.date()), "fwd_xs_20": float(y),
                          "recorded_at": as_of.isoformat(),
                          "provenance": "ML_PANEL_YFINANCE_STOCK_LABEL_NOT_OPTION_EXECUTION"})
        seen.add(key)
    if additions:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf8") as f:
            for row in additions:
                f.write(json.dumps(row, sort_keys=True) + "\n")
    return len(additions)


def prospective(panel: pd.DataFrame, h: dict, p: dict, digest: str, journal: Path, as_of: date,
                frozen_outcomes: Path | None = None):
    preds = [r for r in _read_jsonl(journal)
             if r.get("protocol_sha256") == digest and r.get("hypothesis_id") == h["id"]]
    grouped = {}
    if frozen_outcomes is not None:
        labels = {(r["date"], r["ticker"]): r for r in _read_jsonl(frozen_outcomes)
                  if r.get("protocol_sha256") == digest}
        for row in preds:
            label = labels.get((row["date"], row["ticker"]))
            if label is None or date.fromisoformat(label["label_end"]) >= as_of:
                continue
            if datetime.fromisoformat(row["frozen_at"]).date() >= date.fromisoformat(label["label_end"]):
                continue
            grouped.setdefault(row["date"], []).append(
                (float(label["fwd_xs_20"]), row["baseline_pred"], row["challenger_pred"]))
    else:
        # For deterministic unit tests only; operational run uses the locked
        # append-only outcomes journal above.
        outcomes = panel[["date", "ticker", p["target"], p["label_end"]]].copy()
        outcomes["date"] = outcomes["date"].dt.strftime("%Y-%m-%d")
        observed = outcomes.set_index(["date", "ticker"])
        for row in preds:
            key = (row["date"], row["ticker"])
            if key not in observed.index:
                continue
            outcome = observed.loc[key]
            end = pd.to_datetime(outcome[p["label_end"]], errors="coerce")
            if pd.isna(end) or end.date() >= as_of:
                continue
            if datetime.fromisoformat(row["frozen_at"]).date() >= end.date():
                continue
            y = outcome[p["target"]]
            if pd.isna(y):
                continue
            grouped.setdefault(row["date"], []).append((float(y), row["baseline_pred"], row["challenger_pred"]))
    metrics = []
    for day, rows in sorted(grouped.items()):
        if len(rows) < p["evaluation"]["min_cross_section"]:
            continue
        data = pd.DataFrame(rows, columns=[p["target"], "b", "c"])
        m = _date_metrics(data, data["b"].to_numpy(), data["c"].to_numpy(), p)
        if m:
            metrics.append({"date": day, **m})
    return summarize(metrics, p, forward=True)


def run(panel: pd.DataFrame, *, archive=None, protocol_path: Path = PROTOCOL,
        out: Path = OUT, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    p, digest = load_protocol(protocol_path)
    archive = archive or ExternalArchive()
    # A protocol is immutable once the first forecast has been frozen.
    previous = out / REPORT
    if previous.exists():
        prior = json.loads(previous.read_text(encoding="utf8"))
        if prior.get("protocol_version") == p["version"] and prior.get("protocol_sha256") != digest:
            raise ValueError("FD3 protocol changed without version bump: refusing contaminated experiment")
    for row in _read_jsonl(out / JOURNAL):
        if row.get("protocol_version") == p["version"] and row["protocol_sha256"] != digest:
            raise ValueError("FD3 immutable journal version/hash mismatch")
    store = PITResearchStore(archive)
    enhanced = store.enrich_panel(panel)
    out.mkdir(parents=True, exist_ok=True)
    report = {"protocol_version": p["version"], "protocol_sha256": digest,
              "as_of": now.date().isoformat(), "research_only": True,
              "orders_created": 0, "production_influence": "NONE",
              "features": {}, "hypotheses": {}, "next_research_priorities": []}
    for h in p["hypotheses"]:
        f = h["feature"]
        report["features"][f] = {
            "coverage": float(enhanced[f].notna().mean()),
            "dates_available": int(enhanced.loc[enhanced[f].notna(), "date"].nunique()),
        }
    report["new_frozen_predictions"] = freeze_latest(
        enhanced, p, digest, out / JOURNAL, now)
    report["new_frozen_stock_outcomes"] = freeze_matured_outcomes(
        enhanced, p, digest, out / JOURNAL, out / OUTCOMES, now.date())
    for h in p["hypotheses"]:
        f = h["feature"]
        retro = retrospective(enhanced, h, p, now.date())
        forward = prospective(enhanced, h, p, digest, out / JOURNAL, now.date(), out / OUTCOMES)
        report["hypotheses"][h["id"]] = {
            "source": h["source"], "feature": f, "premise": h["premise"],
            "retrospective": retro, "prospective": forward,
            "verdict": "NOT_CONFIRMED",
            "next_action": ("COLLECT_PIT_DATA" if report["features"][f]["dates_available"] < p["evaluation"]["min_train_dates"]
                            else "CONTINUE_PRE_REGISTERED_FORWARD"),
        }
    report["next_research_priorities"] = [
        x["id"] for x in sorted(p["hypotheses"],
            key=lambda h: (-report["features"][h["feature"]]["dates_available"], h["id"]))
    ]
    (out / REPORT).write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-cache", default=os.getenv("ML_PANEL_CACHE", ""))
    parser.add_argument("--as-of", default=None)
    args = parser.parse_args(argv)
    if not args.panel_cache or not Path(args.panel_cache).is_file():
        raise SystemExit("FD3: ML panel snapshot missing; refusing independent market reload")
    now = (datetime.combine(date.fromisoformat(args.as_of), datetime.min.time(),
                            tzinfo=timezone.utc) if args.as_of else datetime.now(timezone.utc))
    report = run(pd.read_pickle(args.panel_cache), now=now)
    print(json.dumps({k: report[k] for k in (
        "protocol_version", "as_of", "features", "new_frozen_predictions", "hypotheses"
    )}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

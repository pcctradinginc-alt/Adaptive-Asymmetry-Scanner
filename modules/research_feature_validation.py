"""FD3 Phase 2: independently auditable, repeated-look-safe paired shadow validation.

Reads ONLY immutable FD3 prediction/stock-outcome journals. No quote/broker,
market API, retraining, LLM, candidate ranking, model tuning or production.
All tests are *diagnostic*: no global alpha ledger slot is claimed, therefore
no confirmed research/option alpha is ever emitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from modules import research_feature_discovery as fd

ROOT = Path("outputs/research/feature_discovery_v3")
LOOKS = (8, 16, 24)
N_PERMUTATIONS = 4096
N_BOOTSTRAP = 2048
SEED = 20261108
SCHEMA = "fd3-paired-shadow-validation-v1"


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     default=str).encode()).hexdigest()


POLICY = {"schema": SCHEMA, "looks": LOOKS, "n_permutations": N_PERMUTATIONS,
          "n_bootstrap": N_BOOTSTRAP, "seed": SEED,
          "placebo": "within_date_label_permutation_two_sided",
          "independence": "non_overlapping_true_label_end",
          "economic_unit": "paired_top_decile_stock_net_not_options"}
POLICY_HASH = _hash(POLICY)

def read_look_journal(path: Path) -> tuple[list[dict], str]:
    """One hash chain, append-only. A changed historical look fails closed."""
    last, out = "GENESIS", []
    for i, event in enumerate(fd._read_jsonl(path)):
        if event.get("prev_hash") != last or event.get("record_hash") != _hash(event.get("record")):
            raise ValueError(f"FD3 phase-2 journal integrity error at row {i + 1}")
        if event["record"].get("validation_policy_sha256") != POLICY_HASH:
            raise ValueError("FD3 statistical policy changed under an existing look")
        last = _hash(event)
        out.append(event["record"])
    keys = [(r["protocol_hash"], r["hypothesis_id"], r["look_n"]) for r in out]
    if len(set(keys)) != len(keys):
        raise ValueError("FD3 phase-2 journal has repeated look IDs")
    return out, last


def _append_look(path: Path, record: dict, prev_hash: str) -> str:
    event = {"prev_hash": prev_hash, "record": record, "record_hash": _hash(record)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps(event, sort_keys=True) + "\n")
        out.flush()
        os.fsync(out.fileno())
    return _hash(event)


def _ts(s) -> datetime:
    value = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("Naive timestamp in FD3 immutable evidence")
    return value.astimezone(timezone.utc)


def source_rows(pred_path: Path, outcome_path: Path, proto_hash: str,
                today: date) -> tuple[dict, dict]:
    """Pair ONLY previously frozen predictions with matured, frozen stock labels.

    Duplicate/colliding records are integrity failures rather than voting or
    last-writer-wins. Independent window selection happens after pairing.
    """
    pred, outcome = {}, {}
    for r in fd._read_jsonl(pred_path):
        if r.get("protocol_sha256") != proto_hash:
            continue
        k = (r["hypothesis_id"], r["date"], r["ticker"])
        if k in pred:
            raise ValueError(f"Duplicate immutable prediction {k}")
        if not r.get("research_only") or r.get("label") != "FUTURE_UNOBSERVED":
            raise ValueError(f"FD3 prediction schema violation {k}")
        pred[k] = r
    for r in fd._read_jsonl(outcome_path):
        if r.get("protocol_sha256") != proto_hash:
            continue
        k = (r["date"], r["ticker"])
        if k in outcome:
            raise ValueError(f"Duplicate immutable stock outcome {k}")
        outcome[k] = r
    cohorts = defaultdict(lambda: defaultdict(list))
    missing = defaultdict(int)
    for (hid, day, ticker), p in sorted(pred.items()):
        o = outcome.get((day, ticker))
        if o is None:
            missing[hid] += 1
            continue
        end = date.fromisoformat(o["label_end"])
        if end >= today or _ts(p["frozen_at"]).date() >= end:
            missing[hid] += 1
            continue
        if date.fromisoformat(str(o.get("recorded_at", "9999-12-31"))[:10]) > today:
            missing[hid] += 1
            continue
        y = float(o["fwd_xs_20"])
        b, c = float(p["baseline_pred"]), float(p["challenger_pred"])
        if not all(np.isfinite(x) for x in (y, b, c)):
            raise ValueError("Non-finite frozen forecast or outcome")
        cohorts[hid][day].append((ticker, y, b, c, end.isoformat()))
    return cohorts, {"predictions": len(pred), "frozen_outcomes": len(outcome),
                     "unmatured_or_missing": dict(missing)}


def independent_days(dated: dict[str, list], p: dict) -> list[dict]:
    """Use one cohort per non-overlapping actual stock-label horizon."""
    rows = []
    for day, cohort in sorted(dated.items()):
        if len(cohort) < p["evaluation"]["min_cross_section"]:
            continue
        df = pd.DataFrame(cohort, columns=["ticker", "fwd_xs_20", "b", "c", "window_end"])
        metric = fd._date_metrics(df, df["b"].to_numpy(), df["c"].to_numpy(), p)
        if metric is None:
            continue
        rows.append({"date": day, "window_end": max(df["window_end"]),
                     "metrics": metric, "df": df})
    return fd.independent(rows, p)


def _null_placebo(blocks: list[dict], n: int, rng: np.random.Generator) -> dict:
    """Within-day label shuffling: tests selection without any stock-feature edge."""
    realized, null = [], []
    for block in blocks:
        realized.append(float(block["metrics"]["top_xs_delta"]))
    observed = float(np.mean(realized))
    for _ in range(n):
        deltas = []
        for block in blocks:
            d = block["df"]
            y = rng.permutation(d["fwd_xs_20"].to_numpy(dtype=float))
            b, c = d["b"].to_numpy(dtype=float), d["c"].to_numpy(dtype=float)
            k = max(1, int(np.ceil(len(y) * .10)))
            # Paired fixed-size portfolios; costs cancel because both pay same.
            deltas.append(float(y[np.argsort(c)[-k:]].mean()
                                - y[np.argsort(b)[-k:]].mean()))
        null.append(float(np.mean(deltas)))
    # Diagnostic only, two-sided. No alpha slot in the global ledger.
    p_raw = (1 + sum(abs(x) >= abs(observed) for x in null)) / (len(null) + 1)
    return {"observed_delta": observed,
            "placebo_p_two_sided": float(p_raw),
            "placebo_mean": float(np.mean(null)),
            "placebo_p95_abs": float(np.percentile(np.abs(null), 95))}


def analyze_look(blocks: list[dict], hypothesis: str, look: int,
                 protocol_hash: str) -> dict:
    seed = int(hashlib.sha256(f"{SEED}|{hypothesis}|{look}|{protocol_hash}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    window = blocks[:look]
    d = np.array([b["metrics"]["top_xs_delta"] for b in window], dtype=float)
    ic = np.array([b["metrics"]["ic_delta"] for b in window], dtype=float)
    res = _null_placebo(window, N_PERMUTATIONS, rng)
    boots = rng.choice(d, size=(N_BOOTSTRAP, len(d)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # Local diagnostic Bonferroni over four fixed hypotheses × three looks.
    # This does NOT substitute the repository's global alpha ledger.
    local_adjust = len(fd.load_protocol()[0]["hypotheses"]) * len(LOOKS)
    return {"schema": SCHEMA, "validation_policy_sha256": POLICY_HASH,
            "protocol_hash": protocol_hash,
            "hypothesis_id": hypothesis, "look_n": look,
            "window_dates": [b["date"] for b in window],
            "n_distinct_tickers_min": min(len(b["df"]) for b in window),
            "last_window_end": window[-1]["window_end"],
            "paired_stock_mean": float(d.mean()),
            "paired_stock_95pct_bootstrap": [float(lo), float(hi)],
            "paired_stock_positive_window_fraction": float(np.mean(d > 0)),
            "paired_rank_ic_mean": float(ic.mean()),
            "placebo": res,
            "local_bonferroni_diagnostic_p": min(1.0, res["placebo_p_two_sided"] * local_adjust),
            "statistics_are_exploratory": True,
            "global_alpha_ledger_slot_allocated": False,
            "option_economics_verified": False,
            "decision": "NO_AUTOMATIC_PROMOTION"}


def run(*, root: Path = ROOT, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    p, digest = fd.load_protocol()
    status_file = root / fd.REPORT
    if not status_file.exists():
        return {"status": "MISSING_FD3_SOURCE", "new_looks": 0, "hypotheses": {}}
    status = json.loads(status_file.read_text(encoding="utf-8"))
    if status.get("protocol_sha256") != digest:
        raise ValueError("FD3 source/protocol mismatch; refuse cross-version attribution")
    cohorts, coverage = source_rows(root / fd.JOURNAL, root / fd.OUTCOMES, digest, now.date())
    existing, last = read_look_journal(root / "paired_looks.jsonl")
    keys = {(r["protocol_hash"], r["hypothesis_id"], r["look_n"]) for r in existing}
    out = {"schema": SCHEMA, "as_of": now.isoformat(), "protocol_sha256": digest,
           "status": "DIAGNOSTIC_SHADOW_ONLY", "validation_policy_sha256": POLICY_HASH,
           "global_alpha_ledger_slot_allocated": False,
           "option_economics_verified": False, "coverage": coverage,
           "hypotheses": {}, "new_looks": 0}
    for h in p["hypotheses"]:
        hid = h["id"]
        windows = independent_days(cohorts.get(hid, {}), p)
        evaluated = []
        for n in LOOKS:
            key = (digest, hid, n)
            if len(windows) < n:
                continue
            stored = next((r for r in existing if (r["protocol_hash"], r["hypothesis_id"], r["look_n"]) == key), None)
            if stored is not None:
                evaluated.append(stored)
                continue
            look = analyze_look(windows, hid, n, digest)
            last = _append_look(root / "paired_looks.jsonl", look, last)
            evaluated.append(look)
            keys.add(key)
            out["new_looks"] += 1
        out["hypotheses"][hid] = {
            "feature": h["feature"], "source": h["source"],
            "independent_forward_windows": len(windows),
            "candidate_rows": sum(len(v) for v in cohorts.get(hid, {}).values()),
            "status": "PENDING_INDEPENDENT_WINDOWS" if len(windows) < LOOKS[0]
                      else "DIAGNOSTIC_REVIEW_ONLY",
            "looks": evaluated, "confirmed_alpha": False,
            "next_action": "MORE_FORWARD_DATA" if len(windows) < LOOKS[0]
                          else "REVIEW_STATISTICS_AND_OPTION_ECONOMICS",
        }
    (root / "paired_validation.json").write_text(
        json.dumps(out, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def main(argv=None) -> int:
    q = argparse.ArgumentParser()
    q.add_argument("--as-of", default=None)
    args = q.parse_args(argv)
    now = (datetime.fromisoformat(args.as_of).replace(tzinfo=timezone.utc)
           if args.as_of else datetime.now(timezone.utc))
    output = run(now=now)
    print(json.dumps({"status": output["status"], "new_looks": output["new_looks"],
                      "hypotheses": {k: v["status"] for k, v in output["hypotheses"].items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

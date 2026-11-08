"""FD3 prospective signal adapter for the EXISTING quote-verified paper link.

Executed by the weekly option-link register step only; no network, market
capture or order logic here. The normal option-link capture budget and its
pre-registered fixed contract rules are retained unchanged.

Comparators: 1 top-ranked BASE and CHALLENGER stock per selected hypothesis.
Two out of four fixed hypotheses rotate deterministically by calendar week;
NO source selection uses subsequent returns or model performance.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import yaml

FD3_CONFIG = Path("config/research_feature_discovery_v3.yaml")
FD3_PREDICTIONS = Path("outputs/research/feature_discovery_v3/predictions.jsonl")


def _ts(s):
    try:
        d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        return d.astimezone(timezone.utc) if d.tzinfo else None
    except (ValueError, TypeError):
        return None


def _read_jsonl(path: Path):
    if not path.exists():
        return []
    out = []
    for i, s in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if s.strip():
            try:
                row = json.loads(s)
            except ValueError as exc:
                raise ValueError(f"FD3 malformed prediction line {i}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"FD3 non-object prediction line {i}")
            out.append(row)
    return out


def _digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     default=str).encode()).hexdigest()


def build_batches(root: Path, opts: dict, link_start: date, max_age: timedelta,
                  now: datetime, stats: dict) -> list[dict]:
    root = Path(root)
    start = _ts(opts.get("registration_start_utc"))
    if start is None or start > now:
        stats["fd3_not_registered_yet"] += 1
        return []
    config = root / FD3_CONFIG
    if not config.exists():
        stats["fd3_protocol_missing"] += 1
        return []
    raw = config.read_bytes()
    protocol, protocol_hash = yaml.safe_load(raw), hashlib.sha256(raw).hexdigest()
    if not isinstance(protocol, dict) or protocol.get("research_only") is not True:
        raise ValueError("FD3 source protocol must be research-only")
    if protocol.get("version") != "fd3-2026-11-08":
        raise ValueError("FD3 source version changed without option-link review")
    feature_status = root / "outputs/research/feature_discovery_v3/status.json"
    if not feature_status.exists():
        stats["fd3_source_status_missing"] += 1
        return []
    published = json.loads(feature_status.read_text(encoding="utf-8"))
    if published.get("protocol_sha256") != protocol_hash:
        raise ValueError("FD3 source protocol mismatches sealed feature status")
    ids = sorted(h["id"] for h in protocol["hypotheses"])
    if len(ids) != 4 or len(set(ids)) != len(ids):
        raise ValueError("FD3 source hypothesis family changed")
    groups = defaultdict(list)
    seen = set()
    for r in _read_jsonl(root / FD3_PREDICTIONS):
        if r.get("protocol_sha256") != protocol_hash:
            continue
        if not r.get("research_only") or r.get("label") != "FUTURE_UNOBSERVED":
            raise ValueError("FD3 non-forward or mutable row")
        day, hid, ticker = r.get("date"), r.get("hypothesis_id"), r.get("ticker")
        if not day or hid not in ids or not ticker:
            continue
        created = _ts(r.get("frozen_at"))
        if created is None or created <= start or created.date() < link_start:
            stats["fd3_before_registration"] += 1
            continue
        if created > now or now - created > max_age:
            stats["fd3_stale_or_future"] += 1
            continue
        pair = (hid, day, ticker)
        if pair in seen:
            raise ValueError("FD3 duplicated frozen prediction")
        seen.add(pair)
        base, challenger = r.get("baseline_pred"), r.get("challenger_pred")
        try:
            b, c = float(base), float(challenger)
        except (TypeError, ValueError) as e:
            raise ValueError("FD3 missing frozen scores") from e
        if not math.isfinite(b) or not math.isfinite(c):
            raise ValueError("FD3 non-finite prediction")
        groups[(day, hid)].append((str(ticker), b, c, created))
    batches = []
    limit = int(opts["max_hypotheses_per_signal_date"])
    for day in sorted({day for day, _ in groups}):
        day_date = date.fromisoformat(day)
        rotating = (day_date.toordinal() // 7) % len(ids)
        scheduled = {ids[(rotating + i) % len(ids)] for i in range(limit)}
        for hid in sorted(scheduled):
            rows = groups.get((day, hid)) or []
            if len(rows) < int(opts["min_issuers"]) or len({x[0] for x in rows}) != len(rows):
                stats["fd3_missing_full_cohort"] += 1
                continue  # NEVER replace a missing preregistered family.
            frozen = {x[3] for x in rows}
            if len(frozen) != 1:
                raise ValueError("FD3 changing freeze timestamps across matched cohort")
            registered = max(frozen)
            # Within-date cross-sectional rank, ticker tie break stable.
            universe = sorted(x[0] for x in rows)
            sealed_ref = _digest([{"ticker": t, "base": b, "challenger": c}
                                  for t, b, c, _ in sorted(rows)])
            for arm, idx, comparator in (("BASE", 1, "REFERENCE"),
                                         ("CHALLENGER", 2, "TREATMENT")):
                ordered = sorted(rows, key=lambda x: (-x[idx], x[0]))
                best = ordered[:int(opts["top_k"])]
                ref = {"journal": str(FD3_PREDICTIONS), "hash": sealed_ref,
                       "created_at": registered.isoformat(), "prediction_date": day,
                       "n_universe": len(universe),
                       "fd3_arm": arm, "fd3_hypothesis": hid}
                batches.append({
                    "hypothesis_id": f"FD3:{hid}:{arm}",
                    "hypothesis_source": "fd3_feature", "comparison_group": comparator,
                    "hypothesis_version": protocol_hash, "hypothesis_hash": protocol_hash,
                    "hypothesis_registered_at": start.isoformat(),
                    "model_id": f"FD3:{hid}:{arm}",
                    "experiment_version": f"fd3:{protocol_hash}:{hid}:{arm}",
                    "prediction_ref": ref, "signal_time": registered.isoformat(),
                    "pit_data_date": day,
                    "top": [(x[0], float(x[idx])) for x in best],
                    "bottom": [], "universe": universe, "expected": None,
                })
            stats["fd3_paired_batches"] += 1
    return batches

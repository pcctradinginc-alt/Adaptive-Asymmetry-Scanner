"""Alpha-to-Options link (Phase A): hypothesis -> option decision -> outcome.

RESEARCH / PAPER ONLY. No orders, no LLM, no production/champion/gate/sizing or
risk-limit influence. Reuses the existing stores; no second orchestrator:

* signals   come from the existing prospective journals (ML registry models in
            outputs/research/ml_predictions, AUTO-* factor predictions) and are
            frozen by the weekly ML research job (``register``);
* decisions are made by the existing RTH options-capture job (``decide``) with
            a pre-fixed contract rule (config/alpha_option_link.yaml); every
            observed quote goes to the existing quote archive
            (modules/option_quotes) and is referenced by its record_id;
* outcomes  are settled from archived RTH quotes only (``settle``).

Integrity: append-only, content-hashed journals with deterministic logical IDs,
idempotent runs, strict point-in-time order (prediction -> registration ->
decision -> exit), explicit unknowns, versioned corrections that never delete
or re-label, and documented untradable/rejected signals. Missing data is never
treated as a result. Predictions that pre-date the link are NOT linked
retroactively; they are reported as PRE_LINK.

Capture-path functions (``decide``, ``due_capture_targets``) must stay
importable with requests + pyyaml only (no pandas at module import).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from datetime import time as dt_time
from pathlib import Path

import yaml

from .registry import EvidenceIntegrityError, _atomic_json, canonical, digest

CONFIG = Path("config/alpha_option_link.yaml")
ROOT = Path("outputs/research/autonomous_alpha/option_link")
# Decisions are written by the options-capture job, which may only commit the
# quote archive (tests/test_ops_readiness*): they live next to the quotes they
# reference. Everything else is written by the weekly research job.
DECISIONS_ROOT = Path("outputs/options_archive/alpha_link")
KINDS = ("signals", "decisions", "outcomes", "corrections")
REPORT = Path("outputs/research/autonomous_alpha/option_link.json")
ML_PRED_DIR = Path("outputs/research/ml_predictions")
ML_REGISTRY = Path("config/model_registry.yaml")
ML_REGISTRY_LOG = Path("outputs/research/ml_registry_log.json")
FACTOR_PRED_DIR = Path("outputs/research/autonomous_alpha/factor_predictions")
TRADE_OUTCOMES = Path("outputs/options_archive/trade_outcomes.jsonl")

STRUCTURES = ("LONG_CALL", "BULL_CALL_SPREAD")
CONTROL_ID = "CONTROL-UNIVERSE-RANDOM"
EVIDENCE_CLASSES = ("EXECUTED", "QUOTE_VERIFIED_PAPER", "UNVERIFIABLE")
CORRECTION_ACTIONS = ("EXCLUDE_DATA_ERROR", "DOWNGRADE_EVIDENCE")
MAX_ROWS = 50000
MAX_SIGNALS_HARD = 40
MAX_UNDERLYINGS_HARD = 25
MAX_CAPTURE_HARD = 60
LEG_PAIR_MAX_SECONDS = 900
SCHEMAS = {
    "signals": "alpha-link-signal-v1",
    "decisions": "alpha-link-decision-v1",
    "outcomes": "alpha-link-outcome-v1",
    "corrections": "alpha-link-correction-v1",
}
_ID_FIELD = {"signals": "signal_id", "decisions": "decision_id",
             "outcomes": "outcome_id", "corrections": "correction_id"}
_TIME_FIELD = {"signals": "registered_at", "decisions": "decided_at",
               "outcomes": "recorded_at", "corrections": "recorded_at"}


# ── configuration ────────────────────────────────────────────────────────────

def config(root: Path = Path(".")) -> dict:
    """Load and validate the frozen link policy (hard caps cannot be relaxed)."""
    p = Path(root) / CONFIG
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or cfg.get("version") != "alpha-option-link-v1":
        raise EvidenceIntegrityError("alpha option link policy invalid")
    if cfg.get("production_influence") != "NONE" or cfg.get("allow_llm") is not False:
        raise EvidenceIntegrityError("alpha option link must have no production/LLM access")
    date.fromisoformat(str(cfg["link_start"]))
    if not 1 <= int(cfg.get("max_signals_per_date", 0)) <= MAX_SIGNALS_HARD:
        raise EvidenceIntegrityError("signal budget exceeds hard cap")
    if not 1 <= int(cfg.get("max_underlyings_per_decision_run", 0)) <= MAX_UNDERLYINGS_HARD:
        raise EvidenceIntegrityError("decision budget exceeds hard cap")
    if not 1 <= int(cfg.get("max_capture_symbols_per_run", 0)) <= MAX_CAPTURE_HARD:
        raise EvidenceIntegrityError("capture budget exceeds hard cap")
    if not 1 <= int(cfg.get("registration_max_age_days", 0)) <= 7:
        raise EvidenceIntegrityError("registration freshness cannot be relaxed")
    if not 1 <= int(cfg.get("decision_max_lag_trading_days", 0)) <= 5:
        raise EvidenceIntegrityError("decision freshness cannot be relaxed")
    if not 30 <= int(cfg.get("decision_time_budget_seconds", 0)) <= 400:
        raise EvidenceIntegrityError("decision time budget invalid")
    ctrl = cfg.get("control") or {}
    if not 0 <= int(ctrl.get("per_signal_date", -1)) <= 10 or not ctrl.get("seed"):
        raise EvidenceIntegrityError("control group definition invalid")
    for name in ("ml_registry", "autonomous_factor"):
        src = (cfg.get("sources") or {}).get(name) or {}
        if not 0 <= int(src.get("top_k", -1)) <= 5 or not 0 <= int(src.get("bottom_k", -1)) <= 10:
            raise EvidenceIntegrityError("source tail sizes invalid: " + name)
    rule = cfg.get("rule") or {}
    if tuple(rule.get("structures") or ()) != STRUCTURES or rule.get("direction") != "BULLISH":
        raise EvidenceIntegrityError("only LONG_CALL and BULL_CALL_SPREAD are allowed")
    if not rule.get("rule_version"):
        raise EvidenceIntegrityError("rule version missing")
    if int(rule.get("horizon_trading_days", 0)) != 20:
        raise EvidenceIntegrityError("horizon must match the 20-day hypotheses")
    if not 1 <= int(rule.get("exit_max_lag_trading_days", 0)) <= 8:
        raise EvidenceIntegrityError("exit freshness cannot be relaxed")
    if not 30 <= int(rule.get("dte_min", 0)) < int(rule.get("dte_max", 0)) <= 180:
        raise EvidenceIntegrityError("DTE window invalid (must outlive the horizon)")
    if rule.get("long_strike") != "ATM_NEAREST_TIE_LOWER":
        raise EvidenceIntegrityError("unknown long strike rule")
    lo, hi = (float(x) for x in rule.get("short_strike_window") or (0, 0))
    if not 1.0 < lo < float(rule.get("short_strike_target", 0)) < hi <= 1.5:
        raise EvidenceIntegrityError("spread short-leg window invalid")
    if int(rule.get("contracts_per_position", 0)) != 1:
        raise EvidenceIntegrityError("research position size is fixed at one contract")
    checks = (("max_leg_spread_pct", 0.01, 1.0), ("min_open_interest", 0, 100000),
              ("min_bid", 0.0, 5.0), ("max_iv", 0.05, 5.0), ("max_capital_usd", 100, 50000),
              ("commission_usd_per_contract_per_side", 0.0, 5.0),
              ("adverse_slippage_fraction_of_spread", 0.0, 1.0))
    for key, a, b in checks:
        v = float(rule.get(key, -1))
        if not a <= v <= b:
            raise EvidenceIntegrityError("rule parameter out of range: " + key)
    if rule.get("edge_gate") != "NONE":
        raise EvidenceIntegrityError("v1 records the edge but never gates on it")
    cov = cfg.get("coverage") or {}
    if int(cov.get("min_verified_outcomes", 0)) < 10 or int(cov.get("min_distinct_signal_dates", 0)) < 4:
        raise EvidenceIntegrityError("coverage floor cannot be relaxed")
    return cfg


def rule_hash(cfg: dict) -> str:
    return digest(cfg["rule"])


# ── append-only journals ─────────────────────────────────────────────────────

def _dir(root: Path, kind: str) -> Path:
    return Path(root) / (DECISIONS_ROOT if kind == "decisions" else ROOT) / kind


def configured(root: Path = Path(".")) -> bool:
    return (Path(root) / CONFIG).is_file()


def _logical_key(kind: str, rec: dict) -> str:
    return str(rec[_ID_FIELD[kind]])


def _read_kind(root: Path, kind: str) -> tuple[list[dict], dict]:
    """Validated records of one journal plus integrity facts.

    A changed line (hash mismatch) or a foreign schema fails closed. Identical
    duplicates (git union merge) are tolerated. Two different records under
    the same logical ID are never resolved by outcome: the EARLIEST by its
    own timestamp wins and the conflict is reported.
    """
    directory = _dir(root, kind)
    by_id: dict[str, dict] = {}
    conflicts = 0
    n = 0
    if directory.exists():
        for path in sorted(directory.glob("????-??.jsonl")):
            with path.open(encoding="utf-8") as fh:
                for lineno, line in enumerate(fh, 1):
                    if not line.strip():
                        continue
                    n += 1
                    if n > MAX_ROWS:
                        raise EvidenceIntegrityError(f"{kind} journal over record budget")
                    try:
                        event = json.loads(line)
                        rec = event["payload"]
                        if not isinstance(rec, dict) or event.get("sha256") != digest(rec):
                            raise ValueError("content hash mismatch")
                        if rec.get("schema") != SCHEMAS[kind]:
                            raise ValueError("unexpected record schema")
                        key = _logical_key(kind, rec)
                    except (TypeError, ValueError, KeyError) as exc:
                        raise EvidenceIntegrityError(f"broken {kind} journal {path}:{lineno}: {exc}") from exc
                    prev = by_id.get(key)
                    if prev is None:
                        by_id[key] = rec
                    elif digest(prev) != digest(rec):
                        conflicts += 1
                        if str(rec[_TIME_FIELD[kind]]) < str(prev[_TIME_FIELD[kind]]):
                            by_id[key] = rec
    rows = sorted(by_id.values(), key=lambda r: (str(r[_TIME_FIELD[kind]]), _logical_key(kind, r)))
    return rows, {"rows": n, "unique": len(rows), "conflicting_duplicates": conflicts}


def _append(root: Path, kind: str, records: list[dict], existing: list[dict]) -> int:
    known = {_logical_key(kind, r) for r in existing}
    written = 0
    for rec in records:
        key = _logical_key(kind, rec)
        if key in known:
            continue
        if rec.get("schema") != SCHEMAS[kind]:
            raise EvidenceIntegrityError("refusing to append foreign schema")
        month = str(rec[_TIME_FIELD[kind]])[:7]
        directory = _dir(root, kind)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / f"{month}.jsonl").open("a", encoding="utf-8") as out:
            out.write(canonical({"sha256": digest(rec), "payload": rec}) + "\n")
            out.flush()
            os.fsync(out.fileno())
        known.add(key)
        written += 1
    return written


def read_all(root: Path = Path(".")) -> dict:
    """All journals with corrections applied as a VIEW (originals unchanged)."""
    root = Path(root)
    data, integrity = {}, {}
    for kind in KINDS:
        data[kind], integrity[kind] = _read_kind(root, kind)
    effective = apply_corrections(data, data["corrections"])
    effective["integrity"] = integrity
    return effective


# ── time helpers ─────────────────────────────────────────────────────────────

def _ts(x) -> datetime | None:
    if not x:
        return None
    try:
        t = datetime.fromisoformat(str(x).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else None


def _iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).isoformat(timespec="seconds")


def _ny_day(t: datetime) -> date:
    from modules import market_snapshot as ms
    return t.astimezone(ms.NY_TZ).date()


def _ny_midnight_utc(d: date) -> datetime:
    from modules import market_snapshot as ms
    return datetime.combine(d, dt_time(0, 0), tzinfo=ms.NY_TZ).astimezone(timezone.utc)


def add_trading_days(d: date, n: int) -> date:
    """n-th NYSE trading day after d (n >= 1); n == 0 returns d."""
    from modules import market_snapshot as ms
    cur = d
    for _ in range(int(n)):
        cur = ms.next_trading_day(cur)
    return cur


def _num(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# ── signal sources (existing journals only) ──────────────────────────────────

def ml_spec_hash(spec: dict) -> str:
    """Mirror of modules.ml_research.spec_hash (equality is tested)."""
    core = {k: spec.get(k) for k in ("model", "target", "features", "params_grid", "rule_feature")}
    return hashlib.sha256(json.dumps(core, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _ml_registry(root: Path) -> dict:
    p = Path(root) / ML_REGISTRY
    if not p.exists():
        return {}
    reg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return {m["id"]: m for m in reg.get("models") or [] if isinstance(m, dict) and m.get("id")}


def _ml_expected(root: Path) -> dict:
    """Frozen ex-ante expectation per model: locked-holdout mean top-vs-universe
    20d return (net of base stock costs). Never a forward outcome."""
    p = Path(root) / ML_REGISTRY_LOG
    if not p.exists():
        return {}
    try:
        book = json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        return {}
    out = {}
    for mid, entry in (book or {}).items():
        lr = (entry or {}).get("locked_result") or {}
        mean = _num((lr.get("base") or {}).get("mean"))
        if lr.get("status") == "ok" and mean is not None:
            out[mid] = {"value": mean, "evaluated_at": lr.get("evaluated_at"),
                        "source": "ml_registry_log.locked_result.base.mean"}
    return out


def _ml_predictions(root: Path) -> tuple[list[dict], int]:
    rows, broken = [], 0
    d = Path(root) / ML_PRED_DIR
    for p in sorted(d.glob("*.jsonl")) if d.exists() else []:
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                broken += 1
                continue
            if isinstance(r, dict) and r.get("model_id") and isinstance(r.get("rank_pct"), dict):
                rows.append(r)
            else:
                broken += 1
    return rows, broken


def ml_prediction_hash(row: dict) -> str:
    """Hash of the immutable part of an ML prediction (``realized`` is filled
    in later by ml_research and is excluded on purpose)."""
    return digest({k: v for k, v in row.items() if k != "realized"})


def _tails(ranked: list[tuple[str, float]], top_k: int, bottom_k: int):
    """ranked: (ticker, score) -> (top names best-first, bottom names worst-first)."""
    asc = sorted(ranked, key=lambda x: (x[1], x[0]))
    desc = sorted(ranked, key=lambda x: (-x[1], x[0]))
    top = desc[:top_k]
    top_names = {t for t, _ in top}
    bottom = [x for x in asc if x[0] not in top_names][:bottom_k]
    return top, bottom


def _sources(root: Path, cfg: dict, now: datetime) -> tuple[list[dict], dict]:
    """Normalized hypothesis predictions from the existing journals."""
    stats = Counter()
    batches = []
    max_age = timedelta(days=int(cfg["registration_max_age_days"]))
    link_start = date.fromisoformat(str(cfg["link_start"]))
    src = cfg["sources"]
    if src["ml_registry"].get("enabled"):
        reg = _ml_registry(root)
        expected = _ml_expected(root)
        rows, broken = _ml_predictions(root)
        stats["ml_broken_lines"] += broken
        for r in rows:
            created = _ts(r.get("created_at"))
            spec = reg.get(r["model_id"])
            if created is None:
                stats["ml_no_created_at"] += 1
                continue
            if created.date() < link_start:
                stats["ml_pre_link_rows"] += 1
                continue
            if created > now:
                stats["ml_future_rows"] += 1
                continue
            if now - created > max_age:
                stats["ml_late_not_registered"] += 1
                continue
            if spec is None:
                stats["ml_unregistered_model"] += 1
                continue
            reg_at = _ts(str(spec.get("registered_at")))
            if reg_at is None or created <= reg_at:
                stats["ml_pre_registration"] += 1
                continue
            if r.get("spec_hash") != ml_spec_hash(spec):
                stats["ml_spec_hash_mismatch"] += 1
                continue
            ranked = [(str(t), float(v)) for t, v in r["rank_pct"].items() if _num(v) is not None]
            top, bottom = _tails(ranked, int(src["ml_registry"]["top_k"]), int(src["ml_registry"]["bottom_k"]))
            exp = expected.get(r["model_id"])
            batches.append({
                "hypothesis_id": r["model_id"], "hypothesis_source": "ml_registry",
                "comparison_group": "REFERENCE" if spec.get("role") == "benchmark" else "TREATMENT",
                "hypothesis_version": r["spec_hash"], "hypothesis_hash": digest(spec),
                "hypothesis_registered_at": str(spec.get("registered_at")),
                "model_id": r["model_id"],
                "experiment_version": f"ml_research:{r['spec_hash']}:{r.get('train_cutoff')}:{r.get('code_sha')}",
                "prediction_ref": {"journal": str(ML_PRED_DIR), "hash": ml_prediction_hash(r),
                                   "created_at": r.get("created_at"), "prediction_date": r["prediction_date"],
                                   "train_cutoff": r.get("train_cutoff"), "features_hash": r.get("features_hash"),
                                   "code_sha": r.get("code_sha"), "n_universe": r.get("n")},
                "signal_time": _iso(created), "pit_data_date": r["prediction_date"],
                "top": top, "bottom": bottom, "universe": sorted(t for t, _ in ranked),
                "expected": exp,
            })
    if src["autonomous_factor"].get("enabled"):
        directory = Path(root) / FACTOR_PRED_DIR
        if directory.exists() and any(directory.glob("*.jsonl")):
            from .factor_forward import (
                _read as read_factor,  # needs pandas: weekly job only
            )
            for e in read_factor(directory):
                p = e["payload"]
                if p.get("schema") != "alpha-factor-pred-v1":
                    continue
                created = _ts(p.get("prediction_recorded_at"))
                if created is None or created.date() < link_start:
                    stats["auto_pre_link_rows"] += 1
                    continue
                if created > now or now - created > max_age:
                    stats["auto_late_or_future"] += 1
                    continue
                reg_at = _ts(p.get("registered_at"))
                if reg_at is None or created <= reg_at:
                    stats["auto_pre_registration"] += 1
                    continue
                k_top, k_bot = int(src["autonomous_factor"]["top_k"]), int(src["autonomous_factor"]["bottom_k"])
                # factor_forward stores tails sorted by ascending score
                n_top = len(p["top"])
                top = [(t, None) for t in reversed(p["top"][max(0, n_top - k_top):])]
                bottom = [(t, None) for t in p["bottom"][:k_bot]]
                batches.append({
                    "hypothesis_id": p["hypothesis_id"], "hypothesis_source": "autonomous_factor",
                    "comparison_group": "TREATMENT",
                    "hypothesis_version": p["spec_hash"], "hypothesis_hash": p["spec_hash"],
                    "hypothesis_registered_at": p["registered_at"], "model_id": p["hypothesis_id"],
                    "experiment_version": f"factor_forward:{p['spec_hash'][:16]}",
                    "prediction_ref": {"journal": str(FACTOR_PRED_DIR), "hash": e["sha256"],
                                       "created_at": p["prediction_recorded_at"],
                                       "prediction_date": p["prediction_date"],
                                       "n_universe": p.get("n_universe")},
                    "signal_time": _iso(created), "pit_data_date": p["prediction_date"],
                    "top": top, "bottom": bottom, "universe": [], "expected": None,
                })
    return batches, dict(stats)


def _panel_context(panel, day: str, tickers: set[str]) -> tuple[dict, dict]:
    """PIT features/regime ONLY from rows of the prediction date itself."""
    if panel is None:
        return {}, {"status": "UNAVAILABLE"}
    import pandas as pd

    from modules import research_lab as lab
    p = panel.loc[pd.to_datetime(panel["date"]) == pd.Timestamp(day)]
    if p.empty:
        return {}, {"status": "UNAVAILABLE"}
    cols = sorted(c for c in lab.ALLOWED_NAMES if c in p.columns and not c.startswith(("fwd_", "label_")))
    feats = {}
    for _, row in p.loc[p["ticker"].astype(str).isin(tickers)].iterrows():
        vals = {}
        for c in cols:
            v = _num(row[c])
            vals[c] = round(v, 6) if v is not None else None
        feats[str(row["ticker"])] = vals
    first = p.iloc[0]
    vix, trend = _num(first.get("vix")), _num(first.get("spy_trend_200"))
    label = "UNKNOWN"
    if vix is not None and trend is not None:
        label = ("vix_lt_20" if vix < 20 else "vix_ge_20") + "|" + ("spy_uptrend" if trend > 0 else "spy_downtrend")
    return feats, {"status": "OK" if label != "UNKNOWN" else "PARTIAL", "vix": vix,
                   "spy_trend_200": trend, "label": label}


def _control_sample(universe: list[str], exclude: set[str], day: str, n: int, seed: str) -> list[str]:
    pool = sorted(set(universe) - exclude)
    key = lambda t: hashlib.sha256(f"{seed}|{day}|{t}".encode()).hexdigest()
    return sorted(pool, key=key)[:n]


def register(root: Path = Path("."), *, panel=None, cache: Path | None = None,
             now: datetime | None = None, persist: bool = True) -> dict:
    """Freeze option-relevant signals from fresh prospective predictions.

    Idempotent: deterministic signal IDs; already registered signals are never
    re-written. Only predictions created after the hypothesis registration,
    after link_start and at most ``registration_max_age_days`` ago qualify.
    """
    root = Path(root)
    cfg = config(root)
    now = now or datetime.now(timezone.utc)
    if not cfg.get("enabled"):
        return {"status": "DISABLED", "n_new_signals": 0}
    if panel is None and cache is not None:
        from .factor_forward import _trusted_panel
        panel = _trusted_panel(Path(cache))
    existing, _ = _read_kind(root, "signals")
    known = {s["signal_id"] for s in existing}
    batches, stats = _sources(root, cfg, now)
    rh, rule = rule_hash(cfg), cfg["rule"]
    by_day: dict[str, list[dict]] = defaultdict(list)
    for b in batches:
        by_day[b["pit_data_date"]].append(b)
    new = []
    for day in sorted(by_day):
        group = sorted(by_day[day], key=lambda b: (
            {"REFERENCE": 0, "TREATMENT": 1}[b["comparison_group"]],
            b["hypothesis_source"] != "ml_registry", b["hypothesis_registered_at"], b["hypothesis_id"]))
        universe = sorted({t for b in group for t in b["universe"]})
        tickers = {t for b in group for t, _ in b["top"] + b["bottom"]}
        n_ctrl = int(cfg["control"]["per_signal_date"])
        control = _control_sample(universe, tickers, day, n_ctrl, str(cfg["control"]["seed"])) if universe else []
        feats, regime = _panel_context(panel, day, tickers | set(control))
        budget = int(cfg["max_signals_per_date"]) - len(control)
        signal_time = max((b["signal_time"] for b in group), default=_iso(now))
        entries = []
        for b in group:
            for role, names in (("TOP", b["top"]), ("BOTTOM", b["bottom"])):
                for pos, (ticker, score) in enumerate(names, 1):
                    entries.append((b, role, pos, ticker, score))
        if control:
            ctrl = {"hypothesis_id": CONTROL_ID, "hypothesis_source": "control", "comparison_group": "CONTROL",
                    "hypothesis_version": digest(cfg["control"])[:16], "hypothesis_hash": digest(cfg["control"]),
                    "hypothesis_registered_at": str(cfg["link_start"]), "model_id": None,
                    "experiment_version": f"control:{digest(cfg['control'])[:16]}",
                    "prediction_ref": {"journal": "universe_of_same_date_predictions", "hash": digest(universe),
                                       "prediction_date": day, "n_universe": len(universe)},
                    "signal_time": signal_time, "pit_data_date": day, "expected": None}
            entries = [(ctrl, "CONTROL", i, t, None) for i, t in enumerate(control, 1)] + entries
        for b, role, pos, ticker, score in entries:
            identity = {"hypothesis_id": b["hypothesis_id"], "hypothesis_version": b["hypothesis_version"],
                        "prediction_hash": b["prediction_ref"]["hash"], "ticker": ticker, "role": role}
            sid = "ALS-" + digest(identity)[:20]
            if sid in known:
                continue
            tradable, reason = role in ("TOP", "CONTROL"), None
            if role == "BOTTOM":
                reason = "NO_ALLOWED_STRUCTURE"  # bearish view; only bullish call structures allowed
            elif role == "TOP":
                if budget <= 0:
                    tradable, reason = False, "SIGNAL_BUDGET_EXCEEDED"
                else:
                    budget -= 1
            exp = b.get("expected")
            new.append({
                "schema": SCHEMAS["signals"], "signal_id": sid, "link_version": cfg["version"],
                "hypothesis_id": b["hypothesis_id"], "hypothesis_source": b["hypothesis_source"],
                "hypothesis_version": b["hypothesis_version"], "hypothesis_hash": b["hypothesis_hash"],
                "hypothesis_registered_at": b["hypothesis_registered_at"], "model_id": b["model_id"],
                "experiment_version": b["experiment_version"], "prediction_ref": b["prediction_ref"],
                "signal_time": b["signal_time"], "pit_data_date": day, "registered_at": _iso(now),
                "ticker": ticker, "role": role, "rank_position": pos,
                "score": round(score, 6) if isinstance(score, float) else None,
                "direction": "BEARISH" if role == "BOTTOM" else "BULLISH",
                "horizon_trading_days": int(rule["horizon_trading_days"]),
                "features": feats.get(ticker), "features_status": "OK" if ticker in feats else "UNAVAILABLE",
                "regime": regime, "comparison_group": b["comparison_group"],
                "tradable": tradable, "untradable_reason": reason,
                "structures": list(STRUCTURES) if tradable else [],
                "rule_version": rule["rule_version"], "rule_hash": rh,
                "expected_excess_return_20d": exp["value"] if exp else None,
                "expected_return_source": (f"{exp['source']}@{exp['evaluated_at']}" if exp else "UNKNOWN"),
                "order_placed": False, "production_influence": "NONE",
            })
            known.add(sid)
    written = _append(root, "signals", new, existing) if persist else len(new)
    return {"status": "OK", "n_new_signals": written, "n_prediction_batches": len(batches),
            "source_stats": stats}


# ── RTH decisions (options-capture job; requests + pyyaml only) ──────────────

def short_strike(strikes, long_strike: float, window, target: float) -> float | None:
    """Spread short-leg rule, identical to options_designer.pick_spread_leg_strike
    for calls (tested), without importing the heavy production module."""
    lo, hi = float(window[0]) * long_strike, float(window[1]) * long_strike
    cands = [float(s) for s in strikes if lo <= float(s) <= hi]
    if not cands:
        return None
    goal = float(target) * long_strike
    return min(cands, key=lambda s: abs(s - goal))


def _flat(raw: dict) -> dict:
    g = raw.get("greeks") or {}
    extra = {k: g[k] for k in ("mid_iv", "smv_vol", "delta", "gamma", "theta", "vega") if k in g}
    return {**{k: v for k, v in raw.items() if k != "greeks"}, **extra}


def _leg(rec: dict, side: str, usable: bool) -> dict:
    return {"occ_symbol": rec["occ_symbol"], "side": side, "strike": rec.get("strike"),
            "expiry": rec.get("expiry"), "bid": rec.get("bid"), "ask": rec.get("ask"),
            "mid": rec.get("mid"), "spread_pct": rec.get("spread_pct"), "iv": rec.get("iv"),
            "delta": rec.get("delta"), "open_interest": rec.get("open_interest"),
            "quote_time": rec.get("quote_time"), "retrieved_at": rec.get("retrieved_at"),
            "quote_record_id": rec.get("record_id"), "quality_status": rec.get("quality_status"),
            "quality_flags": rec.get("quality_flags"), "usable_rth": bool(usable)}


def _observe(ticker: str, uq: dict | None, now: datetime, cfg: dict, *, root: Path,
             fixed_clock: bool) -> dict | None:
    """One binding chain observation. None = transient failure (retry later)."""
    from modules import market_snapshot as ms
    from modules import option_quotes as oq
    rule = cfg["rule"]
    uq = uq or {}
    bid, ask, last = _num(uq.get("bid")), _num(uq.get("ask")), _num(uq.get("last"))
    spot = (bid + ask) / 2 if bid and ask and 0 < bid <= ask else last
    if not spot or spot <= 0:
        return None
    underlying = {"bid": bid, "ask": ask, "last": last, "spot": round(spot, 6),
                  "quote_ts": uq.get("quote_ts"), "source": uq.get("source") or "tradier"}
    expirations = ms._fetch_expirations(ticker)
    if not expirations:
        return None
    today = _ny_day(now)
    chosen = None
    for exp in expirations:
        try:
            dte = (date.fromisoformat(str(exp)[:10]) - today).days
        except ValueError:
            continue
        if int(rule["dte_min"]) <= dte <= int(rule["dte_max"]):
            chosen = (str(exp)[:10], dte)
            break
    base = {"underlying": underlying, "observed_at": _iso(now)}
    if chosen is None:
        return {**base, "status": "NO_EXPIRY_IN_DTE_WINDOW"}
    chain = ms._fetch_chain(ticker, chosen[0])
    if not chain:
        return None
    calls = []
    for o in chain:
        k = _num(o.get("strike"))
        if o.get("option_type") == "call" and k and k > 0:
            calls.append((k, o))
    if not calls:
        return {**base, "status": "NO_CALLS", "expiry": chosen[0], "dte": chosen[1]}
    long_k, long_raw = min(calls, key=lambda x: (abs(x[0] - spot), x[0]))
    # Quote time = actual retrieval (tests pass a fixed clock).
    retrieved = now if fixed_clock else datetime.now(timezone.utc)
    legs = {}

    def _archive(raw):
        rec = oq.normalize(_flat(raw), underlying=ticker, source="tradier",
                           context="alpha_link_entry", retrieved_at=retrieved)
        if rec is None:
            return None
        oq.archive([rec], root=Path(root) / oq.ROOT)
        return rec
    lrec = _archive(long_raw)
    if lrec is None:
        return None
    legs["long"] = _leg(lrec, "long", oq.usable_rth_quote(lrec))
    sk = short_strike([k for k, _ in calls], long_k, rule["short_strike_window"], rule["short_strike_target"])
    if sk is not None:
        srec = _archive(next(o for k, o in calls if k == sk))
        if srec is not None:
            legs["short"] = _leg(srec, "short", oq.usable_rth_quote(srec))
    return {**base, "observed_at": _iso(retrieved), "status": "OBSERVED", "expiry": chosen[0],
            "dte": chosen[1], "legs": legs}


def _gates(legs: list[dict], rule: dict) -> list[str]:
    reasons = []
    for leg in legs:
        tag = leg["side"].upper()
        if not leg["usable_rth"]:
            reasons.append(f"{tag}_QUOTE_NOT_USABLE_RTH")
        bid, sp, oi = _num(leg["bid"]), _num(leg["spread_pct"]), _num(leg["open_interest"])
        if bid is None or bid < float(rule["min_bid"]):
            reasons.append(f"{tag}_BID_BELOW_MIN")
        if sp is None or sp > float(rule["max_leg_spread_pct"]):
            reasons.append(f"{tag}_SPREAD_TOO_WIDE")
        if oi is None or oi < float(rule["min_open_interest"]):
            reasons.append(f"{tag}_OPEN_INTEREST_BELOW_MIN")
    iv = _num(legs[0]["iv"])
    if iv is None:
        reasons.append("IV_UNKNOWN")
    elif iv > float(rule["max_iv"]):
        reasons.append("IV_ABOVE_MAX")
    return reasons


def _decision(sig: dict, structure: str, obs: dict, cfg: dict, now: datetime) -> dict:
    rule = cfg["rule"]
    lag = int(rule["exit_max_lag_trading_days"])
    now = _ts(obs.get("observed_at")) or now  # a decision is never earlier than its quotes
    day = _ny_day(now)
    exit_target = add_trading_days(day, int(rule["horizon_trading_days"]))
    rec = {
        "schema": SCHEMAS["decisions"], "signal_id": sig["signal_id"], "structure": structure,
        "decision_id": "ALD-" + digest({"signal_id": sig["signal_id"], "structure": structure,
                                        "rule_version": rule["rule_version"]})[:20],
        "hypothesis_id": sig["hypothesis_id"], "hypothesis_version": sig["hypothesis_version"],
        "comparison_group": sig["comparison_group"], "ticker": sig["ticker"],
        "pit_data_date": sig["pit_data_date"], "signal_registered_at": sig["registered_at"],
        "rule_version": rule["rule_version"], "rule_hash": rule_hash(cfg),
        "decided_at": _iso(now), "decision_day": day.isoformat(),
        "underlying": obs.get("underlying"), "expiry": obs.get("expiry"), "dte": obs.get("dte"),
        "legs": [], "decision": "ABSTAIN", "abstain_reasons": [],
        "net_debit_ask": None, "capital_usd": None, "max_loss_usd": None, "breakeven_move": None,
        "predicted_edge": {"status": "UNKNOWN", "kind": "HEURISTIC_UNCALIBRATED", "gating": "NONE"},
        "exit_target_day": exit_target.isoformat(),
        "exit_deadline_day": add_trading_days(exit_target, lag).isoformat(),
        "evidence_class_at_entry": "UNVERIFIABLE", "order_placed": False, "production_influence": "NONE",
    }
    if obs.get("status") != "OBSERVED":
        rec["abstain_reasons"] = [obs.get("status") or "NOT_OBSERVED"]
        return rec
    legs = obs["legs"]
    chosen = [legs["long"]] + ([legs["short"]] if structure == "BULL_CALL_SPREAD" and "short" in legs else [])
    rec["legs"] = chosen
    if structure == "BULL_CALL_SPREAD" and "short" not in legs:
        rec["abstain_reasons"] = ["NO_SHORT_LEG_IN_WINDOW"]
        return rec
    reasons = _gates(chosen, rule)
    long_ask = _num(chosen[0]["ask"])
    short_bid = _num(chosen[1]["bid"]) if len(chosen) > 1 else 0.0
    spot = _num((obs.get("underlying") or {}).get("spot"))
    contracts = int(rule["contracts_per_position"])
    fee = float(rule["commission_usd_per_contract_per_side"]) * contracts * len(chosen)
    if long_ask is not None and short_bid is not None:
        debit = long_ask - short_bid
        rec["net_debit_ask"] = round(debit, 6)
        if debit <= 0:
            reasons.append("NONPOSITIVE_DEBIT")
        else:
            rec["capital_usd"] = round(debit * 100 * contracts + fee, 4)
            rec["max_loss_usd"] = round(debit * 100 * contracts + 2 * fee, 4)
            if rec["capital_usd"] > float(rule["max_capital_usd"]):
                reasons.append("CAPITAL_ABOVE_MAX")
            if spot:
                rec["breakeven_move"] = round((float(chosen[0]["strike"]) + debit) / spot - 1, 6)
    else:
        reasons.append("PRICE_MISSING")
    exp = _num(sig.get("expected_excess_return_20d"))
    if exp is not None and rec["breakeven_move"] is not None:
        rec["predicted_edge"] = {"status": "AVAILABLE", "kind": "HEURISTIC_UNCALIBRATED", "gating": "NONE",
                                 "expected_excess_return_20d": exp,
                                 "required_move_to_breakeven": rec["breakeven_move"],
                                 "edge": round(exp - rec["breakeven_move"], 6),
                                 "source": sig.get("expected_return_source")}
    if all(leg["usable_rth"] and leg["quote_record_id"] for leg in chosen):
        rec["evidence_class_at_entry"] = "QUOTE_VERIFIED_PAPER"
    rec["abstain_reasons"] = sorted(set(reasons))
    rec["decision"] = "ABSTAIN" if reasons else "TRADE"
    return rec


def _check_rule(cfg: dict, decisions: list[dict]) -> None:
    rv, rh = cfg["rule"]["rule_version"], rule_hash(cfg)
    if any(d["rule_version"] == rv and d["rule_hash"] != rh for d in decisions):
        raise EvidenceIntegrityError("contract rule modified under an already used rule_version")


def decide(root: Path = Path("."), *, now: datetime | None = None, persist: bool = True) -> dict:
    """Binding RTH decisions for registered signals (called by rth_marks)."""
    from modules import market_snapshot as ms
    from modules import option_quotes as oq
    root = Path(root)
    if not configured(root):
        return {"status": "NOT_CONFIGURED"}
    cfg = config(root)
    fixed_clock = now is not None
    now = now or datetime.now(timezone.utc)
    if not cfg.get("enabled"):
        return {"status": "DISABLED"}
    signals, _ = _read_kind(root, "signals")
    decisions, _ = _read_kind(root, "decisions")
    _check_rule(cfg, decisions)
    done = {(d["signal_id"], d["structure"]) for d in decisions}
    today = _ny_day(now)
    lag = int(cfg["decision_max_lag_trading_days"])
    pending_live, new = [], []
    for s in signals:
        if not s["tradable"]:
            continue
        reg = _ts(s["registered_at"])
        todo = [st for st in s["structures"] if (s["signal_id"], st) not in done]
        if not todo or reg is None or reg >= now:
            continue
        if today > add_trading_days(_ny_day(reg), lag):
            for st in todo:
                new.append(_decision(s, st, {"status": "DECISION_WINDOW_MISSED"}, cfg, now))
        else:
            pending_live.append((s, todo))
    res = {"status": "OK", "pending": len(pending_live), "window_missed": len(new)}
    in_window = oq.mark_window(now) == "IN_WINDOW"
    if pending_live and not in_window:
        res["status"] = "OUTSIDE_MARK_WINDOW"
    elif pending_live and not ms._use_tradier():
        res["status"] = "NO_DATA_SOURCE"
    elif pending_live:
        first = {}
        for s, _ in pending_live:
            first[s["ticker"]] = min(first.get(s["ticker"], s["registered_at"]), s["registered_at"])
        tickers = sorted(first, key=lambda t: (first[t], t))[: int(cfg["max_underlyings_per_decision_run"])]
        uq = ms.fetch_underlying_quotes(tickers)
        started = time.monotonic()
        observed, transient = 0, []
        for t in tickers:
            if time.monotonic() - started > float(cfg["decision_time_budget_seconds"]):
                transient.append(t)
                continue
            obs = _observe(t, uq.get(t), now, cfg, root=root, fixed_clock=fixed_clock)
            if obs is None:
                transient.append(t)
                continue
            observed += 1
            for s, todo in pending_live:
                if s["ticker"] == t:
                    new.extend(_decision(s, st, obs, cfg, now) for st in todo)
        res.update(underlyings_observed=observed, underlyings_retry_later=sorted(transient))
    res["n_new_decisions"] = _append(root, "decisions", new, decisions) if persist else len(new)
    res["n_trade"] = sum(d["decision"] == "TRADE" for d in new)
    return res


def due_capture_targets(*, today: date, root: Path = Path(".")) -> dict[str, datetime]:
    """Exit-quote capture targets of open TRADE decisions (whole positions only)."""
    root = Path(root)
    if not configured(root):
        return {}
    cfg = config(root)
    if not cfg.get("enabled"):
        return {}
    decisions, _ = _read_kind(root, "decisions")
    outcomes, _ = _read_kind(root, "outcomes")
    settled = {o["decision_id"] for o in outcomes}
    jobs = []
    for d in decisions:
        if d["decision"] != "TRADE" or d["decision_id"] in settled:
            continue
        target, deadline = date.fromisoformat(d["exit_target_day"]), date.fromisoformat(d["exit_deadline_day"])
        if target <= today <= deadline:
            jobs.append((deadline, d["decision_id"], [leg["occ_symbol"] for leg in d["legs"]], target))
    jobs.sort()
    budget = int(cfg["max_capture_symbols_per_run"])
    out: dict[str, datetime] = {}
    for _, _, legs, target in jobs:
        add = set(legs) - set(out)
        if len(out) + len(add) > budget:
            continue
        for sym in legs:
            start = _ny_midnight_utc(target)
            out[sym] = max(out.get(sym, start), start)
    return out


# ── settlement (weekly research job; archived quotes only) ───────────────────

def _exit_quotes(d: dict, quotes: list[dict], now: datetime) -> tuple[str, list[dict]] | None:
    from modules import option_quotes as oq
    target, deadline = d["exit_target_day"], d["exit_deadline_day"]
    want = [leg["occ_symbol"] for leg in d["legs"]]
    per_day: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for q in quotes:
        sym = q.get("occ_symbol")
        t = _ts(q.get("retrieved_at"))
        day = oq.quote_day(q)
        if sym not in want or t is None or t > now or day is None:
            continue
        if not target <= day <= deadline or not oq.usable_rth_quote(q):
            continue
        per_day[day][sym].append(q)
    for day in sorted(per_day):
        legs = per_day[day]
        if any(not legs.get(s) for s in want):
            continue
        anchor = sorted(legs[want[0]], key=lambda q: str(q["retrieved_at"]))
        for a in anchor:
            ta = _ts(a["retrieved_at"])
            chosen = [a]
            for s in want[1:]:
                match = sorted((q for q in legs[s]
                                if abs((_ts(q["retrieved_at"]) - ta).total_seconds()) <= LEG_PAIR_MAX_SECONDS),
                               key=lambda q: str(q["retrieved_at"]))
                if not match:
                    break
                chosen.append(match[0])
            if len(chosen) == len(want):
                return day, chosen
    return None


def settle(root: Path = Path("."), *, now: datetime | None = None, persist: bool = True) -> dict:
    """Exit records for TRADE decisions from archived RTH quotes only."""
    from modules import option_quotes as oq
    root = Path(root)
    cfg = config(root)
    now = now or datetime.now(timezone.utc)
    decisions, _ = _read_kind(root, "decisions")
    outcomes, _ = _read_kind(root, "outcomes")
    _check_rule(cfg, decisions)
    settled = {o["decision_id"] for o in outcomes}
    today = _ny_day(now)
    open_trades = [d for d in decisions if d["decision"] == "TRADE" and d["decision_id"] not in settled
                   and date.fromisoformat(d["exit_target_day"]) <= today]
    quotes = []
    if open_trades:
        lo = min(d["exit_target_day"] for d in open_trades)
        hi = max(d["exit_deadline_day"] for d in open_trades)
        quotes = oq.read((date.fromisoformat(lo) - timedelta(days=1)).isoformat(),
                         (date.fromisoformat(hi) + timedelta(days=1)).isoformat(), root=root / oq.ROOT)
    new, pending = [], 0
    for d in open_trades:
        found = _exit_quotes(d, quotes, now)
        deadline = date.fromisoformat(d["exit_deadline_day"])
        if found is None and today <= deadline:
            pending += 1
            continue
        rec = {"schema": SCHEMAS["outcomes"], "outcome_id": "ALO-" + d["decision_id"][4:],
               "decision_id": d["decision_id"], "signal_id": d["signal_id"],
               "hypothesis_id": d["hypothesis_id"], "structure": d["structure"],
               "comparison_group": d["comparison_group"], "ticker": d["ticker"],
               "recorded_at": _iso(now), "exit_target_day": d["exit_target_day"],
               "exit_deadline_day": d["exit_deadline_day"], "order_placed": False,
               "production_influence": "NONE"}
        if found is None:
            rec.update(status="EXIT_MISSING", exit_day=None, exit_legs=[],
                       evidence_class="UNVERIFIABLE", missing_reason="NO_USABLE_RTH_EXIT_QUOTE_IN_WINDOW")
        else:
            day, qs = found
            rec.update(status="EXIT_OBSERVED", exit_day=day, missing_reason=None,
                       exit_legs=[{"occ_symbol": q["occ_symbol"], "bid": q.get("bid"), "ask": q.get("ask"),
                                   "mid": q.get("mid"), "quote_time": q.get("quote_time"),
                                   "retrieved_at": q.get("retrieved_at"), "quote_record_id": q.get("record_id"),
                                   "quality_status": q.get("quality_status")} for q in qs],
                       evidence_class=("QUOTE_VERIFIED_PAPER"
                                       if d["evidence_class_at_entry"] == "QUOTE_VERIFIED_PAPER" else "UNVERIFIABLE"))
        new.append(rec)
    written = _append(root, "outcomes", new, outcomes) if persist else len(new)
    return {"status": "OK", "n_new_outcomes": written, "n_pending_exit": pending,
            "n_exit_missing": sum(r["status"] == "EXIT_MISSING" for r in new)}


# ── corrections (versioned, never delete or re-label) ────────────────────────

def record_correction(root: Path, *, kind: str, target_id: str, action: str, reason: str,
                      recorded_by: str, now: datetime | None = None) -> dict:
    """Human-initiated, versioned correction. Allowed: exclude a record for a
    documented data error, or downgrade its evidence class. Never upgrades,
    never edits identity, prices or outcomes, never deletes a line."""
    root = Path(root)
    if kind not in ("signals", "decisions", "outcomes") or action not in CORRECTION_ACTIONS:
        raise EvidenceIntegrityError("correction kind/action not allowed")
    if not reason or not str(reason).strip() or not recorded_by:
        raise EvidenceIntegrityError("correction needs a reason and an author")
    rows, _ = _read_kind(root, kind)
    if target_id not in {_logical_key(kind, r) for r in rows}:
        raise EvidenceIntegrityError("correction target does not exist")
    prior, _ = _read_kind(root, "corrections")
    version = 1 + sum(c["target_id"] == target_id for c in prior)
    now = now or datetime.now(timezone.utc)
    rec = {"schema": SCHEMAS["corrections"], "target_kind": kind, "target_id": target_id,
           "action": action, "reason": str(reason).strip(), "recorded_by": str(recorded_by),
           "recorded_at": _iso(now), "version": version}
    rec["correction_id"] = "ALC-" + digest({k: rec[k] for k in ("target_kind", "target_id", "action",
                                                                  "version")})[:20]
    _append(root, "corrections", [rec], prior)
    return rec


def apply_corrections(data: dict, corrections: list[dict]) -> dict:
    by_target = defaultdict(list)
    for c in sorted(corrections, key=lambda c: (c["target_id"], c["version"])):
        by_target[c["target_id"]].append(c)
    out = {}
    for kind in KINDS:
        rows = []
        for r in data.get(kind, []):
            cs = by_target.get(_logical_key(kind, r)) if kind != "corrections" else None
            if not cs:
                rows.append(r)
                continue
            view = dict(r)
            view["corrections"] = [{"version": c["version"], "action": c["action"], "reason": c["reason"]}
                                   for c in cs]
            if any(c["action"] == "EXCLUDE_DATA_ERROR" for c in cs):
                view["excluded_by_correction"] = True
            if any(c["action"] == "DOWNGRADE_EVIDENCE" for c in cs):
                for f in ("evidence_class", "evidence_class_at_entry"):
                    if f in view:
                        view[f] = "UNVERIFIABLE"
            rows.append(view)
        out[kind] = rows
    return out


# ── executed-trade attribution check (forward_options) ───────────────────────

BROKER_FILL_FIELDS = ("order_id", "fill_price", "filled_at")


def broker_execution_documented(row: dict) -> tuple[bool, str]:
    """EXECUTED evidence requires a documented BROKER execution for entry AND exit:
    ``broker_execution = {"broker": str, "entry": {order_id, fill_price, filled_at},
    "exit": {order_id, fill_price, filled_at}}``. Archived bid/ask quotes, paper
    bookings and quote-derived outcomes never qualify, however well verified -
    they stay QUOTE_VERIFIED_PAPER at best. Nothing is ever upgraded later."""
    ex = row.get("broker_execution")
    if not isinstance(ex, dict) or not ex:
        return False, "NO_BROKER_EXECUTION_RECORD"
    if not isinstance(ex.get("broker"), str) or not ex["broker"].strip():
        return False, "BROKER_MISSING"
    for side in ("entry", "exit"):
        fill = ex.get(side)
        if not isinstance(fill, dict) or any(fill.get(k) in (None, "") for k in BROKER_FILL_FIELDS):
            return False, f"{side.upper()}_FILL_INCOMPLETE"
        if _num(fill.get("fill_price")) is None or _ts(fill.get("filled_at")) is None:
            return False, f"{side.upper()}_FILL_INVALID"
    if _ts(ex["exit"]["filled_at"]) <= _ts(ex["entry"]["filled_at"]):
        return False, "FILL_ORDER_INVALID"
    return True, "BROKER_EXECUTION_DOCUMENTED"


def trade_evidence_class(row: dict, *, quote_verifiable: bool) -> str:
    """Evidence class of a real trade record (derived, never written back)."""
    if broker_execution_documented(row)[0]:
        return "EXECUTED"
    return "QUOTE_VERIFIED_PAPER" if quote_verifiable else "UNVERIFIABLE"


def ex_ante_attribution(row: dict, signals_by_id: dict) -> tuple[str | None, str]:
    """A real trade counts for a hypothesis ONLY if its record carries a signal
    ID that was registered BEFORE the trade entry. Nothing else attributes."""
    sid = row.get("alpha_signal_id")
    if not sid:
        return None, "NO_EX_ANTE_SIGNAL_ID"
    sig = signals_by_id.get(sid)
    if sig is None:
        return None, "UNKNOWN_SIGNAL_ID"
    reg = _ts(sig.get("registered_at"))
    entry = _ts(row.get("entry_time") or row.get("opened_at"))
    if entry is None:
        try:
            entry = _ny_midnight_utc(date.fromisoformat(str(row.get("entry_date"))[:10]))
        except ValueError:
            entry = None
    if reg is None or entry is None or reg >= entry:
        return None, "SIGNAL_NOT_REGISTERED_BEFORE_ENTRY"
    if sig.get("ticker") != row.get("ticker"):
        return None, "TICKER_MISMATCH"
    return sig["hypothesis_id"], "ATTRIBUTED_EX_ANTE"


# ── coverage report ──────────────────────────────────────────────────────────

def coverage(root: Path = Path("."), *, now: datetime | None = None, persist: bool = True) -> dict:
    root = Path(root)
    cfg = config(root)
    now = now or datetime.now(timezone.utc)
    data = read_all(root)
    signals, decisions, outcomes = data["signals"], data["decisions"], data["outcomes"]
    _check_rule(cfg, decisions)
    today = _ny_day(now)
    lag = int(cfg["decision_max_lag_trading_days"])
    cov = cfg["coverage"]
    dec_by_sig = defaultdict(dict)
    for d in decisions:
        dec_by_sig[d["signal_id"]][d["structure"]] = d
    out_by_dec = {o["decision_id"]: o for o in outcomes}
    hyp: dict[str, dict] = {}
    for s in signals:
        h = hyp.setdefault(s["hypothesis_id"], {
            "hypothesis_source": s["hypothesis_source"], "comparison_group": s["comparison_group"],
            "versions": set(), "signals": 0, "untradable": Counter(), "signal_dates": set(),
            "excluded_by_correction": 0, "pending_decision": 0, "overdue_decision": 0,
            "by_structure": {st: {"TRADE": 0, "ABSTAIN": 0, "abstain_reasons": Counter(),
                                  "EXIT_OBSERVED": 0, "EXIT_MISSING": 0, "EXIT_PENDING": 0,
                                  "evidence": Counter(), "verified_dates": set()} for st in STRUCTURES}})
        h["signals"] += 1
        h["versions"].add(s["hypothesis_version"])
        h["signal_dates"].add(s["pit_data_date"])
        if s.get("excluded_by_correction"):
            h["excluded_by_correction"] += 1
            continue
        if not s["tradable"]:
            h["untradable"][s["untradable_reason"]] += 1
            continue
        for st in s["structures"]:
            d = dec_by_sig[s["signal_id"]].get(st)
            if d is None:
                overdue = today > add_trading_days(_ny_day(_ts(s["registered_at"])), lag)
                h["overdue_decision" if overdue else "pending_decision"] += 1
                continue
            b = h["by_structure"][st]
            if d.get("excluded_by_correction"):
                h["excluded_by_correction"] += 1
                continue
            b[d["decision"]] += 1
            if d["decision"] == "ABSTAIN":
                b["abstain_reasons"].update(d["abstain_reasons"])
                continue
            o = out_by_dec.get(d["decision_id"])
            if o is None or o.get("excluded_by_correction"):
                b["EXIT_PENDING" if o is None else "EXIT_MISSING"] += 1
                continue
            b[o["status"]] += 1
            b["evidence"][o["evidence_class"]] += 1
            if o["evidence_class"] == "QUOTE_VERIFIED_PAPER":
                b["verified_dates"].add(s["pit_data_date"])
    report_h = {}
    for hid, h in sorted(hyp.items()):
        by = {}
        evaluable = False
        for st, b in h["by_structure"].items():
            verified = b["evidence"].get("QUOTE_VERIFIED_PAPER", 0)
            ok = verified >= int(cov["min_verified_outcomes"]) and len(b["verified_dates"]) >= int(
                cov["min_distinct_signal_dates"])
            evaluable = evaluable or ok
            by[st] = {"TRADE": b["TRADE"], "ABSTAIN": b["ABSTAIN"],
                      "abstain_reasons": dict(sorted(b["abstain_reasons"].items())),
                      "EXIT_OBSERVED": b["EXIT_OBSERVED"], "EXIT_MISSING": b["EXIT_MISSING"],
                      "EXIT_PENDING": b["EXIT_PENDING"],
                      "evidence": {c: b["evidence"].get(c, 0) for c in EVIDENCE_CLASSES},
                      "distinct_verified_signal_dates": len(b["verified_dates"]),
                      "status": "COVERAGE_SUFFICIENT" if ok else "NEED_MORE_DATA"}
        report_h[hid] = {"hypothesis_source": h["hypothesis_source"], "comparison_group": h["comparison_group"],
                         "versions": sorted(h["versions"]), "signals": h["signals"],
                         "distinct_signal_dates": len(h["signal_dates"]),
                         "untradable": dict(sorted(h["untradable"].items())),
                         "pending_decision": h["pending_decision"], "overdue_decision": h["overdue_decision"],
                         "excluded_by_correction": h["excluded_by_correction"], "by_structure": by,
                         "status": "COVERAGE_SUFFICIENT" if evaluable else "NEED_MORE_DATA"}
    _, stats = _sources(root, cfg, now)
    ml_rows, _ = _ml_predictions(root)
    link_start = date.fromisoformat(str(cfg["link_start"]))
    pre = [r for r in ml_rows if (_ts(r.get("created_at")) or now).date() < link_start]
    executed = _executed_check(root, {s["signal_id"]: s for s in signals})
    integrity = data["integrity"]
    conflicts = sum(v["conflicting_duplicates"] for v in integrity.values())
    report = {
        "schema": "alpha-option-link-coverage-v1", "as_of": today.isoformat(),
        "generated_at": _iso(now), "link_version": cfg["version"], "link_start": cfg["link_start"],
        "rule_version": cfg["rule"]["rule_version"], "rule_hash": rule_hash(cfg),
        "status": ("COVERAGE_SUFFICIENT" if any(h["status"] == "COVERAGE_SUFFICIENT" for h in report_h.values())
                   else "NEED_MORE_DATA"),
        "n_signals": len(signals), "n_decisions": len(decisions), "n_outcomes": len(outcomes),
        "decisions_by_class": dict(Counter(d["decision"] for d in decisions)),
        "outcomes_by_evidence_class": {c: sum(o["evidence_class"] == c for o in outcomes) for c in EVIDENCE_CLASSES},
        "hypotheses": report_h,
        "pre_link": {"ml_prediction_rows": len(pre), "prediction_dates": sorted({r["prediction_date"] for r in pre}),
                     "status": "NOT_LINKABLE_RETROACTIVELY",
                     "reason": "No option decision was observed when these predictions were made."},
        "source_stats": stats, "executed": executed,
        "integrity": {"journals": integrity, "conflicting_duplicates": conflicts,
                      "corrections": len(data["corrections"]),
                      "status": "OK" if conflicts == 0 else "CONFLICT_DETECTED"},
        "order_placed": False, "production_influence": "NONE", "llm_calls": 0,
        "economic_result": "NOT_EVALUATED_IN_PHASE_A",
    }
    if persist:
        _atomic_json(root / REPORT, report)
    return report


def _executed_check(root: Path, signals_by_id: dict) -> dict:
    p = Path(root) / TRADE_OUTCOMES
    counts, executed = Counter(), Counter()
    n = 0
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                counts["UNREADABLE"] += 1
                continue
            n += 1
            hid, why = ex_ante_attribution(row, signals_by_id)
            counts[why] += 1
            documented, reason = broker_execution_documented(row)
            executed[reason] += 1
            if hid and documented:
                executed["EXECUTED_AND_ATTRIBUTED_EX_ANTE"] += 1
    return {"n_real_trades": n, "attribution": dict(sorted(counts.items())),
            "n_attributed_ex_ante": counts.get("ATTRIBUTED_EX_ANTE", 0),
            "broker_execution_checks": dict(sorted(executed.items())),
            "n_documented_broker_executions": executed.get("BROKER_EXECUTION_DOCUMENTED", 0),
            "n_executed_evidence": executed.get("EXECUTED_AND_ATTRIBUTED_EX_ANTE", 0),
            "note": "The trade outcomes are scanner paper trades valued from archived bid/ask quotes, without "
                    "broker fills: they are never EXECUTED evidence. EXECUTED needs a documented broker execution "
                    "AND an ex-ante alpha_signal_id (neither exists today; trading path unchanged)."}


def run(root: Path = Path("."), *, cache: Path | None = None, panel=None,
        now: datetime | None = None) -> dict:
    """Weekly research step: register -> settle -> coverage (no network)."""
    now = now or datetime.now(timezone.utc)
    reg = register(root, panel=panel, cache=cache, now=now)
    st = settle(root, now=now)
    cov = coverage(root, now=now)
    return {"register": reg, "settle": st, "status": cov["status"], "n_signals": cov["n_signals"],
            "n_decisions": cov["n_decisions"], "n_outcomes": cov["n_outcomes"],
            "integrity": cov["integrity"]["status"], "production_influence": "NONE"}

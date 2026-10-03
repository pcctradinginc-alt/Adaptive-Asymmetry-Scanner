"""modules/promotion_controller.py – kontrollierte Research→Production-Promotion.

    python -m modules.promotion_controller [--notify]     (CI: feedback.yml, nach feedback.py)

Einzige Stelle, die über den PRODUKTIONSEINFLUSS einer Hypothese entscheidet.
Angewendet wird dieser Einfluss ausschließlich vom ProductionIntelligenceAdapter
(modules/production_intelligence_adapter.py), der promotion_state.json liest.

Harte Regeln (Auftrag, getestet in tests/test_promotion.py):
  * Historische Evidenz führt höchstens zu HISTORICALLY_VALIDATED. Danach wird die
    Hypothese als PROSPECTIVE_CHALLENGER eingefroren; für die Promotion zählen NUR
    Beobachtungen mit Entscheidungszeit >= forward_start UND > registered_at, deren
    Regel-Auswertung zum Entscheidungszeitpunkt im Decision-Ledger festgehalten wurde
    (nie nachträglich neu berechnet) und deren spec_hash zum Vertrag passt.
  * Mindest-N, unabhängige Signaltage, Kalenderspanne, Regime-/Sektor-Breite,
    Ausreißer-Robustheit, mehrere Zeitfenster – sonst KEEP_SHADOW / NEED_MORE_DATA.
  * Multiple Testing: Bonferroni über alle je registrierten Verträge derselben Familie
    (production_class) × geplante Looks; jeder Look wird protokolliert.
  * Gewinnt die Gegenrichtung (H_alt) -> REJECT, kein Vorzeichenwechsel.
  * Automatisch höchstens policy.max_automatic_influence (Default ABSTENTION_ONLY);
    darüber nur Empfehlung (INTELLIGENCE PROMOTION CANDIDATE), FULL_PRODUCTION nur
    per menschlicher Freigabe (config/promotion_approvals.yaml, CODEOWNERS).
  * Jeder Statuswechsel append-only mit previous_state, new_state, timestamp, reason,
    evidence_snapshot, metrics, code_version, data_version (Hash-Kette).
  * Demotion vorab im Vertrag festgelegt; Abstufung 25 % -> 10 % -> ABSTENTION -> SHADOW
    -> DEMOTED; Integritätsfehler -> ROLLBACK (sofort ohne Einfluss).

Dieses Modul schreibt NIE config.yaml, Gate-Schwellen, pipeline.py oder Modelle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import statistics
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from modules import hypothesis_contract as hc
from modules.outcomes import RELIABLE_OUTCOME_METHODS

log = logging.getLogger(__name__)

OUT = Path("outputs/intelligence")
STATE_PATH = OUT / "promotion_state.json"
TRANSITIONS = OUT / "promotion_transitions.jsonl"
LOOKS = OUT / "promotion_looks.jsonl"
LEDGER_DIR = OUT / "decision_ledger"
OUTCOMES = OUT / "decision_outcomes.jsonl"
NOTIFIED = OUT / "promotion_notified.json"
APPROVALS = Path("config/promotion_approvals.yaml")
HISTORY = Path("outputs/history.json")

STATES = ("IDEA", "HISTORICAL_RESEARCH", "HISTORICALLY_VALIDATED", "PROSPECTIVE_CHALLENGER", "FORWARD_VALIDATED",
          "GUARDED_PRODUCTION", "LIMITED_PRODUCTION", "FULL_PRODUCTION", "DEMOTED", "REJECTED", "EXPIRED")
TERMINAL = ("DEMOTED", "REJECTED", "EXPIRED")
SHADOW_STATES = ("PROSPECTIVE_CHALLENGER", "FORWARD_VALIDATED")
ACTIVE_STATES = ("GUARDED_PRODUCTION", "LIMITED_PRODUCTION", "FULL_PRODUCTION")
ALLOWED = {
    "IDEA": {"HISTORICAL_RESEARCH", "PROSPECTIVE_CHALLENGER", "REJECTED"},
    "HISTORICAL_RESEARCH": {"HISTORICALLY_VALIDATED", "PROSPECTIVE_CHALLENGER", "REJECTED"},
    "HISTORICALLY_VALIDATED": {"PROSPECTIVE_CHALLENGER", "REJECTED"},
    "PROSPECTIVE_CHALLENGER": {"FORWARD_VALIDATED", "GUARDED_PRODUCTION", "LIMITED_PRODUCTION", "REJECTED",
                               "EXPIRED", "DEMOTED"},
    "FORWARD_VALIDATED": {"GUARDED_PRODUCTION", "LIMITED_PRODUCTION", "PROSPECTIVE_CHALLENGER", "REJECTED",
                          "EXPIRED", "DEMOTED"},
    "GUARDED_PRODUCTION": {"LIMITED_PRODUCTION", "PROSPECTIVE_CHALLENGER", "DEMOTED"},
    "LIMITED_PRODUCTION": {"LIMITED_PRODUCTION", "GUARDED_PRODUCTION", "PROSPECTIVE_CHALLENGER", "DEMOTED",
                           "FULL_PRODUCTION"},
    "FULL_PRODUCTION": {"LIMITED_PRODUCTION", "DEMOTED"},
    "DEMOTED": set(), "REJECTED": set(), "EXPIRED": set(),
}
DECISIONS = ("REJECT", "KEEP_SHADOW", "ALLOW_ABSTENTION", "ALLOW_RERANK", "ALLOW_10_PERCENT_WEIGHT",
             "ALLOW_25_PERCENT_WEIGHT", "RECOMMEND_FULL_PROMOTION", "DEMOTE", "ROLLBACK")
LEVEL_OF_DECISION = {"ALLOW_ABSTENTION": "ABSTENTION_ONLY", "ALLOW_RERANK": "RERANK_ONLY",
                     "ALLOW_10_PERCENT_WEIGHT": "WEIGHT_10", "ALLOW_25_PERCENT_WEIGHT": "WEIGHT_25"}
# Einfluss-Leiter (ohne SCORE_LIMITED als eigene Weight-Stufe: Score-Klasse nutzt RERANK -> SCORE)
LADDER = {"abstention": ["NONE", "ABSTENTION_ONLY"],
          "rerank": ["NONE", "RERANK_ONLY"],
          "score": ["NONE", "RERANK_ONLY", "SCORE_LIMITED"],
          "weight": ["NONE", "RERANK_ONLY", "WEIGHT_10", "WEIGHT_25"],
          "research_only": ["NONE"]}


# Nur Outcomes aus echten Optionsquotes zählen als Evidenz (modules/outcomes.py).
REAL_OUTCOME_METHODS = RELIABLE_OUTCOME_METHODS


class TransitionError(ValueError):
    pass


# ── Versionen ───────────────────────────────────────────────────────────────
def code_version() -> str:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"][:12]
    try:
        return subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True,
                              timeout=5).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _file_hash(paths) -> str:
    h = hashlib.sha256()
    for p in sorted(Path(x) for x in paths):
        if p.is_file():
            h.update(p.name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def data_version(ledger_dir: Path, outcomes: Path) -> str:
    files = list(ledger_dir.glob("*.jsonl")) if ledger_dir.exists() else []
    return _file_hash(files + [outcomes])


def policy_hash(policy: dict) -> str:
    return hashlib.sha256(json.dumps(policy, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ── Zustandsübergänge (append-only, Hash-Kette) ─────────────────────────────
def _chain_hash(prev: str, e: dict) -> str:
    body = json.dumps({k: v for k, v in e.items() if k != "entry_hash"}, sort_keys=True, default=str)
    return hashlib.sha256((prev + body).encode()).hexdigest()


def read_transitions(path: Path | None = None) -> tuple[list[dict], list[str]]:
    path = path or TRANSITIONS
    out, problems, prev = [], [], ""
    if not path.exists():
        return [], []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        e = json.loads(line)
        if e.get("prev_hash") != prev or _chain_hash(prev, e) != e.get("entry_hash"):
            problems.append(f"Transition-Log manipuliert bei Zeile {i + 1}")
        prev = e.get("entry_hash", "")
        out.append(e)
    return out, problems


def current_states(path: Path | None = None) -> dict[str, dict]:
    """Letzter Übergang je Vertrag (key) -> {state, influence_level, ...}."""
    entries, _ = read_transitions(path)
    cur: dict[str, dict] = {}
    for e in entries:
        cur[e["key"]] = e
    return cur


def transition(key: str, new_state: str, *, reason: str, evidence_snapshot: dict | None = None,
               metrics: dict | None = None, influence_level: str = "NONE", spec_hash: str = "",
               decision: str | None = None, path: Path | None = None, now: str | None = None,
               code_ver: str | None = None, data_ver: str | None = None) -> dict:
    path = path or TRANSITIONS
    if new_state not in STATES:
        raise TransitionError(f"unbekannter Zustand {new_state}")
    entries, problems = read_transitions(path)
    if problems:
        raise TransitionError("; ".join(problems))
    prev_state = None
    for e in entries:
        if e["key"] == key:
            prev_state = e["new_state"]
    if prev_state is None:
        if new_state not in ("IDEA", "HISTORICAL_RESEARCH", "PROSPECTIVE_CHALLENGER"):
            raise TransitionError(f"{key}: erster Zustand muss IDEA/HISTORICAL_RESEARCH/PROSPECTIVE_CHALLENGER sein")
    elif new_state not in ALLOWED[prev_state] and new_state != prev_state:
        raise TransitionError(f"{key}: Übergang {prev_state} -> {new_state} nicht erlaubt")
    if new_state == "FULL_PRODUCTION" and decision != "HUMAN_APPROVAL":
        raise TransitionError("FULL_PRODUCTION nur per menschlicher Freigabe")
    e = {"key": key, "previous_state": prev_state, "new_state": new_state,
         "timestamp": now or datetime.now(timezone.utc).isoformat(timespec="seconds"), "reason": reason,
         "decision": decision, "influence_level": influence_level, "spec_hash": spec_hash,
         "evidence_snapshot": evidence_snapshot or {}, "metrics": metrics or {},
         "code_version": code_ver or code_version(), "data_version": data_ver or "",
         "prev_hash": entries[-1]["entry_hash"] if entries else ""}
    e["entry_hash"] = _chain_hash(e["prev_hash"], e)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(e, sort_keys=True, default=str) + "\n")
    return e


# ── Decision-Ledger / Outcomes ──────────────────────────────────────────────
def read_jsonl_dir(d: Path) -> list[dict]:
    rows = []
    if d.exists():
        for f in sorted(d.glob("*.jsonl")):
            rows += [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    return rows


def read_outcomes(path: Path | None = None) -> dict[str, dict]:
    path = path or OUTCOMES
    out: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                e = json.loads(line)
                out.setdefault(e["decision_id"], e)          # erstes Outcome gilt (append-only)
    return out


def resolve_outcomes(history: dict, ledger_dir: Path | None = None, outcomes_path: Path | None = None,
                     now: str | None = None) -> int:
    """Verbindet Decision-Ledger-Zeilen mit realisierten Outcomes aus history.json:
    umgesetzte Trades -> closed_trades; abstinierte Trades -> counterfactual_closed
    (gleicher Lebenszyklus wie echte Trades). Append-only, je decision_id einmal;
    outcome_method wird mitgeführt (nur Quote-basierte Outcomes zählen als Evidenz)."""
    ledger_dir, outcomes_path = ledger_dir or LEDGER_DIR, outcomes_path or OUTCOMES
    have = read_outcomes(outcomes_path)
    closed = {(t.get("ticker"), str(t.get("entry_date", ""))[:10]): t for t in history.get("closed_trades") or []
              if t.get("outcome") is not None}
    shadow = {(t.get("ticker"), str(t.get("entry_date", ""))[:10]): t for t in history.get("counterfactual_closed") or []
              if t.get("outcome") is not None}
    n = 0
    for r in read_jsonl_dir(ledger_dir):
        if r["decision_id"] in have:
            continue
        k = (r.get("ticker"), str(r.get("date"))[:10])
        if r.get("final_production_decision") == "TRADE" and k in closed:
            t, src = closed[k], "closed_trade"
        elif r.get("final_production_decision") == "ABSTAIN" and k in shadow:
            t, src = shadow[k], "counterfactual_trade"
        else:
            continue
        e = {"decision_id": r["decision_id"], "outcome": float(t["outcome"]), "source": src,
             "outcome_method": t.get("outcome_method") or "unknown", "close_reason": t.get("close_reason"),
             "close_date": t.get("close_date"), "mfe": t.get("mfe"), "mae": t.get("mae"),
             "resolved_at": now or datetime.now(timezone.utc).isoformat(timespec="seconds")}
        outcomes_path.parent.mkdir(parents=True, exist_ok=True)
        with open(outcomes_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(e, sort_keys=True) + "\n")
        have[r["decision_id"]] = e
        n += 1
    return n


# ── Kennzahlen ──────────────────────────────────────────────────────────────
def _ts(x) -> datetime:
    d = datetime.fromisoformat(str(x).replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def group_metrics(vals: list[float], probs: list[float | None] | None = None,
                  mfe: list | None = None, mae: list | None = None) -> dict:
    if not vals:
        return {"n": 0}
    wins = [v for v in vals if v > 0]
    losses = [v for v in vals if v <= 0]
    eq, peak, mdd = 0.0, 0.0, 0.0
    for v in vals:
        eq += v
        peak = max(peak, eq)
        mdd = min(mdd, eq - peak)
    m = {"n": len(vals), "win_rate": round(len(wins) / len(vals), 4), "average_return": round(statistics.fmean(vals), 5),
         "median_return": round(statistics.median(vals), 5), "expectancy": round(statistics.fmean(vals), 5),
         "profit_factor": round(sum(wins) / abs(sum(losses)), 3) if losses and sum(losses) != 0 else None,
         "max_drawdown": round(mdd, 4)}
    m["mfe"] = round(statistics.fmean([x for x in mfe if x is not None]), 4) if mfe and any(
        x is not None for x in mfe) else None
    m["mae"] = round(statistics.fmean([x for x in mae if x is not None]), 4) if mae and any(
        x is not None for x in mae) else None
    if probs and any(p is not None for p in probs):
        pairs = [(p, 1.0 if v > 0 else 0.0) for p, v in zip(probs, vals) if p is not None]
        m["brier"] = round(statistics.fmean([(p - y) ** 2 for p, y in pairs]), 4)
        m["ece"] = ece(pairs)
    else:
        m["brier"] = m["ece"] = None
    return m


def ece(pairs: list[tuple[float, float]], bins: int = 5) -> float | None:
    if not pairs:
        return None
    tot, err = len(pairs), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [(p, y) for p, y in pairs if (lo <= p < hi) or (b == bins - 1 and p == 1.0)]
        if sel:
            err += len(sel) / tot * abs(statistics.fmean(p for p, _ in sel) - statistics.fmean(y for _, y in sel))
    return round(err, 4)


def _sharpe(xs: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    sd = statistics.stdev(xs)
    return round(statistics.fmean(xs) / sd, 3) if sd > 0 else None


def _sortino(xs: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    dn = [min(0.0, x) ** 2 for x in xs]
    dd = math.sqrt(statistics.fmean(dn))
    return round(statistics.fmean(xs) / dd, 3) if dd > 0 else None


def observations(contract: dict, spec_hash: str, rows: list[dict], outcomes: dict[str, dict],
                 since: datetime | None = None) -> list[dict]:
    """NUR Zukunftsdaten: Entscheidungszeit >= forward_start und > registered_at (und >= since),
    Auswertung zum Entscheidungszeitpunkt festgehalten, gleicher spec_hash, Outcome bekannt."""
    k = hc.key(contract)
    fwd, reg = _ts(contract["forward_start"]), _ts(contract["registered_at"])
    out = []
    for r in rows:
        t = _ts(r["timestamp"])
        if t < fwd or t <= reg or (since is not None and t < since):
            continue
        h = (r.get("intelligence") or {}).get(k)
        if not h or h.get("spec_hash") != spec_hash or not h.get("evaluable") or not h.get("in_scope"):
            continue
        if r.get("champion_decision") != "TRADE":
            continue
        o = outcomes.get(r["decision_id"])
        if o is None or o.get("outcome") is None or o.get("outcome_method") not in REAL_OUTCOME_METHODS:
            continue
        out.append({"decision_id": r["decision_id"], "date": str(r["date"])[:10], "ts": t, "fired": bool(h.get("fired")),
                    "outcome": float(o["outcome"]), "prob": r.get("champion_probability"),
                    "sector": r.get("sector"), "regime": r.get("regime"), "mfe": o.get("mfe"), "mae": o.get("mae"),
                    "applied": bool(h.get("applied"))})
    return sorted(out, key=lambda x: x["ts"])


def _delta(obs: list[dict], direction: int) -> float | None:
    """Inkrementeller Nutzen der Regel ggü. Champion-only (je Champion-Trade):
    Abstinenz (direction -1): E[durchgelassen] - E[alle]. direction +1: E[gefeuert] - E[alle]."""
    if not obs:
        return None
    allm = statistics.fmean(o["outcome"] for o in obs)
    if direction < 0:
        kept = [o["outcome"] for o in obs if not o["fired"]]
    else:
        kept = [o["outcome"] for o in obs if o["fired"]]
    if not kept:
        return None
    return statistics.fmean(kept) - allm


def _bootstrap(obs: list[dict], direction: int, n: int, seed: int, alpha: float) -> tuple:
    """Block-Bootstrap über unabhängige Signaltage; einseitige Grenzen bei alpha."""
    by_date: dict[str, list[dict]] = {}
    for o in obs:
        by_date.setdefault(o["date"], []).append(o)
    dates = sorted(by_date)
    if len(dates) < 2:
        return None, None
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        sample = [o for d in (rng.choice(dates) for _ in dates) for o in by_date[d]]
        v = _delta(sample, direction)
        if v is not None:
            vals.append(v)
    if len(vals) < 0.9 * n:
        return None, None
    vals.sort()
    lo = vals[max(0, int(alpha * len(vals)) - 1)]
    hi = vals[min(len(vals) - 1, int((1 - alpha) * len(vals)))]
    return round(lo, 5), round(hi, 5)


def _outlier_trims(obs: list[dict], direction: int) -> dict:
    """Effekt nach Entfernen der Beobachtungen, die den Vorteil am stärksten tragen."""
    base = _delta(obs, direction)
    if base is None:
        return {}
    allm = statistics.fmean(o["outcome"] for o in obs)

    def contrib(o):  # Beitrag zum Vorteil der Regel
        if direction < 0:
            return (allm - o["outcome"]) if o["fired"] else (o["outcome"] - allm)
        return (o["outcome"] - allm) if o["fired"] else (allm - o["outcome"])
    ranked = sorted(obs, key=contrib, reverse=True)
    out = {}
    for name, k in (("top1", 1), ("top3", 3), ("top5pct", max(1, math.ceil(0.05 * len(obs))))):
        v = _delta(ranked[k:], direction) if len(ranked) > k else None
        out[name] = round(v, 5) if v is not None else None
    return out


def evidence(contract: dict, spec_hash: str, rows: list[dict], outcomes: dict[str, dict], policy: dict,
             alpha: float, since: datetime | None = None) -> dict:
    direction = int(contract["direction"])
    obs = observations(contract, spec_hash, rows, outcomes, since)
    st = policy.get("statistics") or {}
    ev: dict = {"n_observations": len(obs), "n_independent_dates": len({o["date"] for o in obs}),
                "calendar_span_days": (obs[-1]["ts"] - obs[0]["ts"]).days if len(obs) > 1 else 0,
                "first_observation": obs[0]["date"] if obs else None,
                "last_observation": obs[-1]["date"] if obs else None, "data_kind": "prospective_forward"}
    fired = [o for o in obs if o["fired"]]
    notf = [o for o in obs if not o["fired"]]
    ev["n_fired"] = len(fired)
    ev["n_fired_independent_dates"] = len({o["date"] for o in fired})
    ev["all"] = group_metrics([o["outcome"] for o in obs], [o["prob"] for o in obs],
                              [o["mfe"] for o in obs], [o["mae"] for o in obs])
    ev["fired"] = group_metrics([o["outcome"] for o in fired], [o["prob"] for o in fired])
    ev["not_fired"] = group_metrics([o["outcome"] for o in notf], [o["prob"] for o in notf])
    kept = notf if direction < 0 else fired
    ev["policy"] = group_metrics([o["outcome"] for o in kept], [o["prob"] for o in kept])
    d = _delta(obs, direction)
    ev["delta_expectancy"] = round(d, 5) if d is not None else None
    ev["ci"] = list(_bootstrap(obs, direction, int(st.get("bootstrap_n", 2000)),
                               int(st.get("bootstrap_seed", 41)), alpha)) if obs else [None, None]
    ev["outlier_trims"] = _outlier_trims(obs, direction)
    # Zeitfenster: drei gleich lange Kalenderfenster
    win = []
    if len(obs) > 1:
        t0, t1 = obs[0]["ts"], obs[-1]["ts"]
        step = (t1 - t0) / 3
        for i in range(3):
            a, b = t0 + step * i, t0 + step * (i + 1)
            part = [o for o in obs if (a <= o["ts"] < b) or (i == 2 and o["ts"] == t1)]
            v = _delta(part, direction)
            win.append(round(v, 5) if v is not None else None)
    ev["time_windows"] = win
    ev["regimes"] = sorted({o["regime"] for o in obs if o["regime"]})
    secs = [o["sector"] for o in fired if o["sector"]]
    ev["max_sector_share_fired"] = round(max(secs.count(s) for s in set(secs)) / len(secs), 3) if secs else None
    ev["dominant_sector_fired"] = max(set(secs), key=secs.count) if secs else None
    if direction < 0:      # Abstinenz-Bilanz
        ev["abstention"] = {"n_blocked": len(fired),
                            "blocked_trade_win_rate": ev["fired"].get("win_rate"),
                            "blocked_trade_expectancy": ev["fired"].get("expectancy"),
                            "blocked_trade_mean_return": ev["fired"].get("average_return"),
                            "avoided_losses": round(-sum(o["outcome"] for o in fired if o["outcome"] < 0), 4),
                            "missed_winners": sum(1 for o in fired if o["outcome"] > 0),
                            "missed_winner_return": round(sum(o["outcome"] for o in fired if o["outcome"] > 0), 4),
                            "net_value_of_abstention": round(-sum(o["outcome"] for o in fired) / len(obs), 5)
                            if obs else None,
                            "precision": round(sum(1 for o in fired if o["outcome"] <= 0) / len(fired), 4)
                            if fired else None,
                            "false_positive_rate": round(sum(1 for o in fired if o["outcome"] > 0)
                                                         / max(1, sum(1 for o in obs if o["outcome"] > 0)), 4)
                            if obs else None}
    return ev


# ── Multiple Testing ────────────────────────────────────────────────────────
def family_alpha(contract: dict, registry_entries: list[dict], policy: dict) -> dict:
    st = policy.get("statistics") or {}
    fam = contract["production_class"]
    members = {e["key"] for e in registry_entries if (e.get("contract") or {}).get("production_class") == fam}
    variants = sum(1 for e in registry_entries if e.get("hypothesis_id") == contract["hypothesis_id"])
    m = max(1, len(members))
    looks = int(st.get("planned_looks", 12))
    return {"hypothesis_family": fam, "family_size": m, "number_of_variants": variants, "planned_looks": looks,
            "alpha_effective": float(st.get("alpha_family", 0.05)) / (m * looks)}


def looks_used(key: str, path: Path | None = None) -> int:
    path = path or LOOKS
    if not path.exists():
        return 0
    return sum(1 for x in path.read_text().splitlines() if x.strip() and json.loads(x)["key"] == key)


LOOK_INTERVAL_DAYS = 28      # Promotion-Entscheidungen nur an geplanten (monatlichen) Looks


def last_look_at(key: str, path: Path | None = None) -> datetime | None:
    path = path or LOOKS
    if not path.exists():
        return None
    ts = [json.loads(x)["at"] for x in path.read_text().splitlines() if x.strip() and json.loads(x)["key"] == key]
    return _ts(max(ts)) if ts else None


def record_look(key: str, ev: dict, path: Path | None = None, now: str | None = None) -> None:
    path = path or LOOKS
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"key": key, "at": now or datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "n": ev.get("n_observations"), "delta": ev.get("delta_expectancy"),
                             "ci": ev.get("ci")}) + "\n")


# ── Entscheidung ────────────────────────────────────────────────────────────
def insufficiency(contract: dict, ev: dict, policy: dict) -> list[str]:
    fl = policy.get("evidence_floors") or {}
    pc = contract["promotion_criteria"]
    need = []
    if ev["n_observations"] < int(contract["minimum_sample_size"]):
        need.append(f"n {ev['n_observations']}/{contract['minimum_sample_size']}")
    if ev["n_independent_dates"] < int(contract["minimum_independent_dates"]):
        need.append(f"unabhängige Tage {ev['n_independent_dates']}/{contract['minimum_independent_dates']}")
    if ev["calendar_span_days"] < int(contract["minimum_calendar_span"]):
        need.append(f"Kalenderspanne {ev['calendar_span_days']}/{contract['minimum_calendar_span']} T")
    if ev["n_fired"] < int(pc.get("min_fired_observations", 1)):
        need.append(f"Regel-Treffer {ev['n_fired']}/{pc.get('min_fired_observations')}")
    if ev["n_fired_independent_dates"] < int(pc.get("min_fired_independent_dates", 1)):
        need.append(f"Treffer-Tage {ev['n_fired_independent_dates']}/{pc.get('min_fired_independent_dates')}")
    scoped_regime = "all" not in (contract.get("regime_scope") or ["all"])
    if not scoped_regime and len(ev["regimes"]) < int(fl.get("min_regimes_observed", 2)):
        need.append(f"nur {len(ev['regimes'])} Marktregime beobachtet")
    scoped_sector = "all" not in (contract.get("sector_scope") or ["all"])
    share = ev.get("max_sector_share_fired")
    if not scoped_sector and share is not None and share > float(fl.get("max_single_sector_share", 0.6)):
        need.append(f"Sektor {ev.get('dominant_sector_fired')} dominiert ({share:.0%}) – nur sektor-gescopte "
                    f"neue Hypothese zulässig")
    return need


def promotion_checks(contract: dict, ev: dict, policy: dict) -> tuple[bool, list[str]]:
    pc = contract["promotion_criteria"]
    fl = policy.get("evidence_floors") or {}
    fails = []
    d, (lo, _hi) = ev["delta_expectancy"], (ev["ci"] or [None, None])
    if d is None or d <= float(pc.get("delta_expectancy_min", 0.0)):
        fails.append(f"Δ Expectancy {d} nicht > {pc.get('delta_expectancy_min')}")
    if lo is None or lo <= float(pc.get("ci_lower_min", 0.0)):
        fails.append(f"CI-Untergrenze {lo} nicht > {pc.get('ci_lower_min')} (alpha korrigiert)")
    if pc.get("require_outlier_robust", True):
        bad = [k for k, v in (ev.get("outlier_trims") or {}).items() if v is None or v <= 0]
        if bad or not ev.get("outlier_trims"):
            fails.append(f"nicht ausreißer-robust (Effekt verschwindet ohne {bad or 'Daten'})")
    need_w = int(pc.get("require_time_windows_positive", fl.get("min_time_windows_positive", 2)))
    pos_w = sum(1 for v in ev.get("time_windows") or [] if v is not None and v > 0)
    if pos_w < need_w:
        fails.append(f"Effekt nur in {pos_w}/3 Zeitfenstern positiv (nötig {need_w})")
    if pc.get("calibration_not_worse", True):
        bp, ba = (ev.get("policy") or {}).get("brier"), (ev.get("all") or {}).get("brier")
        if bp is not None and ba is not None and bp > ba + 0.005:
            fails.append(f"Kalibrierung schlechter (Brier {bp} > {ba})")
    ddp, dda = (ev.get("policy") or {}).get("max_drawdown"), (ev.get("all") or {}).get("max_drawdown")
    if ddp is not None and dda is not None and ddp < dda - float(pc.get("max_drawdown_not_worse_by", 0.10)):
        fails.append(f"Drawdown deutlich schlechter ({ddp} vs {dda})")
    return (not fails), fails


def _cap_level(level: str, contract: dict, policy: dict) -> tuple[str, bool]:
    """-> (automatisch erlaubte Stufe, wurde gekappt?)."""
    order = hc.INFLUENCE_LEVELS
    auto = policy.get("max_automatic_influence", "ABSTENTION_ONLY")
    cap = min(order.index(level), order.index(auto) if auto in order else 0)
    return order[cap], order.index(level) > cap


def decide(contract: dict, cur: dict | None, ev: dict, policy: dict, *, integrity_ok: bool,
           looks: int, alpha_info: dict, now: datetime, post_ev: dict | None = None,
           look_due: bool = True) -> dict:
    """-> {decision, new_state, influence_level, reasons, next_requirement, recommendation}."""
    state = (cur or {}).get("new_state", "PROSPECTIVE_CHALLENGER")
    level = (cur or {}).get("influence_level", "NONE")
    cls = contract["production_class"]
    ladder = LADDER.get(cls, ["NONE"])
    res = {"decision": "KEEP_SHADOW", "new_state": state, "influence_level": level, "reasons": [],
           "next_requirement": None, "recommendation": None}
    if not integrity_ok:
        res.update(decision="ROLLBACK" if level != "NONE" else "KEEP_SHADOW",
                   new_state="DEMOTED" if level != "NONE" else state, influence_level="NONE",
                   reasons=["Integritätsfehler (Hash/Registry/Kette) – kein Einfluss"])
        return res
    if state in TERMINAL:
        res.update(decision="REJECT" if state == "REJECTED" else "KEEP_SHADOW", influence_level="NONE",
                   reasons=[f"terminal: {state}"])
        return res
    lo_hi = ev.get("ci") or [None, None]
    # H_alt gewinnt: Effekt signifikant in Gegenrichtung
    if ev["n_observations"] >= int(contract["minimum_sample_size"]) and lo_hi[1] is not None and lo_hi[1] < 0:
        if state in ACTIVE_STATES:
            res.update(decision="DEMOTE", new_state="DEMOTED", influence_level="NONE",
                       reasons=["Effektumkehr (H_alt) nach Promotion"])
        else:
            res.update(decision="REJECT", new_state="REJECTED", influence_level="NONE",
                       reasons=["Gegenrichtung (H_alt) signifikant – REJECT, kein Vorzeichenwechsel; "
                                "ggf. neue Hypothese registrieren und erneut prospektiv testen"])
        return res
    if state in ACTIVE_STATES:
        return _monitor(contract, state, level, ladder, post_ev or ev, ev, policy, res)
    # Shadow-Zustände
    planned = int(alpha_info.get("planned_looks", 12))
    need = insufficiency(contract, ev, policy)
    if need:
        if looks >= planned:
            res.update(decision="REJECT", new_state="EXPIRED", reasons=["Looks erschöpft ohne ausreichende Evidenz"])
        else:
            res.update(reasons=["NEED_MORE_DATA: " + "; ".join(need)], next_requirement="; ".join(need))
        return res
    if not look_due:
        res.update(reasons=[f"Evidenzminimum erreicht; Entscheidung erst am nächsten geplanten Look "
                            f"(Intervall {LOOK_INTERVAL_DAYS} T, Alpha-Spending)"])
        return res
    ok, fails = promotion_checks(contract, ev, policy)
    if not ok:
        if looks >= planned:
            res.update(decision="REJECT", new_state="EXPIRED", reasons=["Looks erschöpft"] + fails)
        else:
            res.update(reasons=fails, next_requirement="; ".join(fails))
        return res
    # Forward validiert -> nächste Stufe der Leiter
    target = ladder[1] if len(ladder) > 1 else "NONE"
    target = min(target, contract["maximum_initial_influence"], key=hc.INFLUENCE_LEVELS.index)
    allowed, capped = _cap_level(target, contract, policy)
    if allowed == "NONE":
        res.update(new_state="FORWARD_VALIDATED", reasons=["Forward validiert; automatische Stufe NONE (Policy)"],
                   recommendation=target if target != "NONE" else None)
        return res
    dec = {v: k for k, v in LEVEL_OF_DECISION.items()}.get(allowed, "KEEP_SHADOW")
    if allowed == "SCORE_LIMITED":
        dec = "ALLOW_RERANK"          # Score-Einfluss nur über eigene Freigabe nach Rerank-Evidenz
    res.update(decision=dec, new_state="GUARDED_PRODUCTION" if allowed == "ABSTENTION_ONLY" else "LIMITED_PRODUCTION",
               influence_level=allowed, reasons=["Promotion-Gate bestanden (nur Forward-Daten)"],
               recommendation=target if capped else None)
    return res


def _monitor(contract, state, level, ladder, post_ev, ev, policy, res) -> dict:
    """Aktive Hypothese: vorab festgelegte Demotion-Kriterien auf dem rollierenden Fenster
    seit Promotion; Abstufung genau eine Stufe der Leiter."""
    dc = {**(policy.get("demotion_defaults") or {}), **contract["demotion_criteria"]}
    reasons = []
    win = int(dc.get("rolling_window_observations", 30))
    if post_ev["n_observations"] >= win:
        if int(contract["direction"]) < 0:
            ab = post_ev.get("abstention") or {}
            nv = ab.get("net_value_of_abstention")
            if nv is not None and nv <= float(dc.get("abstention_net_value_min", 0.0)):
                reasons.append(f"Abstinenz-Nettowert {nv} <= {dc.get('abstention_net_value_min')}")
            be, pe = (post_ev.get("fired") or {}).get("expectancy"), (post_ev.get("not_fired") or {}).get("expectancy")
            if be is not None and pe is not None and be >= pe:
                reasons.append(f"blockierte Trades nicht schlechter (E {be} >= {pe})")
        d = post_ev.get("delta_expectancy")
        if d is not None and d < float(dc.get("rolling_expectancy_min", 0.0)):
            reasons.append(f"rollierende Δ Expectancy {d} < {dc.get('rolling_expectancy_min')}")
        e_p = (post_ev.get("policy") or {}).get("ece")
        if e_p is not None and e_p > float(dc.get("ece_max", 0.10)):
            reasons.append(f"Kalibrierung ECE {e_p} > {dc.get('ece_max')}")
    if reasons:
        i = ladder.index(level) if level in ladder else 0
        lower = ladder[i - 1] if i > 0 else "NONE"
        if lower == "NONE":
            new_state = "PROSPECTIVE_CHALLENGER"          # zurück auf SHADOW, sammelt weiter
        elif lower == "ABSTENTION_ONLY":
            new_state = "GUARDED_PRODUCTION"
        else:
            new_state = "LIMITED_PRODUCTION"
        res.update(decision="DEMOTE", new_state=new_state, influence_level=lower, reasons=reasons)
        return res
    res.update(decision={"ABSTENTION_ONLY": "ALLOW_ABSTENTION", "RERANK_ONLY": "ALLOW_RERANK",
                         "SCORE_LIMITED": "ALLOW_RERANK", "WEIGHT_10": "ALLOW_10_PERCENT_WEIGHT",
                         "WEIGHT_25": "ALLOW_25_PERCENT_WEIGHT"}.get(level, "KEEP_SHADOW"),
               reasons=[f"Monitoring ok ({post_ev['n_observations']} Beobachtungen seit Promotion)"])
    i = ladder.index(level) if level in ladder else 0
    if i + 1 < len(ladder):
        nxt = ladder[i + 1]
        allowed, capped = _cap_level(nxt, contract, policy)
        ok, _ = promotion_checks(contract, post_ev, policy)
        if ok and not insufficiency(contract, post_ev, policy):
            if allowed == nxt and not capped:
                res.update(decision={"RERANK_ONLY": "ALLOW_RERANK", "SCORE_LIMITED": "ALLOW_RERANK",
                                     "WEIGHT_10": "ALLOW_10_PERCENT_WEIGHT",
                                     "WEIGHT_25": "ALLOW_25_PERCENT_WEIGHT"}.get(nxt, res["decision"]),
                           new_state="LIMITED_PRODUCTION", influence_level=nxt,
                           reasons=[f"weitere unabhängige Forward-Evidenz seit Promotion -> {nxt}"])
            else:
                res["recommendation"] = nxt
        else:
            res["next_requirement"] = f"für {nxt}: erneut volle Forward-Evidenz seit Promotion"
    else:
        res["recommendation"] = "FULL_PRODUCTION (nur menschliche Freigabe)"
        res["decision"] = "RECOMMEND_FULL_PROMOTION" if level == "WEIGHT_25" else res["decision"]
    return res


# ── Freigaben (Mensch) ──────────────────────────────────────────────────────
def load_approvals(path: Path | None = None) -> dict:
    path = path or APPROVALS
    if not path.exists():
        return {}
    d = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {a["key"]: a for a in d.get("approvals") or [] if a.get("key")}


# ── Lauf ────────────────────────────────────────────────────────────────────
def run(*, contracts: list[dict] | None = None, policy: dict | None = None, now: datetime | None = None,
        registry: Path | None = None, transitions: Path | None = None, ledger_dir: Path | None = None,
        outcomes_path: Path | None = None, looks_path: Path | None = None, state_path: Path | None = None,
        history: dict | None = None, approvals: dict | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    now_s = now.isoformat(timespec="seconds")
    policy = policy if policy is not None else hc.load_policy()
    contracts = contracts if contracts is not None else hc.load()
    registry, transitions = registry or hc.REGISTRY, transitions or TRANSITIONS
    ledger_dir, outcomes_path = ledger_dir or LEDGER_DIR, outcomes_path or OUTCOMES
    looks_path, state_path = looks_path or LOOKS, state_path or STATE_PATH
    approvals = approvals if approvals is not None else load_approvals()
    if history is None and HISTORY.exists():
        history = json.loads(HISTORY.read_text(encoding="utf-8"))
    if history:
        resolve_outcomes(history, ledger_dir, outcomes_path, now_s)
    reg_status = hc.register(contracts, registry, policy, now_s)
    reg_entries, reg_problems = hc.read_registry(registry)
    _, tr_problems = read_transitions(transitions)
    integrity = {"registry": reg_problems, "transitions": tr_problems}
    rows = read_jsonl_dir(ledger_dir)
    outs = read_outcomes(outcomes_path)
    cv, dv = code_version(), data_version(ledger_dir, outcomes_path)
    reg_hash = {e["key"]: e["spec_hash"] for e in reg_entries}
    state = {"generated": now_s, "policy_version": policy.get("version"), "policy_hash": policy_hash(policy),
             "max_automatic_influence": policy.get("max_automatic_influence", "ABSTENTION_ONLY"),
             "code_version": cv, "data_version": dv, "integrity": integrity, "hypotheses": {},
             "notices": []}
    cur_all = current_states(transitions) if not tr_problems else {}
    for c in contracts:
        k = hc.key(c)
        st = reg_status.get(k, {})
        h = hc.spec_hash(c)
        integrity_ok = (st.get("status") == "VALID" and reg_hash.get(k) == h and not reg_problems and not tr_problems)
        cur = cur_all.get(k)
        if cur is None and integrity_ok:
            cur = transition(k, "PROSPECTIVE_CHALLENGER", reason="registriert und eingefroren (Forward ab forward_start)",
                             spec_hash=h, decision="KEEP_SHADOW", path=transitions, now=now_s, code_ver=cv, data_ver=dv)
        if cur is not None and cur.get("spec_hash") and cur["spec_hash"] != h:
            integrity_ok = False
        ai = family_alpha(c, reg_entries, policy)
        ev = evidence(c, h, rows, outs, policy, ai["alpha_effective"]) if integrity_ok else {
            "n_observations": 0, "n_independent_dates": 0, "calendar_span_days": 0, "n_fired": 0,
            "n_fired_independent_dates": 0, "ci": [None, None], "regimes": []}
        post_ev = None
        if cur and cur.get("new_state") in ACTIVE_STATES and integrity_ok:
            post_ev = evidence(c, h, rows, outs, policy, ai["alpha_effective"], since=_ts(cur["timestamp"]))
        lu = looks_used(k, looks_path)
        last = last_look_at(k, looks_path)
        look_due = last is None or (now - last).days >= LOOK_INTERVAL_DAYS
        if integrity_ok and look_due and (cur or {}).get("new_state") in SHADOW_STATES \
                and not insufficiency(c, ev, policy):
            record_look(k, ev, looks_path, now_s)
            lu += 1
        d = decide(c, cur, ev, policy, integrity_ok=integrity_ok, looks=lu, alpha_info=ai, now=now, post_ev=post_ev,
                   look_due=look_due)
        appr = approvals.get(k) or {}
        if appr.get("rollback") and (cur or {}).get("influence_level", "NONE") != "NONE":
            d.update(decision="ROLLBACK", new_state="DEMOTED", influence_level="NONE",
                     reasons=[f"menschlicher Rollback: {appr.get('reason', '')}"])
        if cur is not None and d["new_state"] in ACTIVE_STATES and cur["new_state"] == "PROSPECTIVE_CHALLENGER":
            cur = transition(k, "FORWARD_VALIDATED", reason="Promotion-Gate auf reinen Forward-Daten bestanden",
                             evidence_snapshot=_snapshot(ev), metrics={"delta_expectancy": ev.get("delta_expectancy"),
                                                                       "ci": ev.get("ci")},
                             spec_hash=h, decision="KEEP_SHADOW", path=transitions, now=now_s, code_ver=cv,
                             data_ver=dv)
        if cur is not None and (d["new_state"] != cur["new_state"] or d["influence_level"] != cur.get("influence_level")):
            try:
                cur = transition(k, d["new_state"], reason="; ".join(d["reasons"]), evidence_snapshot=_snapshot(ev),
                                 metrics={"delta_expectancy": ev.get("delta_expectancy"), "ci": ev.get("ci")},
                                 influence_level=d["influence_level"], spec_hash=h, decision=d["decision"],
                                 path=transitions, now=now_s, code_ver=cv, data_ver=dv)
            except TransitionError as e:
                d["reasons"].append(f"Übergang verweigert: {e}")
        if d["decision"] in ("ALLOW_ABSTENTION", "ALLOW_RERANK", "ALLOW_10_PERCENT_WEIGHT", "ALLOW_25_PERCENT_WEIGHT",
                             "RECOMMEND_FULL_PROMOTION") or d.get("recommendation"):
            state["notices"].append(_notice(c, cur, d, ev))
        state["hypotheses"][k] = {
            "hypothesis_id": c["hypothesis_id"], "version": c["version"], "title": c.get("title"),
            "description": c.get("research_question"), "source_type": c["source_type"],
            "production_class": c["production_class"], "spec_hash": h, "registry_status": st.get("status"),
            "registry_errors": st.get("errors"), "state": (cur or {}).get("new_state", "IDEA"),
            "influence_level": (cur or {}).get("influence_level", "NONE") if integrity_ok else "NONE",
            "sector_scope": c.get("sector_scope"), "regime_scope": c.get("regime_scope"),
            "forward_start": c["forward_start"], "registered_at": c["registered_at"],
            "decision": d["decision"], "reasons": d["reasons"], "next_requirement": d["next_requirement"],
            "recommendation": d["recommendation"], "looks_used": lu, "multiple_testing": ai,
            "evidence": _snapshot(ev), "post_promotion_evidence": _snapshot(post_ev) if post_ev else None,
            "integrity_ok": integrity_ok, "last_transition": (cur or {}).get("timestamp")}
    inv = hc.research_inventory()
    state["multiple_testing"] = {
        "number_of_hypotheses_tested": len(reg_entries) + len(inv),
        "promotion_contracts_registered": len(reg_entries),
        "research_inventory": len(inv),
        "families": sorted({(e.get("contract") or {}).get("production_class") for e in reg_entries} - {None})}
    state["evaluation"] = evaluate_arms(rows, outs)
    state["state_hash"] = state_digest(state)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return state


def state_digest(state: dict) -> str:
    core = {k: v for k, v in state.items() if k != "state_hash"}
    return hashlib.sha256(json.dumps(core, sort_keys=True, default=str).encode()).hexdigest()


def _snapshot(ev: dict | None) -> dict:
    if not ev:
        return {}
    return {k: v for k, v in ev.items() if k != "ts"}


def _notice(c, cur, d, ev) -> dict:
    ab = ev.get("abstention") or {}
    return {"title": "INTELLIGENCE PROMOTION CANDIDATE", "hypothesis": hc.key(c), "description": c.get("title"),
            "current_level": (cur or {}).get("influence_level", "NONE"),
            "proposed_level": d.get("recommendation") or d.get("influence_level"),
            "automatic": d["decision"] in LEVEL_OF_DECISION and not d.get("recommendation"),
            "forward_n": ev.get("n_observations"), "calendar_span_days": ev.get("calendar_span_days"),
            "delta_expectancy": ev.get("delta_expectancy"), "ci": ev.get("ci"),
            "calibration": {"brier_policy": (ev.get("policy") or {}).get("brier"),
                            "brier_champion": (ev.get("all") or {}).get("brier")},
            "drawdown": {"policy": (ev.get("policy") or {}).get("max_drawdown"),
                         "champion": (ev.get("all") or {}).get("max_drawdown")},
            "evidence": {"outlier_trims": ev.get("outlier_trims"), "time_windows": ev.get("time_windows"),
                         "net_value_of_abstention": ab.get("net_value_of_abstention")},
            "risks": "kleine Forward-Stichprobe; Regime-Wechsel; Ausreißer – Demotion-Kriterien aktiv"}


# ── Champion vs. Adaptive (nur prospektive Daten) ───────────────────────────
def evaluate_arms(rows: list[dict], outs: dict[str, dict]) -> dict:
    """CHAMPION ONLY vs. +ABSTENTION vs. +RERANK vs. +LIMITED INTELLIGENCE auf denselben
    prospektiven Champion-Trades. Abstinierte Trades werden über ihr Shadow-Outcome bewertet."""
    res = [(r, outs.get(r["decision_id"])) for r in rows if r.get("champion_decision") == "TRADE"]
    n_approx = sum(1 for _, o in res if o and o.get("outcome") is not None
                   and o.get("outcome_method") not in REAL_OUTCOME_METHODS)
    res = [(r, o) for r, o in res if o and o.get("outcome") is not None
           and o.get("outcome_method") in REAL_OUTCOME_METHODS]
    arms = {}

    def arm(name, sel_fn):
        kept = [(r, o) for r, o in res if sel_fn(r)]
        dropped = [(r, o) for r, o in res if not sel_fn(r)]
        vals = [o["outcome"] for _, o in kept]
        m = group_metrics(vals, [r.get("champion_probability") for r, _ in kept],
                          [o.get("mfe") for _, o in kept], [o.get("mae") for _, o in kept])
        by_day: dict[str, list[float]] = {}
        for r, o in kept:
            by_day.setdefault(str(r["date"])[:10], []).append(o["outcome"])
        daily = [statistics.fmean(v) for _, v in sorted(by_day.items())]
        m.update(sharpe=_sharpe(daily), sortino=_sortino(daily), trade_count=len(kept),
                 missed_winners=sum(1 for _, o in dropped if o["outcome"] > 0),
                 avoided_losers=sum(1 for _, o in dropped if o["outcome"] <= 0))
        arms[name] = m
    arm("CHAMPION_ONLY", lambda r: True)
    arm("ADAPTIVE_ACTUAL", lambda r: r.get("final_production_decision") == "TRADE")
    arm("CHAMPION_PLUS_ABSTENTION_SHADOW", lambda r: r.get("intelligence_decision") != "ABSTAIN")
    # Ranking-Arme: Precision@K / Mittel Top-K je Tag, Champion- vs. Intelligence-Rang
    by_day: dict[str, list] = {}
    for r, o in res:
        by_day.setdefault(str(r["date"])[:10], []).append((r, o))
    rank = {}
    for name, fld in (("champion", "champion_rank"), ("intelligence", "intelligence_rank")):
        top1, top3 = [], []
        for _, items in by_day.items():
            items = [x for x in items if x[0].get(fld) is not None]
            items.sort(key=lambda x: x[0][fld])
            if items:
                top1.append(items[0][1]["outcome"])
                top3 += [x[1]["outcome"] for x in items[:3]]
        rank[name] = {"precision_at_1": round(sum(1 for v in top1 if v > 0) / len(top1), 4) if top1 else None,
                      "precision_at_3": round(sum(1 for v in top3 if v > 0) / len(top3), 4) if top3 else None,
                      "mean_return_top1": round(statistics.fmean(top1), 5) if top1 else None,
                      "mean_return_top3": round(statistics.fmean(top3), 5) if top3 else None}
    arms["RANKING"] = rank
    arms["data_kind"] = "prospective_forward_only"
    arms["n_resolved_decisions"] = len(res)
    arms["n_excluded_approximate_outcomes"] = n_approx
    return arms


# ── Benachrichtigung ────────────────────────────────────────────────────────
def notify(state: dict, *, send: bool = False, path: Path | None = None) -> list[dict]:
    path = path or NOTIFIED
    done = set(json.loads(path.read_text())) if path.exists() else set()
    new = [n for n in state.get("notices") or []
           if f"{n['hypothesis']}:{n['proposed_level']}" not in done]
    if not new:
        return []
    text = "\n\n".join(render_notice(n) for n in new)
    if send:
        try:
            from modules.mailer import send_mail
            send_mail("INTELLIGENCE PROMOTION CANDIDATE", "<pre>" + text + "</pre>", text)
        except Exception as e:  # noqa: BLE001 – Mailfehler darf nichts blockieren
            log.warning(f"Promotion-Mail nicht gesendet: {e}")
    done |= {f"{n['hypothesis']}:{n['proposed_level']}" for n in new}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(done)))
    return new


def render_notice(n: dict) -> str:
    return "\n".join([n["title"], f"Hypothesis: {n['hypothesis']} – {n.get('description')}",
                      f"Current level: {n['current_level']}", f"Proposed level: {n['proposed_level']}"
                      + ("" if n.get("automatic") else " (Empfehlung – Mensch/PR entscheidet)"),
                      f"Forward N: {n['forward_n']}", f"Calendar span: {n['calendar_span_days']} Tage",
                      f"Δ Expectancy: {n['delta_expectancy']} (CI {n['ci']})", f"Calibration: {n['calibration']}",
                      f"Drawdown: {n['drawdown']}", f"Evidence: {n['evidence']}", f"Risks: {n['risks']}"])


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--notify", action="store_true")
    args = ap.parse_args(argv)
    state = run()
    sent = notify(state, send=args.notify)
    for k, h in state["hypotheses"].items():
        ev = h["evidence"]
        print(f"{k}: {h['state']} / {h['influence_level']} – {h['decision']} "
              f"(n={ev.get('n_observations')}, Tage={ev.get('n_independent_dates')}, Δ={ev.get('delta_expectancy')}) "
              f"{'; '.join(h['reasons'])[:200]}")
    print(f"Benachrichtigungen: {len(sent)} · Integrität: {state['integrity']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

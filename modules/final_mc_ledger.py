"""modules/final_mc_ledger.py – prospektiver Shadow-Ledger der Population FINAL_MC_SURVIVOR.

Neue wissenschaftliche Fragestellung (eigene, versionierte Verträge `eligible_stage:
FINAL_MC_SURVIVOR`): Unterscheiden sich Final-MC-Survivors, bei denen eine
Abstention-Regel feuert, von denen ohne Treffer? Vergleichsgruppe ist IMMER dieselbe
Population (triggered vs. non-triggered), nie die Champion-Trades.

Ablauf (append-only, nichts wird nachträglich neu berechnet):
  1. pipeline.py, direkt nach Final MC: record_survivors() friert je Survivor die
     Vertragsauswertung (spec_hash, fired, evaluable), SystemState, Regime, VIX,
     Modelluneinigkeit, erwarteten Drawdown und die Champion-Wahrscheinlichkeit ein.
  2. pipeline.py, Laufende: record_downstream() hält fest, was spätere Gates taten
     (ROI-Teil-Gates `fail_gates`, Score, Korrelation, Adapter, finale Champion-
     Entscheidung). Rein beschreibend – kein Survivor wird dadurch zum Trade.
  3. feedback.py: resolve_outcomes() schreibt je Horizont (20/45/60 Kalendertage) die
     Richtungsrendite des Basiswerts ab Schlusskurs des Signaltags sowie MFE/MAE.
  4. promotion_controller: evidence() – nur Beobachtungen ab forward_start, gleicher
     spec_hash, Outcome bekannt; N, unabhängige Ereignis-Cluster, Signaltage und
     Kalenderspanne; Block-Bootstrap über Signaltage UND über Ereignis-Cluster
     (beide Untergrenzen müssen halten).

Kein Pfad dieses Moduls verändert eine Produktionsentscheidung.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import statistics
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from modules import hypothesis_contract as hc

log = logging.getLogger(__name__)

STAGE = "FINAL_MC_SURVIVOR"
CHAMPION_STAGE = "CHAMPION_TRADE"
DIR = Path("outputs/intelligence/final_mc_ledger")
DOWNSTREAM = Path("outputs/intelligence/final_mc_downstream.jsonl")
OUTCOMES = Path("outputs/intelligence/final_mc_outcomes.jsonl")
HORIZONS = (20, 45, 60)                 # Kalendertage nach dem Signaltag
OUTCOME_METHOD = "underlying_close_daily"
CLUSTER_GAP_DAYS = 10                   # gleicher Ticker innerhalb von 10 Tagen = gleiches Ereignis


def stage_of(c: dict) -> str:
    """v1-Verträge tragen kein Feld -> Champion-Population (Spezifikation bleibt unverändert)."""
    return c.get("eligible_stage") or CHAMPION_STAGE


def stage_contracts(contracts: list[dict], stage: str = STAGE) -> list[dict]:
    return [c for c in contracts if stage_of(c) == stage]


# ── 1. Survivor aufzeichnen ─────────────────────────────────────────────────
def _registered(contracts: list[dict], registry: Path | None) -> dict[str, dict]:
    reg, problems = hc.read_registry(registry)
    if problems:
        log.warning(f"final_mc_ledger: Registry-Probleme – keine Vertragsauswertung ({problems[:2]})")
        return {}
    ok = {e["key"]: e["spec_hash"] for e in reg}
    return {hc.key(c): c for c in contracts if ok.get(hc.key(c)) == hc.spec_hash(c)}


def record_survivors(survivors: list[dict], *, today: str, vix=None, ctx: dict | None = None,
                     contracts: list[dict] | None = None, registry: Path | None = None,
                     ledger_dir: Path | None = None, now: datetime | None = None) -> list[dict]:
    """Je Final-MC-Survivor eine Beobachtung mit eingefrorener Auswertung der FINAL_MC-Verträge."""
    from modules import production_intelligence_adapter as pia
    now = now or datetime.now(timezone.utc)
    ledger_dir = ledger_dir or DIR
    contracts = stage_contracts(contracts if contracts is not None else hc.load())
    valid = _registered(contracts, registry)
    if ctx is None:
        ctx = pia.research_context(today)
    regime = pia.regime_label(vix)
    off = set(ctx.get("data_disabled_signals") or [])
    missing = set(ctx.get("data_unavailable_features") or [])
    # Idempotent: je (Tag, Ticker) genau eine Beobachtung – ein wiederholter Lauf (manueller
    # Neustart) erzeugt keine Pseudo-Stichprobe; die erste, eingefrorene Auswertung gilt.
    seen = {(str(r["date"])[:10], r["ticker"]) for r in read_rows(ledger_dir) if str(r["date"])[:7] == today[:7]}
    rows = []
    for i, s in enumerate(survivors):
        t = s.get("ticker")
        if not t or (today, t) in seen:
            continue
        seen.add((today, t))
        env = pia.candidate_env(s, ctx, vix)
        sector = s.get("sector") or (s.get("info") or {}).get("sector")
        trig = {}
        for k, c in valid.items():
            usable = not (k in off or c["hypothesis_id"] in off or set(c.get("features") or []) & missing)
            scope = hc.in_scope(c, sector, regime) if usable else False
            f = hc.fires(c, env) if scope else None
            trig[k] = {"spec_hash": hc.spec_hash(c), "in_scope": scope, "evaluable": f is not None,
                       "fired": bool(f), "signal_value": hc.evaluate_signal(c, env) if scope else None,
                       "data_available": usable}
        sim = s.get("simulation") or {}
        da = s.get("deep_analysis") or {}
        oid = hashlib.sha256(f"{STAGE}|{today}|{t}|{now.isoformat()}|{i}".encode()).hexdigest()[:16]
        rows.append({
            "observation_id": oid, "stage": STAGE, "timestamp": now.isoformat(timespec="seconds"),
            "date": today, "ticker": t, "sector": sector, "regime": regime, "vix": vix,
            "direction": da.get("direction") or "BULLISH",
            "reference_price": sim.get("current_price"),
            "champion_probability": env.get("champion_probability"),
            "model_disagreement": env.get("ml_disagreement_sd"), "predicted_drawdown_60": env.get("ml_exp_dd_60"),
            "system_state": {"version": ctx.get("system_state_version"), "safe_mode": ctx.get("safe_mode_active"),
                             "drift_level": ctx.get("drift_level")},
            "contracts": trig,
            "env_hash": hashlib.sha256(json.dumps(env, sort_keys=True, default=str).encode()).hexdigest()[:16]})
    if rows:
        from modules.atomic_io import append_jsonl
        append_jsonl(ledger_dir / f"{today[:7]}.jsonl", rows, sort_keys=True, ensure_ascii=False)
    return rows


# ── 2. Downstream (beschreibend) ────────────────────────────────────────────
def record_downstream(observations: list[dict], outcome_by_ticker: dict[str, dict], *,
                      path: Path | None = None, now: datetime | None = None) -> int:
    """outcome_by_ticker[ticker] = {champion_decision, reason, fail_gates}. Fehlender Ticker -> UNKNOWN."""
    path = path or DOWNSTREAM
    now_s = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    if not observations:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for o in observations:
            d = outcome_by_ticker.get(o["ticker"]) or {}
            fh.write(json.dumps({"observation_id": o["observation_id"], "recorded_at": now_s,
                                 "champion_decision": d.get("champion_decision") or "UNKNOWN",
                                 "reason": d.get("reason"), "fail_gates": d.get("fail_gates")},
                                sort_keys=True, default=str) + "\n")
    return len(observations)


def downstream_map(*, final_tickers: set[str], roi_rejects: list[dict], reject_stats: dict,
                   blocked_by_intelligence: set[str] | None = None,
                   other_reasons: dict[str, str] | None = None) -> dict[str, dict]:
    """Was ist nach Final MC mit jedem Survivor passiert? (Champion = umgesetzter Vorschlag)."""
    out: dict[str, dict] = {}
    for reason, v in (reject_stats or {}).items():
        for t in v.get("tickers") or []:
            out[t] = {"champion_decision": "NO_TRADE", "reason": reason}
    for t, why in (other_reasons or {}).items():          # Score-/Korrelationsfilter (kein reject())
        out[t] = {"champion_decision": "NO_TRADE", "reason": why}
    for r in roi_rejects or []:
        out[r["ticker"]] = {"champion_decision": "NO_TRADE", "reason": "roi_gate", "fail_gates": r.get("fail_gates")}
    for t in blocked_by_intelligence or set():
        out[t] = {"champion_decision": "TRADE", "reason": "intelligence_abstention_applied"}
    for t in final_tickers:
        out[t] = {"champion_decision": "TRADE", "reason": None}
    return out


# ── 3. Outcomes (verzögert) ─────────────────────────────────────────────────
def read_jsonl(path: Path) -> list[dict]:
    from modules.atomic_io import read_jsonl as _read
    return _read(path)                           # abgeschnittene letzte Zeile (Abbruch) -> ignoriert


def read_rows(ledger_dir: Path | None = None) -> list[dict]:
    d = ledger_dir or DIR
    rows = []
    if d.exists():
        for f in sorted(d.glob("*.jsonl")):
            rows += read_jsonl(f)
    return rows


def read_outcomes(path: Path | None = None) -> dict[tuple[str, int], dict]:
    out: dict[tuple[str, int], dict] = {}
    for e in read_jsonl(path or OUTCOMES):
        out.setdefault((e["observation_id"], int(e["horizon"])), e)      # erstes gilt (append-only)
    return out


def path_outcome(bars: list[tuple[date, float, float, float]], signal_day: date, horizon: int,
                 direction: str) -> dict | None:
    """bars: (Tag, High, Low, Close) aufsteigend. Einstieg = Schlusskurs am Signaltag (bzw. erster
    Handelstag danach); Ausstieg = letzter Schlusskurs <= Signaltag + horizon. Richtungsbereinigt.
    None, wenn das Fenster noch nicht vollständig vorliegt."""
    entry = next(((d, c) for d, _, _, c in bars if d >= signal_day), None)
    if entry is None or entry[1] is None or entry[1] <= 0:
        return None
    end = signal_day + timedelta(days=horizon)
    if not bars or bars[-1][0] < end:
        return None                                  # Horizont noch nicht erreicht
    window = [b for b in bars if entry[0] < b[0] <= end]
    if not window:
        return None
    sign = -1.0 if str(direction).upper() == "BEARISH" else 1.0
    p0 = entry[1]
    ret = sign * (window[-1][3] / p0 - 1.0)
    fav = [sign * ((h if sign > 0 else l) / p0 - 1.0) for _, h, l, _ in window]
    adv = [sign * ((l if sign > 0 else h) / p0 - 1.0) for _, h, l, _ in window]
    return {"outcome": round(ret, 6), "mfe": round(max(fav), 6), "mae": round(min(adv), 6),
            "entry_date": entry[0].isoformat(), "exit_date": window[-1][0].isoformat()}


def _yf_bars(ticker: str, start: date, end: date) -> list[tuple[date, float, float, float]]:
    import yfinance as yf
    df = yf.Ticker(ticker).history(start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(),
                                   auto_adjust=True)
    if df is None or df.empty:
        return []
    return [(ix.date(), float(r["High"]), float(r["Low"]), float(r["Close"])) for ix, r in df.iterrows()]


def resolve_outcomes(*, today: date | None = None, ledger_dir: Path | None = None, path: Path | None = None,
                     bars_fn=None) -> int:
    """Fällige Horizonte auflösen (append-only, je (Beobachtung, Horizont) genau einmal)."""
    today = today or datetime.now(timezone.utc).date()
    path = path or OUTCOMES
    bars_fn = bars_fn or _yf_bars
    have = read_outcomes(path)
    due: dict[str, list[tuple[dict, int]]] = {}
    for r in read_rows(ledger_dir):
        d0 = date.fromisoformat(str(r["date"])[:10])
        for h in HORIZONS:
            if (r["observation_id"], h) not in have and d0 + timedelta(days=h) < today:
                due.setdefault(r["ticker"], []).append((r, h))
    n = 0
    for ticker, items in due.items():
        start = min(date.fromisoformat(str(r["date"])[:10]) for r, _ in items)
        end = max(date.fromisoformat(str(r["date"])[:10]) + timedelta(days=h) for r, h in items)
        try:
            bars = bars_fn(ticker, start, min(end, today))
        except Exception as e:  # noqa: BLE001 – Kursquelle aus: später erneut
            log.warning(f"final_mc_ledger: Kurse {ticker} nicht abrufbar ({e})")
            continue
        for r, h in items:
            res = path_outcome(bars, date.fromisoformat(str(r["date"])[:10]), h, r.get("direction"))
            if res is None:
                continue
            e = {"observation_id": r["observation_id"], "horizon": h, **res, "outcome_method": OUTCOME_METHOD,
                 "resolved_at": today.isoformat()}
            from modules.atomic_io import append_jsonl
            append_jsonl(path, [e], sort_keys=True)
            have[(r["observation_id"], h)] = e
            n += 1
    return n


# ── 4. Evidenz (nur dieselbe Population) ────────────────────────────────────
def _ts(x) -> datetime:
    d = datetime.fromisoformat(str(x).replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def primary_horizon(contract: dict) -> int:
    return int(contract.get("horizon_days") or 45)


def assign_clusters(obs: list[dict], gap_days: int = CLUSTER_GAP_DAYS) -> None:
    """Gleicher Ticker mit <= gap_days Abstand zum vorigen Signal = ein abhängiges Ereignis-Cluster."""
    last: dict[str, tuple[date, str]] = {}
    for o in sorted(obs, key=lambda x: (x["date"], x["ticker"])):
        d = date.fromisoformat(o["date"])
        prev = last.get(o["ticker"])
        cid = prev[1] if prev and (d - prev[0]).days <= gap_days else f"{o['ticker']}:{o['date']}"
        o["cluster"] = cid
        last[o["ticker"]] = (d, cid)


def observations(contract: dict, spec_hash: str, rows: list[dict], outcomes: dict, *,
                 horizon: int | None = None, since: datetime | None = None) -> list[dict]:
    k = hc.key(contract)
    h = horizon or primary_horizon(contract)
    fwd, reg = _ts(contract["forward_start"]), _ts(contract["registered_at"])
    out = []
    for r in rows:
        if r.get("stage") != STAGE:
            continue
        t = _ts(r["timestamp"])
        if t < fwd or t <= reg or (since is not None and t < since):
            continue
        ev = (r.get("contracts") or {}).get(k)
        if not ev or ev.get("spec_hash") != spec_hash or not ev.get("evaluable") or not ev.get("in_scope"):
            continue
        o = outcomes.get((r["observation_id"], h))
        if o is None or o.get("outcome") is None or o.get("outcome_method") != OUTCOME_METHOD:
            continue
        out.append({"decision_id": r["observation_id"], "date": str(r["date"])[:10], "ts": t, "ticker": r["ticker"],
                    "fired": bool(ev.get("fired")), "outcome": float(o["outcome"]),
                    "prob": r.get("champion_probability"), "sector": r.get("sector"), "regime": r.get("regime"),
                    "mfe": o.get("mfe"), "mae": o.get("mae"), "applied": False})
    assign_clusters(out)
    return sorted(out, key=lambda x: x["ts"])


def _tail(vals: list[float], q: float = 0.10) -> float | None:
    if len(vals) < 5:
        return None
    s = sorted(vals)
    k = max(1, int(math.floor(q * len(s))))
    return round(statistics.fmean(s[:k]), 5)


def _block_bootstrap(obs: list[dict], key: str, direction: int, n: int, seed: int, alpha: float):
    from modules import promotion_controller as pc
    blocks: dict[str, list[dict]] = {}
    for o in obs:
        blocks.setdefault(o[key], []).append(o)
    ids = sorted(blocks)
    if len(ids) < 2:
        return None, None
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        v = pc._delta([o for b in (rng.choice(ids) for _ in ids) for o in blocks[b]], direction)
        if v is not None:
            vals.append(v)
    if len(vals) < 0.9 * n:
        return None, None
    vals.sort()
    return (round(vals[max(0, int(alpha * len(vals)) - 1)], 5),
            round(vals[min(len(vals) - 1, int((1 - alpha) * len(vals)))], 5))


def evidence(contract: dict, spec_hash: str, rows: list[dict], outcomes: dict, policy: dict, alpha: float,
             since: datetime | None = None) -> dict:
    """Gleiche Struktur wie promotion_controller.evidence (damit insufficiency/decide greifen),
    plus Cluster, Tail, MFE/MAE je Gruppe und alle Horizonte (nur beschreibend)."""
    from modules import promotion_controller as pc
    direction = int(contract["direction"])
    st = policy.get("statistics") or {}
    obs = observations(contract, spec_hash, rows, outcomes, since=since)
    fired = [o for o in obs if o["fired"]]
    notf = [o for o in obs if not o["fired"]]
    ev: dict = {"stage": STAGE, "data_kind": "prospective_forward", "horizon_days": primary_horizon(contract),
                "n_observations": len(obs), "n_independent_dates": len({o["date"] for o in obs}),
                "n_event_clusters": len({o["cluster"] for o in obs}),
                "calendar_span_days": (obs[-1]["ts"] - obs[0]["ts"]).days if len(obs) > 1 else 0,
                "first_observation": obs[0]["date"] if obs else None,
                "last_observation": obs[-1]["date"] if obs else None,
                "n_fired": len(fired), "n_fired_independent_dates": len({o["date"] for o in fired}),
                "n_fired_event_clusters": len({o["cluster"] for o in fired})}

    def grp(g):
        m = pc.group_metrics([o["outcome"] for o in g], None, [o["mfe"] for o in g], [o["mae"] for o in g])
        m["downside_tail_10pct"] = _tail([o["outcome"] for o in g])
        return m
    ev["all"], ev["fired"], ev["not_fired"] = grp(obs), grp(fired), grp(notf)
    ev["policy"] = grp(notf if direction < 0 else fired)
    d = pc._delta(obs, direction)
    ev["delta_expectancy"] = round(d, 5) if d is not None else None
    ev["triggered_minus_non_triggered"] = (round(ev["fired"]["expectancy"] - ev["not_fired"]["expectancy"], 5)
                                           if fired and notf else None)
    bn, seed = int(st.get("bootstrap_n", 2000)), int(st.get("bootstrap_seed", 41))
    ci_dates = _block_bootstrap(obs, "date", direction, bn, seed, alpha) if obs else (None, None)
    ci_clusters = _block_bootstrap(obs, "cluster", direction, bn, seed, alpha) if obs else (None, None)
    ev["ci_by_date"], ev["ci_by_cluster"] = list(ci_dates), list(ci_clusters)
    # konservativ: engere Aussage gilt nur, wenn BEIDE Blockungen sie tragen
    lows = [x for x in (ci_dates[0], ci_clusters[0])]
    highs = [x for x in (ci_dates[1], ci_clusters[1])]
    ev["ci"] = [None if None in lows else min(lows), None if None in highs else max(highs)]
    ev["outlier_trims"] = pc._outlier_trims(obs, direction)
    win = []
    if len(obs) > 1:
        t0, t1 = obs[0]["ts"], obs[-1]["ts"]
        step = (t1 - t0) / 3
        for i in range(3):
            a, b = t0 + step * i, t0 + step * (i + 1)
            part = [o for o in obs if (a <= o["ts"] < b) or (i == 2 and o["ts"] == t1)]
            v = pc._delta(part, direction)
            win.append(round(v, 5) if v is not None else None)
    ev["time_windows"] = win
    ev["regimes"] = sorted({o["regime"] for o in obs if o["regime"]})
    secs = [o["sector"] for o in fired if o["sector"]]
    ev["max_sector_share_fired"] = round(max(secs.count(s) for s in set(secs)) / len(secs), 3) if secs else None
    ev["dominant_sector_fired"] = max(set(secs), key=secs.count) if secs else None
    if direction < 0:
        ev["abstention"] = {"n_blocked": len(fired),
                            "avoided_loss_potential": round(-sum(o["outcome"] for o in fired if o["outcome"] < 0), 4),
                            "missed_gain": round(sum(o["outcome"] for o in fired if o["outcome"] > 0), 4),
                            "net_value_of_abstention": round(-sum(o["outcome"] for o in fired) / len(obs), 5)
                            if obs else None,
                            "precision": round(sum(1 for o in fired if o["outcome"] <= 0) / len(fired), 4)
                            if fired else None}
    ev["secondary_horizons"] = {}
    for h in HORIZONS:
        if h == ev["horizon_days"]:
            continue
        o2 = observations(contract, spec_hash, rows, outcomes, horizon=h, since=since)
        f2 = [o["outcome"] for o in o2 if o["fired"]]
        n2 = [o["outcome"] for o in o2 if not o["fired"]]
        ev["secondary_horizons"][str(h)] = {"n": len(o2), "fired_expectancy": round(statistics.fmean(f2), 5) if f2
                                            else None, "non_fired_expectancy": round(statistics.fmean(n2), 5)
                                            if n2 else None}
    return ev


def insufficiency_extra(contract: dict, ev: dict) -> list[str]:
    """Cluster-Anforderungen der FINAL_MC-Verträge (zusätzlich zu den gemeinsamen Untergrenzen):
    viele korrelierte Beobachtungen ersetzen keine unabhängigen Ereignisse."""
    need = []
    m = contract.get("minimum_independent_event_clusters")
    if m is not None and ev.get("n_event_clusters", 0) < int(m):
        need.append(f"Ereignis-Cluster {ev.get('n_event_clusters', 0)}/{m}")
    mf = (contract.get("promotion_criteria") or {}).get("min_fired_event_clusters")
    if mf is not None and ev.get("n_fired_event_clusters", 0) < int(mf):
        need.append(f"Treffer-Cluster {ev.get('n_fired_event_clusters', 0)}/{mf}")
    return need


# ── Übersicht (Report) ──────────────────────────────────────────────────────
def population_summary(rows: list[dict] | None = None, downstream: list[dict] | None = None,
                       since: str | None = None) -> dict:
    rows = rows if rows is not None else read_rows()
    downstream = downstream if downstream is not None else read_jsonl(DOWNSTREAM)
    if since:
        rows = [r for r in rows if str(r["date"]) >= since]
    obs = [{"ticker": r["ticker"], "date": str(r["date"])[:10]} for r in rows]
    assign_clusters(obs)
    ds = {d["observation_id"]: d for d in downstream}
    gates: dict[str, int] = {}
    for r in rows:
        for g in ((ds.get(r["observation_id"]) or {}).get("fail_gates") or {}).values():
            if g:
                gates[g] = gates.get(g, 0) + 1
    months = sorted({r["date"][:7] for r in rows})
    return {"n": len(rows), "event_clusters": len({o["cluster"] for o in obs}),
            "dates": len({o["date"] for o in obs}), "months": len(months),
            "per_month": round(len(rows) / len(months), 1) if months else None,
            "champion_trades": sum(1 for r in rows if (ds.get(r["observation_id"]) or {}).get("champion_decision")
                                   == "TRADE"),
            "roi_fail_gates": gates}

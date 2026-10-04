"""modules/universe_v2_ledger.py – Shadow-Ledger der Population V2_CANDIDATE (UNIVERSE_V2).

V2-Kandidaten entstehen LLM-frei aus den nicht-LLM Champion-Stufen (Relative Volume VIX-skaliert,
News-Präsenz, Pre-MC sigma-only) auf RESEARCH_UNIVERSE_V2 (modules/universe_v2.py). Jeder Kandidat
wird mit universe_version, Market Cap + Bucket, Liquidität, Options-Liquidität, Spreads, geschätzter
Slippage/Ausführungskosten, execution_quality, Risikoflags, SystemState und eingefrorener Auswertung
der V2-Segment-Verträge gespeichert. Kein Kandidat wird dadurch zum Trade.

Outcomes je Horizont (20/45/60 T): raw_underlying_return, MFE/MAE, option_theoretical_return,
estimated_spread_cost, estimated_slippage, estimated_execution_cost, net_realizable_return,
liquid_option_available, outcome_status (inkl. DELISTED_WORST_CASE, CORPORATE_ACTION).

Evidenz (Segment-Verträge): ausschließlich net_realizable_return; Segment (z. B. SMALL, nicht in V1)
gegen die V1-Referenz derselben Population (in_universe_v1 = 1). Nie mit V1-Champion- oder
Final-MC-Evidenz verrechnet.
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

from modules import final_mc_ledger as fml
from modules import hypothesis_contract as hc
from modules import universe_v2 as uv

log = logging.getLogger(__name__)

STAGE = "V2_CANDIDATE"
DIR = Path("outputs/universe/v2_ledger")
OUTCOMES = Path("outputs/universe/v2_outcomes.jsonl")
RECOMMENDATIONS = Path("outputs/universe/v2_recommendations")
EVIDENCE_STATUSES = ("QUOTED_EXIT", "ESTIMATED_EXIT_SPREAD", "DELISTED_WORST_CASE")
TOP_K = 3


def v2_contracts(contracts: list[dict] | None = None) -> list[dict]:
    return [c for c in (contracts if contracts is not None else hc.load()) if fml.stage_of(c) == STAGE]


# ── 1. Kandidaten aufzeichnen ───────────────────────────────────────────────
def record_candidates(cands: list[dict], *, today: str, vix=None, ctx: dict | None = None,
                      contracts: list[dict] | None = None, registry: Path | None = None,
                      ledger_dir: Path | None = None, now: datetime | None = None,
                      v1_members: set[str] | None = None) -> list[dict]:
    """cands: {ticker, assessment (universe_v2.assess), hit_rate, rel_volume, news_count, direction}.
    Idempotent je (Tag, Ticker); Vertragsauswertung eingefroren."""
    from modules import production_intelligence_adapter as pia
    now = now or datetime.now(timezone.utc)
    ledger_dir = ledger_dir or DIR
    valid = fml._registered(v2_contracts(contracts), registry)
    if ctx is None:
        ctx = pia.research_context(today)
    regime = pia.regime_label(vix)
    v1 = v1_members if v1_members is not None else set()
    seen = {(str(r["date"])[:10], r["ticker"]) for r in fml.read_rows(ledger_dir) if str(r["date"])[:7] == today[:7]}
    rows = []
    for i, c in enumerate(cands):
        t, a = c.get("ticker"), c.get("assessment") or {}
        if not t or (today, t) in seen or not a.get("research_ok"):
            continue
        seen.add((today, t))
        env = {"market_cap_bucket": a.get("market_cap_bucket"), "in_universe_v1": 1 if t in v1 else 0,
               "liquidity_bucket": a.get("liquidity_bucket"), "options_liquidity_bucket": a.get("options_liquidity_bucket"),
               "execution_quality": a.get("execution_quality"), "tradeable_v2": 1 if a.get("tradeable_ok") else 0,
               "hit_rate": c.get("hit_rate"), "vix": vix, "safe_mode_active": ctx.get("safe_mode_active")}
        try:                                         # Commodity × Market-Cap-Bucket prospektiv testbar (RESEARCH)
            from modules import commodity_intelligence as _cmd
            cmd_env = _cmd.decision_features(t, ctx.get("commodity"), ticker_known=True)
        except Exception:  # noqa: BLE001 – optional
            cmd_env = {}
        env.update({k: v for k, v in cmd_env.items() if k != "commodity_data_version"})
        trig = {}
        for k, ct in valid.items():
            f = hc.fires(ct, env)
            trig[k] = {"spec_hash": hc.spec_hash(ct), "in_scope": True, "evaluable": f is not None, "fired": bool(f)}
        oid = hashlib.sha256(f"{STAGE}|{today}|{t}|{now.isoformat()}|{i}".encode()).hexdigest()[:16]
        rows.append({
            "observation_id": oid, "stage": STAGE, "universe_version": uv.UNIVERSE_V2,
            "timestamp": now.isoformat(timespec="seconds"), "date": today, "ticker": t, "regime": regime, "vix": vix,
            "direction": c.get("direction") or "BULLISH", "in_universe_v1": env["in_universe_v1"],
            "hit_rate": c.get("hit_rate"), "rel_volume": c.get("rel_volume"), "news_count": c.get("news_count"),
            "market_cap": a.get("market_cap"), "market_cap_source": a.get("market_cap_source"),
            "market_cap_bucket": a.get("market_cap_bucket"), "underlying_liquidity": a.get("underlying_liquidity"),
            "liquidity_bucket": a.get("liquidity_bucket"), "options_liquidity": a.get("options_liquidity"),
            "options_liquidity_bucket": a.get("options_liquidity_bucket"), "spread_metrics": a.get("spread_metrics"),
            "estimated_slippage": a.get("estimated_slippage"), "execution_cost": a.get("execution_cost"),
            "execution_quality": a.get("execution_quality"), "tradeable_v2": bool(a.get("tradeable_ok")),
            "tradeable_fail": a.get("tradeable_fail"), "risk_flags": a.get("risk_flags"),
            "reference_option": a.get("reference_option"),
            "commodity": {"data_version": cmd_env.get("commodity_data_version"),
                          "exposure": {k: v for k, v in cmd_env.items() if k.startswith("cmdexp_") and v},
                          "features": {k: v for k, v in cmd_env.items() if k.startswith("cmd_") and v is not None}},
            "reference_price": (a.get("underlying_liquidity") or {}).get("price"),
            "system_state": {"version": ctx.get("system_state_version"), "safe_mode": ctx.get("safe_mode_active"),
                             "drift_level": ctx.get("drift_level")},
            "contracts": trig, "production_effect": "NONE"})
    if rows:
        from modules.atomic_io import append_jsonl
        append_jsonl(ledger_dir / f"{today[:7]}.jsonl", rows, sort_keys=True, ensure_ascii=False)
    return rows


# ── 2. Outcomes (verzögert, theoretisch + netto) ────────────────────────────
def read_outcomes(path: Path | None = None) -> dict[tuple[str, int], dict]:
    return fml.read_outcomes(path or OUTCOMES)


def _yf_bars_and_splits(ticker: str, start: date, end: date):
    import yfinance as yf
    df = yf.Ticker(ticker).history(start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(),
                                   auto_adjust=True, actions=True)
    if df is None or df.empty:
        return [], []
    bars = [(ix.date(), float(r["High"]), float(r["Low"]), float(r["Close"])) for ix, r in df.iterrows()]
    splits = [ix.date() for ix, r in df.iterrows() if float(r.get("Stock Splits") or 0) not in (0.0, 1.0)]
    return bars, splits


def _option_close(symbol: str, on: date) -> float | None:
    """Tageschluss der Option (Tradier-Historie) am letzten Handelstag <= on."""
    js = uv._tradier("markets/history", {"symbol": symbol, "interval": "daily",
                                         "start": (on - timedelta(days=7)).isoformat(), "end": on.isoformat()})
    days = ((js or {}).get("history") or {}) or {}
    d = days.get("day") or []
    d = [d] if isinstance(d, dict) else d
    return float(d[-1]["close"]) if d and d[-1].get("close") is not None else None


def resolve_outcomes(*, today: date | None = None, ledger_dir: Path | None = None, path: Path | None = None,
                     bars_fn=None, option_close_fn=None, cfg: dict | None = None) -> int:
    """Append-only, je (Beobachtung, Horizont) genau einmal. Fehlende Kurse nach Horizont + Titel im
    letzten Snapshot STALE/fehlend -> Delisting (konservativ). Split im Fenster -> CORPORATE_ACTION
    (Optionskontrakt angepasst: Netto-Outcome nicht berechenbar, gezählt, nicht als Evidenz)."""
    today = today or datetime.now(timezone.utc).date()
    path = path or OUTCOMES
    cfg = cfg or uv.load_cfg()
    bars_fn = bars_fn or _yf_bars_and_splits
    option_close_fn = option_close_fn or _option_close
    horizons = [int(h) for h in cfg["outcomes"]["horizons_days"]]
    have = read_outcomes(path)
    snap = uv.latest_snapshot()
    stale = {r["ticker"] for r in (snap or {}).get("records") or [] if r.get("status") == "STALE"}
    n = 0
    for r in fml.read_rows(ledger_dir or DIR):
        d0 = date.fromisoformat(str(r["date"])[:10])
        due = [h for h in horizons if (r["observation_id"], h) not in have and d0 + timedelta(days=h) < today]
        if not due:
            continue
        try:
            bars, splits = bars_fn(r["ticker"], d0, min(today, d0 + timedelta(days=max(due))))
        except Exception as e:  # noqa: BLE001 – Kursquelle aus: später erneut
            log.warning(f"v2_ledger: Kurse {r['ticker']} nicht abrufbar ({e})")
            continue
        ref = r.get("reference_option") or {}
        for h in due:
            end = d0 + timedelta(days=h)
            po = fml.path_outcome(bars, d0, h, r.get("direction"))
            delisted = po is None and r["ticker"] in stale and (not bars or bars[-1][0] < end)
            if po is None and not delisted:
                continue
            ca = any(d0 < s <= end for s in splits)
            exit_mid = None
            if ref.get("symbol") and not ca and not delisted:
                try:
                    exit_mid = option_close_fn(ref["symbol"], end)
                except Exception as e:  # noqa: BLE001
                    log.debug(f"v2_ledger: Optionshistorie {ref.get('symbol')} ({e})")
            net = uv.net_outcome(raw_underlying_return=(po or {}).get("outcome"), entry_bid=ref.get("bid"),
                                 entry_ask=ref.get("ask"), exit_mid=exit_mid, open_interest=ref.get("open_interest"),
                                 cfg=cfg, delisted=delisted)
            if ca:
                net.update(outcome_status="CORPORATE_ACTION", net_realizable_return=None, option_theoretical_return=None)
            e = {"observation_id": r["observation_id"], "horizon": h, "mfe": (po or {}).get("mfe"),
                 "mae": (po or {}).get("mae"), **net, "resolved_at": today.isoformat()}
            from modules.atomic_io import append_jsonl
            append_jsonl(path, [e], sort_keys=True)
            have[(r["observation_id"], h)] = e
            n += 1
    return n


# ── 3. Evidenz je Segment (nur netto, Segment vs. V1-Referenz) ──────────────
def observations(contract: dict, spec_hash: str, rows: list[dict], outcomes: dict, *, since: datetime | None = None,
                 horizon: int | None = None) -> list[dict]:
    k = hc.key(contract)
    h = horizon or int(contract.get("horizon_days") or 45)
    fwd, reg = fml._ts(contract["forward_start"]), fml._ts(contract["registered_at"])
    out = []
    for r in rows:
        if r.get("stage") != STAGE or uv.universe_version_of(r) != uv.UNIVERSE_V2:
            continue
        t = fml._ts(r["timestamp"])
        if t < fwd or t <= reg or (since is not None and t < since):
            continue
        ev = (r.get("contracts") or {}).get(k)
        if not ev or ev.get("spec_hash") != spec_hash or not ev.get("evaluable"):
            continue
        o = outcomes.get((r["observation_id"], h))
        if o is None or o.get("outcome_status") not in EVIDENCE_STATUSES or o.get("net_realizable_return") is None:
            continue
        out.append({"date": str(r["date"])[:10], "ts": t, "ticker": r["ticker"], "fired": bool(ev.get("fired")),
                    "reference": bool(r.get("in_universe_v1")), "net": float(o["net_realizable_return"]),
                    "theo": o.get("option_theoretical_return"), "raw": o.get("raw_underlying_return"),
                    "spread": o.get("estimated_spread_cost"), "slippage": o.get("estimated_slippage"),
                    "cost": o.get("estimated_execution_cost"), "mfe": o.get("mfe"), "mae": o.get("mae"),
                    "prob": r.get("hit_rate"), "regime": r.get("regime"), "sector": None,
                    "quality": r.get("execution_quality"), "delisted": o.get("outcome_status") == "DELISTED_WORST_CASE"})
    fml.assign_clusters(out)
    return sorted(out, key=lambda x: x["ts"])


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.fmean(xs), 6) if xs else None


def _metrics(g: list[dict]) -> dict:
    from modules import promotion_controller as pc
    m = pc.group_metrics([o["net"] for o in g], [o["prob"] for o in g], [o["mfe"] for o in g], [o["mae"] for o in g])
    if not g:
        return m
    m.update(net_expectancy=m.get("expectancy"), raw_expectancy=_mean(o["raw"] for o in g),
             theoretical_expectancy=_mean(o["theo"] for o in g), avg_spread=_mean(o["spread"] for o in g),
             avg_slippage=_mean(o["slippage"] for o in g), avg_execution_cost=_mean(o["cost"] for o in g),
             downside_tail_10pct=fml._tail([o["net"] for o in g]), trade_count=len(g),
             n_delisted=sum(1 for o in g if o["delisted"]),
             execution_quality_ok_share=round(sum(1 for o in g if o["quality"] in ("GOOD", "FAIR")) / len(g), 4),
             precision_at_k=precision_at_k(g))
    return m


def precision_at_k(g: list[dict], k: int = TOP_K) -> float | None:
    """Je Tag die k Kandidaten mit höchster Modellwahrscheinlichkeit: Anteil mit Netto > 0."""
    by: dict[str, list[dict]] = {}
    for o in g:
        if o["prob"] is not None:
            by.setdefault(o["date"], []).append(o)
    top = [o for d in by.values() for o in sorted(d, key=lambda x: -x["prob"])[:k]]
    return round(sum(1 for o in top if o["net"] > 0) / len(top), 4) if top else None


def _boot_mean(vals_by_block: dict[str, list[float]], n: int, seed: int, alpha: float):
    ids = sorted(vals_by_block)
    if len(ids) < 2:
        return None, None
    rng, out = random.Random(seed), []
    for _ in range(n):
        s = [v for b in (rng.choice(ids) for _ in ids) for v in vals_by_block[b]]
        if s:
            out.append(statistics.fmean(s))
    out.sort()
    return (round(out[max(0, int(alpha * len(out)) - 1)], 6), round(out[min(len(out) - 1, int((1 - alpha) * len(out)))], 6))


def evidence(contract: dict, spec_hash: str, rows: list[dict], outcomes: dict, policy: dict, alpha: float,
             since: datetime | None = None) -> dict:
    """Struktur kompatibel zu promotion_controller.decide: 'fired' = Segment, 'all' = V1-Referenz,
    delta_expectancy = Netto-Expectancy des Segments (absolut), CI über Signaltage UND Cluster."""
    st = policy.get("statistics") or {}
    obs = observations(contract, spec_hash, rows, outcomes, since=since)
    seg = [o for o in obs if o["fired"]]
    ref = [o for o in obs if o["reference"] and not o["fired"]]
    ev: dict = {"stage": STAGE, "universe_version": uv.UNIVERSE_V2, "data_kind": "prospective_forward",
                "outcome_basis": "net_realizable_return", "horizon_days": int(contract.get("horizon_days") or 45),
                "n_observations": len(seg), "n_independent_dates": len({o["date"] for o in seg}),
                "n_event_clusters": len({o["cluster"] for o in seg}),
                "calendar_span_days": (seg[-1]["ts"] - seg[0]["ts"]).days if len(seg) > 1 else 0,
                "first_observation": seg[0]["date"] if seg else None, "last_observation": seg[-1]["date"] if seg else None,
                "n_fired": len(seg), "n_fired_independent_dates": len({o["date"] for o in seg}),
                "n_fired_event_clusters": len({o["cluster"] for o in seg}), "n_reference": len(ref)}
    ev["fired"] = ev["policy"] = _metrics(seg)
    ev["all"] = ev["not_fired"] = ev["reference_v1"] = _metrics(ref)
    ev["delta_expectancy"] = ev["fired"].get("net_expectancy")
    ev["delta_vs_v1_reference"] = (round(ev["fired"]["net_expectancy"] - ev["all"]["net_expectancy"], 6)
                                   if seg and ref else None)
    bn, seed = int(st.get("bootstrap_n", 2000)), int(st.get("bootstrap_seed", 41))
    by_d, by_c = {}, {}
    for o in seg:
        by_d.setdefault(o["date"], []).append(o["net"])
        by_c.setdefault(o["cluster"], []).append(o["net"])
    cd, cc = _boot_mean(by_d, bn, seed, alpha), _boot_mean(by_c, bn, seed, alpha)
    ev["ci_by_date"], ev["ci_by_cluster"] = list(cd), list(cc)
    lows, highs = [cd[0], cc[0]], [cd[1], cc[1]]
    ev["ci"] = [None if None in lows else min(lows), None if None in highs else max(highs)]
    nets = sorted((o["net"] for o in seg), reverse=True)
    ev["outlier_trims"] = {name: (round(statistics.fmean(nets[k:]), 6) if len(nets) > k else None)
                           for name, k in (("top1", 1), ("top3", 3), ("top5pct", max(1, math.ceil(0.05 * len(nets)))))}
    win = []
    if len(seg) > 1:
        t0, t1 = seg[0]["ts"], seg[-1]["ts"]
        step = (t1 - t0) / 3
        for i in range(3):
            a, b = t0 + step * i, t0 + step * (i + 1)
            part = [o["net"] for o in seg if (a <= o["ts"] < b) or (i == 2 and o["ts"] == t1)]
            win.append(round(statistics.fmean(part), 6) if part else None)
    ev["time_windows"] = win
    ev["regimes"] = sorted({o["regime"] for o in seg if o["regime"]})
    ev["max_sector_share_fired"] = None
    return ev


def segment_checks(contract: dict, ev: dict) -> list[str]:
    """Zusätzliche, vorab im Vertrag fixierte Kriterien (nur Segment-Verträge)."""
    pc_ = contract.get("promotion_criteria") or {}
    seg, ref = ev.get("fired") or {}, ev.get("all") or {}
    fails = []
    tol = pc_.get("precision_at_k_not_worse_by")
    if tol is not None:
        ps, pr = seg.get("precision_at_k"), ref.get("precision_at_k")
        if ps is None or (pr is not None and ps < pr - float(tol)):
            fails.append(f"Precision@{TOP_K} {ps} schlechter als V1-Referenz {pr} - {tol}")
    tol = pc_.get("calibration_not_worse_by")
    if tol is not None:
        bs, br = seg.get("brier"), ref.get("brier")
        if bs is None or (br is not None and bs > br + float(tol)):
            fails.append(f"Kalibrierung (Brier) {bs} schlechter als V1-Referenz {br} + {tol}")
    m = pc_.get("min_execution_quality_share")
    if m is not None and (seg.get("execution_quality_ok_share") or 0) < float(m):
        fails.append(f"Execution-Qualität GOOD/FAIR {seg.get('execution_quality_ok_share')} < {m}")
    tol = pc_.get("tail_not_worse_than_reference_by")
    if tol is not None:
        ts_, tr_ = seg.get("downside_tail_10pct"), ref.get("downside_tail_10pct")
        if ts_ is None or (tr_ is not None and ts_ < tr_ - float(tol)):
            fails.append(f"Tail-Risiko {ts_} schlechter als V1-Referenz {tr_} - {tol}")
    return fails


def segment_demotion_reasons(contract: dict, post_ev: dict) -> list[str]:
    dc = contract.get("demotion_criteria") or {}
    seg = post_ev.get("fired") or {}
    r = []
    if "min_net_expectancy" in dc and seg.get("net_expectancy") is not None and seg["net_expectancy"] < float(dc["min_net_expectancy"]):
        r.append(f"Netto-Expectancy {seg['net_expectancy']} < {dc['min_net_expectancy']}")
    if "max_avg_slippage" in dc and seg.get("avg_slippage") is not None and seg["avg_slippage"] > float(dc["max_avg_slippage"]):
        r.append(f"Slippage {seg['avg_slippage']} > {dc['max_avg_slippage']}")
    if "min_execution_quality_share" in dc and (seg.get("execution_quality_ok_share") or 0) < float(dc["min_execution_quality_share"]):
        r.append(f"Execution-Qualität {seg.get('execution_quality_ok_share')} < {dc['min_execution_quality_share']}")
    if "min_precision_at_k" in dc and seg.get("precision_at_k") is not None and seg["precision_at_k"] < float(dc["min_precision_at_k"]):
        r.append(f"Precision@{TOP_K} {seg['precision_at_k']} < {dc['min_precision_at_k']}")
    if "max_drawdown_min" in dc and seg.get("max_drawdown") is not None and seg["max_drawdown"] < float(dc["max_drawdown_min"]):
        r.append(f"MaxDD {seg['max_drawdown']} < {dc['max_drawdown_min']}")
    return r


# ── 4. Empfehlungs-Gate (Trade-Mail) ────────────────────────────────────────
def segment_of(row: dict) -> str:
    return f"{row.get('market_cap_bucket')}{'' if not row.get('in_universe_v1') else '_V1'}"


def recommendations(rows: list[dict], levels: dict[str, str], contracts: list[dict]) -> list[dict]:
    """Nur Kandidaten, deren Segment-Vertrag TRADE_RECOMMENDATION_ENABLED erreicht hat UND die die
    Produktions-Gates (tradeable_v2) ohne Risikoflag bestehen. Sonst leer (Standard)."""
    enabled = {hc.key(c) for c in contracts if levels.get(hc.key(c)) == "TRADE_RECOMMENDATION_ENABLED"}
    out = []
    for r in rows:
        hits = [k for k, v in (r.get("contracts") or {}).items() if k in enabled and v.get("fired")]
        if hits and r.get("tradeable_v2") and not r.get("risk_flags"):
            out.append({**r, "enabled_by": hits})
    return out


# ── 5. Bericht ──────────────────────────────────────────────────────────────
def bucket_summary(rows: list[dict] | None = None, outcomes: dict | None = None, horizon: int = 45) -> dict:
    """Beschreibend je Market-Cap-Bucket (alle Daten, auch vor forward_start, klar gekennzeichnet)."""
    rows = rows if rows is not None else fml.read_rows(DIR)
    outcomes = outcomes if outcomes is not None else read_outcomes()
    by: dict[str, list] = {}
    for r in rows:
        if r.get("stage") != STAGE:
            continue
        o = outcomes.get((r["observation_id"], horizon)) or {}
        by.setdefault(r.get("market_cap_bucket") or "UNKNOWN", []).append((r, o))
    out = {}
    for b, items in sorted(by.items()):
        nets = [o["net_realizable_return"] for _, o in items if o.get("net_realizable_return") is not None]
        out[b] = {"signals": len(items), "tradeable": sum(1 for r, _ in items if r.get("tradeable_v2")),
                  "resolved": len(nets), "net_expectancy": _mean(nets),
                  "avg_option_spread": _mean((r.get("spread_metrics") or {}).get("option_spread_rel") for r, _ in items),
                  "avg_slippage": _mean(r.get("estimated_slippage") for r, _ in items),
                  "avg_execution_cost": _mean(r.get("execution_cost") for r, _ in items)}
    return out

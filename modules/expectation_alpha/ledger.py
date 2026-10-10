"""modules/expectation_alpha/ledger.py – append-only EA-Ledger, Outcomes und Promotion-Evidenz.

Dateien (outputs/expectation_alpha/):
  context/YYYY-MM.jsonl         ein Kontext je Lauf (Gaps, Regime, Signale, Provenienz)
  candidates/YYYY-MM-DD.jsonl   eine These je Kandidat und Tag. Vertragsauswertungen sind zum
                                Entscheidungszeitpunkt eingefroren.
  downstream.jsonl              was der Champion danach tat (rein beschreibend)
  outcomes.jsonl                je (Beobachtung, Expression, Variante, Horizont) genau einmal
  runs.jsonl                    Lauf-Protokoll (Zählungen, Verteilungen, Fehler, Laufzeit)

Outcomes werden nur in outcomes.jsonl geschrieben, nie in den Entscheidungsdatensatz. Die Entscheidung ist
vor dem ersten Outcome-Tag eingefroren.
Horizonte in Handelstagen. Einstieg ist der erste US-Schlusskurs nach der Entscheidung
(`entry_not_before`): Entscheidung vor 16:00 New York -> Schluss des Entscheidungstags, sonst Folgetag.
Varianten:
  immediate     Einstieg am Entscheidungstag
  triggered     nur WAIT, nur UNDERLYING: Der eingefrorene Trigger wird je Schlusskurs geprüft, nur mit bis
                dahin bekannten Kursen. Einstieg zum Schluss des Folgetags nach dem ersten Triggertag;
                gleicher Ausstiegstag wie immediate. Ohne Trigger: NO_ENTRY mit Rendite 0, also Cash,
                gekennzeichnet.
  kill_managed  Ausstieg beim ersten eingefrorenen Kill-Ereignis, sonst am Horizont.
"""
from __future__ import annotations

import logging
import math
import statistics
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from modules import final_mc_ledger as fml
from modules import hypothesis_contract as hc
from modules.atomic_io import append_jsonl, read_jsonl
from modules.expectation_alpha import cross_asset_confirmation as cac
from modules.expectation_alpha import timing as tm
from modules.expectation_alpha.schemas import ERROR, STAGE, WAIT, rnd

log = logging.getLogger(__name__)

ROOT = Path("outputs/expectation_alpha")
OUTCOME_METHOD = "ea_close_to_close_v1"
OUTCOME_KINDS = ("fired_vs_not_fired", "paired_wait_delta", "paired_expression_delta")
VARIANTS = ("immediate", "triggered", "kill_managed")


def paths(root: Path | None = None) -> dict[str, Path]:
    r = Path(root or ROOT)
    return {"root": r, "context": r / "context", "candidates": r / "candidates", "downstream": r / "downstream.jsonl",
            "outcomes": r / "outcomes.jsonl", "runs": r / "runs.jsonl", "evaluation_json": r / "evaluation.json",
            "evaluation_md": r / "evaluation.md", "proposals": r / "proposals.json"}


# ── Schreiben ───────────────────────────────────────────────────────────────
def write_context(ctx: dict, root: Path | None = None) -> bool:
    p = paths(root)["context"] / f"{ctx['date'][:7]}.jsonl"
    if any(r.get("context_id") == ctx["context_id"] for r in read_jsonl(p)):
        return False
    append_jsonl(p, [ctx], sort_keys=True, ensure_ascii=False, default=str)
    return True


def read_contexts(root: Path | None = None) -> list[dict]:
    d = paths(root)["context"]
    out = []
    if d.exists():
        for f in sorted(d.glob("*.jsonl")):
            out += read_jsonl(f)
    return sorted(out, key=lambda r: r.get("decision_time") or "")


def registered_contracts(contracts: list[dict] | None = None, registry: Path | None = None) -> dict[str, dict]:
    """EA-Verträge, deren spec_hash in der Registry steht (unregistriert/verändert -> keine Auswertung)."""
    cs = fml.stage_contracts(contracts if contracts is not None else hc.load(), STAGE)
    return fml._registered(cs, registry)


def in_population(c: dict, env: dict) -> bool:
    """population_filter (sicherer AST). Fehlende Merkmale -> nicht in der Population (nie 0)."""
    f = c.get("population_filter") or "1"
    try:
        return bool(hc._ev(hc.parse_signal(f), env))
    except (KeyError, TypeError, ZeroDivisionError, hc.ContractError):
        return False


def freeze_contracts(row: dict, valid: dict[str, dict]) -> dict:
    env = row.get("env") or {}
    out = {}
    for k, c in valid.items():
        scope = hc.in_scope(c, row.get("sector"), row.get("regime")) and in_population(c, env)
        f = hc.fires(c, env) if scope else None
        out[k] = {"spec_hash": hc.spec_hash(c), "in_scope": bool(scope), "evaluable": f is not None,
                  "fired": bool(f), "signal_value": hc.evaluate_signal(c, env) if scope else None,
                  "outcome_kind": c.get("ea_outcome_kind")}
    return out


def record_candidates(rows: list[dict], *, root: Path | None = None, contracts: list[dict] | None = None,
                      registry: Path | None = None) -> list[dict]:
    """Append-only, idempotent je observation_id (wiederholter Lauf erzeugt keine Pseudo-Stichprobe)."""
    if not rows:
        return []
    valid = registered_contracts(contracts, registry)
    p = paths(root)["candidates"] / f"{rows[0]['date']}.jsonl"
    seen = {r.get("observation_id") for r in read_jsonl(p)}
    new = []
    for r in rows:
        if r["observation_id"] in seen:
            continue
        seen.add(r["observation_id"])
        r["contracts"] = freeze_contracts(r, valid)
        new.append(r)
    if new:
        append_jsonl(p, new, sort_keys=True, ensure_ascii=False, default=str)
    return new


def record_downstream(observations: list[dict], outcome_by_ticker: dict[str, dict], *, root: Path | None = None,
                      now: datetime | None = None) -> int:
    if not observations:
        return 0
    return fml.record_downstream(observations, outcome_by_ticker, path=paths(root)["downstream"], now=now)


def record_run(entry: dict, root: Path | None = None) -> None:
    append_jsonl(paths(root)["runs"], [entry], sort_keys=True, ensure_ascii=False, default=str)


def read_rows(root: Path | None = None) -> list[dict]:
    d = paths(root)["candidates"]
    rows = []
    if d.exists():
        for f in sorted(d.glob("*.jsonl")):
            rows += read_jsonl(f)
    return rows


def read_outcomes(root: Path | None = None) -> dict[tuple, dict]:
    out: dict[tuple, dict] = {}
    for e in read_jsonl(paths(root)["outcomes"]):
        out.setdefault((e["observation_id"], e["expression"], e["variant"], int(e["horizon"])), e)  # erstes gilt
    return out


# ── Outcomes ────────────────────────────────────────────────────────────────
def _path(bars: list, i0: int, i1: int, sign: float, cost: float) -> dict:
    p0 = bars[i0][3]
    seg = bars[i0 + 1: i1 + 1]
    ret = sign * (bars[i1][3] / p0 - 1.0)
    fav = [sign * (((h if sign > 0 else lo) or c) / p0 - 1.0) for _, h, lo, c in seg]
    adv = [sign * (((lo if sign > 0 else h) or c) / p0 - 1.0) for _, h, lo, c in seg]
    eq = [sign * (c / p0 - 1.0) for *_, c in seg]
    peak, mdd = 0.0, 0.0
    for e in eq:
        peak = max(peak, e)
        mdd = min(mdd, e - peak)
    return {"outcome": round(ret, 6), "outcome_net": round(ret - 2 * cost, 6),
            "mfe": round(max(fav), 6) if fav else 0.0, "mae": round(min(adv), 6) if adv else 0.0,
            "max_drawdown": round(mdd, 6), "entry_date": bars[i0][0].isoformat(),
            "exit_date": bars[i1][0].isoformat(), "holding_days": i1 - i0}


def _frozen_spec(row: dict) -> list[dict]:
    return [{"signal": s["signal"], "expected": s["expected"], "deadband": s["deadband"],
             "weight": s.get("weight", 1.0)} for s in (row.get("cross_asset_confirmation") or {}).get("signals") or []]


def _metrics_on(sig_row: pd.Series | None, row: dict, spec: list[dict], min_av: int) -> dict:
    if sig_row is None:
        return {}
    vals = {k: rnd(v, 6) for k, v in sig_row.items()}
    c = cac.confirm(spec, vals)
    etf, d = row.get("sector_etf"), row.get("direction_sign") or 1
    rs = vals.get(f"sector_rs_20d:{etf}") if etf else None
    return {"confirmation_ratio": c["confirmation_ratio"] if c["n_available"] >= min_av else None,
            "n_available": c["n_available"], "sector_rs_20d_aligned": None if rs is None else d * rs}


def _cond(m: dict, cond: dict) -> bool:
    v = m.get(cond["metric"])
    return v is not None and hc._OPS[cond["op"]](v, float(cond["value"]))


def trigger_day(row: dict, sig: pd.DataFrame, bars: list, i0: int) -> int | None:
    """Erster Tag j in [i0, i0 + max_wait) mit erfülltem Trigger, nur mit Schlusskursen <= j."""
    trig = row.get("wait_trigger") or {}
    spec, min_av = _frozen_spec(row), int(trig.get("min_confirmation_available", 3))
    for j in range(i0, min(len(bars), i0 + int(trig.get("max_wait_trading_days", 10)))):
        ts = pd.Timestamp(bars[j][0])
        m = _metrics_on(sig.loc[ts] if ts in sig.index else None, row, spec, min_av)
        if any(_cond(m, c) for c in trig.get("any_of") or []):
            return j
    return None


def kill_events(row: dict, bars: list, i0: int, i_end: int, sig: pd.DataFrame | None,
                contexts: list[dict]) -> dict:
    """Erstes Auftreten jeder eingefrorenen Kill Condition auf dem Pfad (Handelstag-Index relativ zu i0)."""
    kill = row.get("kill_conditions") or {}
    if not tm.verify_kill(kill):
        return {"integrity": "KILL_HASH_MISMATCH"}
    k = kill["conditions"]
    sign = float(row.get("direction_sign") or 1)
    p0 = bars[i0][3]
    ev: dict = {}
    stop = (k.get("risk_stop") or {}).get("return_le")
    cat = (k.get("catalyst_failure") or {}).get("deadline_trading_days")
    tinv = (k.get("time_invalidation") or {}).get("trading_days")
    for j in range(i0 + 1, i_end + 1):
        r = sign * (bars[j][3] / p0 - 1.0)
        if stop is not None and "risk_stop" not in ev and r <= float(stop):
            ev["risk_stop"] = j - i0
        if cat and "catalyst_failure" not in ev and j - i0 == int(cat) and r <= 0:
            ev["catalyst_failure"] = j - i0
        if tinv and "time_invalidation" not in ev and j - i0 == int(tinv):
            ev["time_invalidation"] = j - i0
    mcf = k.get("market_confirmation_failure") or {}
    if sig is not None and mcf:
        spec, run = _frozen_spec(row), 0
        for j in range(i0 + 1, i_end + 1):
            ts = pd.Timestamp(bars[j][0])
            m = _metrics_on(sig.loc[ts] if ts in sig.index else None, row, spec, int(mcf.get("min_available", 3)))
            run = run + 1 if _cond(m, mcf) else 0
            if run >= int(mcf.get("consecutive_days", 5)):
                ev["market_confirmation_failure"] = j - i0
                break
    ei = k.get("economic_invalidation") or {}
    if ei.get("evaluable"):
        dates = [b[0] for b in bars]
        for c in contexts:
            cd = date.fromisoformat(str(c.get("date"))[:10])
            if cd <= bars[i0][0] or cd > bars[i_end][0]:
                continue
            g = (c.get("gaps") or {}).get(ei["domain"]) or {}
            if g.get("status") == "OK" and g.get("sign") not in (None, 0) and g["sign"] != ei["gap_sign_at_decision"]:
                j = next((i for i, d in enumerate(dates) if d >= cd), None)
                if j is not None:
                    ev["economic_invalidation"] = j - i0
                break
    return ev


def _bars_fn_default(sym: str, start: date, end: date) -> list:
    return fml._yf_bars(sym, start, end)


def resolve_outcomes(*, today: date | None = None, root: Path | None = None, bars_fn=None,
                     cfg: dict | None = None) -> dict:
    """Fällige (Beobachtung, Expression, Variante, Horizont) auflösen – append-only, je Schlüssel einmal.
    Nur abgeschlossene Handelstage (< today)."""
    from modules.expectation_alpha import config as eacfg
    cfg = cfg or eacfg.load()
    today = today or datetime.now(timezone.utc).date()
    bars_fn = bars_fn or _bars_fn_default
    P = paths(root)
    have = read_outcomes(root)
    rows = [r for r in read_rows(root) if r.get("direction_sign") in (1, -1)]
    horizons = [int(h) for h in (cfg.get("outcomes") or {}).get("horizons") or [20, 60, 120, 250]]
    contexts = read_contexts(root)
    cache: dict[str, list] = {}
    stats = {"resolved": 0, "pending": 0, "errors": 0, "kill_hash_mismatch": 0}

    def bars_of(sym: str, start: date) -> list:
        if sym not in cache:
            try:
                b = [x for x in bars_fn(sym, start, today) if x[0] < today and x[3] and x[3] > 0]
            except Exception as e:  # noqa: BLE001 – Kursquelle aus: später erneut
                log.warning(f"expectation_alpha: Kurse {sym} nicht abrufbar ({e})")
                b = []
            cache[sym] = b
        return cache[sym]

    start_all = min((date.fromisoformat(r["date"]) for r in rows), default=today) - timedelta(days=45)
    sig_cache: dict[str, pd.DataFrame] = {}

    def signals_for(row) -> pd.DataFrame | None:
        need = set(cac.CYCLICALS + cac.DEFENSIVES + ("SPY", "IWM", "HYG", "LQD", "^TNX", "^IRX", "TIP", "IEF", "GLD",
                                                    "TLT", "XLE", "^VIX", "^VIX3M", "HG=F", "GC=F"))
        if row.get("sector_etf"):
            need.add(row["sector_etf"])
        key = ",".join(sorted(need))
        if key not in sig_cache:
            cols = {s: pd.Series({pd.Timestamp(d): c for d, _, _, c in bars_of(s, start_all)}, dtype=float)
                    for s in sorted(need)}
            px = pd.DataFrame(cols).sort_index()
            sig_cache[key] = cac.signal_frame(px.dropna(subset=["SPY"]) if "SPY" in px else px)
        return sig_cache[key]

    new: list[dict] = []
    for r in rows:
        # Einstieg = erster Schlusskurs NACH der Entscheidung (thesis.entry_not_before)
        d0 = date.fromisoformat(r.get("entry_not_before") or r["date"])
        sign = float(r["direction_sign"])
        for ex in r.get("expressions") or []:
            if not ex.get("available") or not ex.get("symbol"):
                continue
            variants = ["immediate"] + (["triggered", "kill_managed"] if ex["expression"] == "UNDERLYING" else [])
            if r.get("status") != WAIT:
                variants = [v for v in variants if v != "triggered"]
            wd = int(np.busday_count(d0, today))           # Werktage seit Entscheidung: obere Schranke der Handelstage
            todo = [(v, h) for v in variants for h in horizons
                    if (r["observation_id"], ex["expression"], v, h) not in have and wd > h]
            stats["pending"] += sum(1 for v in variants for h in horizons
                                    if (r["observation_id"], ex["expression"], v, h) not in have and wd <= h)
            if not todo:
                continue
            bars = bars_of(ex["symbol"], start_all)
            i0 = next((i for i, b in enumerate(bars) if b[0] >= d0), None)
            if i0 is None:
                stats["pending"] += len(todo)
                continue
            if (bars[i0][0] - d0).days > 5:
                stats["errors"] += 1                         # Einstieg nicht am Entscheidungstag -> nicht bewertbar
                continue
            cost = float(ex.get("cost_per_side", 0.001))
            for v, h in todo:
                i1 = i0 + h
                if i1 >= len(bars):
                    stats["pending"] += 1
                    continue
                base = {"observation_id": r["observation_id"], "expression": ex["expression"], "symbol": ex["symbol"],
                        "variant": v, "horizon": h, "outcome_method": OUTCOME_METHOD,
                        "resolved_at": today.isoformat(), "decision_date": r["date"], "status": r.get("status")}
                if v == "immediate":
                    e = {**base, **_path(bars, i0, i1, sign, cost), "entry": "ENTERED"}
                elif v == "triggered":
                    sig = signals_for(r)
                    j = trigger_day(r, sig, bars, i0)
                    j = None if j is None else j + int((r.get("wait_trigger") or {}).get("entry_lag_days", 1))
                    if j is None or j >= i1:
                        e = {**base, "entry": "NO_ENTRY", "outcome": 0.0, "outcome_net": 0.0, "mfe": 0.0, "mae": 0.0,
                             "max_drawdown": 0.0, "entry_date": None, "exit_date": bars[i1][0].isoformat(),
                             "holding_days": 0, "trigger_day": None, "note": "kein Trigger -> Cash (Rendite 0)"}
                    else:
                        e = {**base, **_path(bars, j, i1, sign, cost), "entry": "ENTERED", "entry_day": j - i0}
                else:
                    ke = kill_events(r, bars, i0, i1, signals_for(r), contexts)
                    if ke.get("integrity"):
                        stats["kill_hash_mismatch"] += 1
                        e = {**base, "entry": "DATA_BAD", "outcome": None, "outcome_net": None,
                             "kill_events": ke}
                    else:
                        first = min(ke.values()) if ke else None
                        i_exit = i0 + first if first is not None and first < h else i1
                        e = {**base, **_path(bars, i0, i_exit, sign, cost), "entry": "ENTERED", "kill_events": ke,
                             "exit_reason": (min(ke, key=ke.get) if first is not None and first < h else "horizon")}
                new.append(e)
                have[(r["observation_id"], ex["expression"], v, h)] = e
                stats["resolved"] += 1
    if new:
        append_jsonl(P["outcomes"], new, sort_keys=True, default=str)
    return stats


# ── Promotion-Evidenz (gleiches ev-Format wie promotion_controller.evidence) ───
def outcome_map(outs: dict, expression: str = "UNDERLYING", variant: str = "immediate") -> dict:
    """{(obs, h): {outcome (kostenbereinigt), mfe, mae, outcome_method}} für fml.evidence."""
    m = {}
    for (oid, ex, v, h), e in outs.items():
        if ex == expression and v == variant and e.get("outcome_net") is not None:
            m[(oid, h)] = {"outcome": e["outcome_net"], "mfe": e.get("mfe"), "mae": e.get("mae"),
                           "outcome_method": e.get("outcome_method")}
    return m


def paired_observations(contract: dict, spec_hash: str, rows: list[dict], outs: dict, *, horizon: int,
                        since: datetime | None = None) -> list[dict]:
    """Gepaarte Vergleiche je Beobachtung (Treatment − Kontrolle am selben Ausstiegstag)."""
    k, kind = hc.key(contract), contract.get("ea_outcome_kind")
    fwd, reg = fml._ts(contract["forward_start"]), fml._ts(contract["registered_at"])
    out = []
    for r in rows:
        if r.get("stage") != STAGE:
            continue
        t = fml._ts(r["timestamp"])
        if t < fwd or t <= reg or (since is not None and t < since):
            continue
        ev = (r.get("contracts") or {}).get(k)
        if not ev or ev.get("spec_hash") != spec_hash or not ev.get("in_scope") or not ev.get("evaluable"):
            continue
        oid = r["observation_id"]
        if kind == "paired_wait_delta":
            tr, ct = outs.get((oid, "UNDERLYING", "triggered", horizon)), outs.get((oid, "UNDERLYING", "immediate", horizon))
        elif kind == "paired_expression_delta":
            sel = (r.get("selected_expression") or {}).get("selected_expression")
            tr, ct = outs.get((oid, sel, "immediate", horizon)), outs.get((oid, "UNDERLYING", "immediate", horizon))
        else:
            raise ValueError(kind)
        if not tr or not ct or tr.get("outcome_net") is None or ct.get("outcome_net") is None \
                or tr.get("outcome_method") != OUTCOME_METHOD or ct.get("outcome_method") != OUTCOME_METHOD:
            continue
        out.append({"decision_id": oid, "date": r["date"], "ts": t, "ticker": r["ticker"], "sector": r.get("sector"),
                    "regime": r.get("regime"), "treatment": float(tr["outcome_net"]), "control": float(ct["outcome_net"]),
                    "delta": float(tr["outcome_net"]) - float(ct["outcome_net"]),
                    "mfe": tr.get("mfe"), "mae": tr.get("mae"), "control_mae": ct.get("mae"), "fired": True})
    fml.assign_clusters(out)
    return sorted(out, key=lambda x: x["ts"])


def _mean_boot(obs: list[dict], key: str, n: int, seed: int, alpha: float):
    """Einseitige Grenzen (alpha) des mittleren gepaarten Deltas, Block-Bootstrap über `key` (vektorisiert)."""
    blocks: dict[str, list[float]] = {}
    for o in obs:
        blocks.setdefault(o[key], []).append(o["delta"])
    if len(blocks) < 2:
        return None, None
    sums = np.array([sum(v) for v in blocks.values()])
    cnts = np.array([len(v) for v in blocks.values()], dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(blocks), size=(n, len(blocks)))
    vals = np.sort(sums[idx].sum(axis=1) / cnts[idx].sum(axis=1))
    return (round(float(vals[max(0, int(alpha * n) - 1)]), 5), round(float(vals[min(n - 1, int((1 - alpha) * n))]), 5))


def paired_evidence(contract: dict, spec_hash: str, rows: list[dict], outs: dict, policy: dict, alpha: float,
                    since: datetime | None = None) -> dict:
    from modules import promotion_controller as pc
    h = int(contract.get("horizon_days") or 60)
    st = policy.get("statistics") or {}
    obs = paired_observations(contract, spec_hash, rows, outs, horizon=h, since=since)
    ev: dict = {"stage": STAGE, "data_kind": "prospective_forward", "comparison": contract.get("ea_outcome_kind"),
                "horizon_days": h, "n_observations": len(obs), "n_independent_dates": len({o["date"] for o in obs}),
                "n_event_clusters": len({o["cluster"] for o in obs}),
                "calendar_span_days": (obs[-1]["ts"] - obs[0]["ts"]).days if len(obs) > 1 else 0,
                "first_observation": obs[0]["date"] if obs else None,
                "last_observation": obs[-1]["date"] if obs else None,
                "n_fired": len(obs), "n_fired_independent_dates": len({o["date"] for o in obs}),
                "n_fired_event_clusters": len({o["cluster"] for o in obs})}
    ev["fired"] = ev["policy"] = pc.group_metrics([o["treatment"] for o in obs], None, [o["mfe"] for o in obs],
                                                  [o["mae"] for o in obs])
    ev["all"] = ev["not_fired"] = pc.group_metrics([o["control"] for o in obs], None, None,
                                                   [o["control_mae"] for o in obs])
    d = statistics.fmean(o["delta"] for o in obs) if obs else None
    ev["delta_expectancy"] = round(d, 5) if d is not None else None
    bn, seed = int(st.get("bootstrap_n", 2000)), int(st.get("bootstrap_seed", 41))
    cd = _mean_boot(obs, "date", bn, seed, alpha) if obs else (None, None)
    cc = _mean_boot(obs, "cluster", bn, seed, alpha) if obs else (None, None)
    ev["ci_by_date"], ev["ci_by_cluster"] = list(cd), list(cc)
    ev["ci"] = [None if None in (cd[0], cc[0]) else min(cd[0], cc[0]),
                None if None in (cd[1], cc[1]) else max(cd[1], cc[1])]
    ranked = sorted(obs, key=lambda o: o["delta"], reverse=True)
    trims = {}
    for name, kk in (("top1", 1), ("top3", 3), ("top5pct", max(1, math.ceil(0.05 * len(obs))))):
        rest = ranked[kk:]
        trims[name] = round(statistics.fmean(o["delta"] for o in rest), 5) if rest else None
    ev["outlier_trims"] = trims if obs else {}
    win = []
    if len(obs) > 1:
        t0, t1 = obs[0]["ts"], obs[-1]["ts"]
        step = (t1 - t0) / 3
        for i in range(3):
            a, b = t0 + step * i, t0 + step * (i + 1)
            part = [o["delta"] for o in obs if (a <= o["ts"] < b) or (i == 2 and o["ts"] == t1)]
            win.append(round(statistics.fmean(part), 5) if part else None)
    ev["time_windows"] = win
    ev["regimes"] = sorted({o["regime"] for o in obs if o["regime"]})
    secs = [o["sector"] for o in obs if o["sector"]]
    ev["max_sector_share_fired"] = round(max(secs.count(s) for s in set(secs)) / len(secs), 3) if secs else None
    ev["dominant_sector_fired"] = max(set(secs), key=secs.count) if secs else None
    ev["mae_delta"] = (round(statistics.fmean((o["mae"] or 0.0) - (o["control_mae"] or 0.0) for o in obs
                                              if o["mae"] is not None and o["control_mae"] is not None), 5)
                       if any(o["mae"] is not None and o["control_mae"] is not None for o in obs) else None)
    return ev


def evidence(contract: dict, spec_hash: str, rows: list[dict], outs: dict, policy: dict, alpha: float,
             since: datetime | None = None) -> dict:
    """Einstieg für promotion_controller: Evidenz nur aus dem EA-Ledger, nur ab forward_start, gleicher spec_hash."""
    kind = contract.get("ea_outcome_kind") or "fired_vs_not_fired"
    if kind == "fired_vs_not_fired":
        hs = tuple(int(h) for h in contract.get("secondary_horizons_days") or (20, 60, 120, 250))
        return fml.evidence(contract, spec_hash, rows, outcome_map(outs), policy, alpha, since,
                            stage=STAGE, outcome_method=OUTCOME_METHOD, horizons=hs)
    return paired_evidence(contract, spec_hash, rows, outs, policy, alpha, since)


def population_summary(rows: list[dict] | None = None, root: Path | None = None) -> dict:
    rows = rows if rows is not None else read_rows(root)
    obs = [{"ticker": r["ticker"], "date": r["date"]} for r in rows]
    fml.assign_clusters(obs)
    by_status: dict[str, int] = {}
    by_group: dict[str, int] = {}
    for r in rows:
        by_status[r.get("status")] = by_status.get(r.get("status"), 0) + 1
        by_group[r.get("group")] = by_group.get(r.get("group"), 0) + 1
    return {"n": len(rows), "dates": len({r["date"] for r in rows}), "event_clusters": len({o["cluster"] for o in obs}),
            "by_status": by_status, "by_group": by_group,
            "errors": by_status.get(ERROR, 0)}

"""modules/expectation_alpha/thesis.py – Kandidat (Deep Analysis) -> These mit Erwartungs-Kontext.

Liest das Deep-Analysis-Ergebnis read-only und verändert es nie. Kontext je Kandidat:
* macro_alignment = Mittel über die im Sektor-Mapping hinterlegten, verfügbaren Domänen von
  Richtung × Sensitivität × clip(gap_z / gap_z_ref, −1, 1);
* sector_alignment = Richtung × clip(Sektor-RS_63T / sector_rs_ref, −1, 1);
* context_status:
  - +1 bei macro_alignment ≥ positive_min und Sektor nicht klar dagegen;
  - −1 bei macro_alignment ≤ negative_max oder Sektor ≤ sector_contradiction;
  - 0 sonst;
  - None ohne verfügbares macro_alignment (INSUFFICIENT_DATA, Gruppe X).
Forschungsgruppen: A starke News + Kontext +1, B starke + 0, C starke + −1, D schwache + +1,
E schwache + 0/−1, X Kontext unbekannt.

Es gibt keine additive Gesamtpunktzahl; alle Bausteine bleiben getrennt.
"""
from __future__ import annotations

import hashlib

import numpy as np

from modules.expectation_alpha import cross_asset_confirmation as cac
from modules.expectation_alpha import expression as ex
from modules.expectation_alpha import timing as tm
from modules.expectation_alpha.schemas import (ABSTAIN, ERROR, NEWS_STRONG, NEWS_WEAK, OK, STAGE, TRADE, WAIT,
                                               finite, rnd)

TIMING_STATE = {TRADE: "ENTER_NOW", WAIT: "WAIT_FOR_TRIGGER", ABSTAIN: "NO_ENTRY", ERROR: "UNKNOWN"}


def _clip(x: float) -> float:
    return float(max(-1.0, min(1.0, x)))


def direction_of(da: dict) -> int | None:
    d = str((da or {}).get("direction") or "").upper()
    return 1 if d == "BULLISH" else -1 if d == "BEARISH" else None


def news_edge(analysis: dict, impact_min: int, surprise_min: int) -> dict:
    da = analysis.get("deep_analysis") or {}
    imp, sur = finite(da.get("impact")), finite(da.get("surprise"))
    known = imp is not None and sur is not None
    isf = bool(known and imp >= impact_min and sur >= surprise_min)
    return {"impact": imp, "surprise": sur, "ttm": da.get("time_to_materialization"),
            "catalyst": (str(da.get("catalyst"))[:240] if da.get("catalyst") else None),
            "direction": da.get("direction"), "isf_pass": isf if known else None,
            "impact_min": impact_min, "surprise_min": surprise_min,
            "strength": (NEWS_STRONG if isf else NEWS_WEAK) if known else None,
            "rule": "impact >= impact_min and surprise >= surprise_min (Champion-ISF-Floor, unverändert)"}


def candidate_context(direction: int, sector: str | None, ctx: dict, cfg: dict) -> dict:
    cc = cfg.get("context") or {}
    sm = (cfg.get("sector_map") or {}).get(sector) if sector else None
    sens = (sm or {}).get("sensitivities") or {}
    etf = (sm or {}).get("etf")
    contrib, missing = {}, []
    for dom, s in sens.items():
        g = (ctx.get("gaps") or {}).get(dom) or {}
        gz = finite(g.get("gap_z"))
        if g.get("status") == OK and gz is not None:
            contrib[dom] = round(direction * int(s) * _clip(gz / float(cc.get("gap_z_ref", 2.0))), 4)
        else:
            missing.append(dom)
    macro = round(float(np.mean(list(contrib.values()))), 4) if contrib else None
    rs = finite(((ctx.get("signals") or {}).get("values") or {}).get(f"sector_rs_63d:{etf}")) if etf else None
    sector_al = round(direction * _clip(rs / float(cc.get("sector_rs_ref", 0.05))), 4) if rs is not None else None
    contra = float(cc.get("sector_contradiction", -0.5))
    if macro is None:
        status = None
    elif macro <= float(cc.get("negative_max", -0.25)) or (sector_al is not None and sector_al <= contra):
        status = -1
    elif macro >= float(cc.get("positive_min", 0.25)):
        status = 1
    else:
        status = 0
    primary = max(contrib, key=lambda d: abs(contrib[d])) if contrib else None
    return {"sector": sector, "sector_etf": etf, "sensitivities": sens, "mapped": sm is not None,
            "domain_alignment": contrib, "missing_domains": missing, "macro_alignment": macro,
            "sector_rs_63d": rnd(rs, 5), "sector_alignment": sector_al, "context_status": status,
            "primary_domain": primary}


def group_of(strength: str | None, ctx_status) -> str:
    if ctx_status is None or strength is None:
        return "X"
    if strength == NEWS_STRONG:
        return {1: "A", 0: "B", -1: "C"}[ctx_status]
    return "D" if ctx_status == 1 else "E"


def _evidence(direction: int, cctx: dict, conf: dict, news: dict) -> tuple[list[str], list[str]]:
    pro, con = [], []
    if news.get("strength") == NEWS_STRONG:
        pro.append(f"News stark (Impact {news.get('impact')}, Surprise {news.get('surprise')})")
    elif news.get("strength") == NEWS_WEAK:
        con.append(f"News schwach (Impact {news.get('impact')}, Surprise {news.get('surprise')})")
    for dom, v in (cctx.get("domain_alignment") or {}).items():
        (pro if v > 0 else con if v < 0 else []).append(f"Gap {dom} {'stützt' if v > 0 else 'gegen'} These ({v:+.2f})")
    sa = cctx.get("sector_alignment")
    if sa is not None and sa != 0:
        (pro if sa > 0 else con).append(f"Sektor-RS 63T {'mit' if sa > 0 else 'gegen'} These ({sa:+.2f})")
    for s in conf.get("signals") or []:
        if s["state"] == "CONFIRMING":
            pro.append(f"{s['signal']} bestätigt")
        elif s["state"] == "CONFLICTING":
            con.append(f"{s['signal']} widerspricht")
    return pro, con


def entry_not_before(decision_time: str) -> str:
    """Frühester Einstiegstag: der erste US-Schlusskurs NACH der Entscheidung (16:00 New York, Sommer-/Winterzeit
    korrekt). Entscheidung vor Börsenschluss -> Schluss des Entscheidungstags; danach -> Folgetag (Signal und
    Ausführung nie auf demselben, bereits bekannten Schlusskurs)."""
    from datetime import datetime, time, timedelta
    from zoneinfo import ZoneInfo
    ny = ZoneInfo("America/New_York")
    t = datetime.fromisoformat(decision_time).astimezone(ny)
    close = datetime.combine(t.date(), time(16, 0), ny)
    return (t.date() if t < close else t.date() + timedelta(days=1)).isoformat()


def observation_id(date: str, ticker: str, event_key: str | None) -> str:
    return hashlib.sha256(f"{STAGE}|{date}|{ticker}|{event_key or ''}".encode()).hexdigest()[:16]


def build_thesis(analysis: dict, ctx: dict, cfg: dict, *, impact_min: int, surprise_min: int,
                 run_errors: list[str] | None = None) -> dict:
    """Eine These je Kandidat (vollständig, auch bei ERROR). Mutiert `analysis` nie."""
    ticker = analysis.get("ticker")
    da = analysis.get("deep_analysis") or {}
    sector = analysis.get("sector") or (analysis.get("info") or {}).get("sector")
    news = news_edge(analysis, impact_min, surprise_min)
    direction = direction_of(da)
    errors = list(run_errors or [])
    if direction is None:
        errors.append("RICHTUNG_FEHLT")
    if news["strength"] is None:
        errors.append("NEWS_EDGE_FEHLT (impact/surprise)")
    d = direction or 1
    cctx = candidate_context(d, sector, ctx, cfg) if direction is not None else candidate_context(1, None, {}, cfg)
    spec, amb = cac.candidate_spec(d, cctx["sector_etf"], cctx["sensitivities"], cfg)
    conf = cac.confirm(spec, ((ctx.get("signals") or {}).get("values") or {}), amb)
    reg = ctx.get("regime") or {}
    ru = reg.get("regime_uncertainty")
    prim = cctx["primary_domain"]
    gap = ((ctx.get("gaps") or {}).get(prim) or {}) if prim else {}
    dim = (cfg.get("regime_dimension") or {}).get(prim) if prim else None
    dim_state = ((reg.get("dimensions") or {}).get(dim) or {}) if dim else {}
    sens_p = int((cctx["sensitivities"] or {}).get(prim, 0)) if prim else 0
    st_num = {"high": 1, "low": -1, "neutral": 0}.get(dim_state.get("state"))
    regime_support = (d * sens_p * st_num) if st_num is not None and prim else None
    dec = tm.decide(news=news, context=cctx, confirmation=conf, regime_uncertainty=ru, errors=errors, cfg=cfg)
    kill = tm.kill_conditions(ttm=news.get("ttm"), primary_domain=prim, gap_sign=gap.get("sign"), cfg=cfg)
    exprs = ex.registered(ticker, cctx["sector_etf"], cfg)
    sel = ex.select(news, cctx, exprs, cfg)
    acc = finite(((gap.get("gap_roc") or {}).get("acceleration")))
    pro, con = _evidence(d, cctx, conf, news)
    date = ctx["date"]
    oid = observation_id(date, ticker, news.get("catalyst"))
    expectation_gap = None
    if prim:
        expectation_gap = {"domain": prim, "kind": gap.get("kind"), "unit": gap.get("unit"),
                           "gap_raw": gap.get("gap_raw"), "gap_z": gap.get("gap_z"),
                           "gap_percentile": gap.get("gap_percentile"), "sign": gap.get("sign"),
                           "aligned": int(np.sign(cctx["domain_alignment"][prim])),
                           "acceleration": rnd(acc, 6),
                           "acceleration_aligned": (int(np.sign(d * sens_p * acc)) if acc is not None and acc != 0
                                                    else None),
                           "history": gap.get("history")}
    env = contract_env(news, cctx, conf, dec["status"], expectation_gap, ru, sel)
    return {
        "observation_id": oid, "thesis_id": oid, "stage": STAGE, "schema_version": ctx.get("schema_version"),
        "timestamp": ctx["decision_time"], "decision_time": ctx["decision_time"], "date": date,
        "entry_not_before": entry_not_before(ctx["decision_time"]),
        "candidate_id": f"{ticker}:{(news.get('catalyst') or '')[:40]}", "asset": ticker, "ticker": ticker,
        "sector": sector, "sector_etf": cctx["sector_etf"], "direction": da.get("direction"),
        "direction_sign": direction, "horizon": int((cfg.get("outcomes") or {}).get("primary_horizon", 60)),
        "horizons": list((cfg.get("outcomes") or {}).get("horizons") or []),
        "news_edge": news, "catalyst": news.get("catalyst"),
        "model_expectation": (gap.get("model") or {}).get("value") if prim else None,
        "market_expectation": (gap.get("market") or {}).get("value") if prim else None,
        "expectation_gap": expectation_gap,
        "macro_alignment": cctx["macro_alignment"], "sector_alignment": cctx["sector_alignment"],
        "domain_alignment": cctx["domain_alignment"], "missing_domains": cctx["missing_domains"],
        "context_status": cctx["context_status"], "group": group_of(news["strength"], cctx["context_status"]),
        "regime_state": {k: (v or {}).get("state") for k, v in (reg.get("dimensions") or {}).items()},
        "regime_uncertainty": ru, "regime_support": regime_support,
        "regime_dimension": dim, "regime_transition_probability": ((dim_state.get("transition") or {}).get("value")),
        "regime": ctx.get("vix_regime"), "vix": ctx.get("vix"),
        "cross_asset_confirmation": conf, "confirmation_ratio": conf["confirmation_ratio"],
        "evidence_for": pro, "evidence_against": con,
        "status": dec["status"], "status_reasons": dec["reasons"], "timing_state": TIMING_STATE[dec["status"]],
        "wait_trigger": dec["wait_trigger"], "kill_conditions": kill,
        "expressions": exprs, "selected_expression": sel,
        "data_snapshot": {"context_id": ctx.get("context_id"), "context_hash": ctx.get("context_hash"),
                          "signals_date": (ctx.get("signals") or {}).get("date")},
        "code_version": ctx.get("code_version"), "config_hash": ctx.get("config_hash"),
        "sector_map_version": cfg.get("sector_map_version"), "sector_map_hash": ctx.get("sector_map_hash"),
        "env": env, "mode": "shadow", "production_influence": "NONE",
    }


def contract_env(news, cctx, conf, status, gap, ru, sel) -> dict:
    """Flache Merkmale für EA-Verträge (eingefroren). Fehlend bleibt None (Vertrag dann nicht auswertbar)."""
    return {
        "ea_news_strong": None if news.get("strength") is None else int(news["strength"] == NEWS_STRONG),
        "ea_context_status": cctx.get("context_status"),
        "ea_macro_alignment": cctx.get("macro_alignment"),
        "ea_sector_alignment": cctx.get("sector_alignment"),
        "ea_gap_aligned": None if not gap else int(gap["aligned"] > 0),
        "ea_gap_abs_z": None if not gap or gap.get("gap_z") is None else abs(float(gap["gap_z"])),
        "ea_gap_accel_aligned": None if not gap or gap.get("acceleration_aligned") is None
        else int(gap["acceleration_aligned"] > 0),
        "ea_confirmation_ratio": conf.get("confirmation_ratio"),
        "ea_confirmation_available": conf.get("n_available"),
        "ea_regime_uncertainty": ru,
        "ea_status": status,
        "ea_status_valid": int(status != ERROR),
        "ea_is_abstain": int(status == ABSTAIN),
        "ea_is_wait": int(status == WAIT),
        "ea_selected_is_default": int(sel.get("selected_expression") == "UNDERLYING"),
    }

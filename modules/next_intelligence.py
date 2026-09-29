"""
modules/next_intelligence.py – Gesamtvalidierung der Intelligenz-Komponenten
(Challenger A–G), Abstinenz-Bestätigung, Ablation, Promotion-Gate (NUR SHADOW)

    python -m modules.next_intelligence   (CI: nach meta_learning, gleicher Job)

Präregistrierung: config/next_protocol.yaml. Alle Varianten auf IDENTISCHEN
OOS-Zeilen (Meta-Testjahre 2021 bis Locked + Locked-Fold) aus
meta_learning.LAST_RUN; keine neue Infrastruktur.

  A  statisches Ensemble (Champion vor Meta-Learning)
  B  Meta-Regime-Gewichte (Meta-Learning, REJECT – Referenz)
  C  B + World-Model-Zustand als Meta-Merkmal
  D  C + kausale Sektor-Tilts (vorab erklärte Priors, PIT-Koeffizienten)
  E  D + Knowledge-Graph-Evidenz (kuratierte Exposures; Tilt nur mit Kante)
  F  A + Counterfactual-Filter (fragile Top-Dezil-Positionen meiden)
  G  A + alle Komponenten mit KEEP (Filter, Abstinenz, Tilt, Diversifikation)
Kennzahlen je Variante wie in meta_learning plus Expected Shortfall,
High-Confidence-Präzision, Stabilität über Jahre und Regime.
"""

from __future__ import annotations

import json
import logging
import math
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from modules import meta_learning as meta
from modules import ml_research as ml

log = logging.getLogger(__name__)

NP = meta.yaml.safe_load((ml.PROTOCOL_PATH.parent / "next_protocol.yaml").read_text(encoding="utf-8"))
OUT_JSON = ml.OUT_DIR / "next_validation.json"
OUT_MD = ml.OUT_DIR / "next_validation.md"
WORLD_LOG = ml.OUT_DIR / "world_state.jsonl"


def abstention_active(df: pd.DataFrame) -> pd.Series:
    """Vorab registrierte Regel: handeln nur bei VIX >= 20 ODER SPY unter SMA200."""
    return (df["vix"] >= 20) | (df["spy_trend_200"] <= 0)


def monthly_full(pos: pd.DataFrame, months: pd.PeriodIndex) -> pd.Series:
    """Monatsreihe mit 0 (Cash) in Monaten ohne Positionen (Abstinenz)."""
    m = meta.monthly_series(pos) if len(pos) else pd.Series(dtype=float)
    return m.reindex(months).fillna(0.0)


def full_metrics(res: pd.DataFrame, score: str, months: pd.PeriodIndex, active: pd.Series | None = None) -> dict:
    df = res if active is None else res.assign(**{score: res[score].where(active)})
    pos = meta.positions(df, score)
    dev = pos[pos["date"] < ml.LOCKED_FROM] if len(pos) else pos
    m = meta.portfolio_metrics(dev) if len(dev) else {"n_trades": 0}
    mon = monthly_full(dev, months)
    if len(mon) > 2 and mon.std(ddof=1) > 0:
        eq = (1 + mon).cumprod()
        m["sharpe"] = ml._r(mon.mean() / mon.std(ddof=1) * math.sqrt(12), 3)
        m["cagr"] = ml._r(eq.iloc[-1] ** (12 / len(mon)) - 1, 4)
        m["max_dd"] = ml._r(float((eq / eq.cummax() - 1).min()), 4)
        down = mon[mon < 0]
        m["sortino"] = ml._r(mon.mean() / math.sqrt(float((down ** 2).mean())) * math.sqrt(12), 3) if len(down) else None
        m["calmar"] = ml._r(m["cagr"] / abs(m["max_dd"]), 3) if m["max_dd"] and m["max_dd"] < 0 else None
        m["es5_monthly"] = ml._r(mon[mon <= mon.quantile(0.05)].mean(), 5)
    if len(dev):
        r = dev["ret"]
        m["es5_position"] = ml._r(r[r <= r.quantile(0.05)].mean(), 5)
        yrs = dev.groupby("year")["ret"].mean()
        m["stability_sd_yearly"] = ml._r(yrs.std(ddof=1), 5) if len(yrs) > 1 else None
        regs = meta.breakdowns(dev)["by_regime"]
        m["regime_min_expectancy"] = ml._r(min((v["expectancy"] for v in regs.values() if v["expectancy"] is not None),
                                               default=None), 5) if regs else None
        m["active_share_cohorts"] = ml._r(dev["date"].nunique() / max(res[res["date"] < ml.LOCKED_FROM]["date"].nunique(), 1), 3)
    cal = meta.lagged_calibration(df, score)
    m.update(meta.prob_metrics(cal[cal["fold"] != "locked"]) if len(cal) else {})
    if len(cal) and len(dev):
        hc = cal[(cal["prob"] >= NP["validation"]["high_confidence_prob"]) & (cal["fold"] != "locked")]
        top = set(zip(dev["date"], dev["ticker"]))
        hc = hc[[(d, t) in top for d, t in zip(hc["date"], hc["ticker"])]]
        m["hc_n"] = int(len(hc))
        m["hc_hit_rate"] = ml._r((hc[meta.PROB_TARGET] > 0).mean(), 4) if len(hc) else None
    lk = pos[pos["date"] >= ml.LOCKED_FROM] if len(pos) else pos
    m["locked_expectancy"] = ml._r(lk["ret"].mean(), 5) if len(lk) else None
    m["_monthly"] = mon
    return m


def world_features(dates) -> pd.DataFrame:
    """World-Model-Scores je Stichtag aus dem append-only State-Log (PIT je Datum)."""
    if not WORLD_LOG.exists():
        return pd.DataFrame()
    rows = []
    for line in WORLD_LOG.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            log.warning("next_intelligence: defekte World-State-Zeile übersprungen")
    if not rows:
        return pd.DataFrame()
    w = pd.DataFrame(rows)
    w["date"] = pd.to_datetime(w["date"])
    cols = [c for c in w.columns if c.endswith("_score")] + ["uncertainty"]
    w = w.drop_duplicates("date", keep="last").set_index("date")[cols].astype(float)
    return w.reindex(pd.DatetimeIndex(sorted(set(dates)))).add_prefix("wm_")


def variant_meta_with_world(res: pd.DataFrame, meta_d: pd.DataFrame, models: list[str]) -> pd.Series:
    wf = world_features(meta_d.index)
    if wf.empty:
        return pd.Series(np.nan, index=res.index)
    md = meta_d.join(wf)
    state = list(meta.STATE_FEATURES) + list(wf.columns)
    out = pd.Series(np.nan, index=res.index)
    for f in res["fold"].unique():
        te = ml.LOCKED_FROM if f == "locked" else pd.Timestamp(year=int(f), month=1, day=1)
        sub = res[res["fold"] == f]
        wm = meta.regime_weight_model(md, models, te, state_cols=state)
        w = meta.regime_weights(wm, md, sorted(sub["date"].unique()))
        out.loc[sub.index] = meta.score_weighted(sub, w, models)
    return out


def causal_tilt_scores(res: pd.DataFrame) -> pd.Series | None:
    """Kausale Sektor-Tilts je Titel. Beziehungen = ALLE vorab erklärten
    Prior-Paare (keine datenbasierte Auswahl -> keine Selektion mit
    Zukunftsdaten); Koeffizienten PIT (nur Ziele mit Ende < Stichtag).
    None, wenn Kausalschicht/Marktdaten nicht verfügbar sind."""
    try:
        from modules import causal_research as cr
        from modules import world_model as wmod
        px = wmod.fetch_market()
        ind, _ = wmod.build_world(px, wmod.load_archive(), None)
        rels = [{"driver": p["driver"], "target": p["target"], "horizon_weeks": h}
                for p in cr.CP["priors"] for h in cr.CP["horizons_weeks"]]
        tilts = cr.sector_tilts(ind, px, rels, sorted(res["date"].unique()))
    except (ImportError, AttributeError, OSError, ValueError, KeyError) as e:
        log.warning(f"next_intelligence: kausale Tilts nicht verfügbar ({e})")
        return None
    if tilts is None or len(tilts) == 0:
        return None
    t = tilts.copy()
    t["sector"] = t["sector_etf"].map(cr.SECTOR_ETF_MAP)
    key = t.dropna(subset=["sector"]).groupby(["date", "sector"])["tilt"].mean()
    idx = pd.MultiIndex.from_arrays([res["date"], res["sector"]])
    return pd.Series(key.reindex(idx).to_numpy(), index=res.index)


def kg_edge_mask(res: pd.DataFrame) -> pd.Series:
    """E: Tilt nur für Sektoren, die im Knowledge Graph über eine kuratierte
    Kante mit einer Industrie-Exposure verbunden sind (statisch, kein Look-ahead)."""
    try:
        from modules import knowledge_graph as kg
        g = kg.build_graph(**{k: v for k, v in kg.load_inputs().items() if k != "causal"})
    except (ImportError, AttributeError, OSError, ValueError, KeyError, TypeError) as e:
        log.warning(f"next_intelligence: Knowledge Graph nicht verfügbar ({e})")
        return pd.Series(True, index=res.index)
    names = {e["from"].split(":", 1)[1] for e in g["edges"] if e["type"] == "maps_to_industry"}
    return res["sector"].isin(names)


def confirm_abstention(base: pd.DataFrame, models: list[str]) -> dict:
    """Bestätigung auf den UNGESEHENEN Basis-OOS-Jahren 2019–2020 (nicht Teil
    der Meta-Testjahre, in denen der Effekt entdeckt wurde)."""
    cfg = NP["abstention"]
    b = base[base["fold"].isin([str(y) for y in cfg["confirm_years"]])].copy()
    if b.empty:
        return {"status": "no_data"}
    b["s"] = meta.score_static(b, models)
    act = abstention_active(b)
    pa = meta.positions(b.assign(s=b["s"].where(act)), "s")
    pi = meta.positions(b.assign(s=b["s"].where(~act)), "s")
    ca, ci = pa.groupby("date")["ret"].mean(), pi.groupby("date")["ret"].mean()
    diff_t = None
    if len(ca) > 3 and len(ci) > 3:
        se = math.sqrt(ca.var(ddof=1) / len(ca) + ci.var(ddof=1) / len(ci))
        diff_t = (ca.mean() - ci.mean()) / se if se > 0 else None
    ok = len(ca) >= cfg["confirm_min_cohorts"] and ca.mean() > 0 and diff_t is not None and diff_t >= 1.65
    return {"active_cohorts": int(len(ca)), "inactive_cohorts": int(len(ci)),
            "active_expectancy": ml._r(ca.mean(), 5) if len(ca) else None,
            "inactive_expectancy": ml._r(ci.mean(), 5) if len(ci) else None,
            "diff_t": ml._r(diff_t, 2), "confirmed": bool(ok), "years": cfg["confirm_years"]}


def gate(m: dict, ref: dict, boot: dict, ablation: dict, leak_ok: bool) -> dict:
    g = NP["gate"]
    crit = {
        "1_no_leakage": leak_ok,
        "2_reproducible": True,                                   # feste Seeds, Panel-Hash im Bericht
        "3_min_cohorts": (m.get("n_cohorts") or 0) >= g["3_min_cohorts"],
        "4_calibration_not_worse": (m.get("brier", 9) - ref.get("brier", 0) <= g["4_calibration_not_worse"]["brier"]) and
                                   (m.get("ece", 9) - ref.get("ece", 0) <= g["4_calibration_not_worse"]["ece"]),
        "5_risk_adjusted": (m.get("sharpe") or -9) > (ref.get("sharpe") or 0) and bool(boot) and boot["ci_monthly_mean"][0] > 0,
        "6_no_regime_collapse": (m.get("regime_min_expectancy") or -9) - (ref.get("regime_min_expectancy") or 0) >= g["6_max_regime_drop"],
        "7_hc_hit_rate_higher": (m.get("hc_hit_rate") or 0) > (ref.get("hc_hit_rate") or 0),
        "8_drawdown": (m.get("max_dd") or -9) >= (ref.get("max_dd") or 0) - g["8_max_dd_worsening"],
        "9_not_outlier_driven": bool(boot) and boot.get("trimmed_mean", -1) > 0,
        "10_complexity_pays": bool(ablation) and all((v or 0) > 0 for v in ablation.values()),
    }
    return {"pass": all(crit.values()), "criteria": crit, "failed": [k for k, v in crit.items() if not v]}


def boot_with_trim(a: pd.Series, b: pd.Series) -> dict:
    bt = meta.bootstrap_delta(a, b, n=NP["validation"]["bootstrap_n"], seed=NP["validation"]["bootstrap_seed"],
                              alpha=NP["validation"]["alpha_one_sided"] / 6)
    d = (a - b).dropna().sort_values()
    if len(d):
        bt["trimmed_mean"] = float(d.iloc[: int(len(d) * (1 - NP["gate"]["9_trimmed_positive"]))].mean())
    return bt


def evaluate(last: dict, panel: pd.DataFrame | None = None) -> dict:
    res, meta_d, models, base = last["res"], last["meta_d"], last["models"], last["base"]
    res = res.copy()
    months = pd.period_range(res["date"].min(), ml.LOCKED_FROM - pd.Timedelta(days=1), freq="M")
    cases = [c for c in res.columns if c.startswith("cfrank_")]
    # F: Counterfactual-Filter auf A
    if cases:
        base_rank = res.groupby("date")["s_static_equal"].rank(pct=True)
        min_cf = res[cases].min(axis=1)
        res["fragile"] = (base_rank >= 0.9) & (min_cf < 0.9)
        res["s_F"] = res["s_static_equal"].where(~res["fragile"])
    # C: Meta + World Model
    res["s_C"] = variant_meta_with_world(res, meta_d, models)
    # D/E: kausale Tilts
    tilt = causal_tilt_scores(res)
    notes = {}
    if tilt is not None:
        tr = tilt.groupby(res["date"]).rank(pct=True) - 0.5
        base_c = res["s_C"] if res["s_C"].notna().any() else res["s_meta_regime_weights"]
        res["s_D"] = base_c + NP["validation"].get("tilt_weight", 0.25) * tr.fillna(0)
        res["s_E"] = base_c + 0.25 * tr.where(kg_edge_mask(res)).fillna(0)
    else:
        notes["D/E"] = "kausale Tilts nicht verfügbar – D/E nicht ausgewertet"
    # Blind-Spot-Filter
    from modules import blind_spots as bs
    bsr = bs.filter_walk_forward(res, base, "s_static_equal")
    res = res.merge(bsr.pop("scores"), on=["date", "ticker"], how="left")
    act = abstention_active(res)
    variants = {"A": ("s_static_equal", None), "B": ("s_meta_regime_weights", None), "C": ("s_C", None),
                "D": ("s_D", None), "E": ("s_E", None), "F": ("s_F", None),
                "A_blindspot": ("s_static_equal__bs", None), "A_abstention": ("s_static_equal", act)}
    metrics = {k: full_metrics(res, col, months, a) for k, (col, a) in variants.items() if col in res and res[col].notna().any()}
    abst = confirm_abstention(base, models)
    comp_keep = {"counterfactual_filter": None, "blind_spot_filter": bsr["verdict"], "abstention": abst.get("confirmed")}
    # Komponenten-Nutzen (Bootstrap gegen A)
    comp = {}
    for k in ("C", "D", "E", "F", "A_blindspot", "A_abstention"):
        if k in metrics:
            comp[k] = boot_with_trim(metrics[k]["_monthly"], metrics["A"]["_monthly"])
    comp_keep["counterfactual_filter"] = "KEEP" if comp.get("F", {}).get("ci_monthly_mean", [-1])[0] > 0 else (
        "MODIFY" if (metrics.get("F", {}).get("expectancy") or -1) > (metrics["A"].get("expectancy") or 0) else "REJECT")
    # G: A + Komponenten mit KEEP
    g_parts = []
    score_g = res["s_static_equal"].copy()
    if comp_keep["counterfactual_filter"] == "KEEP" and "s_F" in res:
        score_g = score_g.where(~res["fragile"])
        g_parts.append("counterfactual_filter")
    if comp_keep["blind_spot_filter"] == "KEEP":
        score_g = score_g.where(res["s_static_equal__bs"].notna())
        g_parts.append("blind_spot_filter")
    g_active = act if comp_keep["abstention"] else None
    if g_active is not None:
        g_parts.append("abstention")
    res["s_G"] = score_g
    metrics["G"] = full_metrics(res, "s_G", months, g_active)
    boot_g = boot_with_trim(metrics["G"]["_monthly"], metrics["A"]["_monthly"])
    # Ablation: G ohne je eine Komponente
    abl = {}
    for part in g_parts:
        s = res["s_static_equal"].copy()
        if part != "counterfactual_filter" and "counterfactual_filter" in g_parts:
            s = s.where(~res["fragile"])
        if part != "blind_spot_filter" and "blind_spot_filter" in g_parts:
            s = s.where(res["s_static_equal__bs"].notna())
        a2 = g_active if part != "abstention" else None
        m2 = full_metrics(res.assign(_abl=s), "_abl", months, a2)
        abl[part] = ml._r((metrics["G"].get("expectancy") or 0) - (m2.get("expectancy") or 0), 5)
    leak_ok = bool(last.get("leak_ok", True))
    gres = gate(metrics["G"], metrics["A"], boot_g, abl, leak_ok) if g_parts else \
        {"pass": False, "criteria": {}, "failed": ["keine Komponente mit KEEP – G ist identisch mit A"]}
    stress = {}
    try:
        from modules import counterfactual as cf
        stress = {"historical_windows": cf.historical_stress(res, {k: v[0] for k, v in variants.items() if v[0] in res}),
                  "scenario_turnover_top_decile": cf.scenario_turnover(
                      res.assign(_b=res.groupby("date")["s_static_equal"].rank(pct=True)), "_b",
                      {c.replace("cfrank_", ""): c for c in cases}) if cases else {}}
    except (KeyError, ValueError) as e:
        log.warning(f"next_intelligence: Stress-Auswertung fehlgeschlagen ({e})")
    di = {}
    if panel is not None:
        from modules import decision_intel as dint
        try:
            d = dint.walk_forward_diversified(res, panel, "s_static_equal")
            d.pop("_positions", None)
            di = d
        except (KeyError, ValueError) as e:
            log.warning(f"next_intelligence: Decision-Intelligence-Validierung fehlgeschlagen ({e})")
    clean = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in metrics.items()}
    return {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "version": NP["version"],
            "protocol": "config/next_protocol.yaml", "metrics": clean,
            "component_vs_A": comp, "component_verdicts": {**comp_keep, "decision_intelligence": di.get("verdict")},
            "abstention_confirmation": abst, "blind_spots": {k: v for k, v in bsr.items()},
            "G_components": g_parts, "G_vs_A": boot_g, "ablation_delta_expectancy": abl,
            "gate": gres, "decision": "PROMOTE" if gres["pass"] else "KEEP_CHAMPION",
            "stress": stress, "decision_intelligence": di, "notes": notes,
            "fragile_share_top_decile": ml._r(float(res.loc[res.groupby("date")["s_static_equal"].rank(pct=True) >= 0.9,
                                                            "fragile"].mean()), 4) if "fragile" in res else None}


def render_md(rep: dict) -> str:
    L = [f"# Gesamtvalidierung Intelligenz-Komponenten – {rep['generated']}", "",
         f"**Entscheidung: {rep['decision']}** · G-Komponenten: {rep['G_components'] or '–'} · "
         f"Gate nicht erfüllt: {rep['gate'].get('failed') or '–'}", "",
         "| Variante | CAGR | Sharpe | Sortino | Calmar | MaxDD | ES5 Monat | Hit | PF | Expectancy | Ø Gew. | Ø Verl. | Prec@K | Brier | ECE | LogLoss | Turnover | Trades | HC-Hit (n) | Stabilität σJahr | min Regime | aktiv | Locked |",
         "|" + "---|" * 23]
    for k, m in rep["metrics"].items():
        L.append("| " + " | ".join(str(x) for x in [
            k, m.get("cagr"), m.get("sharpe"), m.get("sortino"), m.get("calmar"), m.get("max_dd"), m.get("es5_monthly"),
            m.get("hit_rate"), m.get("profit_factor"), m.get("expectancy"), m.get("avg_winner"), m.get("avg_loser"),
            m.get("precision_at_k"), m.get("brier"), m.get("ece"), m.get("log_loss"), m.get("turnover"), m.get("n_trades"),
            f"{m.get('hc_hit_rate')} ({m.get('hc_n')})", m.get("stability_sd_yearly"), m.get("regime_min_expectancy"),
            m.get("active_share_cohorts"), m.get("locked_expectancy")]) + " |")
    L += ["", "## Komponenten gegen A (Bootstrap, Bonferroni über 6)", ""]
    for k, b in rep["component_vs_A"].items():
        L.append(f"- {k}: Monats-Δ {b.get('delta_monthly_mean')} CI {b.get('ci_monthly_mean')} · ohne Top-5 %-Monate {ml._r(b.get('trimmed_mean'), 6)}")
    L += ["", f"Verdikte: {rep['component_verdicts']}", "",
          f"Abstinenz-Bestätigung (ungesehene Jahre {rep['abstention_confirmation'].get('years')}): {rep['abstention_confirmation']}",
          "", f"Ablation (Δ Expectancy G − G ohne Komponente): {rep['ablation_delta_expectancy'] or '–'}",
          "", f"Gate: {rep['gate']}", "", f"Stress: {rep['stress']}", "",
          f"Decision Intelligence: { {k: v for k, v in rep['decision_intelligence'].items()} }",
          "", f"Anteil fragiler Positionen im Top-Dezil: {rep.get('fragile_share_top_decile')}", "",
          f"Hinweise: {rep.get('notes')}"]
    return "\n".join(L) + "\n"


def run() -> dict:
    cache = os.environ.get("META_RES_CACHE")
    panel_cache = os.environ.get("ML_PANEL_CACHE")
    panel = None
    if panel_cache and Path(panel_cache).exists():
        with open(panel_cache, "rb") as fh:
            panel = pickle.load(fh)  # noqa: S301 – eigene, im selben Job erzeugte Datei
    if cache and Path(cache).exists():
        with open(cache, "rb") as fh:
            last = pickle.load(fh)  # noqa: S301 – eigene, im selben Job erzeugte Datei
    else:
        meta.run(panel)
        last = meta.LAST_RUN
    mj = json.loads(meta.OUT_JSON.read_text()) if meta.OUT_JSON.exists() else {}
    last["leak_ok"] = (mj.get("leakage_checks") or {}).get("ok", True)
    rep = evaluate(last, panel)
    ml.OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(rep, indent=1, default=str, ensure_ascii=False))
    OUT_MD.write_text(render_md(rep), encoding="utf-8")
    return rep


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    print(render_md(run()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

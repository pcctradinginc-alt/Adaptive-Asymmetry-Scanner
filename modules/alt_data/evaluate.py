"""Inkrementelle Bewertung von Alternative-Data-Quellen (Research/SHADOW).

    python -m modules.alt_data.evaluate        (CI: nach ml_research, ML_PANEL_CACHE)

Nicht "funktioniert Feature X?", sondern:
    BASELINE (bestehende Registry-Spec)  vs.  BASELINE + ausgewählte Quellen-Features
auf identischen Walk-Forward-Zeilen (Testjahre vor dem kontaminierten Locked),
plus Redundanz-/Abdeckungsprüfung, Sektor-/Regime-Aufschlüsselung und ein
deterministischer Source Value Score. Protokoll: config/alt_data_protocol.yaml.
"""
from __future__ import annotations

import json
import logging
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from modules import meta_learning as meta
from modules import ml_research as ml
from modules.alt_data.registry import SOURCES

log = logging.getLogger(__name__)
PROTOCOL_PATH = Path("config/alt_data_protocol.yaml")
AP = yaml.safe_load(PROTOCOL_PATH.read_text())
OUT_JSON = ml.OUT_DIR / "alt_data_validation.json"
OUT_MD = ml.OUT_DIR / "alt_data_validation.md"
SCOREBOARD = ml.OUT_DIR / "source_scoreboard.json"


def _xs_rank_corr(panel: pd.DataFrame, a: str, b: str) -> float | None:
    vals = []
    for _, g in panel[["date", a, b]].dropna().groupby("date"):
        if len(g) >= 30 and g[a].nunique() > 1 and g[b].nunique() > 1:
            vals.append(g[a].rank().corr(g[b].rank()))
    return float(np.mean(vals)) if vals else None


def feature_screen(panel: pd.DataFrame, feats: list[str], years: list[int]) -> dict:
    """Abdeckung, Redundanz (max |ρ| zu bestehenden Features) und IC je Feature – NUR Auswahljahre."""
    sel = panel[panel["date"].dt.year.isin(years)]
    out = {}
    for f in feats:
        if f not in sel:
            out[f] = {"coverage": 0.0}
            continue
        cov = float(sel[f].notna().mean())
        corrs = {b: _xs_rank_corr(sel, f, b) for b in ml.STOCK_FEATURES}
        corrs = {k: v for k, v in corrs.items() if v is not None}
        mx = max(corrs.items(), key=lambda kv: abs(kv[1])) if corrs else (None, None)
        ic = ml.daily_ic(sel[sel[f].notna()], f).mean() if cov > 0 else np.nan
        out[f] = {"coverage": round(cov, 3), "max_abs_corr_existing": None if mx[1] is None else round(abs(mx[1]), 3),
                  "most_similar_existing": mx[0], "incremental_information_score":
                  None if mx[1] is None else round(1 - abs(mx[1]), 3),
                  "selection_ic": None if pd.isna(ic) else round(float(ic), 4)}
    return out


def select_features(screen: dict) -> list[str]:
    fs = AP["feature_selection"]
    ok = [f for f, s in screen.items() if s.get("coverage", 0) >= fs["min_feature_coverage"]
          and (s.get("max_abs_corr_existing") is None or s["max_abs_corr_existing"] < fs["redundancy_max_abs_rank_corr"])
          and s.get("selection_ic") is not None]
    ok.sort(key=lambda f: -abs(screen[f]["selection_ic"]))
    return ok[:fs["max_features_per_source"]]


def _oos(panel: pd.DataFrame, spec: dict) -> pd.DataFrame:
    wf = ml.walk_forward(panel, spec, ml.LOCKED_FROM, importance=False)
    if wf.get("status") != "ok":
        return pd.DataFrame()
    o = wf["oos"]
    return o[o["date"].dt.year.isin(AP["evaluation"]["dev_years"])]


def _metrics(oos: pd.DataFrame) -> tuple[dict, pd.Series, pd.DataFrame]:
    o = meta.add_rel_target(oos.assign(fold=oos["date"].dt.year.astype(str)))
    pos = meta.positions(o, "score")
    pm = meta.portfolio_metrics(pos)
    cal = meta.lagged_calibration(o, "score")
    prob = meta.prob_metrics(cal) if len(cal) else {}
    ic = ml.daily_ic(oos, "score").mean()
    m = {"ic": None if pd.isna(ic) else round(float(ic), 5), **{k: pm.get(k) for k in
         ("sharpe", "expectancy", "max_dd", "precision_at_k", "hit_rate", "n_trades")},
         **{k: prob.get(k) for k in ("brier", "ece", "log_loss")}}
    return m, (meta.monthly_series(pos) if len(pos) else pd.Series(dtype=float)), pos


def breakdown(pos_b: pd.DataFrame, pos_v: pd.DataFrame, panel: pd.DataFrame) -> dict:
    cols = [c for c in ("sector", "vix") if c in panel]
    key = panel[["date", "ticker"] + cols].drop_duplicates(["date", "ticker"])
    out = {"sector": {}, "regime": {}}
    for name, pos in (("base", pos_b), ("variant", pos_v)):
        p = pos.merge(key, on=["date", "ticker"], how="left")
        p["sector"] = p["sector"] if "sector" in p else "unknown"
        p["regime"] = np.where(p["vix"] >= 20, "vix_ge_20", "vix_lt_20") if "vix" in p else "unknown"
        for col in ("sector", "regime"):
            for k, g in p.groupby(col):
                out[col].setdefault(str(k), {})[name] = round(float(g["ret"].mean()), 5)
                out[col][str(k)][f"n_{name}"] = int(len(g))
    for col in out:
        for k, v in out[col].items():
            if "base" in v and "variant" in v:
                v["delta"] = round(v["variant"] - v["base"], 5)
    return out


def evaluate_source(panel: pd.DataFrame, sid: str, src: dict, specs: dict[str, dict]) -> dict:
    av = src["availability_col"]
    dev = panel[panel["date"].dt.year.isin(AP["evaluation"]["dev_years"])]
    coverage = float(dev[av].mean()) if av in dev else 0.0
    screen = feature_screen(panel, src["features"], AP["feature_selection"]["selection_years"])
    chosen = select_features(screen)
    res = {"source": sid, "coverage_dev": round(coverage, 3), "screen": screen, "selected_features": chosen,
           "baselines": {}}
    if not chosen or coverage < AP["evaluation"]["min_coverage"]:
        res["verdict"] = "REJECT"
        res["verdict_reason"] = ("keine nicht-redundante Feature mit ausreichender Abdeckung"
                                 if not chosen else f"Abdeckung {coverage:.0%} < {AP['evaluation']['min_coverage']:.0%}")
        return res
    n_tests = len(specs) * len(SOURCES)
    for bid, spec in specs.items():
        ob = _oos(panel, spec)
        ov = _oos(panel, {**spec, "id": f"{bid}+{sid}", "extra_features": chosen})
        if ob.empty or ov.empty:
            res["baselines"][bid] = {"status": "no_data"}
            continue
        common = ob[["date", "ticker"]].merge(ov[["date", "ticker"]], on=["date", "ticker"])
        ob = ob.merge(common, on=["date", "ticker"])
        ov = ov.merge(common, on=["date", "ticker"])
        mb, sb, pb = _metrics(ob)
        mv, sv, pv = _metrics(ov)
        boot = meta.bootstrap_delta(sv, sb, n=AP["evaluation"]["bootstrap_n"], seed=AP["evaluation"]["bootstrap_seed"],
                                    alpha=AP["evaluation"]["alpha_one_sided"] / n_tests)
        years = {}
        for y in AP["evaluation"]["dev_years"]:
            yb, yv = sb[sb.index.year == y], sv[sv.index.year == y]
            if len(yb) and len(yv):
                years[str(y)] = round(float(yv.mean() - yb.mean()), 5)
        res["baselines"][bid] = {"base": mb, "with_source": mv, "bootstrap": boot,
                                 "delta": {k: (None if mb.get(k) is None or mv.get(k) is None
                                               else round(mv[k] - mb[k], 6)) for k in mb},
                                 "delta_by_year": years, "breakdown": breakdown(pb, pv, panel)}
    res["verdict"], res["verdict_reason"] = decide(res, coverage)
    return res


def decide(res: dict, coverage: float) -> tuple[str, str]:
    bl = [b for b in res["baselines"].values() if b.get("bootstrap")]
    if not bl:
        return "REJECT", "keine auswertbare Baseline"
    keep = [b for b in bl if (b["bootstrap"].get("ci_monthly_mean") or [-1])[0] > 0
            and (b["delta"].get("brier") is None or b["delta"]["brier"] <= 0.0005)]
    if keep and coverage >= AP["evaluation"]["min_coverage"]:
        return "KEEP", "Bootstrap-Untergrenze > 0 (Bonferroni) und Brier nicht schlechter"
    pos_all = all((b["bootstrap"].get("delta_monthly_mean") or -1) > 0 for b in bl)
    pos_seg = any(v.get("delta", -1) > 0.002 and v.get("n_variant", 0) >= 200
                  for b in bl for v in b["breakdown"]["sector"].values())
    if pos_all or pos_seg:
        return "MODIFY", ("Δ > 0 bei allen Baselines, CI schließt 0 ein" if pos_all
                          else "nur in einzelnen Sektoren positiv")
    return "REJECT", "kein inkrementeller Nutzen gegenüber der Baseline"


def source_value_score(res: dict, health: dict | None, mapping_quality: float | None) -> dict:
    """Deterministisch aus Messwerten (Gewichte im Protokoll)."""
    w = AP["source_value_score"]["weights"]
    deltas = [b["bootstrap"]["delta_monthly_mean"] for b in res["baselines"].values()
              if b.get("bootstrap") and b["bootstrap"].get("delta_monthly_mean") is not None]
    inc = float(np.clip(0.5 + (np.mean(deltas) if deltas else -0.005) / 0.01, 0, 1))
    years = [v for b in res["baselines"].values() for v in (b.get("delta_by_year") or {}).values()]
    stability = float(np.mean([v > 0 for v in years])) if years else 0.0
    fresh = 0.0
    if health and health.get("last_observation"):
        age = (datetime.now(timezone.utc) - pd.Timestamp(health["last_observation"]).to_pydatetime()).days
        fresh = float(np.clip(1 - age / 120, 0, 1))
    comp = {"incremental_value": inc, "coverage": float(res.get("coverage_dev") or 0), "freshness": fresh,
            "stability": stability, "mapping_quality": float(mapping_quality or 0),
            "api_reliability": float(1 - (health or {}).get("error_rate", 1.0)), "revision_risk": 1.0}
    return {"score": round(sum(w[k] * comp[k] for k in w), 3), "components": {k: round(v, 3) for k, v in comp.items()}}


def run(panel: pd.DataFrame) -> dict:
    reg = ml.load_registry()
    st, _ = ml.check_registry(reg)
    specs = {s["id"]: s for s in reg.get("models") or [] if s["id"] in AP["evaluation"]["baselines"]
             and st.get(s["id"]) == "valid"}
    rep = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "protocol": AP["version"],
           "sources": {}}
    board = {}
    for sid, src in SOURCES.items():
        r = evaluate_source(panel, sid, src, specs)
        health_path = Path("outputs/external_data/sec/health.json") if sid == "sec_deep_events" else None
        health = json.loads(health_path.read_text()) if health_path and health_path.exists() else None
        er = Path("outputs/entity/entity_report.json")
        mq = None
        if er.exists():
            e = json.loads(er.read_text()).get("sec", {})
            mq = (e.get("mapped") or 0) / max(1, e.get("wanted") or 0)
        r["source_value"] = source_value_score(r, health, mq)
        rep["sources"][sid] = r
        board[sid] = {"coverage": r["coverage_dev"], "freshness": r["source_value"]["components"]["freshness"],
                      "data_quality": round(1 - (health or {}).get("error_rate", 1.0), 3) if health else None,
                      "active_features": r["selected_features"], "oos_value": r["source_value"]["components"]["incremental_value"],
                      "forward_value": None, "source_value_score": r["source_value"]["score"],
                      "status": {"KEEP": "HISTORICALLY_VALIDATED", "MODIFY": "SHADOW", "REJECT": "REJECTED"}[r["verdict"]],
                      "verdict": r["verdict"]}
    ml.OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(rep, indent=1, default=str, ensure_ascii=False))
    OUT_MD.write_text(render_md(rep), encoding="utf-8")
    SCOREBOARD.write_text(json.dumps({"generated": rep["generated"], "sources": board}, indent=1))
    return rep


def render_md(rep: dict) -> str:
    L = [f"# Alternative Data – inkrementelle Validierung ({rep['generated']}, Protokoll {rep['protocol']})", ""]
    for sid, r in rep["sources"].items():
        L += [f"## {sid}: **{r['verdict']}** – {r['verdict_reason']}", "",
              f"Abdeckung (Dev-Jahre): {r['coverage_dev']} · ausgewählte Features: {r['selected_features'] or '–'} · "
              f"Source Value Score {r['source_value']['score']} {r['source_value']['components']}", "",
              "| Feature | Abdeckung | max |ρ| bestehend | ähnlichstes | Info-Score | IC (Auswahljahre) |", "|---|---|---|---|---|---|"]
        for f, s in r["screen"].items():
            L.append(f"| {f} | {s.get('coverage')} | {s.get('max_abs_corr_existing')} | {s.get('most_similar_existing')} | "
                     f"{s.get('incremental_information_score')} | {s.get('selection_ic')} |")
        for bid, b in r["baselines"].items():
            if b.get("status") == "no_data":
                continue
            L += ["", f"### Baseline {bid}", "", "| Metrik | Baseline | + Quelle | Δ |", "|---|---|---|---|"]
            for k in b["base"]:
                L.append(f"| {k} | {b['base'][k]} | {b['with_source'][k]} | {b['delta'][k]} |")
            L += ["", f"Bootstrap: {b['bootstrap']}", f"Δ je Jahr: {b['delta_by_year']}",
                  f"Regime: {b['breakdown']['regime']}"]
        L.append("")
    return "\n".join(L)


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    cache = os.environ.get("ML_PANEL_CACHE")
    if cache and Path(cache).exists():
        with open(cache, "rb") as fh:
            panel = pickle.load(fh)  # noqa: S301 – eigene, im selben Job erzeugte Datei
    else:
        panel = ml.build_research_panel("full")
    from modules.alt_data.feature_store import attach
    if any(c not in panel for s in SOURCES.values() for c in s["features"]):
        panel = attach(panel)
    print(render_md(run(panel)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

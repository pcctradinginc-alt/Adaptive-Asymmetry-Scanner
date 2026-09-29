"""
modules/meta_cognition.py – Meta-Cognition, Alpha Decay, Research Value
Attribution, Safe Mode (NUR SHADOW)

    python -m modules.meta_cognition      (CI: am Ende des monatlichen/wöchentlichen Laufs)

Konsolidiert ALLE gemessenen Research-Ergebnisse zu einem Machine Intelligence
State. Jede Aussage trägt ihre Metrik; ohne Messung gibt es keine Aussage.
  WHAT DO WE KNOW / UNCERTAIN / SYSTEMATICALLY WRONG / STALE ASSUMPTIONS /
  REDUNDANT MODELS / DECAYING FEATURES / LOW DATA QUALITY / HIGH DISAGREEMENT /
  RESEARCH TRACKS WITH REAL OOS VALUE / WASTED RESEARCH
Alpha Decay: rollender Rank-IC je Merkmal und Modell (13/52 Wochen, nur fertige
Labels), CUSUM-Strukturbruch, Decay-Flag. Research Value: Experimente,
Annahmen und OOS-Beitrag je Research-Track. Safe Mode: Auslöser aus
config/next_protocol.yaml -> outputs/research/safe_mode.json (HC-Alerts aus,
Rückfall auf den stabilen Champion, Warnung im Weekly Report).
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
import yaml

log = logging.getLogger(__name__)

OUT = Path("outputs/research")
NP = yaml.safe_load((Path(__file__).resolve().parent.parent / "config" / "next_protocol.yaml").read_text(encoding="utf-8"))
CUSUM_CRIT = 1.36                   # 5 %-Schwelle des Kolmogorov-artigen CUSUM-Tests


def _load(name: str, default=None):
    p = OUT / name
    try:
        return json.loads(p.read_text()) if p.exists() else default
    except (OSError, json.JSONDecodeError) as e:
        log.warning(f"meta_cognition: {name} nicht lesbar ({e})")
        return default


# ── Alpha Decay ──────────────────────────────────────────────────────────────

def cusum_break(ic: pd.Series) -> dict:
    """CUSUM der Abweichungen vom Mittel; Statistik = max|S| / (σ·√n)."""
    s = ic.dropna()
    if len(s) < 52 or s.std(ddof=1) == 0:
        return {"break": False, "stat": None}
    dev = (s - s.mean()).cumsum()
    stat = float(dev.abs().max() / (s.std(ddof=1) * math.sqrt(len(s))))
    return {"break": stat > CUSUM_CRIT, "stat": round(stat, 3), "at": str(dev.abs().idxmax().date()) if stat > CUSUM_CRIT else None}


def decay_profile(ic: pd.Series, label_end: pd.Series | None = None) -> dict:
    s = ic.dropna()
    if label_end is not None:
        s = s[label_end.reindex(s.index) < s.index.max()]            # nur fertige Labels
    if len(s) < 60:
        return {"n": int(len(s)), "status": "insufficient_history"}
    rec, prev = s.tail(52), s.iloc[-104:-52]
    se = math.sqrt(rec.var(ddof=1) / len(rec) + prev.var(ddof=1) / max(len(prev), 1)) if len(prev) > 3 else None
    t = (rec.mean() - prev.mean()) / se if se else None
    cb = cusum_break(s)
    decaying = bool(t is not None and t <= -2 and prev.mean() > 0)
    return {"n": int(len(s)), "ic_all": round(float(s.mean()), 4), "ic_prev_52w": round(float(prev.mean()), 4) if len(prev) else None,
            "ic_last_52w": round(float(rec.mean()), 4), "ic_last_13w": round(float(s.tail(13).mean()), 4),
            "delta_t": round(t, 2) if t is not None else None, "structural_break": cb,
            "status": "decaying" if decaying else ("break" if cb["break"] else "stable")}


def feature_decay(panel: pd.DataFrame) -> dict:
    from modules import ml_research as ml
    out = {}
    for f in ml.STOCK_FEATURES:
        ic = ml.daily_ic(panel.assign(_s=panel[f]), "_s")
        le = panel.groupby("date")["label_end_20"].max()
        out[f] = decay_profile(ic, le)
    return out


# ── Research Value Attribution ──────────────────────────────────────────────

def research_value(hyp_db: dict, ml_rep: dict, meta: dict) -> dict:
    tracks: dict = {}
    for hid, r in (hyp_db.get("hypotheses") or {}).items():
        src = r.get("source") or "config"
        tr = tracks.setdefault(src, {"experiments": 0, "accepted": 0, "oos_contribution": 0.0, "rejected": 0})
        if r.get("status") in ("blocked_data", "prior_result"):
            continue
        tr["experiments"] += 1
        if r.get("canonical_status") == "ACCEPTED":
            tr["accepted"] += 1
            tr["oos_contribution"] += ((r.get("walk_forward") or {}).get("base") or {}).get("mean") or 0.0
        if r.get("canonical_status") == "REJECTED":
            tr["rejected"] += 1
    disc = (hyp_db.get("discovery") or {})
    if disc:
        tracks["discovery_engine"] = {"experiments": disc.get("n_tested", 0), "accepted": disc.get("n_survivors", 0),
                                      "oos_contribution": 0.0, "rejected": disc.get("n_tested", 0) - disc.get("n_survivors", 0)}
    ml_models = (ml_rep or {}).get("models") or {}
    tracks["ml_models"] = {"experiments": len(ml_models),
                           "accepted": sum(1 for m in ml_models.values() if (m.get("decision") or {}).get("verdict") == "promote_recommended"),
                           "oos_contribution": 0.0, "rejected": sum(1 for m in ml_models.values()
                                                                   if (m.get("decision") or {}).get("verdict") == "rejected_so_far")}
    tracks["meta_learning"] = {"experiments": len((meta or {}).get("approaches") or {}),
                               "accepted": int(((meta or {}).get("decision") or {}).get("verdict") == "PROMOTE"),
                               "oos_contribution": 0.0, "rejected": 0}
    for t in tracks.values():
        n = max(t["experiments"], 1)
        rate = t["accepted"] / n
        t["research_efficiency"] = "HIGH" if rate >= 0.1 and t["oos_contribution"] > 0 else ("MEDIUM" if rate > 0 else "LOW")
        t["oos_contribution"] = round(t["oos_contribution"], 5)
    return tracks


# ── Safe Mode ───────────────────────────────────────────────────────────────

def safe_mode(meta: dict, world: dict, health: dict | None, forward: list[float] | None, ml_rep: dict) -> dict:
    tr = NP["safe_mode"]["triggers"]
    reasons = []
    drift = (meta or {}).get("drift") or {}
    if tr["feature_drift"] and drift.get("feature_drift_flag"):
        reasons.append("FEATURE/DATA DRIFT: Regime-Merkmale außerhalb des Trainingsbereichs "
                       f"({[k for k, v in (drift.get('feature_drift') or {}).items() if v.get('out_of_range')]})")
    mi = (meta or {}).get("model_intelligence") or {}
    if mi:
        share = sum(1 for m in mi.values() if m.get("trend") == "deteriorating") / len(mi)
        if share >= tr["model_drift_share"]:
            reasons.append(f"MODEL DRIFT: {share:.0%} der Modelle 'deteriorating'")
    cal = (ml_rep or {}).get("calibration") or {}
    ece = (((meta or {}).get("approaches") or {}).get((meta or {}).get("active_ensemble", "static_equal"), {})
           .get("metrics", {}) or {}).get("ece")
    if tr["calibration_failure"] and (cal and not cal.get("interval_calibrated") or (ece is not None and ece > 0.05)):
        reasons.append(f"CALIBRATION FAILURE: Intervalle kalibriert={cal.get('interval_calibrated')}, ECE={ece}")
    if health:
        st = [str(v.get("status", "")).upper() for v in health.values() if isinstance(v, dict)]
        fail = sum(1 for s in st if s in ("FAIL", "FAILED", "ERROR")) / max(len(st), 1)
        if fail >= tr["pipeline_fail_share"]:
            reasons.append(f"PIPELINE FAILURE: {fail:.0%} der Quellen FAIL")
    if ((meta or {}).get("disagreement") or {}).get("current_level") == tr["disagreement_level"]:
        reasons.append("EXCESSIVE MODEL DISAGREEMENT")
    wu = ((world or {}).get("current") or {}).get("uncertainty")
    if wu is not None and wu >= tr["world_model_uncertainty"]:
        reasons.append(f"WORLD MODEL INSTABILITY: Unsicherheit {wu}")
    if forward and len(forward) >= 12:
        f = pd.Series(forward[-12:])
        t = f.mean() / f.std(ddof=1) * math.sqrt(len(f)) if f.std(ddof=1) > 0 else 0
        if t <= tr["performance_breakdown_t"]:
            reasons.append(f"PERFORMANCE BREAKDOWN: Forward-Expectancy t={t:.2f}")
    return {"active": bool(reasons), "reasons": reasons, "fallback": "stabiler Champion (statisches Ensemble); keine HC-Alerts",
            "updated": datetime.now(timezone.utc).isoformat(timespec="seconds")}


# ── Machine Intelligence State ───────────────────────────────────────────────

def machine_state(inp: dict, decay: dict, model_decay: dict, rv: dict, sm: dict) -> dict:
    meta, nextv, world, ml_rep = inp.get("meta") or {}, inp.get("nextv") or {}, inp.get("world") or {}, inp.get("ml") or {}
    hyp = inp.get("hyp") or {}
    know = []
    ref = ((meta.get("approaches") or {}).get("static_equal") or {}).get("metrics") or {}
    if ref:
        know.append(f"Champion (statisches Ensemble) OOS 2021–2025H1: Expectancy {ref.get('expectancy')} je Position, "
                    f"Sharpe {ref.get('sharpe')}, Trefferquote {ref.get('hit_rate')}")
    reg = ((meta.get("approaches") or {}).get("static_equal") or {}).get("by_regime") or {}
    if reg:
        know.append("Regime-Abhängigkeit: " + ", ".join(f"{k} {v.get('expectancy')}" for k, v in reg.items()))
    ab = nextv.get("abstention_confirmation") or {}
    if ab:
        know.append(f"Abstinenz-Regel auf ungesehenen Jahren {ab.get('years')}: aktiv {ab.get('active_expectancy')} "
                    f"vs. inaktiv {ab.get('inactive_expectancy')} (t={ab.get('diff_t')}) -> bestätigt={ab.get('confirmed')}")
    for d in (hyp.get("hypotheses") or {}).values():
        if d.get("canonical_status") == "ACCEPTED":
            know.append(f"Akzeptierte Hypothese {d.get('id')}: {d.get('title')}")
    uncertain = []
    cal = ml_rep.get("calibration") or {}
    if cal:
        uncertain.append(f"80-%-Intervalle: roh {cal.get('coverage')}, korrigiert {cal.get('coverage_conformal')}; "
                         f"P(>+10 %) Skill {cal.get('p_up_skill_recal')}")
    wc = (world.get("current") or {})
    if wc:
        uncertain.append(f"World-Model-Unsicherheit {wc.get('uncertainty')}; "
                         f"nicht verfügbar: {[k[:-6] for k, v in wc.items() if k.endswith('_state') and v in ('unavailable', 'no_data')]}")
    wrong = [f"{c['id']}: {c['common_properties']} (Fehler {c['typical_error']}, Lift {c['lift']}, Abdeckung {c['existing_model_coverage']})"
             for c in nextv.get("blind_spot_clusters") or []]
    buckets = (meta.get("calibration_buckets") or {}).get("static_equal") or []
    wrong += [f"Überkonfident im Bucket {b['bucket']}: prognostiziert {b.get('predicted')}, realisiert {b.get('win_rate')} (n={b['n']})"
              for b in buckets if b.get("flag") == "overconfident"]
    stale = [f"{k}: Wert {v.get('value')} außerhalb Trainingsband [{v.get('p01')}, {v.get('p99')}]"
             for k, v in ((meta.get("drift") or {}).get("feature_drift") or {}).items() if v.get("out_of_range")]
    redundant = ["rs_63 ≡ mom_3m (identische Querschnittsränge, Audit A1)"]
    contrib = {m: v.get("contribution") for m, v in (meta.get("model_intelligence") or {}).items()}
    redundant += [f"{m}: Leave-one-out-Beitrag {c} (≤ 0 = entbehrlich)" for m, c in contrib.items() if c is not None and c <= 0]
    decaying = [f"{f}: IC {v.get('ic_prev_52w')} -> {v.get('ic_last_52w')} (t={v.get('delta_t')})" for f, v in decay.items()
                if v.get("status") == "decaying"]
    decaying += [f"Modell {m}: {v.get('prior_ic')} -> {v.get('recent_ic')} (t={v.get('trend_t')})"
                 for m, v in (meta.get("model_intelligence") or {}).items() if v.get("trend") == "deteriorating"]
    breaks = [f"{f}: Strukturbruch um {v['structural_break'].get('at')} (CUSUM {v['structural_break'].get('stat')})"
              for f, v in decay.items() if (v.get("structural_break") or {}).get("break")]
    dq = []
    health = inp.get("health") or {}
    for k, v in health.items():
        if isinstance(v, dict) and str(v.get("status", "")).upper() in ("FAIL", "WARN", "STALE"):
            dq.append(f"{k}: {v.get('status')}")
    dis = meta.get("disagreement") or {}
    valuable = [f"{k}: {v['accepted']}/{v['experiments']} akzeptiert, OOS {v['oos_contribution']}" for k, v in rv.items()
                if v["research_efficiency"] in ("HIGH", "MEDIUM")]
    waste = [f"{k}: {v['accepted']}/{v['experiments']} akzeptiert" for k, v in rv.items()
             if v["research_efficiency"] == "LOW" and v["experiments"] >= 5]
    calib_label = "GOOD" if cal.get("interval_calibrated") and (ref.get("ece") or 1) <= 0.05 else "WEAK"
    return {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "self_assessment": {"overall_calibration": calib_label,
                            "world_model_uncertainty": ("HIGH" if (wc.get("uncertainty") or 0) >= 0.6 else
                                                        "MODERATE" if (wc.get("uncertainty") or 0) >= 0.3 else "LOW"),
                            "world_model_validation": ((world.get("validation") or {}).get("verdict")),
                            "meta_learning": (meta.get("decision") or {}).get("verdict"),
                            "next_architecture": nextv.get("decision"),
                            "safe_mode": sm.get("active")},
        "what_do_we_know": know, "what_are_we_uncertain_about": uncertain,
        "where_are_we_systematically_wrong": wrong, "which_assumptions_are_stale": stale,
        "which_models_are_redundant": redundant, "which_features_are_decaying": decaying + breaks,
        "where_is_data_quality_low": dq,
        "where_is_model_disagreement_high": [f"aktuell {dis.get('current_level')} (mittlere Rang-Streuung {dis.get('current_mean_rank_sd')})"]
        if dis.get("current_level") else [],
        "research_tracks_with_real_oos_value": valuable, "research_tracks_wasting_resources": waste,
        "highest_research_priority": ((inp.get("director") or {}).get("candidates") or [{}])[0].get("question"),
        "feature_decay": decay, "model_decay": model_decay, "research_value": rv, "safe_mode": sm,
    }


def run() -> dict:
    inp = {"meta": _load("meta_learning.json", {}), "nextv": _load("next_validation.json", {}),
           "world": _load("world_model.json", {}), "ml": _load("ml_research.json", {}),
           "hyp": _load("hypothesis_db.json", {}), "director": _load("research_candidates.json", {})}
    hp = Path("outputs/external_data/health/source_health.json")
    try:
        inp["health"] = json.loads(hp.read_text()) if hp.exists() else {}
    except (OSError, json.JSONDecodeError) as e:
        log.warning(f"meta_cognition: Source-Health nicht lesbar ({e})")
        inp["health"] = {}
    decay = {}
    cache = os.environ.get("ML_PANEL_CACHE")
    if cache and Path(cache).exists():
        with open(cache, "rb") as fh:
            decay = feature_decay(pickle.load(fh))  # noqa: S301 – eigene, im selben Job erzeugte Datei
    model_decay = {m: {"trend": v.get("trend"), "trend_t": v.get("trend_t"), "recent_ic": v.get("recent_ic"),
                       "prior_ic": v.get("prior_ic")} for m, v in ((inp["meta"].get("model_intelligence")) or {}).items()}
    rv = research_value(inp["hyp"], inp["ml"], inp["meta"])
    fwd = [e.get("actual_return") for e in _memory_outcomes()]
    sm = safe_mode(inp["meta"], inp["world"], inp["health"], [x for x in fwd if x is not None], inp["ml"])
    st = machine_state(inp, decay, model_decay, rv, sm)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "safe_mode.json").write_text(json.dumps(sm, indent=1, ensure_ascii=False))
    (OUT / "machine_state.json").write_text(json.dumps(st, indent=1, ensure_ascii=False, default=str))
    (OUT / "machine_state.md").write_text(render_md(st), encoding="utf-8")
    return st


def _memory_outcomes() -> list[dict]:
    try:
        from modules import prediction_memory as pm
        m = pm.load_memory()
        if m.empty or "actual_return" not in m:
            return []
        return m.sort_values("signal_date")[["actual_return"]].dropna().to_dict("records")
    except (OSError, ValueError, KeyError) as e:
        log.warning(f"meta_cognition: Prognose-Gedächtnis nicht lesbar ({e})")
        return []


def render_md(st: dict) -> str:
    L = [f"# Machine Intelligence State – {st['generated']}", "", "## Selbsteinschätzung", ""]
    L += [f"- {k}: **{v}**" for k, v in st["self_assessment"].items()]
    for key, title in (("what_do_we_know", "Was wissen wir?"), ("what_are_we_uncertain_about", "Worüber sind wir unsicher?"),
                       ("where_are_we_systematically_wrong", "Wo liegen wir systematisch falsch?"),
                       ("which_assumptions_are_stale", "Welche Annahmen sind veraltet?"),
                       ("which_models_are_redundant", "Welche Modelle sind redundant?"),
                       ("which_features_are_decaying", "Welche Merkmale verlieren Kraft?"),
                       ("where_is_data_quality_low", "Wo ist die Datenqualität niedrig?"),
                       ("where_is_model_disagreement_high", "Wo ist die Uneinigkeit hoch?"),
                       ("research_tracks_with_real_oos_value", "Research-Tracks mit echtem OOS-Wert"),
                       ("research_tracks_wasting_resources", "Research-Tracks ohne Ertrag")):
        L += ["", f"## {title}", ""] + ([f"- {x}" for x in st.get(key) or []] or ["- (keine gemessene Aussage)"])
    L += ["", f"Höchste Research-Priorität: {st.get('highest_research_priority')}", "",
          f"Safe Mode: **{'AKTIV' if st['safe_mode']['active'] else 'aus'}** – {st['safe_mode']['reasons'] or '–'}"]
    return "\n".join(L) + "\n"


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    print(render_md(run()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

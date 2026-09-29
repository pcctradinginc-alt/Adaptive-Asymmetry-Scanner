"""
modules/blind_spots.py – Unknown-Unknown-Detektor (NUR SHADOW)

Sucht systematisch Situationen, in denen das System unerwartet schlecht ist:
Große Fehlprognosen = Top-Dezil-Positionen des Referenz-Ensembles, deren
Netto-Relativrendite (rel20) im schlechtesten Dezil liegt. Für messbare
Eigenschaften (Sektor, Volatilität, Liquidität, Momentum, jüngste Bewegung,
Nähe zum Hoch, Beta, Lotterie-Profil, VIX-Regime, Trend) und deren Paare wird
der Lift gegenüber allen Top-Dezil-Positionen berechnet; Signifikanz per
Binomial-z, Benjamini-Hochberg. Cluster = Lift >= 1.5, n >= 30, signifikant.

Nutzen wird gemessen (config/next_protocol.yaml, blind_spots.keep_rule):
Cluster aus den Vorjahren -> Positionen im Folgejahr meiden -> Δ Expectancy.
Cluster fließen als Research-Tracks in den Research Director.
"""

from __future__ import annotations

import logging
import math
from itertools import combinations

import numpy as np
import pandas as pd

from modules import meta_learning as meta
from modules import ml_research as ml

log = logging.getLogger(__name__)

BS = meta.yaml.safe_load((ml.PROTOCOL_PATH.parent / "next_protocol.yaml").read_text(encoding="utf-8"))["blind_spots"]


def properties(df: pd.DataFrame) -> pd.DataFrame:
    """Messbare Eigenschaften je Zeile (Ränge sind um 0 zentriert)."""
    p = pd.DataFrame(index=df.index)
    p["sector"] = df.get("sector", pd.Series("unknown", index=df.index)).fillna("unknown")
    tert = lambda s, lo, hi: np.where(s < -0.17, lo, np.where(s > 0.17, hi, "mid"))  # noqa: E731
    p["volatility"] = tert(df["vol_60"], "low_vol", "high_vol")
    p["liquidity"] = np.where(df["log_dollar_vol"] > 0, "liquid", "less_liquid")
    p["momentum"] = tert(df["mom_12_1"], "loser_12m", "winner_12m")
    p["recent_move"] = np.where(df["ret_5d"].abs() > 0.4, "extreme_5d_move", "normal_5d_move")
    p["near_high"] = np.where(df["dist_52w_high"] > 0.3, "near_52w_high", "below_52w_high")
    p["beta"] = tert(df["beta_126"], "low_beta", "high_beta")
    p["lottery"] = np.where(df["max_ret_21"] > 0.4, "lottery_profile", "no_lottery")
    p["vix"] = np.where(df["vix"] >= 20, "vix_ge_20", "vix_lt_20")
    p["trend"] = np.where(df["spy_trend_200"] > 0, "uptrend", "downtrend")
    return p


def top_rows(res: pd.DataFrame, score: str) -> pd.DataFrame:
    rk = res.groupby("date")[score].rank(pct=True)
    return res[(rk >= 1 - ml.TOP_Q) & res[meta.PROB_TARGET].notna()]


def find_clusters(top: pd.DataFrame, q: float = BS["big_error_quantile"]) -> list[dict]:
    if len(top) < 200:
        return []
    thr = top[meta.PROB_TARGET].quantile(q)
    big = top[meta.PROB_TARGET] <= thr
    base = float(big.mean())
    props = properties(top)
    tests = []
    single = [(c, v) for c in props.columns for v in props[c].unique()]
    combos = [((a, va), (b, vb)) for (a, va), (b, vb) in combinations(single, 2) if a != b]
    for cond in [((c, v),) for c, v in single] + combos:
        m = np.ones(len(top), dtype=bool)
        for c, v in cond:
            m &= (props[c].to_numpy() == v)
        n_all = int(m.sum())
        n_big = int((m & big.to_numpy()).sum())
        if n_all < 50 or n_big < BS["min_cluster_n"]:
            continue
        rate = n_big / n_all
        lift = rate / base if base > 0 else 0
        z = (rate - base) / math.sqrt(base * (1 - base) / n_all)
        p = 0.5 * math.erfc(z / math.sqrt(2))
        tests.append({"properties": {c: v for c, v in cond}, "n": n_big, "n_all": n_all, "lift": lift, "z": z, "p": p,
                      "typical_error": float(top.loc[m & big.to_numpy(), meta.PROB_TARGET].mean()),
                      "segment_expectancy": float(top.loc[m, meta.PROB_TARGET].mean())})
    if not tests:
        return []
    from modules.research_lab import benjamini_hochberg
    sig = benjamini_hochberg({str(i): t["p"] for i, t in enumerate(tests)}, BS["fdr_q"])
    out = [t for i, t in enumerate(tests) if str(i) in sig and t["lift"] >= BS["min_lift"]]
    # redundante Paare entfernen: Paar nur, wenn Lift deutlich über beiden Einzelteilen
    singles = {tuple(t["properties"].items()): t["lift"] for t in out if len(t["properties"]) == 1}
    keep = []
    for t in sorted(out, key=lambda t: -t["lift"]):
        items = tuple(t["properties"].items())
        if len(items) == 2 and max(singles.get((items[0],), 0), singles.get((items[1],), 0)) * 1.2 > t["lift"]:
            continue
        keep.append(t)
    return keep[:15]


def match(df: pd.DataFrame, clusters: list[dict]) -> pd.Series:
    if not clusters or df.empty:
        return pd.Series(False, index=df.index)
    props = properties(df)
    hit = pd.Series(False, index=df.index)
    for c in clusters:
        m = pd.Series(True, index=df.index)
        for k, v in c["properties"].items():
            m &= props[k] == v
        hit |= m
    return hit


def filter_walk_forward(res: pd.DataFrame, base: pd.DataFrame, score: str) -> dict:
    """Cluster aus Basis-OOS-Zeilen mit Label vor dem Testjahr -> Folgejahr meiden."""
    folds = [f for f in sorted(res["fold"].unique()) if f != "locked"] + \
        (["locked"] if "locked" in set(res["fold"]) else [])
    rows, filt = [], []
    for f in folds:
        start = ml.LOCKED_FROM if f == "locked" else pd.Timestamp(year=int(f), month=1, day=1)
        hist = base[(base["label_end_20"] < start)]
        if score not in hist:
            hist = hist.assign(**{score: meta.score_static(hist, [c[2:] for c in hist.columns if c.startswith("r_")])})
        cl = find_clusters(top_rows(hist, score))
        cur = res[res["fold"] == f].copy()
        bad = match(cur, cl)
        cur[f"{score}__bs"] = cur[score].where(~bad)
        filt.append(cur)
        rows.append({"fold": f, "n_clusters": len(cl), "share_excluded_top": round(float(
            bad[cur.groupby("date")[score].rank(pct=True) >= 0.9].mean()), 4) if len(cur) else None})
    allr = pd.concat(filt)
    pa, pb = meta.positions(allr, score), meta.positions(allr, f"{score}__bs")
    dev_a, dev_b = pa[pa["date"] < ml.LOCKED_FROM], pb[pb["date"] < ml.LOCKED_FROM]
    boot = meta.bootstrap_delta(meta.monthly_series(dev_b), meta.monthly_series(dev_a))
    ma, mb = meta.portfolio_metrics(dev_a), meta.portfolio_metrics(dev_b)
    keep = bool(boot) and boot["ci_monthly_mean"][0] > 0 and (mb.get("hit_rate") or 0) >= (ma.get("hit_rate") or 0)
    return {"per_fold": rows, "reference": ma, "filtered": mb, "bootstrap": boot,
            "locked": {"reference": meta.portfolio_metrics(pa[pa["date"] >= ml.LOCKED_FROM]).get("expectancy"),
                       "filtered": meta.portfolio_metrics(pb[pb["date"] >= ml.LOCKED_FROM]).get("expectancy")},
            "verdict": "KEEP" if keep else ("MODIFY" if (mb.get("expectancy") or 0) > (ma.get("expectancy") or 0) else "REJECT"),
            "scores": allr[["date", "ticker", f"{score}__bs"]]}


def describe(clusters: list[dict], failure_profiles: dict) -> list[dict]:
    """Ausgabe-Format UNKNOWN_CLUSTER_###; Abdeckung = ob ein Modell-Failure-
    Profil das Segment bereits als 'fails' kennt."""
    known = {seg.split(" (")[0] for p in (failure_profiles or {}).values() for seg in p.get("fails", [])}
    out = []
    for i, c in enumerate(clusters, 1):
        vals = set(c["properties"].values())
        covered = bool(vals & known)
        out.append({"id": f"UNKNOWN_CLUSTER_{i:03d}", "n": c["n"], "n_segment": c["n_all"],
                    "typical_error": round(c["typical_error"], 4), "segment_expectancy": round(c["segment_expectancy"], 4),
                    "lift": round(c["lift"], 2), "p": c["p"], "common_properties": c["properties"],
                    "existing_model_coverage": "MEDIUM" if covered else "LOW",
                    "recommendation": "dedizierten Research-Track anlegen" if not covered else "bekannt – Failure-Profil beobachten"})
    return out

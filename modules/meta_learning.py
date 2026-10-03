"""
modules/meta_learning.py – Meta-Learning-Schicht (NUR SHADOW)

    python -m modules.meta_learning        (CI: .github/workflows/ml_research.yml, full)

Frage: "Welchem Basismodell sollte ich unter den aktuellen Bedingungen wie stark
vertrauen?" – und liefert die Kombination OUT-OF-SAMPLE tatsächlich mehr als
der bisherige Champion (statisches Ensemble)?

Präregistrierung/Gate: config/meta_protocol.yaml (geschützt). Gemeinsame
Konstanten (Kosten, Locked-Holdout): config/research_protocol.yaml.

Aufbau (keine zweite Infrastruktur: Feature-Store, Modelle, Registry und
Kennzahlen kommen aus modules/ml_research.py):
  1. Basis-OOS-Prognosen: jedes registrierte Modell jahresweise walk-forward
     (Training nur mit Labels vor dem Testjahr) + ein Locked-Fold.
  2. Meta-Merkmale je (Stichtag, Modell), nur aus damals Bekanntem:
     Regime (VIX, Trend, Zinsen, Kurve), historische Modellleistung (Rank-IC
     der letzten 13/52 Wochen, NUR Stichtage mit fertigem Label), Kalibrierungs-
     Steigung, Modell-Uneinigkeit.
  3. Varianten: bestes Einzelmodell (ex ante), statisches Ensemble,
     IC-gewichtetes Ensemble, Meta-Regime-Gewichte (primär), Meta-Stacking.
  4. Meta-Walk-Forward: Meta-Learner für Jahr Y nur mit Basis-OOS-Zeilen,
     deren Label vor Y endet (kein Stacking-Leakage).
  5. Kennzahlen, Bootstrap-Deltas, Robustheit, Ablationen, Kalibrierungs-
     Buckets, Disagreement-Test, Failure-Profile, Promotion-Gate, Safe-Mode.

Ergebnis: outputs/research/meta_learning.{json,md}, meta_state.json,
hc_thresholds.json. Es gibt keine Orderausführung und keine Produktionswirkung.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from modules import ml_research as ml

log = logging.getLogger(__name__)

META_PROTOCOL_PATH = Path(__file__).resolve().parent.parent / "config" / "meta_protocol.yaml"
MP = yaml.safe_load(META_PROTOCOL_PATH.read_text(encoding="utf-8"))
OUT_JSON = ml.OUT_DIR / "meta_learning.json"
OUT_MD = ml.OUT_DIR / "meta_learning.md"
STATE_PATH = ml.OUT_DIR / "meta_state.json"
HC_THRESHOLDS_PATH = ml.OUT_DIR / "hc_thresholds.json"

STATE_FEATURES = ("vix", "vix_chg_21", "spy_trend_200", "spy_mom_63", "tnx", "curve_10y_3m")
TRAIL_W = int(MP["periods"]["trailing_weeks"])
RECENT_W = int(MP["periods"]["recent_weeks"])
_STACK_PARAMS = {"max_iter": 200, "learning_rate": 0.05, "max_depth": 3, "min_samples_leaf": 500}
RIDGE_ALPHA = 10.0
LAST_RUN: dict = {}          # letzte Auswertung (res, meta_d, models) für nachgelagerte Module im selben Job


def _mcol(mid: str) -> str:
    return f"r_{mid}"


# ── 1. Basis-OOS-Prognosen ───────────────────────────────────────────────────

def base_oos(panel: pd.DataFrame, specs: list[dict], locked_from: pd.Timestamp = ml.LOCKED_FROM,
             first_test_year: int = ml.FIRST_TEST_YEAR, with_counterfactuals: bool = False) -> tuple[pd.DataFrame, dict]:
    """-> (Zeilen mit Querschnittsrang r_<id> und Rohprognose p_<id> je Modell,
    Provenienz je Fold). Fold = Testjahr oder 'locked'."""
    folds = []
    for y in range(first_test_year, (locked_from - pd.Timedelta(days=1)).year + 1):
        ts = pd.Timestamp(year=y, month=1, day=1)
        te = min(pd.Timestamp(year=y + 1, month=1, day=1), locked_from)
        folds.append((str(y), ts, te, ts))
    folds.append(("locked", locked_from, pd.Timestamp.max, locked_from))
    parts, prov = [], {}
    for name, ts, te, cutoff in folds:
        test = panel[(panel["date"] >= ts) & (panel["date"] < te) & panel["label_end_20"].notna()]
        if name != "locked":
            test = test[test["label_end_20"] < locked_from]
        if test.empty:
            continue
        test = test.copy()
        n_models = 0
        prov[name] = {"train_label_end_before": str(cutoff.date()), "n_test": int(len(test))}
        for spec in specs:
            mid = spec["id"]
            train = ml.purged(panel, cutoff, ml._horizon(spec))
            if spec.get("model") != "rule" and train[spec["target"]].notna().sum() < 1000:
                test[f"p_{mid}"] = np.nan
                continue
            params = ml.select_params(spec, train, cutoff)
            m = ml.fit(spec, train, params)
            test[f"p_{mid}"] = m.predict(test)
            if with_counterfactuals:                    # Szenario-Ränge je Modell aufsummieren
                from modules import counterfactual as cf
                for case in cf.ALL_CASES:
                    r = pd.Series(m.predict(cf.apply_case(test, case)), index=test.index).groupby(test["date"]).rank(pct=True)
                    test[f"cfsum_{case}"] = test.get(f"cfsum_{case}", 0.0) + r
            n_models = n_models + 1 if with_counterfactuals else n_models
            prov[name][mid] = {"params": params,
                               "train_max_label_end": str(train["label_end_20"].max().date()) if len(train) else None}
        test["fold"] = name
        if with_counterfactuals and n_models:
            from modules import counterfactual as cf
            for case in cf.ALL_CASES:
                test[f"cfrank_{case}"] = (test[f"cfsum_{case}"] / n_models).groupby(test["date"]).rank(pct=True)
                test = test.drop(columns=[f"cfsum_{case}"])
        parts.append(test)
    df = pd.concat(parts, ignore_index=True)
    for spec in specs:
        mid = spec["id"]
        df[_mcol(mid)] = df.groupby("date")[f"p_{mid}"].rank(pct=True)
    return df, prov


# ── 2. Meta-Merkmale ─────────────────────────────────────────────────────────

def per_date_ic(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Rank-IC je Stichtag und Modell + Label-Ende des Stichtags."""
    rows = {}
    for d, g in df.groupby("date"):
        y = g["fwd_xs_20"]
        r = {"label_end": g["label_end_20"].max()}
        for mid in models:
            x = g[_mcol(mid)]
            ok = x.notna() & y.notna()
            r[mid] = x[ok].rank().corr(y[ok].rank()) if ok.sum() >= ml.MIN_CROSS_SECTION else np.nan
            xc = x[ok] - 0.5
            r[f"slope_{mid}"] = float((xc * y[ok]).sum() / (xc * xc).sum()) if ok.sum() >= ml.MIN_CROSS_SECTION else np.nan
        rows[d] = r
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def trailing_stats(ic: pd.DataFrame, models: list[str], window: int, as_of_dates) -> pd.DataFrame:
    """Mittlerer IC der letzten `window` Stichtage, deren Label VOR dem
    jeweiligen Stichtag feststand (PIT, keine unfertigen Labels)."""
    out = {}
    idx = ic.index
    le = ic["label_end"]
    for d in as_of_dates:
        past = ic[(idx < d) & (le < d)].tail(window)
        r = {}
        for mid in models:
            s = past[mid].dropna()
            r[f"tic_{mid}"] = float(s.mean()) if len(s) >= 8 else np.nan
            sl = past[f"slope_{mid}"].dropna()
            r[f"tslope_{mid}"] = float(sl.mean()) if len(sl) >= 8 else np.nan
            r[f"thit_{mid}"] = float((s > 0).mean()) if len(s) >= 8 else np.nan
        out[d] = r
    return pd.DataFrame.from_dict(out, orient="index")


def disagreement_features(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    R = df[[_mcol(m) for m in models]]
    P = df[[f"p_{m}" for m in models if not m.startswith("momentum")]]
    bull = (R > 0.5).mean(axis=1)
    return pd.DataFrame({
        "dis_rank_sd": R.std(axis=1),
        "dis_rank_range": R.max(axis=1) - R.min(axis=1),
        "dis_bull_bear": np.minimum(bull, 1 - bull),                 # 0 = einig, 0.5 = maximal gespalten
        "dis_pred_sd": P.std(axis=1) if P.shape[1] >= 2 else np.nan,  # Streuung der erwarteten Renditen
    }, index=df.index)


def build_meta_frame(df: pd.DataFrame, models: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if PROB_TARGET not in df:
        df = add_rel_target(df)
    df = df.join(disagreement_features(df, models))
    ic = per_date_ic(df, models)
    dates = sorted(df["date"].unique())
    tr = trailing_stats(ic, models, TRAIL_W, dates)
    rc = trailing_stats(ic, models, RECENT_W, dates).add_prefix("recent_")
    date_state = df.groupby("date")[list(STATE_FEATURES)].first()
    date_state["dis_mean"] = df.groupby("date")["dis_rank_sd"].mean()
    meta_d = date_state.join(tr).join(rc).join(ic)
    return df.join(meta_d[[c for c in meta_d.columns if c.startswith(("tic_", "tslope_", "thit_", "recent_"))]]
                   .rename_axis("date"), on="date"), meta_d


# ── 3. Varianten ─────────────────────────────────────────────────────────────

def _ridge_fit(X: np.ndarray, y: np.ndarray):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    m = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                      Ridge(alpha=RIDGE_ALPHA))
    m.fit(X, y)
    resid = y - m.predict(X)
    n, p = X.shape
    sd = float(np.std(resid, ddof=1)) * math.sqrt(1 + p / max(n, 1))
    return m, sd


def _phi(x):
    return 0.5 * (1 + np.vectorize(math.erf)(np.asarray(x, float) / math.sqrt(2)))


def regime_weight_model(meta_d: pd.DataFrame, models: list[str], train_end: pd.Timestamp,
                        use_regime=True, use_trailing=True, use_disagreement=True,
                        state_cols: tuple | list = STATE_FEATURES) -> dict:
    """Je Basismodell: Ridge  IC(d) ~ Regime + eigene Historie + Uneinigkeit,
    trainiert nur auf Stichtagen, deren Label vor train_end feststand."""
    tr = meta_d[(meta_d["label_end"] < train_end)]
    out = {}
    for mid in models:
        cols = []
        if use_regime:
            cols += [c for c in state_cols if c in meta_d]
        if use_trailing:
            cols += [f"tic_{mid}", f"tslope_{mid}", f"thit_{mid}", f"recent_tic_{mid}"]
        if use_disagreement:
            cols += ["dis_mean"]
        t = tr[tr[mid].notna()]
        if len(t) < 60 or not cols:
            out[mid] = None
            continue
        m, sd = _ridge_fit(t[cols].to_numpy(float), t[mid].to_numpy(float))
        out[mid] = {"model": m, "cols": cols, "sd": sd, "n": int(len(t))}
    return out


def regime_weights(wm: dict, meta_d: pd.DataFrame, dates) -> pd.DataFrame:
    """-> je (Stichtag, Modell): Gewicht, erwartete Zuverlässigkeit (IC),
    P(Modell liefert Mehrwert), Konfidenz des Gewichts."""
    rows = []
    for d in dates:
        s = meta_d.loc[[d]]
        preds = {}
        for mid, w in wm.items():
            if w is None:
                continue
            p = float(w["model"].predict(s[w["cols"]].to_numpy(float))[0])
            preds[mid] = (p, w["sd"], w["n"])
        pos = {k: max(v[0], 0.0) for k, v in preds.items()}
        tot = sum(pos.values())
        for mid, (p, sd, n) in preds.items():
            rows.append({"date": d, "model": mid,
                         "weight": pos[mid] / tot if tot > 0 else 1.0 / max(len(preds), 1),
                         "fallback_equal": tot <= 0,
                         "expected_reliability": p, "p_adds_value": float(_phi(p / sd)) if sd > 0 else np.nan,
                         "confidence": float(abs(p) / sd) if sd > 0 else np.nan})
    return pd.DataFrame(rows)


def score_static(df, models):
    return df[[_mcol(m) for m in models]].mean(axis=1)


def score_weighted(df, weights_by_date: pd.DataFrame, models):
    W = weights_by_date.pivot(index="date", columns="model", values="weight").reindex(columns=models).fillna(0.0)
    Wr = W.reindex(df["date"]).to_numpy()
    R = df[[_mcol(m) for m in models]].fillna(0.5).to_numpy() - 0.5
    return pd.Series((Wr * R).sum(axis=1), index=df.index)


def trailing_ic_weights(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows = []
    for d, g in df.groupby("date"):
        vals = {m: g[f"tic_{m}"].iloc[0] for m in models}
        pos = {m: max(v, 0.0) if np.isfinite(v) else 0.0 for m, v in vals.items()}
        tot = sum(pos.values())
        for m in models:
            rows.append({"date": d, "model": m, "weight": pos[m] / tot if tot > 0 else 1.0 / len(models)})
    return pd.DataFrame(rows)


STACK_BASE = ("dis_rank_sd", "dis_rank_range", "dis_bull_bear", "dis_pred_sd")


def stacking_cols(models, use_regime=True, use_disagreement=True, use_trailing=True, use_sector=True):
    cols = [_mcol(m) for m in models] + ["log_dollar_vol", "vol_60"]
    if use_disagreement:
        cols += list(STACK_BASE)
    if use_regime:
        cols += list(STATE_FEATURES)
    if use_trailing:
        cols += [f"tic_{m}" for m in models]
    if use_sector:
        cols += ["sector_code"]
    return cols


def fit_stacking(train: pd.DataFrame, cols: list[str]):
    """-> (Modell, tatsächlich genutzte Spalten). Im Training komplett leere
    Spalten (z.B. Historie in den ersten Jahren) werden ausgelassen."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    t = train[train["fwd_xs_20"].notna()]
    cols = [c for c in cols if t[c].notna().any()]
    cat = [cols.index("sector_code")] if "sector_code" in cols else None
    m = HistGradientBoostingRegressor(random_state=0, early_stopping=False, categorical_features=cat, **_STACK_PARAMS)
    m.fit(t[cols].to_numpy(float), t["fwd_xs_20"].clip(-0.5, 0.5).to_numpy(float))
    return m, cols


def best_single(meta_d: pd.DataFrame, models: list[str], train_end) -> str:
    tr = meta_d[meta_d["label_end"] < train_end]
    means = {m: tr[m].mean() for m in models if tr[m].notna().sum() >= 20}
    return max(means, key=means.get) if means else models[0]


# ── 4. Meta-Walk-Forward ─────────────────────────────────────────────────────

VARIANT_ABLATIONS = {
    "meta_regime_weights": {},
    "meta_regime_weights__no_regime": {"use_regime": False},
    "meta_regime_weights__no_failure_memory": {"use_trailing": False},
    "meta_regime_weights__no_disagreement": {"use_disagreement": False},
    "meta_stacking": {},
    "meta_stacking__no_regime": {"use_regime": False},
    "meta_stacking__no_disagreement": {"use_disagreement": False},
    "meta_stacking__no_failure_memory": {"use_trailing": False},
    "meta_stacking__no_sector": {"use_sector": False},
}


def meta_walk_forward(df: pd.DataFrame, meta_d: pd.DataFrame, models: list[str],
                      first_meta_year: int = int(MP["periods"]["first_meta_test_year"])) -> tuple[pd.DataFrame, dict]:
    """Scores aller Varianten auf Meta-Testzeilen (Jahre ab first_meta_year + locked)."""
    folds = [f for f in sorted(df["fold"].unique()) if f != "locked" and int(f) >= first_meta_year] + \
        (["locked"] if "locked" in set(df["fold"]) else [])
    out, prov, weights_log = [], {}, []
    for f in folds:
        test = df[df["fold"] == f].copy()
        start = test["date"].min() if f != "locked" else ml.LOCKED_FROM
        train_end = pd.Timestamp(year=int(f), month=1, day=1) if f != "locked" else ml.LOCKED_FROM
        train = df[(df["fold"] != "locked") & (df["label_end_20"] < train_end)]
        prov[f] = {"meta_train_label_end_before": str(train_end.date()), "n_meta_train_rows": int(len(train)),
                   "meta_train_max_label_end": str(train["label_end_20"].max().date()) if len(train) else None,
                   "test_start": str(pd.Timestamp(start).date())}
        test["s_static_equal"] = score_static(test, models)
        bs = best_single(meta_d, models, train_end)
        prov[f]["best_single"] = bs
        test["s_best_single_ex_ante"] = test[_mcol(bs)]
        test["s_trailing_ic_weighted"] = score_weighted(test, trailing_ic_weights(test, models), models)
        dates = sorted(test["date"].unique())
        for name, kw in VARIANT_ABLATIONS.items():
            if name.startswith("meta_regime_weights"):
                wm = regime_weight_model(meta_d, models, train_end, **kw)
                w = regime_weights(wm, meta_d, dates)
                test[f"s_{name}"] = score_weighted(test, w, models)
                if name == "meta_regime_weights":
                    weights_log.append(w.assign(fold=f))
            else:
                cols = stacking_cols(models, **kw)
                if len(train) < 5000:
                    test[f"s_{name}"] = np.nan
                    continue
                m, used = fit_stacking(train, cols)
                test[f"s_{name}"] = m.predict(test[used].to_numpy(float))
        out.append(test)
    res = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    wl = pd.concat(weights_log, ignore_index=True) if weights_log else pd.DataFrame()
    return res, {"folds": prov, "weights": wl}


# ── 5. Kennzahlen ────────────────────────────────────────────────────────────

def positions(df: pd.DataFrame, score: str, q: float = ml.TOP_Q, every: int = 1) -> pd.DataFrame:
    """Top-Quantil je Stichtag als einzelne Positionen (netto gegen Querschnittsmittel)."""
    rows = []
    dates = sorted(df["date"].unique())[::every]
    for d in dates:
        g = df[(df["date"] == d) & df[score].notna() & df["fwd_xs_20"].notna()]
        if len(g) < ml.MIN_CROSS_SECTION:
            continue
        k = max(1, int(round(len(g) * q)))
        u = g["fwd_xs_20"].mean()
        strong = set(g.nlargest(k, "fwd_xs_20")["ticker"])
        good = set(g.nlargest(max(1, int(round(len(g) * 0.2))), "fwd_xs_20")["ticker"])
        top = g.nlargest(k, score)
        for _, r in top.iterrows():
            rows.append({"date": d, "ticker": r["ticker"], "sector": r.get("sector", "unknown"),
                         "ret": r["fwd_xs_20"] - u - 2 * ml.COST_BASE, "mfe": r.get("mfe_20"), "mae": r.get("mae_20"),
                         "in_top_quintile": r["ticker"] in good, "is_strong": r["ticker"] in strong,
                         "n_strong": len(strong), "vix": r.get("vix"), "spy_trend_200": r.get("spy_trend_200"),
                         "year": pd.Timestamp(d).year})
    return pd.DataFrame(rows)


def portfolio_metrics(pos: pd.DataFrame) -> dict:
    if pos.empty:
        return {"n_trades": 0}
    coh = pos.groupby("date")["ret"].mean()
    mon = coh.groupby(coh.index.to_period("M")).mean()
    n_m = len(mon)
    eq = (1 + mon).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    cagr = float(eq.iloc[-1] ** (12 / n_m) - 1) if n_m else np.nan
    sd = mon.std(ddof=1)
    down = mon[mon < 0]
    dsd = math.sqrt(float((down ** 2).mean())) if len(down) else np.nan
    r = pos["ret"]
    w, l = r[r > 0], r[r <= 0]
    sets = pos.groupby("date")["ticker"].apply(set)
    turn = [1 - len(a & b) / max(len(b), 1) for a, b in zip(sets.iloc[:-1], sets.iloc[1:])]
    strong_tot = pos.groupby("date")["n_strong"].first().sum()
    return {
        "cagr": ml._r(cagr, 4), "sharpe": ml._r(mon.mean() / sd * math.sqrt(12), 3) if n_m > 2 and sd > 0 else None,
        "sortino": ml._r(mon.mean() / dsd * math.sqrt(12), 3) if dsd and dsd > 0 else None,
        "max_dd": ml._r(dd, 4), "calmar": ml._r(cagr / abs(dd), 3) if dd < 0 else None,
        "profit_factor": ml._r(w.sum() / abs(l.sum()), 3) if l.sum() < 0 else None,
        "hit_rate": ml._r((r > 0).mean(), 4), "avg_winner": ml._r(w.mean(), 5), "avg_loser": ml._r(l.mean(), 5),
        "payoff": ml._r(w.mean() / abs(l.mean()), 3) if len(l) and l.mean() < 0 else None,
        "expectancy": ml._r(r.mean(), 5), "precision_at_k": ml._r(pos["in_top_quintile"].mean(), 4),
        "recall_strong": ml._r(pos["is_strong"].sum() / strong_tot, 4) if strong_tot else None,
        "turnover": ml._r(np.mean(turn), 4) if turn else None, "exposure": 1.0,
        "n_trades": int(len(pos)), "n_cohorts": int(coh.shape[0]), "n_months": int(n_m),
        "avg_mfe": ml._r(pos["mfe"].mean(), 4), "avg_mae": ml._r(pos["mae"].mean(), 4),
    }


def monthly_series(pos: pd.DataFrame) -> pd.Series:
    coh = pos.groupby("date")["ret"].mean()
    return coh.groupby(coh.index.to_period("M")).mean()


PROB_TARGET = "rel20"   # Netto-Rendite relativ zum Querschnitt (die gehandelte Größe), nicht "schlägt SPY"


def add_rel_target(df: pd.DataFrame) -> pd.DataFrame:
    """rel20 = fwd_xs_20 − Querschnittsmittel des Stichtags − Round-Trip-Kosten.
    Bis 2026-09-29 bezog sich die Wahrscheinlichkeit auf fwd_xs_20 > 0 (schlägt
    SPY); das war in Mega-Cap-Jahren strukturell < 50 % und passte nicht zur
    Bewertung der Positionen (Validierung: überkonfidente Buckets)."""
    df = df.copy()
    df[PROB_TARGET] = df["fwd_xs_20"] - df.groupby("date")["fwd_xs_20"].transform("mean") - 2 * ml.COST_BASE
    return df


def lagged_calibration(res: pd.DataFrame, score: str) -> pd.DataFrame:
    """P(rel20 > 0) aus dem Score: isotonische Abbildung, gefittet auf den
    OOS-Scores des VORJAHRES derselben Variante (echt out-of-sample).
    Zusätzlich erwartete Netto-Relativrendite je Score (ebenfalls Vorjahr)."""
    from sklearn.isotonic import IsotonicRegression
    out = []
    years = sorted({f for f in res["fold"].unique() if f != "locked"})
    order = years + (["locked"] if "locked" in set(res["fold"]) else [])
    for i, f in enumerate(order):
        if i == 0:
            continue
        prev = res[(res["fold"] == order[i - 1]) & res[score].notna() & res[PROB_TARGET].notna()]
        cur = res[(res["fold"] == f) & res[score].notna()].copy()
        if "label_end_20" in prev and not cur.empty:   # Purge: nur Labels, die VOR dem Testbeginn feststanden (Audit F05)
            prev = prev[prev["label_end_20"] < cur["date"].min()]
        if len(prev) < 1000 or cur.empty:
            continue
        rk_prev = prev.groupby("date")[score].rank(pct=True)
        rk_cur = cur.groupby("date")[score].rank(pct=True)
        iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip").fit(rk_prev, (prev[PROB_TARGET] > 0))
        iso_r = IsotonicRegression(out_of_bounds="clip").fit(rk_prev, prev[PROB_TARGET])
        cur["prob"] = iso.predict(rk_cur)
        cur["exp_xs20"] = iso_r.predict(rk_cur)
        out.append(cur[["date", "ticker", "fold", "prob", "exp_xs20", PROB_TARGET, "fwd_xs_20", "mae_20"]
                       + (["mfe_20"] if "mfe_20" in cur else [])])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def prob_metrics(cal: pd.DataFrame) -> dict:
    if cal.empty:
        return {}
    c = cal[cal[PROB_TARGET].notna()]
    y = (c[PROB_TARGET] > 0).astype(float)
    p = c["prob"].clip(1e-4, 1 - 1e-4)
    bins = np.clip((p * 10).astype(int), 0, 9)
    ece = float(sum(abs(y[bins == b].mean() - p[bins == b].mean()) * (bins == b).mean()
                    for b in range(10) if (bins == b).any()))
    return {"brier": ml._r(((p - y) ** 2).mean(), 5), "log_loss": ml._r(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean(), 5),
            "ece": ml._r(ece, 5), "base_rate": ml._r(y.mean(), 4), "n": int(len(c))}


def probability_validation(buckets: list[dict], min_n: int = 100, tol: float = 0.05) -> dict:
    """Audit F12 / Remediation P1-4: Wahrscheinlichkeiten gelten nur als
    validiert, wenn (a) mindestens zwei Buckets n >= min_n haben, (b) jeder
    davon |realisiert − vorhergesagt| <= tol erfüllt und (c) die realisierte
    Trefferquote über diese Buckets nicht fällt (Monotonie). Sonst werden
    keine High-Confidence-Alerts mit Prozentangaben erzeugt."""
    big = [b for b in buckets if (b.get("n") or 0) >= min_n and b.get("win_rate") is not None]
    reasons = []
    if len(big) < 2:
        reasons.append(f"nur {len(big)} Bucket(s) mit n >= {min_n}")
    bad = [b["bucket"] for b in big if b.get("calibration_error") is None or abs(b["calibration_error"]) > tol]
    if bad:
        reasons.append(f"Kalibrierungsfehler > {tol} in {bad}")
    wins = [b["win_rate"] for b in big]
    if any(b < a for a, b in zip(wins, wins[1:])):
        reasons.append(f"Trefferquote nicht monoton steigend {wins}")
    return {"probability_validated": not reasons, "probability_validation_reasons": reasons}


def calibration_buckets(cal: pd.DataFrame) -> list[dict]:
    edges = MP["calibration_buckets"]
    tol, min_n = MP["calibration_flag_tolerance"], MP["min_bucket_n"]
    out = []
    c = cal[cal[PROB_TARGET].notna()]
    for lo, hi in zip(edges[:-1], edges[1:]):
        g = c[(c["prob"] >= lo) & (c["prob"] < hi)]
        n = len(g)
        rec = {"bucket": f"{int(lo * 100)}–{int(hi * 100)} %" if hi < 1 else f">= {int(lo * 100)} %", "n": n}
        if n:
            win = float((g[PROB_TARGET] > 0).mean())
            pred = float(g["prob"].mean())
            err = win - pred
            rec.update(win_rate=ml._r(win, 4), predicted=ml._r(pred, 4), avg_return=ml._r(g[PROB_TARGET].mean(), 5),
                       median_return=ml._r(g[PROB_TARGET].median(), 5), avg_drawdown=ml._r(g["mae_20"].mean(), 5),
                       avg_mae=ml._r(g["mae_20"].mean(), 5),
                       avg_mfe=ml._r(g["mfe_20"].mean(), 5) if "mfe_20" in g else None,
                       expected_return=ml._r(g["exp_xs20"].mean(), 5),
                       expected_value=ml._r(g[PROB_TARGET].mean(), 5),
                       calibration_error=ml._r(err, 4),
                       flag="low_n" if n < min_n else "overconfident" if err < -tol else
                       "underconfident" if err > tol else "calibrated")
        else:
            rec["flag"] = "empty"
        out.append(rec)
    return out


def breakdowns(pos: pd.DataFrame) -> dict:
    out = {"by_year": {}, "by_regime": {}, "by_sector": {}}
    for y, g in pos.groupby("year"):
        out["by_year"][str(y)] = {"expectancy": ml._r(g["ret"].mean(), 5), "n": int(len(g)),
                                  "hit_rate": ml._r((g["ret"] > 0).mean(), 3)}
    for name, m in (("vix_lt_20", pos["vix"] < 20), ("vix_ge_20", pos["vix"] >= 20),
                    ("spy_uptrend", pos["spy_trend_200"] > 0), ("spy_downtrend", pos["spy_trend_200"] <= 0)):
        g = pos[m.fillna(False)]
        out["by_regime"][name] = {"expectancy": ml._r(g["ret"].mean(), 5), "n_cohorts": int(g["date"].nunique()),
                                  "hit_rate": ml._r((g["ret"] > 0).mean(), 3)}
    for s, g in pos.groupby("sector"):
        if len(g) >= 50:
            out["by_sector"][s] = {"expectancy": ml._r(g["ret"].mean(), 5), "n": int(len(g))}
    return out


def bootstrap_delta(a: pd.Series, b: pd.Series, n: int = MP["bootstrap"]["n"], seed: int = MP["bootstrap"]["seed"],
                    alpha: float = MP["bootstrap"]["alpha_one_sided"] / MP["variants"]["n_meta_variants"]) -> dict:
    """Gepaarter Block-Bootstrap über Monate: Δ Mittel und Δ Sharpe (a − b)."""
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(j) < 12:
        return {}
    rng = np.random.default_rng(seed)
    A, B = j["a"].to_numpy(), j["b"].to_numpy()
    dm, ds = [], []
    for _ in range(n):
        i = rng.integers(0, len(j), len(j))
        a_, b_ = A[i], B[i]
        dm.append(a_.mean() - b_.mean())
        sa, sb = a_.std(ddof=1), b_.std(ddof=1)
        ds.append((a_.mean() / sa if sa > 0 else 0) * math.sqrt(12) - (b_.mean() / sb if sb > 0 else 0) * math.sqrt(12))
    q = [alpha, 1 - alpha]
    return {"delta_monthly_mean": ml._r(A.mean() - B.mean(), 6),
            "ci_monthly_mean": [ml._r(v, 6) for v in np.quantile(dm, q)],
            "ci_sharpe": [ml._r(v, 3) for v in np.quantile(ds, q)], "alpha_one_sided": alpha, "n_months": int(len(j))}


# ── 6. Disagreement-Test, Failure-Profile, Modell-Intelligenz ───────────────

def disagreement_test(res: pd.DataFrame, ref_score: str = "s_static_equal") -> dict:
    """Ist hohe Übereinstimmung mit besseren Ergebnissen verbunden? Innerhalb
    des Top-Dezils der Referenz: niedrige vs. hohe Uneinigkeit (Median-Split)."""
    out = {}
    dev = res[res["fold"] != "locked"]
    for col in ("dis_rank_sd", "dis_bull_bear", "dis_rank_range", "dis_pred_sd"):
        rows = []
        for d, g in dev.groupby("date"):
            g = g[g[ref_score].notna() & g["fwd_xs_20"].notna() & g[col].notna()]
            if len(g) < ml.MIN_CROSS_SECTION:
                continue
            top = g.nlargest(max(2, int(round(len(g) * ml.TOP_Q))), ref_score)
            med = top[col].median()
            lo, hi = top[top[col] <= med], top[top[col] > med]
            if len(lo) and len(hi):
                rows.append({"date": d, "diff": lo["fwd_xs_20"].mean() - hi["fwd_xs_20"].mean()})
        if not rows:
            continue
        s = pd.DataFrame(rows).set_index("date")["diff"]
        m = s.groupby(s.index.to_period("M")).mean()
        y = s.groupby(s.index.year).mean()
        t = m.mean() / m.std(ddof=1) * math.sqrt(len(m)) if len(m) > 2 and m.std(ddof=1) > 0 else None
        supported = t is not None and t >= MP["disagreement_rule"]["min_t"] and \
            (y > 0).mean() >= MP["disagreement_rule"]["must_be_positive_share_years"]
        out[col] = {"low_minus_high_mean": ml._r(s.mean(), 5), "t_months": ml._r(t, 2),
                    "share_years_positive": ml._r((y > 0).mean(), 3), "empirically_supported": bool(supported)}
    out["use_in_score"] = any(v.get("empirically_supported") for v in out.values() if isinstance(v, dict))
    out["probability_disagreement"] = "n/a – Basismodelle liefern Renditen/Ränge, keine Wahrscheinlichkeiten"
    return out


def failure_profiles(res: pd.DataFrame, models: list[str]) -> dict:
    """Wo liefert jedes Modell messbar (t >= 2) bzw. versagt (t <= -2)?
    Segmente je Stichtag/Zeile; IC innerhalb Segment je Stichtag, t über Monate."""
    dev = res[res["fold"] != "locked"].copy()
    dev["vol_bucket"] = pd.cut(dev["vol_60"], [-1, -0.17, 0.17, 1], labels=["low_vol", "mid_vol", "high_vol"])
    dev["liq_bucket"] = np.where(dev["log_dollar_vol"] > 0, "liquid", "less_liquid")
    dev["vix_bucket"] = np.where(dev["vix"] >= 20, "vix_ge_20", "vix_lt_20")
    dev["trend_bucket"] = np.where(dev["spy_trend_200"] > 0, "uptrend", "downtrend")
    dev["mkt_move"] = np.where(dev["spy_mom_63"] > 0, "market_up_3m", "market_down_3m")
    out = {}
    for mid in models:
        prof = []
        for dim in ("vix_bucket", "trend_bucket", "mkt_move", "vol_bucket", "liq_bucket", "sector"):
            for seg, g in dev.groupby(dim, observed=True):
                ics = []
                for d, gd in g.groupby("date"):
                    ok = gd[_mcol(mid)].notna() & gd["fwd_xs_20"].notna()
                    if ok.sum() >= 15:
                        ics.append((d, gd.loc[ok, _mcol(mid)].rank().corr(gd.loc[ok, "fwd_xs_20"].rank())))
                if len(ics) < 24:
                    continue
                s = pd.Series(dict(ics)).dropna()
                m = s.groupby(pd.DatetimeIndex(s.index).to_period("M")).mean()
                t = m.mean() / m.std(ddof=1) * math.sqrt(len(m)) if len(m) > 2 and m.std(ddof=1) > 0 else 0.0
                prof.append({"dimension": dim, "segment": str(seg), "mean_ic": ml._r(s.mean(), 4), "t": ml._r(t, 2),
                             "n_dates": int(len(s)),
                             "verdict": "works" if t >= 2 else "fails" if t <= -2 else "neutral"})
        works = [f"{p['segment']} (IC {p['mean_ic']}, t {p['t']})" for p in prof if p["verdict"] == "works"]
        fails = [f"{p['segment']} (IC {p['mean_ic']}, t {p['t']})" for p in prof if p["verdict"] == "fails"]
        out[mid] = {"segments": prof, "works": works, "fails": fails}
    return out


def model_intelligence(meta_d: pd.DataFrame, models: list[str], current_weights: dict) -> dict:
    out = {}
    lab = meta_d[meta_d["label_end"].notna()]
    for mid in models:
        s = lab[mid].dropna()
        if s.empty:
            continue
        rec, prev = s.tail(RECENT_W), s.iloc[-(RECENT_W + TRAIL_W):-RECENT_W]
        t = None
        if len(rec) > 3 and len(prev) > 3:
            se = math.sqrt(rec.var(ddof=1) / len(rec) + prev.var(ddof=1) / len(prev))
            t = (rec.mean() - prev.mean()) / se if se > 0 else None
        trend = "stable" if t is None or abs(t) < 2 else ("improving" if t > 0 else "deteriorating")
        out[mid] = {"oos_ic": ml._r(s.mean(), 4), "recent_ic": ml._r(rec.mean(), 4), "prior_ic": ml._r(prev.mean(), 4),
                    "trend": trend, "trend_t": ml._r(t, 2),
                    "calibration_slope_recent": ml._r(lab[f"slope_{mid}"].dropna().tail(RECENT_W).mean(), 5)
                    if f"slope_{mid}" in lab else None,
                    "meta_weight": (current_weights.get(mid) or {}).get("weight")}
    return out


def contributions(res: pd.DataFrame, models: list[str]) -> dict:
    """Leave-one-out: Expectancy des statischen Ensembles minus ohne Modell."""
    dev = res[res["fold"] != "locked"]
    base = portfolio_metrics(positions(dev.assign(_s=score_static(dev, models)), "_s")).get("expectancy")
    out = {}
    for mid in models:
        rest = [m for m in models if m != mid]
        e = portfolio_metrics(positions(dev.assign(_s=score_static(dev, rest)), "_s")).get("expectancy")
        out[mid] = ml._r((base or 0) - (e or 0), 5)
    return out


# ── 7. Gate, Safe-Mode, High-Confidence-Schwellen ────────────────────────────

def promotion_gate(res: pd.DataFrame, meta: str, ref: str, pos_meta: pd.DataFrame, pos_ref: pd.DataFrame,
                   cal_meta: dict, cal_ref: dict, leakage_ok: bool) -> dict:
    g = MP["promotion_gate"]
    dev_m = pos_meta[pos_meta["date"] < ml.LOCKED_FROM]
    dev_r = pos_ref[pos_ref["date"] < ml.LOCKED_FROM]
    mm, mr = monthly_series(dev_m), monthly_series(dev_r)
    pm, pr = portfolio_metrics(dev_m), portfolio_metrics(dev_r)
    boot = bootstrap_delta(mm, mr)
    d = (mm - mr).dropna()
    years = d.groupby(d.index.year).mean()
    half = len(d) // 2
    trimmed = d.sort_values().iloc[: int(len(d) * (1 - g["trimmed_share"]))] if len(d) else d
    reg = {}
    bm, br = breakdowns(dev_m)["by_regime"], breakdowns(dev_r)["by_regime"]
    for k in bm:
        if (bm[k].get("n_cohorts") or 0) >= 24 and bm[k].get("expectancy") is not None and br.get(k, {}).get("expectancy") is not None:
            reg[k] = ml._r(bm[k]["expectancy"] - br[k]["expectancy"], 5)
    lk_m, lk_r = pos_meta[pos_meta["date"] >= ml.LOCKED_FROM], pos_ref[pos_ref["date"] >= ml.LOCKED_FROM]
    lk_delta = (lk_m["ret"].mean() - lk_r["ret"].mean()) if len(lk_m) and len(lk_r) else None
    crit = {
        "reproducible_leakage_checks": {"pass": leakage_ok},
        "enough_oos": {"pass": (pm.get("n_cohorts") or 0) >= g["min_cohorts"] and (pm.get("n_months") or 0) >= g["min_months"],
                       "value": [pm.get("n_cohorts"), pm.get("n_months")]},
        "delta_sharpe_positive": {"pass": (pm.get("sharpe") or -9) - (pr.get("sharpe") or 0) > g["delta_sharpe_min"],
                                  "value": ml._r((pm.get("sharpe") or 0) - (pr.get("sharpe") or 0), 3)},
        "bootstrap_ci_positive": {"pass": bool(boot) and boot["ci_monthly_mean"][0] > g["ci_lower_delta_monthly_gt"],
                                  "value": boot.get("ci_monthly_mean")},
        "years_positive": {"pass": len(years) > 0 and (years > 0).mean() >= g["min_share_years_positive"],
                           "value": ml._r((years > 0).mean(), 3) if len(years) else None},
        "halves_positive": {"pass": half > 0 and d.iloc[:half].mean() > 0 and d.iloc[half:].mean() > 0,
                            "value": [ml._r(d.iloc[:half].mean(), 5), ml._r(d.iloc[half:].mean(), 5)] if half else None},
        "no_regime_collapse": {"pass": all(v >= g["max_regime_delta_drop"] for v in reg.values()), "value": reg},
        "calibration_not_worse": {"pass": (cal_meta.get("brier", 9) - cal_ref.get("brier", 0) <= g["max_brier_worsening"])
                                  and (cal_meta.get("ece", 9) - cal_ref.get("ece", 0) <= g["max_ece_worsening"]),
                                  "value": {"brier": [cal_meta.get("brier"), cal_ref.get("brier")],
                                            "ece": [cal_meta.get("ece"), cal_ref.get("ece")]}},
        "not_outlier_driven": {"pass": len(trimmed) > 0 and trimmed.mean() > 0, "value": ml._r(trimmed.mean(), 6) if len(trimmed) else None},
        "locked_delta_positive": {"pass": lk_delta is not None and lk_delta > 0, "value": ml._r(lk_delta, 5)},
    }
    all_pass = all(c["pass"] for c in crit.values())
    point_pos = bool(boot) and boot["delta_monthly_mean"] > 0
    if all_pass:
        verdict = "PROMOTE"
    elif point_pos and crit["reproducible_leakage_checks"]["pass"] and crit["no_regime_collapse"]["pass"] \
            and not crit["bootstrap_ci_positive"]["pass"]:
        verdict = "NEED_MORE_DATA"
    else:
        verdict = "REJECT"
    return {"verdict": verdict, "criteria": crit, "bootstrap": boot,
            "reasons": [k for k, c in crit.items() if not c["pass"]]}


def calibrate_hc_rule(cal: pd.DataFrame, res: pd.DataFrame, score: str, use_agreement: bool) -> dict:
    """I: Schwellen nur auf Kalibrierjahren (alle Dev-Jahre außer dem letzten)
    wählen; Validierung auf letztem Dev-Jahr + Locked. Regel nur aktiv, wenn die
    untere Schranke der Netto-Expectancy in den Kalibrierjahren > 0 ist."""
    hc = MP["high_confidence"]
    if cal.empty:
        return {"enabled": False, "disabled_reason": "keine kalibrierten OOS-Wahrscheinlichkeiten"}
    j = cal.merge(res[["date", "ticker", "dis_rank_sd", "log_dollar_vol", "fwd_xs_20"]].rename(
        columns={"fwd_xs_20": "_y"}), on=["date", "ticker"], how="left")
    j["net"] = j[PROB_TARGET]
    folds = [f for f in sorted(j["fold"].unique()) if f != "locked"]
    if len(folds) < 2:
        return {"enabled": False, "disabled_reason": "zu wenige Jahre für Kalibrierung + Validierung"}
    calib, valid = j[j["fold"].isin(folds[:-1])], j[~j["fold"].isin(folds[:-1])]
    grid_sd = hc["agreement_sd_grid"] if use_agreement else [9.0]
    best, tested = None, 0
    for p in hc["prob_grid"]:
        for sd in grid_sd:
            tested += 1
            m = (calib["prob"] >= p) & (calib["dis_rank_sd"] <= sd) & (calib["log_dollar_vol"] > hc["min_dollar_vol_rank"])
            x = calib.loc[m, "net"].dropna()
            if len(x) < hc["min_n_calibration"]:
                continue
            lcb = x.mean() - hc["lcb_z"] * x.std(ddof=1) / math.sqrt(len(x))
            if best is None or lcb > best["lcb"]:
                best = {"prob": p, "agreement_sd": sd, "lcb": float(lcb), "n": int(len(x)), "mean": float(x.mean())}
    if best is None or best["lcb"] <= 0:
        return {"enabled": False, "n_rules_tested": tested, "best": best,
                "disabled_reason": "keine Regel mit positiver unterer Schranke der Netto-Expectancy (Kalibrierjahre)"}
    mv = (valid["prob"] >= best["prob"]) & (valid["dis_rank_sd"] <= best["agreement_sd"]) & \
        (valid["log_dollar_vol"] > hc["min_dollar_vol_rank"])
    xv = valid.loc[mv, "net"].dropna()
    val = {"n": int(len(xv)), "mean": ml._r(xv.mean(), 5) if len(xv) else None,
           "hit_rate": ml._r((xv > 0).mean(), 3) if len(xv) else None}
    enabled = len(xv) >= 30 and (xv.mean() > 0)
    return {"enabled": bool(enabled), "rule": best, "validation": val, "n_rules_tested": tested,
            "calibration_folds": folds[:-1], "validation_folds": sorted(set(valid["fold"])),
            "score": score, "agreement_used": use_agreement,
            "disabled_reason": None if enabled else "Regel hält in der Validierung nicht (n < 30 oder Expectancy <= 0)"}


def prob_maps(res: pd.DataFrame, score: str) -> dict:
    """Abbildung Querschnittsrang -> P(Überrendite > 0) bzw. erwartete 20d-
    Überrendite für LIVE-Signale, gelernt auf dem jüngsten Dev-Fold mit
    fertigen Labels (nie auf dem Locked-Holdout). Nie für die Bewertung verwendet."""
    from sklearn.isotonic import IsotonicRegression
    folds = [f for f in sorted(res["fold"].unique()) if f != "locked"]
    use = folds[-1] if folds else None        # nie auf dem (kontaminierten) Locked-Holdout fitten (Audit F02)
    if use is None:
        return {}
    d = res[(res["fold"] == use) & res[score].notna() & res[PROB_TARGET].notna()]
    rk = d.groupby("date")[score].rank(pct=True)
    iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip").fit(rk, (d[PROB_TARGET] > 0))
    iso_r = IsotonicRegression(out_of_bounds="clip").fit(rk, d[PROB_TARGET])
    return {"fold": use, "n": int(len(d)),
            "prob": {"x": [ml._r(v, 5) for v in iso.X_thresholds_], "y": [ml._r(v, 5) for v in iso.y_thresholds_]},
            "exp_xs20": {"x": [ml._r(v, 5) for v in iso_r.X_thresholds_], "y": [ml._r(v, 6) for v in iso_r.y_thresholds_]}}


# ── 8. Gegenwart: aktuelle Gewichte ─────────────────────────────────────────

def current_weights(panel: pd.DataFrame, res_all: pd.DataFrame, meta_d: pd.DataFrame, models: list[str],
                    latest_ranks: dict[str, dict]) -> dict:
    """Gewichte für den jüngsten Stichtag: Meta-Learner auf allen fertigen
    Labels, Zustand des jüngsten Stichtags."""
    latest = panel["date"].max()
    snap = panel[panel["date"] == latest]
    if snap.empty or not latest_ranks:
        return {}
    df = snap[["date", "ticker"] + list(STATE_FEATURES)].copy()
    for m in models:
        df[_mcol(m)] = df["ticker"].map(latest_ranks.get(m, {}))
        df[f"p_{m}"] = df[_mcol(m)]
    df = df.join(disagreement_features(df, models))
    tr = trailing_stats(meta_d, models, TRAIL_W, [latest]).iloc[0]
    rc = trailing_stats(meta_d, models, RECENT_W, [latest]).add_prefix("recent_").iloc[0]
    state = df[list(STATE_FEATURES)].iloc[0].to_dict()
    row = {**state, **tr.to_dict(), **rc.to_dict(), "dis_mean": float(df["dis_rank_sd"].mean())}
    md = pd.DataFrame([row], index=[latest])
    wm = regime_weight_model(meta_d, models, latest + pd.Timedelta(days=1))
    w = regime_weights(wm, md, [latest])
    return {r["model"]: {k: (ml._r(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k not in ("date", "model")}
            for _, r in w.iterrows()}


# ── Lauf ─────────────────────────────────────────────────────────────────────

def leakage_checks(prov: dict, meta_prov: dict) -> tuple[bool, list[str]]:
    issues = []
    for f, p in prov.items():
        cut = pd.Timestamp(p["train_label_end_before"])
        for mid, v in p.items():
            if isinstance(v, dict) and v.get("train_max_label_end") and pd.Timestamp(v["train_max_label_end"]) >= cut:
                issues.append(f"Basis {mid} Fold {f}: Trainingslabel bis {v['train_max_label_end']} >= {cut.date()}")
    for f, p in meta_prov.items():
        if p.get("meta_train_max_label_end") and pd.Timestamp(p["meta_train_max_label_end"]) >= pd.Timestamp(p["meta_train_label_end_before"]):
            issues.append(f"Meta Fold {f}: Label {p['meta_train_max_label_end']} nicht vor {p['meta_train_label_end_before']}")
        if pd.Timestamp(p["test_start"]) < pd.Timestamp(p["meta_train_label_end_before"]) and f != "locked":
            issues.append(f"Meta Fold {f}: Test beginnt vor Trainingsende")
    return not issues, issues


def panel_hash(panel: pd.DataFrame) -> str:
    num = ["fwd_xs_20"] + list(ml.ALL_FEATURES)
    h = pd.util.hash_pandas_object(pd.concat([panel[["date", "ticker"]], panel[num].round(8)], axis=1),
                                   index=False).to_numpy()
    return hashlib.sha256(h.tobytes()).hexdigest()[:16]


def evaluate(panel: pd.DataFrame, specs: list[dict], latest_ranks: dict | None = None) -> dict:
    models = [s["id"] for s in specs]
    base, prov = base_oos(panel, specs, with_counterfactuals=True)
    base["sector_code"] = base["sector"].astype("category").cat.codes.astype(float)
    base = add_rel_target(base)
    df, meta_d = build_meta_frame(base, models)
    res, mprov = meta_walk_forward(df, meta_d, models)
    leak_ok, leak_issues = leakage_checks(prov, mprov["folds"])
    LAST_RUN.update(res=res, meta_d=meta_d, models=models, base=df)
    names = [c[2:] for c in res.columns if c.startswith("s_")]
    ref, meta = MP["variants"]["reference"], MP["variants"]["primary_meta"]
    approaches, pos_by, cal_by = {}, {}, {}
    for n in names:
        pos = positions(res, f"s_{n}")
        cal = lagged_calibration(res, f"s_{n}")
        pos_by[n], cal_by[n] = pos, cal
        approaches[n] = {"metrics": {**portfolio_metrics(pos[pos["date"] < ml.LOCKED_FROM]), **prob_metrics(
            cal[cal["fold"] != "locked"])},
            "locked": {**portfolio_metrics(pos[pos["date"] >= ml.LOCKED_FROM]), **prob_metrics(cal[cal["fold"] == "locked"])},
            **breakdowns(pos[pos["date"] < ml.LOCKED_FROM])}
    deltas = {}
    for n in names:
        if n == ref:
            continue
        a, b = approaches[n]["metrics"], approaches[ref]["metrics"]
        deltas[n] = {k: ml._r((a.get(k) or 0) - (b.get(k) or 0), 5) for k in
                     ("sharpe", "cagr", "max_dd", "expectancy", "profit_factor", "hit_rate", "brier", "ece",
                      "precision_at_k", "recall_strong", "turnover")}
        deltas[n]["bootstrap"] = bootstrap_delta(monthly_series(pos_by[n][pos_by[n]["date"] < ml.LOCKED_FROM]),
                                                 monthly_series(pos_by[ref][pos_by[ref]["date"] < ml.LOCKED_FROM]))
    robustness = {}
    for n in (meta, "meta_stacking", ref, "best_single_ex_ante"):
        if f"s_{n}" not in res:
            continue
        dev = res[res["fold"] != "locked"]
        robustness[n] = {
            "rebalance_4w": portfolio_metrics(positions(dev, f"s_{n}", every=4)).get("expectancy"),
            "liquid_half": portfolio_metrics(positions(dev[dev["log_dollar_vol"] > 0], f"s_{n}")).get("expectancy"),
            "less_liquid_half": portfolio_metrics(positions(dev[dev["log_dollar_vol"] <= 0], f"s_{n}")).get("expectancy"),
        }
    gate = promotion_gate(res, meta, ref, pos_by[meta], pos_by[ref],
                          approaches[meta]["metrics"], approaches[ref]["metrics"], leak_ok)
    gate_stack = promotion_gate(res, "meta_stacking", ref, pos_by["meta_stacking"], pos_by[ref],
                                approaches["meta_stacking"]["metrics"], approaches[ref]["metrics"], leak_ok) \
        if "meta_stacking" in pos_by else None
    dis = disagreement_test(res)
    if latest_ranks:
        R = pd.DataFrame({m: latest_ranks.get(m, {}) for m in models}).dropna(how="all")
        cur = float(R.std(axis=1).mean()) if len(R) else None
        hist = meta_d["dis_mean"].dropna()
        dis["current_mean_rank_sd"] = ml._r(cur, 4)
        dis["current_level"] = None if cur is None or hist.empty else \
            "HIGH" if cur > hist.quantile(0.9) else "LOW" if cur < hist.quantile(0.1) else "NORMAL"
    active = meta if gate["verdict"] == "PROMOTE" else ref
    hc = calibrate_hc_rule(cal_by[active], res, active, dis.get("use_in_score", False))
    cw = current_weights(panel, res, meta_d, models, latest_ranks or {})
    ablations = {n: {"expectancy": approaches[n]["metrics"].get("expectancy"),
                     "sharpe": approaches[n]["metrics"].get("sharpe"), "brier": approaches[n]["metrics"].get("brier"),
                     "delta_vs_full_expectancy": ml._r((approaches[n]["metrics"].get("expectancy") or 0) -
                                                       (approaches[n.split("__")[0]]["metrics"].get("expectancy") or 0), 5)}
                 for n in names if "__" in n}
    ablations["without_dynamic_weighting (= static_equal)"] = {
        "expectancy": approaches[ref]["metrics"].get("expectancy"), "sharpe": approaches[ref]["metrics"].get("sharpe")}
    ablations["without_historical_analogies"] = "n/a – Analogie-Engine ist nicht Teil der Basismodelle/Meta-Merkmale"
    ablations["without_alternative_data"] = "n/a – kein Basismodell nutzt alternative Daten (PIT-Historie zu kurz)"
    latest = panel["date"].max()
    snap = panel[panel["date"] == latest].iloc[0] if (panel["date"] == latest).any() else None
    tr_rng = res[res["fold"] != "locked"].groupby("date")[list(STATE_FEATURES)].first()
    drift = {}
    if snap is not None:
        for c in STATE_FEATURES:
            lo, hi = tr_rng[c].quantile(0.01), tr_rng[c].quantile(0.99)
            v = snap[c]
            drift[c] = {"value": ml._r(v, 4), "p01": ml._r(lo, 4), "p99": ml._r(hi, 4),
                        "out_of_range": bool(pd.notna(v) and (v < lo or v > hi))}
    mi = model_intelligence(meta_d, models, cw)
    for mid, c in contributions(res, models).items():
        if mid in mi:
            mi[mid]["contribution"] = c
    return {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "meta_version": MP["meta_version"], "panel_hash": panel_hash(panel),
        "protocol": "config/meta_protocol.yaml", "reference": ref, "primary_meta": meta,
        "models": models, "base_provenance": prov, "meta_provenance": mprov["folds"],
        "leakage_checks": {"ok": leak_ok, "issues": leak_issues},
        "approaches": approaches, "deltas": deltas, "robustness": robustness,
        "decision": gate, "decision_stacking": gate_stack, "ablations": ablations,
        "calibration_buckets": {n: calibration_buckets(cal_by[n]) for n in (ref, meta, "meta_stacking") if n in cal_by},
        "disagreement": dis, "failure_profiles": failure_profiles(res, models),
        "model_intelligence": mi, "current_weights": cw, "active_ensemble": active,
        "hc_rule": hc, "prob_map": prob_maps(res, f"s_{active}"),
        "current_regime": {"vix": ml._r(snap["vix"], 2) if snap is not None else None,
                           "spy_trend_200": ml._r(snap["spy_trend_200"], 4) if snap is not None else None,
                           "confidence": ("LOW" if snap is None or any(v["out_of_range"] for v in drift.values())
                                          else "MEDIUM" if abs((snap["vix"] or 20) - 20) < 2 or abs(snap["spy_trend_200"] or 0) < 0.02
                                          else "HIGH")},
        "drift": {"feature_drift": drift,
                  "feature_drift_flag": any(v["out_of_range"] for v in drift.values()),
                  "model_drift_flag": any(v.get("trend") == "deteriorating" for v in mi.values())},
        "weights_history_tail": mprov["weights"].tail(60).to_dict("records") if len(mprov["weights"]) else [],
    }


def render_md(rep: dict) -> str:
    L = [f"# Meta-Learning-Validierung – {rep['generated']}", "",
         f"Version {rep['meta_version']} · Panel-Hash {rep['panel_hash']} · Referenz: {rep['reference']} · "
         f"Primär: {rep['primary_meta']} · Leakage-Checks: {'OK' if rep['leakage_checks']['ok'] else 'FEHLER'}",
         f"**Entscheidung ({rep['primary_meta']}): {rep['decision']['verdict']}** – nicht erfüllt: "
         f"{', '.join(rep['decision']['reasons']) or '–'}",
         f"Aktives Ensemble für Research-Signale: {rep['active_ensemble']}", "",
         "## Out-of-Sample (Meta-Testjahre vor Locked)", "",
         "| Variante | CAGR | Sharpe | Sortino | MaxDD | Calmar | PF | Hit | Ø Gew. | Ø Verl. | Payoff | Expectancy | Brier | ECE | Prec@K | Recall stark | Turnover | Trades |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, a in rep["approaches"].items():
        m = a["metrics"]
        L.append("| " + " | ".join(str(x) for x in [n, m.get("cagr"), m.get("sharpe"), m.get("sortino"), m.get("max_dd"),
                                                   m.get("calmar"), m.get("profit_factor"), m.get("hit_rate"),
                                                   m.get("avg_winner"), m.get("avg_loser"), m.get("payoff"),
                                                   m.get("expectancy"), m.get("brier"), m.get("ece"),
                                                   m.get("precision_at_k"), m.get("recall_strong"), m.get("turnover"),
                                                   m.get("n_trades")]) + " |")
    L += ["", "## Deltas gegenüber Referenz (Bootstrap, Bonferroni)", ""]
    for n, d in rep["deltas"].items():
        b = d.get("bootstrap", {})
        L.append(f"- **{n}**: Δ Sharpe {d['sharpe']}, Δ Expectancy {d['expectancy']}, Δ MaxDD {d['max_dd']}, "
                 f"Δ PF {d['profit_factor']}, Δ Hit {d['hit_rate']}, Δ Brier {d['brier']}, Δ ECE {d['ece']}, "
                 f"Δ Prec@K {d['precision_at_k']} · Monats-Δ {b.get('delta_monthly_mean')} CI {b.get('ci_monthly_mean')}")
    L += ["", "## Gate", ""]
    for k, c in rep["decision"]["criteria"].items():
        L.append(f"- {'✅' if c['pass'] else '❌'} {k}: {c.get('value')}")
    L += ["", "## Locked-Holdout", ""]
    for n, a in rep["approaches"].items():
        L.append(f"- {n}: Expectancy {a['locked'].get('expectancy')}, Sharpe {a['locked'].get('sharpe')}, "
                 f"Trades {a['locked'].get('n_trades')}")
    L += ["", "## Ablationen", ""] + [f"- {k}: {v}" for k, v in rep["ablations"].items()]
    L += ["", "## Robustheit (Expectancy)", ""] + [f"- {k}: {v}" for k, v in rep["robustness"].items()]
    L += ["", "## Disagreement (G)", ""] + [f"- {k}: {v}" for k, v in rep["disagreement"].items()]
    L += ["", "## Failure-Profile (F, nur gemessene Segmente mit |t| >= 2)", ""]
    for mid, p in rep["failure_profiles"].items():
        L.append(f"- **{mid}** – funktioniert: {', '.join(p['works']) or '–'}; versagt: {', '.join(p['fails']) or '–'}")
    L += ["", "## Kalibrierungs-Buckets (P(Überrendite 20d > 0), Vorjahres-Isotonie)", ""]
    for n, bs in rep["calibration_buckets"].items():
        L.append(f"**{n}**")
        L.append("| Bucket | N | Trefferquote | Prognose | Ø Rendite | Median | Ø Drawdown | EV netto | Fehler | Flag |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for b in bs:
            L.append(f"| {b['bucket']} | {b['n']} | {b.get('win_rate')} | {b.get('predicted')} | {b.get('avg_return')} | "
                     f"{b.get('median_return')} | {b.get('avg_drawdown')} | {b.get('expected_value')} | "
                     f"{b.get('calibration_error')} | {b.get('flag')} |")
    L += ["", "## Modell-Intelligenz", ""] + [f"- {k}: {v}" for k, v in rep["model_intelligence"].items()]
    L += ["", "## High-Confidence-Regel (I)", "", f"- {rep['hc_rule']}"]
    return "\n".join(L) + "\n"


def run(panel: pd.DataFrame | None = None) -> dict:
    import os
    import pickle
    if panel is None:
        cache = os.environ.get("ML_PANEL_CACHE")
        if cache and Path(cache).exists():
            with open(cache, "rb") as fh:
                panel = pickle.load(fh)  # noqa: S301 – eigene, im selben Job erzeugte Datei
        else:
            panel = ml.build_research_panel()
    reg = ml.load_registry()
    status, _ = ml.check_registry(reg)
    specs = [s for s in reg.get("models") or [] if status.get(s["id"]) == "valid"]
    latest = panel["date"].max().date().isoformat()
    ranks = {r["model_id"]: r["rank_pct"] for r in ml._read_predictions() if r["prediction_date"] == latest}
    rep = evaluate(panel, specs, ranks)
    cache = os.environ.get("META_RES_CACHE")
    if cache:
        with open(cache, "wb") as fh:
            pickle.dump(LAST_RUN, fh)
    # Kein eigener Safe-Mode-Flag mehr (war immer false -> widersprüchlich): kanonisch ist
    # ausschließlich outputs/state/system_state.json (modules/system_state.py).
    try:
        from modules import system_state as _ss
        _sv = _ss.safe_mode_view(_ss.current())
    except Exception as e:  # noqa: BLE001 – unbekannt = Safe Mode (Meta nicht aktiv)
        _sv = {"active": True, "state_version": None, "reasons": [f"SystemState-Fehler: {e}"]}
    state = {"meta_active": rep["decision"]["verdict"] == "PROMOTE" and not _sv["active"],
             "system_state_version": _sv.get("state_version"),
             "active_ensemble": rep["active_ensemble"],
             "meta_version": rep["meta_version"],
             "reasons": ([f"Meta-Learning nicht promoted ({rep['decision']['verdict']}) – Referenz aktiv"]
                         if rep["decision"]["verdict"] != "PROMOTE" else [])
                        + ([f"SystemState Safe Mode: {'; '.join(_sv['reasons'])}"] if _sv["active"] else []),
             "updated": rep["generated"]}
    ml.OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str))
    OUT_MD.write_text(render_md(rep), encoding="utf-8")
    STATE_PATH.write_text(json.dumps(state, indent=1))
    HC_THRESHOLDS_PATH.write_text(json.dumps({**rep["hc_rule"], "generated": rep["generated"],
                                              "prob_map": rep["prob_map"],
                                              "active_ece": rep["approaches"][rep["active_ensemble"]]["metrics"].get("ece"),
                                              **probability_validation((rep["calibration_buckets"] or {}).get(
                                                  rep["active_ensemble"]) or []),
                                              "current_weights": rep["current_weights"],
                                              "feature_drift_flag": rep["drift"]["feature_drift_flag"],
                                              "failure_profiles": {m: {"works": p["works"], "fails": p["fails"]}
                                                                   for m, p in rep["failure_profiles"].items()},
                                              "meta_version": rep["meta_version"], "active_ensemble": rep["active_ensemble"],
                                              "models": rep["models"]}, indent=1, default=str))
    return rep


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    rep = run()
    print(render_md(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

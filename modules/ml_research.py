"""
modules/ml_research.py – kontrollierter ML-Research-Kreislauf (NUR SHADOW)

    python -m modules.ml_research --mode full     (monatlich: Walk-Forward, Locked, Attribution)
    python -m modules.ml_research --mode weekly   (wöchentlich: Prognosen + Forward-Auswertung)

Präregistrierung: docs/research/PREREG_ml_research_2026-09-29.md
Registry (menschlich gepflegt, hier nur gelesen): config/model_registry.yaml

Kreislauf:
  Kursdaten + PIT-Makro -> historischer Feature-Store (wöchentliche Querschnitte)
  -> Labels (Forward-Rendite, MFE/MAE, Asymmetrie) -> gepurgter Walk-Forward je
  Modell -> Locked-Holdout -> Forward-Shadow-Prognose-Ledger -> Attribution ->
  Champion/Challenger-EMPFEHLUNG. Es gibt keine automatische Promotion und
  keine Produktionswirkung (Scoring/Gates/PPO bleiben unberührt).

Zeitliche Regeln (Tests: tests/test_ml_research.py):
  * Features nur aus Daten bis Close_t; Makro nur mit available_at < Tag t.
  * Entry Open_{t+1}, Exit Close_{t+h}; MFE/MAE aus High/Low von t+1..t+h.
  * Training nur mit Zeilen, deren Label-Ende VOR dem Teststart liegt (Purge).
  * Hyperparameter nur auf dem inneren Validierungsjahr (vor dem Testjahr).
  * Locked-Holdout (ab locked_from) nie in Entwicklung/Selektion; Test-Labels
    der Entwicklung enden vor locked_from.

Bewertungsgröße: Top-Dezil minus Querschnittsmittel desselben Stichtags
(neutralisiert den Survivorship-Bias der heutigen Indexliste weitgehend),
netto 10 bp je Seite; Stress 25 bp.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger(__name__)

REGISTRY_PATH = Path("config/model_registry.yaml")
PROTOCOL_PATH = Path(__file__).resolve().parent.parent / "config" / "research_protocol.yaml"


def load_protocol(path: Path = PROTOCOL_PATH) -> dict:
    """Geschütztes Bewertungsprotokoll (Kosten, Zeiträume, Kriterien)."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


PROTOCOL = load_protocol()
OUT_DIR = Path("outputs/research")
PRED_DIR = OUT_DIR / "ml_predictions"
REGISTRY_LOG = OUT_DIR / "ml_registry_log.json"
SECTOR_CACHE = OUT_DIR / "sector_map.json"            # aktueller Sektor (Referenz, NICHT point-in-time)
SECTOR_HISTORY = OUT_DIR / "sector_history.jsonl"     # zeitgestempelte Beobachtungen (point-in-time)

LABEL_HORIZONS = (20, 60)
TRAIN_START = PROTOCOL["periods"]["train_start"]
FIRST_TEST_YEAR = int(PROTOCOL["periods"]["first_test_year"])
LOCKED_FROM = pd.Timestamp(PROTOCOL["periods"]["locked_from"])
LOCKED_STATUS = PROTOCOL["periods"].get("locked_status", "CLEAN")
FORWARD_HOLDOUT_FROM = PROTOCOL["periods"].get("forward_holdout_from")
COST_BASE = float(PROTOCOL["costs"]["base_per_side"])
COST_STRESS = float(PROTOCOL["costs"]["stress_per_side"])
TOP_Q = float(PROTOCOL["portfolio"]["top_quantile"])
MIN_CROSS_SECTION = int(PROTOCOL["portfolio"]["min_cross_section"])
TARGET_CLIP = {"fwd_xs_20": 0.5, "fwd_xs_60": 0.8, "asym_20": 5.0, "asym_60": 5.0}

STOCK_FEATURES = ("mom_12_1", "mom_3m", "rev_1m", "ret_5d", "rs_63", "vol_20", "vol_60", "vol_ratio",
                  "relvol_5_60", "log_dollar_vol", "dist_52w_high", "max_ret_21", "beta_126")
DATE_FEATURES = ("vix", "vix_chg_21", "spy_trend_200", "spy_mom_63", "tnx", "curve_10y_3m",
                 "cpi_yoy", "fed_assets_13w_chg", "usd_63d_chg", "wti_63d_chg")
ALL_FEATURES = STOCK_FEATURES + DATE_FEATURES
FEATURE_GROUPS = {
    "momentum": ("mom_12_1", "mom_3m", "rev_1m", "ret_5d", "rs_63", "dist_52w_high"),
    "risk": ("vol_20", "vol_60", "vol_ratio", "max_ret_21", "beta_126"),
    "liquidity": ("relvol_5_60", "log_dollar_vol"),
    "market": ("vix", "vix_chg_21", "spy_trend_200", "spy_mom_63", "tnx", "curve_10y_3m"),
    "macro": ("cpi_yoy", "fed_assets_13w_chg", "usd_63d_chg", "wti_63d_chg"),
}


# ── Feature-Store + Labels ───────────────────────────────────────────────────

def _fwd_window(s: pd.Series, h: int, how: str) -> pd.Series:
    """max/min von s über t+1..t+h (nur vollständige Fenster)."""
    rev = s.iloc[::-1].rolling(h, min_periods=h)
    agg = rev.max() if how == "max" else rev.min()
    return agg.iloc[::-1].shift(-1)


def ticker_frame(df: pd.DataFrame, spy: pd.DataFrame, horizons=LABEL_HORIZONS) -> pd.DataFrame:
    """Alle Features/Labels eines Tickers auf dem SPY-Kalender (täglich)."""
    cal = spy.index
    df = df.reindex(cal)
    c, o, v = df["Close"], df["Open"], df["Volume"]
    hi = df["High"] if "High" in df else c
    lo = df["Low"] if "Low" in df else c
    spy_c, spy_o = spy["Close"], spy["Open"]
    ret, spy_ret = c.pct_change(fill_method=None), spy_c.pct_change(fill_method=None)
    f = pd.DataFrame(index=cal)
    f["close"] = c
    f["mom_12_1"] = c.shift(21) / c.shift(252) - 1.0
    f["mom_3m"] = c / c.shift(63) - 1.0
    f["rev_1m"] = c / c.shift(21) - 1.0
    f["ret_5d"] = c / c.shift(5) - 1.0
    f["rs_63"] = f["mom_3m"] - (spy_c / spy_c.shift(63) - 1.0)
    f["vol_20"] = ret.rolling(20, min_periods=18).std()
    f["vol_60"] = ret.rolling(60, min_periods=50).std()
    f["vol_ratio"] = f["vol_20"] / f["vol_60"]
    f["relvol_5_60"] = v.rolling(5, min_periods=5).mean() / v.rolling(60, min_periods=50).mean()
    f["log_dollar_vol"] = np.log10((c * v).rolling(20, min_periods=18).mean().clip(lower=1.0))
    f["dist_52w_high"] = c / hi.rolling(252, min_periods=200).max() - 1.0
    f["max_ret_21"] = ret.rolling(21, min_periods=18).max()
    cov = ret.rolling(126, min_periods=100).cov(spy_ret)
    f["beta_126"] = cov / spy_ret.rolling(126, min_periods=100).var()
    entry = o.shift(-1)
    spy_entry = spy_o.shift(-1)
    idx = pd.Series(cal, index=cal)
    for h in horizons:
        fwd = c.shift(-h) / entry - 1.0
        f[f"fwd_ret_{h}"] = fwd
        f[f"fwd_xs_{h}"] = fwd - (spy_c.shift(-h) / spy_entry - 1.0)
        f[f"mfe_{h}"] = _fwd_window(hi, h, "max") / entry - 1.0
        f[f"mae_{h}"] = _fwd_window(lo, h, "min") / entry - 1.0
        f[f"asym_{h}"] = (f[f"mfe_{h}"] + f[f"mae_{h}"]) / (f["vol_20"] * math.sqrt(h))
        f[f"label_end_{h}"] = idx.shift(-h)
    return f.replace([np.inf, -np.inf], np.nan)


def weekly_dates(cal: pd.DatetimeIndex, start: str = "2014-06-01") -> list[pd.Timestamp]:
    """Letzter Handelstag jeder Kalenderwoche."""
    s = pd.Series(cal, index=cal)
    s = s[s.index >= pd.Timestamp(start)]
    return list(s.groupby(s.index.to_period("W-FRI")).max())


def market_features(spy: pd.DataFrame, vix: pd.Series | None, tnx: pd.Series | None,
                    irx: pd.Series | None) -> pd.DataFrame:
    cal = spy.index
    c = spy["Close"]
    m = pd.DataFrame(index=cal)
    m["spy_trend_200"] = c / c.rolling(200, min_periods=180).mean() - 1.0
    m["spy_mom_63"] = c / c.shift(63) - 1.0
    vx = vix.reindex(cal).ffill() if vix is not None else pd.Series(np.nan, index=cal)
    m["vix"] = vx
    m["vix_chg_21"] = vx - vx.shift(21)
    t = tnx.reindex(cal).ffill() if tnx is not None else pd.Series(np.nan, index=cal)
    i = irx.reindex(cal).ffill() if irx is not None else pd.Series(np.nan, index=cal)
    m["tnx"] = t
    m["curve_10y_3m"] = t - i
    return m


def macro_features(observations, dates: list[pd.Timestamp]) -> pd.DataFrame:
    """PIT-Makro je Stichtag (nur bis zum Vortag veröffentlichte Vintages)."""
    from modules.external import regime
    cols = ("cpi_yoy", "fed_assets_13w_chg", "usd_63d_chg", "wti_63d_chg")
    rows = {}
    by_metric: dict = {}
    for o in observations or []:
        by_metric.setdefault(o.metric, []).append(o)
    obs = [o for lst in by_metric.values() for o in lst]
    for d in dates:
        as_of = datetime(d.year, d.month, d.day, tzinfo=timezone.utc) - timedelta(seconds=1)
        st = regime.regime_state(obs, as_of) if obs else {}
        rows[d] = {k: st.get(k) for k in cols}
    return pd.DataFrame.from_dict(rows, orient="index", columns=list(cols)).astype(float)


def build_panel(frames: dict[str, pd.DataFrame], spy: pd.DataFrame, vix=None, tnx=None, irx=None,
                macro_obs=None, extra_dates=(), sectors: dict | None = None,
                start: str = "2014-06-01", membership: dict | None = None) -> pd.DataFrame:
    """sectors: {ticker: [(observed_at, sector), ...]} (point-in-time). Ein Sektor gilt erst ab dem
    Tag, an dem er beobachtet wurde; davor 'unknown' (nie heutiger Sektor rückwirkend)."""
    """Historische Trainingsmatrix: eine Zeile je (Stichtag, Ticker).
    membership {ticker: [[start,end],...]}: nur Zeilen, an deren Stichtag der
    Titel Indexmitglied war (Point-in-Time-Universum, Audit F01). Die
    Querschnittsränge werden NACH dem Filter gebildet."""
    cal = spy.index
    dates = sorted(set(weekly_dates(cal, start)) | {pd.Timestamp(d) for d in extra_dates if pd.Timestamp(d) in cal})
    parts = []
    for ticker, df in frames.items():
        f = ticker_frame(df, spy)
        f = f.loc[f.index.isin(dates)]
        f = f[f["close"].notna()]
        if f.empty:
            continue
        f = f.assign(ticker=ticker)
        parts.append(f)
    if not parts:
        return pd.DataFrame()
    p = pd.concat(parts)
    p.index.name = "date"
    p = p.reset_index()
    if membership is not None:
        from modules.universe import is_member
        keep = [is_member(membership.get(t, []), d) for t, d in zip(p["ticker"], p["date"].dt.strftime("%Y-%m-%d"))]
        p = p[keep].reset_index(drop=True)
        if p.empty:
            return pd.DataFrame()
    mk = market_features(spy, vix, tnx, irx)
    p = p.join(mk, on="date")
    mac = macro_features(macro_obs, dates)
    p = p.join(mac, on="date")
    p["sector"] = pit_sector(p, sectors or {})
    # Querschnitts-Ränge (-0.5..0.5) je Stichtag: robust gegen Niveau-Drift
    for col in STOCK_FEATURES:
        p[col] = p.groupby("date")[col].rank(pct=True) - 0.5
    return p.sort_values(["date", "ticker"]).reset_index(drop=True)


# ── Modelle ──────────────────────────────────────────────────────────────────

def _make_model(kind: str, params: dict):
    if kind == "elastic_net":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import ElasticNet
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                             ElasticNet(random_state=0, max_iter=5000, **params))
    if kind == "hist_gbm":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(random_state=0, early_stopping=False, l2_regularization=1.0, **params)
    raise ValueError(f"unbekanntes Modell: {kind}")


def _grid(spec: dict) -> list[dict]:
    grid = [{}]
    for k, vals in (spec.get("params_grid") or {}).items():
        grid = [{**g, k: v} for g in grid for v in vals]
    return grid


def feature_list(spec: dict) -> list[str]:
    """features: "all" oder Liste aus Feature-Namen und "group:<name>".
    extra_features: nur registrierte Alternative-Data-Features (SHADOW-
    Challenger, modules/alt_data/registry.py); Champion-Specs nutzen sie nie."""
    f = spec.get("features", "all")
    if f == "all":
        out = list(ALL_FEATURES)
    else:
        out = []
        for x in f:
            names = FEATURE_GROUPS.get(x[6:], ()) if str(x).startswith("group:") else (x,)
            out += [n for n in names if n in ALL_FEATURES and n not in out]
    extra = spec.get("extra_features") or []
    if extra:
        from modules.alt_data.registry import ALT_FEATURES
        unknown = [x for x in extra if x not in ALT_FEATURES]
        if unknown:
            raise ValueError(f"extra_features nicht registriert: {unknown}")
        out += [x for x in extra if x not in out]
    return out


def _xy(df: pd.DataFrame, spec: dict, cols: list[str]):
    tgt = spec["target"]
    d = df[df[tgt].notna()]
    clip = TARGET_CLIP.get(tgt, 1.0)
    return d[cols].to_numpy(float), d[tgt].clip(-clip, clip).to_numpy(float)


class FittedModel:
    def __init__(self, spec: dict, params: dict | None = None, model=None, cols: list[str] | None = None):
        self.spec, self.params, self.model = spec, params or {}, model
        self.cols = cols or []

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.spec.get("model") == "rule":
            return df[self.spec["rule_feature"]].to_numpy(float)
        return self.model.predict(df[self.cols].to_numpy(float))


def fit(spec: dict, train: pd.DataFrame, params: dict | None = None) -> FittedModel:
    """Merkmale, die im Training komplett fehlen (z.B. Reihe beginnt später),
    werden für dieses Modell ausgelassen – nie mit Zukunftswissen gefüllt."""
    if spec.get("model") == "rule":
        return FittedModel(spec)
    t = train[train[spec["target"]].notna()]
    cols = [c for c in feature_list(spec) if t[c].notna().any()]
    if not cols:
        raise ValueError("keine Trainingsmerkmale")
    X, y = _xy(t, spec, cols)
    m = _make_model(spec["model"], params or {})
    m.fit(X, y)
    return FittedModel(spec, params, m, cols)


def purged(panel: pd.DataFrame, before: pd.Timestamp, horizon: int = 20, start: str = TRAIN_START) -> pd.DataFrame:
    """Nur Zeilen, deren Label bis `before` vollständig feststand (Purge)."""
    le = panel[f"label_end_{horizon}"]
    return panel[(panel["date"] >= pd.Timestamp(start)) & le.notna() & (le < before)]


def _horizon(spec: dict) -> int:
    return int(str(spec.get("target", "fwd_xs_20")).rsplit("_", 1)[-1])


def daily_ic(df: pd.DataFrame, score: str, target: str = "fwd_xs_20") -> pd.Series:
    out = {}
    for d, g in df.groupby("date"):
        g = g[[score, target]].dropna()
        if len(g) >= MIN_CROSS_SECTION and g[score].nunique() > 1:
            out[d] = g[score].rank().corr(g[target].rank())
    return pd.Series(out, dtype=float)


def select_params(spec: dict, train: pd.DataFrame, test_start: pd.Timestamp) -> dict:
    """Hyperparameter auf dem inneren Validierungsjahr (Jahr vor dem Test)."""
    grid = _grid(spec)
    if len(grid) <= 1 or spec.get("model") == "rule":
        return grid[0]
    h = _horizon(spec)
    val_start = pd.Timestamp(year=test_start.year - 1, month=1, day=1)
    inner = purged(train, val_start, h)
    val = train[(train["date"] >= val_start)]
    if inner.empty or val.empty:
        return grid[0]
    best, best_ic = grid[0], -np.inf
    for g in grid:
        m = fit(spec, inner, g)
        v = val.assign(_s=m.predict(val))
        ic = daily_ic(v, "_s", spec["target"]).mean()
        if np.isfinite(ic) and ic > best_ic:
            best, best_ic = g, ic
    return best


# ── Kennzahlen ───────────────────────────────────────────────────────────────

def cohort_returns(df: pd.DataFrame, score: str, h: int = 20, q: float = TOP_Q) -> pd.DataFrame:
    """Je Stichtag: Top-Dezil vs. Querschnittsmittel (brutto), Long-Short,
    MFE/MAE des Top-Dezils."""
    rows = []
    tgt = f"fwd_xs_{h}"
    for d, g in df.groupby("date"):
        g = g[g[score].notna() & g[tgt].notna()]
        if len(g) < MIN_CROSS_SECTION:
            continue
        k = max(1, int(round(len(g) * q)))
        s = g.sort_values(score)
        top, bot = s.tail(k), s.head(k)
        u = g[tgt].mean()
        rows.append({"date": d, "top_vs_univ": top[tgt].mean() - u, "top_xs": top[tgt].mean(),
                     "long_short": top[tgt].mean() - bot[tgt].mean(), "univ_xs": u,
                     "top_mfe": top[f"mfe_{h}"].mean(), "top_mae": top[f"mae_{h}"].mean(),
                     "univ_mfe": g[f"mfe_{h}"].mean(), "univ_mae": g[f"mae_{h}"].mean(), "n": len(g)})
    return pd.DataFrame(rows)


def perf(cohorts: pd.DataFrame, cost_per_side: float = COST_BASE, col: str = "top_vs_univ") -> dict:
    """Monatsaggregation (Kohorten überlappen -> t/Sharpe über Monatsmittel)."""
    if cohorts.empty:
        return {"n_cohorts": 0}
    r = cohorts.set_index("date")[col] - 2 * cost_per_side
    m = r.groupby(r.index.to_period("M")).mean()
    y = r.groupby(r.index.year).mean()
    sd = m.std(ddof=1)
    eq = (1 + m).cumprod()
    dd = float((eq / eq.cummax() - 1).min()) if len(eq) else None
    up, down = cohorts["top_mfe"].mean(), cohorts["top_mae"].mean()
    return {
        "n_cohorts": int(len(r)), "n_months": int(len(m)),
        "mean": _r(r.mean()), "median": _r(r.median()), "hit_rate": _r((r > 0).mean(), 3),
        "t_months": _r(m.mean() / sd * math.sqrt(len(m)), 2) if len(m) > 2 and sd > 0 else None,
        "sharpe_ann": _r(m.mean() / sd * math.sqrt(12), 2) if len(m) > 2 and sd > 0 else None,
        "max_dd": _r(dd), "years_positive_share": _r((y > 0).mean(), 3) if len(y) else None,
        "top_mfe": _r(up), "top_mae": _r(down),
        "top_asymmetry": _r(up / abs(down), 3) if down and down < 0 else None,
        "univ_asymmetry": _r(cohorts["univ_mfe"].mean() / abs(cohorts["univ_mae"].mean()), 3)
        if cohorts["univ_mae"].mean() < 0 else None,
    }


def ic_stats(ic: pd.Series) -> dict:
    if ic.empty:
        return {"n_dates": 0}
    m = ic.groupby(ic.index.to_period("M")).mean()
    sd = m.std(ddof=1)
    return {"n_dates": int(len(ic)), "mean_ic": _r(ic.mean()),
            "t_months": _r(m.mean() / sd * math.sqrt(len(m)), 2) if len(m) > 2 and sd > 0 else None}


def _r(x, nd: int = 5):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if math.isfinite(x) else None


# ── Walk-Forward, Locked, Attribution ────────────────────────────────────────

def permutation_importance(model: FittedModel, test: pd.DataFrame, target: str,
                           repeats: int = 2, seed: int = 0) -> dict:
    """IC-Verlust bei Permutation: Aktienmerkmale innerhalb des Stichtags,
    Datumsmerkmale über Stichtage hinweg."""
    if model.spec.get("model") == "rule" or test.empty:
        return {}
    rng = np.random.default_rng(seed)
    base = daily_ic(test.assign(_s=model.predict(test)), "_s", target).mean()
    out = {}
    for f in model.cols:
        drops = []
        for _ in range(repeats):
            t = test.copy()
            if f in STOCK_FEATURES:
                t[f] = t.groupby("date")[f].transform(lambda s: s.sample(frac=1.0, random_state=int(rng.integers(1e9))).to_numpy())
            else:
                dv = t.groupby("date")[f].first()
                perm = dict(zip(dv.index, rng.permutation(dv.to_numpy())))
                t[f] = t["date"].map(perm)
            drops.append(base - daily_ic(t.assign(_s=model.predict(t)), "_s", target).mean())
        out[f] = float(np.nanmean(drops))
    return out


def walk_forward(panel: pd.DataFrame, spec: dict, locked_from: pd.Timestamp,
                 first_test_year: int = FIRST_TEST_YEAR, importance: bool = True) -> dict:
    h = _horizon(spec)
    preds, params_by_year, imp = [], {}, {}
    last_year = (locked_from - pd.Timedelta(days=1)).year
    for y in range(first_test_year, last_year + 1):
        ts = pd.Timestamp(year=y, month=1, day=1)
        te = min(pd.Timestamp(year=y + 1, month=1, day=1), locked_from)
        train = purged(panel, ts, h)
        test = panel[(panel["date"] >= ts) & (panel["date"] < te)]
        test = test[test["label_end_20"].notna() & (test["label_end_20"] < locked_from)]
        if train.empty or test.empty or (spec.get("model") != "rule" and train[spec["target"]].notna().sum() < 1000):
            continue
        params = select_params(spec, train, ts)
        m = fit(spec, train, params)
        preds.append(test.assign(score=m.predict(test)))
        params_by_year[y] = params
        if importance:
            imp[y] = permutation_importance(m, test, "fwd_xs_20")
    if not preds:
        return {"status": "no_data"}
    oos = pd.concat(preds)
    return {"status": "ok", "oos": oos, "params_by_year": params_by_year, "importance_by_year": imp}


def evaluate_oos(oos: pd.DataFrame) -> dict:
    coh = cohort_returns(oos, "score")
    return {"base": perf(coh, COST_BASE), "stress": perf(coh, COST_STRESS),
            "long_short_gross": perf(coh, 0.0, "long_short"), "ic": ic_stats(daily_ic(oos, "score")),
            "_cohorts": coh}


def locked_eval(panel: pd.DataFrame, spec: dict, locked_from: pd.Timestamp) -> dict:
    h = _horizon(spec)
    train = purged(panel, locked_from, h)
    test = panel[(panel["date"] >= locked_from) & panel["fwd_xs_20"].notna()]
    if train.empty or test.empty:
        return {"status": "no_data"}
    params = select_params(spec, train, locked_from)
    m = fit(spec, train, params)
    res = evaluate_oos(test.assign(score=m.predict(test)))
    res.pop("_cohorts", None)
    return {"status": "ok", "params": params, **res}


def aggregate_importance(imp_by_year: dict) -> dict:
    if not imp_by_year:
        return {}
    df = pd.DataFrame(imp_by_year).T
    feat = df.mean().sort_values(ascending=False)
    groups = {g: _r(sum(feat.get(f, 0.0) for f in fs), 5) for g, fs in FEATURE_GROUPS.items()}
    stable = {f: _r((df[f] > 0).mean(), 2) for f in df.columns}
    return {"feature_ic_drop": {k: _r(v, 5) for k, v in feat.items()}, "groups": groups,
            "share_years_positive": stable}


def sector_attribution(oos: pd.DataFrame, min_names: int = 8) -> dict:
    """Mittlerer Rank-IC innerhalb jedes Sektors (wo trennt das Modell?)."""
    out = {}
    for sec, g in oos.groupby("sector"):
        ics = []
        for _, gd in g.groupby("date"):
            gd = gd[["score", "fwd_xs_20"]].dropna()
            if len(gd) >= min_names and gd["score"].nunique() > 1:
                ics.append(gd["score"].rank().corr(gd["fwd_xs_20"].rank()))
        ics = [x for x in ics if np.isfinite(x)]
        if len(ics) >= 20:
            sd = float(np.std(ics, ddof=1))
            out[sec] = {"n_dates": len(ics), "mean_ic": _r(np.mean(ics)),
                        "t_naive": _r(np.mean(ics) / sd * math.sqrt(len(ics)), 2) if sd > 0 else None}
    return dict(sorted(out.items(), key=lambda kv: -(kv[1]["mean_ic"] or 0)))


def regime_attribution(cohorts: pd.DataFrame, panel: pd.DataFrame) -> dict:
    if cohorts.empty:
        return {}
    dfeat = panel.groupby("date")[["vix", "spy_trend_200"]].first()
    c = cohorts.join(dfeat, on="date")
    out = {}
    for name, mask in (("vix_lt_20", c["vix"] < 20), ("vix_ge_20", c["vix"] >= 20),
                       ("spy_uptrend", c["spy_trend_200"] > 0), ("spy_downtrend", c["spy_trend_200"] <= 0)):
        out[name] = perf(c[mask.fillna(False)], COST_BASE)
    return out


# ── Registry, Lock-Log, Entscheidung ─────────────────────────────────────────

def load_registry(path: Path = REGISTRY_PATH) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def spec_hash(spec: dict) -> str:
    core = {k: spec.get(k) for k in ("model", "target", "features", "params_grid", "rule_feature")}
    return hashlib.sha256(json.dumps(core, sort_keys=True, default=str).encode()).hexdigest()[:16]


def check_registry(reg: dict, log_path: Path = REGISTRY_LOG) -> tuple[dict, dict]:
    """-> (status je id, aktualisiertes Log). Geänderte Spezifikation unter
    bestehender id -> invalid_modified (neue id nötig)."""
    book = json.loads(log_path.read_text()) if log_path.exists() else {}
    status = {}
    for spec in reg.get("models") or []:
        mid, hsh = spec["id"], spec_hash(spec)
        e = book.setdefault(mid, {"spec_hash": hsh, "first_seen": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                  "locked_evaluations": 0})
        status[mid] = "valid" if e["spec_hash"] == hsh else "invalid_modified"
    return status, book


def decide(res: dict, bench: dict | None, crit: dict, fwd: dict | None, locked: dict | None) -> dict:
    """Vorab festgelegte Promotionskriterien -> Empfehlung (nie Aktion)."""
    reasons, ok = [], True
    b = res.get("base", {})
    if (b.get("t_months") or -9) < crit.get("wf_min_t_months", 2.0) or (b.get("mean") or -1) <= 0:
        ok = False
        reasons.append(f"Walk-Forward netto {b.get('mean')} mit t={b.get('t_months')} (< {crit.get('wf_min_t_months')})")
    if (b.get("years_positive_share") or 0) < crit.get("wf_min_years_positive", 0.6):
        ok = False
        reasons.append(f"nur {b.get('years_positive_share')} der Testjahre positiv")
    if (res.get("stress", {}).get("mean") or -1) <= 0:
        ok = False
        reasons.append("bei 25 bp/Seite nicht mehr positiv")
    ic = res.get("ic", {})
    if (ic.get("t_months") or -9) < crit.get("min_ic_t", 2.0) or (ic.get("mean_ic") or -1) <= 0:
        ok = False
        reasons.append(f"Rank-IC {ic.get('mean_ic')} (t={ic.get('t_months')}) nicht signifikant")
    if bench:
        bb = bench.get("base", {})
        if (b.get("sharpe_ann") or -9) < (bb.get("sharpe_ann") or 0) + crit.get("min_sharpe_margin", 0.1):
            ok = False
            reasons.append(f"Sharpe {b.get('sharpe_ann')} nicht > Benchmark {bb.get('sharpe_ann')} + Marge")
        if crit.get("max_dd_not_worse") and (b.get("max_dd") or -1) < (bb.get("max_dd") or -1):
            ok = False
            reasons.append(f"Max-DD {b.get('max_dd')} schlechter als Benchmark {bb.get('max_dd')}")
    if crit.get("locked_positive"):
        lb = (locked or {}).get("base", {})
        if (lb.get("mean") or -1) <= 0:
            ok = False
            reasons.append(f"Locked-Holdout netto {lb.get('mean')} nicht > 0")
        if LOCKED_STATUS == "CONTAMINATED":                 # kann nur ablehnen, nie bestätigen (Audit F02)
            reasons.append("Locked-Holdout KONTAMINIERT – nur informativ; bindend ist der Forward-Shadow")
    fb = (fwd or {}).get("base", {})
    fwd_ready = (fb.get("n_cohorts") or 0) >= crit.get("fwd_min_cohorts", 26)
    if not fwd_ready:
        reasons.append(f"Forward-Shadow: {fb.get('n_cohorts', 0)}/{crit.get('fwd_min_cohorts', 26)} fertige Kohorten")
    elif (fb.get("mean") or -1) <= 0:
        ok = False
        reasons.append(f"Forward-Shadow netto {fb.get('mean')} nicht > 0")
    if not ok:
        verdict = "rejected_so_far"
    elif not fwd_ready:
        verdict = "running_forward"
    else:
        verdict = "promote_recommended"
    return {"verdict": verdict, "reasons": reasons}


# ── Prognose-Ledger (Forward-Shadow) ─────────────────────────────────────────

def _read_predictions(pred_dir: Path = PRED_DIR) -> list[dict]:
    rows = []
    for p in sorted(pred_dir.glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    log.warning(f"ml_research: defekte Prognosezeile in {p.name} übersprungen")
    return rows


def _write_predictions(rows: list[dict], pred_dir: Path = PRED_DIR) -> None:
    pred_dir.mkdir(parents=True, exist_ok=True)
    by_month: dict = {}
    for r in rows:
        by_month.setdefault(r["prediction_date"][:7], []).append(r)
    for month, rs in by_month.items():
        (pred_dir / f"{month}.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in rs), encoding="utf-8")


def _code_sha() -> str | None:
    try:
        import subprocess
        return subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def predict_latest(panel: pd.DataFrame, reg: dict, status: dict, pred_dir: Path = PRED_DIR) -> list[dict]:
    """Trainiert jedes gültige Modell auf allen fertigen Labels und schreibt die
    Prognose des jüngsten Stichtags (idempotent je Datum+Modell)."""
    latest = panel["date"].max()
    snap = panel[panel["date"] == latest]
    existing = _read_predictions(pred_dir)
    have = {(r["prediction_date"], r["model_id"]) for r in existing}
    new = []
    fhash = hashlib.sha256(pd.util.hash_pandas_object(snap[list(ALL_FEATURES)].round(6), index=False)
                           .to_numpy().tobytes()).hexdigest()[:16]
    for spec in reg.get("models") or []:
        mid = spec["id"]
        if status.get(mid) != "valid" or (latest.date().isoformat(), mid) in have:
            continue
        cutoff = latest + pd.Timedelta(days=1)
        train = purged(panel, cutoff, _horizon(spec))
        try:
            params = select_params(spec, train, cutoff)
            m = fit(spec, train, params)
        except ValueError as e:
            log.warning(f"ml_research: {mid} nicht trainierbar: {e}")
            continue
        sc = pd.Series(m.predict(snap), index=snap["ticker"].to_numpy()).dropna()
        ranks = sc.rank(pct=True).round(4)
        new.append({"prediction_date": latest.date().isoformat(), "model_id": mid, "spec_hash": spec_hash(spec),
                    "train_cutoff": str(train["label_end_20"].max().date()) if len(train) else None,
                    "params": params, "code_sha": _code_sha(), "features_hash": fhash,
                    "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "n": int(len(ranks)), "rank_pct": ranks.to_dict(), "realized": None})
    if new:
        _write_predictions(existing + new, pred_dir)
    return new


def forward_eval(panel: pd.DataFrame, reg: dict, pred_dir: Path = PRED_DIR) -> dict:
    """Nur Prognosen NACH registered_at mit vollständigen Labels; schreibt die
    realisierten Werte in den Ledger zurück."""
    rows = _read_predictions(pred_dir)
    if not rows:
        return {}
    lab = panel.set_index(["date", "ticker"])[["fwd_xs_20", "mfe_20", "mae_20", "label_end_20"]]
    reg_at = {s["id"]: pd.Timestamp(s.get("registered_at")) for s in reg.get("models") or []}
    per_model: dict = {}
    changed = False
    for r in rows:
        d = pd.Timestamp(r["prediction_date"])
        try:
            g = lab.loc[d]
        except KeyError:
            continue
        g = g[g["fwd_xs_20"].notna()]
        df = pd.DataFrame({"ticker": list(r["rank_pct"]), "score": list(r["rank_pct"].values())})
        df = df.join(g, on="ticker").dropna(subset=["fwd_xs_20"])
        if len(df) < MIN_CROSS_SECTION:
            continue
        df["date"] = d
        if r.get("realized") is None:
            coh = cohort_returns(df, "score")
            r["realized"] = {"top_vs_univ_gross": _r(coh["top_vs_univ"].iloc[0]),
                             "ic": _r(daily_ic(df, "score").iloc[0]) if len(daily_ic(df, "score")) else None,
                             "top_mfe": _r(coh["top_mfe"].iloc[0]), "top_mae": _r(coh["top_mae"].iloc[0])}
            changed = True
        ra = reg_at.get(r["model_id"])
        created = pd.Timestamp(r.get("created_at") or r["prediction_date"])
        if ra is not None and created > ra:
            per_model.setdefault(r["model_id"], []).append(df)
    if changed:
        _write_predictions(rows, pred_dir)
    out = {}
    for mid, dfs in per_model.items():
        allp = pd.concat(dfs)
        coh = cohort_returns(allp, "score")
        out[mid] = {"base": perf(coh, COST_BASE), "ic": ic_stats(daily_ic(allp, "score"))}
    return out


def latest_scores(max_age_days: int = 8, pred_dir: Path = PRED_DIR, today=None) -> dict[str, dict]:
    """Jüngste Shadow-Ränge je Modell {model_id: {ticker: rank_pct}} für den
    Candidate-Ledger (nur Beobachtung, kein Scoring)."""
    today = pd.Timestamp(today or datetime.now(timezone.utc).date())
    best: dict = {}
    for r in _read_predictions(pred_dir):
        d = pd.Timestamp(r["prediction_date"])
        if (today - d).days > max_age_days or d > today:
            continue
        if r["model_id"] not in best or d > pd.Timestamp(best[r["model_id"]]["prediction_date"]):
            best[r["model_id"]] = r
    return {mid: r["rank_pct"] for mid, r in best.items()}


# ── Unsicherheit, Analogien, Kontrafaktisches ────────────────────────────────

UNC = PROTOCOL["uncertainty"]
CARDS_PATH = OUT_DIR / "ml_cards.json"
_UNC_PARAMS = {"max_iter": 200, "learning_rate": 0.05, "max_depth": 3, "min_samples_leaf": 300}


def fit_uncertainty(train: pd.DataFrame) -> dict:
    """Quantil-Modelle für die 60-Tage-Rendite (80-%-Intervall + Median),
    Median-Drawdown (MAE_60) und P(Rendite_60 > up_threshold)."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    t = train[train["fwd_ret_60"].notna() & train["mae_60"].notna()]
    cols = [c for c in ALL_FEATURES if t[c].notna().any()]
    X = t[cols].to_numpy(float)
    y = t["fwd_ret_60"].clip(-0.9, 1.5).to_numpy(float)
    lo, hi = UNC["interval"]
    out = {"cols": cols}
    for name, q in (("q_lo", lo), ("q_mid", 0.5), ("q_hi", hi)):
        out[name] = HistGradientBoostingRegressor(loss="quantile", quantile=q, random_state=0,
                                                  early_stopping=False, **_UNC_PARAMS).fit(X, y)
    out["mae_mid"] = HistGradientBoostingRegressor(loss="quantile", quantile=0.5, random_state=0, early_stopping=False,
                                                   **_UNC_PARAMS).fit(X, t["mae_60"].clip(-0.95, 0).to_numpy(float))
    up = (t["fwd_ret_60"] > UNC["up_threshold"]).astype(int).to_numpy()
    out["p_up"] = HistGradientBoostingClassifier(random_state=0, early_stopping=False, **_UNC_PARAMS).fit(X, up) \
        if up.min() != up.max() else None
    out["base_rate_up"] = float(up.mean())
    return out


def predict_uncertainty(models: dict, df: pd.DataFrame) -> pd.DataFrame:
    X = df[models["cols"]].to_numpy(float)
    q = np.sort(np.column_stack([models[k].predict(X) for k in ("q_lo", "q_mid", "q_hi")]), axis=1)
    out = pd.DataFrame({"q_lo": q[:, 0], "q_mid": q[:, 1], "q_hi": q[:, 2],
                        "mae_mid": np.minimum(models["mae_mid"].predict(X), 0.0)}, index=df.index)
    out["p_up"] = models["p_up"].predict_proba(X)[:, 1] if models["p_up"] is not None else models["base_rate_up"]
    return out


def calibration_wf(panel: pd.DataFrame, locked_from: pd.Timestamp, first_test_year: int = FIRST_TEST_YEAR) -> dict:
    """Weiß das System, wann es wenig weiß? Walk-Forward-Prüfung: Abdeckung des
    80-%-Intervalls und Brier-Score von P(>+10 %) gegen die Basisrate – roh UND
    nach PIT-Korrektur: konformale Verbreiterung (CQR) und isotonische
    Rekalibrierung, beide NUR aus den OOS-Residuen des Vorjahres gelernt."""
    from sklearn.isotonic import IsotonicRegression
    rows, prev = [], None
    target = UNC["interval"][1] - UNC["interval"][0]
    for y in range(first_test_year, (locked_from - pd.Timedelta(days=1)).year + 1):
        ts = pd.Timestamp(year=y, month=1, day=1)
        train = purged(panel, ts, 60)
        test = panel[(panel["date"] >= ts) & (panel["date"] < min(ts + pd.DateOffset(years=1), locked_from))]
        test = test[test["label_end_60"].notna() & (test["label_end_60"] < locked_from) & test["fwd_ret_60"].notna()]
        if len(train) < 5000 or test.empty:
            continue
        m = fit_uncertainty(train)
        p = predict_uncertainty(m, test)
        r = test["fwd_ret_60"]
        up = (r > UNC["up_threshold"]).astype(float)
        row = {"year": y, "n": int(len(test)),
               "coverage_80": _r(((r >= p["q_lo"]) & (r <= p["q_hi"])).mean(), 4),
               "below_lo": _r((r < p["q_lo"]).mean(), 4), "above_hi": _r((r > p["q_hi"]).mean(), 4),
               "brier": _r(((p["p_up"] - up) ** 2).mean(), 5),
               "brier_base": _r(((m["base_rate_up"] - up) ** 2).mean(), 5),
               "mae_coverage_50": _r((test["mae_60"] >= p["mae_mid"]).mean(), 4)}
        if prev is not None:
            lo, hi = p["q_lo"] - prev["qhat"], p["q_hi"] + prev["qhat"]
            p_rc = np.interp(p["p_up"], prev["iso_x"], prev["iso_y"])
            row.update(coverage_80_conformal=_r(((r >= lo) & (r <= hi)).mean(), 4),
                       brier_recal=_r(((p_rc - up) ** 2).mean(), 5), qhat_used=_r(prev["qhat"], 4))
        # Nonkonformität dieses Jahres -> Korrektur für das FOLGEJAHR
        E = np.maximum(p["q_lo"] - r, r - p["q_hi"])
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p["p_up"], up)
        prev = {"qhat": float(np.quantile(E, target)), "iso_x": iso.X_thresholds_.tolist(),
                "iso_y": iso.y_thresholds_.tolist()}
        rows.append(row)
    if not rows:
        return {"status": "no_data"}
    df = pd.DataFrame(rows)
    cov = float(np.average(df["coverage_80"], weights=df["n"]))
    brier, base = float(np.average(df["brier"], weights=df["n"])), float(np.average(df["brier_base"], weights=df["n"]))
    out = {"status": "ok", "by_year": rows, "coverage": _r(cov, 4), "target": target,
           "interval_calibrated_raw": abs(cov - target) <= UNC["coverage_tolerance"],
           "brier": _r(brier, 5), "brier_base": _r(base, 5), "p_up_skill": _r(1 - brier / base, 4) if base > 0 else None}
    c = df.dropna(subset=["coverage_80_conformal"]) if "coverage_80_conformal" in df else df.iloc[0:0]
    if len(c):
        cc = float(np.average(c["coverage_80_conformal"], weights=c["n"]))
        br = float(np.average(c["brier_recal"], weights=c["n"]))
        bb = float(np.average(c["brier_base"], weights=c["n"]))
        worst = float((c["coverage_80_conformal"] - target).abs().max())
        out.update(coverage_conformal=_r(cc, 4), worst_year_gap_conformal=_r(worst, 4),
                   brier_recal=_r(br, 5), p_up_skill_recal=_r(1 - br / bb, 4) if bb > 0 else None,
                   correction={"qhat": _r(prev["qhat"], 5), "iso_x": [_r(x, 5) for x in prev["iso_x"]],
                               "iso_y": [_r(x, 5) for x in prev["iso_y"]], "from_year": rows[-1]["year"]})
        # kalibriert nur, wenn die KORRIGIERTE Abdeckung im Mittel passt; Skill nur, wenn > 0
        out["interval_calibrated"] = abs(cc - target) <= UNC["coverage_tolerance"]
        out["p_up_informative"] = (out["p_up_skill_recal"] or 0) > 0
    else:
        out["interval_calibrated"] = out["interval_calibrated_raw"]
        out["p_up_informative"] = (out["p_up_skill"] or 0) > 0
    return out


def _level(x: float | None, cuts: tuple[float, float], labels=("LOW", "MEDIUM", "HIGH")) -> str:
    if x is None or not np.isfinite(x):
        return "UNKNOWN"
    return labels[0] if x < cuts[0] else labels[1] if x < cuts[1] else labels[2]


def analogs(hist: pd.DataFrame, query: pd.DataFrame, k: int) -> pd.DataFrame:
    """k nächste historische Situationen (Aktienmerkmals-Ränge) mit fertigem
    60-Tage-Label vor dem Stichtag: Verteilung dessen, was danach geschah."""
    from sklearn.neighbors import NearestNeighbors
    cols = list(STOCK_FEATURES)
    h = hist[hist["fwd_ret_60"].notna()]
    if len(h) < k or query.empty:
        return pd.DataFrame(index=query.index)
    nn = NearestNeighbors(n_neighbors=k).fit(h[cols].fillna(0.0).to_numpy(float))
    _, idx = nn.kneighbors(query[cols].fillna(0.0).to_numpy(float))
    r = h["fwd_ret_60"].to_numpy()[idx]
    mae = h["mae_60"].to_numpy()[idx]
    mfe = h["mfe_60"].to_numpy()[idx] if "mfe_60" in h else np.full(idx.shape, np.nan)
    years = pd.DatetimeIndex(h["date"]).year.to_numpy()[idx]
    return pd.DataFrame({"analog_median_ret_60": np.nanmedian(r, axis=1), "analog_share_positive": (r > 0).mean(axis=1),
                         "analog_median_mae_60": np.nanmedian(mae, axis=1),
                         "analog_median_mfe_60": np.nanmedian(mfe, axis=1), "analog_n": k,
                         "analog_years": [f"{a.min()}-{a.max()}" for a in years]}, index=query.index)


def counterfactual_drivers(models: dict, row: pd.DataFrame, top: int = 3) -> dict:
    """Welche Merkmale tragen die Median-Prognose? Jedes Aktienmerkmal wird auf
    neutral (Querschnittsmitte) gesetzt; die größten Rückgänge sind das, was
    kippen müsste, damit das Setup schlechter wird."""
    base = float(predict_uncertainty(models, row)["q_mid"].iloc[0])
    deltas = {}
    for f in STOCK_FEATURES:
        if f not in models["cols"] or pd.isna(row[f].iloc[0]):
            continue
        r2 = row.copy()
        r2[f] = 0.0
        deltas[f] = base - float(predict_uncertainty(models, r2)["q_mid"].iloc[0])
    pos = sorted(((k, v) for k, v in deltas.items() if v > 0), key=lambda kv: -kv[1])[:top]
    neg = sorted(((k, v) for k, v in deltas.items() if v < 0), key=lambda kv: kv[1])[:top]
    return {"supports": {k: _r(v, 4) for k, v in pos}, "weighs_against": {k: _r(v, 4) for k, v in neg}}


def build_cards(panel: pd.DataFrame, rank_by_model: dict[str, dict], calibration: dict | None = None,
                top_n_drivers: int = 60) -> dict:
    """Entscheidungskarte je Ticker des jüngsten Stichtags."""
    latest = panel["date"].max()
    snap = panel[panel["date"] == latest]
    train = purged(panel, latest + pd.Timedelta(days=1), 60)
    if len(train) < 5000 or snap.empty:
        return {}
    m = fit_uncertainty(train)
    u = predict_uncertainty(m, snap)
    an = analogs(train, snap, int(UNC["analogs_k"]))
    ranks = pd.DataFrame(rank_by_model).reindex(snap["ticker"])
    challengers = [c for c in ranks.columns if not c.startswith("momentum")]
    disagree = ranks[challengers].std(axis=1) if len(challengers) >= 2 else pd.Series(np.nan, index=ranks.index)
    mean_rank = ranks[challengers].mean(axis=1) if challengers else pd.Series(np.nan, index=ranks.index)
    feat_cov = snap[list(ALL_FEATURES)].notna().mean(axis=1)
    d = snap.iloc[0]
    date_avail = float(pd.Series([d[c] for c in DATE_FEATURES]).notna().mean())
    near_edge = (abs((d["vix"] or 20) - 20) < 2) or (abs(d["spy_trend_200"] or 0) < 0.02)
    regime_conf = "HIGH" if date_avail >= 0.9 and not near_edge else "MEDIUM" if date_avail >= 0.7 else "LOW"
    top_tickers = set(mean_rank.sort_values(ascending=False).head(top_n_drivers).index)
    cards = {}
    corr = (calibration or {}).get("correction") or {}
    if corr:
        u["q_lo"] = u["q_lo"] - corr["qhat"]
        u["q_hi"] = u["q_hi"] + corr["qhat"]
        u["p_up"] = np.interp(u["p_up"], corr["iso_x"], corr["iso_y"])
    for i, row in snap.iterrows():
        t = row["ticker"]
        c = {"expected_return_60": _r(u.at[i, "q_mid"], 4),
             "interval_80": [_r(u.at[i, "q_lo"], 4), _r(u.at[i, "q_hi"], 4)],
             "expected_drawdown_60": _r(u.at[i, "mae_mid"], 4),
             "p_return_gt_10": _r(u.at[i, "p_up"], 3),
             "model_disagreement": _level(disagree.get(t), (0.15, 0.25), ("LOW", "MEDIUM", "HIGH")),
             "model_disagreement_sd": _r(disagree.get(t), 3), "mean_challenger_rank": _r(mean_rank.get(t), 3),
             "data_quality": _level(feat_cov.at[i], (0.8, 0.95)), "regime_confidence": regime_conf,
             "interval_calibrated": (calibration or {}).get("interval_calibrated"),
             "interval_corrected": bool(corr), "p_up_informative": (calibration or {}).get("p_up_informative"),
             "p_up_skill": (calibration or {}).get("p_up_skill_recal", (calibration or {}).get("p_up_skill"))}
        if i in an.index and len(an.columns):
            c["analogs"] = {k: (_r(v, 4) if not isinstance(v, str) else v) for k, v in an.loc[i].items()}
            c["analogs"]["oos_validated"] = False          # Audit P3-3: Nutzen nie OOS gemessen
        if t in top_tickers:
            c["counterfactual"] = counterfactual_drivers(m, snap.loc[[i]])
        cards[t] = c
    return {"date": str(latest.date()), "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "note": "SHADOW – Forschungskarte, keine Handelsempfehlung", "cards": cards}


def latest_cards(max_age_days: int = 8, path: Path | None = None, today=None) -> dict:
    path = path or CARDS_PATH
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    today = pd.Timestamp(today or datetime.now(timezone.utc).date())
    if (today - pd.Timestamp(data.get("date", "1900-01-01"))).days > max_age_days:
        return {}
    return data.get("cards", {})


# ── Daten (nur CI, Netzwerk) ─────────────────────────────────────────────────

def fetch_data(tickers: list[str], start: str = "2014-01-01"):
    import yfinance as yf
    frames: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), 80):
        chunk = tickers[i:i + 80]
        data = yf.download(chunk, start=start, auto_adjust=True, group_by="ticker", progress=False, threads=True)
        for t in chunk:
            try:
                df = data[t][["Open", "High", "Low", "Close", "Volume"]].dropna(how="all")
            except (KeyError, TypeError):
                log.warning(f"ml_research: keine Daten für {t}")
                continue
            if len(df):
                frames[t] = df
    series = {}
    for sym, adj in (("SPY", True), ("^VIX", False), ("^TNX", False), ("^IRX", False)):
        d = yf.download(sym, start=start, auto_adjust=adj, progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        series[sym] = d
    return frames, series["SPY"][["Open", "Close"]], series["^VIX"]["Close"], \
        series["^TNX"]["Close"], series["^IRX"]["Close"]


def survivorship_report(panel: pd.DataFrame, membership: dict, label: str = "fwd_xs_20") -> dict:
    """Quantifiziert den verbleibenden Survivorship-Bias (Kurse delisteter Titel fehlen bei Yahoo):
    Anteil der Indexmitglieder je Stichtag ohne Panelzeile und eine Worst-Case-Schranke für das
    Querschnittsmittel, falls die fehlenden Titel am 10-%-Quantil gelegen hätten."""
    from modules.universe import is_member
    if panel.empty or not membership:
        return {"available": False}
    present = panel.groupby("date")["ticker"].apply(set)
    rows = []
    for d, have in present.items():
        ds = str(pd.Timestamp(d).date())
        members = {t for t, iv in membership.items() if is_member(iv, ds)}
        if not members:
            continue
        miss = len(members - have) / len(members)
        x = panel.loc[panel["date"] == d, label].dropna() if label in panel else pd.Series(dtype=float)
        shift = miss * (float(x.quantile(0.10)) - float(x.mean())) if len(x) >= 20 else None
        rows.append((pd.Timestamp(d).year, miss, shift))
    if not rows:
        return {"available": False}
    df = pd.DataFrame(rows, columns=["year", "missing", "shift"])
    return {"available": True, "label": label,
            "mean_missing_member_share": _r(float(df["missing"].mean()), 4),
            "max_missing_member_share": _r(float(df["missing"].max()), 4),
            "missing_share_by_year": {int(y): _r(float(g["missing"].mean()), 4) for y, g in df.groupby("year")},
            "worst_case_xs_mean_shift": _r(float(df["shift"].dropna().mean()), 5) if df["shift"].notna().any() else None,
            "note": "Schranke: fehlende Mitglieder am 10-%-Quantil des Stichtags; verschiebt das Querschnittsmittel "
                    "(Benchmark der Top-Dezil-Rendite) um diesen Betrag – Top-Dezil-Überrenditen darunter sind "
                    "nicht von Survivorship unterscheidbar"}


def pit_sector(p: pd.DataFrame, history: dict) -> pd.Series:
    """Sektor je (date, ticker) = letzte Beobachtung mit observed_at <= date; sonst 'unknown'."""
    out = pd.Series("unknown", index=p.index, dtype=object)
    for t, idx in p.groupby("ticker").groups.items():
        obs = sorted((pd.Timestamp(d), sec) for d, sec in (history.get(t) or []) if sec)
        if not obs:
            continue
        dates = p.loc[idx, "date"]
        for i, d in zip(idx, dates):
            cur = None
            for od, sec in obs:
                if od <= d:
                    cur = sec
                else:
                    break
            if cur:
                out.at[i] = cur
    return out


def read_sector_history(path: Path | None = None) -> dict:
    path = path or SECTOR_HISTORY
    h: dict[str, list] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                h.setdefault(r["ticker"], []).append((r["observed_at"], r["sector"]))
    return h


def record_sector_observations(obs: list[dict], path: Path | None = None) -> int:
    """Append-only; je (ticker, sector) nur die ERSTE Beobachtung (frühester bekannter Zeitpunkt)."""
    path = path or SECTOR_HISTORY
    have = {(t, s) for t, v in read_sector_history(path).items() for _, s in v}
    new = [o for o in obs if o.get("sector") and (o["ticker"], o["sector"]) not in have]
    new.sort(key=lambda o: o["observed_at"])
    seen = set()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for o in new:
            if (o["ticker"], o["sector"]) in seen:
                continue
            seen.add((o["ticker"], o["sector"]))
            fh.write(json.dumps({k: o[k] for k in ("ticker", "sector", "observed_at", "source")}) + "\n")
    return len(seen)


def sector_history(tickers: list[str], budget_s: float = 600.0, today: str | None = None) -> dict:
    """Point-in-time-Sektoren: Candidate-Ledger (Entscheidungsdatum = exakt PIT), danach aktueller
    Abruf (sector_map) mit Abrufdatum als observed_at. Nie rückwirkend."""
    today = today or datetime.now(timezone.utc).date().isoformat()
    obs = []
    for p in sorted(Path("outputs/candidate_ledger").glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            sec = (r.get("features") or {}).get("sector")
            if sec and r.get("ticker") and r.get("date"):
                obs.append({"ticker": r["ticker"], "sector": sec, "observed_at": str(r["date"])[:10],
                            "source": "candidate_ledger"})
    for t, sec in sector_map(tickers, budget_s).items():
        obs.append({"ticker": t, "sector": sec, "observed_at": today, "source": "sector_map_retrieval"})
    record_sector_observations(obs)
    return read_sector_history()


def sector_map(tickers: list[str], budget_s: float = 600.0) -> dict:
    """Sektor je Ticker: Cache -> Candidate-Ledger -> yfinance-Info (Budget)."""
    import time
    m = json.loads(SECTOR_CACHE.read_text()) if SECTOR_CACHE.exists() else {}
    for p in sorted(Path("outputs/candidate_ledger").glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            sec = (r.get("features") or {}).get("sector")
            if sec and r.get("ticker") not in m:
                m[r["ticker"]] = sec
    missing = [t for t in tickers if t not in m]
    t0 = time.monotonic()
    if missing:
        import yfinance as yf
        for t in missing:
            if time.monotonic() - t0 > budget_s:
                log.warning(f"ml_research: Sektor-Budget erschöpft, {len(missing)} offen")
                break
            try:
                sec = (yf.Ticker(t).info or {}).get("sector")
            except Exception as e:  # noqa: BLE001 – yfinance wirft uneinheitlich; Sektor ist optional
                log.warning(f"ml_research: Sektor {t} nicht abrufbar: {e}")
                continue
            if sec:
                m[t] = sec
    SECTOR_CACHE.parent.mkdir(parents=True, exist_ok=True)
    SECTOR_CACHE.write_text(json.dumps(dict(sorted(m.items())), indent=0))
    return m


def load_macro(archive_root: str = "outputs/external_data") -> list:
    try:
        from modules.external import regime
        from modules.external.archive import ExternalArchive
        return ExternalArchive(archive_root).load(regime.SOURCE_ID)
    except (OSError, ValueError, KeyError) as e:
        log.warning(f"ml_research: Makro-Archiv nicht lesbar ({e}) -> Makro-Features NaN")
        return []


# ── Report ───────────────────────────────────────────────────────────────────

def _survivorship_line(u: dict) -> str:
    if u.get("pit"):
        return (f"Survivorship: PIT-Universum ({u.get('n_tickers_ever')} Titel je Mitglied, {u.get('n_changes')} Änderungen; "
                f"entfernte Titel mit Kursen {u.get('n_removed_with_prices')}/{u.get('n_removed_since_start')} – "
                f"Restbias: {u.get('residual_bias')}).")
    return "Survivorship: heutige Indexliste (Querschnittsvergleich dämpft den Bias)."


def render_md(rep: dict) -> str:
    L = [f"# ML-Research (Shadow) – {rep['generated']}", "",
         f"Panel: {rep.get('n_rows')} Zeilen, {rep.get('n_tickers')} Ticker, {rep.get('period')} · "
         f"Locked-Holdout ab {rep.get('locked_from')} · Champion: {rep.get('champion') or '— (keiner)'}",
         "Bewertung: Top-Dezil minus Querschnittsmittel, 20 Handelstage, netto 10 bp/Seite. " +
         _survivorship_line(rep.get("universe") or {}) + " Keine Produktionswirkung.", "",
         "| Modell | Status | WF netto | t | Sharpe | Max-DD | Jahre + | IC | IC t | 25bp | Locked netto | Forward n | Forward netto | Asym Top/Univ | Verdikt |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for mid, m in rep.get("models", {}).items():
        b, s, ic = m.get("wf", {}).get("base", {}), m.get("wf", {}).get("stress", {}), m.get("wf", {}).get("ic", {})
        lk = (m.get("locked") or {}).get("base", {})
        fw = (m.get("forward") or {}).get("base", {})
        L.append(f"| {mid} | {m.get('registry_status')} | {b.get('mean')} | {b.get('t_months')} | {b.get('sharpe_ann')} | "
                 f"{b.get('max_dd')} | {b.get('years_positive_share')} | {ic.get('mean_ic')} | {ic.get('t_months')} | "
                 f"{s.get('mean')} | {lk.get('mean')} | {fw.get('n_cohorts', 0)} | {fw.get('mean')} | "
                 f"{b.get('top_asymmetry')}/{b.get('univ_asymmetry')} | {m.get('decision', {}).get('verdict')} |")
    cal = rep.get("calibration") or {}
    if cal.get("status") == "ok":
        L += ["", "## Unsicherheit (Walk-Forward-Kalibrierung)", "",
              f"- 80-%-Intervall der 60-Tage-Rendite deckt roh {cal['coverage']} ab (Soll {cal['target']}); "
              f"nach konformaler Vorjahres-Korrektur {cal.get('coverage_conformal')} "
              f"(schlechtestes Jahr ±{cal.get('worst_year_gap_conformal')}) -> "
              f"{'kalibriert' if cal['interval_calibrated'] else 'NICHT kalibriert'}",
              f"- P(>+10 %) nach isotonischer Vorjahres-Rekalibrierung: Skill {cal.get('p_up_skill_recal')}",
              f"- P(Rendite_60 > +10 %): Brier {cal['brier']} vs. Basisrate {cal['brier_base']} "
              f"(Skill {cal['p_up_skill']}; <= 0 heißt: keine Information über die Basisrate hinaus)"]
    for mid, m in rep.get("models", {}).items():
        L += ["", f"## {mid}", ""]
        for r in m.get("decision", {}).get("reasons", []):
            L.append(f"- {r}")
        imp = m.get("attribution", {}).get("importance", {})
        if imp.get("groups"):
            L.append(f"- Feature-Gruppen (IC-Verlust bei Permutation): {imp['groups']}")
            top = list(imp.get("feature_ic_drop", {}).items())[:6]
            L.append(f"- Wichtigste Features: {top}")
        sec = m.get("attribution", {}).get("sectors", {})
        if sec:
            L.append("- Sektoren (Rank-IC innerhalb Sektor): " +
                     ", ".join(f"{k} {v['mean_ic']} (t {v['t_naive']})" for k, v in sec.items()))
        rg = m.get("attribution", {}).get("regimes", {})
        if rg:
            L.append("- Regime (netto): " + ", ".join(f"{k} {v.get('mean')} (n {v.get('n_cohorts')})" for k, v in rg.items()))
        if m.get("params_by_year"):
            L.append(f"- Hyperparameter je Testjahr (innere Validierung): {m['params_by_year']}")
    return "\n".join(L) + "\n"


UNIVERSE_INFO: dict = {}


def build_research_panel(mode: str = "full") -> pd.DataFrame:
    """Point-in-Time-Universum (S&P 500 inkl. später entfernter Titel, Zeilen
    nur während der Mitgliedschaft). Ist die Historie nicht verfügbar, bricht
    der Lauf ab – kein stiller Rückfall auf die heutige Liste (Audit F01).

    Replay (Audit F14 / P2-5): ML_PANEL_SNAPSHOT=<pickle> lädt einen
    eingefrorenen Panel-Snapshot statt neu bei Yahoo zu laden -> identische
    Eingangsdaten für Reproduktionsläufe."""
    import os
    import pickle
    snap = os.environ.get("ML_PANEL_SNAPSHOT")
    if snap:
        with open(snap, "rb") as fh:
            panel = pickle.load(fh)  # noqa: S301 – eigener, als CI-Artefakt gesicherter Snapshot
        UNIVERSE_INFO.clear()
        UNIVERSE_INFO.update({"replay_snapshot": snap})
        log.info(f"ml_research: REPLAY aus Snapshot {snap} ({len(panel)} Zeilen)")
        return panel
    from modules.universe import research_universe
    uni = research_universe("2014-01-01")
    tickers = uni["tickers"]
    frames, spy, vix, tnx, irx = fetch_data(tickers)
    removed = uni["removed_since_start"]
    with_data = [t for t in removed if t in frames]
    UNIVERSE_INFO.clear()
    UNIVERSE_INFO.update({"pit": True, "source": uni["source"], "n_changes": uni["n_changes"],
                          "first_change": uni["first_change"], "n_tickers_ever": len(tickers),
                          "n_removed_since_start": len(removed), "n_removed_with_prices": len(with_data),
                          "removed_price_coverage": _r(len(with_data) / len(removed), 3) if removed else None,
                          "residual_bias": "entfernte Titel ohne Yahoo-Kurse fehlen weiterhin",
                          "sector_assignment": "point-in-time: Sektor erst ab erster Beobachtung (Candidate-"
                                               "Ledger/Abrufdatum, outputs/research/sector_history.jsonl); davor "
                                               "'unknown' – historische Sektor-Attribution daher meist unknown"})
    log.info(f"ml_research: PIT-Universum {len(tickers)} Titel, davon {len(removed)} entfernt "
             f"({len(with_data)} mit Kursen)")
    now = datetime.now(timezone.utc)
    if now.hour < 21:      # heutiger Balken evtl. unfertig (US-Close 20/21 UTC) -> nie als Close_t nutzen
        spy = spy[spy.index.date < now.date()]
    log.info(f"ml_research: {len(frames)}/{len(tickers)} Ticker geladen")
    pred_dates = {r["prediction_date"] for r in _read_predictions()}
    panel = build_panel(frames, spy, vix, tnx, irx, load_macro(), extra_dates=pred_dates | {str(spy.index.max().date())},
                        sectors=sector_history(tickers) if mode == "full" else read_sector_history(),
                        membership=uni["intervals"])
    if not panel.empty:
        UNIVERSE_INFO["sector_pit_coverage"] = _r(float((panel["sector"] != "unknown").mean()), 4)
        UNIVERSE_INFO["survivorship"] = survivorship_report(panel, uni["intervals"])
    from modules.alt_data.feature_store import attach    # SHADOW-Spalten; nicht in ALL_FEATURES
    return attach(panel) if not panel.empty else panel


def run(mode: str = "full") -> dict:
    import os
    import pickle
    reg = load_registry()
    status, book = check_registry(reg)
    locked_from = LOCKED_FROM
    panel = build_research_panel(mode)
    cache = os.environ.get("ML_PANEL_CACHE")
    if cache:
        with open(cache, "wb") as fh:
            pickle.dump(panel, fh)
    from modules.meta_learning import panel_hash
    rep = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "mode": mode,
           "n_rows": int(len(panel)), "n_tickers": int(panel["ticker"].nunique()),
           "panel_hash": panel_hash(panel), "code_sha": _code_sha(),
           "period": f"{panel['date'].min().date()}..{panel['date'].max().date()}",
           "locked_from": str(locked_from.date()), "locked_status": LOCKED_STATUS,
           "forward_holdout_from": FORWARD_HOLDOUT_FROM, "champion": reg.get("champion"),
           "prereg": "docs/research/PREREG_ml_research_2026-09-29.md", "models": {},
           "universe": dict(UNIVERSE_INFO)}
    new = predict_latest(panel, reg, status)
    rep["new_predictions"] = [f"{r['model_id']}@{r['prediction_date']}" for r in new]
    fwd = forward_eval(panel, reg)
    calib = calibration_wf(panel, locked_from) if mode == "full" else \
        (json.loads((OUT_DIR / "ml_research.json").read_text()).get("calibration")
         if (OUT_DIR / "ml_research.json").exists() else None)
    rep["calibration"] = calib
    latest = panel["date"].max().date().isoformat()
    ranks = {r["model_id"]: r["rank_pct"] for r in _read_predictions() if r["prediction_date"] == latest}
    cards = build_cards(panel, ranks, calib)
    if cards:
        CARDS_PATH.parent.mkdir(parents=True, exist_ok=True)
        CARDS_PATH.write_text(json.dumps(cards, indent=1, ensure_ascii=False, default=str))
        rep["cards"] = {"date": cards["date"], "n": len(cards["cards"])}
    crit = PROTOCOL["promotion_criteria"]
    bench_id = reg.get("champion") or next((s["id"] for s in reg.get("models", []) if s.get("role") == "benchmark"), None)
    if mode == "full":
        results = {}
        for spec in reg.get("models") or []:
            mid = spec["id"]
            if status[mid] != "valid":
                rep["models"][mid] = {"registry_status": status[mid]}
                continue
            wf = walk_forward(panel, spec, locked_from)
            if wf["status"] != "ok":
                rep["models"][mid] = {"registry_status": "valid", "wf": {}, "note": wf["status"]}
                continue
            ev = evaluate_oos(wf["oos"])
            coh = ev.pop("_cohorts")
            if book[mid].get("locked_result") is not None:      # je spec_hash nur EINE Locked-Auswertung (F02)
                locked = {**book[mid]["locked_result"], "reused": True}
            else:
                locked = locked_eval(panel, spec, locked_from)
                book[mid]["locked_evaluations"] = book[mid].get("locked_evaluations", 0) + 1
                book[mid]["locked_result"] = {k: v for k, v in locked.items() if k in ("status", "params", "base", "stress", "ic")}
                book[mid]["locked_result"]["evaluated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            locked["holdout_status"] = LOCKED_STATUS
            results[mid] = ev
            rep["models"][mid] = {
                "registry_status": "valid", "role": spec.get("role"), "wf": ev, "locked": locked,
                "forward": fwd.get(mid), "params_by_year": {str(k): v for k, v in wf["params_by_year"].items()},
                "attribution": {"importance": aggregate_importance(wf["importance_by_year"]),
                                "sectors": sector_attribution(wf["oos"]),
                                "regimes": regime_attribution(coh, panel)},
            }
        for mid, m in rep["models"].items():
            if m.get("role") == "challenger" and mid in results:
                m["decision"] = decide(results[mid], results.get(bench_id), crit, fwd.get(mid), m.get("locked"))
        rep["benchmark"] = bench_id
        rep["locked_evaluations_total"] = sum(v.get("locked_evaluations", 0) for v in book.values())
        REGISTRY_LOG.parent.mkdir(parents=True, exist_ok=True)
        REGISTRY_LOG.write_text(json.dumps(book, indent=2, sort_keys=True))
        (OUT_DIR / "ml_research.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str))
        (OUT_DIR / "ml_research.md").write_text(render_md(rep), encoding="utf-8")
    else:
        REGISTRY_LOG.parent.mkdir(parents=True, exist_ok=True)
        REGISTRY_LOG.write_text(json.dumps(book, indent=2, sort_keys=True))
        rep["forward"] = fwd
        (OUT_DIR / "ml_forward.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str))
    return rep


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("full", "weekly"), default="full")
    rep = run(ap.parse_args().mode)
    print(render_md(rep) if rep.get("mode") == "full" else json.dumps(rep.get("forward"), indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
modules/world_model.py – Market/Economic World Model (NUR SHADOW)

    python -m modules.world_model        (CI: .github/workflows/ml_research.yml)

Versionierter, messbarer Zustand von Wirtschaft und Markt je Stichtag, nur aus
PIT-Daten gebildet:
  * ALFRED-Vintages aus dem Archiv (available_at = Veröffentlichungstag):
    fred_regime_macro (CPI, Fed-Bilanz, USD, WTI, NFCI ab 2025),
    fred_us_macro (INDPRO, UMCSENT), bts_freight_tsi (Fracht-TSI),
    fred_world_macro (PAYEMS, ICSA, RSAFS, ISRATIO)
  * Marktpreise bis Close t (SPY, VIX, VIX3M, Zinsen, HYG/LQD, IEF, Sektor-ETFs,
    Kupfer/Gold)
  * Marktbreite aus dem Feature-Store (heutiges Universum -> Survivorship-Hinweis)
Keine LLM-Klassifikation. Jede Dimension ist der Mittelwert expandierend
normierter Indikatoren (z-Score nur mit Vergangenheit); Label high/neutral/low;
Unsicherheit aus Abdeckung und Abstand zur Schwelle. Nicht verfügbare
Dimensionen (Earnings-Revisionen PIT) werden als "unavailable" ausgewiesen.

Validierung (config/intelligence_protocol.yaml, world_model): Sagt der Zustand
Drawdowns, Volatilität, Sektorrotation, Aktien vs. Anleihen, Regimewechsel und
die Wirksamkeit der Querschnittssignale besser voraus als die bestehende
Regime-Engine? Entscheidung KEEP / MODIFY / REJECT nach vorab festgelegter Regel.
"""

from __future__ import annotations

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

PROTOCOL_PATH = Path(__file__).resolve().parent.parent / "config" / "intelligence_protocol.yaml"
WP = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))["world_model"]
OUT_DIR = Path("outputs/research")
STATE_LOG = OUT_DIR / "world_state.jsonl"
OUT_JSON = OUT_DIR / "world_model.json"
OUT_MD = OUT_DIR / "world_model.md"

MARKET_SYMBOLS = ("SPY", "^VIX", "^VIX3M", "^TNX", "^IRX", "HYG", "LQD", "IEF", "XLK", "XLF", "XLE", "XLI",
                  "XLV", "XLY", "XLP", "XLU", "XLB", "HG=F", "GC=F")
CYCLICALS, DEFENSIVES = ("XLI", "XLY", "XLB", "XLF"), ("XLP", "XLU", "XLV")

# Makro-Indikatoren: (Quelle, Metrik, Transformation, Höchstalter der jüngsten Periode in Tagen)
MACRO = {
    "indpro_yoy": ("fred_us_macro", "us_indpro", ("chg_days", 365), 75),
    "indpro_3m": ("fred_us_macro", "us_indpro", ("chg_days", 91), 75),
    "umcsent": ("fred_us_macro", "us_umcsent", ("level", None), 75),
    "cpi_yoy": ("fred_regime_macro", "us_cpi", ("chg_days", 365), 75),
    "cpi_trend": ("fred_regime_macro", "us_cpi", ("accel", None), 75),
    "fed_assets_13w": ("fred_regime_macro", "fed_total_assets", ("chg_days", 91), 21),
    "usd_63d": ("fred_regime_macro", "usd_broad", ("chg_days", 91), 10),
    "wti_63d": ("fred_regime_macro", "wti", ("chg_days", 91), 10),
    "nfci_neg": ("fred_regime_macro", "nfci", ("neg_level", None), 21),
    "nfci_credit_neg": ("fred_regime_macro", "nfci_credit", ("neg_level", None), 21),
    "freight_yoy": ("bts_freight_tsi", "us_freight_tsi", ("chg_days", 365), 120),
    "payems_3m": ("fred_world_macro", "us_payems", ("chg_days", 91), 75),
    "claims_13w_neg": ("fred_world_macro", "us_initial_claims", ("neg_chg_days", 91), 21),
    "retail_yoy": ("fred_world_macro", "us_retail_sales", ("chg_days", 365), 75),
    "inv_sales_yoy": ("fred_world_macro", "us_inventory_sales_ratio", ("chg_days", 365), 100),
}

# Dimension -> Indikatoren (Vorzeichen so, dass +1 = "mehr" der Dimension)
DIMENSIONS = {
    "growth": ("indpro_yoy", "copper_gold_63d", "cyc_def_63d"),
    "inflation": ("cpi_yoy", "cpi_trend"),
    "liquidity": ("fed_assets_13w", "nfci_neg"),
    "interest_rates": ("tnx", "tnx_63d", "curve"),
    "credit_conditions": ("hyg_lqd_63d", "nfci_credit_neg"),
    "risk_appetite": ("spy_trend_200", "xly_xlp_63d", "breadth_13w"),
    "volatility": ("vix", "vix_term", "spy_rv20"),
    "earnings_momentum": (),
    "consumer_demand": ("umcsent", "retail_yoy"),
    "industrial_activity": ("indpro_3m", "xli_spy_63d"),
    "freight_supply_chain": ("freight_yoy",),
    "inventories": ("inv_sales_yoy",),
    "commodities": ("wti_63d", "copper_63d"),
    "fx_usd": ("usd_63d",),
    "labour_market": ("payems_3m", "claims_13w_neg"),
    "breadth": ("breadth_13w", "breadth_near_high"),
}
UNAVAILABLE_REASON = {"earnings_momentum": "keine PIT-Historie von Gewinnrevisionen (nur kommerziell, I/B/E/S o.ä.)"}


# ── PIT-Makro effizient ──────────────────────────────────────────────────────

def pit_snapshots(observations, metric: str, dates: list[pd.Timestamp]) -> dict:
    """Je Stichtag die zum Tagesbeginn bekannte Reihe {Periode: Wert} (jüngste
    Vintage mit available_at < Stichtag). Eine Sortierung, ein Durchlauf."""
    obs = sorted((o for o in observations if o.metric == metric and o.available_at is not None and o.value is not None),
                 key=lambda o: o.available_at)
    out, best, i = {}, {}, 0
    for d in sorted(dates):
        cut = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
        while i < len(obs) and obs[i].available_at < cut:
            o = obs[i]
            cur = best.get(o.observation_time)
            if cur is None or o.available_at >= cur[0]:
                best[o.observation_time] = (o.available_at, float(o.value))
            i += 1
        out[d] = pd.Series({k: v[1] for k, v in best.items()}).sort_index() if best else pd.Series(dtype=float)
    return out


def _value_before(s: pd.Series, t) -> float | None:
    s = s[s.index <= t]
    return float(s.iloc[-1]) if len(s) else None


def macro_indicator(snap: pd.Series, how: tuple, max_age: int, as_of: pd.Timestamp) -> float:
    if snap is None or snap.empty:
        return np.nan
    last_t = snap.index[-1]
    if (pd.Timestamp(as_of).tz_localize("UTC") - pd.Timestamp(last_t)).days > max_age:
        return np.nan                                                 # veraltet -> fehlend, nie Default
    kind, arg = how
    v = float(snap.iloc[-1])
    if kind == "level":
        return v
    if kind == "neg_level":
        return -v
    if kind in ("chg_days", "neg_chg_days"):
        prev = _value_before(snap, last_t - pd.Timedelta(days=arg))
        if not prev:
            return np.nan
        c = v / prev - 1.0
        return -c if kind == "neg_chg_days" else c
    if kind == "accel":                                               # 3-Monats-Rate (annualisiert) minus Jahresrate
        p3 = _value_before(snap, last_t - pd.Timedelta(days=91))
        p12 = _value_before(snap, last_t - pd.Timedelta(days=365))
        if not p3 or not p12:
            return np.nan
        return ((v / p3) ** 4 - 1.0) - (v / p12 - 1.0)
    return np.nan


def macro_frame(archive_obs: dict[str, list], dates: list[pd.Timestamp]) -> pd.DataFrame:
    cols = {}
    snaps: dict = {}
    for name, (src, metric, how, max_age) in MACRO.items():
        key = (src, metric)
        if key not in snaps:
            snaps[key] = pit_snapshots(archive_obs.get(src, []), metric, dates)
        cols[name] = [macro_indicator(snaps[key][d], how, max_age, d) for d in dates]
    return pd.DataFrame(cols, index=pd.DatetimeIndex(dates))


# ── Markt ────────────────────────────────────────────────────────────────────

def market_frame(px: pd.DataFrame) -> pd.DataFrame:
    """px: Tagesschlusskurse (Spalten = Symbole). Alles nur bis Close t."""
    def chg(sym, n=63):
        return px[sym] / px[sym].shift(n) - 1.0 if sym in px else pd.Series(np.nan, index=px.index)
    m = pd.DataFrame(index=px.index)
    spy = px["SPY"]
    m["spy_trend_200"] = spy / spy.rolling(200, min_periods=180).mean() - 1.0
    m["spy_rv20"] = spy.pct_change(fill_method=None).rolling(20, min_periods=18).std() * math.sqrt(252)
    m["vix"] = px.get("^VIX")
    m["vix_term"] = px["^VIX"] / px["^VIX3M"] if "^VIX3M" in px else np.nan
    m["tnx"] = px.get("^TNX")
    m["tnx_63d"] = px["^TNX"] - px["^TNX"].shift(63) if "^TNX" in px else np.nan
    m["curve"] = px["^TNX"] - px["^IRX"] if "^IRX" in px else np.nan
    m["hyg_lqd_63d"] = (px["HYG"] / px["LQD"]).pct_change(63, fill_method=None) if "HYG" in px and "LQD" in px else np.nan
    m["xly_xlp_63d"] = chg("XLY") - chg("XLP")
    m["xli_spy_63d"] = chg("XLI") - chg("SPY")
    m["cyc_def_63d"] = pd.concat([chg(s) for s in CYCLICALS], axis=1).mean(axis=1) - \
        pd.concat([chg(s) for s in DEFENSIVES], axis=1).mean(axis=1)
    m["copper_63d"] = chg("HG=F")
    m["copper_gold_63d"] = (px["HG=F"] / px["GC=F"]).pct_change(63, fill_method=None) if "HG=F" in px and "GC=F" in px else np.nan
    return m.replace([np.inf, -np.inf], np.nan)


def breadth_frame(panel: pd.DataFrame | None) -> pd.DataFrame:
    """Marktbreite aus den wöchentlichen Schlusskursen des Feature-Stores."""
    if panel is None or panel.empty or "close" not in panel:
        return pd.DataFrame()
    p = panel[["date", "ticker", "close"]].sort_values(["ticker", "date"])
    g = p.groupby("ticker")["close"]
    p["up13"] = (p["close"] / g.shift(13) - 1.0) > 0
    p["near_high"] = p["close"] >= 0.95 * g.transform(lambda s: s.rolling(52, min_periods=40).max())
    valid13 = g.shift(13).notna()
    b = pd.DataFrame({"breadth_13w": p[valid13].groupby("date")["up13"].mean(),
                      "breadth_near_high": p.groupby("date")["near_high"].mean()})
    return b


# ── Zustand ──────────────────────────────────────────────────────────────────

def expanding_z(ind: pd.DataFrame, min_hist: int = WP["zscore_min_history_weeks"]) -> pd.DataFrame:
    """z-Score je Indikator nur mit Werten VOR dem Stichtag (kein Look-ahead)."""
    mu = ind.expanding(min_periods=min_hist).mean().shift(1)
    sd = ind.expanding(min_periods=min_hist).std().shift(1)
    return ((ind - mu) / sd).clip(-4, 4)


def world_states(z: pd.DataFrame, thr: float = WP["state_threshold"]) -> pd.DataFrame:
    rows = {}
    for d, zr in z.iterrows():
        r = {}
        unc = []
        for dim, inds in DIMENSIONS.items():
            if not inds:
                r[f"{dim}_score"] = np.nan
                r[f"{dim}_state"] = "unavailable"
                continue
            vals = [zr.get(i) for i in inds if i in zr and pd.notna(zr.get(i))]
            cov = len(vals) / len(inds)
            if not vals:
                r[f"{dim}_score"] = np.nan
                r[f"{dim}_state"] = "no_data"
                continue
            sc = float(np.mean(vals))
            r[f"{dim}_score"] = sc
            r[f"{dim}_state"] = "high" if sc > thr else "low" if sc < -thr else "neutral"
            margin = min(1.0, abs(abs(sc) - thr) / thr)
            r[f"{dim}_uncertainty"] = round(1.0 - cov * margin, 3)
            unc.append(r[f"{dim}_uncertainty"])
        r["uncertainty"] = float(np.mean(unc)) if unc else np.nan
        r["n_dims_available"] = len(unc)
        rows[d] = r
    return pd.DataFrame.from_dict(rows, orient="index")


def build_world(px: pd.DataFrame, archive_obs: dict, panel: pd.DataFrame | None,
                dates: list[pd.Timestamp] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """-> (Indikatoren roh je Stichtag, Zustände je Stichtag). Stichtage =
    letzter Handelstag je Woche."""
    mk = market_frame(px)
    if dates is None:
        s = pd.Series(mk.index, index=mk.index)
        dates = list(s.groupby(s.index.to_period("W-FRI")).max())
    mk = mk.reindex(dates)
    mac = macro_frame(archive_obs, dates)
    br = breadth_frame(panel).reindex(dates) if panel is not None else pd.DataFrame(index=dates)
    ind = pd.concat([mk, mac, br], axis=1)
    z = expanding_z(ind)
    return ind, world_states(z).join(z.add_prefix("z_"))


def state_hash(row: dict) -> str:
    return hashlib.sha256(json.dumps(row, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ── Validierung ──────────────────────────────────────────────────────────────

def targets(px: pd.DataFrame, dates: list[pd.Timestamp], panel: pd.DataFrame | None = None) -> pd.DataFrame:
    spy = px["SPY"]
    idx = spy.index
    pos = {d: idx.get_indexer([d])[0] for d in dates}
    rows = {}
    ret = spy.pct_change(fill_method=None)
    vix, trend = px.get("^VIX"), spy / spy.rolling(200, min_periods=180).mean() - 1.0

    def fwd(sym, i, h):
        if sym not in px or i + h >= len(idx):
            return np.nan
        return px[sym].iloc[i + h] / px[sym].iloc[i] - 1.0
    for d in dates:
        i = pos[d]
        if i < 0:
            continue
        r = {}
        for name, spec in WP["targets"].items():
            h = spec["horizon"]
            if i + h >= len(idx):
                r[name] = np.nan
                continue
            r[f"end_{name}"] = idx[i + h]
            if name in ("dd60", "dd60_bin"):
                dd = float(spy.iloc[i + 1:i + h + 1].min() / spy.iloc[i] - 1.0)
                r[name] = dd if name == "dd60" else float(dd < spec["threshold"])
            elif name == "rv20":
                r[name] = float(np.log(ret.iloc[i + 1:i + h + 1].std() * math.sqrt(252)))
            elif name == "rot60":
                r[name] = np.nanmean([fwd(s, i, h) for s in CYCLICALS]) - np.nanmean([fwd(s, i, h) for s in DEFENSIVES])
            elif name == "sb60":
                r[name] = fwd("SPY", i, h) - fwd("IEF", i, h)
            elif name == "regime_flip20":
                lab = lambda j: (bool(vix.iloc[j] >= 20) if vix is not None else False, bool(trend.iloc[j] > 0))  # noqa: E731
                now = lab(i)
                r[name] = float(any(lab(j) != now for j in range(i + 1, i + h + 1)))
        rows[d] = r
    t = pd.DataFrame.from_dict(rows, orient="index")
    if panel is not None and not panel.empty:
        for name, feat in (("ic_mom20", "mom_12_1"), ("ic_rev20", "rev_1m")):
            ic = {}
            for d, g in panel.groupby("date"):
                g = g[[feat, "fwd_xs_20"]].dropna()
                if len(g) >= 50:
                    ic[d] = g[feat].rank().corr(g["fwd_xs_20"].rank())
            t[name] = pd.Series(ic).reindex(t.index)
            le = panel.groupby("date")["label_end_20"].max()
            t[f"end_{name}"] = le.reindex(t.index)
    return t


def _fit_predict(kind: str, Xtr, ytr, Xte):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "binary":
        if len(np.unique(ytr)) < 2:
            return np.full(len(Xte), float(np.mean(ytr)))
        m = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                          LogisticRegression(C=WP["logistic_c"], max_iter=2000))
        m.fit(Xtr, ytr.astype(int))
        return m.predict_proba(Xte)[:, 1]
    m = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                      Ridge(alpha=WP["ridge_alpha"]))
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def walk_forward_compare(feats: dict[str, pd.DataFrame], tg: pd.DataFrame, locked_from: pd.Timestamp) -> dict:
    """Je Ziel und Featureset OOS-Prognosen (jahresweise, Training nur mit
    Zielen, deren Horizont vor dem Testjahr endet). -> Verluste je Stichtag."""
    out = {}
    last_year = (locked_from - pd.Timedelta(days=1)).year
    for name, spec in WP["targets"].items():
        if name not in tg:
            continue
        kind = spec["kind"]
        preds = {k: [] for k in list(feats) + ["climatology"]}
        for y in range(WP["first_test_year"], last_year + 1):
            ts = pd.Timestamp(year=y, month=1, day=1)
            te = min(pd.Timestamp(year=y + 1, month=1, day=1), locked_from)
            endc = f"end_{name}"
            tr = tg[(tg.index < ts) & tg[name].notna() & (tg[endc] < ts)]
            te_ = tg[(tg.index >= ts) & (tg.index < te) & tg[name].notna() & (tg[endc] < locked_from)]
            if len(tr) < 100 or te_.empty:
                continue
            preds["climatology"].append(pd.Series(tr[name].mean(), index=te_.index))
            for k, F in feats.items():
                Xtr, Xte = F.reindex(tr.index), F.reindex(te_.index)
                cols = [c for c in F.columns if Xtr[c].notna().any()]
                preds[k].append(pd.Series(_fit_predict(kind, Xtr[cols].to_numpy(float), tr[name].to_numpy(float),
                                                       Xte[cols].to_numpy(float)), index=te_.index))
        if not preds["climatology"]:
            continue
        y_true = tg[name]
        res = {"kind": kind}
        loss = {}
        for k, parts in preds.items():
            p = pd.concat(parts)
            yt = y_true.reindex(p.index)
            loss[k] = (p - yt) ** 2
            res[k] = {"mse": float(loss[k].mean()), "n": int(len(p))}
            if kind == "binary":
                try:
                    from sklearn.metrics import roc_auc_score
                    res[k]["auc"] = float(roc_auc_score(yt, p)) if yt.nunique() == 2 else None
                except ValueError:
                    res[k]["auc"] = None
        clim = res["climatology"]["mse"]
        for k in feats:
            res[k]["skill_vs_climatology"] = float(1 - res[k]["mse"] / clim) if clim > 0 else None
        out[name] = {"summary": res, "_loss": loss}
    return out


def block_bootstrap_gain(loss_base: pd.Series, loss_new: pd.Series, n: int, seed: int, alpha: float) -> dict:
    """Verlustdifferenz (Basis − Neu) > 0 heißt: Neu besser. Monats-Blöcke."""
    d = (loss_base - loss_new).dropna()
    m = d.groupby(pd.DatetimeIndex(d.index).to_period("M")).mean()
    if len(m) < 12:
        return {"n_months": int(len(m))}
    rng = np.random.default_rng(seed)
    arr = m.to_numpy()
    boots = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n)]
    lo, hi = np.quantile(boots, [alpha, 1 - alpha])
    rel = float(d.mean() / loss_base.mean()) if loss_base.mean() > 0 else None
    return {"mean_gain": float(arr.mean()), "ci": [float(lo), float(hi)], "relative_mse_reduction": rel,
            "n_months": int(len(m)), "significantly_better": bool(lo > 0), "significantly_worse": bool(hi < 0)}


def validate(ind_states: pd.DataFrame, baseline: pd.DataFrame, tg: pd.DataFrame,
             locked_from: pd.Timestamp) -> dict:
    wm_cols = [c for c in ind_states.columns if c.endswith("_score")] + ["uncertainty"]
    wm = ind_states[wm_cols]
    feats = {"baseline": baseline, "world_model": wm, "combined": baseline.join(wm, rsuffix="_wm")}
    wf = walk_forward_compare(feats, tg, locked_from)
    n_t = max(len(wf), 1)
    alpha = WP["alpha_one_sided"] / n_t
    results, better, worse = {}, [], []
    for name, r in wf.items():
        L = r["_loss"]
        prim = block_bootstrap_gain(L["baseline"], L["world_model"], WP["bootstrap_n"], WP["bootstrap_seed"], alpha)
        sec = block_bootstrap_gain(L["baseline"], L["combined"], WP["bootstrap_n"], WP["bootstrap_seed"], alpha)
        results[name] = {**r["summary"], "wm_vs_baseline": prim, "combined_vs_baseline": sec,
                         "desc": WP["targets"][name]["desc"]}
        if prim.get("significantly_better"):
            better.append(name)
        if prim.get("significantly_worse"):
            worse.append(name)
    dec = WP["decision"]
    if len(better) >= dec["keep_min_targets_better"] and len(worse) <= dec["keep_max_targets_worse"]:
        verdict = "KEEP"
    elif better:
        verdict = "MODIFY"
    else:
        verdict = "REJECT"
    return {"targets": results, "better": better, "worse": worse, "verdict": verdict,
            "alpha_per_target": alpha, "rule": dec}


# ── Daten (nur CI, Netzwerk) + Lauf ──────────────────────────────────────────

def fetch_market(start: str = "2012-01-01") -> pd.DataFrame:
    import yfinance as yf
    data = yf.download(list(MARKET_SYMBOLS), start=start, auto_adjust=True, progress=False, threads=True)
    px = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
    px = px.dropna(subset=["SPY"])
    now = datetime.now(timezone.utc)
    if now.hour < 21:                                            # unfertiger Tagesbalken nie als Close_t
        px = px[px.index.date < now.date()]
    return px


def load_archive(root: str = "outputs/external_data") -> dict:
    from modules.external.archive import ExternalArchive
    arch = ExternalArchive(root)
    out = {}
    for src in {v[0] for v in MACRO.values()}:
        try:
            out[src] = arch.load(src)
        except (OSError, ValueError, KeyError) as e:
            log.warning(f"world_model: Archivquelle {src} nicht lesbar ({e}) -> Indikatoren fehlen")
            out[src] = []
    return out


def render_md(rep: dict) -> str:
    cur = rep["current"]
    L = [f"# World Model ({rep['version']}) – {rep['generated']}", "",
         f"Stichtag {cur.get('date')} · Unsicherheit {cur.get('uncertainty')} · Dimensionen mit Daten "
         f"{cur.get('n_dims_available')}/{len(DIMENSIONS)} · Hash {cur.get('state_hash')}", "",
         "| Dimension | Zustand | Score | Unsicherheit | Vorwoche |", "|---|---|---|---|---|"]
    prev = rep.get("previous") or {}
    for dim in DIMENSIONS:
        L.append(f"| {dim} | {cur.get(f'{dim}_state')} | {cur.get(f'{dim}_score')} | "
                 f"{cur.get(f'{dim}_uncertainty')} | {prev.get(f'{dim}_state')} |")
    for dim, why in UNAVAILABLE_REASON.items():
        L.append(f"\n- {dim}: nicht verfügbar – {why}")
    v = rep.get("validation") or {}
    if v:
        L += ["", f"## Validierung gegen bestehende Regime-Engine: **{v['verdict']}**", "",
              f"Signifikant besser bei: {v['better'] or '–'} · signifikant schlechter bei: {v['worse'] or '–'} "
              f"(einseitig, Bonferroni-α je Ziel {round(v['alpha_per_target'], 4)})", "",
              "| Ziel | Art | Skill Basis | Skill WM | Skill kombiniert | WM−Basis MSE-Reduktion | CI | kombiniert−Basis | AUC Basis/WM |",
              "|---|---|---|---|---|---|---|---|---|"]
        for name, r in v["targets"].items():
            p, s = r["wm_vs_baseline"], r["combined_vs_baseline"]
            L.append(f"| {name} ({r['desc']}) | {r['kind']} | {_f(r['baseline'].get('skill_vs_climatology'))} | "
                     f"{_f(r['world_model'].get('skill_vs_climatology'))} | {_f(r['combined'].get('skill_vs_climatology'))} | "
                     f"{_f(p.get('relative_mse_reduction'))} | {[_f(x, 6) for x in p.get('ci', [])]} | "
                     f"{_f(s.get('relative_mse_reduction'))} | {_f(r['baseline'].get('auc'))}/{_f(r['world_model'].get('auc'))} |")
    return "\n".join(L) + "\n"


def _f(x, nd=3):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), nd)


def run(panel: pd.DataFrame | None = None, validate_now: bool = True) -> dict:
    import os
    import pickle
    if panel is None:
        cache = os.environ.get("ML_PANEL_CACHE")
        if cache and Path(cache).exists():
            with open(cache, "rb") as fh:
                panel = pickle.load(fh)  # noqa: S301 – eigene, im selben Job erzeugte Datei
        else:
            from modules import ml_research as ml
            panel = ml.build_research_panel("weekly")          # Breite + Signal-IC-Ziele
    px = fetch_market()
    arch = load_archive()
    ind, st = build_world(px, arch, panel)
    st = st.dropna(how="all", subset=[c for c in st.columns if c.endswith("_score")])
    from modules import ml_research as ml
    locked = ml.LOCKED_FROM
    rep = {"version": WP["version"], "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "no_llm_ground_truth": True}
    last = st.index.max()
    cur = {k: (None if isinstance(v, float) and not math.isfinite(v) else (round(v, 4) if isinstance(v, float) else v))
           for k, v in st.loc[last].items() if not k.startswith("z_")}
    cur["date"] = str(last.date())
    cur["state_hash"] = state_hash(cur)
    rep["current"] = cur
    if len(st) > 1:
        pv = st.iloc[-2]
        rep["previous"] = {k: v for k, v in pv.items() if k.endswith("_state")}
        rep["changes"] = [f"{d}: {pv.get(f'{d}_state')} -> {cur.get(f'{d}_state')}" for d in DIMENSIONS
                          if pv.get(f"{d}_state") != cur.get(f"{d}_state")]
    rep["indicator_coverage"] = {c: round(float(ind[c].notna().mean()), 3) for c in ind.columns}
    if validate_now:
        tg = targets(px, list(st.index), panel)
        base_cols = WP["baseline_features"]
        base = pd.DataFrame(index=st.index)
        mk = market_frame(px).reindex(st.index)
        base["vix"] = mk["vix"]
        base["vix_chg_21"] = (px["^VIX"] - px["^VIX"].shift(21)).reindex(st.index)
        base["spy_trend_200"] = mk["spy_trend_200"]
        base["spy_mom_63"] = (px["SPY"] / px["SPY"].shift(63) - 1).reindex(st.index)
        base["tnx"] = mk["tnx"]
        base["curve_10y_3m"] = mk["curve"]
        mac = macro_frame(arch, list(st.index))
        base["cpi_yoy"], base["fed_assets_13w_chg"] = mac["cpi_yoy"], mac["fed_assets_13w"]
        base["usd_63d_chg"], base["wti_63d_chg"] = mac["usd_63d"], mac["wti_63d"]
        rep["validation"] = validate(st, base[base_cols], tg, locked)
        # Locked-Holdout: nur Bericht (einmal je Version, Zählung im Log)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    have = set()
    if STATE_LOG.exists():
        for line in STATE_LOG.read_text().splitlines():
            try:
                have.add(json.loads(line)["date"])
            except (json.JSONDecodeError, KeyError):
                log.warning("world_model: defekte Zeile im State-Log übersprungen")
    with open(STATE_LOG, "a", encoding="utf-8") as fh:            # append-only, versioniert
        for d, row in st.iterrows():
            ds = str(d.date())
            if ds in have:
                continue
            rec = {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in row.items()
                   if not k.startswith("z_")}
            rec.update(date=ds, version=WP["version"])
            rec["state_hash"] = state_hash(rec)
            fh.write(json.dumps(rec, default=str) + "\n")
    OUT_JSON.write_text(json.dumps(rep, indent=1, default=str, ensure_ascii=False))
    OUT_MD.write_text(render_md(rep), encoding="utf-8")
    return rep


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    print(render_md(run()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

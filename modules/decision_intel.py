"""
modules/decision_intel.py – Decision Intelligence (NUR SHADOW, keine Orders)

Der beste Einzeltrade ist nicht automatisch der beste Portfolio-Trade.
Je Kandidat: erwartete Rendite, kalibrierte Wahrscheinlichkeit, Downside,
Expected Shortfall, Korrelation zum Kandidaten-Universum, Faktor-Exposures,
Sektor-Konzentration, Liquidität, Tail-Risiko-Beitrag – und ein Portfolio
Utility Score.

Robust statt Mittelwert/Kovarianz-Optimierung: Kovarianz mit Ledoit-Wolf-
Shrinkage aus 104 Wochen, erwartete Renditen nur als OOS-kalibrierte Rang-
Abbildung (keine historischen Mittelwerte je Titel), gierige Auswahl mit
Sektor-Obergrenze und Strafterm für Grenzrisiko. Validierung (next_protocol,
decision_intelligence.keep_rule): Walk-Forward gegen das plain Top-Dezil.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from modules import meta_learning as meta
from modules import ml_research as ml

log = logging.getLogger(__name__)

DI = meta.yaml.safe_load((ml.PROTOCOL_PATH.parent / "next_protocol.yaml").read_text(encoding="utf-8"))["decision_intelligence"]


def weekly_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Wöchentliche Renditen je Ticker aus den Schlusskursen des Feature-Stores."""
    px = panel.pivot_table(index="date", columns="ticker", values="close").sort_index()
    return px.pct_change(fill_method=None)


def shrunk_cov(rets: pd.DataFrame) -> pd.DataFrame | None:
    from sklearn.covariance import LedoitWolf
    r = rets.dropna(axis=1, thresh=int(len(rets) * 0.8)).fillna(0.0)
    if r.shape[0] < 26 or r.shape[1] < 2:
        return None
    lw = LedoitWolf().fit(r.to_numpy())
    return pd.DataFrame(lw.covariance_, index=r.columns, columns=r.columns)


def greedy_select(cands: pd.DataFrame, cov: pd.DataFrame | None, k: int,
                  max_sector_share: float = DI["max_sector_share"], lam: float = DI["risk_aversion"]) -> list[str]:
    """cands: index=ticker, Spalten exp (erwartete Netto-Relativrendite), sector.
    Utility_i = exp_i − λ · (Grenzvarianz des gleichgewichteten Portfolios)."""
    chosen: list[str] = []
    sec_count: dict = {}
    cap = max(1, int(math.ceil(k * max_sector_share)))
    pool = cands.sort_values("exp", ascending=False)
    while len(chosen) < k and len(pool):
        best, best_u = None, -np.inf
        for t, r in pool.iterrows():
            if sec_count.get(r["sector"], 0) >= cap:
                continue
            u = r["exp"]
            if cov is not None and t in cov.index and chosen:
                c_in = [c for c in chosen if c in cov.index]
                if c_in:
                    u -= lam * float(cov.loc[t, c_in].mean()) * 52 / 4      # ~ 20-Tage-Skala
            if u > best_u:
                best, best_u = t, u
        if best is None:
            break
        chosen.append(best)
        sec_count[pool.loc[best, "sector"]] = sec_count.get(pool.loc[best, "sector"], 0) + 1
        pool = pool.drop(index=best)
    return chosen


def walk_forward_diversified(res: pd.DataFrame, panel: pd.DataFrame, score: str, exp_col: str | None = None) -> dict:
    """Je Stichtag: Pool = Top 20 % nach Score, Auswahl k = Top-Dezil-Größe per
    greedy_select (Kovarianz nur aus Wochen VOR dem Stichtag)."""
    rets = weekly_returns(panel)
    rows_plain, rows_div = [], []
    for d, g in res.groupby("date"):
        g = g[g[score].notna() & g[meta.PROB_TARGET].notna()]
        if len(g) < ml.MIN_CROSS_SECTION:
            continue
        k = max(1, int(round(len(g) * ml.TOP_Q)))
        rk = g[score].rank(pct=True)
        pool = g[rk >= 0.8].set_index("ticker")
        pool = pool.assign(exp=pool[exp_col] if exp_col and exp_col in pool else pool[score].rank(pct=True),
                           sector=pool["sector"].fillna("unknown"))
        hist = rets[rets.index < d].tail(DI["lookback_weeks"])
        cov = shrunk_cov(hist[[t for t in pool.index if t in hist.columns]])
        div = greedy_select(pool, cov, k)
        plain = list(g.nlargest(k, score)["ticker"])
        gi = g.set_index("ticker")
        for lst, rows in ((plain, rows_plain), (div, rows_div)):
            for t in lst:
                rows.append({"date": d, "ticker": t, "ret": gi.at[t, meta.PROB_TARGET], "sector": gi.at[t, "sector"],
                             "mfe": gi.at[t, "mfe_20"], "mae": gi.at[t, "mae_20"], "in_top_quintile": False,
                             "is_strong": False, "n_strong": 1, "vix": gi.at[t, "vix"],
                             "spy_trend_200": gi.at[t, "spy_trend_200"], "year": pd.Timestamp(d).year})
    pa, pb = pd.DataFrame(rows_plain), pd.DataFrame(rows_div)
    out = {}
    for name, p in (("top_decile", pa), ("diversified", pb)):
        dev = p[p["date"] < ml.LOCKED_FROM]
        m = meta.portfolio_metrics(dev)
        mon = meta.monthly_series(dev)
        m["es5_monthly"] = ml._r(mon[mon <= mon.quantile(0.05)].mean(), 5) if len(mon) >= 20 else None
        m["sector_hhi"] = ml._r(float(dev.groupby(["date", "sector"]).size().groupby("date").apply(
            lambda s: ((s / s.sum()) ** 2).sum()).mean()), 4) if len(dev) else None
        out[name] = m
    a, b = out["top_decile"], out["diversified"]
    keep = (b.get("sharpe") or -9) >= (a.get("sharpe") or 0) and (
        (b.get("es5_monthly") or -9) > (a.get("es5_monthly") or -9) or (b.get("max_dd") or -9) > (a.get("max_dd") or -9))
    out["verdict"] = "KEEP" if keep else ("MODIFY" if (b.get("max_dd") or -9) > (a.get("max_dd") or -9) else "REJECT")
    out["_positions"] = {"top_decile": pa, "diversified": pb}
    return out


def candidate_profile(tickers: list[str], panel: pd.DataFrame, info: dict[str, dict]) -> dict:
    """Gegenwart: Portfolio-Kennzahlen je Kandidat (info: exp, prob, downside
    aus Karten/Kalibrierung)."""
    if not tickers:
        return {}
    rets = weekly_returns(panel).tail(DI["lookback_weeks"])
    cols = [t for t in tickers if t in rets.columns]
    cov = shrunk_cov(rets[cols]) if len(cols) >= 2 else None
    snap = panel[panel["date"] == panel["date"].max()].set_index("ticker")
    four = (1 + rets).rolling(4).apply(np.prod, raw=True) - 1
    out = {}
    n = len(cols)
    w = np.full(n, 1.0 / n) if n else np.array([])
    port_var = float(w @ cov.to_numpy() @ w) if cov is not None else None
    sectors = snap["sector"].reindex(tickers).fillna("unknown")
    for t in tickers:
        r4 = four[t].dropna() if t in four else pd.Series(dtype=float)
        es = float(r4[r4 <= r4.quantile(0.05)].mean()) if len(r4) >= 20 else None
        corr = None
        mrc = None
        if cov is not None and t in cov.index:
            sd = np.sqrt(np.diag(cov.to_numpy()))
            cm = cov.to_numpy() / np.outer(sd, sd)
            i = list(cov.index).index(t)
            corr = float(np.delete(cm[i], i).mean()) if n > 1 else None
            mrc = float((cov.to_numpy() @ w)[i] * w[i] / port_var) if port_var else None
        e = info.get(t, {})
        util = (e.get("exp") or 0) - DI["risk_aversion"] * (mrc or 0) * (math.sqrt(port_var * 52 / 4) if port_var else 0)
        out[t] = {"expected_return": e.get("exp"), "calibrated_probability": e.get("prob"), "downside": e.get("downside"),
                  "expected_shortfall_4w_5pct": ml._r(es, 4), "drawdown_estimate": e.get("drawdown"),
                  "asymmetry": e.get("asymmetry"), "avg_corr_to_candidates": ml._r(corr, 3),
                  "factor_exposure": {f: ml._r(snap.at[t, f], 3) if t in snap.index else None
                                      for f in ("beta_126", "mom_12_1", "vol_60", "log_dollar_vol")},
                  "sector": sectors.get(t), "sector_share_in_candidates": ml._r(float((sectors == sectors.get(t)).mean()), 3),
                  "liquidity_rank": ml._r(snap.at[t, "log_dollar_vol"], 3) if t in snap.index else None,
                  "tail_risk_contribution": ml._r(mrc, 3), "portfolio_utility": ml._r(util, 5)}
    return out

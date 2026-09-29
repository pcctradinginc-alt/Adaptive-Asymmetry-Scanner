"""
modules/price_event_study.py – historische, PIT-saubere Volumen-Event-Studie
mit jahresweisem Walk-Forward (Präregistrierung:
docs/research/PREREG_price_event_study_2026-09-29.md).

    python -m modules.price_event_study      (CI: .github/workflows/research.yml)

Zeitliche Regeln (Tests: tests/test_price_event_study.py):
  * Features nur aus Daten bis einschließlich Close_t (Event-Tag).
  * Entry Open_{t+1}, Exit Close_{t+h}; Rendite minus SPY im selben Fenster.
  * Terzilgrenzen/Richtungen nur aus Trainingsjahren (< Testjahr).
Keine Produktionswirkung; Ergebnisse -> outputs/research/price_event_study.{json,md}.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

OUT_DIR = Path("outputs/research")
HORIZONS = (1, 5, 20, 60)
RELVOL_MIN = 2.0
PRICE_MIN = 5.0
DOLLAR_VOL_MIN = 20e6
DEDUP_DAYS = 5
COST_BASE_PER_SIDE = 0.0010
COST_STRESS_PER_SIDE = 0.0025
TEST_START_YEAR = 2019
TRAIN_START_YEAR = 2015
DECISION_H = 20


# ── Events ───────────────────────────────────────────────────────────────────

def build_events(frames: dict[str, pd.DataFrame], spy: pd.DataFrame,
                 vix: pd.Series | None = None, horizons=HORIZONS) -> pd.DataFrame:
    """frames: ticker -> DataFrame[Open, Close, Volume] (DatetimeIndex, bereinigt).
    Alle Frames werden auf den SPY-Kalender gebracht (fehlende Tage = NaN)."""
    cal = spy.index
    spy_c, spy_o = spy["Close"], spy["Open"]
    spy_ret = spy_c.pct_change()
    trend_up = spy_c > spy_c.rolling(200).mean()
    vix_s = vix.reindex(cal).ffill() if vix is not None else pd.Series(np.nan, index=cal)
    out = []
    for ticker, df in frames.items():
        df = df.reindex(cal)
        c, o, v = df["Close"], df["Open"], df["Volume"]
        if c.notna().sum() < 300:
            continue
        ret = c.pct_change()
        avg_v = v.shift(1).rolling(20, min_periods=20).mean()
        dollar_v = (c * v).shift(1).rolling(20, min_periods=20).mean()
        relvol = v / avg_v
        sigma20 = ret.shift(1).rolling(20, min_periods=20).std()
        ev_ret_adj = ret - spy_ret
        r2 = c / c.shift(2) - 1.0
        z2 = r2 / (sigma20 * math.sqrt(2))
        rs35 = (c / c.shift(35) - 1.0) - (spy_c / spy_c.shift(35) - 1.0)
        mask = (relvol >= RELVOL_MIN) & (c >= PRICE_MIN) & (dollar_v >= DOLLAR_VOL_MIN) \
            & sigma20.notna() & ev_ret_adj.notna()
        idx = np.flatnonzero(mask.to_numpy())
        last = -10**9
        keep = []
        for i in idx:                       # Cluster-Entdopplung: max. 1 Event je 5 Tage
            if i - last >= DEDUP_DAYS:
                keep.append(i)
                last = i
        if not keep:
            continue
        rows = {
            "date": cal[keep], "ticker": ticker,
            "ev_ret_adj": ev_ret_adj.iloc[keep].to_numpy(),
            "z2": z2.iloc[keep].to_numpy(), "rs35": rs35.iloc[keep].to_numpy(),
            "relvol": relvol.iloc[keep].to_numpy(), "sigma20": sigma20.iloc[keep].to_numpy(),
            "vix": vix_s.iloc[keep].to_numpy(), "trend_up": trend_up.iloc[keep].to_numpy(),
        }
        entry = o.shift(-1)
        spy_entry = spy_o.shift(-1)
        for h in horizons:
            fwd = c.shift(-h) / entry - 1.0
            spy_fwd = spy_c.shift(-h) / spy_entry - 1.0
            rows[f"fwd_{h}"] = (fwd - spy_fwd).iloc[keep].to_numpy()
            ex = pd.Series(cal, index=cal).shift(-h)
            rows[f"exit_{h}"] = ex.iloc[keep].to_numpy()
        out.append(pd.DataFrame(rows))
    if not out:
        return pd.DataFrame()
    ev = pd.concat(out, ignore_index=True)
    ev["year"] = pd.to_datetime(ev["date"]).dt.year
    return ev.sort_values("date").reset_index(drop=True)


# ── Statistik ────────────────────────────────────────────────────────────────

def trade_metrics(trades: pd.DataFrame, cost_per_side: float) -> dict:
    """trades: date, signed (Brutto-Richtungsrendite). Kennzahlen netto."""
    t = trades.dropna(subset=["signed"])
    if len(t) == 0:
        return {"n": 0}
    net = t["signed"] - 2 * cost_per_side
    cohort = net.groupby(pd.to_datetime(t["date"])).mean()
    monthly = cohort.groupby(cohort.index.to_period("M")).mean()
    m = monthly.to_numpy()
    sd = m.std(ddof=1) if len(m) > 1 else float("nan")
    downside = np.sqrt(np.mean(np.minimum(m, 0) ** 2)) if len(m) else float("nan")
    curve = np.cumsum(m)
    dd = float(np.min(curve - np.maximum.accumulate(curve))) if len(curve) else float("nan")
    by_year = net.groupby(pd.to_datetime(t["date"]).dt.year).mean()
    return {
        "n": int(len(t)), "n_dates": int(cohort.size), "n_months": int(len(m)),
        "mean": _r(net.mean()), "median": _r(net.median()), "hit_rate": _r((net > 0).mean()),
        "t_months": _r(m.mean() / (sd / math.sqrt(len(m)))) if len(m) > 2 and sd > 0 else None,
        "sharpe_ann": _r(m.mean() / sd * math.sqrt(12)) if len(m) > 2 and sd > 0 else None,
        "sortino_ann": _r(m.mean() / downside * math.sqrt(12)) if downside and downside > 0 else None,
        "max_dd_monthly_cohorts": _r(dd),
        "years_positive_share": _r((by_year > 0).mean()) if len(by_year) else None,
        "by_year": {int(y): _r(v) for y, v in by_year.items()},
    }


def _r(x, nd=5):
    return round(float(x), nd) if x is not None and np.isfinite(x) else None


# ── Hypothesen (präregistriert) ──────────────────────────────────────────────

def _signed(ev: pd.DataFrame, h: int, sign: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"date": ev["date"], "signed": sign * ev[f"fwd_{h}"],
                         "vix": ev["vix"], "trend_up": ev["trend_up"]})


def hypothesis_trades(name: str, train: pd.DataFrame, test: pd.DataFrame, h: int) -> pd.DataFrame:
    """Trades der Hypothese im Testjahr; Parameter NUR aus `train`."""
    sgn = np.sign(test["ev_ret_adj"])
    if name == "H1_drift":
        return _signed(test, h, sgn)
    if name == "H1L_long_up":
        up = test[test["ev_ret_adj"] > 0]
        return _signed(up, h, pd.Series(1.0, index=up.index))
    if name == "H2_small_move_gate":          # |z2| <= Train-Terzil 1/3
        q = train["z2"].abs().quantile(1 / 3)
        sel = test[test["z2"].abs() <= q]
        return _signed(sel, h, np.sign(sel["ev_ret_adj"]))
    if name == "H2_large_move":               # Vergleich: oberes Terzil
        q = train["z2"].abs().quantile(2 / 3)
        sel = test[test["z2"].abs() >= q]
        return _signed(sel, h, np.sign(sel["ev_ret_adj"]))
    if name == "H3_up_rs_positive":
        sel = test[(test["ev_ret_adj"] > 0) & (test["rs35"] > 0)]
        return _signed(sel, h, pd.Series(1.0, index=sel.index))
    if name == "H3_up_rs_negative":
        sel = test[(test["ev_ret_adj"] > 0) & (test["rs35"] <= 0)]
        return _signed(sel, h, pd.Series(1.0, index=sel.index))
    if name == "H4_high_relvol":
        q = train["relvol"].quantile(2 / 3)
        sel = test[test["relvol"] >= q]
        return _signed(sel, h, np.sign(sel["ev_ret_adj"]))
    if name == "H4_low_relvol":
        q = train["relvol"].quantile(1 / 3)
        sel = test[test["relvol"] <= q]
        return _signed(sel, h, np.sign(sel["ev_ret_adj"]))
    raise ValueError(name)


HYPOTHESES = ("H1_drift", "H1L_long_up", "H2_small_move_gate", "H2_large_move",
              "H3_up_rs_positive", "H3_up_rs_negative", "H4_high_relvol", "H4_low_relvol")


def walk_forward(ev: pd.DataFrame, name: str, h: int) -> pd.DataFrame:
    """Testjahre ab TEST_START_YEAR; Training = alle Vorjahre ab TRAIN_START_YEAR,
    nur Events, deren Exit VOR dem Testjahr liegt (kein Outcome aus der Zukunft)."""
    parts = []
    for y in sorted(ev["year"].unique()):
        if y < TEST_START_YEAR:
            continue
        start = pd.Timestamp(f"{y}-01-01")
        train = ev[(ev["year"] >= TRAIN_START_YEAR) & (pd.to_datetime(ev[f"exit_{h}"]) < start)]
        test = ev[ev["year"] == y]
        if len(train) < 200 or len(test) == 0:
            continue
        parts.append(hypothesis_trades(name, train, test, h))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["date", "signed"])


def regime_split(trades: pd.DataFrame, cost: float) -> dict:
    out = {}
    if trades.empty:
        return out
    for label, m in (("vix_low", trades["vix"] < 20), ("vix_high", trades["vix"] >= 20),
                     ("trend_up", trades["trend_up"] == True),       # noqa: E712
                     ("trend_down", trades["trend_up"] == False)):   # noqa: E712
        sub = trades[m.fillna(False)]
        met = trade_metrics(sub, cost)
        out[label] = {"n": met.get("n"), "mean": met.get("mean"), "t_months": met.get("t_months")}
    return out


def decide(base: dict, stress: dict, regimes: dict) -> dict:
    reasons = []
    ok = True
    if not (base.get("mean") or 0) > 0 or (base.get("t_months") or 0) < 2.0:
        ok = False
        reasons.append(f"OOS-Mittel/t nicht ausreichend (mean={base.get('mean')}, t={base.get('t_months')})")
    if (base.get("years_positive_share") or 0) < 0.6:
        ok = False
        reasons.append(f"nur {base.get('years_positive_share')} der Jahre positiv")
    signs = [v.get("mean") for v in regimes.values() if v.get("n")]
    if not signs or any(s is None or s <= 0 for s in signs):
        ok = False
        reasons.append("nicht in allen Regimen positiv")
    if not (stress.get("mean") or 0) > 0:
        ok = False
        reasons.append("bei 25 bp/Seite nicht mehr positiv")
    return {"supported": ok, "reasons": reasons}


def evaluate(ev: pd.DataFrame) -> dict:
    res = {"n_events": int(len(ev)), "n_tickers": int(ev["ticker"].nunique()) if len(ev) else 0,
           "period": [str(pd.to_datetime(ev["date"]).min().date()), str(pd.to_datetime(ev["date"]).max().date())]
           if len(ev) else None, "hypotheses": {}}
    for name in HYPOTHESES:
        per_h = {}
        for h in HORIZONS:
            tr = walk_forward(ev, name, h)
            base = trade_metrics(tr, COST_BASE_PER_SIDE)
            stress = trade_metrics(tr, COST_STRESS_PER_SIDE)
            regimes = regime_split(tr, COST_BASE_PER_SIDE)
            per_h[f"h{h}"] = {"base_cost": base, "stress_cost": {k: stress.get(k) for k in ("mean", "t_months", "sharpe_ann")},
                              "regimes": regimes}
            if h == DECISION_H:
                per_h["decision"] = decide(base, stress, regimes)
        res["hypotheses"][name] = per_h
    return res


# ── Daten (nur CI, Netzwerk) ─────────────────────────────────────────────────

def fetch_frames(tickers: list[str], start: str = "2014-01-01") -> tuple[dict, pd.DataFrame, pd.Series]:
    import yfinance as yf
    frames: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), 80):
        chunk = tickers[i:i + 80]
        data = yf.download(chunk, start=start, auto_adjust=True, group_by="ticker",
                           progress=False, threads=True)
        for t in chunk:
            try:
                df = data[t][["Open", "Close", "Volume"]].dropna(how="all")
            except (KeyError, TypeError):
                log.warning(f"price_event_study: keine Daten für {t}")
                continue
            if len(df):
                frames[t] = df
    spy = yf.download("SPY", start=start, auto_adjust=True, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    vix = yf.download("^VIX", start=start, auto_adjust=False, progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    return frames, spy[["Open", "Close"]], vix["Close"]


def render_md(res: dict) -> str:
    lines = [f"# Volumen-Event-Studie (Walk-Forward, OOS ab {TEST_START_YEAR})", "",
             f"Events: {res['n_events']} · Ticker: {res['n_tickers']} · Zeitraum: {res['period']}",
             "Präregistrierung: docs/research/PREREG_price_event_study_2026-09-29.md",
             "Renditen marktbereinigt (minus SPY), netto 10 bp/Seite; Survivorship-Bias: heutige Indexliste.", "",
             "| Hypothese | h | n | Mittel | Median | Hit | t (Monate) | Sharpe | Jahre + | 25bp-Mittel | Entscheid |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, per in res["hypotheses"].items():
        for h in HORIZONS:
            b = per[f"h{h}"]["base_cost"]
            s = per[f"h{h}"]["stress_cost"]
            dec = per.get("decision", {}) if h == DECISION_H else {}
            verdict = ("gestützt" if dec.get("supported") else "nicht gestützt") if dec else ""
            lines.append(f"| {name} | {h} | {b.get('n')} | {b.get('mean')} | {b.get('median')} | {b.get('hit_rate')} | "
                         f"{b.get('t_months')} | {b.get('sharpe_ann')} | {b.get('years_positive_share')} | "
                         f"{s.get('mean')} | {verdict} |")
    lines += ["", "## Entscheidungen (h=20)", ""]
    for name, per in res["hypotheses"].items():
        d = per.get("decision", {})
        lines.append(f"- **{name}**: {'GESTÜTZT' if d.get('supported') else 'nicht gestützt'}"
                     + (f" – {'; '.join(d.get('reasons', []))}" if d.get("reasons") else ""))
    return "\n".join(lines) + "\n"


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    from modules.universe import get_universe
    tickers = sorted(set(get_universe()))
    frames, spy, vix = fetch_frames(tickers)
    log.info(f"price_event_study: {len(frames)}/{len(tickers)} Ticker geladen")
    ev = build_events(frames, spy, vix)
    res = evaluate(ev)
    res.update({"generated": datetime.utcnow().isoformat(timespec="seconds"),
                "tickers_requested": len(tickers), "tickers_loaded": len(frames),
                "prereg": "docs/research/PREREG_price_event_study_2026-09-29.md"})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "price_event_study.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    (OUT_DIR / "price_event_study.md").write_text(render_md(res), encoding="utf-8")
    print(render_md(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

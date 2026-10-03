"""
modules/surprise_engine.py – Expectation / Surprise Engine mit präregistrierter Walk-Forward-Studie.

Surprise = beobachtete Realität − Markterwartung, getrennt nach
  Fundamental Outcome   fund_surprise = (berichtetes EPS − Konsens-EPS) / max(|Konsens|, EPS_FLOOR), [−2, 2]
  Market Reaction       react_z = marktbereinigte Rendite im Reaktionsfenster / (σ20 · √Tage)
  Unpriced Surprise     unpriced = Perzentil(fund_surprise) − Perzentil(react_z), Perzentile aus TRAININGSJAHREN

Zeitliche Regeln (PIT, Tests: tests/test_surprise_engine.py):
  * Konsens-EPS ist die Erwartung VOR der Meldung; das Ist-EPS ist ab dem Meldezeitpunkt bekannt.
  * Meldung vor Börsenöffnung -> Reaktionstag = Meldetag; nach Börsenschluss -> nächster Handelstag;
    Uhrzeit unbekannt -> konservativ zweitägiges Fenster (Meldetag + Folgetag).
  * Entry = Open des Handelstags NACH dem Reaktionsfenster; Exit = Close nach h Handelstagen;
    Rendite minus SPY im selben Fenster. Lag-Test: Entry 5 Handelstage später.
  * Perzentile/Terzile nur aus Trainingsjahren (< Testjahr, Exit vor Testjahr).
Präregistrierung: docs/research/PREREG_surprise_engine_2026-10-03.md. Keine Produktionswirkung;
Ergebnisse -> outputs/research/surprise_study.{json,md} -> Research Memory.

Datenlücken (DATA_GAP, nie simuliert): historische implizite Vola / Implied Move je Meldung
(Optionsarchiv), Positionierung (Short Interest-Historie), Prediction Markets.

    python -m modules.surprise_engine      (CI: .github/workflows/research.yml)
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from modules.price_event_study import (COST_BASE_PER_SIDE, COST_STRESS_PER_SIDE, TEST_START_YEAR,
                                       TRAIN_START_YEAR, regime_split, trade_metrics)

log = logging.getLogger(__name__)

OUT_DIR = Path("outputs/research")
HORIZONS = (20, 60)
DECISION_H = 20
LAG_DAYS = 5
EPS_FLOOR = 0.05
PLACEBO_N = 100
SEED = 23
FDR_Q = 0.10
PREREG = "docs/research/PREREG_surprise_engine_2026-10-03.md"
DATA_GAPS = ["historische implizite Volatilität/Implied Move je Meldung (Optionsarchiv)",
             "Positionierung (Short-Interest-Historie, Dealer-Gamma historisch)",
             "Prediction Markets (keine kostenlose PIT-Historie)",
             "PIT-Sektorzuordnung (Sektor-Analyse nur mit heutiger Zuordnung, beschreibend)"]


# ── Kennzahlen je Meldung (auch für einzelne Kandidaten nutzbar) ─────────────
def fund_surprise(eps_actual, eps_estimate) -> float | None:
    try:
        a, e = float(eps_actual), float(eps_estimate)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(a) and math.isfinite(e)):
        return None
    return float(np.clip((a - e) / max(abs(e), EPS_FLOOR), -2.0, 2.0))


def reaction_window(ann_ts: pd.Timestamp, cal: pd.DatetimeIndex) -> tuple[int, int] | None:
    """(Index des letzten Closes VOR der Reaktion, Index des Reaktions-Closes) im Kalender."""
    ts = pd.Timestamp(ann_ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("America/New_York")
    day = pd.Timestamp(ts.date())
    pos = int(cal.searchsorted(day))                 # erster Handelstag >= Meldedatum
    if pos >= len(cal) or pos < 1:
        return None
    is_trading_day = cal[pos] == day
    minutes = ts.hour * 60 + ts.minute if ts.tzinfo is not None else None
    if minutes is None or minutes == 0:              # Uhrzeit unbekannt -> Meldetag + Folgetag
        end = pos + 1 if is_trading_day else pos
        start = pos - 1
    elif minutes < 9 * 60 + 30:                      # vor Börsenöffnung
        start, end = pos - 1, pos
    elif minutes >= 16 * 60 or not is_trading_day:   # nach Schluss / Wochenende
        start, end = (pos, pos + 1) if is_trading_day else (pos - 1, pos)
    else:                                            # während der Handelszeit
        start, end = pos - 1, pos
    if end >= len(cal) or start < 0:
        return None
    return start, end


def build_events(earn: pd.DataFrame, frames: dict[str, pd.DataFrame], spy: pd.DataFrame,
                 vix: pd.Series | None = None, horizons=HORIZONS, lag: int = LAG_DAYS) -> pd.DataFrame:
    """earn: ticker, ann_ts, eps_estimate, eps_actual. frames: ticker -> [Open, Close]."""
    cal = spy.index
    spy_c, spy_o = spy["Close"], spy["Open"]
    trend_up = spy_c > spy_c.rolling(200).mean()
    vix_s = vix.reindex(cal).ffill() if vix is not None else pd.Series(np.nan, index=cal)
    rows = []
    for ticker, g in earn.groupby("ticker"):
        df = frames.get(ticker)
        if df is None:
            continue
        df = df.reindex(cal)
        c, o = df["Close"], df["Open"]
        sigma20 = c.pct_change().shift(1).rolling(20, min_periods=20).std()
        for _, e in g.iterrows():
            fs = fund_surprise(e.get("eps_actual"), e.get("eps_estimate"))
            win = reaction_window(e["ann_ts"], cal)
            if fs is None or win is None:
                continue
            s, t = win
            if not (np.isfinite(c.iloc[s]) and np.isfinite(c.iloc[t]) and np.isfinite(sigma20.iloc[s])):
                continue
            ndays = t - s
            react = (c.iloc[t] / c.iloc[s] - 1.0) - (spy_c.iloc[t] / spy_c.iloc[s] - 1.0)
            rz = react / (sigma20.iloc[s] * math.sqrt(ndays)) if sigma20.iloc[s] > 0 else np.nan
            r = {"ticker": ticker, "date": cal[t], "fund_surprise": fs, "react_abn": react, "react_z": rz,
                 "vix": vix_s.iloc[t], "trend_up": trend_up.iloc[t]}
            for label, ent in (("", t + 1), ("lag_", t + 1 + lag)):
                for h in horizons:
                    ex = ent + h - 1
                    if ex < len(cal) and np.isfinite(o.iloc[ent]) and np.isfinite(c.iloc[ex]):
                        r[f"{label}fwd_{h}"] = (c.iloc[ex] / o.iloc[ent] - 1.0) - (spy_c.iloc[ex] / spy_o.iloc[ent] - 1.0)
                        r[f"{label}exit_{h}"] = cal[ex]
                    else:
                        r[f"{label}fwd_{h}"], r[f"{label}exit_{h}"] = np.nan, pd.NaT
            rows.append(r)
    if not rows:
        return pd.DataFrame()
    ev = pd.DataFrame(rows)
    ev["year"] = pd.to_datetime(ev["date"]).dt.year
    return ev.sort_values("date").reset_index(drop=True)


# ── Präregistrierte Hypothesen ──────────────────────────────────────────────
HYPOTHESES = {
    "S1_pead": "Richtung der fundamentalen Überraschung setzt sich fort (PEAD), long/short",
    "S1L_pead_long": "Oberes Trainings-Terzil der fundamentalen Überraschung, nur long",
    "S2_unpriced": "Unpriced Surprise (Perzentil Fundamental − Perzentil Reaktion), oberes vs. unteres Terzil",
    "S2L_unpriced_long": "Unpriced Surprise oberes Terzil, nur long (produktionsnah: Long Calls)",
    "S3_reaction_only": "KONTROLLE: Richtung der Kursreaktion allein (ohne Fundamentaldaten)",
    "S4_disagreement_long": "Positive Überraschung, aber negative Marktreaktion -> long",
}
USES_FUNDAMENTALS = ("S1_pead", "S1L_pead_long", "S2_unpriced", "S2L_unpriced_long", "S4_disagreement_long")


def _pct(train: pd.Series, x: pd.Series) -> pd.Series:
    ref = np.sort(train.dropna().to_numpy())
    if len(ref) == 0:
        return pd.Series(np.nan, index=x.index)
    return pd.Series(np.searchsorted(ref, x.to_numpy(), side="right") / len(ref), index=x.index)


def hypothesis_trades(name: str, train: pd.DataFrame, test: pd.DataFrame, h: int, prefix: str = "") -> pd.DataFrame:
    """Trades im Testjahr; alle Schwellen/Perzentile NUR aus `train`."""
    col = f"{prefix}fwd_{h}"

    def out(sel, sign):
        return pd.DataFrame({"date": sel["date"], "signed": sign * sel[col], "vix": sel["vix"],
                             "trend_up": sel["trend_up"], "ticker": sel["ticker"]})
    if name == "S1_pead":
        sel = test[test["fund_surprise"] != 0]
        return out(sel, np.sign(sel["fund_surprise"]))
    if name == "S1L_pead_long":
        sel = test[test["fund_surprise"] >= train["fund_surprise"].quantile(2 / 3)]
        return out(sel, 1.0)
    if name in ("S2_unpriced", "S2L_unpriced_long"):
        u_tr = _pct(train["fund_surprise"], train["fund_surprise"]) - _pct(train["react_z"], train["react_z"])
        u = _pct(train["fund_surprise"], test["fund_surprise"]) - _pct(train["react_z"], test["react_z"])
        hi, lo = u_tr.quantile(2 / 3), u_tr.quantile(1 / 3)
        if name == "S2L_unpriced_long":
            return out(test[u >= hi], 1.0)
        sel = test[(u >= hi) | (u <= lo)]
        return out(sel, np.where(u[sel.index] >= hi, 1.0, -1.0))
    if name == "S3_reaction_only":
        sel = test[test["react_z"].notna() & (test["react_z"] != 0)]
        return out(sel, np.sign(sel["react_z"]))
    if name == "S4_disagreement_long":
        return out(test[(test["fund_surprise"] > 0) & (test["react_z"] < 0)], 1.0)
    raise ValueError(name)


def walk_forward(ev: pd.DataFrame, name: str, h: int, prefix: str = "") -> pd.DataFrame:
    parts = []
    for y in sorted(ev["year"].unique()):
        if y < TEST_START_YEAR:
            continue
        start = pd.Timestamp(f"{y}-01-01")
        train = ev[(ev["year"] >= TRAIN_START_YEAR) & (pd.to_datetime(ev[f"{prefix}exit_{h}"]) < start)]
        test = ev[ev["year"] == y]
        if len(train) < 200 or len(test) == 0:
            continue
        parts.append(hypothesis_trades(name, train, test, h, prefix))
    cols = ["date", "signed", "vix", "trend_up", "ticker"]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


# ── Robustheit ───────────────────────────────────────────────────────────────
def _p_two_sided(t) -> float | None:
    if t is None:
        return None
    return float(math.erfc(abs(t) / math.sqrt(2)))


def benjamini_hochberg(pvals: dict[str, float | None], q: float = FDR_Q) -> dict[str, bool]:
    items = sorted(((k, p) for k, p in pvals.items() if p is not None), key=lambda kv: kv[1])
    m = len(items)
    cutoff = 0
    for i, (_, p) in enumerate(items, start=1):
        if p <= q * i / m:
            cutoff = i
    passed = {k for k, _ in items[:cutoff]}
    return {k: k in passed for k in pvals}


def placebo_p(ev: pd.DataFrame, name: str, h: int, actual_mean: float | None, n: int = PLACEBO_N,
              seed: int = SEED) -> float | None:
    """Anteil der Permutationen (fund_surprise innerhalb des Jahres gemischt) mit Mittel >= tatsächlichem."""
    if actual_mean is None or name not in USES_FUNDAMENTALS:
        return None
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n):
        perm = ev.copy()
        perm["fund_surprise"] = perm.groupby("year")["fund_surprise"].transform(
            lambda s: s.to_numpy()[rng.permutation(len(s))])
        m = trade_metrics(walk_forward(perm, name, h), COST_BASE_PER_SIDE).get("mean")
        hits += int(m is not None and m >= actual_mean)
    return round((hits + 1) / (n + 1), 4)


def _half(ticker: str) -> int:
    return int(hashlib.sha256(ticker.encode()).hexdigest(), 16) % 2


def replication(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {}
    yrs = sorted(pd.to_datetime(trades["date"]).dt.year.unique())
    mid = yrs[len(yrs) // 2] if yrs else None
    halves = trades["ticker"].map(_half)
    d = pd.to_datetime(trades["date"]).dt.year
    out = {}
    for label, m in (("universe_A", halves == 0), ("universe_B", halves == 1),
                     ("period_early", d < mid), ("period_late", d >= mid)):
        met = trade_metrics(trades[m], COST_BASE_PER_SIDE)
        out[label] = {"n": met.get("n"), "mean": met.get("mean"), "t_months": met.get("t_months")}
    return out


def decide(base: dict, stress: dict, regimes: dict, lag: dict, rep: dict, bh_ok: bool, placebo: float | None,
           uses_fund: bool) -> dict:
    reasons = []
    core = (base.get("mean") or 0) > 0 and (base.get("t_months") or 0) >= 2.0
    if not core:
        reasons.append(f"OOS-Mittel/t nicht ausreichend (mean={base.get('mean')}, t={base.get('t_months')})")
    if (base.get("years_positive_share") or 0) < 0.6:
        reasons.append(f"nur {base.get('years_positive_share')} der Jahre positiv")
    if not (stress.get("mean") or 0) > 0:
        reasons.append("bei 25 bp/Seite nicht positiv")
    if not bh_ok:
        reasons.append(f"nicht signifikant nach Benjamini-Hochberg (q={FDR_Q})")
    if uses_fund and (placebo is None or placebo >= 0.05):
        reasons.append(f"Placebo nicht übertroffen (p={placebo})")
    if not (lag.get("mean") or 0) > 0:
        reasons.append(f"Lag-Test ({LAG_DAYS} T später) nicht positiv")
    if any(not ((v or {}).get("mean") or 0) > 0 for v in rep.values()) or not rep:
        reasons.append("Replikation (Universumshälften / Zeithälften) nicht durchgehend positiv")
    if any((v.get("n") or 0) and (v.get("mean") is None or v["mean"] <= 0) for v in regimes.values()):
        reasons.append("nicht in allen Regimen positiv")
    verdict = "KEEP" if not reasons else ("MODIFY" if core else "REJECT")
    return {"verdict": verdict, "reasons": reasons}


# ── Gegenhypothese aus dem ersten Lauf (Nachregistrierung 2026-10-03, vor jeder Holdout-Auswertung) ──
# Befund Lauf 2026-10-03 (OOS 2019+): S3_reaction_only signifikant NEGATIV (t=-2,59, BH) ->
# H_alt gewann. Laut Protokoll kein Vorzeichenwechsel auf denselben Daten, sondern eine NEUE
# Hypothese, geprüft ausschließlich auf Daten, die bisher nie Testperiode waren:
REVERSAL_ID = "S5_reaction_reversal"
REVERSAL_DESC = ("Earnings-Reaktion kehrt sich um: gegen die Richtung der abnormalen Reaktion positionieren "
                 "(Gegenhypothese zu S3, Holdout = Jahre vor dem OOS-Start, nie Testperiode)")


def holdout_reversal(ev: pd.DataFrame, horizons=HORIZONS) -> dict:
    """Replikation der Gegenhypothese NUR auf Jahren < TEST_START_YEAR. Keine Parameter (reines
    Vorzeichen), daher kein Training nötig; zusätzlich Lag-Test und Universums-Hälften."""
    ho = ev[ev["year"] < TEST_START_YEAR]
    ho = ho[ho["react_z"].notna() & (ho["react_z"] != 0)]
    out = {"description": REVERSAL_DESC, "holdout_years": sorted(int(y) for y in ho["year"].unique()),
           "registered": "2026-10-03 nach Lauf 1, vor Holdout-Auswertung"}
    for h in horizons:
        sign = -np.sign(ho["react_z"])
        tr = pd.DataFrame({"date": ho["date"], "signed": sign * ho[f"fwd_{h}"], "vix": ho["vix"],
                           "trend_up": ho["trend_up"], "ticker": ho["ticker"]})
        lag = pd.DataFrame({"date": ho["date"], "signed": sign * ho[f"lag_fwd_{h}"]})
        base = trade_metrics(tr, COST_BASE_PER_SIDE)
        out[f"h{h}"] = {"base_cost": base,
                        "stress_cost": {k: trade_metrics(tr, COST_STRESS_PER_SIDE).get(k) for k in ("mean", "t_months")},
                        "lag": {k: trade_metrics(lag, COST_BASE_PER_SIDE).get(k) for k in ("n", "mean", "t_months")},
                        "replication": replication(tr), "regimes": regime_split(tr, COST_BASE_PER_SIDE),
                        "p_value": _p_two_sided(base.get("t_months"))}
    d = out.get(f"h{DECISION_H}") or {}
    b = d.get("base_cost") or {}
    reasons = []
    core = (b.get("mean") or 0) > 0 and (b.get("t_months") or 0) >= 2.0
    if not core:
        reasons.append(f"Holdout-Mittel/t nicht ausreichend (mean={b.get('mean')}, t={b.get('t_months')})")
    if not ((d.get("stress_cost") or {}).get("mean") or 0) > 0:
        reasons.append("bei 25 bp/Seite nicht positiv")
    if not ((d.get("lag") or {}).get("mean") or 0) > 0:
        reasons.append("Lag-Test nicht positiv")
    if not d.get("replication") or any(not ((v or {}).get("mean") or 0) > 0 for v in d["replication"].values()):
        reasons.append("Replikation nicht durchgehend positiv")
    # Holdout bestanden -> nur PROSPECTIVE-Kandidat (Forward entscheidet), nie Produktion
    out["decision"] = {"verdict": "KEEP" if not reasons else ("MODIFY" if core else "REJECT"),
                       "reasons": reasons,
                       "next": ("als prospektiven Challenger vorschlagen (Forward entscheidet)" if not reasons
                                else "verworfen bzw. weiter offen; Befund bleibt in der Research Memory")}
    return out


def evaluate(ev: pd.DataFrame, placebo_n: int = PLACEBO_N) -> dict:
    res = {"n_events": int(len(ev)), "n_tickers": int(ev["ticker"].nunique()) if len(ev) else 0,
           "period": [str(pd.to_datetime(ev["date"]).min().date()), str(pd.to_datetime(ev["date"]).max().date())]
           if len(ev) else None, "hypotheses": {}, "data_gaps": DATA_GAPS}
    pvals, cache = {}, {}
    for name in HYPOTHESES:
        for h in HORIZONS:
            tr = walk_forward(ev, name, h)
            base = trade_metrics(tr, COST_BASE_PER_SIDE)
            cache[(name, h)] = (tr, base)
            pvals[f"{name}|h{h}"] = _p_two_sided(base.get("t_months"))
    bh = benjamini_hochberg(pvals)
    for name, desc in HYPOTHESES.items():
        per = {"description": desc}
        for h in HORIZONS:
            tr, base = cache[(name, h)]
            stress = trade_metrics(tr, COST_STRESS_PER_SIDE)
            regimes = regime_split(tr, COST_BASE_PER_SIDE)
            lag = trade_metrics(walk_forward(ev, name, h, prefix="lag_"), COST_BASE_PER_SIDE)
            rep = replication(tr)
            per[f"h{h}"] = {"base_cost": base, "stress_cost": {k: stress.get(k) for k in ("mean", "t_months")},
                            "regimes": regimes, "lag": {k: lag.get(k) for k in ("n", "mean", "t_months")},
                            "replication": rep, "p_value": pvals[f"{name}|h{h}"], "bh_significant": bh[f"{name}|h{h}"]}
            if h == DECISION_H:
                pl = placebo_p(ev, name, h, base.get("mean"), n=placebo_n) if (base.get("mean") or 0) > 0 else None
                per[f"h{h}"]["placebo_p"] = pl
                per["decision"] = decide(base, stress, regimes, lag, rep, bh[f"{name}|h{h}"], pl,
                                         name in USES_FUNDAMENTALS)
        res["hypotheses"][name] = per
    ctrl = res["hypotheses"].get("S3_reaction_only", {}).get(f"h{DECISION_H}", {}).get("base_cost", {})
    res["control_mean"] = ctrl.get("mean")
    res["counter_hypotheses"] = {REVERSAL_ID: holdout_reversal(ev)}
    return res


# ── Daten (nur CI, Netzwerk) ─────────────────────────────────────────────────
def fetch_earnings(tickers: list[str], limit: int = 40) -> pd.DataFrame:
    import yfinance as yf
    rows = []
    for i, t in enumerate(tickers):
        try:
            df = yf.Ticker(t).get_earnings_dates(limit=limit)
        except Exception as e:  # noqa: BLE001 – einzelne Ticker dürfen fehlen (gezählt)
            log.debug(f"surprise_engine: keine Earnings für {t}: {e}")
            continue
        if df is None or df.empty:
            continue
        for ts, r in df.iterrows():
            rows.append({"ticker": t, "ann_ts": ts, "eps_estimate": r.get("EPS Estimate"),
                         "eps_actual": r.get("Reported EPS")})
        if i and i % 100 == 0:
            log.info(f"surprise_engine: Earnings {i}/{len(tickers)}")
    return pd.DataFrame(rows)


def render_md(res: dict) -> str:
    L = [f"# Surprise Engine – Walk-Forward-Studie (OOS ab {TEST_START_YEAR})", "",
         f"Events: {res['n_events']} · Ticker: {res['n_tickers']} · Zeitraum: {res['period']}",
         f"Präregistrierung: {PREREG}",
         "Renditen marktbereinigt (minus SPY), netto 10 bp/Seite; PIT-Universum (S&P 500 inkl. entfernter Titel).", "",
         "| Hypothese | h | n | Mittel | t | Jahre + | 25bp | Lag | p | BH | Placebo p | Verdikt |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, per in res["hypotheses"].items():
        for h in HORIZONS:
            x = per[f"h{h}"]
            b = x["base_cost"]
            L.append(f"| {name} | {h} | {b.get('n')} | {b.get('mean')} | {b.get('t_months')} | "
                     f"{b.get('years_positive_share')} | {x['stress_cost'].get('mean')} | {x['lag'].get('mean')} | "
                     f"{x['p_value'] if x['p_value'] is None else round(x['p_value'], 4)} | "
                     f"{'ja' if x['bh_significant'] else 'nein'} | {x.get('placebo_p', '–')} | "
                     f"{per['decision']['verdict'] if h == DECISION_H else ''} |")
    L += ["", f"## Entscheidungen (h={DECISION_H})", ""]
    for name, per in res["hypotheses"].items():
        d = per["decision"]
        L.append(f"- **{name}** ({per['description']}): {d['verdict']}"
                 + (f" – {'; '.join(d['reasons'])}" if d["reasons"] else ""))
    for cid, c in (res.get("counter_hypotheses") or {}).items():
        b = (c.get(f"h{DECISION_H}") or {}).get("base_cost") or {}
        L += ["", f"## Gegenhypothese {cid} (Holdout {c.get('holdout_years')})", "", c.get("description", ""), "",
              f"n={b.get('n')}, Mittel={b.get('mean')}, t={b.get('t_months')}, Jahre+={b.get('years_positive_share')} "
              f"→ **{c['decision']['verdict']}**" + (f" – {'; '.join(c['decision']['reasons'])}"
                                                    if c["decision"]["reasons"] else ""),
              f"Nächster Schritt: {c['decision']['next']}"]
    L += ["", "## Datenlücken (nie simuliert)", ""] + [f"- {g}" for g in res.get("data_gaps") or []]
    return "\n".join(L) + "\n"


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    from modules.price_event_study import fetch_frames
    from modules.universe import is_member, research_universe
    uni = research_universe(f"{TRAIN_START_YEAR}-01-01")
    tickers = uni["tickers"]
    frames, spy, vix = fetch_frames(tickers)
    earn = fetch_earnings(list(frames))
    log.info(f"surprise_engine: {len(earn)} Meldungen für {earn['ticker'].nunique() if len(earn) else 0} Ticker")
    ev = build_events(earn, frames, spy, vix) if len(earn) else pd.DataFrame()
    if not ev.empty:
        ev = ev[[is_member(uni["intervals"].get(t, []), str(pd.Timestamp(d).date()))
                 for t, d in zip(ev["ticker"], ev["date"])]].reset_index(drop=True)
    res = evaluate(ev) if not ev.empty else {"n_events": 0, "n_tickers": 0, "period": None, "hypotheses": {},
                                             "data_gaps": DATA_GAPS}
    res.update({"generated": datetime.utcnow().isoformat(timespec="seconds"), "prereg": PREREG,
                "tickers_requested": len(tickers), "tickers_with_prices": len(frames),
                "tickers_with_earnings": int(earn["ticker"].nunique()) if len(earn) else 0})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "surprise_study.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str))
    md = render_md(res) if res["hypotheses"] else "# Surprise Engine\n\nKeine Events (Datenabruf leer).\n"
    (OUT_DIR / "surprise_study.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

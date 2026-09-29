"""
modules/counterfactual.py – Counterfactual Engine + Stress-Umgebung (NUR SHADOW)

Frage je Kandidat: "Was müsste sich ändern, damit diese Prognose falsch wird?"
  * Makro-/Markt-Szenarien verschieben die Datumsmerkmale (VIX, Zinsen, Crash,
    USD, Öl, Inflation, Liquidität) – vorab festgelegt in config/next_protocol.yaml.
  * Merkmals-Flips spiegeln Aktienmerkmals-Gruppen (Momentum-Umkehr,
    Volatilitäts-Umkehr).
  * Die trainierten Basismodelle werden unter jedem Szenario neu ausgewertet;
    daraus: Ensemble-Rang je Szenario, erwartete Rendite je Szenario,
    dominante Annahme, fragil/robust.
Fragil = fällt unter EINEM einzelnen Szenario oder Flip aus dem Top-Dezil.
High-Confidence darf nie auf einer einzigen fragilen Annahme beruhen.

Stress: dieselben Szenarien für das ganze Top-Dezil (Umschlag der Auswahl)
plus historische Stressfenster (echte OOS-Daten, keine synthetischen Renditen
als Evidenz). Synthetische Verschiebungen dienen nur der Robustheitsprüfung.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger(__name__)

NP = yaml.safe_load((Path(__file__).resolve().parent.parent / "config" / "next_protocol.yaml").read_text(encoding="utf-8"))
SCENARIOS: dict = NP["counterfactual"]["scenarios"]
FLIPS = {"momentum_reversal": ("mom_12_1", "mom_3m", "rs_63", "dist_52w_high"),
         "volatility_flip": ("vol_20", "vol_60", "max_ret_21", "beta_126")}
ALL_CASES = list(SCENARIOS) + NP["counterfactual"]["feature_flips"]
TOP_RANK = 0.9

# Historische Stressfenster (Datum der Signal-Kohorten)
STRESS_WINDOWS = {
    "covid_crash_2020": ("2020-02-14", "2020-04-30"),
    "rate_inflation_shock_2022": ("2022-01-01", "2022-10-31"),
    "q4_selloff_2018": ("2018-10-01", "2018-12-31"),
    "regional_banks_2023": ("2023-03-01", "2023-05-31"),
    "tariff_shock_2025": ("2025-03-20", "2025-05-15"),
}


def apply_case(df: pd.DataFrame, case: str) -> pd.DataFrame:
    """Kopie mit verschobenen Datumsmerkmalen bzw. gespiegelten Aktienmerkmalen."""
    d = df.copy()
    if case in SCENARIOS:
        for col, shift in SCENARIOS[case].items():
            if col in d:
                d[col] = d[col] + shift
    elif case in FLIPS:
        for col in FLIPS[case]:
            if col in d:
                d[col] = -d[col]                  # Querschnittsränge sind um 0 zentriert
    return d


def ensemble_rank(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    s = df[cols].mean(axis=1)
    return s.groupby(df["date"]).rank(pct=True)


def fragility(df: pd.DataFrame, base_col: str, case_cols: dict[str, str]) -> pd.DataFrame:
    """je Zeile: schlimmster Fall, Rangverlust, fragil (nur für Top-Dezil sinnvoll)."""
    base = df[base_col]
    drops = pd.DataFrame({c: base - df[col] for c, col in case_cols.items()}, index=df.index)
    worst = drops.idxmax(axis=1)
    worst_drop = drops.max(axis=1)
    min_rank = pd.DataFrame({c: df[col] for c, col in case_cols.items()}).min(axis=1)
    return pd.DataFrame({"cf_worst_case": worst, "cf_worst_drop": worst_drop, "cf_min_rank": min_rank,
                         "cf_fragile": (base >= TOP_RANK) & (min_rank < TOP_RANK),
                         "cf_n_breaking": sum((df[col] < TOP_RANK).astype(int) for col in case_cols.values())},
                        index=df.index)


def historical_stress(res: pd.DataFrame, score_cols: dict[str, str]) -> dict:
    """Expectancy/Trefferquote des Top-Dezils je Variante in echten Stressfenstern."""
    from modules import meta_learning as meta
    out = {}
    for w, (a, b) in STRESS_WINDOWS.items():
        sub = res[(res["date"] >= pd.Timestamp(a)) & (res["date"] <= pd.Timestamp(b))]
        if sub["date"].nunique() < 3:
            out[w] = {"n_cohorts": int(sub["date"].nunique()), "note": "außerhalb des OOS-Zeitraums"}
            continue
        out[w] = {}
        for name, col in score_cols.items():
            if col not in sub:
                continue
            m = meta.portfolio_metrics(meta.positions(sub, col))
            out[w][name] = {"expectancy": m.get("expectancy"), "hit_rate": m.get("hit_rate"),
                            "n_cohorts": m.get("n_cohorts")}
    return out


def scenario_turnover(df: pd.DataFrame, base_col: str, case_cols: dict[str, str]) -> dict:
    """Anteil des Top-Dezils, der unter einem Szenario herausfällt (je Stichtag gemittelt)."""
    out = {}
    top = df[base_col] >= TOP_RANK
    for c, col in case_cols.items():
        out_share = (df.loc[top, col] < TOP_RANK).groupby(df.loc[top, "date"]).mean()
        out[c] = round(float(out_share.mean()), 4) if len(out_share) else None
    return out


def latest_counterfactuals(panel: pd.DataFrame, specs: list[dict]) -> dict:
    """Gegenwart: Modelle auf allen fertigen Labels, jüngster Stichtag unter
    jedem Fall neu ausgewertet. Zusätzlich 60d-Median-Rendite je Szenario
    (Quantil-Modell). -> {ticker: {...}}"""
    from modules import ml_research as ml
    latest = panel["date"].max()
    snap = panel[panel["date"] == latest]
    cutoff = latest + pd.Timedelta(days=1)
    base_cols, case_cols = [], {c: [] for c in ALL_CASES}
    frame = snap[["date", "ticker"]].copy()
    for spec in specs:
        train = ml.purged(panel, cutoff, ml._horizon(spec))
        try:
            params = ml.select_params(spec, train, cutoff)
            m = ml.fit(spec, train, params)
        except ValueError as e:
            log.warning(f"counterfactual: {spec['id']} nicht trainierbar: {e}")
            continue
        frame[f"b_{spec['id']}"] = pd.Series(m.predict(snap), index=snap.index).rank(pct=True)
        base_cols.append(f"b_{spec['id']}")
        for c in ALL_CASES:
            frame[f"{c}_{spec['id']}"] = pd.Series(m.predict(apply_case(snap, c)), index=snap.index).rank(pct=True)
            case_cols[c].append(f"{c}_{spec['id']}")
    if not base_cols:
        return {}
    frame["base_rank"] = frame[base_cols].mean(axis=1).rank(pct=True)
    ccols = {}
    for c, cols in case_cols.items():
        frame[f"rank_{c}"] = frame[cols].mean(axis=1).rank(pct=True)
        ccols[c] = f"rank_{c}"
    fr = fragility(frame, "base_rank", ccols)
    train60 = ml.purged(panel, cutoff, 60)
    exp60 = {}
    if len(train60) > 5000:
        um = ml.fit_uncertainty(train60)
        exp60["base"] = ml.predict_uncertainty(um, snap)["q_mid"]
        for c in ALL_CASES:
            exp60[c] = ml.predict_uncertainty(um, apply_case(snap, c))["q_mid"]
    out = {}
    for i, r in frame.iterrows():
        rec = {"base_rank": round(float(r["base_rank"]), 4),
               "scenario_ranks": {c: round(float(r[f"rank_{c}"]), 4) for c in ALL_CASES},
               "worst_case": fr.at[i, "cf_worst_case"], "worst_rank_drop": round(float(fr.at[i, "cf_worst_drop"]), 4),
               "fragile": bool(fr.at[i, "cf_fragile"]), "n_cases_breaking_top_decile": int(fr.at[i, "cf_n_breaking"])}
        if exp60:
            rec["expected_return_60"] = {k: round(float(v.loc[i]), 4) for k, v in exp60.items()}
        rec["dominant_assumption"] = rec["worst_case"] if rec["fragile"] else None
        out[r["ticker"]] = rec
    return {"date": str(latest.date()), "cases": ALL_CASES, "per_ticker": out,
            "scenario_turnover_top_decile": scenario_turnover(frame, "base_rank", ccols)}

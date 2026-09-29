"""
modules/causal_research.py – Causal Research Layer (NUR SHADOW, Phase 3)

    python -m modules.causal_research        (CI: Netzwerk nur über world_model.fetch_market/load_archive)

Frage: Welche World-Model-Indikatoren laufen Sektor-Relativrenditen (Sektor-ETF
minus SPY) mit Vorlauf voraus – und wie belastbar ist diese Evidenz?

Treiber = Roh-Indikatoren des World Models (wöchentlich, PIT). Ziele = Sektor-
Relativrendite vorwärts über h = 4 und 13 Wochen (Close t -> Close t+h Wochen).
Geprüft werden NUR vorab erklärte ökonomische Paare (config/intelligence_protocol.yaml,
causal_research.priors) – keine Paar-Suche, keine Parameter-Suche, Priors werden
nie an Daten angepasst, keine LLM-Aussagen.

Tests je (Treiber, Ziel, h):
  1. Korrelation zeitgleich (Treiber_t vs. Relativrendite der letzten 4 Wochen)
  2. Lead/Lag: Korr(Treiber_t, Ziel t->t+h) gegen Korr(Treiber_t, Ziel t-h->t)
  3. Granger-artig OUT-OF-SAMPLE: Walk-Forward (jahresweise, Training nur mit Zielen,
     deren Ende vor dem Testjahr liegt), Ridge "eigene Vergangenheit + Markt-
     Kontrollen" gegen dasselbe plus Treiber; Monats-Block-Bootstrap (einseitig)
  4. konditionale Robustheit: partielle Korrelation gegeben Kontrollen
  5. Stabilität: gleiches Vorzeichen in beiden Hälften und >= 60 % der Jahre
  6. Replikation: gleiches Vorzeichen beim anderen Horizont (4 <-> 13 Wochen)
Mehrfachtest: Benjamini-Hochberg über ALLE getesteten Kombinationen (n_tested wird ausgewiesen).

Evidence-Level (deterministisch): correlation < predictive_relationship <
temporally_leading < causal_hypothesis. "causal_evidence" wird NIE automatisch
vergeben (Feld immer "none"): dafür wäre ein Quasi-Experiment nötig, das hier
nicht vorliegt. causal_confidence ist höchstens MODERATE.

sector_tilts(): PIT-Walk-Forward – für Stichtag t werden Koeffizienten nur mit
Zielen geschätzt, deren Ende < t liegt.

Vorab-Regel (Protokoll): KEEP wenn >= 1 Beziehung temporally_leading (nach BH) UND
die Sektor-Tilts eine positive Rang-IC mit Bootstrap-Untergrenze > 0 haben;
MODIFY wenn nur predictive_relationship; sonst REJECT.
Ausgabe: outputs/research/causal_research.{json,md}. Keine Produktionswirkung.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger(__name__)

PROTOCOL_PATH = Path(__file__).resolve().parent.parent / "config" / "intelligence_protocol.yaml"
CP = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))["causal_research"]
OUT_DIR = Path("outputs/research")
OUT_JSON = OUT_DIR / "causal_research.json"
OUT_MD = OUT_DIR / "causal_research.md"

SECTOR_ETFS = ("XLK", "XLF", "XLE", "XLI", "XLV", "XLY", "XLP", "XLU", "XLB")
# ETF -> yfinance-Sektorname (wie outputs/research/sector_map.json)
SECTOR_ETF_MAP = {
    "XLK": "Technology", "XLF": "Financial Services", "XLE": "Energy", "XLI": "Industrials",
    "XLV": "Healthcare", "XLY": "Consumer Cyclical", "XLP": "Consumer Defensive",
    "XLU": "Utilities", "XLB": "Basic Materials",
}
LEVELS = ("correlation", "predictive_relationship", "temporally_leading", "causal_hypothesis")


# ── Hilfen ───────────────────────────────────────────────────────────────────

def _f(x, nd: int = 4):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), nd)


def _sign(x) -> int:
    return 0 if x is None or not math.isfinite(x) or x == 0 else (1 if x > 0 else -1)


def _grid_returns(px: pd.DataFrame, dates: list, h: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """(fwd, past, end): Sektor-Relativrendite vorwärts t->t+h und rückwärts t-h->t auf dem
    Wochenraster `dates`; end = Datum des Ziel-Endes (Close t+h Wochen)."""
    cols = [e for e in SECTOR_ETFS if e in px]
    P = px.reindex(pd.DatetimeIndex(dates))
    spy = P["SPY"]
    fwd = (P[cols].shift(-h).div(P[cols]) - 1.0).sub(spy.shift(-h) / spy - 1.0, axis=0)
    past = (P[cols].div(P[cols].shift(h)) - 1.0).sub(spy / spy.shift(h) - 1.0, axis=0)
    end = pd.Series(pd.DatetimeIndex(dates), index=pd.DatetimeIndex(dates)).shift(-h)
    return fwd.replace([np.inf, -np.inf], np.nan), past.replace([np.inf, -np.inf], np.nan), end


def _pair_frame(ind: pd.DataFrame, px: pd.DataFrame, driver: str, etf: str, h: int, controls: list[str]) -> pd.DataFrame:
    dates = list(ind.index)
    fwd, past, end = _grid_returns(px, dates, h)
    _, past4, _ = _grid_returns(px, dates, 4)
    df = pd.DataFrame({"x": ind[driver], "y": fwd[etf], "past": past[etf], "past4": past4[etf], "end": end})
    for c in controls:
        df[c] = ind[c] if c in ind else np.nan
    return df


def _ridge_fit_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, alpha: float) -> np.ndarray:
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    A = (Xtr - mu) / sd
    ym = ytr.mean()
    try:
        w = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (ytr - ym))
    except np.linalg.LinAlgError:
        log.warning("causal_research: Ridge-Lösung singulär -> Konstante")
        w = np.zeros(A.shape[1])
    return ((Xte - mu) / sd) @ w + ym


def benjamini_hochberg(p: list[float]) -> list[float]:
    """Benjamini-Hochberg-q-Werte (monotone Step-up-Korrektur) in Eingabereihenfolge."""
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    q = np.empty(m)
    prev = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        prev = min(prev, float(p[i]) * m / rank)
        q[i] = prev
    return [float(x) for x in q]


def month_block_bootstrap(d: pd.Series, n: int, seed: int, alpha: float) -> dict:
    """Bootstrap über Monatsmittel (Blöcke). p_one_sided = P(Mittel <= 0) (Nullhypothese: kein Gewinn)."""
    d = d.dropna()
    if d.empty:
        return {"n_months": 0}
    m = d.groupby(pd.DatetimeIndex(d.index).to_period("M")).mean()
    if len(m) < 12:
        return {"n_months": int(len(m)), "mean": float(m.mean())}
    rng = np.random.default_rng(seed)
    arr = m.to_numpy()
    boots = arr[rng.integers(0, len(arr), (n, len(arr)))].mean(axis=1)
    return {"n_months": int(len(m)), "mean": float(arr.mean()), "lo": float(np.quantile(boots, alpha)),
            "hi": float(np.quantile(boots, 1 - alpha)), "p_one_sided": float((np.sum(boots <= 0) + 1) / (n + 1))}


# ── Einzeltests ──────────────────────────────────────────────────────────────

def walk_forward_gain(df: pd.DataFrame, base_cols: list[str], full_cols: list[str], locked_from: pd.Timestamp,
                      cfg: dict) -> pd.Series:
    """OOS-Verlustdifferenz (Basis − Voll, quadratischer Fehler) je Teststichtag. Training nur mit
    Zielen, deren Ende vor dem Testjahr liegt; Testziele enden vor dem Locked-Holdout."""
    d = df.dropna(subset=full_cols + ["y", "end"])
    out = []
    last_year = (locked_from - pd.Timedelta(days=1)).year
    for yr in range(cfg["first_test_year"], last_year + 1):
        ts, te = pd.Timestamp(year=yr, month=1, day=1), min(pd.Timestamp(year=yr + 1, month=1, day=1), locked_from)
        tr = d[(d.index < ts) & (d["end"] < ts)]
        tt = d[(d.index >= ts) & (d.index < te) & (d["end"] < locked_from)]
        if len(tr) < cfg["min_train_samples"] or tt.empty:
            continue
        ytr, yte = tr["y"].to_numpy(float), tt["y"].to_numpy(float)
        pb = _ridge_fit_predict(tr[base_cols].to_numpy(float), ytr, tt[base_cols].to_numpy(float), cfg["ridge_alpha"])
        pf = _ridge_fit_predict(tr[full_cols].to_numpy(float), ytr, tt[full_cols].to_numpy(float), cfg["ridge_alpha"])
        out.append(pd.Series((yte - pb) ** 2 - (yte - pf) ** 2, index=tt.index))
    return pd.concat(out) if out else pd.Series(dtype=float)


def _partial_corr(d: pd.DataFrame, controls: list[str]) -> float | None:
    d = d.dropna(subset=["x", "y"] + controls)
    if len(d) < 30 or not controls:
        return None
    Z = np.column_stack([np.ones(len(d))] + [d[c].to_numpy(float) for c in controls])
    rx = d["x"].to_numpy(float) - Z @ np.linalg.lstsq(Z, d["x"].to_numpy(float), rcond=None)[0]
    ry = d["y"].to_numpy(float) - Z @ np.linalg.lstsq(Z, d["y"].to_numpy(float), rcond=None)[0]
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _corr(a: pd.Series, b: pd.Series) -> float | None:
    d = pd.concat([a, b], axis=1).dropna()
    if len(d) < 30 or d.iloc[:, 0].std() == 0 or d.iloc[:, 1].std() == 0:
        return None
    return float(d.iloc[:, 0].corr(d.iloc[:, 1]))


def pair_stats(ind: pd.DataFrame, px: pd.DataFrame, driver: str, etf: str, h: int, locked_from: pd.Timestamp,
               cfg: dict) -> dict:
    """Alle Tests (1)-(5) für ein (Treiber, Ziel, h); (6) und BH folgen in analyze()."""
    controls = [c for c in cfg["controls"] if c in ind.columns]
    df = _pair_frame(ind, px, driver, etf, h, controls)
    d = df[df["end"] < locked_from]                                   # nie in den Locked-Holdout hinein
    s: dict = {"n": int(d.dropna(subset=["x", "y"]).shape[0])}
    s["corr_contemporaneous"] = _corr(d["x"], d["past4"])
    s["corr_lead"] = _corr(d["x"], d["y"])
    s["corr_lag"] = _corr(d["x"], d["past"])
    s["lead_gt_lag"] = bool(s["corr_lead"] is not None and (s["corr_lag"] is None or abs(s["corr_lead"]) > abs(s["corr_lag"])))
    base = ["past"] + controls
    gain = walk_forward_gain(df, base, base + ["x"], locked_from, cfg)
    bs = month_block_bootstrap(gain, cfg["bootstrap_n"], cfg["bootstrap_seed"], 0.05)
    dd = df.dropna(subset=["x", "y", "past", "end"] + controls)
    dd = dd[dd["end"] < locked_from]
    s["oos"] = {"n_oos": int(len(gain)), "mean_gain": _f(bs.get("mean"), 8), "n_months": bs.get("n_months", 0),
                "p_one_sided": bs.get("p_one_sided"), "ci": [_f(bs.get("lo"), 8), _f(bs.get("hi"), 8)]}
    pc = _partial_corr(dd, controls + ["past"])
    s["partial_corr"] = pc
    # Stabilität: beide Hälften und Jahre
    e = d.dropna(subset=["x", "y"]).sort_index()
    obs = _sign(s["corr_lead"])
    halves = []
    if len(e) >= 60:
        mid = len(e) // 2
        halves = [_sign(_corr(e["x"].iloc[:mid], e["y"].iloc[:mid])), _sign(_corr(e["x"].iloc[mid:], e["y"].iloc[mid:]))]
    yrs = [_sign(_corr(g["x"], g["y"])) for _, g in e.groupby(e.index.year) if len(g) >= cfg["min_year_obs"]]
    s["stability"] = {"half_signs": halves, "both_halves_same": bool(halves and obs != 0 and all(x == obs for x in halves)),
                      "n_years": len(yrs), "year_share_same": _f(sum(1 for x in yrs if x == obs) / len(yrs)) if yrs and obs else None}
    return s


# ── Evidence-Level ───────────────────────────────────────────────────────────

def _grade_robustness(raw: float | None, partial: float | None, cfg: dict) -> str:
    if raw is None or partial is None or _sign(raw) != _sign(partial):
        return "LOW"
    r = abs(partial) / abs(raw) if raw else 0.0
    return "HIGH" if r >= cfg["robustness_ratio"]["high"] else "MEDIUM" if r >= cfg["robustness_ratio"]["medium"] else "LOW"


def _grade_stability(st: dict, cfg: dict) -> str:
    share_ok = st.get("year_share_same") is not None and st["year_share_same"] >= cfg["stability_min_year_share"] \
        and st.get("n_years", 0) >= 3
    if st.get("both_halves_same") and share_ok:
        return "HIGH"
    return "MEDIUM" if (st.get("both_halves_same") or share_ok) else "LOW"


def assess(stats: dict, prior_sign: int | None, q: float | None, other: dict | None, cfg: dict) -> dict:
    """Deterministische Einstufung -> {"level", "evidence": {...}, "causal_evidence": "none", "observed_sign"}.
    `other` = Stats desselben Paars beim anderen Horizont (Replikation)."""
    obs = _sign(stats.get("corr_lead"))
    gain = stats["oos"].get("mean_gain")
    predictive = bool(q is not None and q <= cfg["fdr_q"] and gain is not None and gain > 0)
    leading = bool(predictive and stats.get("lead_gt_lag"))
    if prior_sign is None or obs == 0:
        plaus = "UNKNOWN"
    else:
        plaus = "HIGH" if obs == prior_sign else "LOW"
    stab = _grade_stability(stats["stability"], cfg)
    if other is None or obs == 0:
        rep = "LOW"
    elif _sign(other.get("corr_lead")) == obs:
        og = other["oos"].get("mean_gain")
        rep = "HIGH" if (og is not None and og > 0) else "MEDIUM"
    else:
        rep = "LOW"
    rob = _grade_robustness(stats.get("corr_lead"), stats.get("partial_corr"), cfg)
    hypothesis = bool(leading and plaus == "HIGH" and stab == "HIGH" and rep in ("HIGH", "MEDIUM"))
    level = "causal_hypothesis" if hypothesis else "temporally_leading" if leading else \
        "predictive_relationship" if predictive else "correlation"
    lead_grade = "HIGH" if leading else "MEDIUM" if (predictive or (
        stats["oos"].get("p_one_sided") is not None and stats["oos"]["p_one_sided"] < 0.05 and gain and gain > 0)) else "LOW"
    conf = "MODERATE" if (hypothesis and rob in ("HIGH", "MEDIUM")) else "LOW"   # HIGH nie ohne Quasi-Experiment
    return {"level": level, "observed_sign": obs, "causal_evidence": "none",
            "evidence": {"predictive_lead": lead_grade, "conditional_robustness": rob, "stability": stab,
                         "replication": rep, "economic_plausibility": plaus, "causal_confidence": conf}}


# ── Gesamtanalyse ────────────────────────────────────────────────────────────

def analyze(ind: pd.DataFrame, px: pd.DataFrame, locked_from: pd.Timestamp, priors: list[dict] | None = None,
            cfg: dict | None = None) -> dict:
    """Testet alle vorab erklärten Paare × Horizonte, BH über alle, stuft ein. -> {"n_tested","fdr_q","relations"}."""
    cfg = {**CP, **(cfg or {})}
    priors = CP["priors"] if priors is None else priors
    raw = {}
    for pr in priors:
        drv, etf = pr["driver"], pr["target"]
        if drv not in ind.columns or etf not in px.columns or "SPY" not in px.columns:
            log.warning(f"causal_research: Paar {drv}->{etf} übersprungen (Daten fehlen)")
            continue
        for h in cfg["horizons_weeks"]:
            raw[(drv, etf, h)] = pair_stats(ind, px, drv, etf, h, locked_from, cfg)
    keys = list(raw)
    ps = [raw[k]["oos"].get("p_one_sided") if raw[k]["oos"].get("p_one_sided") is not None else 1.0 for k in keys]
    qs = dict(zip(keys, benjamini_hochberg(ps)))
    prior_sign = {(p["driver"], p["target"]): int(p["sign"]) for p in priors}
    rels = []
    for k in keys:
        drv, etf, h = k
        other_h = [x for x in cfg["horizons_weeks"] if x != h]
        other = raw.get((drv, etf, other_h[0])) if other_h else None
        a = assess(raw[k], prior_sign.get((drv, etf)), qs[k], other, cfg)
        st = {**raw[k], "q_bh": _f(qs[k], 6)}
        rels.append({"driver": drv, "target": etf, "horizon_weeks": h, "level": a["level"],
                     "causal_evidence": a["causal_evidence"], "evidence": a["evidence"], "stats": _clean(st),
                     "prior_sign": prior_sign.get((drv, etf)), "observed_sign": a["observed_sign"]})
    rels.sort(key=lambda r: (-LEVELS.index(r["level"]), r["stats"].get("q_bh") or 1.0, r["driver"], r["target"], r["horizon_weeks"]))
    counters = {"causal_hypothesis": 0, "other": 0}
    for r in rels:
        key = "causal_hypothesis" if r["level"] == "causal_hypothesis" else "other"
        counters[key] += 1
        r["id"] = ("CAUSAL_HYPOTHESIS_" if key == "causal_hypothesis" else "REL_") + f"{counters[key]:03d}"
    return {"n_tested": len(keys), "fdr_q": cfg["fdr_q"], "relations": rels}


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (float, np.floating)):
        return _f(float(o), 6)
    if isinstance(o, np.integer):
        return int(o)
    return o


# ── PIT-Sektor-Tilts ─────────────────────────────────────────────────────────

def sector_tilts(ind: pd.DataFrame, px: pd.DataFrame, relations: list, dates, cfg: dict | None = None) -> pd.DataFrame:
    """-> DataFrame[date, sector_etf, tilt]. Für Stichtag t je Beziehung eine univariate Ridge
    (Ziel ~ Treiber), expandierend nur mit Zielen, deren Ende < t; Beitrag = Koeffizient × z(Treiber_t)
    / Std(Ziel im Training) (horizontneutral). Tilt = Mittel der Beiträge je Sektor-ETF (erwartete
    Sektor-Relativrendite, Rangordnung zählt). Kein Prior-Vorzeichen fließt ein."""
    tcfg = {**CP["tilt"], **(cfg or {})}
    cols = ["date", "sector_etf", "tilt"]
    dates = [pd.Timestamp(d) for d in dates if pd.Timestamp(d) in ind.index]
    prepared = []
    for r in relations:
        drv, etf, h = r["driver"], r["target"], int(r["horizon_weeks"])
        if drv not in ind.columns or etf not in px.columns:
            continue
        df = _pair_frame(ind, px, drv, etf, h, [])
        prepared.append((etf, df["x"].to_numpy(float), df["y"].to_numpy(float),
                         df["end"].to_numpy("datetime64[ns]"), df.index))
    rows = []
    for t in dates:
        tn = np.datetime64(t)
        per_sector: dict[str, list[float]] = {}
        for etf, x, y, end, idx in prepared:
            i = idx.get_loc(t)
            if not np.isfinite(x[i]):
                continue
            m = (end < tn) & np.isfinite(x) & np.isfinite(y)                # Ziel-Ende strikt vor t
            if m.sum() < tcfg["min_train"]:
                continue
            xs, ys = x[m], y[m]
            mu, sd, ym = xs.mean(), xs.std(), ys.mean()
            ysd = ys.std()
            if sd <= 0 or ysd <= 0:
                continue
            z = (xs - mu) / sd
            w = float(z @ (ys - ym) / (z @ z + tcfg["ridge_alpha"]))
            per_sector.setdefault(etf, []).append(w * (x[i] - mu) / sd / ysd)
        for etf, v in per_sector.items():
            rows.append((t, etf, float(np.mean(v))))
    return pd.DataFrame(rows, columns=cols)


def evaluate_tilts(tilts: pd.DataFrame, px: pd.DataFrame, grid, locked_from: pd.Timestamp, cfg: dict | None = None) -> dict:
    """Rang-IC (Spearman über Sektoren je Stichtag) der Tilts gegen die realisierte Sektor-Relativrendite
    (Horizont tilt.eval_horizon_weeks), nur Ziele mit Ende < Locked-Holdout; Monats-Block-Bootstrap."""
    full = {**CP, **(cfg or {})}
    tc = {**CP["tilt"], **(full.get("tilt") or {})}
    if tilts is None or tilts.empty:
        return {"n_dates": 0, "positive": False}
    fwd, _, end = _grid_returns(px, list(grid), tc["eval_horizon_weeks"])
    piv = tilts.pivot(index="date", columns="sector_etf", values="tilt")
    ics = {}
    for t, row in piv.iterrows():
        e_t = end.get(t)
        if t not in fwd.index or e_t is None or pd.isna(e_t) or e_t >= locked_from:
            continue
        d = pd.concat([row, fwd.loc[t]], axis=1).dropna()
        if len(d) >= tc["min_sectors"]:
            ics[t] = d.iloc[:, 0].rank().corr(d.iloc[:, 1].rank())
    s = pd.Series(ics).dropna()
    if s.empty:
        return {"n_dates": 0, "positive": False}
    bs = month_block_bootstrap(s, full["bootstrap_n"], full["bootstrap_seed"], tc["bootstrap_alpha"])
    lo = bs.get("lo")
    return {"n_dates": int(len(s)), "ic_mean": _f(float(s.mean())), "ci_lo": _f(lo), "ci_hi": _f(bs.get("hi")),
            "n_months": bs.get("n_months", 0), "positive": bool(lo is not None and lo > 0 and s.mean() > 0)}


def decide(relations: list[dict], tilt_eval: dict) -> tuple[str, str]:
    """Vorab-Regel (Protokoll causal_research.decision)."""
    lead = [r for r in relations if LEVELS.index(r["level"]) >= LEVELS.index("temporally_leading")]
    pred = [r for r in relations if LEVELS.index(r["level"]) >= LEVELS.index("predictive_relationship")]
    if len(lead) >= CP["decision"]["keep_min_leading"] and tilt_eval.get("positive"):
        return "KEEP", f"{len(lead)} Beziehung(en) temporally_leading nach BH und Tilt-Rang-IC-Untergrenze > 0"
    if pred:
        return "MODIFY", (f"{len(pred)} predictive_relationship-Beziehung(en); "
                          + ("Tilt-Rang-IC nicht signifikant positiv" if lead else "keine temporally_leading nach BH")
                          + " -> nur Beobachtung/Report")
    return "REJECT", "keine Beziehung überlebt BH auf dem OOS-Granger-Test"


# ── Lauf (CI, Netzwerk) ──────────────────────────────────────────────────────

def render_md(rep: dict) -> str:
    L = [f"# Causal Research ({rep['version']}) – {rep['generated']}", "",
         f"Verdikt: **{rep['verdict']}** – {rep['verdict_reason']}", "",
         f"Getestete Kombinationen: {rep['n_tested']} (Benjamini-Hochberg q={rep['fdr_q']}). "
         "`causal_evidence` ist nie automatisch vergeben (immer none). Priors sind vorab festgelegt.", ""]
    te = rep.get("tilt_eval") or {}
    L.append(f"Sektor-Tilts (PIT-Walk-Forward): Rang-IC {te.get('ic_mean')} · CI [{te.get('ci_lo')}, {te.get('ci_hi')}] · n={te.get('n_dates')}")
    L += ["", "| ID | Treiber | Ziel | h | Level | Prior | beob. | q(BH) | Lead/Lag | Stabil. | Repl. | Plaus. | Kausal-Konf. |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rep["relations"]:
        e, s = r["evidence"], r["stats"]
        L.append(f"| {r['id']} | {r['driver']} | {r['target']} | {r['horizon_weeks']} | {r['level']} | {r['prior_sign']} | "
                 f"{r['observed_sign']} | {s.get('q_bh')} | {_f(s.get('corr_lead'), 3)}/{_f(s.get('corr_lag'), 3)} | "
                 f"{e['stability']} | {e['replication']} | {e['economic_plausibility']} | {e['causal_confidence']} |")
    return "\n".join(L) + "\n"


def run(validate_now: bool = True) -> dict:
    from modules import ml_research as ml
    from modules import world_model as wm
    px = wm.fetch_market()
    arch = wm.load_archive()
    ind, _ = wm.build_world(px, arch, None)
    locked = ml.LOCKED_FROM
    res = analyze(ind, px, locked)
    sel = [r for r in res["relations"] if r["level"] in CP["tilt"]["tilt_levels"]]
    first = pd.Timestamp(year=CP["first_test_year"], month=1, day=1)
    dates = [d for d in ind.index if d >= first]
    tilts = sector_tilts(ind, px, sel, dates)
    te = evaluate_tilts(tilts, px, ind.index, locked)
    # Robustheit: ohne Auswahl (alle Prior-Paare, keine Selektion nach Ergebnis)
    te_all = evaluate_tilts(sector_tilts(ind, px, res["relations"], dates), px, ind.index, locked)
    verdict, why = decide(res["relations"], te)
    rep = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "version": CP["version"],
           "n_tested": res["n_tested"], "fdr_q": res["fdr_q"], "relations": res["relations"],
           "tilt_eval": te, "tilt_eval_selection_free": te_all,
           "selection_note": "Beziehungs-Auswahl nutzt OOS-Tests bis zum Locked-Holdout (leichter Selektionsbias); "
                             "tilt_eval_selection_free nutzt alle Prior-Paare ohne Auswahl.",
           "verdict": verdict, "verdict_reason": why}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(rep, indent=1, default=str, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(render_md(rep), encoding="utf-8")
    return rep


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    print(render_md(run()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

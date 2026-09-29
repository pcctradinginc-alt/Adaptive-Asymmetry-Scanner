"""
modules/factor_monitor.py – Faktor-Performance-Datenbank, Walk-Forward-
Bewertung, Factor Decay, Regime-Analyse und adaptive SHADOW-Gewichte.

    python -m modules.factor_monitor            (monatlich im Workflow)

Ändert NIE Produktion (Scores, Gates, PPO, Gewichte). Ergebnisse:
  outputs/research/factor_report.json / .md      aktueller Befund je Feature
  outputs/research/factor_performance.jsonl      Faktor-Performance-Datenbank
                                                 (ein Snapshot je Lauf/Feature)
  outputs/research/factor_weights_shadow.json    adaptive Gewichte (nur Shadow)

Grundsätze (gegen Overfitting):
  * Keine zufälligen Splits: Walk-Forward über Kalendermonate, Gewichte für
    Monat m ausschließlich aus Daten mit Entry-Datum < Monat m, und nur aus
    Outcomes, die zum Monatsbeginn bereits realisiert waren.
  * Effektive Stichprobe = Anzahl unabhängiger Entry-Tage, nicht Zeilen.
  * Shrinkage: IC_shrunk = IC · n_eff / (n_eff + PRIOR_N) → neutral bei wenig
    Daten; Gewicht 0 unter MIN_EFF_N oder wenn das 90%-KI die Null enthält.
  * Decay: EWMA-IC über Monats-ICs (Halbwertszeit DECAY_HALFLIFE_M) statt
    Gesamt-IC → nachlassende Prognosekraft senkt das Gewicht automatisch.
  * Regime: IC je Regime nur berichtet, Regime-Gewichte erst mit >= MIN_REGIME_N
    Tagen je Regime und signifikantem Unterschied (z-Test der Fisher-z).
  * Outcomes getrennt: `net` (Options-P&L inkl. Spread/Prämie, history.json)
    vs. `gross` (Underlying-Forward-Return, Ledger) → Kennzeichnung
    "nur vor Kosten profitabel".
"""

from __future__ import annotations

import json
import logging
import math
import statistics
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

log = logging.getLogger(__name__)

HISTORY_PATH = Path("outputs/history.json")
LEDGER_DIR = Path("outputs/candidate_ledger")
REPORTS_DIR = Path("outputs/daily_reports")
OUT_DIR = Path("outputs/research")

PRIOR_N = 60                 # Shrinkage-Stärke (Pseudo-Tage bei IC=0)
MIN_EFF_N = 30               # Mindestzahl unabhängiger Tage für ein Gewicht != 0
MIN_REGIME_N = 20
DECAY_HALFLIFE_M = 3.0       # Monate
MIN_COVERAGE = 0.2
REDUNDANT_RHO = 0.8
LEAK_IC = 0.5                # |Rank-IC| darüber bei n>=30 ist verdächtig
LEAK_TOKENS = ("ret", "outcome", "mfe", "mae", "exit", "close", "pnl", "realized", "return", "peak")
PSI_DRIFT = 0.25
Z90 = 1.645
LEDGER_HORIZONS = (1, 5, 20, 45, 60, 120)


# ── Statistik-Helfer ─────────────────────────────────────────────────────────

def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _ranks(v: list[float]) -> list[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0
        i = j + 1
    return r


def pearson(x: list[float], y: list[float]) -> float | None:
    n = len(x)
    if n < 3:
        return None
    mx, my = statistics.fmean(x), statistics.fmean(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx == 0 or sy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def spearman(x: list[float], y: list[float]) -> float | None:
    return pearson(_ranks(x), _ranks(y)) if len(x) >= 3 else None


def fisher_ci(r: float | None, n_eff: int, z: float = Z90) -> tuple[float | None, float | None]:
    if r is None or n_eff < 4:
        return None, None
    r = max(min(r, 0.999), -0.999)
    f = math.atanh(r)
    se = 1.0 / math.sqrt(n_eff - 3)
    return math.tanh(f - z * se), math.tanh(f + z * se)


def shrink(ic: float | None, n_eff: int, prior_n: int = PRIOR_N) -> float:
    if ic is None or n_eff <= 0:
        return 0.0
    return ic * n_eff / (n_eff + prior_n)


def ewma(values: list[float], halflife: float = DECAY_HALFLIFE_M) -> float | None:
    """Ältester Wert zuerst; jüngere Werte gewichtet stärker."""
    if not values:
        return None
    lam = 0.5 ** (1.0 / halflife)
    num = den = 0.0
    for age, v in enumerate(reversed(values)):
        w = lam ** age
        num += w * v
        den += w
    return num / den


def psi(expected: list[float], actual: list[float], bins: int = 10) -> float | None:
    """Population Stability Index zwischen Referenz- und aktueller Verteilung."""
    if len(expected) < 20 or len(actual) < 20:
        return None
    qs = sorted(expected)
    edges = [qs[int(len(qs) * i / bins)] for i in range(1, bins)]

    def hist(v):
        c = [0] * bins
        for x in v:
            c[sum(1 for e in edges if x > e)] += 1
        return [max(k / len(v), 1e-4) for k in c]
    e, a = hist(expected), hist(actual)
    return sum((ai - ei) * math.log(ai / ei) for ai, ei in zip(a, e))


def perf_stats(returns: list[float]) -> dict:
    """Kennzahlen einer Folge von Trade-/Perioden-Returns (chronologisch)."""
    r = [x for x in returns if _finite(x)]
    if not r:
        return {"n": 0}
    mean = statistics.fmean(r)
    sd = statistics.pstdev(r) if len(r) > 1 else 0.0
    downside = [x for x in r if x < 0]
    dsd = math.sqrt(sum(x * x for x in downside) / len(r)) if downside else 0.0
    gains, losses = sum(x for x in r if x > 0), -sum(x for x in r if x < 0)
    eq, peak, mdd = 1.0, 1.0, 0.0
    for x in r:
        eq *= (1.0 + max(x, -1.0) * 0.1)     # 10 % Einsatz je Trade für die DD-Kurve
        peak = max(peak, eq)
        mdd = min(mdd, eq / peak - 1.0)
    return {"n": len(r), "hit_rate": round(sum(1 for x in r if x > 0) / len(r), 4),
            "mean": round(mean, 4), "median": round(statistics.median(r), 4),
            "sharpe_per_trade": round(mean / sd, 4) if sd > 0 else None,
            "sortino_per_trade": round(mean / dsd, 4) if dsd > 0 else None,
            "profit_factor": round(gains / losses, 4) if losses > 0 else None,
            "max_drawdown_10pct_sizing": round(mdd, 4)}


# ── Daten laden ──────────────────────────────────────────────────────────────

def _num_features(feats: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in (feats or {}).items():
        if _finite(v):
            out[prefix + k] = float(v)
        elif isinstance(v, bool):
            out[prefix + k] = 1.0 if v else 0.0
    return out


def load_events(history_path: Path = HISTORY_PATH, ledger_dir: Path = LEDGER_DIR) -> list[dict]:
    """Einheitliche Ereignisliste: {date, ticker, source, features, outcomes}.
    outcomes['net'] = Options-P&L (inkl. Spread/Prämie) bei geschlossenen und
    Schatten-Trades; outcomes['gross_<h>d'] = richtungsbereinigter Underlying-
    Return aus dem Ledger; outcomes['net_45d'] = real_strat_ret_45d (Ledger)."""
    events = []
    try:
        h = json.loads(Path(history_path).read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        log.warning(f"factor_monitor: history.json nicht lesbar: {e}")
        h = {}
    for src in ("closed_trades", "shadow_trades"):
        for t in h.get(src) or []:
            out = t.get("outcome")
            d = str(t.get("entry_date") or "")[:10]
            if not d or not _finite(out):
                continue
            if not t.get("close_date"):
                continue            # nur realisierte Outcomes (kein Zwischenstand)
            events.append({"date": d, "ticker": t.get("ticker"), "source": src,
                           "outcome_date": str(t.get("close_date") or "")[:10] or None,
                           "features": _num_features(t.get("features")),
                           "outcomes": {"net": float(out)}})
    for f in sorted(Path(ledger_dir).glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            oc = r.get("outcomes") or {}
            outcomes = {f"gross_{h_}d": float(oc[f"ret_{h_}d"]) for h_ in LEDGER_HORIZONS
                        if _finite(oc.get(f"ret_{h_}d"))}
            if _finite(oc.get("real_strat_ret_45d")):
                outcomes["net_45d"] = float(oc["real_strat_ret_45d"])
            feats = _num_features(r.get("features"))
            feats.update(_num_features((r.get("external") or {}).get("primitives"), "ext."))
            events.append({"date": str(r.get("date"))[:10], "ticker": r.get("ticker"),
                           "source": "ledger", "outcome_date": None,
                           "status": r.get("status"), "features": feats, "outcomes": outcomes,
                           "sector": (r.get("features") or {}).get("sector")})
    events.sort(key=lambda e: e["date"])
    return events


def load_regimes(reports_dir: Path = REPORTS_DIR) -> dict[str, dict]:
    """Regime je Tag aus den (damals geschriebenen) Tagesreports: VIX-Niveau.
    Weitere Regime (Trend, Zinsen, Inflation) werden ergänzt, sobald sie
    PIT-sauber je Tag im Ledger stehen (external.states)."""
    out = {}
    for f in sorted(Path(reports_dir).glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        vix = (d.get("stats") or {}).get("vix")
        if _finite(vix):
            out[f.stem] = {"vix": "high_vol" if vix >= 20 else "low_vol"}
    return out


def indpro_regimes(dates: list[str], archive_root: str = "outputs/external_data") -> dict[str, dict]:
    """Expansion/Kontraktion je Tag aus US-Industrieproduktion (ALFRED-
    Vintages): nur Werte mit available_at <= Tag (PIT, kein Restatement-Bias),
    Jahresveränderung der jüngsten damals bekannten Periode."""
    try:
        from modules.external.archive import ExternalArchive
        obs = [o for o in ExternalArchive(archive_root).load("fred_us_macro")
               if o.metric == "us_indpro" and _finite(o.value)]
    except Exception as e:  # noqa: BLE001
        log.warning(f"factor_monitor: INDPRO-Regime nicht verfügbar: {e}")
        return {}
    out = {}
    for d in sorted(set(dates)):
        known: dict = {}
        for o in obs:
            if o.available_at.date().isoformat() <= d:
                prev = known.get(o.observation_time)
                if prev is None or o.available_at > prev.available_at:
                    known[o.observation_time] = o
        if not known:
            continue
        last = max(known)
        yago = [t for t in known if t.year == last.year - 1 and t.month == last.month]
        if yago:
            yoy = known[last].value / known[yago[0]].value - 1.0
            out[d] = {"cycle": "expansion" if yoy > 0 else "contraction"}
    return out


def market_regimes(dates: list[str]) -> dict[str, dict]:
    """Markt-Regime je Tag aus Schlusskursen des VORTAGS (PIT): Trend (SPY vs.
    SMA200, Seitwärts bei |60d-Return| < 3 %), Zinsniveau (^TNX >= 4 %),
    Zinskurve (^TNX - ^IRX < 0 = invertiert). Netzwerk nötig; Fehler -> {}."""
    if not dates:
        return {}
    try:
        import yfinance as yf
        start = date.fromordinal(date.fromisoformat(min(dates)).toordinal() - 400).isoformat()
        closes = {}
        for sym in ("SPY", "^TNX", "^IRX"):
            h = yf.Ticker(sym).history(start=start)["Close"]
            closes[sym] = [(ts.date().isoformat(), float(v)) for ts, v in h.items() if _finite(float(v))]
    except Exception as e:  # noqa: BLE001
        log.warning(f"factor_monitor: Markt-Regime nicht verfügbar: {e}")
        return {}
    out = {}
    for d in dates:
        spy = [v for t, v in closes.get("SPY", []) if t < d]
        tnx = [v for t, v in closes.get("^TNX", []) if t < d]
        irx = [v for t, v in closes.get("^IRX", []) if t < d]
        reg = {}
        if len(spy) >= 200:
            r60 = spy[-1] / spy[-61] - 1.0
            reg["trend"] = ("sideways" if abs(r60) < 0.03 else
                            "bull" if spy[-1] > statistics.fmean(spy[-200:]) else "bear")
        if tnx:
            reg["rates"] = "high_rates" if tnx[-1] >= 4.0 else "low_rates"
        if tnx and irx:
            reg["curve"] = "inverted" if tnx[-1] - irx[-1] < 0 else "normal"
        if reg:
            out[d] = reg
    return out


def merge_regimes(*maps: dict) -> dict[str, dict]:
    out: dict = defaultdict(dict)
    for m in maps:
        for d, r in (m or {}).items():
            out[d].update(r)
    return dict(out)


# ── Kernbewertung ────────────────────────────────────────────────────────────

def _pairs(events, feature, outcome_key):
    xs, ys, ds = [], [], []
    for e in events:
        x, y = e["features"].get(feature), e["outcomes"].get(outcome_key)
        if x is not None and y is not None:
            xs.append(x), ys.append(y), ds.append(e["date"])
    return xs, ys, ds


def evaluate_feature(events: list[dict], feature: str, outcome_key: str,
                     regimes: dict | None = None) -> dict:
    relevant = [e for e in events if outcome_key in e["outcomes"]]
    xs, ys, ds = _pairs(relevant, feature, outcome_key)
    n, n_eff = len(xs), len(set(ds))
    res = {"feature": feature, "outcome": outcome_key, "n": n, "n_eff_dates": n_eff,
           "coverage": round(n / len(relevant), 4) if relevant else 0.0}
    if n < 3 or len(set(xs)) < 2:
        res["status"] = "dead"
        return res
    ic, ric = pearson(xs, ys), spearman(xs, ys)
    lo, hi = fisher_ci(ric, n_eff)
    by_month = defaultdict(list)
    for x, y, d in zip(xs, ys, ds):
        by_month[d[:7]].append((x, y))
    monthly = []
    for m in sorted(by_month):
        pts = by_month[m]
        if len(pts) >= 8 and len({p[0] for p in pts}) > 1:
            r = spearman([p[0] for p in pts], [p[1] for p in pts])
            if r is not None:
                monthly.append((m, r))
    mic = [r for _, r in monthly]
    ewma_ic = ewma(mic) if mic else None
    recent_ic = statistics.fmean(mic[-3:]) if len(mic) >= 4 else None
    prior_ic = statistics.fmean(mic[:-3]) if len(mic) >= 4 else None
    ic_ir = (statistics.fmean(mic) / statistics.pstdev(mic)) if len(mic) >= 3 and statistics.pstdev(mic) > 0 else None
    sign_consistency = (sum(1 for r in mic if ric and r * ric > 0) / len(mic)) if mic and ric else None

    # Terzil-Spread (Top minus Bottom) und Performance des Top-Terzils
    order = sorted(range(n), key=lambda i: xs[i])
    k = max(n // 3, 1)
    top = [ys[i] for i in order[-k:]]
    bot = [ys[i] for i in order[:k]]
    res.update({
        "ic": _r(ic), "rank_ic": _r(ric), "rank_ic_ci90": [_r(lo), _r(hi)],
        "monthly_rank_ic": [[m, _r(r)] for m, r in monthly], "ewma_rank_ic": _r(ewma_ic),
        "ic_ir": _r(ic_ir), "sign_consistency": _r(sign_consistency),
        "recent_3m_rank_ic": _r(recent_ic), "prior_rank_ic": _r(prior_ic),
        "tercile_spread": _r(statistics.fmean(top) - statistics.fmean(bot)),
        "top_tercile": perf_stats(top), "all": perf_stats(ys),
    })
    if regimes:
        per = defaultdict(lambda: ([], [], set()))
        for x, y, d in zip(xs, ys, ds):
            for dim, val in (regimes.get(d) or {}).items():
                a, b, s = per[f"{dim}={val}"]
                a.append(x), b.append(y), s.add(d)
        res["regimes"] = {name: {"n": len(a), "n_eff_dates": len(s), "rank_ic": _r(spearman(a, b))}
                          for name, (a, b, s) in sorted(per.items()) if len(a) >= 5}
        res["regime_dependent"] = _regime_dependent(res["regimes"])
    return res


def _regime_dependent(reg: dict) -> bool:
    items = [(v["rank_ic"], v["n_eff_dates"]) for v in reg.values()
             if v.get("rank_ic") is not None and v["n_eff_dates"] >= MIN_REGIME_N]
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            (r1, n1), (r2, n2) = items[i], items[j]
            z = (math.atanh(max(min(r1, .999), -.999)) - math.atanh(max(min(r2, .999), -.999))) \
                / math.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
            if abs(z) > 1.96:
                return True
    return False


def _r(x, nd: int = 4):
    return round(x, nd) if _finite(x) else None


def classify(res: dict, feature: str) -> list[str]:
    tags = []
    if res.get("status") == "dead" or res.get("coverage", 0) < MIN_COVERAGE:
        tags.append("dead")
        return tags
    lo, hi = res.get("rank_ic_ci90", [None, None])
    ric = res.get("rank_ic")
    if any(t in feature.lower() for t in LEAK_TOKENS) or \
            (ric is not None and abs(ric) > LEAK_IC and res.get("n", 0) >= 30):
        tags.append("potential_leakage")
    if res.get("sign_consistency") is not None and len(res.get("monthly_rank_ic", [])) >= 4 \
            and res["sign_consistency"] < 0.6:
        tags.append("unstable")
    if lo is not None and (lo > 0 or hi < 0) and res.get("n_eff_dates", 0) >= MIN_EFF_N \
            and "unstable" not in tags:
        tags.append("high_value")
    ew = res.get("ewma_rank_ic")
    rec, pri = res.get("recent_3m_rank_ic"), res.get("prior_rank_ic")
    decay_ewma = ric and ew is not None and abs(ric) > 0.05 and (ew * ric < 0 or abs(ew) < 0.5 * abs(ric))
    decay_recent = (rec is not None and pri is not None and abs(pri) > 0.05
                    and (rec * pri < 0 or abs(rec) < 0.5 * abs(pri)))
    if decay_ewma or decay_recent:
        tags.append("decaying")
    if res.get("regime_dependent"):
        tags.append("regime_dependent")
    return tags


def redundancy(events: list[dict], features: list[str]) -> list[dict]:
    pairs = []
    for i, a in enumerate(features):
        for b in features[i + 1:]:
            xs, ys = [], []
            for e in events:
                va, vb = e["features"].get(a), e["features"].get(b)
                if va is not None and vb is not None:
                    xs.append(va), ys.append(vb)
            if len(xs) >= 20 and len(set(xs)) > 1 and len(set(ys)) > 1:
                rho = spearman(xs, ys)
                if rho is not None and abs(rho) >= REDUNDANT_RHO:
                    pairs.append({"a": a, "b": b, "rho": _r(rho), "n": len(xs)})
    return pairs


def feature_drift(events: list[dict], feature: str, recent_days: int = 30) -> float | None:
    vals = [(e["date"], e["features"][feature]) for e in events if feature in e["features"]]
    if not vals:
        return None
    last = max(d for d, _ in vals)
    cutoff = date.fromordinal(date.fromisoformat(last).toordinal() - recent_days).isoformat()
    ref = [v for d, v in vals if d < cutoff]
    cur = [v for d, v in vals if d >= cutoff]
    p = psi(ref, cur)
    return _r(p)


# ── Walk-Forward adaptive Gewichte ───────────────────────────────────────────

def weights_from(events: list[dict], features: list[str], outcome_key: str) -> dict[str, float]:
    """Shrinkage-Gewichte aus EWMA-Rank-IC (Decay); 0 ohne ausreichende
    Evidenz. Summe der Beträge normiert auf 1 (nur Richtung + relative Stärke)."""
    w = {}
    for f in features:
        r = evaluate_feature(events, f, outcome_key)
        n_eff = r.get("n_eff_dates", 0)
        lo, hi = r.get("rank_ic_ci90", [None, None]) if r.get("status") != "dead" else (None, None)
        base = r.get("ewma_rank_ic") if r.get("ewma_rank_ic") is not None else r.get("rank_ic")
        rec = r.get("recent_3m_rank_ic")
        if base is not None and rec is not None and (rec * base < 0 or abs(rec) < abs(base)):
            base = rec                      # Decay: konservativ den jüngsten IC nehmen
        if n_eff < MIN_EFF_N or lo is None or (lo <= 0 <= hi):
            w[f] = 0.0
        else:
            w[f] = shrink(base, n_eff)
    s = sum(abs(v) for v in w.values())
    return {k: (v / s if s else 0.0) for k, v in w.items()}


def _zscores(events, features):
    stats = {}
    for f in features:
        v = [e["features"][f] for e in events if f in e["features"]]
        if len(v) >= 2 and statistics.pstdev(v) > 0:
            stats[f] = (statistics.fmean(v), statistics.pstdev(v))
    return stats


def composite(e: dict, weights: dict, zstats: dict) -> float | None:
    s, used = 0.0, 0
    for f, w in weights.items():
        if w == 0 or f not in zstats or f not in e["features"]:
            continue
        m, sd = zstats[f]
        s += w * max(min((e["features"][f] - m) / sd, 3.0), -3.0)   # winsorisiert ±3σ
        used += 1
    return s if used else None


def walk_forward(events: list[dict], features: list[str], outcome_key: str,
                 min_train_months: int = 2) -> dict:
    """Expandierendes Fenster über Monate. Training nur mit Ereignissen, deren
    Entry < Testmonat UND deren Outcome zum Testbeginn realisiert war
    (outcome_date < Testmonat, sofern bekannt) → kein Look-ahead."""
    rel = [e for e in events if outcome_key in e["outcomes"]]
    months = sorted({e["date"][:7] for e in rel})
    folds, oos_adaptive, oos_equal = [], [], []
    for i, m in enumerate(months):
        if i < min_train_months:
            continue
        start = f"{m}-01"
        train = [e for e in rel if e["date"] < start and (e.get("outcome_date") is None
                                                          or e["outcome_date"] < start)]
        test = [e for e in rel if e["date"][:7] == m]
        if len(train) < 20 or len(test) < 5:
            continue
        w = weights_from(train, features, outcome_key)
        eq = {f: 1.0 / len(features) for f in features}
        zs = _zscores(train, features)
        for ws, sink in ((w, oos_adaptive), (eq, oos_equal)):
            for e in test:
                c = composite(e, ws, zs)
                if c is not None:
                    sink.append((c, e["outcomes"][outcome_key], m))
        folds.append({"test_month": m, "n_train": len(train), "n_test": len(test),
                      "weights": {k: _r(v) for k, v in w.items() if v}})

    def summarize(pts):
        if len(pts) < 10:
            return {"n": len(pts)}
        ric = spearman([p[0] for p in pts], [p[1] for p in pts])
        top = [p[1] for p in pts if p[0] > 0]
        return {"n": len(pts), "oos_rank_ic": _r(ric), "positive_score": perf_stats(top),
                "all": perf_stats([p[1] for p in pts])}
    abstained = sum(1 for f in folds if not f["weights"])
    return {"outcome": outcome_key, "folds": folds,
            "adaptive_abstained_folds": abstained,
            "adaptive_note": ("Enthaltung: keine Faktor-Evidenz (n_eff/KI) -> kein Score, kein Trade"
                              if folds and abstained == len(folds) else None),
            "adaptive": summarize(oos_adaptive), "equal_weight": summarize(oos_equal)}


# ── Lauf ─────────────────────────────────────────────────────────────────────

def run(history_path: Path = HISTORY_PATH, ledger_dir: Path = LEDGER_DIR,
        reports_dir: Path = REPORTS_DIR, out_dir: Path = OUT_DIR,
        today: date | None = None, with_market_regimes: bool = False) -> dict:
    today = today or datetime.utcnow().date()
    events = load_events(history_path, ledger_dir)
    dates = sorted({e["date"] for e in events})
    regimes = merge_regimes(load_regimes(reports_dir),
                            market_regimes(dates) if with_market_regimes else {},
                            indpro_regimes(dates) if with_market_regimes else {})
    outcome_keys = sorted({k for e in events for k in e["outcomes"]})
    report = {"generated": today.isoformat(), "n_events": len(events),
              "sources": dict(_count(e["source"] for e in events)),
              "outcomes": {}, "note": "SHADOW: keine Produktionswirkung. Gewichte nur Forschung.",
              "regime_coverage": dict(_count(f"{k}={v}" for r in regimes.values() for k, v in r.items())),
              "regimes_missing": ["inflation (keine PIT-Inflationsreihe im Archiv)",
                                  "credit_spreads", "dollar", "oil", "liquidity"]}
    db_rows = []
    weights_out = {}
    for ok in outcome_keys:
        rel = [e for e in events if ok in e["outcomes"]]
        feats = sorted({f for e in rel for f in e["features"]})
        feats = [f for f in feats
                 if sum(1 for e in rel if f in e["features"]) >= max(10, MIN_COVERAGE * len(rel))]
        per = {}
        for f in feats:
            r = evaluate_feature(rel, f, ok, regimes)
            r["tags"] = classify(r, f)
            r["psi_drift_30d"] = feature_drift(rel, f)
            if r["psi_drift_30d"] is not None and r["psi_drift_30d"] > PSI_DRIFT:
                r["tags"].append("data_drift")
            per[f] = r
            db_rows.append({"run_date": today.isoformat(), "outcome": ok, "feature": f,
                            **{k: r.get(k) for k in ("n", "n_eff_dates", "coverage", "ic", "rank_ic",
                                                     "rank_ic_ci90", "ewma_rank_ic", "ic_ir",
                                                     "tercile_spread", "tags")},
                            "regimes": r.get("regimes")})
        usable = [f for f in feats if "dead" not in per[f]["tags"]
                  and "potential_leakage" not in per[f]["tags"]]
        wf = walk_forward(rel, usable, ok) if usable else {"folds": []}
        weights_out[ok] = weights_from(rel, usable, ok) if usable else {}
        report["outcomes"][ok] = {
            "n": len(rel), "n_eff_dates": len({e["date"] for e in rel}),
            "features": per, "redundant_pairs": redundancy(rel, usable),
            "walk_forward": wf,
            "summary": {tag: sorted(f for f, r in per.items() if tag in r["tags"])
                        for tag in ("high_value", "dead", "unstable", "decaying", "regime_dependent",
                                    "potential_leakage", "data_drift")},
        }
    report["cost_check"] = _cost_check(report)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "factor_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (out_dir / "factor_weights_shadow.json").write_text(json.dumps(
        {"generated": today.isoformat(), "weights": weights_out,
         "method": f"EWMA-Rank-IC (HWZ {DECAY_HALFLIFE_M} M) · n/(n+{PRIOR_N}); 0 bei n_eff<{MIN_EFF_N} "
                   "oder 90%-KI inkl. 0", "production_use": False}, indent=2))
    with open(out_dir / "factor_performance.jsonl", "a", encoding="utf-8") as fh:
        for row in db_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out_dir / "factor_report.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def _count(it):
    c = defaultdict(int)
    for x in it:
        c[x] += 1
    return c


def _cost_check(report: dict) -> list[dict]:
    """Features mit positivem Brutto-Terzil-Spread (Underlying), aber nicht-
    positivem Netto-Spread (Optionsstrategie inkl. Kosten, net_45d)."""
    gross = (report["outcomes"].get("gross_45d") or {}).get("features", {})
    net = (report["outcomes"].get("net_45d") or {}).get("features", {})
    out = []
    for f, g in gross.items():
        n = net.get(f)
        if n and (g.get("tercile_spread") or 0) > 0 and (n.get("tercile_spread") or 0) <= 0:
            out.append({"feature": f, "gross_spread": g["tercile_spread"],
                        "net_spread": n["tercile_spread"], "flag": "only_profitable_before_costs"})
    return out


def render_markdown(rep: dict) -> str:
    lines = [f"# Faktor-Report {rep['generated']}", "", rep["note"], "",
             f"Ereignisse: {rep['n_events']} ({rep['sources']})", ""]
    for ok, block in rep["outcomes"].items():
        lines += [f"## Outcome `{ok}` (n={block['n']}, unabhängige Tage={block['n_eff_dates']})", "",
                  "| Feature | n | Tage | Rank-IC | 90%-KI | EWMA-IC | Terzil-Spread | Tags |",
                  "|---|---|---|---|---|---|---|---|"]
        for f, r in sorted(block["features"].items(), key=lambda kv: -abs(kv[1].get("rank_ic") or 0)):
            lines.append(f"| {f} | {r.get('n')} | {r.get('n_eff_dates')} | {r.get('rank_ic')} | "
                         f"{r.get('rank_ic_ci90')} | {r.get('ewma_rank_ic')} | {r.get('tercile_spread')} | "
                         f"{', '.join(r.get('tags', []))} |")
        wf = block.get("walk_forward", {})
        lines += ["", f"Walk-Forward OOS: adaptiv {wf.get('adaptive')} vs. gleichgewichtet "
                      f"{wf.get('equal_weight')}", ""]
        if block.get("redundant_pairs"):
            lines += ["Redundant: " + "; ".join(f"{p['a']}~{p['b']} (ρ={p['rho']})"
                                                for p in block["redundant_pairs"]), ""]
    if rep.get("cost_check"):
        lines += ["## Nur vor Kosten profitabel", ""] + \
                 [f"- {c['feature']}: brutto {c['gross_spread']} / netto {c['net_spread']}"
                  for c in rep["cost_check"]]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    r = run(with_market_regimes=True)
    print(json.dumps({k: {"n": v["n"], "summary": v["summary"],
                          "wf_adaptive": v["walk_forward"].get("adaptive"),
                          "wf_equal": v["walk_forward"].get("equal_weight")}
                      for k, v in r["outcomes"].items()}, indent=2, ensure_ascii=False))

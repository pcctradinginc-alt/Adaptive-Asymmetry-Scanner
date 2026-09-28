"""
modules/alpha_discovery.py – systematische, snooping-feste Alpha-Suche.

Datenbasis: Candidate Ledger. JEDER analysierte Kandidat (auch verworfene)
bekommt richtungsbereinigte Underlying-Renditen nach 5/20/45/120 Tagen
(outcomes.ret_{h}d) -> um ein Vielfaches mehr Beobachtungen als echte Trades.

Verfahren (alle Schwellen a priori, nie aus den Ergebnissen abgeleitet):
  1. Merkmale je Zeile: numerische features.*, external.primitives.*,
     kategoriale Merkmale (Richtung, Sektor, Prescreen-Kategorie, externe
     Zustände/Relation/Exposure, Gate-Status). Outcome-nahe Schlüssel sind
     ausgeschlossen (LEAK_TOKENS).
  2. Zielgröße: ret_{h}d, TAGES-BEREINIGT (minus Tagesmittel aller
     Kandidaten desselben Tages). Das entfernt Markt- und Kalendereffekte --
     genau die Vermengung, die die Fracht-Rückwärtsprobe unbrauchbar machte.
     Merkmale, die innerhalb eines Tages konstant sind (Makro/extern), werden
     stattdessen über Tage getestet (Tagesmittel, n = Anzahl Tage).
  3. Querschnitt: Rang-IC je Tag, Mittel über Tage, t = IC / SE (Tage sind
     die unabhängigen Einheiten, Fama-MacBeth-Logik). Kategorial: Tagesmittel
     der bereinigten Rendite der Kategorie, t über Tage.
  4. Chronologische Teilung: die ersten 60 % der Tage = Entdeckung, die
     letzten 40 % = interne Bestätigung. Entdeckung: Benjamini-Hochberg über
     ALLE Tests (q <= 0.10). Bestätigung: gleiches Vorzeichen, p <= 0.10
     (einseitig), ökonomische Mindestgröße (Terzil-Spread).
  5. Treffer -> eingefrorener Challenger-Vorschlag (Hash-Lock). Registriert
     wird er mit start_date = Folgetag: die ENDGÜLTIGE Bestätigung passiert
     ausschließlich auf zukünftigen Daten (challenger.py, Alpha-Spending,
     menschliche Promotion). Deckel: max. MAX_NEW_PER_RUN je Lauf.

Ausgabe: outputs/research/alpha_discovery.{json,md}. EXPLORATORY bis zur
prospektiven Bestätigung.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import statistics
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger(__name__)

LEDGER_DIR = Path("outputs/candidate_ledger")
OUT_DIR = Path("outputs/research")
AUTO_PROPOSALS_PATH = Path("config/challenger_proposals_auto.yaml")

HORIZONS = (20, 45)                 # Primär-Horizonte der Suche
MIN_ROWS = 300                      # Mindestdaten für einen Lauf
MIN_DATES = 30
MIN_ROWS_PER_DATE = 3               # Querschnitt braucht mehrere Kandidaten/Tag
MIN_LEVEL_ROWS = 30                 # Kategorie-Level
MIN_LEVEL_DATES = 10
DISCOVERY_FRACTION = 0.6
FDR_Q = 0.10
HOLDOUT_P = 0.10
MIN_SPREAD = {5: 0.01, 20: 0.02, 45: 0.03, 120: 0.05}   # Terzil-Spread (bereinigt)
MAX_NEW_PER_RUN = 2
LEAK_TOKENS = ("ret", "outcome", "mfe", "mae", "exit", "close", "pnl", "realized")
CATEGORICAL_KEYS = (
    "direction", "status", "reject_reason",
    "features.sector", "features.prescreen_category", "features.dealer_gamma_sign",
    "features.insider_cluster", "features.strategy_source", "features.rl_robust_action",
    "external.relation.relation", "external.states.us_freight_state",
    "external.states.eu_freight_state", "external.states.global_freight_state",
    "external.states.global_maritime_state",
    "external.ticker_exposure.road_freight_relevance",
    "external.ticker_exposure.maritime_relevance", "external.ticker_exposure.weather_relevance",
)


# ── Laden ────────────────────────────────────────────────────────────────────

def _get(d, path):
    for k in path.split("."):
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _leaky(name: str) -> bool:
    low = name.lower()
    return any(tok in low for tok in LEAK_TOKENS)


def load_rows(ledger_dir: Path = LEDGER_DIR, horizon: int = 20) -> list[dict]:
    """Nur Zeilen mit reifer Zielgröße und Richtung (= Deep-Analyse erreicht)."""
    rows = []
    for f in sorted(Path(ledger_dir).glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            y = _get(r, f"outcomes.ret_{horizon}d")
            if y is None or r.get("direction") not in ("BULLISH", "BEARISH") or not r.get("date"):
                continue
            num = {}
            for k, v in (r.get("features") or {}).items():
                if isinstance(v, bool) or not isinstance(v, (int, float)) or _leaky(k):
                    continue
                if math.isfinite(float(v)):
                    num[f"features.{k}"] = float(v)
            for k, v in ((r.get("external") or {}).get("primitives") or {}).items():
                if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)):
                    num[f"external.primitives.{k}"] = float(v)
            cat = {}
            for k in CATEGORICAL_KEYS:
                v = _get(r, k)
                if v is not None and v != "":
                    cat[k] = str(v)
            rows.append({"date": str(r["date"])[:10], "y": float(y), "num": num, "cat": cat})
    return rows


# ── Statistik ────────────────────────────────────────────────────────────────

def _demean_by_date(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["date"]].append(r["y"])
    means = {d: statistics.fmean(v) for d, v in by.items()}
    for r in rows:
        r["y_adj"] = r["y"] - means[r["date"]]
    return rows


def _t_p(values: list[float]) -> tuple[float | None, float | None]:
    """t-Statistik des Mittels über unabhängige Einheiten + zweiseitiges p."""
    n = len(values)
    if n < 3:
        return None, None
    sd = statistics.stdev(values)
    if sd == 0:
        return None, None
    t = statistics.fmean(values) / (sd / math.sqrt(n))
    try:
        from scipy import stats
        p = float(2 * stats.t.sf(abs(t), n - 1))
    except Exception:
        p = math.erfc(abs(t) / math.sqrt(2))
    return t, p


def _spearman(x, y):
    try:
        from scipy import stats
        r = stats.spearmanr(x, y).statistic
        return None if r is None or not math.isfinite(r) else float(r)
    except Exception:
        return None


def _is_date_constant(rows, key) -> bool:
    by = defaultdict(set)
    for r in rows:
        if key in r["num"]:
            by[r["date"]].add(round(r["num"][key], 9))
    multi = [d for d, v in by.items() if len(v) > 1]
    return len(multi) <= 0.2 * max(len(by), 1)


def test_numeric(rows, key) -> dict | None:
    """Querschnitt (Rang-IC je Tag) oder Zeitreihe (über Tage) je nach Merkmal."""
    if _is_date_constant(rows, key):
        by = defaultdict(list)
        for r in rows:
            if key in r["num"]:
                by[r["date"]].append((r["num"][key], r["y"]))
        pts = [(statistics.fmean(a for a, _ in v), statistics.fmean(b for _, b in v)) for v in by.values()]
        if len(pts) < 10:
            return None
        rho = _spearman([a for a, _ in pts], [b for _, b in pts])
        if rho is None:
            return None
        n = len(pts)
        t = rho * math.sqrt((n - 2) / max(1e-12, 1 - rho ** 2))
        try:
            from scipy import stats
            p = float(2 * stats.t.sf(abs(t), n - 2))
        except Exception:
            p = math.erfc(abs(t) / math.sqrt(2))
        return {"kind": "time_series", "stat": rho, "t": t, "p": p, "n_units": n}
    by = defaultdict(list)
    for r in rows:
        if key in r["num"]:
            by[r["date"]].append((r["num"][key], r["y_adj"]))
    ics = []
    for v in by.values():
        if len(v) >= MIN_ROWS_PER_DATE and len({a for a, _ in v}) > 1:
            ic = _spearman([a for a, _ in v], [b for _, b in v])
            if ic is not None:
                ics.append(ic)
    if len(ics) < 10:
        return None
    t, p = _t_p(ics)
    if t is None:
        return None
    return {"kind": "cross_section", "stat": statistics.fmean(ics), "t": t, "p": p, "n_units": len(ics)}


def test_level(rows, key, level) -> dict | None:
    by = defaultdict(list)
    for r in rows:
        if r["cat"].get(key) == level:
            by[r["date"]].append(r["y_adj"])
    n_rows = sum(len(v) for v in by.values())
    if n_rows < MIN_LEVEL_ROWS or len(by) < MIN_LEVEL_DATES:
        return None
    t, p = _t_p([statistics.fmean(v) for v in by.values()])
    if t is None:
        return None
    return {"kind": "category", "stat": statistics.fmean(r for v in by.values() for r in v),
            "t": t, "p": p, "n_units": len(by), "n_rows": n_rows}


def tercile_spread(rows, key) -> tuple[float | None, float | None, float | None]:
    """(Spread oben-unten der bereinigten Rendite, unteres, oberes Terzil-Cutoff)."""
    vals = sorted(r["num"][key] for r in rows if key in r["num"])
    if len(vals) < 30:
        return None, None, None
    lo, hi = vals[len(vals) // 3], vals[2 * len(vals) // 3]
    top = [r["y_adj"] for r in rows if key in r["num"] and r["num"][key] >= hi]
    bot = [r["y_adj"] for r in rows if key in r["num"] and r["num"][key] <= lo]
    if not top or not bot or lo == hi:
        return None, lo, hi
    return statistics.fmean(top) - statistics.fmean(bot), lo, hi


def benjamini_hochberg(pvals: list[float], q: float = FDR_Q) -> list[bool]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    passed = [False] * m
    k_max = 0
    for rank, i in enumerate(order, start=1):
        if pvals[i] <= q * rank / m:
            k_max = rank
    for rank, i in enumerate(order, start=1):
        if rank <= k_max:
            passed[i] = True
    return passed


# ── Lauf ─────────────────────────────────────────────────────────────────────

def _split(rows):
    dates = sorted({r["date"] for r in rows})
    cut = dates[int(len(dates) * DISCOVERY_FRACTION)] if dates else None
    return [r for r in rows if r["date"] < cut], [r for r in rows if r["date"] >= cut], cut


def discover(rows: list[dict], horizon: int) -> dict:
    rows = _demean_by_date([dict(r) for r in rows])
    n_dates = len({r["date"] for r in rows})
    res = {"horizon": horizon, "n_rows": len(rows), "n_dates": n_dates, "findings": [], "tested": 0}
    if len(rows) < MIN_ROWS or n_dates < MIN_DATES:
        res["status"] = f"INSUFFICIENT_DATA (benötigt {MIN_ROWS} Zeilen / {MIN_DATES} Tage)"
        return res
    disc, hold, cut = _split(rows)
    res["discovery_until"], res["holdout_from"] = cut, cut
    tests = []
    num_keys = sorted({k for r in rows for k in r["num"]})
    for k in num_keys:
        t = test_numeric(disc, k)
        if t:
            tests.append(("num", k, None, t))
    cat_levels = sorted({(k, v) for r in rows for k, v in r["cat"].items()})
    for k, lvl in cat_levels:
        t = test_level(disc, k, lvl)
        if t:
            tests.append(("cat", k, lvl, t))
    res["tested"] = len(tests)
    if not tests:
        res["status"] = "NO_TESTABLE_FEATURES"
        return res
    passed = benjamini_hochberg([t[3]["p"] for t in tests])
    for (kind, key, lvl, t), ok in zip(tests, passed):
        if not ok:
            continue
        h = test_numeric(hold, key) if kind == "num" else test_level(hold, key, lvl)
        if not h:
            continue
        same_sign = (h["t"] > 0) == (t["t"] > 0)
        one_sided_p = h["p"] / 2 if same_sign else 1 - h["p"] / 2
        finding = {"kind": kind, "feature": key, "level": lvl, "discovery": t, "holdout": h,
                   "replicated": bool(same_sign and one_sided_p <= HOLDOUT_P)}
        if kind == "num":
            spread, lo, hi = tercile_spread(disc, key)
            finding.update({"tercile_spread": spread, "cut_low": lo, "cut_high": hi})
            finding["economic"] = spread is not None and abs(spread) >= MIN_SPREAD.get(horizon, 0.02)
        else:
            finding["economic"] = abs(t["stat"]) >= MIN_SPREAD.get(horizon, 0.02) / 2
        finding["accepted"] = finding["replicated"] and finding["economic"]
        res["findings"].append(finding)
    res["findings"].sort(key=lambda f: -abs(f["holdout"]["t"]))
    res["status"] = "OK"
    return res


# ── Gate-Wirksamkeit ─────────────────────────────────────────────────────────

def gate_efficacy(ledger_dir: Path = LEDGER_DIR, horizon: int = 20) -> dict:
    """Tagesbereinigte Rendite je Endstatus/Reject-Grund gegenüber den
    vorgeschlagenen Kandidaten. Ein Gate, dessen Verworfene BESSER laufen als
    die Durchgelassenen, vernichtet Alpha (Kandidat für einen Challenger).
    Zeilen ohne Richtung (vor der Deep-Analyse verworfen) = Long-Rendite."""
    rows = []
    for f in sorted(Path(ledger_dir).glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            y = _get(r, f"outcomes.ret_{horizon}d")
            if y is None or not r.get("date"):
                continue
            grp = "proposed" if r.get("status") == "proposed" else (r.get("reject_reason") or r.get("status") or "unknown")
            rows.append({"date": str(r["date"])[:10], "y": float(y), "grp": grp})
    if not rows:
        return {"status": "NO_DATA"}
    _demean_by_date(rows)
    out = {}
    for grp in sorted({r["grp"] for r in rows}):
        by = defaultdict(list)
        for r in rows:
            if r["grp"] == grp:
                by[r["date"]].append(r["y_adj"])
        t, p = _t_p([statistics.fmean(v) for v in by.values()])
        out[grp] = {"n": sum(len(v) for v in by.values()), "n_dates": len(by),
                    "mean_adj": round(statistics.fmean(x for v in by.values() for x in v), 4),
                    "t_vs_day_mean": None if t is None else round(t, 2),
                    "p": None if p is None else round(p, 4)}
    return {"status": "OK", "horizon": horizon, "groups": out,
            "note": "positiv = besser als der Tagesdurchschnitt aller Kandidaten"}


# ── Vorschläge ───────────────────────────────────────────────────────────────

def _baseline():
    # alle Kandidaten, die die Deep-Analyse erreicht haben (Signal-Alpha, nicht Trade)
    return [{"field": "direction", "op": "in", "value": ["BULLISH", "BEARISH"]}]


def proposal_for(finding: dict, horizon: int, today: date) -> dict:
    key, lvl = finding["feature"], finding["level"]
    positive = finding["discovery"]["t"] > 0
    if finding["kind"] == "num":
        cond = ({"field": key, "op": ">=", "value": finding["cut_high"]} if positive
                else {"field": key, "op": "<=", "value": finding["cut_low"]})
        desc = f"{key} {'>=' if positive else '<='} {cond['value']:.4g}"
    else:
        cond = ({"field": key, "op": "==", "value": lvl} if positive
                else {"not": {"field": key, "op": "==", "value": lvl}})
        desc = f"{key} {'==' if positive else '!='} {lvl}"
    slug = hashlib.sha1(f"{key}|{lvl}|{horizon}|{positive}".encode()).hexdigest()[:8]
    return {
        "id": f"auto_{slug}",
        "source_hypothesis": "alpha_discovery",
        "hypothesis": (f"Automatisch entdeckt ({today.isoformat()}): Kandidaten mit {desc} haben eine "
                       f"höhere richtungsbereinigte {horizon}d-Rendite als alle analysierten Kandidaten. "
                       f"Entdeckung IC/t={finding['discovery']['stat']:.3f}/{finding['discovery']['t']:.2f}, "
                       f"interne Bestätigung t={finding['holdout']['t']:.2f}. Endgültig nur prospektiv."),
        "rule": _baseline() + [cond],
        "baseline_rule": _baseline(),
        "metric": f"outcomes.ret_{horizon}d",
        "min_n": 60, "min_clusters": 20,
        "horizon_days": horizon, "max_duration_days": 180,
    }


def run(ledger_dir: Path = LEDGER_DIR, out_dir: Path = OUT_DIR,
        proposals_path: Path = AUTO_PROPOSALS_PATH, today: date | None = None) -> dict:
    import yaml
    today = today or datetime.now(timezone.utc).date()
    report = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "label": "EXPLORATORY — Bestätigung nur prospektiv über challengers.yaml",
              "results": {}, "new_proposals": []}
    candidates = []
    for h in HORIZONS:
        res = discover(load_rows(ledger_dir, h), h)
        report["results"][str(h)] = res
        candidates += [(f, h) for f in res["findings"] if f.get("accepted")]
    candidates.sort(key=lambda fh: -abs(fh[0]["holdout"]["t"]))
    report["gate_efficacy"] = {str(h): gate_efficacy(ledger_dir, h) for h in HORIZONS}

    existing = yaml.safe_load(proposals_path.read_text()) if proposals_path.exists() else None
    existing = existing or {"proposals": []}
    known = {p["id"] for p in existing.get("proposals", [])}
    new = []
    for f, h in candidates:
        p = proposal_for(f, h, today)
        if p["id"] in known or len(new) >= MAX_NEW_PER_RUN:
            continue
        from modules.challenger_registrar import spec_sha256
        p["frozen_on"] = today.isoformat()
        p["spec_sha256"] = spec_sha256({k: v for k, v in p.items() if k != "spec_sha256"})
        new.append(p)
    if new:
        existing["proposals"] = existing.get("proposals", []) + new
        proposals_path.parent.mkdir(parents=True, exist_ok=True)
        proposals_path.write_text(yaml.safe_dump(existing, sort_keys=False, allow_unicode=True))
        report["new_proposals"] = [p["id"] for p in new]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "alpha_discovery.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    r = run()
    print(json.dumps({h: {k: v for k, v in res.items() if k != "findings"} | {
        "accepted": [f["feature"] + (f"={f['level']}" if f["level"] else "") for f in res["findings"] if f["accepted"]]}
        for h, res in r["results"].items()}, indent=2, ensure_ascii=False))
    print("neue Vorschläge:", r["new_proposals"])

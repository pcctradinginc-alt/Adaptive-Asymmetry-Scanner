"""
scripts/incremental_analysis.py – Base vs. Base + externe Familie (EXPLORATIV).

Beantwortet NICHT "korreliert Feature X mit Rendite", sondern: verbessert ein
VORAB festgelegter, nicht getunter Filter das BESTEHENDE Modell, wenn man
verlorene Gewinner mitzählt?

Regeln (fest, aus den bestehenden Zustandsschwellen classify_state: z <= -0.5
CONTRACTION): BULLISH-Kandidat wird entfernt, wenn die Familie für ihn
kontrahiert (bzw. Wetterstörung >= 0.5 bei weather_relevance HIGH). Keine
Schwelle wird aus den Outcomes abgeleitet.

Datenquellen:
  A) 117+ geschlossene Trades (outputs/history.json): nur PIT-sichere
     Familien (ALFRED: US-Frachtindex, UMCSENT, INDPRO) zum Einstiegszeitpunkt.
     EU-Fracht/Shipping/Wetter sind dort NICHT rückwirkend PIT-sicher -> N/A.
  B) Candidate Ledger (outputs/candidate_ledger/*.jsonl): eingefrorener
     Kontext + reife Outcomes, alle Familien (ab 2026-09-28).

Ausgabe: outputs/research/incremental_analysis.{json,md}. Ergebnisse sind
EXPLORATORY ONLY; Bestätigung nur prospektiv über challengers.yaml.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT_DIR = Path("outputs/research")
LEDGER_DIR = Path("outputs/candidate_ledger")
CONTRACTING = {"CONTRACTION", "STRONG_CONTRACTION"}
EXPOSED = {"MEDIUM", "HIGH"}
BIG_WIN = 1.0          # +100 % = "großer asymmetrischer Gewinner"
LARGE_LOSS = -0.8
N_BOOT = 2000
ORTHO_FEATURES = ("impact", "surprise", "mismatch", "z_score", "eps_drift", "sigma_30d")


def _stats(v: list[float]) -> dict:
    if not v:
        return {"n": 0}
    return {"n": len(v), "mean": round(statistics.fmean(v), 4), "median": round(statistics.median(v), 4),
            "win_rate": round(sum(x > 0 for x in v) / len(v), 3),
            "large_loss_rate": round(sum(x <= LARGE_LOSS for x in v) / len(v), 3)}


def evaluate_filter(rows: list[dict], excluded: list[bool], seed: int = 11) -> dict:
    """rows: {date, outcome}; excluded: Filter entfernt die Zeile."""
    base = [r["outcome"] for r in rows]
    kept = [r["outcome"] for r, ex in zip(rows, excluded) if not ex]
    removed = [r["outcome"] for r, ex in zip(rows, excluded) if ex]
    b, k = _stats(base), _stats(kept)
    res = {
        "base": b, "with_filter": k,
        "n_removed": len(removed),
        "removed_dates": len({r["date"] for r, ex in zip(rows, excluded) if ex}),
        "avoided_losers": sum(1 for x in removed if x < 0),
        "lost_winners": sum(1 for x in removed if x > 0),
        "lost_big_winners": sum(1 for x in removed if x >= BIG_WIN),
        "net_removed_return": round(sum(removed), 4),
        # Erwartete Rendite je Signal (entfernte Signale = 0): enthält die
        # Opportunitätskosten verlorener Gewinner
        "expected_return_per_signal_base": round(sum(base) / len(base), 4) if base else None,
        "expected_return_per_signal_filter": round(sum(kept) / len(base), 4) if base else None,
    }
    if b.get("n") and k.get("n"):
        res["delta_mean"] = round(k["mean"] - b["mean"], 4)
        res["delta_median"] = round(k["median"] - b["median"], 4)
        res["delta_large_loss_rate"] = round(k["large_loss_rate"] - b["large_loss_rate"], 4)
        res["delta_expected_per_signal"] = round(res["expected_return_per_signal_filter"]
                                                 - res["expected_return_per_signal_base"], 4)
        res["cluster_ci_delta_expected_per_signal"] = _cluster_ci(rows, excluded, seed)
    return res


def _cluster_ci(rows, excluded, seed):
    """95%-CI der Δ erwarteten Rendite je Signal, Bootstrap über Signaltage."""
    by_date: dict = {}
    for r, ex in zip(rows, excluded):
        by_date.setdefault(r["date"], []).append((r["outcome"], ex))
    dates = sorted(by_date)
    if len(dates) < 5:
        return None
    rnd = random.Random(seed)
    deltas = []
    for _ in range(N_BOOT):
        sample = [x for d in (rnd.choice(dates) for _ in dates) for x in by_date[d]]
        n = len(sample)
        deltas.append((sum(o for o, ex in sample if not ex) - sum(o for o, _ in sample)) / n)
    deltas.sort()
    return [round(deltas[int(0.025 * N_BOOT)], 4), round(deltas[int(0.975 * N_BOOT) - 1], 4)]


def sample_structure(rows: list[dict], regime_key: str | None = None) -> dict:
    dates = sorted({r["date"] for r in rows})
    out = {"n_trades": len(rows), "n_independent_dates": len(dates),
           "calendar_span": [dates[0], dates[-1]] if dates else None,
           "n_months": len({d[:7] for d in dates}),
           "n_sectors": len({r.get("sector") for r in rows if r.get("sector")}) or None}
    if regime_key:
        regimes = Counter(r.get(regime_key) for r in rows)
        out["n_regimes"] = len([k for k in regimes if k is not None])
        out["regimes"] = dict(regimes)
        # Konfundierung: fällt das Regime mit dem Kalendermonat zusammen?
        month_by_regime = {}
        for r in rows:
            month_by_regime.setdefault(r.get(regime_key), Counter())[r["date"][:7]] += 1
        out["regime_x_month"] = {str(k): dict(v) for k, v in month_by_regime.items()}
    return out


def _corr(a, b):
    try:
        return round(statistics.correlation(a, b), 3)
    except Exception:
        return None


def orthogonality(rows: list[dict], feature: str) -> dict:
    """Korrelation des externen Features mit den bestehenden Scanner-Features,
    je Trade und je Signaltag (Tagesmittel; ein Makrowert je Tag)."""
    out = {}
    for f in ORTHO_FEATURES:
        pairs = [(r[feature], r["features"].get(f)) for r in rows
                 if r.get(feature) is not None and isinstance(r["features"].get(f), (int, float))]
        if len(pairs) < 10:
            continue
        by_date: dict = {}
        for r in rows:
            if r.get(feature) is not None and isinstance(r["features"].get(f), (int, float)):
                by_date.setdefault(r["date"], []).append((r[feature], r["features"][f]))
        daily = [(statistics.fmean(x for x, _ in v), statistics.fmean(y for _, y in v)) for v in by_date.values()]
        out[f] = {"per_trade": _corr(*zip(*pairs)),
                  "per_date": _corr(*zip(*daily)) if len(daily) >= 5 else None, "n_dates": len(daily)}
    return out


# ── A) geschlossene Trades mit PIT-sicheren ALFRED-Familien ────────────────

def closed_trade_rows(history: dict, archive) -> list[dict]:
    from modules.external.features import classify_state
    from modules.external.sources import real_economy_features as ref
    from modules.external.sources import road_freight_features as rff
    rows, cache = [], {}
    for t in history.get("closed_trades", []):
        if t.get("outcome") is None or not t.get("entry_date"):
            continue
        d = str(t["entry_date"])[:10]
        if d not in cache:
            T = datetime.fromisoformat(d).replace(tzinfo=timezone.utc) + timedelta(hours=14)
            us = archive.as_of("bts_freight_tsi", T)
            fm = archive.as_of("fred_us_macro", T)
            z = rff.us_freight_tsi_z(us)
            cache[d] = {"us_freight_z": z, "us_freight_state": classify_state(z),
                        "us_survey_z": ref.us_survey_z(fm), "us_hard_z": ref.us_hard_z(fm)}
        strat = str(t.get("strategy", ""))
        rows.append({"date": d, "ticker": t.get("ticker"), "outcome": float(t["outcome"]),
                     "direction": "BEARISH" if ("PUT" in strat) else "BULLISH",
                     "sector": (t.get("features") or {}).get("sector"),
                     "features": t.get("features") or {}, **cache[d]})
    return rows


# ── B) Candidate Ledger mit eingefrorenem Kontext ──────────────────────────

LEDGER_FAMILIES = {
    "us_freight": ("states.us_freight_state", "road_freight_relevance"),
    "eu_freight": ("states.eu_freight_state", "road_freight_relevance"),
    "asia_freight": ("states.asia_freight_state", "road_freight_relevance"),
    "shipping": ("states.global_maritime_state", "maritime_relevance"),
}


def _path(d, p):
    for k in p.split("."):
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def ledger_rows(ledger_dir: Path = LEDGER_DIR, metric: str = "outcomes.real_strat_ret_45d") -> list[dict]:
    rows = []
    for f in sorted(Path(ledger_dir).glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            o = _path(r, metric)
            if r.get("status") != "proposed" or o is None or not r.get("external"):
                continue
            rows.append({"date": str(r.get("date"))[:10], "outcome": float(o), "direction": r.get("direction"),
                         "sector": (r.get("features") or {}).get("sector"),
                         "features": r.get("features") or {}, "external": r["external"]})
    return rows


def ledger_family_masks(rows):
    masks = {}
    for fam, (state_path, rel_key) in LEDGER_FAMILIES.items():
        masks[fam] = [r["direction"] == "BULLISH"
                      and _path(r["external"], f"ticker_exposure.{rel_key}") in EXPOSED
                      and _path(r["external"], state_path) in CONTRACTING for r in rows]
    masks["road_plus_shipping"] = [
        r["direction"] == "BULLISH"
        and (_path(r["external"], "ticker_exposure.road_freight_relevance") in EXPOSED
             or _path(r["external"], "ticker_exposure.maritime_relevance") in EXPOSED)
        and _path(r["external"], "states.global_freight_state") in CONTRACTING
        and _path(r["external"], "states.global_maritime_state") in CONTRACTING for r in rows]
    masks["weather"] = [
        r["direction"] == "BULLISH"
        and _path(r["external"], "ticker_exposure.weather_relevance") == "HIGH"
        and (_path(r["external"], "primitives.weather_disruption_index") or 0) >= 0.5 for r in rows]
    return masks


def run() -> dict:
    from modules.external.archive import ExternalArchive
    history = json.loads(Path("outputs/history.json").read_text())
    archive = ExternalArchive(root="outputs/external_data")
    ct = closed_trade_rows(history, archive)
    us_mask = [r["direction"] == "BULLISH" and r["us_freight_state"] in CONTRACTING for r in ct]
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label": "EXPLORATORY ONLY — keine Produktions- oder Promotionsaussage",
        "closed_trades_pit_safe": {
            "sample": sample_structure(ct, "us_freight_state"),
            "us_freight_filter": evaluate_filter(ct, us_mask),
            "orthogonality_us_freight_z": orthogonality(ct, "us_freight_z"),
            "not_pit_safe_families": ["eu_freight", "asia_freight", "shipping", "weather"],
        },
    }
    lr = ledger_rows()
    result["ledger"] = {"sample": sample_structure(lr)}
    if lr:
        result["ledger"]["families"] = {fam: evaluate_filter(lr, m) for fam, m in ledger_family_masks(lr).items()}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "incremental_analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False))

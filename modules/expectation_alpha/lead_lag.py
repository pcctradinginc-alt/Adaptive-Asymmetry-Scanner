"""modules/expectation_alpha/lead_lag.py – Lead-Lag-Diagnostik je Feature/Familie (nur Research/SHADOW).

Frage: Welche EA-Features bzw. Quellfamilien stehen auf den VORAB festgelegten Horizonten
(config outcomes.horizons, z. B. 20/60/120/250 Handelstage) in Beziehung zu späteren Outcomes?

Regeln:
* Der Featurewert stammt aus dem eingefrorenen Entscheidungs-Snapshot (Ledger-Zeile, `env`/These). Es wird
  nichts neu berechnet und keine revidierte Information verwendet.
* Der Outcome stammt aus outcomes.jsonl und existiert erst nach Ablauf des Horizonts. Zeilen mit exit_date
  vor oder gleich dem Entscheidungstag werden verworfen (Lookahead-Schutz).
* Alle Horizonte werden immer berichtet. Es gibt keinen „besten“ Horizont, keine Auswahl nach Ergebnis,
  kein Gewicht und keine Strategieänderung (kein P-Hacking). Die Ergebnisse dienen nur als Grundlage
  späterer, präregistrierter Hypothesen.
* Unter `evaluation.min_n_report` Beobachtungen gibt es keine Effektaussage (NEED_MORE_DATA).
Statistik (wiederverwendet aus factor_monitor):
* Spearman-IC mit Fisher-CI über unabhängige Signaltage;
* Spread oberes − unteres Terzil (binäre Features: 1 − 0) mit Block-Bootstrap-CI über Signaltage;
* Trefferquote, MAE/MFE und Regime-Aufschlüsselung.
"""
from __future__ import annotations

import statistics
from datetime import date

import numpy as np

from modules.expectation_alpha.schemas import ERROR, rnd

VERSION = "lead-lag-v1"
# Feature -> (Familie, Quelle im eingefrorenen Snapshot). Fest registriert, nicht nach Ergebnis erweitert.
FEATURES: dict[str, tuple[str, str]] = {
    "macro_alignment":              ("expectation_gap", "row:macro_alignment"),
    "gap_abs_z":                    ("expectation_gap", "env:ea_gap_abs_z"),
    "gap_aligned":                  ("expectation_gap", "env:ea_gap_aligned"),
    "gap_percentile":               ("percentile", "gap:gap_percentile"),
    "gap_acceleration_aligned":     ("rate_of_change", "env:ea_gap_accel_aligned"),
    "sector_alignment":             ("relative", "row:sector_alignment"),
    "confirmation_ratio_raw":       ("cross_asset", "env:ea_confirmation_ratio"),
    "confirmation_ratio_effective": ("cross_asset", "env:ea_effective_confirmation_ratio"),
    "confirmation_family_count":    ("cross_asset", "env:ea_confirmation_family_count"),
    "regime_uncertainty":           ("regime", "row:regime_uncertainty"),
    "regime_transition_probability": ("regime_transition", "row:regime_transition_probability"),
    "verified_claim_fraction":      ("claims", "env:ea_verified_claim_fraction"),
    "news_impact":                  ("news", "news:impact"),
    "news_surprise":                ("news", "news:surprise"),
}
BOOT_N = 500


def feature_value(row: dict, spec: str):
    src, key = spec.split(":", 1)
    if src == "row":
        v = row.get(key)
    elif src == "env":
        v = (row.get("env") or {}).get(key)
    elif src == "gap":
        v = (row.get("expectation_gap") or {}).get(key)
    elif src == "news":
        v = (row.get("news_edge") or {}).get(key)
    else:
        raise ValueError(spec)
    if v is None or isinstance(v, bool) or isinstance(v, str):
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _pairs(rows: list[dict], outs: dict, spec: str, h: int) -> list[dict]:
    out = []
    for r in rows:
        if r.get("status") == ERROR:
            continue
        x = feature_value(r, spec)
        o = outs.get((r["observation_id"], "UNDERLYING", "immediate", h))
        if x is None or not o or o.get("outcome_net") is None:
            continue
        exit_d = o.get("exit_date")
        if not exit_d or date.fromisoformat(str(exit_d)[:10]) <= date.fromisoformat(str(r["date"])[:10]):
            continue                                       # Outcome nicht nach der Entscheidung -> nie verwenden
        out.append({"date": str(r["date"])[:10], "x": x, "y": float(o["outcome_net"]), "mae": o.get("mae"),
                    "mfe": o.get("mfe"), "regime": r.get("regime") or "unknown"})
    return out


def _split(p: list[dict]) -> tuple[list[dict], list[dict], str]:
    xs = sorted({q["x"] for q in p})
    if len(xs) <= 2:                                       # binär/zweiwertig: oberer vs. unterer Wert
        hi, lo = xs[-1], xs[0]
        return [q for q in p if q["x"] == hi], [q for q in p if q["x"] == lo], "binary"
    v = sorted(q["x"] for q in p)
    t1, t2 = v[len(v) // 3], v[(2 * len(v)) // 3]
    return [q for q in p if q["x"] >= t2], [q for q in p if q["x"] <= t1], "tercile"


def _spread(p: list[dict]) -> float | None:
    top, bot, _ = _split(p)
    if not top or not bot or top is bot:
        return None
    return statistics.fmean(q["y"] for q in top) - statistics.fmean(q["y"] for q in bot)


def _spread_ci(p: list[dict], seed: int) -> list:
    by: dict[str, list[dict]] = {}
    for q in p:
        by.setdefault(q["date"], []).append(q)
    ids = sorted(by)
    if len(ids) < 2:
        return [None, None]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(BOOT_N):
        sample = [q for i in rng.integers(0, len(ids), len(ids)) for q in by[ids[i]]]
        s = _spread(sample) if len({q["x"] for q in sample}) > 1 else None
        if s is not None:
            vals.append(s)
    if len(vals) < 0.9 * BOOT_N:
        return [None, None]
    lo, hi = np.quantile(vals, [0.05, 0.95])
    return [rnd(lo, 5), rnd(hi, 5)]


def _block(p: list[dict], min_n: int, seed: int) -> dict:
    from modules import factor_monitor as fm
    n, days = len(p), len({q["date"] for q in p})
    res: dict = {"n": n, "independent_signal_days": days}
    if n < min_n or len({q["x"] for q in p}) < 2:
        res["status"] = "NEED_MORE_DATA"
        return res
    ic = fm.spearman([q["x"] for q in p], [q["y"] for q in p])
    lo, hi = fm.fisher_ci(ic, days)
    top, bot, kind = _split(p)
    res.update(status="OK", ic=rnd(ic, 4), ic_ci90=[rnd(lo, 4), rnd(hi, 4)], split=kind,
               forward_return_spread=rnd(_spread(p), 5), spread_ci90=_spread_ci(p, seed),
               hit_rate_top=rnd(sum(1 for q in top if q["y"] > 0) / len(top), 4) if top else None,
               hit_rate_bottom=rnd(sum(1 for q in bot if q["y"] > 0) / len(bot), 4) if bot else None,
               mae_top=rnd(statistics.fmean([q["mae"] for q in top if q["mae"] is not None]), 5)
               if any(q["mae"] is not None for q in top) else None,
               mfe_top=rnd(statistics.fmean([q["mfe"] for q in top if q["mfe"] is not None]), 5)
               if any(q["mfe"] is not None for q in top) else None)
    return res


def run(rows: list[dict], outs: dict, cfg: dict) -> dict:
    """Je Feature × vorab festem Horizont. Kein „bester Horizont“, keine Auswahl, kein Produktionseinfluss."""
    horizons = [int(h) for h in (cfg.get("outcomes") or {}).get("horizons") or [20, 60, 120, 250]]
    ev = cfg.get("evaluation") or {}
    min_n, seed = int(ev.get("min_n_report", 30)), int(ev.get("bootstrap_seed", 47))
    feats: dict = {}
    n_ok = 0
    for name, (family, spec) in FEATURES.items():
        per_h: dict = {}
        for h in horizons:
            p = _pairs(rows, outs, spec, h)
            b = _block(p, min_n, seed)
            regimes = {}
            for rg in sorted({q["regime"] for q in p}):
                sub = [q for q in p if q["regime"] == rg]
                regimes[rg] = _block(sub, min_n, seed) if sub else {"n": 0}
            b["regime_breakdown"] = regimes
            per_h[str(h)] = b
            n_ok += b["status"] == "OK"
        feats[name] = {"family": family, "source": spec, "horizons": per_h}
    fam: dict = {}
    for name, f in feats.items():
        fam.setdefault(f["family"], []).append(name)
    return {"version": VERSION, "mode": "SHADOW – Research, keine Strategieänderung",
            "horizons_preregistered": horizons, "selection": "keine (alle Horizonte berichtet)",
            "lead_lag_status": "OK" if n_ok else "NEED_MORE_DATA", "n_blocks_ok": n_ok,
            "families": fam, "features": feats}

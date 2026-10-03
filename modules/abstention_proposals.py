"""modules/abstention_proposals.py – Beobachtung -> falsifizierbare Abstinenz-Hypothese
-> historischer Walk-Forward -> eingefrorener VERTRAGSENTWURF (Self-Improvement-Eingang).

    python -m modules.abstention_proposals      (CI: feedback.yml, nach dem PromotionController)

Lernt aus echten, verlässlichen Champion-Outcomes (modules/outcomes.py), bei welchen zum
Entscheidungszeitpunkt bekannten Merkmalen Champion-Trades systematisch schlechter laufen.

Ablauf (alle Regeln fest, nichts aus den Testdaten gewählt):
  1. Daten: geschlossene Champion-Trades mit verlässlichem Outcome, chronologisch.
  2. Split: erste 60 % = Kalibrierung, letzte 40 % = Walk-Forward-Test.
  3. Je Merkmal und Richtung (oberes/unteres Quintil der KALIBRIERUNG) eine Regel
     "Merkmal > q80" bzw. "Merkmal < q20". Kandidat nur, wenn in der Kalibrierung die
     Schwanz-Expectancy unter der Rest-Expectancy liegt.
  4. Test (nur später liegende Trades): Schwanz-Expectancy < Rest-Expectancy, mindestens
     MIN_TEST_TAIL Treffer und Differenz-Bootstrap-Untergrenze < 0 bei Bonferroni über
     alle getesteten Regeln.
  5. Überlebende -> vollständiger Vertragsentwurf (Schema hypothesis_contract) in
     outputs/intelligence/contract_proposals.json, Status PROPOSED, mit Similarity-Check.

Es wird NICHTS registriert und NICHTS angewendet: historische Evidenz erreicht höchstens
HISTORICALLY_VALIDATED. Registrierung = menschlicher PR in config/promotion_hypotheses.yaml
(CODEOWNERS); danach zählt ausschließlich prospektive Forward-Evidenz.
"""
from __future__ import annotations

import json
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path

from modules import hypothesis_contract as hc
from modules.outcomes import is_reliable_outcome

HISTORY = Path("outputs/history.json")
OUT = Path("outputs/intelligence/contract_proposals.json")
CALIBRATION_SHARE = 0.60
MIN_TEST_TAIL = 5
MIN_TRADES = 40
ALPHA = 0.05
BOOT_N = 2000
SEED = 17
# Zum Entscheidungszeitpunkt bekannte numerische Merkmale der Champion-Trades
FEATURES = ("impact", "surprise", "mismatch", "z_score", "sigma_30d", "price_move_48h", "quick_mc_hit_rate",
            "eps_drift", "final_mc_hit_rate",
            # Abstention Intelligence (modules/abstention_intelligence.py): erst ab 2026-10 in Trade-Features;
            # ältere Trades ohne Wert zählen nicht (None, nie 0)
            "risk_p_model_wrong", "risk_regime_mismatch", "risk_unknown_risk", "risk_model_disagreement",
            "risk_counterfactual_fragility", "risk_alpha_decay", "risk_abstain_score")


def _value(t: dict, f: str):
    if f == "final_mc_hit_rate":
        v = (t.get("simulation") or {}).get("hit_rate")
    else:
        v = (t.get("features") or {}).get(f)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _q(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(round(p * (len(xs) - 1)))))]


def _diff_ci_upper(tail: list[float], rest: list[float], alpha: float) -> float | None:
    """Obere einseitige Grenze von E[tail] - E[rest] (Bootstrap)."""
    if len(tail) < 2 or len(rest) < 2:
        return None
    rng = random.Random(SEED)
    vals = sorted(statistics.fmean(rng.choices(tail, k=len(tail))) - statistics.fmean(rng.choices(rest, k=len(rest)))
                  for _ in range(BOOT_N))
    return vals[min(len(vals) - 1, int((1 - alpha) * len(vals)))]


def propose(history: dict, existing: list[dict] | None = None, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    trades = sorted((t for t in history.get("closed_trades") or [] if t.get("outcome") is not None
                     and is_reliable_outcome(t)), key=lambda t: str(t.get("entry_date", "")))
    res = {"generated": now.isoformat(timespec="seconds"), "data_kind": "historical_walk_forward",
           "n_trades": len(trades), "candidates_tested": 0, "proposals": [], "rejected": [],
           "note": "Nur Entwürfe. Registrierung per menschlichem PR; Promotion nur aus Forward-Daten."}
    if len(trades) < MIN_TRADES:
        res["status"] = f"zu wenige verlässliche Trades ({len(trades)}/{MIN_TRADES})"
        return res
    cut = int(len(trades) * CALIBRATION_SHARE)
    cal, test = trades[:cut], trades[cut:]
    res.update(calibration_period=[cal[0]["entry_date"], cal[-1]["entry_date"]],
               test_period=[test[0]["entry_date"], test[-1]["entry_date"]])
    cands = []
    for f in FEATURES:
        cv = [(_value(t, f), t["outcome"]) for t in cal if _value(t, f) is not None]
        if len(cv) < 20:
            continue
        xs = [v for v, _ in cv]
        for side, p, op in (("high", 0.8, ">"), ("low", 0.2, "<")):
            th = round(_q(xs, p), 6)
            tail = [o for v, o in cv if (v > th if op == ">" else v < th)]
            rest = [o for v, o in cv if not (v > th if op == ">" else v < th)]
            if len(tail) >= 4 and rest and statistics.fmean(tail) < statistics.fmean(rest):
                cands.append((f, op, th, statistics.fmean(tail) - statistics.fmean(rest), len(tail)))
    res["candidates_tested"] = len(cands)
    alpha = ALPHA / max(1, len(cands))
    existing = existing if existing is not None else [e["contract"] for e in hc.read_registry()[0]]
    for f, op, th, d_cal, n_cal in cands:
        tv = [(_value(t, f), t["outcome"]) for t in test if _value(t, f) is not None]
        tail = [o for v, o in tv if (v > th if op == ">" else v < th)]
        rest = [o for v, o in tv if not (v > th if op == ">" else v < th)]
        stats = {"feature": f, "rule": f"{f} {op} {th}", "calibration_delta": round(d_cal, 4),
                 "calibration_n_tail": n_cal, "test_n_tail": len(tail), "test_n_rest": len(rest),
                 "test_delta": round(statistics.fmean(tail) - statistics.fmean(rest), 4) if tail and rest else None,
                 "test_ci_upper": None}
        if len(tail) < MIN_TEST_TAIL or not rest:
            res["rejected"].append({**stats, "reason": f"zu wenige Test-Treffer ({len(tail)}/{MIN_TEST_TAIL})"})
            continue
        ub = _diff_ci_upper(tail, rest, alpha)
        stats["test_ci_upper"] = round(ub, 4) if ub is not None else None
        if stats["test_delta"] is None or stats["test_delta"] >= 0 or ub is None or ub >= 0:
            res["rejected"].append({**stats, "reason": "Walk-Forward-Test bestätigt nicht (Bonferroni)"})
            continue
        draft = _draft(f, op, th, stats, now)
        sims = hc.similar_existing(draft, existing)
        sims += [(hc.key(o), 1.0) for o in existing if o.get("hypothesis_id") == draft["hypothesis_id"]]
        if sims:
            res["rejected"].append({**stats, "reason": f"ähnlich zu {sims} – bestehende Hypothese fortführen"})
            continue
        res["proposals"].append({"status": "PROPOSED", "max_state_from_history": "HISTORICALLY_VALIDATED",
                                 "walk_forward": stats, "contract_draft": draft,
                                 "validation_errors": hc.validate({**draft, "registered_at": now.isoformat(),
                                                                   "forward_start": now.isoformat()[:10] + "T23:59:59Z"})})
    res["status"] = f"{len(res['proposals'])} Vorschlag/Vorschläge"
    return res


def _draft(f: str, op: str, th: float, stats: dict, now: datetime) -> dict:
    base = hc.load()[0] if hc.load() else {}
    hid = f"PROM-ABST-AUTO-{f.upper()}-{'HI' if op == '>' else 'LO'}"
    return {**{k: base.get(k) for k in ("universe", "population", "outcome_horizon", "primary_metric",
                                         "secondary_metrics", "baseline", "minimum_sample_size",
                                         "minimum_independent_dates", "minimum_calendar_span",
                                         "promotion_criteria", "demotion_criteria")},
            "hypothesis_id": hid, "version": 1, "title": f"Abstinenz bei {f} {op} {th} (automatischer Vorschlag)",
            "created_at": now.date().isoformat(), "registered_at": None, "forward_start": None,
            "created_by": "abstention_proposals (Walk-Forward auf Champion-Outcomes)", "source_type": "regime_failure"
            if f in ("sigma_30d", "price_move_48h") else "meta_learning_failure",
            "research_question": f"Haben Champion-Trades mit {f} {op} {th} eine niedrigere Expectancy als die übrigen?",
            "economic_rationale": f"Walk-Forward auf verlässlichen Champion-Outcomes: Kalibrierung Δ {stats['calibration_delta']}, "
                                  f"Test Δ {stats['test_delta']} (n={stats['test_n_tail']}); ökonomische Begründung vor "
                                  f"Registrierung durch Menschen zu ergänzen.",
            "exposure": "Einzeltrade (Champion-Merkmal)", "signal_definition": f, "features": [f],
            "thresholds": {"op": op, "value": th}, "direction": -1, "sector_scope": ["all"], "regime_scope": ["all"],
            "production_class": "abstention", "maximum_initial_influence": "ABSTENTION_ONLY",
            "failure_condition": "Blockierte Trades nicht schlechter als durchgelassene (Forward)"}


def main() -> int:
    hist = json.loads(HISTORY.read_text(encoding="utf-8")) if HISTORY.exists() else {}
    res = propose(hist)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Abstinenz-Vorschläge: {res['status']} · getestet {res['candidates_tested']} · verworfen {len(res['rejected'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

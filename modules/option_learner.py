"""modules/option_learner.py – lernt aus geprüften Netto-Options-Ergebnissen und leitet Empfehlungen ab (Research).

Kreislauf (autonom, kein LLM, keine Markt-API, keine Order):
  Kandidaten-Ledger (Merkmale + Kontrakt zum Entscheidungszeitpunkt eingefroren)
  -> Ausdrucksformen je Kandidat: Produktions-Strategie (single|spread), Einzel-Option (real_option) und die
     Research-Kontraktvariante (real_option_alt); derselbe Kontrakt zählt nur einmal
  -> Label: Ausführbare Rendite minus Kommission, NUR wenn research_evidence_discovery.provenance == VERIFIED
     (Entry und Exit aus dem Quote-Archiv nachgerechnet)
  -> Ridge-Regression mit festen Hyperparametern auf standardisierten Merkmalen
  -> Walk-Forward je Signaltag: trainiert nur auf Labels, deren Mark-Tag VOR dem Testtag liegt (Embargo)
  -> Prognose-Journal: jede Tagesprognose append-only, bevor ihr Label existiert (prospektiv), mit
     Experimentversion (experiment_id: Merkmale, Label, Kosten, Training, Auswahlregel) und Modellinstanz
     (model_id: Trainingsstichtag, Label-Manifest, Koeffizienten-Fingerprint; Register model_instances.jsonl)
  -> Forward-Auswertung des Journals, sobald die Labels geprüft vorliegen: nur Prognosen der aktuellen
     Experimentversion, am Signaltag geschrieben, von einer registrierten, vorher trainierten Modellinstanz;
     fortlaufend neu trainierte Instanzen derselben Version bilden eine gemeinsame Lernkurve
  -> EMPFEHLUNG nur, wenn die Freigabe (release_gate) alle Evidenzanforderungen erneut bestätigt: Walk-Forward
     VALIDATED UND Forward FORWARD_POSITIVE, jeweils mit Mindest-Testtagen, unabhängigen Perioden, Fallzahl und
     Konfidenzintervall über 0 – fail closed bei fehlenden oder widersprüchlichen Angaben; je Kandidat die
     Ausdrucksform mit der höchsten Prognose > 0. Sonst nur eine als UNVALIDIERT markierte Research-Watchlist.
     Mail nur für das 20-Tage-Hauptmodell; das 5-Tage-Modell ist Frühindikator ohne Handels-Mail.
  -> Lernfortschritt: je Experimentversion wird die erste Modellinstanz als Referenz eingefroren (Gewichte einmalig
     gespeichert); lernende Version und Referenz entscheiden täglich über dieselben Kandidaten (inkl. ABSTAIN).
     Gepaarter Vergleich der realisierten Netto-Ergebnisse bei gleichem Budget je Kandidat, Unsicherheit über
     unabhängige Haltefenster; dazu "nie handeln" und "alles handeln" unter denselben Bedingungen.
  -> Leistungszeugnis: Trainings-Manifest (welche geprüften Labels in welches Modell flossen), append-only
     Status-Historie, Experimentversion und Modellinstanzen, Lernfortschritt gegen die Referenz.
Kein Einfluss auf Champion, Gates, Sizing, Risk oder Promotion.

CLI:
  python -m modules.option_learner evaluate  --as-of YYYY-MM-DD [--horizon 20]
  python -m modules.option_learner recommend --date  YYYY-MM-DD [--horizon 20] [--mail]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

from modules import research_evidence_discovery as red
from modules.atomic_io import (
    append_jsonl,
    atomic_write_json,
    atomic_write_text,
    read_jsonl,
)

log = logging.getLogger(__name__)

OUT = Path("outputs/research/option_learner")
JOURNAL = "predictions.jsonl"
INSTANCES = "model_instances.jsonl"
REFERENCE_DIR = "reference"
REPORT_CARD = "report_card"
HORIZON = 20
COMMISSION_PER_LEG_SIDE = 0.65          # USD je Kontrakt und Seite (wie cost_model der Feedback-Outcomes)
RIDGE_ALPHA = 10.0
MIN_TRAIN_ROWS = 100
MIN_TRAIN_DATES = 8
MIN_OOS_DATES = 20                      # Walk-Forward-Verdikt erst ab so vielen Testtagen
MIN_SELECTED = 30                       # mindestens so viele OOS-Fälle mit Prognose > 0
MIN_INDEPENDENT_PERIODS = 5             # nicht überlappende Haltefenster (Horizont in Kalendertagen) mit Auswahl
IC_T_MIN = 2.0
MIN_COVERAGE = 0.8                      # Anteil der TRADE-Entscheidungen mit geprüftem Ergebnis der gewählten Form
DUE_GRACE_DAYS = 10                     # candidate_ledger.LATE_MARK_DAYS (7) + bis zu 3 Tage späterer Entry
MIN_SPAN_DAYS = 90                      # kürzerer Auswertungszeitraum = höchstens eine Marktphase
MAIL_HORIZON = HORIZON                  # Handels-Mail nur für das Hauptmodell (5 T = Frühindikator)
MIN_FEATURE_COVERAGE = 0.5
BOOTSTRAP = 2000
WATCHLIST_N = 5
WINSOR = (0.01, 0.99)
ALLOWED_STRATEGIES = ("LONG_CALL", "BULL_CALL_SPREAD")   # Call-Strukturen, die die Produktion handeln kann
# Evidenzanforderungen (ändern keine Prognose, daher nicht Teil der Experimentversion)
EVIDENCE = {"min_oos_dates": MIN_OOS_DATES, "min_selected": MIN_SELECTED,
            "min_independent_periods": MIN_INDEPENDENT_PERIODS, "ic_t_min": IC_T_MIN, "bootstrap": "block/horizon",
            "min_coverage": MIN_COVERAGE, "due_grace_days": DUE_GRACE_DAYS}

LEDGER_FEATURES = (
    "impact", "surprise", "sentiment_score", "sentiment_confidence", "sentiment_drift", "news_count",
    "news_novelty", "rel_volume", "scan_day_ret", "short_mom", "log_market_cap", "short_pct_float",
    "options_skew_25d", "options_skew_ratio", "dealer_oi_ratio", "ml_exp_ret_60", "ml_q10_ret_60",
    "ml_q90_ret_60", "ml_exp_dd_60", "ml_p_gt10_60", "ml_disagreement_sd", "ml_hgb_xs20_v1", "ml_enet_xs20_v1",
)
CONTRACT_FEATURES = ("c_entry_spread", "c_is_spread", "c_is_call", "c_iv", "c_dte", "c_moneyness", "c_delta",
                     "c_log_oi", "c_iv_rank", "c_rv_score", "c_term_score")
# (Ledger-Schlüssel, Outcome-Präfix, Name) – real_strategy heißt nach Struktur "single" oder "spread"
SOURCES = (("real_strategy", "real_strat", None), ("real_option", "real_opt", "single"),
           ("real_option_alt", "real_alt", "single_alt"))


def experiment() -> dict:
    """Experiment-/Strategieversion: alles, was eine Prognose oder Entscheidung inhaltlich bestimmt. Jede Änderung
    an Merkmalen, Labeldefinition, Kostenmodell, Trainingsverfahren oder Auswahlregel ergibt eine neue Version;
    deren Forward-Evidenz wird nie mit der einer anderen Version vermischt."""
    return {"name": "option_learner_ridge",
            "features": {"ledger": list(LEDGER_FEATURES), "contract": list(CONTRACT_FEATURES),
                         "min_coverage": MIN_FEATURE_COVERAGE},
            "label": {"target": "executable return (buy ask, sell bid) - commission", "provenance": "VERIFIED",
                      "sources": [s[0] for s in SOURCES]},
            "costs": {"commission_per_leg_side_usd": COMMISSION_PER_LEG_SIDE},
            "training": {"model": "ridge", "alpha": RIDGE_ALPHA, "winsor": list(WINSOR),
                         "min_train_rows": MIN_TRAIN_ROWS, "min_train_dates": MIN_TRAIN_DATES,
                         "embargo": "label_day < decision day"},
            "selection": {"rule": "per candidate the allowed expression with the highest prediction; TRADE if > 0, "
                                  "else ABSTAIN", "allowed_strategies": list(ALLOWED_STRATEGIES),
                          "unit": "one decision per candidate; one candidate per ticker and day"}}


def experiment_id() -> str:
    return hashlib.sha256(json.dumps(experiment(), sort_keys=True).encode()).hexdigest()[:12]


def _num(v):
    return red.number(v)


def _legs_at_decision(rs: dict) -> list[dict]:
    """Leg-Quotes zum Entscheidungszeitpunkt: snapshot_legs (Session/timing-v2), sonst die erfassten Legs."""
    legs = rs.get("snapshot_legs") or rs.get("legs") or []
    side = {g.get("symbol"): g.get("side") for g in rs.get("legs") or [] if isinstance(g, dict)}
    out = [dict(g, side=g.get("side") or side.get(g.get("symbol"))) for g in legs if isinstance(g, dict)]
    return sorted(out, key=lambda g: 0 if g.get("side") == "long" else 1)


def net_quote(legs: list[dict]) -> tuple[float | None, float | None]:
    """(Kaufpreis netto, Mid netto) der Struktur: Long-Ask − Short-Bid bzw. Long-Mid − Short-Mid."""
    if not legs:
        return None, None
    long_, short = legs[0], (legs[1] if len(legs) > 1 else None)
    ask, mid = _num(long_.get("ask")), _num(long_.get("mid"))
    if short is not None:
        sb, sm = _num(short.get("bid")), _num(short.get("mid"))
        ask = None if ask is None or sb is None else ask - sb
        mid = None if mid is None or sm is None else mid - sm
    return ask, mid


def contract_features(row: dict) -> dict:
    """Kontrakt-Merkmale, die zum Entscheidungszeitpunkt bekannt waren (nie der spätere Fill oder Mark)."""
    rs, ro = row.get("real_strategy") or {}, row.get("real_option") or {}
    legs = _legs_at_decision(rs)
    out = {}
    if legs:
        long_, short = legs[0], (legs[1] if len(legs) > 1 else None)
        ask, mid = net_quote(legs)
        if ask is not None and mid and mid > 0:
            out["c_entry_spread"] = ask / mid - 1.0
        out["c_is_spread"] = 1.0 if short is not None else 0.0
        sym = str(long_.get("symbol") or "")
        out["c_is_call"] = 1.0 if len(sym) > 15 and sym[-9] == "C" else 0.0
    snap = ro.get("snapshot_quote") or {}
    out["c_iv"] = _num(ro.get("iv"))
    out["c_dte"] = _num(ro.get("dte"))
    strike, spot = _num(ro.get("strike")), _num(row.get("entry_price"))
    out["c_moneyness"] = strike / spot if strike and spot else None
    out["c_delta"] = _num(ro.get("delta"))
    oi = _num(ro.get("open_interest"))
    out["c_log_oi"] = math.log1p(oi) if oi is not None and oi >= 0 else None
    out["c_iv_rank"] = _num(rs.get("iv_rank"))
    comps = rs.get("iv_rank_components") or {}          # IV relativ zur realisierten Vola und zur Term-Struktur
    out["c_rv_score"] = _num(comps.get("rv_score"))
    out["c_term_score"] = _num(comps.get("term_score"))
    if "c_entry_spread" not in out and snap:
        ask, mid = _num(snap.get("ask")), _num(snap.get("mid"))
        if ask is not None and mid and mid > 0:
            out["c_entry_spread"] = ask / mid - 1.0
    return {k: v for k, v in out.items() if v is not None}


def features(row: dict) -> dict:
    f = {k: _num((row.get("features") or {}).get(k)) for k in LEDGER_FEATURES}
    f.update(contract_features(row))
    return {k: v for k, v in f.items() if v is not None}


def net_label(row: dict, horizon: int) -> float | None:
    """Ausführbare Rendite (Kauf Ask, Verkauf Bid) minus Kommission je Leg und Seite."""
    rs = row.get("real_strategy") or {}
    gross = _num((row.get("outcomes") or {}).get(f"real_strat_ret_{horizon}d"))
    debit = _num(rs.get("net_debit_entry"))
    if gross is None or not debit or debit <= 0:
        return None
    legs = max(1, len(rs.get("legs") or []))
    return gross - 2 * legs * COMMISSION_PER_LEG_SIDE / (debit * 100)


def _label_day(row: dict, horizon: int) -> date | None:
    try:
        return date.fromisoformat(str((row.get("outcomes") or {}).get(f"real_strat_mark_date_{horizon}d")))
    except ValueError:
        return None


def _right(symbol) -> str | None:
    s = str(symbol or "")
    return s[-9] if len(s) > 15 and s[-9] in "CP" else None


def _strategy_like(ro: dict) -> dict:
    """Einzel-Option in der Form von real_strategy, damit Prüfung, Label und Merkmale gleich laufen."""
    sym, snap = ro.get("symbol"), ro.get("snapshot_quote") or {}
    rs = {"strategy": "LONG_CALL" if _right(sym) == "C" else "LONG_PUT",
          "legs": [{"symbol": sym, "side": "long", "bid": ro.get("bid"), "ask": ro.get("ask"), "mid": ro.get("mid")}],
          "net_debit_entry": ro.get("ask"), "net_mid_entry": ro.get("mid"), "expiry": ro.get("expiry")}
    if snap:
        rs["snapshot_legs"] = [{"symbol": sym, "side": "long", "bid": snap.get("bid"), "ask": snap.get("ask"),
                                "mid": snap.get("mid")}]
    rs.update({k: ro[k] for k in ("entry_pending", "entry_status", "entry_date", "entry_quote_record_ids")
               if k in ro})
    return rs


def expressions(row: dict) -> list[dict]:
    """Ausdrucksformen eines Kandidaten als Pseudo-Zeilen im einheitlichen real_strategy/outcomes-Schema."""
    out, seen = [], set()
    oc = row.get("outcomes") or {}
    base_rs = row.get("real_strategy") if isinstance(row.get("real_strategy"), dict) else {}
    for key, prefix, name in SOURCES:
        spec = row.get(key)
        if not isinstance(spec, dict):
            continue
        if key == "real_strategy":
            if not spec.get("legs"):
                continue
            rs, contract = spec, row.get("real_option") or {}
        else:
            if not spec.get("symbol"):
                continue
            rs, contract = _strategy_like(spec), spec
        symbols = tuple(g.get("symbol") for g in sorted(rs.get("legs") or [],
                                                        key=lambda g: 0 if g.get("side") == "long" else 1))
        if symbols in seen:
            continue
        seen.add(symbols)
        mapped = {"real_strat" + k[len(prefix):]: v for k, v in oc.items() if k.startswith(prefix + "_")}
        pseudo = {**row, "real_strategy": {**rs, "iv_rank": rs.get("iv_rank", base_rs.get("iv_rank")),
                                           "iv_rank_components": rs.get("iv_rank_components",
                                                                        base_rs.get("iv_rank_components"))},
                  "real_option": contract, "outcomes": mapped}
        out.append({"expression": name or ("spread" if len(symbols) > 1 else "single"),
                    "contract": list(symbols), "row": pseudo})
    return out


def _sample(row: dict) -> str:
    """Forschungsstichprobe (Call ohne LLM-Richtung) oder produktives Richtungssignal (inkl. Altbestand)."""
    return "research_census" if row.get("option_sample") == "research_census" else "directional"


def _direction(strategy) -> str:
    s = str(strategy or "")
    return {"LONG_CALL": "bullish", "BULL_CALL_SPREAD": "bullish", "LONG_PUT": "bearish",
            "BEAR_PUT_SPREAD": "bearish"}.get(s, "unknown")


def candidates(rows: list[dict], quotes: list[dict], as_of: date, horizon: int = HORIZON,
               counts: dict | None = None) -> list[dict]:
    """Je Ledger-Kandidat alle zum Entscheidungszeitpunkt erfassten Ausdrucksformen mit Merkmalen, Entscheidungs-
    Quotes und Label-Status: VERIFIED (mit y und Mark-Tag), ein Prüf-Grund aus research_evidence_discovery.provenance
    oder NO_LABEL (bis as_of kein Label). `counts` erhält die Prüfergebnisse aller vorhandenen Labels."""
    index = red.quote_index(quotes)
    seen, out = set(), []
    for r in rows:
        if not isinstance(r, dict) or not r.get("date") or not r.get("ticker"):
            continue
        key = (r["date"], r["ticker"], str(r.get("signal_id") or r.get("event_id") or ""))
        if key in seen:
            continue
        seen.add(key)
        exprs = []
        for e in expressions(r):
            pr = e["row"]
            ld, y = _label_day(pr, horizon), net_label(pr, horizon)
            status = "NO_LABEL" if ld is None or ld > as_of or y is None else red.provenance(pr, index, horizon)
            if counts is not None and status != "NO_LABEL":
                counts[status] = counts.get(status, 0) + 1
            ok = status == "VERIFIED"
            ask, mid = net_quote(_legs_at_decision(pr["real_strategy"]))
            exprs.append({"expression": e["expression"], "contract": e["contract"],
                          "strategy": pr["real_strategy"].get("strategy"), "x": features(pr), "ask": ask, "mid": mid,
                          "status": status, "y": y if ok else None, "label_day": ld if ok else None})
        if exprs:
            out.append({"date": r["date"], "signal_id": r.get("signal_id"), "ticker": r["ticker"],
                        "sample": _sample(r), "gate": r.get("reject_reason") or r.get("status") or "unknown",
                        "exprs": exprs})
    return sorted(out, key=lambda c: (c["date"], c["ticker"], str(c["signal_id"])))


def _verified(cands: list[dict]) -> list[dict]:
    out = [{"date": c["date"], "label_day": e["label_day"], "signal_id": c["signal_id"], "ticker": c["ticker"],
            "expression": e["expression"], "contract": e["contract"], "x": e["x"], "y": e["y"], "gate": c["gate"]}
           for c in cands for e in c["exprs"] if e["status"] == "VERIFIED"]
    return sorted(out, key=lambda d: (d["date"], d["ticker"], d["expression"]))


def dataset(rows: list[dict], quotes: list[dict], as_of: date, horizon: int = HORIZON,
            counts: dict | None = None) -> list[dict]:
    """Geprüfte Labels mit Mark-Tag <= as_of je Ausdrucksform (Trainingsdaten): {date, label_day, signal_id, ticker,
    expression, contract, x, y, gate}."""
    return _verified(candidates(rows, quotes, as_of, horizon, counts))


class Model:
    """Median-Imputation + Standardisierung + Ridge (geschlossene Lösung, deterministisch)."""

    def __init__(self, names, med, mu, sd, coef, intercept, clip):
        self.names, self.med, self.mu, self.sd = names, med, mu, sd
        self.coef, self.intercept, self.clip = coef, intercept, clip

    def _matrix(self, xs):
        m = np.array([[x.get(n, self.med[i]) for i, n in enumerate(self.names)] for x in xs], dtype=float)
        return (m - self.mu) / self.sd

    def predict(self, xs) -> np.ndarray:
        if not xs:
            return np.zeros(0)
        return self._matrix(xs) @ self.coef + self.intercept

    def describe(self) -> dict:
        order = np.argsort(-np.abs(self.coef))
        return {"features": len(self.names), "clip": [round(c, 4) for c in self.clip],
                "intercept": round(float(self.intercept), 6),
                "top_coefficients": {self.names[i]: round(float(self.coef[i]), 6) for i in order[:8]}}


def fit(train: list[dict]) -> Model | None:
    if len(train) < MIN_TRAIN_ROWS:
        return None
    names = [n for n in (*LEDGER_FEATURES, *CONTRACT_FEATURES)
             if sum(n in d["x"] for d in train) >= MIN_FEATURE_COVERAGE * len(train)]
    if not names:
        return None
    med = [float(np.median([d["x"][n] for d in train if n in d["x"]])) for n in names]
    raw = np.array([[d["x"].get(n, med[i]) for i, n in enumerate(names)] for d in train], dtype=float)
    mu, sd = raw.mean(axis=0), raw.std(axis=0)
    sd[sd == 0] = 1.0
    x = (raw - mu) / sd
    y = np.array([d["y"] for d in train], dtype=float)
    lo, hi = np.quantile(y, WINSOR)
    y = np.clip(y, lo, hi)
    intercept = float(y.mean())
    coef = np.linalg.solve(x.T @ x + RIDGE_ALPHA * np.eye(len(names)), x.T @ (y - intercept))
    return Model(names, med, mu, sd, coef, intercept, (float(lo), float(hi)))


def _spearman(a, b) -> float | None:
    if len(a) < 5:
        return None
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def _periods(by_date: dict[str, list], horizon: int) -> dict[int, list]:
    """Signaltage in nicht überlappende Haltefenster der Länge `horizon` (Kalendertage) bündeln: Positionen
    innerhalb eines Fensters überlappen und sind nicht unabhängig."""
    days = sorted(by_date)
    if not days:
        return {}
    first = date.fromisoformat(days[0]).toordinal()
    out = defaultdict(list)
    for d in days:
        out[(date.fromisoformat(d).toordinal() - first) // max(1, horizon)].extend(by_date[d])
    return dict(out)


def _block_bootstrap_ci(by_date: dict[str, list[float]], horizon: int, seed: int = 7) -> tuple[float, float] | None:
    """95-%-KI des gepoolten Mittels; gezogen werden ganze Haltefenster (Block-Bootstrap), nicht einzelne Tage."""
    blocks = [b for b in _periods(by_date, horizon).values() if b]
    if len(blocks) < 2:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(BOOTSTRAP):
        vals = [v for b in (rng.choice(blocks) for _ in blocks) for v in b]
        means.append(sum(vals) / len(vals))
    means.sort()
    return means[int(0.025 * BOOTSTRAP)], means[int(0.975 * BOOTSTRAP) - 1]


def score(preds: list[dict], horizon: int = HORIZON) -> dict:
    """Kennzahlen für Prognosen [{date, pred, y}]: Rang-IC je Tag, Mittel der Auswahl (pred > 0) mit Block-Bootstrap
    über nicht überlappende Haltefenster (`independent_periods`)."""
    by_date = defaultdict(list)
    for p in preds:
        by_date[p["date"]].append(p)
    ics = [ic for ps in by_date.values()
           if (ic := _spearman([p["pred"] for p in ps], [p["y"] for p in ps])) is not None]
    sel = defaultdict(list)
    for p in preds:
        if p["pred"] > 0:
            sel[p["date"]].append(p["y"])
    n_sel = sum(len(v) for v in sel.values())
    ic_t = None
    if len(ics) >= 3 and np.std(ics, ddof=1) > 0:
        ic_t = float(np.mean(ics) / (np.std(ics, ddof=1) / math.sqrt(len(ics))))
    ci = _block_bootstrap_ci(sel, horizon) if n_sel else None
    return {"n": len(preds), "dates": len(by_date), "independent_periods": len(_periods(sel, horizon)),
            "mean_all": round(float(np.mean([p["y"] for p in preds])), 6) if preds else None,
            "ic_mean": round(float(np.mean(ics)), 6) if ics else None,
            "ic_t": round(ic_t, 3) if ic_t is not None else None,
            "n_selected": n_sel, "selected_dates": len(sel),
            "mean_selected": round(sum(v for vs in sel.values() for v in vs) / n_sel, 6) if n_sel else None,
            "selected_ci95": [round(c, 6) for c in ci] if ci else None}


def _by_expression(preds: list[dict]) -> dict:
    """Ergebnis jeder einzelnen Optionsform (ungesehen): alle Fälle und nur die mit Prognose > 0."""
    groups = defaultdict(list)
    for p in preds:
        groups[p.get("expression") or "single"].append(p)
    out = {}
    for name, ps in sorted(groups.items()):
        sel = [p["y"] for p in ps if p["pred"] > 0]
        out[name] = {"n": len(ps), "mean_all": round(float(np.mean([p["y"] for p in ps])), 6),
                     "n_selected": len(sel), "mean_selected": round(float(np.mean(sel)), 6) if sel else None}
    return out


def _insufficient(s: dict) -> bool:
    return (s["dates"] < MIN_OOS_DATES or s["n_selected"] < MIN_SELECTED
            or s["independent_periods"] < MIN_INDEPENDENT_PERIODS)


def _due(day: str, as_of: date, horizon: int) -> bool:
    """Ergebnis spätestens jetzt fällig (Mark-Frist des Ledgers abgelaufen) – unabhängig davon, ob es vorliegt."""
    return (as_of - date.fromisoformat(day)).days > horizon + DUE_GRACE_DAYS


def decide(cand: dict, preds) -> dict:
    """Auswahlregel – nur aus Prognosen, nie aus Ergebnissen: je Kandidat genau eine zulässige Ausdrucksform
    (höchste Prognose unter ALLOWED_STRATEGIES, bei Gleichstand die erste); TRADE wenn > 0, sonst ABSTAIN."""
    elig = [(float(p), e) for p, e in zip(preds, cand["exprs"]) if e["strategy"] in ALLOWED_STRATEGIES]
    if not elig:
        return {"action": "ABSTAIN", "reason": "NO_ALLOWED_EXPRESSION", "expression": None, "contract": None,
                "pred": None}
    p, e = max(elig, key=lambda t: t[0])
    return {"action": "TRADE" if p > 0 else "ABSTAIN", "reason": None if p > 0 else "PRED_NOT_POSITIVE",
            "expression": e["expression"], "contract": e["contract"], "pred": round(p, 6)}


def realized(cand: dict, contract) -> dict | None:
    """Die Ausdrucksform genau des gewählten Kontrakts (mit ihrem Label-Status) – nie eine andere Form."""
    return next((e for e in cand["exprs"] if e["contract"] == list(contract or ())), None)


def _representativeness(picks: list[dict]) -> dict:
    """Woraus die Evidenz besteht: Forschungsstichprobe vs. Richtungssignale, Richtung, Zeitraum – mit Warnungen,
    damit eine einseitige Stichprobe nicht unbemerkt als Beleg für alle Marktsituationen gilt."""
    def group(key):
        g = defaultdict(list)
        for p in picks:
            g[p[key]].append(p["y"])
        return {k: {"n": len(v), "mean": round(float(np.mean(v)), 6)} for k, v in sorted(g.items())}
    n, by_sample, by_direction = len(picks), group("sample"), group("direction")
    days = sorted({p["date"] for p in picks})
    span = (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days + 1 if days else 0
    warnings = []
    if n and by_direction.get("bullish", {}).get("n", 0) >= 0.8 * n:
        warnings.append("überwiegend bullische Call-Strukturen: keine Aussage für bearishe Signale")
    if n and by_sample.get("research_census", {}).get("n", 0) >= 0.8 * n:
        warnings.append("überwiegend Forschungsstichprobe (Call ohne Richtungssignal): nicht automatisch auf "
                        "produktive Richtungssignale übertragbar")
    if n and span < MIN_SPAN_DAYS:
        warnings.append(f"nur {span} Kalendertage: höchstens eine Marktphase")
    return {"by_sample": by_sample, "by_direction": by_direction, "span_days": span, "warnings": warnings}


def _paired(by_date: dict[str, list[float]], horizon: int) -> dict:
    vals = [v for vs in by_date.values() for v in vs]
    ci = _block_bootstrap_ci(by_date, horizon) if vals else None
    return {"n": len(vals), "dates": len(by_date), "independent_periods": len(_periods(by_date, horizon)),
            "mean_diff": round(float(np.mean(vals)), 6) if vals else None,
            "ci95": [round(c, 6) for c in ci] if ci else None}


def assess(records: list[dict], horizon: int) -> dict:
    """Bewertet vorab getroffene Auswahlentscheidungen [{cand, dec, preds}] auf fälligen Kandidaten. Gezählt wird nur
    das geprüfte Netto-Ergebnis der gewählten Form; fehlt es, ist der Fall nicht bewertbar (mit Grund) und nie
    positiv. Je Aktie und Tag zählt ein Kandidat (der erste nach Signal-ID), Varianten nie getrennt."""
    seen, picks, expr_preds, vs_default = set(), [], [], defaultdict(list)
    unevaluable, forms, abstained = defaultdict(int), defaultdict(int), 0
    for rec in sorted(records, key=lambda r: (r["cand"]["date"], r["cand"]["ticker"], str(r["cand"]["signal_id"]))):
        c, dec = rec["cand"], rec["dec"]
        if (c["date"], c["ticker"]) in seen:
            continue
        seen.add((c["date"], c["ticker"]))
        expr_preds += [{"date": c["date"], "pred": p, "y": e["y"], "expression": e["expression"]}
                       for e in c["exprs"] if e["status"] == "VERIFIED"
                       and (p := rec["preds"].get(tuple(e["contract"]))) is not None]
        if dec.get("action") != "TRADE":
            abstained += 1
            continue
        e = realized(c, dec.get("contract"))
        if e is None or e["status"] != "VERIFIED":
            unevaluable[e["status"] if e else "CONTRACT_NOT_IN_CANDIDATE"] += 1
            continue
        forms[e["expression"]] += 1
        picks.append({"date": c["date"], "pred": 1.0, "y": e["y"], "sample": c["sample"],
                      "direction": _direction(e["strategy"])})
        default = c["exprs"][0]                                   # Standardform: Produktions-Strategie bzw. real_option
        if default["status"] == "VERIFIED":
            vs_default[c["date"]].append(e["y"] - default["y"])
    sel, rank = score(picks, horizon), score(expr_preds, horizon)
    trades = len(picks) + sum(unevaluable.values())
    lost = sum(p["y"] for p in picks) - sum(unevaluable.values())
    return {"n": len(seen), "dates": len({d for d, _t in seen}), "independent_periods": sel["independent_periods"],
            "mean_all": rank["mean_all"], "ic_mean": rank["ic_mean"], "ic_t": rank["ic_t"],
            "n_selected": sel["n_selected"], "selected_dates": sel["selected_dates"],
            "mean_selected": sel["mean_selected"], "selected_ci95": sel["selected_ci95"],
            "trades": trades, "abstained": abstained, "unevaluable": dict(sorted(unevaluable.items())),
            "coverage": round(len(picks) / trades, 4) if trades else None,
            "mean_selected_unevaluable_as_loss": round(lost / trades, 6) if trades else None,
            "chosen_forms": dict(sorted(forms.items())), "by_expression": _by_expression(expr_preds),
            "vs_default": _paired(vs_default, horizon), "representativeness": _representativeness(picks)}


def compare_policies(pairs: list[dict], horizon: int) -> dict:
    """Gepaarter Vergleich [{cand, learning, reference}] auf denselben fälligen Kandidaten: je Kandidat ein gleiches
    Budget (eine Einheit Prämie; ABSTAIN = Einheit bleibt in Cash = 0), je Tag das Mittel über alle Kandidaten des
    Tages – für lernende Version, Referenz, "alles handeln" (Standardform, sofern zulässig) und "nie handeln" (0).
    Ein Kandidat zählt nur, wenn das Ergebnis jeder gehandelten Wahl geprüft vorliegt (sonst für alle
    ausgeschlossen); je Aktie und Tag einer. Unsicherheit der Tagesdifferenz per Block-Bootstrap.
    Vorab festgelegt: NEED_MORE_DATA unter den Mindestzahlen; LEARNING_BETTER / LEARNING_WORSE, wenn das 95-%-KI
    der mittleren Tagesdifferenz (lernend − Referenz) ganz über / unter 0 liegt; sonst NO_SIGNIFICANT_DIFFERENCE."""
    seen, excluded, differ, daily = set(), defaultdict(int), 0, defaultdict(lambda: defaultdict(list))
    for pr in sorted(pairs, key=lambda p: (p["cand"]["date"], p["cand"]["ticker"], str(p["cand"]["signal_id"]))):
        c = pr["cand"]
        if (c["date"], c["ticker"]) in seen:
            continue
        seen.add((c["date"], c["ticker"]))
        default = c["exprs"][0]
        trade_all = ({"action": "TRADE", "contract": default["contract"]}
                     if default["strategy"] in ALLOWED_STRATEGIES else {"action": "ABSTAIN"})
        r, missing = {}, None
        for name, dec in (("learning", pr["learning"]), ("reference", pr["reference"]), ("trade_all", trade_all)):
            if dec.get("action") != "TRADE":
                r[name] = 0.0
                continue
            e = realized(c, dec.get("contract"))
            if e is None or e["status"] != "VERIFIED":
                missing = e["status"] if e else "CONTRACT_NOT_IN_CANDIDATE"
                break
            r[name] = e["y"]
        if missing:
            excluded[missing] += 1
            continue
        differ += ((pr["learning"].get("action"), pr["learning"].get("contract"))
                   != (pr["reference"].get("action"), pr["reference"].get("contract")))
        for name, v in r.items():
            daily[c["date"]][name].append(v)
    days = sorted(daily)
    day_mean = {d: {k: sum(v) / len(v) for k, v in daily[d].items()} for d in days}

    def diff(a, b):
        res = _paired({d: [day_mean[d][a] - (day_mean[d][b] if b else 0.0)] for d in days}, horizon)
        return {"mean": res["mean_diff"], "ci95": res["ci95"], "independent_periods": res["independent_periods"]}
    lr, n = diff("learning", "reference"), sum(len(daily[d]["learning"]) for d in days)
    if n < MIN_SELECTED or len(days) < MIN_OOS_DATES or lr["independent_periods"] < MIN_INDEPENDENT_PERIODS:
        verdict = "NEED_MORE_DATA"
    elif lr["ci95"] and lr["ci95"][0] > 0:
        verdict = "LEARNING_BETTER"
    elif lr["ci95"] and lr["ci95"][1] < 0:
        verdict = "LEARNING_WORSE"
    else:
        verdict = "NO_SIGNIFICANT_DIFFERENCE"
    return {"candidates": n, "days": len(days), "independent_periods": lr["independent_periods"],
            "decisions_differ": differ, "excluded": dict(sorted(excluded.items())),
            "daily_mean": {k: (round(float(np.mean([day_mean[d][k] for d in days])), 6) if days else None)
                           for k in ("learning", "reference", "trade_all")} | {"no_trade": 0.0},
            "learning_minus_reference": lr, "learning_minus_trade_all": diff("learning", "trade_all"),
            "learning_minus_no_trade": diff("learning", None), "verdict": verdict}


def _verdict(s: dict, kind: str) -> str:
    if _insufficient(s):
        return "NEED_MORE_DATA"
    if (s["coverage"] or 0) < MIN_COVERAGE:
        return "INSUFFICIENT_COVERAGE"
    ci = s["selected_ci95"]
    if kind == "walk_forward":
        return "VALIDATED" if ci and ci[0] > 0 and (s["ic_t"] or 0) > IC_T_MIN else "NOT_VALIDATED"
    if ci and ci[1] < 0:
        return "FORWARD_NEGATIVE"
    return "FORWARD_POSITIVE" if ci and ci[0] > 0 else "INCONCLUSIVE"


def walk_forward(cands: list[dict], as_of: date, horizon: int = HORIZON) -> dict:
    """Historische Simulation der Auswahlregel: je Testtag nur mit Labels trainiert, deren Mark-Tag davor liegt; je
    fälligem Kandidat zuerst genau eine Form gewählt (decide), erst danach ihr geprüftes Ergebnis betrachtet."""
    labels = _verified(cands)
    by_day = defaultdict(list)
    for c in cands:
        if _due(c["date"], as_of, horizon):
            by_day[c["date"]].append(c)
    records, pairs, ref, ref_day = [], [], None, None
    for day in sorted(by_day):
        test_day = date.fromisoformat(day)
        train = [d for d in labels if d["label_day"] < test_day]
        if len({d["date"] for d in train}) < MIN_TRAIN_DATES:
            continue
        model = fit(train)
        if model is None:
            continue
        ref, ref_day = (ref, ref_day) if ref is not None else (model, day)     # erste Instanz bleibt eingefroren
        for c in by_day[day]:
            xs = [e["x"] for e in c["exprs"]]
            preds = model.predict(xs)
            dec = decide(c, preds)
            records.append({"cand": c, "dec": dec,
                            "preds": {tuple(e["contract"]): round(float(p), 6) for e, p in zip(c["exprs"], preds)}})
            pairs.append({"cand": c, "learning": dec, "reference": decide(c, ref.predict(xs))})
    s = assess(records, horizon)
    return {**s, "verdict": _verdict(s, "walk_forward"),
            "vs_reference": {**compare_policies(pairs, horizon), "mode": "retrospective", "reference_cutoff": ref_day}}


def _invalid(j: dict, exp: str, instances: dict) -> str | None:
    """Grund, warum ein Journal-Eintrag keine prospektive Evidenz der Experimentversion `exp` ist (sonst None):
    andere Version, unbekannte Modellinstanz, nicht am Signaltag geschrieben oder mit Labels ab dem Signaltag
    trainiert."""
    if (j.get("experiment_id") or j.get("model_spec")) != exp:
        return "OTHER_EXPERIMENT"
    inst = instances.get(j.get("model_id"))
    if inst is None:
        return "UNKNOWN_MODEL_INSTANCE"
    if inst.get("experiment_id") != exp:
        return "INSTANCE_EXPERIMENT_MISMATCH"
    try:
        created = datetime.fromisoformat(str(j["created_at"])).astimezone(timezone.utc).date()
        signal = date.fromisoformat(str(j["date"]))
    except (KeyError, ValueError, TypeError):
        return "NO_TIMESTAMP"
    if created > signal:
        return "LATE_PREDICTION"
    day = signal.isoformat()
    if str(inst.get("max_label_day") or "9999") >= day or str(inst.get("train_cutoff")) > day:
        return "TRAINED_ON_FUTURE"
    return None


def _jkey(j: dict) -> tuple:
    kind = j.get("kind", "prediction")
    return (kind, j.get("role", "learning"), j.get("date"), j.get("signal_id"), j.get("horizon"),
            tuple(j.get("contract") or ()) if kind == "prediction" else None,
            j.get("experiment_id") or j.get("model_spec"))


def forward(journal: list[dict], cands: list[dict], horizon: int, instances: list[dict] | None = None,
            as_of: date | None = None, exp: str | None = None) -> dict:
    """Prospektiver Test der Auswahlregel: bewertet die am Signaltag im Journal festgehaltenen Entscheidungen
    (kind=decision, role=learning) – nie neu berechnet – auf inzwischen fälligen Kandidaten. Es zählen nur
    unveränderte Einträge der Experimentversion `exp` (Standard: aktuelle Version); fortlaufend neu trainierte
    Modellinstanzen derselben Version gemeinsam."""
    exp = exp or experiment_id()
    as_of = as_of or datetime.now(timezone.utc).date()
    reg = {i.get("model_id"): i for i in instances or [] if isinstance(i, dict)}
    by_cand = {(c["date"], c["signal_id"]): c for c in cands}
    preds, decisions, ref_decisions, seen = defaultdict(dict), {}, {}, set()
    excluded, by_exp = defaultdict(int), defaultdict(int)
    for j in journal:
        if not isinstance(j, dict) or j.get("horizon") != horizon:
            continue
        by_exp[str(j.get("experiment_id") or j.get("model_spec") or "unbekannt")] += 1
        key = _jkey(j)
        reason = _invalid(j, exp, reg) or ("DUPLICATE" if key in seen else None)
        if reason:
            excluded[reason] += 1
            continue
        seen.add(key)
        ck = (j["date"], j.get("signal_id"))
        if j.get("kind") == "decision":
            (decisions if j.get("role", "learning") == "learning" else ref_decisions)[ck] = j
        else:
            preds[ck][tuple(j.get("contract") or ())] = float(j["pred_net"])
    records, pairs, pending, models = [], [], 0, set()
    for ck, j in decisions.items():
        if not _due(j["date"], as_of, horizon):
            pending += 1
            continue
        if ck not in by_cand:
            excluded["CANDIDATE_NOT_IN_LEDGER"] += 1
            continue
        models.add(j["model_id"])
        dec = {"action": j.get("action"), "contract": j.get("contract"), "pred": j.get("pred_net")}
        records.append({"cand": by_cand[ck], "dec": dec, "preds": preds.get(ck, {})})
        if ck in ref_decisions:
            r = ref_decisions[ck]
            pairs.append({"cand": by_cand[ck], "learning": dec,
                          "reference": {"action": r.get("action"), "contract": r.get("contract")}})
    s = assess(records, horizon)
    return {**s, "verdict": _verdict(s, "forward"), "experiment_id": exp, "valid_decisions": len(decisions),
            "valid_predictions": sum(len(v) for v in preds.values()), "pending_decisions": pending,
            "model_instances": len(models), "excluded": dict(sorted(excluded.items())),
            "journal_by_experiment": dict(sorted(by_exp.items())),
            "vs_reference": {**compare_policies(pairs, horizon), "mode": "forward",
                             "reference_models": sorted({r.get("model_id") for r in ref_decisions.values()})}}


def release_gate(wf: dict | None, fw: dict | None) -> dict:
    """Freigabe von Empfehlungen – fail closed. Erlaubt nur bei Walk-Forward VALIDATED UND Forward FORWARD_POSITIVE;
    die Mindestzahlen und Evidenzanforderungen werden aus den Kennzahlen erneut geprüft, damit ein fehlender,
    unvollständiger oder widersprüchlicher Bericht (z. B. Verdikt ohne passende Zahlen) nie freigibt."""
    reasons = []
    for name, s, want in (("walk_forward", wf, "VALIDATED"), ("forward", fw, "FORWARD_POSITIVE")):
        if not isinstance(s, dict) or not s.get("verdict"):
            reasons.append(f"{name}: Bericht fehlt")
            continue
        if s["verdict"] != want:
            reasons.append(f"{name}: {s['verdict']} (verlangt {want})")
        try:
            ci = s.get("selected_ci95")
            checks = ((s["dates"] >= MIN_OOS_DATES, f"Testtage {s['dates']} < {MIN_OOS_DATES}"),
                      (s["n_selected"] >= MIN_SELECTED, f"Auswahl n={s['n_selected']} < {MIN_SELECTED}"),
                      (s["independent_periods"] >= MIN_INDEPENDENT_PERIODS,
                       f"unabhängige Perioden {s['independent_periods']} < {MIN_INDEPENDENT_PERIODS}"),
                      (bool(ci) and ci[0] > 0, f"95-%-KI {ci} nicht über 0"),
                      (s.get("mean_selected") is not None and s["mean_selected"] > 0, "Ø netto der Auswahl nicht > 0"),
                      (s["coverage"] is not None and s["coverage"] >= MIN_COVERAGE,
                       f"Abdeckung geprüfter Ergebnisse {s['coverage']} < {MIN_COVERAGE}"))
            if name == "walk_forward":
                checks += (((s.get("ic_t") or 0) > IC_T_MIN, f"Rang-IC t {s.get('ic_t')} <= {IC_T_MIN}"),)
            else:
                checks += ((s["experiment_id"] == experiment_id(), f"Experimentversion {s['experiment_id']} veraltet"),)
        except (KeyError, TypeError, IndexError):
            reasons.append(f"{name}: Kennzahlen unvollständig")
            continue
        reasons += [f"{name}: {text}" for ok, text in checks if not ok]
    return {"allowed": not reasons, "reasons": reasons}


def _load(as_of: date, horizon: int, ledger: Path, quotes: Path, counts: dict | None = None):
    """(Ledger-Zeilen, Kandidaten mit allen Formen und Label-Status, geprüfte Trainings-Labels)."""
    rows = red.load(ledger, as_of)
    labelled = [r for r in rows if isinstance(r, dict) and any(
        _num((r.get("outcomes") or {}).get(f"{prefix}_ret_{horizon}d")) is not None for _k, prefix, _n in SOURCES)]
    cands = candidates(rows, red.load_quotes(quotes, labelled, as_of), as_of, horizon, counts)
    return rows, cands, _verified(cands)


def _label_ids(data: list[dict]) -> list[str]:
    return sorted(f"{d['date']}|{d['signal_id']}|{'/'.join(d['contract'])}|{d['label_day']}" for d in data)


def instance(model: Model, train: list[dict], cutoff: date) -> dict:
    """Trainierte Modellinstanz: Experimentversion, Trainingsstichtag, Manifest der verwendeten Label-IDs und
    Fingerprint der Gewichte. Die Label-IDs sind aus dem Ledger über (Signal, Kontrakt, Mark-Tag < Stichtag)
    reproduzierbar; gespeichert wird ihr Hash."""
    labels_sha = hashlib.sha256("\n".join(_label_ids(train)).encode()).hexdigest()[:16]
    weights = np.round(np.concatenate([model.coef, [model.intercept], model.mu, model.sd]), 10)
    weights_sha = hashlib.sha256(weights.tobytes() + "|".join(model.names).encode()).hexdigest()[:16]
    exp = experiment_id()
    return {"model_id": hashlib.sha256(f"{exp}|{labels_sha}|{weights_sha}".encode()).hexdigest()[:16],
            "experiment_id": exp, "train_cutoff": cutoff.isoformat(), "n_train_labels": len(train),
            "labels_sha256": labels_sha, "weights_sha256": weights_sha,
            "max_label_day": max(d["label_day"] for d in train).isoformat(), "features": list(model.names)}


def _register(out: Path, inst: dict, horizon: int, now: datetime) -> None:
    """Modellinstanz append-only registrieren (einmal je model_id und Horizont)."""
    path = out / INSTANCES
    known = {(i.get("model_id"), i.get("horizon")) for i in (read_jsonl(path) if path.exists() else [])}
    if (inst["model_id"], horizon) not in known:
        append_jsonl(path, [{**inst, "horizon": horizon, "registered_at": now.isoformat(timespec="seconds")}])


def reference(out: Path, horizon: int, model: Model | None = None, inst: dict | None = None,
              now: datetime | None = None) -> tuple[Model | None, dict | None]:
    """Eingefrorene Referenzversion je Experimentversion und Horizont: die erste trainierte Modellinstanz. Ihre
    Gewichte werden einmalig gespeichert und nie überschrieben; eine neue Experimentversion beginnt eine neue
    Vergleichskohorte (eigene Datei). Ohne `model` wird nur gelesen."""
    path = out / REFERENCE_DIR / f"{horizon}d_{experiment_id()}.json"
    if not path.exists():
        if model is None or inst is None:
            return None, None
        w = {"names": list(model.names), "med": list(model.med), "mu": model.mu.tolist(), "sd": model.sd.tolist(),
             "coef": model.coef.tolist(), "intercept": float(model.intercept), "clip": list(model.clip)}
        atomic_write_json(path, {**{k: v for k, v in inst.items() if k != "features"}, "horizon": horizon,
                                 "frozen_at": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
                                 "weights": w}, indent=1)
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
        w = meta["weights"]
        return Model(w["names"], w["med"], np.array(w["mu"]), np.array(w["sd"]), np.array(w["coef"]),
                     float(w["intercept"]), tuple(w["clip"])), {k: v for k, v in meta.items() if k != "weights"}
    except (OSError, ValueError, KeyError, TypeError) as exc:      # nie stillschweigend neu einfrieren
        log.warning("option_learner: Referenz %s unlesbar (%s) – kein Referenzvergleich", path, exc)
        return None, None


def manifest(data: list[dict], as_of: date) -> dict:
    """Welche geprüften Labels in das Modell zum Stichtag flossen (Hash über Signal, Kontrakt, Mark-Tag)."""
    ids = _label_ids(data)
    by_expr = defaultdict(int)
    for d in data:
        by_expr[d["expression"]] += 1
    return {"cutoff": as_of.isoformat(), "n_labels": len(data),
            "labels_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16],
            "first_signal": ids[0][:10] if ids else None, "last_signal": max((d["date"] for d in data), default=None),
            "by_expression": dict(sorted(by_expr.items())), "all_labels_verified": True,
            "trains_model": len(data) >= MIN_TRAIN_ROWS}


MIN_GATE_N, MIN_GATE_DATES = 20, 10


def by_gate(data: list[dict]) -> dict:
    """Filter-Prüfung (Systemaudit §9): realisiertes Netto je Ablehnungsgrund der geprüften Labels – beschreibend,
    zeigt, ob ein Filter eher schlechte oder auch gute Options-Gelegenheiten verwirft."""
    groups = defaultdict(list)
    for d in data:
        groups[d.get("gate") or "unknown"].append(d)
    out = {}
    for gate, ds in sorted(groups.items()):
        ys = [d["y"] for d in ds]
        out[gate] = {"n": len(ys), "dates": len({d["date"] for d in ds}), "mean": round(float(np.mean(ys)), 6),
                     "win_rate": round(sum(y > 0 for y in ys) / len(ys), 4),
                     "sufficient": len(ys) >= MIN_GATE_N and len({d["date"] for d in ds}) >= MIN_GATE_DATES}
    return out


def _baselines(cmp: dict) -> dict:
    """Modell gegen "nie handeln" (0) und "alles handeln" (Standardform) auf denselben Kandidaten und Tagen, mit
    gleichem Budget je Kandidat (Mittel der Tagesrendite); "schlägt" nur, wenn das gepaarte KI über 0 liegt."""
    dm = cmp.get("daily_mean") or {}
    lt, ln = cmp.get("learning_minus_trade_all") or {}, cmp.get("learning_minus_no_trade") or {}
    return {"no_trade": 0.0, "trade_all": dm.get("trade_all"), "model": dm.get("learning"), "days": cmp.get("days", 0),
            "beats_no_trade": bool(ln.get("ci95") and ln["ci95"][0] > 0),
            "beats_trade_all": bool(lt.get("ci95") and lt["ci95"][0] > 0)}


def _vs_previous(history: list[dict], status: dict) -> dict | None:
    """Beschreibende Lernkurve (aggregiert, nur innerhalb derselben Experimentversion)."""
    prev = next((h for h in reversed(history) if h.get("as_of") < status["as_of"]
                 and h.get("experiment_id") == status["experiment_id"]), None)
    if prev is None:
        return None
    cur, old = status["walk_forward"], prev.get("walk_forward") or {}

    def delta(k):
        return None if cur.get(k) is None or old.get(k) is None else round(cur[k] - old[k], 6)
    return {"previous_as_of": prev["as_of"], "previous_verdict": old.get("verdict"),
            "delta_mean_selected": delta("mean_selected"), "delta_ic_mean": delta("ic_mean"),
            "delta_labels": status["manifest"]["n_labels"] - (prev.get("manifest") or {}).get("n_labels", 0)}


def evaluate(as_of: date, *, horizon: int = HORIZON, ledger: Path = red.LEDGER, quotes: Path = red.QUOTES,
             out: Path = OUT) -> dict:
    counts: dict = {}
    rows, cands, data = _load(as_of, horizon, ledger, quotes, counts)
    journal = read_jsonl(out / JOURNAL) if (out / JOURNAL).exists() else []
    instances = read_jsonl(out / INSTANCES) if (out / INSTANCES).exists() else []
    model = fit(data)
    wf = walk_forward(cands, as_of, horizon)
    week = (as_of.toordinal() - 7)
    recent = [r for r in rows if isinstance(r, dict) and r.get("date")
              and date.fromisoformat(str(r["date"])[:10]).toordinal() > week]
    status = {"as_of": as_of.isoformat(), "horizon": horizon, "experiment": experiment(),
              "experiment_id": experiment_id(), "evidence_requirements": EVIDENCE,
              "verified_labels": len(data), "label_dates": len({d["date"] for d in data}),
              "provenance_counts": dict(sorted(counts.items())),
              "ledger_rows_7d": len(recent),
              "option_snapshots_7d": sum(1 for r in recent if isinstance(r.get("real_option"), dict)),
              "manifest": manifest(data, as_of), "walk_forward": wf,
              "baselines": _baselines(wf.get("vs_reference") or {}),
              "by_gate": by_gate(data),
              "forward": forward(journal, cands, horizon, instances, as_of),
              "model": model.describe() if model else None,
              "production_influence": "NONE", "orders_created": 0}
    mine = [i for i in instances if i.get("horizon") == horizon and i.get("experiment_id") == status["experiment_id"]]
    status["model_instances"] = {"count": len(mine), "latest": {k: v for k, v in mine[-1].items() if k != "features"}
                                 if mine else None}
    status["reference"] = reference(out, horizon)[1]
    hist_path = out / f"status_history_{horizon}d.jsonl"
    history = read_jsonl(hist_path) if hist_path.exists() else []
    def progress(as_of, fw):                     # Verlauf des gepaarten Forward-Vergleichs (gleiche Version)
        cmp = (fw or {}).get("vs_reference") or {}
        return {"as_of": as_of, "verdict": cmp.get("verdict"),
                "mean_diff": (cmp.get("learning_minus_reference") or {}).get("mean")}
    status["learning_trend"] = [progress(h["as_of"], h.get("forward")) for h in history
                                if h.get("experiment_id") == status["experiment_id"]
                                and h["as_of"] < status["as_of"]][-4:] + [progress(status["as_of"], status["forward"])]
    status["vs_previous"] = _vs_previous(history, status)
    atomic_write_json(out / f"status_{horizon}d.json", status, indent=1)
    last = history[-1] if history else {}
    if (last.get("as_of"), (last.get("manifest") or {}).get("labels_sha256"), last.get("experiment_id")) != \
            (status["as_of"], status["manifest"]["labels_sha256"], status["experiment_id"]):
        append_jsonl(hist_path, [{k: status[k] for k in ("as_of", "horizon", "experiment_id", "verified_labels",
                                                          "manifest", "walk_forward", "baselines", "forward")}])
    report_card(out)
    return status


def recommend(day: date, *, horizon: int = HORIZON, ledger: Path = red.LEDGER, quotes: Path = red.QUOTES,
              out: Path = OUT, now: datetime | None = None) -> dict:
    """Prognosen für alle Ausdrucksformen der heutigen Kandidaten und je Kandidat die Auswahlentscheidung (TRADE mit
    genau einer Form oder ABSTAIN) append-only ins Journal; Empfehlung nur bei bestandener Freigabe (release_gate),
    sonst unvalidierte Research-Watchlist."""
    now = now or datetime.now(timezone.utc)
    _rows, cands, data = _load(day, horizon, ledger, quotes)
    journal = read_jsonl(out / JOURNAL) if (out / JOURNAL).exists() else []
    instances = read_jsonl(out / INSTANCES) if (out / INSTANCES).exists() else []
    wf, fw = walk_forward(cands, day, horizon), forward(journal, cands, horizon, instances, day)
    train = [d for d in data if d["label_day"] < day]
    model, exp = fit(train), experiment_id()
    inst = instance(model, train, day) if model is not None else None
    if inst is not None:
        _register(out, inst, horizon, now)
    ref, ref_meta = reference(out, horizon, model, inst, now)
    stamp = {"horizon": horizon, "experiment_id": exp, "created_at": now.isoformat(timespec="seconds"),
             "walk_forward_verdict": (wf or {}).get("verdict"), "forward_verdict": (fw or {}).get("verdict")}
    new, decisions = [], []
    for c in (c for c in cands if c["date"] == day.isoformat() and model is not None):
        preds = model.predict([e["x"] for e in c["exprs"]])
        base = {"date": c["date"], "signal_id": c["signal_id"], "ticker": c["ticker"], "option_sample": c["sample"],
                **stamp, "model_id": inst["model_id"], "train_cutoff": inst["train_cutoff"]}
        new += [{**base, "kind": "prediction", "expression": e["expression"], "contract": e["contract"],
                 "strategy": e["strategy"], "entry_ask_net": e["ask"], "entry_mid_net": e["mid"],
                 "pred_net": round(float(p), 6)} for e, p in zip(c["exprs"], preds)]
        dec = decide(c, preds)
        chosen = realized(c, dec["contract"]) or {}
        decisions.append({**base, "kind": "decision", "role": "learning", "action": dec["action"],
                          "reason": dec["reason"], "expression": dec["expression"], "contract": dec["contract"],
                          "strategy": chosen.get("strategy"), "direction": _direction(chosen.get("strategy")),
                          "entry_ask_net": chosen.get("ask"), "entry_mid_net": chosen.get("mid"),
                          "pred_net": dec["pred"]})
        if ref is not None:                       # eingefrorene Referenz: gleiche Kandidaten, gleiche Regel
            rd = decide(c, ref.predict([e["x"] for e in c["exprs"]]))
            new.append({**base, "model_id": ref_meta["model_id"], "train_cutoff": ref_meta["train_cutoff"],
                        "kind": "decision", "role": "reference", "action": rd["action"], "reason": rd["reason"],
                        "expression": rd["expression"], "contract": rd["contract"], "pred_net": rd["pred"]})
    done = {_jkey(j) for j in journal}
    new = [j for j in new + decisions if _jkey(j) not in done]
    if new:
        append_jsonl(out / JOURNAL, new)
    gate = release_gate(wf, fw)                 # fail closed, auch wenn ein Bericht fehlt oder unvollständig ist
    best = {}
    for d in sorted((d for d in decisions if d["action"] == "TRADE"), key=lambda d: -d["pred_net"]):
        best.setdefault(d["ticker"], d)          # je Aktie höchstens eine Empfehlung
    recs = list(best.values()) if gate["allowed"] else []
    watch = sorted((d for d in decisions if d["contract"]), key=lambda d: -d["pred_net"])[:WATCHLIST_N]
    result = {"date": day.isoformat(), "horizon": horizon, "model_available": model is not None,
              "experiment_id": exp, "model_id": inst and inst["model_id"],
              "reference_model_id": ref_meta and ref_meta["model_id"], "verified_labels": len(data),
              "walk_forward": wf, "forward": fw, "release": gate,
              "mail_eligible": horizon == MAIL_HORIZON and bool(recs),
              "recommendations": recs, "watchlist": [] if recs else watch,
              "journaled": len(new), "decisions": len(decisions),
              "production_influence": "NONE", "orders_created": 0}
    target = out / "recommendations" / f"{day.isoformat()}_{horizon}d.json"
    atomic_write_json(target, result, indent=1)
    atomic_write_text(target.with_suffix(".md"), render(result))
    return result


def _fmt(v, pct=True):
    return "n/a" if v is None else (f"{v * 100:+.1f} %" if pct else f"{v:.2f}")


def render(res: dict) -> str:
    gate = res.get("release") or release_gate(res.get("walk_forward"), res.get("forward"))
    wf, fw = (defaultdict(lambda: None, res.get(k) or {"verdict": "FEHLT"}) for k in ("walk_forward", "forward"))
    early = res["horizon"] != MAIL_HORIZON
    lines = [f"# Options-Lernmodell – {res['date']} ({res['horizon']} T)", "",
             "_Research, keine Order, keine Anlageberatung._", ""]
    if early:
        lines += [(f"_Frühindikator ({res['horizon']} T): keine Handels-Mail; Hauptmaßstab ist das "
                   f"{MAIL_HORIZON}-Tage-Modell._"), ""]
    lines += [f"- Geprüfte Options-Labels: {res['verified_labels']}",
              (f"- Walk-Forward (Auswahlregel): **{wf['verdict']}** – Testtage {wf['dates']} "
               f"({wf['independent_periods']} unabhängige Perioden), gewählt n={wf['n_selected']}, "
               f"Ø netto {_fmt(wf['mean_selected'])} (alle Formen {_fmt(wf['mean_all'])}), "
               f"95-%-KI {wf['selected_ci95']}, Rang-IC {wf['ic_mean']} (t {wf['ic_t']}), "
               f"Abdeckung {wf['coverage']}"),
              (f"- Forward (vorab festgehaltene Entscheidungen): **{fw['verdict']}** – Tage {fw['dates']} "
               f"({fw['independent_periods']} unabhängige Perioden), gewählt n={fw['n_selected']}, "
               f"Ø netto {_fmt(fw['mean_selected'])}, Abdeckung {fw['coverage']}"),
              f"- Freigabe: **{'erteilt' if gate['allowed'] else 'gesperrt'}**",
              (f"- Experimentversion {res.get('experiment_id')}, Modellinstanz {res.get('model_id')}, "
               f"eingefrorene Referenz {res.get('reference_model_id')}"),
              (f"- Lernfortschritt (Forward, gepaart gegen die Referenz): "
               f"**{(fw['vs_reference'] or {}).get('verdict', 'NEED_MORE_DATA')}** – "
               f"{(fw['vs_reference'] or {}).get('days', 0)} Tage, "
               f"{(fw['vs_reference'] or {}).get('candidates', 0)} Kandidaten"), ""]
    warnings = list(dict.fromkeys(w for s in (wf, fw) for w in ((s["representativeness"] or {}).get("warnings") or [])))
    if warnings:
        lines += ["Evidenzbasis – Einschränkungen:", *[f"- {w}" for w in warnings], ""]
    if res["recommendations"]:
        lines += ["## Empfehlungen (Walk-Forward validiert, Forward positiv bestätigt)", "",
                  "| Ticker | Strategie | Form | Kontrakt | Ask / Mid (Entscheidung) | erwartet netto |",
                  "|---|---|---|---|---|---|"]
        lines += [f"| {i['ticker']} | {i['strategy']} | {i['expression']} | {' / '.join(i['contract'])} | "
                  f"{_fmt(i['entry_ask_net'], False)} / {_fmt(i['entry_mid_net'], False)} | {_fmt(i['pred_net'])} |"
                  for i in res["recommendations"]]
        lines += ["", ("Erwartung gilt für Kauf zum Ask und Verkauf zum Bid; ein Limit zum Mid ist günstiger, "
                       "aber nicht garantiert ausgeführt.")]
    else:
        lines += ["## Keine Empfehlung", ""]
        lines += ([f"- Gesperrt: {r}" for r in gate["reasons"]] if gate["reasons"] else
                  ["- Freigabe erteilt, aber heute kein positiver Netto-Erwartungswert."]) + [""]
        if res["watchlist"]:
            lines += ["Research-Watchlist – UNVALIDIERT, keine Empfehlung, nicht handeln:", "",
                      "| Ticker | Strategie | Form | Kontrakt | Prognose netto |", "|---|---|---|---|---|"]
            lines += [f"| {i['ticker']} | {i['strategy']} | {i['expression']} | {' / '.join(i['contract'])} | "
                      f"{_fmt(i['pred_net'])} |"
                      for i in res["watchlist"]]
    return "\n".join(lines) + "\n"


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, ValueError):
        return {}


def _answers(st: dict) -> dict:
    """Antworten auf die Audit-Fragen aus einem Status (Ja | Teilweise | Nein | offen + Beleg)."""
    if not st:
        return {}
    wf, fw, base, man = st["walk_forward"], st["forward"], st["baselines"], st["manifest"]
    counts = st.get("provenance_counts") or {}
    due = sum(counts.values())
    verified = counts.get("VERIFIED", 0)
    fr, wr = fw.get("vs_reference") or {}, wf.get("vs_reference") or {}
    trend = " → ".join(f"{h['as_of'][5:]} {h['verdict'] or '–'} {_fmt(h['mean_diff'])}"
                       for h in st.get("learning_trend") or [])
    expr = "; ".join(f"{k} {_fmt(v['mean_all'])} (n={v['n']})" for k, v in (wf.get("by_expression") or {}).items())
    vd = wf.get("vs_default") or {}
    choice = (f"gewählt {_fmt(wf.get('mean_selected'))} (n={wf.get('n_selected', 0)}, {wf.get('chosen_forms')}, "
              f"nicht bewertbar {wf.get('unevaluable')}); ggü. Standardform {_fmt(vd.get('mean_diff'))} "
              f"(n={vd.get('n', 0)}, KI {vd.get('ci95')})")
    rep = wf.get("representativeness") or {}
    rep_text = ("; ".join(f"{k} {_fmt(v['mean'])} (n={v['n']})" for k, v in (rep.get("by_sample") or {}).items())
                + f"; {rep.get('span_days', 0)} Tage" + "".join(f"; {w}" for w in rep.get("warnings") or []))
    gates = {k: v for k, v in (st.get("by_gate") or {}).items() if v.get("sufficient")}
    gate_text = "; ".join(f"{k} {_fmt(v['mean'])} (n={v['n']})"
                          for k, v in sorted(gates.items(), key=lambda kv: -kv[1]["mean"]))
    gate = release_gate(wf, fw)
    latest = (st.get("model_instances") or {}).get("latest") or {}
    inst_count = (st.get("model_instances") or {}).get("count", 0)
    return {
        "data": ("Ja" if st.get("option_snapshots_7d") else "Nein",
                 f"{st.get('ledger_rows_7d', 0)} Kandidaten, {st.get('option_snapshots_7d', 0)} mit Optionskontrakt (7 T)"),
        "provenance": ("offen" if not due else "Ja" if verified / due >= 0.9 else "Teilweise" if verified else "Nein",
                       f"{verified}/{due} fällige Labels geprüft"),
        "learns": ("Ja" if man["trains_model"] else "offen",
                   f"{man['n_labels']} geprüfte Labels, Manifest {man['labels_sha256']}"),
        "validated": ({"VALIDATED": "Ja", "NOT_VALIDATED": "Nein"}.get(wf["verdict"], "offen"),
                      (f"{wf['verdict']}, {wf['dates']} Testtage, {wf.get('independent_periods', 0)} unabhängige "
                       f"Perioden, Abdeckung {wf.get('coverage')}")),
        "baselines": ("Ja" if base["beats_no_trade"] and base["beats_trade_all"] else
                      "offen" if not base.get("days") else "Nein",
                      (f"je Kandidat und Tag (gleiches Budget, {base.get('days', 0)} Tage, retrospektiv): Modell "
                       f"{_fmt(base.get('model'))} vs. alles {_fmt(base['trade_all'])} vs. nie 0 %")),
        "forward": ({"FORWARD_POSITIVE": "Ja", "FORWARD_NEGATIVE": "Nein"}.get(fw["verdict"], "offen"),
                    (f"{fw['verdict']}, {fw['n']} fällige vorab festgehaltene Entscheidungen, "
                     f"{fw.get('pending_decisions', 0)} noch nicht fällig")),
        "experiment": ("Ja" if fw.get("valid_decisions") else "offen",
                       f"Version {st.get('experiment_id')}; {inst_count} Modellinstanzen"
                       + (f" (zuletzt {latest.get('model_id')}, Stichtag {latest.get('train_cutoff')}, "
                          f"Labels {latest.get('labels_sha256')})" if latest else "")
                       + f"; gültige Forward-Entscheidungen {fw.get('valid_decisions', 0)} (Prognosen "
                       f"{fw.get('valid_predictions', 0)}); ausgeschlossen: "
                       + (", ".join(f"{k} {v}" for k, v in (fw.get("excluded") or {}).items()) or "keine")),
        "improves": ({"LEARNING_BETTER": "Ja", "LEARNING_WORSE": "Nein",
                      "NO_SIGNIFICANT_DIFFERENCE": "Unentschieden"}.get(fr.get("verdict"), "offen"),
                     (f"Forward gegen eingefrorene Referenz {(st.get('reference') or {}).get('model_id')}: "
                      f"{fr.get('verdict', 'NEED_MORE_DATA')} ({fr.get('days', 0)} Tage, {fr.get('candidates', 0)} "
                      f"Kandidaten, Δ {_fmt((fr.get('learning_minus_reference') or {}).get('mean'))}/Tag, KI "
                      f"{(fr.get('learning_minus_reference') or {}).get('ci95')}); retrospektiv "
                      f"{wr.get('verdict')} (Δ {_fmt((wr.get('learning_minus_reference') or {}).get('mean'))}); "
                      f"Verlauf: {trend or '–'}")),
        "expressions": ("offen" if not expr else "Ja",
                        f"je Form: {expr}; {choice}" if expr else "noch keine Testtage"),
        "representative": ("offen" if not rep.get("by_sample") else "Teilweise" if rep.get("warnings") else "Ja",
                           rep_text if rep.get("by_sample") else "noch keine bewerteten Entscheidungen"),
        "gates": ("offen" if not gates else "Ja",
                  gate_text or f"noch keine Gruppe mit >= {MIN_GATE_N} Labels an >= {MIN_GATE_DATES} Tagen"),
        "proven": ("Ja" if gate["allowed"] else "Nein", "nur bei VALIDATED und FORWARD_POSITIVE samt Mindestzahlen"),
        "self_limit": ("Ja", "Empfehlungen freigegeben" if gate["allowed"] else
                       f"Empfehlungen gesperrt ({len(gate['reasons'])} offene Bedingungen, z. B. "
                       f"{gate['reasons'][0]})"),
    }


CARD_QUESTIONS = (
    ("data", "Werden Daten automatisch gesammelt?"),
    ("provenance", "Sind die Options-Ergebnisse gegen das Quote-Archiv geprüft?"),
    ("learns", "Lernt das System aus geprüften Options-Ergebnissen?"),
    ("validated", "Ist das Modell auf ungesehenen Tagen validiert?"),
    ("baselines", "Schlägt es 'nie handeln' und 'alles handeln'?"),
    ("forward", "Bestätigt es sich auf vorab festgehaltenen Prognosen?"),
    ("experiment", "Welche Experimentversion und Modellinstanzen stehen dahinter (ausgeschlossene Prognosen)?"),
    ("improves", "Entscheidet die lernende Version besser als die eingefrorene Referenz (gleiche Signale)?"),
    ("expressions", "Welche Optionsform schneidet ab, was bringt die Wahl des Modells (Ø netto, ungesehen)?"),
    ("representative", "Ist die Evidenz repräsentativ (Forschung vs. Richtungssignale, Zeitraum)?"),
    ("gates", "Was bringen die verworfenen Kandidaten je Filter (Ø netto, beschreibend)?"),
    ("proven", "Ist eine profitable Optionsstrategie nachgewiesen?"),
    ("self_limit", "Begrenzt es sich selbst?"),
)


def report_card(out: Path = OUT, horizons: tuple = (HORIZON, 5)) -> dict:
    """Leistungszeugnis aus den Status-Dateien: beantwortet die Audit-Fragen automatisch je Horizont."""
    answers = {h: _answers(_read_json(out / f"status_{h}d.json")) for h in horizons}
    as_of = max((_read_json(out / f"status_{h}d.json").get("as_of") or "" for h in horizons), default="")
    rows = [{"key": k, "question": q, **{f"{h}d": answers[h].get(k) for h in horizons}} for k, q in CARD_QUESTIONS]
    card = {"as_of": as_of, "horizons": list(horizons), "rows": rows, "production_influence": "NONE"}
    atomic_write_json(out / f"{REPORT_CARD}.json", card, indent=1)
    head = "| Frage | " + " | ".join(f"{h} Tage" for h in horizons) + " |"
    lines = [f"# Leistungszeugnis Options-Lernmodell – {as_of}", "", head,
             "|---|" + "---|" * len(horizons)]
    for r in rows:
        cells = [("–" if r[f"{h}d"] is None else f"**{r[f'{h}d'][0]}** – {r[f'{h}d'][1]}") for h in horizons]
        lines.append(f"| {r['question']} | " + " | ".join(cells) + " |")
    atomic_write_text(out / f"{REPORT_CARD}.md", "\n".join(lines) + "\n")
    return card


def mail(res: dict) -> dict | None:
    """Mail nur bei echten Empfehlungen des Hauptmodells; die Freigabe wird hier erneut geprüft (fail closed) –
    keine tägliche Nullmeldung, keine Mail für den 5-Tage-Frühindikator."""
    if (res.get("horizon") != MAIL_HORIZON or not res.get("recommendations")
            or not release_gate(res.get("walk_forward"), res.get("forward"))["allowed"]):
        return None
    from modules import mailer
    text = render(res)
    html = "<pre style='font-family:monospace'>" + text.replace("&", "&amp;").replace("<", "&lt;") + "</pre>"
    return mailer.send_mail(f"Options-Empfehlungen {res['date']} (Research, validiertes Modell)", html, text)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--as-of", required=True)
    e.add_argument("--horizon", type=int, default=HORIZON)
    r = sub.add_parser("recommend")
    r.add_argument("--date", required=True)
    r.add_argument("--horizon", type=int, default=HORIZON)
    r.add_argument("--mail", action="store_true")
    a = p.parse_args(argv)
    if a.cmd == "evaluate":
        s = evaluate(date.fromisoformat(a.as_of), horizon=a.horizon)
        print(json.dumps({k: s[k] for k in ("verified_labels", "walk_forward", "forward")}, default=str))
    else:
        res = recommend(date.fromisoformat(a.date), horizon=a.horizon)
        print(json.dumps({"recommendations": len(res["recommendations"]), "watchlist": len(res["watchlist"]),
                          "walk_forward": res["walk_forward"]["verdict"], "forward": res["forward"]["verdict"],
                          "journaled": res["journaled"]}))
        if a.mail:
            log.info("option_learner mail: %s", mail(res))


if __name__ == "__main__":
    main()

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
     VALIDATED UND Forward FORWARD_POSITIVE UND sequenzielle Bestätigung an einem vorab festgelegten Prüfzeitpunkt
     (forward_sequential: Prüfplan je Experimentversion, O'Brien-Fleming, eingefrorene Prüfberichte), jeweils mit
     Mindest-Testtagen, unabhängigen Perioden, Fallzahl, Abdeckung und Konfidenzintervall über 0 – fail closed
     bei fehlenden oder widersprüchlichen Angaben; je Kandidat die
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
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import NormalDist

import numpy as np
import yaml

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
VARIANTS_CONFIG = Path("config/option_learner_variants.yaml")
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


def experiment(params: dict | None = None) -> dict:
    """Experiment-/Strategieversion: alles, was eine Prognose oder Entscheidung inhaltlich bestimmt. Jede Änderung
    an Merkmalen, Labeldefinition, Kostenmodell, Trainingsverfahren oder Auswahlregel ergibt eine neue Version;
    deren Forward-Evidenz wird nie mit der einer anderen Version vermischt. `params` (Variante, siehe
    variant_params) ändert genau einen freigegebenen Parameter; ohne params die Basisversion."""
    p = params or {}
    drop = set(p.get("drop") or ())
    spec = {"name": "option_learner_ridge",
            "features": {"ledger": [f for f in LEDGER_FEATURES if f not in drop],
                         "contract": [f for f in CONTRACT_FEATURES if f not in drop],
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
    if p.get("alpha") is not None:
        spec["training"]["alpha"] = float(p["alpha"])
    if p.get("min_pred"):
        spec["selection"]["min_pred"] = float(p["min_pred"])
    return spec


def variant_config(path: Path | None = None) -> dict:
    """Freigegebene Methoden für Varianten-Experimente (Allowlist, nur per menschlichem PR änderbar)."""
    try:
        cfg = yaml.safe_load((path or VARIANTS_CONFIG).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        log.warning("option_learner: Varianten-Konfiguration unlesbar (%s) – keine Varianten", exc)
        return {}
    return cfg if cfg.get("version") == "olv-1" else {}


def validate_overrides(overrides: dict, cfg: dict) -> dict:
    """Genau eine Änderung mit einem freigegebenen Wert aus cfg['approved_methods'] – sonst ValueError. Damit
    laufen automatisch nur vorab freigegebene Methoden, nie beliebige Parameter oder Code."""
    approved = (cfg or {}).get("approved_methods") or {}
    if not isinstance(overrides, dict) or len(overrides) != 1:
        raise ValueError(f"genau eine Änderung je Variante erwartet: {overrides!r}")
    ((method, value),) = overrides.items()
    allowed = approved.get(method)
    if allowed is None:
        raise ValueError(f"Methode {method!r} nicht freigegeben")
    if method == "drop_feature_group":
        known = set(LEDGER_FEATURES) | set(CONTRACT_FEATURES)
        if value not in allowed or not set(allowed[value]) <= known:
            raise ValueError(f"Merkmalsgruppe {value!r} nicht freigegeben oder unbekannt")
    elif float(value) not in [float(v) for v in allowed]:
        raise ValueError(f"Wert {value!r} für {method} nicht freigegeben")
    return {method: value}


def variant_params(overrides: dict, cfg: dict) -> dict:
    """Freigegebene Änderung -> eingefrorene Parameter (Merkmalsgruppe aufgelöst), wie sie registriert werden."""
    ((method, value),) = validate_overrides(overrides, cfg).items()
    if method == "drop_feature_group":
        return {"drop": sorted(cfg["approved_methods"][method][value])}
    return {"alpha": float(value)} if method == "ridge_alpha" else {"min_pred": float(value)}


def experiment_id(params: dict | None = None) -> str:
    return hashlib.sha256(json.dumps(experiment(params), sort_keys=True).encode()).hexdigest()[:12]


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


def fit(train: list[dict], params: dict | None = None) -> Model | None:
    """Ridge auf den geprüften Labels; `params` einer Variante: Merkmale weglassen (drop) bzw. alpha."""
    if len(train) < MIN_TRAIN_ROWS:
        return None
    p = params or {}
    drop, alpha = set(p.get("drop") or ()), float(p.get("alpha") or RIDGE_ALPHA)
    names = [n for n in (*LEDGER_FEATURES, *CONTRACT_FEATURES)
             if n not in drop and sum(n in d["x"] for d in train) >= MIN_FEATURE_COVERAGE * len(train)]
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
    coef = np.linalg.solve(x.T @ x + alpha * np.eye(len(names)), x.T @ (y - intercept))
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


def _block_bootstrap_ci(by_date: dict[str, list[float]], horizon: int, seed: int = 7, alpha: float = 0.05,
                        method: str = "percentile") -> tuple[float, float] | None:
    """(1 − alpha)-KI des gepoolten Mittels (Standard 95 %); gezogen werden ganze Haltefenster (Block-Bootstrap),
    nicht einzelne Tage. method="normal": Mittel ± z·Bootstrap-Standardfehler – für sehr kleine, korrigierte alpha
    (sequenziell, Bonferroni), deren Perzentile 2000 Ziehungen nicht verlässlich auflösen. method="t": wie
    "normal", aber mit dem t-Quantil für B − 1 Freiheitsgrade und dem um √(B/(B−1)) korrigierten Standardfehler
    (B = Zahl der Haltefenster) – bei wenigen unabhängigen Fenstern ist die Normal-Approximation zu optimistisch."""
    blocks = [b for b in _periods(by_date, horizon).values() if b]
    if len(blocks) < 2:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(BOOTSTRAP):
        vals = [v for b in (rng.choice(blocks) for _ in blocks) for v in b]
        means.append(sum(vals) / len(vals))
    if method in ("normal", "t"):
        pooled, se = [v for b in blocks for v in b], float(np.std(means, ddof=1))
        if method == "t":
            from scipy.stats import t as student_t
            q = float(student_t.ppf(1 - alpha / 2, len(blocks) - 1))
            se *= math.sqrt(len(blocks) / (len(blocks) - 1))
        else:
            q = NormalDist().inv_cdf(1 - alpha / 2)
        m = sum(pooled) / len(pooled)
        return m - q * se, m + q * se
    means.sort()
    return means[int(alpha / 2 * BOOTSTRAP)], means[int((1 - alpha / 2) * BOOTSTRAP) - 1]


def score(preds: list[dict], horizon: int = HORIZON, alpha: float = 0.05, method: str = "percentile") -> dict:
    """Kennzahlen für Prognosen [{date, pred, y}]: Rang-IC je Tag, Mittel der Auswahl (pred > 0) mit Block-Bootstrap
    über nicht überlappende Haltefenster (`independent_periods`); `selected_ci95` ist das (1 − alpha)-KI
    (Standard 95 %; die sequenzielle Forward-Prüfung übergibt ihr Teil-Alpha)."""
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
    ci = _block_bootstrap_ci(sel, horizon, alpha=alpha, method=method) if n_sel else None
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


def decide(cand: dict, preds, min_pred: float = 0.0) -> dict:
    """Auswahlregel – nur aus Prognosen, nie aus Ergebnissen: je Kandidat genau eine zulässige Ausdrucksform
    (höchste Prognose unter ALLOWED_STRATEGIES, bei Gleichstand die erste); TRADE wenn > min_pred (Basis 0),
    sonst ABSTAIN."""
    elig = [(float(p), e) for p, e in zip(preds, cand["exprs"]) if e["strategy"] in ALLOWED_STRATEGIES]
    if not elig:
        return {"action": "ABSTAIN", "reason": "NO_ALLOWED_EXPRESSION", "expression": None, "contract": None,
                "pred": None}
    p, e = max(elig, key=lambda t: t[0])
    reason = None if p > min_pred else "PRED_NOT_POSITIVE" if p <= 0 else "PRED_BELOW_MIN"
    return {"action": "ABSTAIN" if reason else "TRADE", "reason": reason,
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


def _paired(by_date: dict[str, list[float]], horizon: int, alpha: float = 0.05, method: str = "percentile") -> dict:
    vals = [v for vs in by_date.values() for v in vs]
    ci = _block_bootstrap_ci(by_date, horizon, alpha=alpha, method=method) if vals else None
    return {"n": len(vals), "dates": len(by_date), "independent_periods": len(_periods(by_date, horizon)),
            "mean_diff": round(float(np.mean(vals)), 6) if vals else None,
            "ci95": [round(c, 6) for c in ci] if ci else None}


def assess(records: list[dict], horizon: int, alpha: float = 0.05, method: str = "percentile") -> dict:
    """Bewertet vorab getroffene Auswahlentscheidungen [{cand, dec, preds}] auf fälligen Kandidaten. Gezählt wird nur
    das geprüfte Netto-Ergebnis der gewählten Form; fehlt es, ist der Fall nicht bewertbar (mit Grund) und nie
    positiv. Je Aktie und Tag zählt ein Kandidat (der erste nach Signal-ID), Varianten nie getrennt.
    alpha/method betreffen nur das KI der Auswahl (Standard: 95 %, Perzentile)."""
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
    sel, rank = score(picks, horizon, alpha, method), score(expr_preds, horizon)
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


def compare_policies(pairs: list[dict], horizon: int, alpha: float = 0.05, due: int | None = None,
                     method: str = "percentile") -> dict:
    """Gepaarter Vergleich [{cand, learning, reference}] auf denselben fälligen Kandidaten: je Kandidat ein gleiches
    Budget (eine Einheit Prämie; ABSTAIN = Einheit bleibt in Cash = 0), je Tag das Mittel über alle Kandidaten des
    Tages – für lernende Version, Referenz, "alles handeln" (Standardform, sofern zulässig) und "nie handeln" (0).
    Ein Kandidat zählt nur, wenn das Ergebnis jeder gehandelten Wahl geprüft vorliegt (sonst für alle
    ausgeschlossen); je Aktie und Tag einer. Unsicherheit der Tagesdifferenz per Block-Bootstrap.
    Gemeinsame Ergebnisabdeckung = verglichene / fällige Kandidaten (`due`, sonst die übergebenen Paare).
    Vorab festgelegt: NEED_MORE_DATA unter den Mindestzahlen; INSUFFICIENT_COVERAGE unter MIN_COVERAGE (fehlende
    Quotes könnten das Ergebnis verzerren – kein Urteil); LEARNING_BETTER / LEARNING_WORSE, wenn das (1 − alpha)-KI
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
        res = _paired({d: [day_mean[d][a] - (day_mean[d][b] if b else 0.0)] for d in days}, horizon, alpha, method)
        return {"mean": res["mean_diff"], "ci95": res["ci95"], "independent_periods": res["independent_periods"]}
    lr, n = diff("learning", "reference"), sum(len(daily[d]["learning"]) for d in days)
    due_n = max(due or 0, len(seen))
    coverage = round(n / due_n, 4) if due_n else None
    if n < MIN_SELECTED or len(days) < MIN_OOS_DATES or lr["independent_periods"] < MIN_INDEPENDENT_PERIODS:
        verdict = "NEED_MORE_DATA"
    elif coverage is None or coverage < MIN_COVERAGE:
        verdict = "INSUFFICIENT_COVERAGE"
    elif lr["ci95"] and lr["ci95"][0] > 0:
        verdict = "LEARNING_BETTER"
    elif lr["ci95"] and lr["ci95"][1] < 0:
        verdict = "LEARNING_WORSE"
    else:
        verdict = "NO_SIGNIFICANT_DIFFERENCE"
    return {"candidates": n, "due_candidates": due_n, "coverage": coverage, "days": len(days),
            "independent_periods": lr["independent_periods"], "ci_level": round(1 - alpha, 6),
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


def _evidence_checks(s: dict, ci_name: str = "95-%-KI") -> tuple:
    """Mindestzahlen und Evidenz aus den Kennzahlen eines Berichts erneut prüfen (KeyError/TypeError: Bericht
    unvollständig)."""
    ci = s.get("selected_ci95")
    return ((s["dates"] >= MIN_OOS_DATES, f"Testtage {s['dates']} < {MIN_OOS_DATES}"),
            (s["n_selected"] >= MIN_SELECTED, f"Auswahl n={s['n_selected']} < {MIN_SELECTED}"),
            (s["independent_periods"] >= MIN_INDEPENDENT_PERIODS,
             f"unabhängige Perioden {s['independent_periods']} < {MIN_INDEPENDENT_PERIODS}"),
            (bool(ci) and ci[0] > 0, f"{ci_name} {ci} nicht über 0"),
            (s.get("mean_selected") is not None and s["mean_selected"] > 0, "Ø netto der Auswahl nicht > 0"),
            (s["coverage"] is not None and s["coverage"] >= MIN_COVERAGE,
             f"Abdeckung geprüfter Ergebnisse {s['coverage']} < {MIN_COVERAGE}"))


def _sequential_reasons(seq: dict | None) -> list[str]:
    """Sperrgründe der sequenziellen Forward-Prüfung (forward_sequential) – fail closed: nur CONFIRMED an einem
    vorab festgelegten Prüfzeitpunkt mit eingefrorenem Bericht, aktueller Experimentversion, dem Teil-Alpha
    dieser Prüfung als KI-Niveau und allen Mindestzahlen samt Entscheidungsabdeckung gibt frei."""
    name = "forward_sequential"
    if not isinstance(seq, dict) or not seq.get("status"):
        return [f"{name}: vorab festgelegte Forward-Prüfung fehlt"]
    if seq["status"] != "CONFIRMED":
        return [f"{name}: {seq['status']} (verlangt CONFIRMED an einem vorab festgelegten Prüfzeitpunkt)"]
    try:
        b, alpha = seq["binding"], float(seq["alpha"])
        k, spend = int(b["look"]), alpha_spending(alpha)
        checks = ((seq["experiment_id"] == experiment_id(), f"Experimentversion {seq['experiment_id']} veraltet"),
                  (0 < alpha <= FORWARD_ALPHA, f"Gesamt-Alpha {alpha} nicht in (0, {FORWARD_ALPHA}]"),
                  (1 <= k <= len(spend) and abs(float(b["ci_level"]) - (1 - spend[k - 1])) < 1e-6,
                   f"KI-Niveau {b.get('ci_level')} entspricht nicht dem Teil-Alpha von Prüfung {k}"),
                  (b["ci_method"] == FORWARD_CI_METHOD, f"KI-Methode {b['ci_method']} statt {FORWARD_CI_METHOD}"),
                  (b["verdict"] == "FORWARD_POSITIVE", f"Prüfung {k}: {b['verdict']}"),
                  *_evidence_checks(b, f"KI ({b.get('ci_level')})"),
                  (b["decision_coverage"] is not None and b["decision_coverage"] >= MIN_COVERAGE,
                   f"Entscheidungsabdeckung {b['decision_coverage']} < {MIN_COVERAGE}"),
                  (bool(b.get("report_id")), f"Prüfbericht zu Prüfung {k} nicht eingefroren"))
    except (KeyError, TypeError, IndexError, ValueError):
        return [f"{name}: Kennzahlen unvollständig"]
    return [f"{name}: {text}" for ok, text in checks if not ok]


def release_gate(wf: dict | None, fw: dict | None, seq: dict | None = None) -> dict:
    """Freigabe von Empfehlungen – fail closed. Erlaubt nur bei Walk-Forward VALIDATED UND Forward FORWARD_POSITIVE
    UND sequenzieller Bestätigung (forward_sequential: CONFIRMED an einem vorab festgelegten Prüfzeitpunkt).
    Die Mindestzahlen und Evidenzanforderungen werden aus den Kennzahlen erneut geprüft, damit ein fehlender,
    unvollständiger oder widersprüchlicher Bericht (z. B. Verdikt ohne passende Zahlen) nie freigibt.

    Warum die dritte Bedingung: Der tägliche Forward-Bericht prüft dieselben, wachsenden Daten jeden Tag mit einem
    nominalen 95-%-KI; wer so lange täglich prüft, bis das KI einmal über 0 liegt, findet auch ohne echten
    Vorteil irgendwann einen "Erfolg". Er bleibt als laufende Kontrolle (schließt die Freigabe wieder, sobald er
    nicht mehr positiv ist), kontrolliert aber die Fehlerrate nicht. Das tut die sequenzielle Prüfung: höchstens
    drei vorab festgelegte Prüfzeitpunkte je Experimentversion mit O'Brien-Fleming-Teil-Alphas, die sich zu
    FORWARD_ALPHA summieren. Da alle Bedingungen zugleich gelten müssen, ist die Wahrscheinlichkeit einer
    Freigabe ohne echten Vorteil höchstens die der sequenziellen Prüfung."""
    reasons = []
    for name, s, want in (("walk_forward", wf, "VALIDATED"), ("forward", fw, "FORWARD_POSITIVE")):
        if not isinstance(s, dict) or not s.get("verdict"):
            reasons.append(f"{name}: Bericht fehlt")
            continue
        if s["verdict"] != want:
            reasons.append(f"{name}: {s['verdict']} (verlangt {want})")
        try:
            checks = _evidence_checks(s)
            if name == "walk_forward":
                checks += (((s.get("ic_t") or 0) > IC_T_MIN, f"Rang-IC t {s.get('ic_t')} <= {IC_T_MIN}"),)
            else:
                checks += ((s["experiment_id"] == experiment_id(), f"Experimentversion {s['experiment_id']} veraltet"),)
        except (KeyError, TypeError, IndexError):
            reasons.append(f"{name}: Kennzahlen unvollständig")
            continue
        reasons += [f"{name}: {text}" for ok, text in checks if not ok]
    reasons += _sequential_reasons(seq)
    return {"allowed": not reasons, "reasons": reasons}


# ── Sequenzielle Forward-Prüfung der Basis (Freigabe) ─────────────────────────────────────────────────────────────
FORWARD_PLAN_DIR = "forward_plans"
FORWARD_LOOK_DIR = "forward_looks"
FORWARD_ALPHA = 0.05                    # zweiseitig, über ALLE Prüfzeitpunkte eines Prüfplans (O'Brien-Fleming)
FORWARD_MAX_DAYS = {5: 120, 20: 300}    # Kalendertage ab Start; Prüfungen bei 1/3, 2/3 und 1 davon
FORWARD_CI_METHOD = "t"                 # t-Quantil (Fenster − 1 Freiheitsgrade): wenige unabhängige Haltefenster
SEQ_FINAL = ("CONFIRMED", "REJECTED_NEGATIVE", "NOT_CONFIRMED")


def _plan_core(plan: dict) -> dict:
    return {k: v for k, v in plan.items() if k != "plan_sha256"}


def forward_plan(out: Path, horizon: int, exp: str | None = None, now: datetime | None = None,
                 register: bool = False) -> dict | None:
    """Vorab festgelegter Prüfplan des Forward-Tests je Experimentversion und Horizont (create-only, nie geändert).

    Start = Tag der Registrierung: es zählen nur ab dann am Signaltag festgehaltene Entscheidungen, deren Ergebnis
    bei der Registrierung also noch unbekannt war. Prüfzeitpunkte bei 1/3, 2/3 und 1 von max_days, Gesamt-Alpha
    FORWARD_ALPHA mit O'Brien-Fleming-Aufteilung. register=True legt den Plan an, falls er fehlt (nur der tägliche
    Scanner – genau ein Schreiber); sonst wird nur gelesen. Ein veränderter oder beschädigter Plan (Hash, Version,
    Horizont) gilt als ungültig und sperrt die Freigabe; er wird nie still ersetzt."""
    exp = exp or experiment_id()
    path = out / FORWARD_PLAN_DIR / f"{horizon}d_{exp}.json"
    if path.exists():
        plan = _read_json(path)
        ok = (plan.get("plan_sha256") == _sha(_plan_core(plan)) and plan.get("experiment_id") == exp
              and plan.get("horizon") == horizon)
        return plan if ok else {"plan_id": plan.get("plan_id") or path.stem, "invalid": True}
    if not register:
        return None
    now = now or datetime.now(timezone.utc)
    start, max_days = now.date(), FORWARD_MAX_DAYS.get(horizon, 15 * horizon)
    days = look_days(max_days)
    plan = {"plan_id": f"FWD-{horizon}d-{exp}", "experiment_id": exp, "horizon": horizon,
            "registered_at": now.isoformat(timespec="seconds"), "start_date": start.isoformat(),
            "max_days": max_days, "look_days": days,
            "look_cutoffs": [(start + timedelta(days=d)).isoformat() for d in days],
            "alpha": FORWARD_ALPHA, "spending": "O'Brien-Fleming (Lan-DeMets)",
            "look_alpha": alpha_spending(FORWARD_ALPHA), "ci_method": FORWARD_CI_METHOD,
            "evidence": EVIDENCE, "min_decision_coverage": MIN_COVERAGE,
            "rule": ("Prüfung k erst, wenn alle Kandidaten bis Stichtag k fällig sind; nur Kandidaten im Zeitraum "
                     "[Start, Stichtag k]; CONFIRMED, sobald das KI zum Teil-Alpha von k ganz über 0 liegt "
                     "(Mindestzahlen und Abdeckung erfüllt); REJECTED_NEGATIVE, wenn ganz unter 0; nach der "
                     "letzten Prüfung ohne Bestätigung NOT_CONFIRMED (Freigabe für diese Version dauerhaft "
                     "gesperrt).")}
    plan["plan_sha256"] = _sha(plan)
    if path.exists():                                   # nie überschreiben
        raise FileExistsError(path)
    atomic_write_json(path, plan, indent=1)
    return plan


def _forward_look(out: Path, plan: dict, k: int, alpha_k: float, cands: list[dict], learn: dict, invalid_ck: dict,
                  horizon: int, as_of: date, now: datetime, freeze: bool) -> dict:
    """Prüfung k des Prüfplans auf ihrer festen Population: ALLE Ledger-Kandidaten im Zeitraum [Start, Stichtag k],
    je Aktie und Tag einer – Kandidaten ohne gültige, am Signaltag festgehaltene Entscheidung zählen als Ausfall
    (Entscheidungsabdeckung). Bewertet werden nur diese Entscheidungen (nie neu berechnet) mit dem KI zum
    Teil-Alpha alpha_k. freeze=True friert den Bericht ein; sonst wird nur ein passender Bericht gesucht."""
    cutoff = plan["look_cutoffs"][k - 1]
    pop, seen = [], set()
    for c in sorted(cands, key=lambda c: (c["date"], c["ticker"], str(c["signal_id"]))):
        if plan["start_date"] <= c["date"] <= cutoff and (c["date"], c["ticker"]) not in seen:
            seen.add((c["date"], c["ticker"]))
            pop.append(c)
    records, why, models = [], defaultdict(int), set()
    for c in pop:
        j = learn.get((c["date"], c["signal_id"]))
        if j is None:
            why[invalid_ck.get((c["date"], c["signal_id"]), "NO_DECISION")] += 1
            continue
        models.add(j.get("model_id"))
        records.append({"cand": c, "dec": {"action": j.get("action"), "contract": j.get("contract"),
                                           "pred": j.get("pred_net")}, "preds": {}})
    s = assess(records, horizon, alpha=alpha_k, method=plan["ci_method"])
    dcov = round(len(records) / len(pop), 4) if pop else None
    verdict = _verdict(s, "forward")
    if verdict != "NEED_MORE_DATA" and (dcov is None or dcov < MIN_COVERAGE):
        verdict = "INSUFFICIENT_COVERAGE"
    stats = {k2: s[k2] for k2 in ("n", "dates", "independent_periods", "n_selected", "selected_dates",
                                  "mean_selected", "selected_ci95", "trades", "abstained", "unevaluable", "coverage")}
    lk = {"look": k, "data_cutoff": cutoff, "look_alpha": alpha_k, "ci_level": round(1 - alpha_k, 8),
          "ci_method": plan["ci_method"],
          "verdict": verdict, "n_population": len(pop), "decision_coverage": dcov,
          "excluded": dict(sorted(why.items())), **stats,
          "warnings": (s.get("representativeness") or {}).get("warnings") or []}
    labels, inputs = [], []
    for r in records:
        c, e = r["cand"], realized(r["cand"], r["dec"]["contract"])
        if r["dec"]["action"] == "TRADE" and e and e["status"] == "VERIFIED":
            labels.append(f"{c['date']}|{c['signal_id']}|{'/'.join(e['contract'])}|{e['label_day']}")
        inputs.append([c["date"], c["signal_id"], r["dec"]["action"], r["dec"]["contract"],
                       [[x["contract"], x["status"], x["y"]] for x in c["exprs"]]])
    report = {"plan_id": plan["plan_id"], "plan_sha256": plan["plan_sha256"], "look": k,
              "looks": len(plan["look_cutoffs"]), "look_day": plan["look_days"][k - 1], "data_cutoff": cutoff,
              "evaluated_as_of": as_of.isoformat(), "horizon": horizon, "experiment_id": plan["experiment_id"],
              "model_ids": sorted(m for m in models if m),
              "decision_ids": [f"{c['date']}|{c['signal_id']}" for c in pop],
              "label_ids": sorted(labels), "n_population": len(pop), "n_compared": len(records),
              "decision_coverage": dcov, "coverage": s["coverage"], "excluded": lk["excluded"],
              "alpha": {"total": plan["alpha"], "look_alpha": alpha_k, "ci_level": lk["ci_level"],
                        "ci_method": plan["ci_method"],
                        "spending": alpha_spending(plan["alpha"])},
              "result": {"verdict": verdict, **stats},
              "release_status": "CONFIRMED" if verdict == "FORWARD_POSITIVE" else "NO_RELEASE",
              "input_sha256": _sha({"pop": [f"{c['date']}|{c['signal_id']}" for c in pop], "inputs": inputs,
                                    "excluded": lk["excluded"], "look_alpha": alpha_k,
                                    "plan_sha256": plan["plan_sha256"]})}
    if freeze:
        rec = freeze_look_report(out, report, now, FORWARD_LOOK_DIR)
    else:                                               # nur lesen: gilt nur ein Bericht mit gleichen Eingangsdaten
        versions = _look_versions(out / FORWARD_LOOK_DIR, f"{plan['plan_id']}_look{k}")
        rec = _read_json(versions[-1]) if versions else None
        rec = rec if rec and rec.get("input_sha256") == report["input_sha256"] else None
    lk["report_id"], lk["report_version"] = (rec["report_id"], rec["version"]) if rec else (None, None)
    return lk


def forward_sequential(out: Path, journal: list[dict], cands: list[dict], horizon: int,
                       instances: list[dict] | None = None, as_of: date | None = None, now: datetime | None = None,
                       register: bool = False, freeze: bool = False) -> dict:
    """Sequenziell korrekte Forward-Prüfung der Basis für die Freigabe (Gruppen-sequenziell, vorab registriert).

    * Prüfplan je Experimentversion und Horizont (forward_plan, create-only): Start, Stichtage, Gesamt-Alpha und
      Teil-Alphas stehen fest, bevor ein Ergebnis bekannt ist.
    * Prüfung k erst, wenn alle Kandidaten bis zu ihrem Stichtag fällig sind (keine künftigen Daten); verpasste
      frühere Prüfungen werden der Reihe nach nachgeholt. Zwischen zwei Prüfungen ändert sich das Ergebnis nicht –
      tägliches Auswerten erzeugt also keine zusätzlichen Tests.
    * Gültig sind nur unveränderte, am Signaltag geschriebene Entscheidungen der aktuellen Experimentversion von
      vorher trainierten, registrierten Modellinstanzen (dieselbe Prüfung wie im Forward-Test).
    * Status: NO_PLAN / PLAN_INVALID / AWAITING_LOOK / RUNNING (Prüfung ohne Entscheidung, weiter) / CONFIRMED
      (KI ganz über 0) / REJECTED_NEGATIVE (KI ganz unter 0) / NOT_CONFIRMED (letzte Prüfung ohne Bestätigung).
      Fehlende oder unzureichende Evidenz (NEED_MORE_DATA, INSUFFICIENT_COVERAGE) bestätigt nie.
    * Jede Prüfung wird als Bericht eingefroren (freeze=True, nur im täglichen Scanner); korrigierte Daten ergeben
      eine versionierte Korrektur, maßgeblich ist die neueste Fassung."""
    exp = experiment_id()
    as_of = as_of or datetime.now(timezone.utc).date()
    now = now or datetime.now(timezone.utc)
    head = {"experiment_id": exp, "horizon": horizon, "evaluated_as_of": as_of.isoformat(), "looks_evaluated": [],
            "binding": None}
    plan = forward_plan(out, horizon, exp, now, register)
    if plan is None:
        return {**head, "status": "NO_PLAN"}
    if plan.get("invalid"):
        return {**head, "status": "PLAN_INVALID", "plan_id": plan.get("plan_id")}
    reg = {i.get("model_id"): i for i in instances or [] if isinstance(i, dict)}
    invalid_ck: dict = {}
    learn = _decisions(journal, exp, reg, horizon, invalid_ck)[0].get("learning", {})
    spend, cutoffs = alpha_spending(float(plan["alpha"])), plan["look_cutoffs"]
    ready = max((k for k, c in enumerate(cutoffs, 1) if _due(c, as_of, horizon)), default=0)
    looks, status = [], "AWAITING_LOOK"
    for k in range(1, ready + 1):
        lk = _forward_look(out, plan, k, spend[k - 1], cands, learn, invalid_ck, horizon, as_of, now, freeze)
        looks.append(lk)
        status = ({"FORWARD_POSITIVE": "CONFIRMED", "FORWARD_NEGATIVE": "REJECTED_NEGATIVE"}.get(lk["verdict"])
                  or ("NOT_CONFIRMED" if k == len(cutoffs) else "RUNNING"))
        if status in SEQ_FINAL:
            break
    nxt = None if status in SEQ_FINAL else next(((k, c) for k, c in enumerate(cutoffs, 1) if k > len(looks)), None)
    return {**head, "status": status, "plan_id": plan["plan_id"], "plan_sha256": plan["plan_sha256"],
            "start_date": plan["start_date"], "look_cutoffs": cutoffs, "looks": len(cutoffs),
            "alpha": plan["alpha"], "look_alpha": spend, "look": len(looks),
            "valid_decisions": sum(1 for d, _s in learn if d >= plan["start_date"]),
            "looks_evaluated": [{k2: lk[k2] for k2 in ("look", "data_cutoff", "verdict", "ci_level", "selected_ci95",
                                                       "n_selected", "decision_coverage", "coverage", "report_id",
                                                       "report_version")} for lk in looks],
            "binding": looks[-1] if looks else None,
            "next_look": {"look": nxt[0], "cutoff": nxt[1],
                          "ready_after": (date.fromisoformat(nxt[1]) + timedelta(days=horizon + DUE_GRACE_DAYS))
                          .isoformat()} if nxt else None}


def _load(as_of: date, horizon: int, ledger: Path, quotes: Path, counts: dict | None = None):
    """(Ledger-Zeilen, Kandidaten mit allen Formen und Label-Status, geprüfte Trainings-Labels)."""
    rows = red.load(ledger, as_of)
    labelled = [r for r in rows if isinstance(r, dict) and any(
        _num((r.get("outcomes") or {}).get(f"{prefix}_ret_{horizon}d")) is not None for _k, prefix, _n in SOURCES)]
    cands = candidates(rows, red.load_quotes(quotes, labelled, as_of), as_of, horizon, counts)
    return rows, cands, _verified(cands)


def _label_ids(data: list[dict]) -> list[str]:
    return sorted(f"{d['date']}|{d['signal_id']}|{'/'.join(d['contract'])}|{d['label_day']}" for d in data)


def instance(model: Model, train: list[dict], cutoff: date, exp: str | None = None) -> dict:
    """Trainierte Modellinstanz: Experimentversion, Trainingsstichtag, Manifest der verwendeten Label-IDs und
    Fingerprint der Gewichte. Die Label-IDs sind aus dem Ledger über (Signal, Kontrakt, Mark-Tag < Stichtag)
    reproduzierbar; gespeichert wird ihr Hash."""
    labels_sha = hashlib.sha256("\n".join(_label_ids(train)).encode()).hexdigest()[:16]
    weights = np.round(np.concatenate([model.coef, [model.intercept], model.mu, model.sd]), 10)
    weights_sha = hashlib.sha256(weights.tobytes() + "|".join(model.names).encode()).hexdigest()[:16]
    exp = exp or experiment_id()
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


VARIANTS = "variants.jsonl"
VARIANT_DIR = "variants"


def variant_registry(out: Path = OUT) -> dict[str, dict]:
    """Zustand je Variante aus dem append-only Register: Ereignis 'registered', später ggf. 'closed'."""
    path, state = out / VARIANTS, {}
    for e in read_jsonl(path) if path.exists() else []:
        if e.get("event") == "registered":
            state.setdefault(e["variant_id"], dict(e))
        elif e.get("event") == "closed" and e.get("variant_id") in state:
            state[e["variant_id"]].update({k: e.get(k) for k in ("status", "closed_at", "close_reason", "closed_look",
                                                                 "look_report")})
    return state


def _open_for_base(v: dict, horizon: int, base: str) -> bool:
    return v.get("horizon") == horizon and not v.get("closed_at") and v.get("base_experiment_id") == base


def active_variants(out: Path, horizon: int, day: date) -> list[dict]:
    """Offene Varianten der aktuellen Basisversion, deren Start (Folgetag der Registrierung) erreicht ist."""
    base = experiment_id()
    return [v for v in variant_registry(out).values()
            if _open_for_base(v, horizon, base) and v["start_date"] <= day.isoformat()]


def register_variants(out: Path, proposals: list[dict], now: datetime, cfg: dict | None = None) -> list[dict]:
    """Registriert automatisch testbare Vorschläge als isolierte Varianten (append-only, Start am Folgetag, also
    erst auf künftigen Signalen). Höchstens max_active_per_horizon offene Varianten je Horizont, je Methode eine;
    eine Änderung, die schon einmal registriert oder falsifiziert wurde, wird nie erneut getestet. Jede
    Registrierung verbraucht einen Slot des dauerhaften Fehlerbudgets (test_budget); ist es erschöpft, wird
    nichts mehr registriert."""
    cfg = cfg if cfg is not None else variant_config()
    reg, base, new = variant_registry(out), experiment_id(), []
    max_active = int(cfg.get("max_active_per_horizon") or 0)
    budget = test_budget(out, cfg)
    for p in proposals:
        if not p.get("auto_experiment"):
            continue
        if budget["used"] >= budget["max_tests"]:           # Fehlerbudget erschöpft: keine neuen Tests
            break
        try:
            change, params = validate_overrides(p["change"], cfg), variant_params(p["change"], cfg)
        except (ValueError, KeyError, TypeError):
            continue
        h, vexp = int(p["horizon"]), experiment_id(params)
        vid = "OLV-" + hashlib.sha256(f"{base}|{h}|{vexp}".encode()).hexdigest()[:10]
        if vid in reg or any(v["horizon"] == h and v["overrides"] == change and str(v.get("status", "")).startswith(
                "FALSIFIED") for v in reg.values()):
            continue
        open_ = [v for v in reg.values() if _open_for_base(v, h, base)]
        if len(open_) >= max_active or any(next(iter(v["overrides"])) == next(iter(change)) for v in open_):
            continue
        max_days = int((p.get("falsification") or {}).get("max_days") or 4 * h + 60)
        ev = {"event": "registered", "variant_id": vid, "horizon": h, "base_experiment_id": base,
              "experiment_id": vexp, "overrides": change, "params": params, "proposal_id": p.get("proposal_id"),
              "gap_ids": p.get("gap_ids"), "falsification": p.get("falsification"),
              "look_days": look_days(max_days),
              "test_budget": {"total_alpha": budget["total_alpha"], "max_tests": budget["max_tests"]},
              "alpha_slot": budget["used"] + 1, "slot_alpha": budget["slot_alpha"],
              "registered_at": now.isoformat(timespec="seconds"),
              "start_date": (now.date() + timedelta(days=1)).isoformat()}
        append_jsonl(out / VARIANTS, [ev])
        reg[vid] = ev
        new.append(ev)
        budget["used"] += 1
    return new


def _run_variant(out: Path, v: dict, cands: list[dict], train: list[dict], day: date, horizon: int,
                 now: datetime) -> int:
    """Eine Variante auf genau denselben Kandidaten und Labels wie die Basis; schreibt nur in ihr eigenes Journal
    variants/<variant_id>.jsonl und ihre Modellinstanz ins Register – nie in Basis-Journal, Freigabe oder Mail."""
    params = v["params"]
    if experiment_id(params) != v["experiment_id"]:
        raise ValueError("Spezifikation weicht von der registrierten Variante ab")
    model = fit(train, params)
    if model is None:
        return 0
    inst = instance(model, train, day, exp=v["experiment_id"])
    _register(out, inst, horizon, now)
    path = out / VARIANT_DIR / f"{v['variant_id']}.jsonl"
    done = {_jkey(j) for j in read_jsonl(path)} if path.exists() else set()
    new = []
    for c in (c for c in cands if c["date"] == day.isoformat()):
        dec = decide(c, model.predict([e["x"] for e in c["exprs"]]), float(params.get("min_pred") or 0.0))
        e = {"date": c["date"], "signal_id": c["signal_id"], "ticker": c["ticker"], "horizon": horizon,
             "kind": "decision", "role": "variant", "variant_id": v["variant_id"], "experiment_id": v["experiment_id"],
             "model_id": inst["model_id"], "train_cutoff": inst["train_cutoff"], "action": dec["action"],
             "reason": dec["reason"], "expression": dec["expression"], "contract": dec["contract"],
             "pred_net": dec["pred"], "created_at": now.isoformat(timespec="seconds")}
        if _jkey(e) not in done:
            new.append(e)
    if new:
        append_jsonl(path, new)
    return len(new)


def _decisions(journal: list[dict], exp: str, instances: dict, horizon: int,
               invalid_by_cand: dict | None = None) -> tuple[dict, dict]:
    """Gültige, unveränderte Entscheidungen der Version `exp` je Rolle {role: {(Tag, Signal): Eintrag}} und die
    Ausschlüsse je Grund (dieselbe Prüfung wie im Forward-Test, die erste Fassung gilt). `invalid_by_cand`
    erhält je Kandidat den Grund seines ersten ungültigen Eintrags."""
    out, seen, excluded = defaultdict(dict), set(), defaultdict(int)
    for j in journal:
        if not isinstance(j, dict) or j.get("horizon") != horizon or j.get("kind") != "decision":
            continue
        key = _jkey(j)
        reason = _invalid(j, exp, instances) or ("DUPLICATE" if key in seen else None)
        if reason:
            excluded[reason] += 1
            if invalid_by_cand is not None and reason != "DUPLICATE":
                invalid_by_cand.setdefault((j.get("date"), j.get("signal_id")), reason)
            continue
        seen.add(key)
        out[j.get("role", "learning")][(j["date"], j.get("signal_id"))] = j
    return out, excluded


LOOK_FRACTIONS = (1 / 3, 2 / 3, 1.0)            # vorab festgelegte Prüfzeitpunkte (Anteil von max_days ab Start)
RELATIVE = {"LEARNING_BETTER": "BETTER_THAN_BASE", "LEARNING_WORSE": "WORSE_THAN_BASE"}
ADOPTION = ("PROPOSE_ADOPTION", "CONFIRMED_ADOPTION")
CLOSING = ("BASE_CHANGED", "FALSIFIED_WORSE", "FALSIFIED_NO_EFFECT", "BETTER_THAN_BASE_NOT_PROFITABLE",
           "CONFIRMED_ADOPTION")
LOOK_DIR = "look_reports"


def look_days(max_days: int) -> list[int]:
    """Prüfzeitpunkte in Tagen ab Start; bei der Registrierung festgeschrieben."""
    return [max(1, round(max_days * f)) for f in LOOK_FRACTIONS]


def alpha_spending(alpha: float) -> list[float]:
    """Teil-Alpha je Prüfzeitpunkt nach der O'Brien-Fleming-Ausgabefunktion (Lan-DeMets):
    α(t) = 2 − 2·Φ(z_{1−α/2} / √t). Frühe Prüfungen verlangen sehr starke Evidenz; die Teil-Alphas summieren
    sich zu alpha, wiederholtes Prüfen derselben wachsenden Forward-Daten erhöht die Fehlerrate also nicht."""
    nd = NormalDist()
    z = nd.inv_cdf(1 - alpha / 2)
    cum = [2 - 2 * nd.cdf(z / math.sqrt(f)) for f in LOOK_FRACTIONS]
    return [c - prev for c, prev in zip(cum, [0.0, *cum[:-1]])]


def test_budget(out: Path = OUT, cfg: dict | None = None) -> dict:
    """Dauerhaftes Fehlerbudget ALLER Varianten-Tests: total_alpha, aufgeteilt in max_tests gleich große
    Bonferroni-Slots; jede registrierte Variante erhält bei der Registrierung genau einen Slot und verteilt ihn
    nach O'Brien-Fleming auf ihre Prüfzeitpunkte. Familienweise Fehlerrate <= Summe der Slots <= total_alpha.
    Die Definition gilt ab der ersten Registrierung unverändert (aus dem Register, nicht aus der Konfiguration);
    verbraucht ist jede jemals registrierte Variante – auch geschlossene, erneut registrierte und solche alter
    Basisversionen. Ein Neustart oder Modellwechsel setzt nichts zurück."""
    path = out / VARIANTS
    events = [e for e in (read_jsonl(path) if path.exists() else []) if e.get("event") == "registered"]
    frozen = next((e["test_budget"] for e in events if isinstance(e.get("test_budget"), dict)), None)
    conf = (cfg if cfg is not None else variant_config()).get("test_budget") or {}
    budget = frozen or {"total_alpha": float(conf.get("total_alpha", 0.05)),
                        "max_tests": int(conf.get("max_tests", 20))}
    used = len(events)
    return {"total_alpha": float(budget["total_alpha"]), "max_tests": int(budget["max_tests"]),
            "slot_alpha": float(budget["total_alpha"]) / int(budget["max_tests"]), "used": used,
            "remaining": max(0, int(budget["max_tests"]) - used), "exhausted": used >= int(budget["max_tests"]),
            "frozen_by_registry": frozen is not None}


def _economic(cmp: dict) -> str:
    """Wirtschaftlicher Erfolg der Variante gegenüber Nicht-Handeln: Tagesrendite bei gleichem Budget je Kandidat,
    dasselbe korrigierte Alpha wie der relative Test (beide müssen bestehen)."""
    if cmp["verdict"] in ("NEED_MORE_DATA", "INSUFFICIENT_COVERAGE"):
        return cmp["verdict"]
    ci = (cmp.get("learning_minus_no_trade") or {}).get("ci95")
    return "POSITIVE_EXECUTABLE_ALPHA" if ci and ci[0] > 0 else "NEGATIVE_NET" if ci and ci[1] < 0 else "NOT_PROVEN"


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _compact(cmp: dict) -> dict:
    return {k: cmp.get(k) for k in ("verdict", "candidates", "due_candidates", "coverage", "days",
                                    "independent_periods", "ci_level", "daily_mean", "learning_minus_reference",
                                    "learning_minus_no_trade")}


def _look_versions(d: Path, stem: str) -> list[Path]:
    return sorted(d.glob(f"{stem}_v*.json"), key=lambda p: int(p.stem.rsplit("_v", 1)[1])) if d.exists() else []


def freeze_look_report(out: Path, report: dict, now: datetime, subdir: str = LOOK_DIR) -> dict:
    """Prüfbericht je Test (Variante bzw. Forward-Prüfplan) und Prüfzeitpunkt unveränderlich einfrieren
    (create-only). Gleiche Eingangsdaten (input_sha256) -> der bestehende Bericht gilt unverändert. Weichen sie
    später ab (nachgelieferte oder korrigierte Daten), entsteht eine ausdrücklich versionierte Korrektur _v<n+1>
    mit Verweis auf die Vorfassung und Grund; frühere Fassungen bleiben unverändert erhalten. Maßgeblich ist die
    neueste Fassung."""
    d = out / subdir
    stem = f"{report.get('variant_id') or report['plan_id']}_look{report['look']}"
    versions = _look_versions(d, stem)
    latest = _read_json(versions[-1]) if versions else None
    if latest and latest.get("input_sha256") == report["input_sha256"]:
        return latest
    ver = len(versions) + 1
    rec = {**report, "report_id": f"{stem}_v{ver}", "version": ver, "created_at": now.isoformat(timespec="seconds"),
           "supersedes": latest.get("report_id") if latest else None,
           "correction_reason": (f"Eingangsdaten seit {latest.get('report_id')} geändert (Fälle "
                                 f"{latest.get('n_compared')} -> {report['n_compared']}, Labels "
                                 f"{len(latest.get('label_ids') or [])} -> {len(report['label_ids'])})")
           if latest else None}
    rec["report_sha256"] = _sha(rec)
    path = d / f"{rec['report_id']}.json"
    if path.exists():                                   # nie überschreiben
        raise FileExistsError(path)
    atomic_write_json(path, rec, indent=1)
    return rec


def evaluate_variants(out: Path, cands: list[dict], horizon: int, as_of: date, journal: list[dict],
                      instances: list[dict], cfg: dict | None = None, now: datetime | None = None) -> dict:
    """Fairer, sequenziell korrekter Vergleich jeder Variante auf identischen, fälligen Kandidaten mit geprüften
    Netto-Ergebnissen, primär gegen die Basis-Lernversion, zusätzlich gegen deren eingefrorene Referenz.

    * Population (Nenner der Abdeckung): ALLE Ledger-Kandidaten im vorregistrierten Prüfzeitraum
      [Start, Stichtag], je Aktie und Tag einer – unabhängig davon, ob eine Variantenentscheidung oder ein
      Ergebnis vorliegt. Fehlende oder ungültige Variantenentscheidungen, fehlende Basis-Entscheidungen und
      fehlende Entry-/Exit-Quotes zählen ausdrücklich als Ausfälle; unter MIN_COVERAGE gibt es kein Urteil.
    * Prüfzeitpunkte: Urteile nur an den bei der Registrierung festgeschriebenen look_days (1/3, 2/3, 1 von
      max_days). Prüfung k nutzt nur Kandidaten bis zu ihrem Stichtag und erst, wenn alle fällig sind; ihr Bericht
      wird eingefroren (freeze_look_report), Korrekturen werden versioniert.
    * Fehlerbudget: jede Variante testet mit ihrem bei der Registrierung zugeteilten Slot aus dem dauerhaften
      Budget (test_budget), verteilt nach O'Brien-Fleming auf die Prüfzeitpunkte. Spätere Registrierungen ändern
      frühere Ergebnisse nicht; eine Variante ohne gültigen Slot bleibt RESEARCH_ONLY (nie eine Erfolgsfreigabe).
    * Getrennte Urteile: relativ (BETTER_THAN_BASE / WORSE_THAN_BASE) und wirtschaftlich gegenüber Nicht-Handeln
      (POSITIVE_EXECUTABLE_ALPHA / NEGATIVE_NET / NOT_PROVEN). Ein Übernahmevorschlag braucht beides.
    * Status: RUNNING, RESEARCH_ONLY, BETTER_THAN_BASE_ONLY, PROPOSE_ADOPTION (offen, wird weiter geprüft);
      geschlossen (Ereignis 'closed'): FALSIFIED_WORSE, FALSIFIED_NO_EFFECT, BETTER_THAN_BASE_NOT_PROFITABLE,
      CONFIRMED_ADOPTION (letzte Prüfung), BASE_CHANGED."""
    cfg = cfg if cfg is not None else variant_config()
    now = now or datetime.now(timezone.utc)
    reg_inst = {i.get("model_id"): i for i in instances or [] if isinstance(i, dict)}
    budget = test_budget(out, cfg)
    base = experiment_id()
    base_dec = _decisions(journal, base, reg_inst, horizon)[0]
    results, closed = {}, []
    for v in (v for v in variant_registry(out).values() if v.get("horizon") == horizon):
        path = out / VARIANT_DIR / f"{v['variant_id']}.jsonl"
        invalid_ck: dict = {}
        own, invalid = _decisions(read_jsonl(path) if path.exists() else [], v["experiment_id"], reg_inst, horizon,
                                  invalid_ck)
        own_var = own.get("variant", {})
        start = date.fromisoformat(v["start_date"])
        days = v.get("look_days") or look_days(int((v.get("falsification") or {}).get("max_days") or 4 * horizon + 60))
        cutoffs = [(start + timedelta(days=d)).isoformat() for d in days]
        ready = max((k for k, c in enumerate(cutoffs, 1) if _due(c, as_of, horizon)), default=0)
        slot = v.get("alpha_slot")
        slot_alpha = v.get("slot_alpha") if slot and v.get("test_budget") and slot <= v["test_budget"]["max_tests"] \
            else None
        spend = alpha_spending(slot_alpha) if slot_alpha else None

        def run_look(k, freeze, v=v, own_var=own_var, invalid_ck=invalid_ck, cutoffs=cutoffs, days=days,
                     slot=slot, slot_alpha=slot_alpha, spend=spend):
            """Prüfung k auf ihrer festen Population [Start, Stichtag k]; friert den Bericht ein (falls freeze)."""
            alpha_k = spend[k - 1] if spend and k else (slot_alpha or 0.05)
            pop, seen = [], set()                      # je Aktie und Tag der erste Kandidat nach Signal-ID
            for c in sorted(cands, key=lambda c: (c["date"], c["ticker"], str(c["signal_id"]))):
                if k and v["start_date"] <= c["date"] <= cutoffs[k - 1] and (c["date"], c["ticker"]) not in seen:
                    seen.add((c["date"], c["ticker"]))
                    pop.append(c)
            pairs, records, why, model_ids = {"learning": [], "reference": []}, [], defaultdict(int), defaultdict(set)
            for c in pop:
                ck = (c["date"], c["signal_id"])
                j, b = own_var.get(ck), base_dec.get("learning", {}).get(ck)
                if j is None:
                    why[invalid_ck.get(ck, "NO_VARIANT_DECISION")] += 1
                    continue
                if b is None:
                    why["NO_BASE_DECISION"] += 1
                    continue
                dec = {"action": j.get("action"), "contract": j.get("contract")}
                records.append({"cand": c, "dec": dec, "preds": {}})
                model_ids["variant"].add(j.get("model_id"))
                model_ids["base"].add(b.get("model_id"))
                pairs["learning"].append({"cand": c, "learning": dec,
                                          "reference": {"action": b.get("action"), "contract": b.get("contract")}})
                if (r := base_dec.get("reference", {}).get(ck)) is not None:
                    model_ids["reference"].add(r.get("model_id"))
                    pairs["reference"].append({"cand": c, "learning": dec, "reference": {
                        "action": r.get("action"), "contract": r.get("contract")}})
            vs_base = compare_policies(pairs["learning"], horizon, alpha_k, due=len(pop), method="normal")
            vs_ref = compare_policies(pairs["reference"], horizon, alpha_k, due=len(pop), method="normal")
            excluded = defaultdict(int, why)
            for reason, n in vs_base["excluded"].items():
                excluded[reason] += n
            if not k:
                relative = economic = "AWAITING_LOOK"
            elif not slot_alpha:
                relative = economic = "RESEARCH_ONLY"
            else:
                relative, economic = RELATIVE.get(vs_base["verdict"], vs_base["verdict"]), _economic(vs_base)
            report = None
            if k and freeze:
                labels = sorted(f"{c['date']}|{c['signal_id']}|{'/'.join(e['contract'])}|{e['label_day']}"
                                for c in (p["cand"] for p in pairs["learning"]) for e in c["exprs"]
                                if e["status"] == "VERIFIED")
                inputs = [[p["cand"]["date"], p["cand"]["signal_id"], p["learning"], p["reference"],
                           [[e["contract"], e["status"], e["y"]] for e in p["cand"]["exprs"]]]
                          for p in pairs["learning"]]
                report = freeze_look_report(out, {
                    "variant_id": v["variant_id"], "look": k, "looks": len(days), "look_day": days[k - 1],
                    "data_cutoff": cutoffs[k - 1], "evaluated_as_of": as_of.isoformat(), "horizon": horizon,
                    "experiment_id": v["experiment_id"], "base_experiment_id": v.get("base_experiment_id"),
                    "current_base_experiment_id": base,
                    "model_ids": {r: sorted(x for x in s if x) for r, s in sorted(model_ids.items())},
                    "decision_ids": [f"{c['date']}|{c['signal_id']}" for c in pop], "label_ids": labels,
                    "n_population": len(pop), "n_compared": vs_base["candidates"], "coverage": vs_base["coverage"],
                    "excluded": dict(sorted(excluded.items())),
                    "test_budget": {"total_alpha": (v.get("test_budget") or {}).get("total_alpha"),
                                    "max_tests": (v.get("test_budget") or {}).get("max_tests"), "slot": slot,
                                    "slot_alpha": slot_alpha, "look_alpha": alpha_k if slot_alpha else None,
                                    "spending": spend},
                    "result": {"relative_verdict": relative, "economic_verdict": economic,
                               "vs_base": _compact(vs_base), "vs_frozen_reference": _compact(vs_ref)},
                    "release_status": ("RESEARCH_ONLY" if not slot_alpha else "ADOPTION_CANDIDATE"
                                       if relative == "BETTER_THAN_BASE" and economic == "POSITIVE_EXECUTABLE_ALPHA"
                                       else "NO_RELEASE"),
                    "input_sha256": _sha({"pop": [f"{c['date']}|{c['signal_id']}" for c in pop], "inputs": inputs,
                                          "excluded": dict(sorted(excluded.items())), "slot_alpha": slot_alpha,
                                          "look_alpha": alpha_k})}, now)
                relative, economic = report["result"]["relative_verdict"], report["result"]["economic_verdict"]
            return {"pop": pop, "records": records, "vs_base": vs_base, "vs_ref": vs_ref, "excluded": excluded,
                    "relative": relative, "economic": economic, "report": report, "alpha": alpha_k}

        def decide_status(k, relative, economic, days=days, slot_alpha=slot_alpha):
            final = k == len(days)
            if not k:
                return "RUNNING"
            if not slot_alpha:
                return "RESEARCH_ONLY"
            if relative == "WORSE_THAN_BASE":
                return "FALSIFIED_WORSE"
            if relative == "BETTER_THAN_BASE" and economic == "POSITIVE_EXECUTABLE_ALPHA":
                return "CONFIRMED_ADOPTION" if final else "PROPOSE_ADOPTION"
            if relative == "BETTER_THAN_BASE":
                return "BETTER_THAN_BASE_NOT_PROFITABLE" if final else "BETTER_THAN_BASE_ONLY"
            return "FALSIFIED_NO_EFFECT" if final else "RUNNING"

        # Jede fällige Prüfung der Reihe nach (auch eine verpasste frühere); eine Schließung beendet die Folge.
        # Geschlossene Varianten (gleiche Basis) werden bis zu ihrer schließenden Prüfung nachgerechnet: gleiche
        # Daten bestätigen den Bericht, korrigierte Daten ergeben eine versionierte Korrektur. Die historische
        # Entscheidung bleibt Status; widerspricht die aktuelle Evidenz, wird das in current_evidence ausgewiesen.
        same_base = v.get("base_experiment_id") == base
        closed_look = v.get("closed_look") or (int(str(v["look_report"]).split("_look")[1].split("_v")[0])
                                               if v.get("look_report") else ready)
        last = min(ready, closed_look) if v.get("closed_at") else ready
        look, lk, status = 0, run_look(0, False), "RUNNING"
        for k in range(1, last + 1):
            look, lk = k, run_look(k, same_base)
            status = decide_status(k, lk["relative"], lk["economic"])
            if status in CLOSING:
                break
        evidence_status = status if same_base else "BASE_CHANGED"
        if v.get("closed_at"):
            status = v.get("status")                     # historische Entscheidung, nie umgeschrieben
        elif not same_base:
            status = "BASE_CHANGED"
        current_evidence = {"status": evidence_status, "look": look, "relative_verdict": lk["relative"],
                            "economic_verdict": lk["economic"],
                            "look_report": lk["report"] and lk["report"]["report_id"],
                            "decision_look_report": v.get("look_report"),
                            "contradicts_decision": bool(v.get("closed_at") and same_base and look
                                                         and (status in ADOPTION) != (evidence_status in ADOPTION))}
        pop, records, vs_base, vs_ref, excluded = lk["pop"], lk["records"], lk["vs_base"], lk["vs_ref"], lk["excluded"]
        relative, economic, report, alpha_look = lk["relative"], lk["economic"], lk["report"], lk["alpha"]
        later = sum(1 for (d, _s) in own_var if not look or d > cutoffs[look - 1])
        if not v.get("closed_at") and status in CLOSING:
            stichtag = cutoffs[look - 1] if look else "-"
            closed.append({"event": "closed", "variant_id": v["variant_id"], "status": status,
                           "closed_at": now.isoformat(timespec="seconds"),
                           "close_reason": f"Prüfung {look}/{len(days)} (Stichtag {stichtag}): relativ {relative}, "
                                           f"netto {economic}, Abdeckung {vs_base['coverage']}",
                           "closed_look": look, "look_report": report and report["report_id"]})
        s = assess(records, horizon)
        nxt = next(((k, c) for k, c in enumerate(cutoffs, 1) if k > look), None)
        results[v["variant_id"]] = {
            "variant_id": v["variant_id"], "overrides": v["overrides"], "experiment_id": v["experiment_id"],
            "base_experiment_id": v.get("base_experiment_id"), "proposal_id": v.get("proposal_id"),
            "gap_ids": v.get("gap_ids"), "start_date": v["start_date"], "days_running": (as_of - start).days,
            "falsification": v.get("falsification"), "look_days": days, "look": look, "looks": len(days),
            "look_cutoff": cutoffs[look - 1] if look else None,
            "next_look": {"look": nxt[0], "cutoff": nxt[1],
                          "ready_after": (date.fromisoformat(nxt[1]) + timedelta(days=horizon + DUE_GRACE_DAYS))
                          .isoformat()} if nxt else None,
            "status": status, "relative_verdict": relative, "economic_verdict": economic,
            "current_evidence": current_evidence,
            "alpha_slot": slot, "slot_alpha": slot_alpha, "alpha_look": round(alpha_look, 8) if slot_alpha else None,
            "look_report": {"report_id": report["report_id"], "version": report["version"],
                            "corrected": report["version"] > 1} if report else None,
            "comparison": {"due_candidates": len(pop), "compared": vs_base["candidates"],
                           "coverage": vs_base["coverage"], "min_coverage": MIN_COVERAGE,
                           "excluded": dict(sorted(excluded.items())), "decisions_after_cutoff": later},
            "valid_decisions": len(own_var), "invalid_entries": dict(sorted(invalid.items())),
            "vs_base": vs_base, "vs_frozen_reference": vs_ref,
            "forward": {k: s[k] for k in ("n", "dates", "independent_periods", "n_selected", "mean_selected",
                                          "selected_ci95", "coverage")} | {"verdict": _verdict(s, "forward")}}
    if closed:
        append_jsonl(out / VARIANTS, closed)
    for r in results.values():
        if r["status"] in ADOPTION or (out / CHANGE_DIR / f"{r['variant_id']}.json").exists():
            r["change_proposal"] = change_proposal(out, r, now)
    return {"horizon": horizon, "base_experiment_id": base, "test_budget": budget, "variants": results}


CHANGE_DIR = "change_proposals"


def _code_change(overrides: dict) -> str:
    ((method, value),) = overrides.items()
    if method == "ridge_alpha":
        return f"modules/option_learner.py: RIDGE_ALPHA = {value} (bisher {RIDGE_ALPHA})"
    if method == "min_pred":
        return f"modules/option_learner.py: Auswahlregel der Basis TRADE erst bei Prognose > {value} (bisher > 0)"
    return (f"modules/option_learner.py: Merkmalsgruppe {value} aus LEDGER_FEATURES/CONTRACT_FEATURES entfernen "
            "(Liste siehe config/option_learner_variants.yaml)")


def _evidence_lines(r: dict) -> list[str]:
    vb, vr = r["vs_base"], r["vs_frozen_reference"]
    lb, ln, lr = vb["learning_minus_reference"], vb["learning_minus_no_trade"], vr["learning_minus_reference"]
    cmp = r["comparison"]
    return [(f"- Prüfzeitpunkt {r['look']}/{r['looks']} (Stichtag {r['look_cutoff']}), Teil-Alpha {r['alpha_look']} "
             f"(O'Brien-Fleming aus Slot {r['alpha_slot']} = {r['slot_alpha']} des dauerhaften Fehlerbudgets; "
             f"Prüfbericht {(r.get('look_report') or {}).get('report_id')})"),
            (f"- Relativ gegen die Basis-Lernversion: **{r['relative_verdict']}** – Δ {lb['mean']} je Kandidat und "
             f"Tag, KI {lb['ci95']}, {vb['days']} Tage, {vb['independent_periods']} unabhängige Perioden"),
            (f"- Netto gegen Nicht-Handeln: **{r['economic_verdict']}** – Tagesrendite {ln['mean']}, KI {ln['ci95']} "
             f"(gleiches Budget je Kandidat, Kauf Ask / Verkauf Bid, Kommission)"),
            (f"- Abdeckung: {cmp['compared']} von {cmp['due_candidates']} fälligen Kandidaten verglichen "
             f"({cmp['coverage']}, Mindestwert {cmp['min_coverage']}); ausgeschlossen: {cmp['excluded'] or 'keine'}"),
            f"- Gegen die eingefrorene Referenz: {vr['verdict']} – Δ {lr['mean']}, KI {lr['ci95']}",
            f"- Basisstrategien bei gleichem Budget (Tagesmittel): {vb['daily_mean']}"]


def _proposal_body(r: dict, now: datetime) -> str:
    lines = [f"# Änderungsvorschlag {r['variant_id']} – Options-Lernmodell", "",
             (f"_Erstellt {now.isoformat(timespec='seconds')} – zur menschlichen Prüfung, keine automatische "
              "Änderung, kein Produktionseinfluss. Die Begründung unten ist eingefroren; maßgeblich ist der "
              "aktuelle Evidenzstatus oben._"), "",
             "## Änderung", "",
             (f"- Variante: `{r['overrides']}` (Experimentversion {r['experiment_id']}, "
              f"Basis {r['base_experiment_id']})"),
             f"- Umsetzung per PR: {_code_change(r['overrides'])}",
             ("- Folge: neue Experimentversion der Basis, neue Forward-Kohorte und neue eingefrorene Referenz; "
              "offene Varianten der alten Basis werden BASE_CHANGED."), "",
             "## Anlass (gemessene Lücke)", "",
             f"- {', '.join(r.get('gap_ids') or [])} (Vorschlag {r.get('proposal_id')})", "",
             "## Evidenz bei Erstellung (prospektiv, identische Kandidaten, geprüfte Netto-Ergebnisse)", "",
             *_evidence_lines(r), "",
             "## Vorab festgelegte Falsifikation", "", f"- {r.get('falsification')}",
             f"- Prüfzeitpunkte (Tage ab Start {r['start_date']}): {r['look_days']}", "",
             "## Prüfpunkte für die menschliche Freigabe", "",
             "- Ist der Netto-Vorteil wirtschaftlich relevant und nicht nur statistisch?",
             "- Ist die Evidenz repräsentativ (Leistungszeugnis: Forschungsstichprobe, Richtung, Zeitraum)?",
             "- Entsteht der Vorteil auf wenigen Tagen oder Aktien (Ausreißer)?"]
    return "\n".join(lines) + "\n"


PROPOSAL_STATE_TEXT = {
    "VALID": "GÜLTIG – Vorteil gegenüber Basis und Nicht-Handeln am letzten Prüfzeitpunkt belegt",
    "WITHDRAWN": "ZURÜCKGEZOGEN – Vorteil nicht mehr belegt; die Begründung unten ist kein Erfolgsnachweis mehr",
    "SUPERSEDED": "ÜBERHOLT – Basisversion geändert; der Vorschlag bezieht sich auf eine alte Basis",
    "REVIEW_REQUIRED": ("PRÜFUNG ERFORDERLICH – die historische Entscheidung bleibt dokumentiert, aber korrigierte "
                        "Daten belegen den Vorteil nicht mehr; kein gültiger Erfolgsvorschlag"),
}


def change_proposal(out: Path, r: dict, now: datetime) -> dict | None:
    """Übernahmevorschlag zur menschlichen Prüfung, erstellt erst bei BETTER_THAN_BASE UND
    POSITIVE_EXECUTABLE_ALPHA an einem Prüfzeitpunkt. Die ursprüngliche Begründung bleibt unverändert erhalten;
    der aktuelle Evidenzstatus (VALID, WITHDRAWN, SUPERSEDED) wird bei jeder Auswertung im Kopf des Vorschlags und
    in current_status nachgeführt, Statuswechsel append-only in status_history. Ändert selbst nichts."""
    path = out / CHANGE_DIR / f"{r['variant_id']}.json"
    rec = _read_json(path)
    if not rec:
        if r["status"] not in ADOPTION:
            return None
        rec = {"variant_id": r["variant_id"], "created_at": now.isoformat(timespec="seconds"),
               "requires_human_pr": True, "code_change": _code_change(r["overrides"]),
               "original": {k: r[k] for k in ("overrides", "experiment_id", "base_experiment_id", "look",
                                              "look_cutoff", "relative_verdict", "economic_verdict", "comparison",
                                              "alpha_look", "alpha_slot", "slot_alpha", "look_report",
                                              "status")},
               "original_markdown": _proposal_body(r, now), "status_history": []}
    ev = r.get("current_evidence") or {}
    state = ("SUPERSEDED" if r["status"] == "BASE_CHANGED" else "REVIEW_REQUIRED" if ev.get("contradicts_decision")
             else "VALID" if r["status"] in ADOPTION else "WITHDRAWN")
    current = {"state": state, "variant_status": r["status"], "evidence_status": ev.get("status"),
               "look": r["look"], "look_cutoff": r["look_cutoff"],
               "relative_verdict": r["relative_verdict"], "economic_verdict": r["economic_verdict"],
               "coverage": r["comparison"]["coverage"], "look_report": ev.get("look_report"),
               "decision_look_report": ev.get("decision_look_report"),
               "evaluated_at": now.isoformat(timespec="seconds")}
    hist = rec.setdefault("status_history", [])
    if not hist or hist[-1]["state"] != state:
        hist.append({"state": state, "since": current["evaluated_at"], "variant_status": r["status"],
                     "look": r["look"]})
    rec["current_status"] = current
    banner = [f"> **Aktueller Evidenzstatus: {PROPOSAL_STATE_TEXT[state]}**", ">",
              (f"> Stand {current['evaluated_at']}, Prüfzeitpunkt {r['look']}/{r['looks']}: Variante {r['status']}, "
               f"relativ {r['relative_verdict']}, netto {r['economic_verdict']}, Abdeckung {current['coverage']}."),
              *([(f"> Historische Entscheidung {r['status']} (Prüfbericht {current['decision_look_report']}); "
                  f"aktuelle Evidenz {current['evidence_status']} (Prüfbericht {current['look_report']}).")]
                if state == "REVIEW_REQUIRED" else []), ""]
    atomic_write_json(path, rec, indent=1)
    atomic_write_text(path.with_suffix(".md"), "\n".join(banner) + "\n" + rec["original_markdown"])
    return {"path": str(path.with_suffix(".md")), "state": state}


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


def feature_coverage(data: list[dict]) -> dict:
    """Anteil der geprüften Labels, in denen jedes Merkmal der Spezifikation befüllt ist (Lückenerkennung)."""
    n = len(data)
    return {f: round(sum(f in d["x"] for d in data) / n, 4) if n else 0.0
            for f in (*LEDGER_FEATURES, *CONTRACT_FEATURES)}


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
              "manifest": manifest(data, as_of), "feature_coverage": feature_coverage(data), "walk_forward": wf,
              "baselines": _baselines(wf.get("vs_reference") or {}),
              "by_gate": by_gate(data),
              "forward": forward(journal, cands, horizon, instances, as_of),
              "forward_sequential": forward_sequential(out, journal, cands, horizon, instances, as_of),
              "model": model.describe() if model else None,
              "production_influence": "NONE", "orders_created": 0}
    mine = [i for i in instances if i.get("horizon") == horizon and i.get("experiment_id") == status["experiment_id"]]
    status["model_instances"] = {"count": len(mine), "latest": {k: v for k, v in mine[-1].items() if k != "features"}
                                 if mine else None}
    status["reference"] = reference(out, horizon)[1]
    status["variants"] = evaluate_variants(out, cands, horizon, as_of, journal, instances)
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
    genau einer Form oder ABSTAIN) append-only ins Journal; Empfehlung nur bei bestandener Freigabe (release_gate,
    inkl. sequenzieller Forward-Prüfung), sonst unvalidierte Research-Watchlist. Der Prüfplan und die
    Prüfberichte werden nur hier geschrieben (ein Schreiber); Prognosen und Entscheidungen hängen nie davon ab."""
    now = now or datetime.now(timezone.utc)
    _rows, cands, data = _load(day, horizon, ledger, quotes)
    journal = read_jsonl(out / JOURNAL) if (out / JOURNAL).exists() else []
    instances = read_jsonl(out / INSTANCES) if (out / INSTANCES).exists() else []
    wf, fw = walk_forward(cands, day, horizon), forward(journal, cands, horizon, instances, day)
    try:                                         # Prüfplan anlegen (falls neu) und fällige Prüfungen einfrieren
        seq = forward_sequential(out, journal, cands, horizon, instances, day, now, register=True, freeze=True)
    except Exception as exc:  # noqa: BLE001 – ein Fehler der Prüfung sperrt nur die Freigabe, nie die Prognosen
        log.warning("option_learner: sequenzielle Forward-Prüfung fehlgeschlagen (%s) – Freigabe gesperrt", exc)
        seq = {"status": "ERROR", "error": str(exc)}
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
    runs = {}
    for v in active_variants(out, horizon, day):              # isoliert: nach der Basis, eigenes Journal
        try:
            runs[v["variant_id"]] = _run_variant(out, v, cands, train, day, horizon, now)
        except Exception as exc:  # noqa: BLE001 – eine Variante darf die Basis nie stören
            log.warning("option_learner: Variante %s fehlgeschlagen (%s) – Basis unberührt", v["variant_id"], exc)
            runs[v["variant_id"]] = f"Fehler: {exc}"
    gate = release_gate(wf, fw, seq)            # fail closed, auch wenn ein Bericht fehlt oder unvollständig ist
    best = {}
    for d in sorted((d for d in decisions if d["action"] == "TRADE"), key=lambda d: -d["pred_net"]):
        best.setdefault(d["ticker"], d)          # je Aktie höchstens eine Empfehlung
    recs = list(best.values()) if gate["allowed"] else []
    watch = sorted((d for d in decisions if d["contract"]), key=lambda d: -d["pred_net"])[:WATCHLIST_N]
    result = {"date": day.isoformat(), "horizon": horizon, "model_available": model is not None,
              "experiment_id": exp, "model_id": inst and inst["model_id"],
              "reference_model_id": ref_meta and ref_meta["model_id"], "verified_labels": len(data),
              "walk_forward": wf, "forward": fw, "forward_sequential": seq, "release": gate,
              "mail_eligible": horizon == MAIL_HORIZON and bool(recs),
              "recommendations": recs, "watchlist": [] if recs else watch,
              "journaled": len(new), "decisions": len(decisions), "variant_runs": runs,
              "production_influence": "NONE", "orders_created": 0}
    target = out / "recommendations" / f"{day.isoformat()}_{horizon}d.json"
    atomic_write_json(target, result, indent=1)
    atomic_write_text(target.with_suffix(".md"), render(result))
    return result


def _fmt(v, pct=True):
    return "n/a" if v is None else (f"{v * 100:+.1f} %" if pct else f"{v:.2f}")


def render(res: dict) -> str:
    seq = res.get("forward_sequential")
    gate = res.get("release") or release_gate(res.get("walk_forward"), res.get("forward"), seq)
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
              f"- Forward sequenziell (vorab festgelegte Prüfzeitpunkte): **{_sequential_text(seq)}**",
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
        lines += ["## Empfehlungen (Walk-Forward validiert, Forward an vorab festgelegtem Prüfzeitpunkt bestätigt)",
                  "",
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


def _improvement_answers(st: dict, imp: dict, horizon: int) -> dict:
    """Autonome Research-Verbesserung: erkannte Lücken, Vorschläge und laufende Varianten-Experimente."""
    gaps = [g for g in (imp or {}).get("gaps") or [] if g.get("horizon") == horizon]
    props = [p for p in (imp or {}).get("proposals") or [] if p.get("horizon") == horizon]
    variants = list(((st or {}).get("variants") or {}).get("variants", {}).values())
    states = [v["status"] for v in variants]
    tested = "; ".join(
        f"{next(iter(v['overrides']))}={next(iter(v['overrides'].values()))}: {v['status']} (Slot "
        f"{v.get('alpha_slot')}, Prüfung "
        f"{v.get('look', 0)}/{v.get('looks', '?')}, relativ {v.get('relative_verdict')}, netto "
        f"{v.get('economic_verdict')}, Abdeckung {(v.get('comparison') or {}).get('coverage')}"
        + (f", nächste Prüfung nach {v['next_look']['ready_after']}" if v.get("next_look") else "") + ")"
        + (f", Vorschlag {v['change_proposal']['state']}" if v.get("change_proposal") else "")
        for v in variants)
    adopted = any(v["status"] in ADOPTION and not (v.get("current_evidence") or {}).get("contradicts_decision")
                  for v in variants)
    budget = ((st or {}).get("variants") or {}).get("test_budget")
    return {
        "gaps": ("Ja" if gaps else "offen",
                 (f"{len(gaps)} Lücke(n), zuerst: {gaps[0]['finding']}; Vorschläge: "
                  f"{sum(p['auto_experiment'] for p in props)} automatisch testbar, "
                  f"{sum(p['approval_required'] for p in props)} zur Freigabe") if gaps else
                 "keine gemessene Lücke (oder research_director --options noch nicht gelaufen)"),
        "improvements": ("Ja" if adopted else "offen" if any(s not in CLOSING for s in states)
                         else "Nein" if variants else "offen",
                         ((tested + ("; gültiger Übernahmevorschlag liegt zur Prüfung vor" if adopted else ""))
                          if variants else "noch keine Variante registriert") + (
                             f"; Fehlerbudget {budget['used']}/{budget['max_tests']} Tests verbraucht"
                             + (" – ERSCHÖPFT, keine neuen Tests" if budget.get("exhausted") else "")
                             if budget else "")),
    }


def _sequential_text(seq: dict) -> str:
    """Kurztext der sequenziellen Forward-Prüfung für Leistungszeugnis und Bericht."""
    if not seq or not seq.get("status"):
        return "noch nicht ausgewertet"
    if seq["status"] in ("NO_PLAN", "PLAN_INVALID", "ERROR"):
        return {"NO_PLAN": "kein Prüfplan (wird beim nächsten Scanner-Lauf vorab festgelegt)",
                "PLAN_INVALID": "Prüfplan verändert oder beschädigt – Freigabe gesperrt",
                "ERROR": f"Fehler: {seq.get('error')}"}[seq["status"]]
    b, nxt = seq.get("binding") or {}, seq.get("next_look")
    text = (f"{seq['status']}, Plan {seq.get('plan_id')} ab {seq.get('start_date')}, Prüfung {seq.get('look', 0)}/"
            f"{seq.get('looks')}")
    if b:
        text += (f" (Stichtag {b.get('data_cutoff')}: {b.get('verdict')}, KI {b.get('selected_ci95')} zum Niveau "
                 f"{b.get('ci_level')}, n={b.get('n_selected')}, Entscheidungsabdeckung {b.get('decision_coverage')}, "
                 f"Bericht {b.get('report_id') or 'nicht eingefroren'})")
    return text + (f"; nächste Prüfung nach {nxt['ready_after']}" if nxt else "")


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
    seq = st.get("forward_sequential") or {}
    gate = release_gate(wf, fw, seq)
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
        "sequential": ({"CONFIRMED": "Ja", "REJECTED_NEGATIVE": "Nein", "NOT_CONFIRMED": "Nein"}.get(
                           seq.get("status"), "offen"), _sequential_text(seq)),
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
        "proven": ("Ja" if gate["allowed"] else "Nein",
                   "nur bei VALIDATED, FORWARD_POSITIVE und sequenzieller Bestätigung samt Mindestzahlen"),
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
    ("sequential", ("Ist das an vorab festgelegten Prüfzeitpunkten statistisch korrekt bestätigt (kein tägliches "
                    "Nachprüfen bis zum Zufallstreffer)?")),
    ("experiment", "Welche Experimentversion und Modellinstanzen stehen dahinter (ausgeschlossene Prognosen)?"),
    ("improves", "Entscheidet die lernende Version besser als die eingefrorene Referenz (gleiche Signale)?"),
    ("expressions", "Welche Optionsform schneidet ab, was bringt die Wahl des Modells (Ø netto, ungesehen)?"),
    ("representative", "Ist die Evidenz repräsentativ (Forschung vs. Richtungssignale, Zeitraum)?"),
    ("gates", "Was bringen die verworfenen Kandidaten je Filter (Ø netto, beschreibend)?"),
    ("proven", "Ist eine profitable Optionsstrategie nachgewiesen?"),
    ("self_limit", "Begrenzt es sich selbst?"),
    ("gaps", "Welche Lücken erkennt es selbst (Daten, Modell, Evidenz)?"),
    ("improvements", "Welche Verbesserungen testet es selbst – mit welchem Ergebnis?"),
)


def report_card(out: Path = OUT, horizons: tuple = (HORIZON, 5)) -> dict:
    """Leistungszeugnis aus den Status-Dateien: beantwortet die Audit-Fragen automatisch je Horizont."""
    imp = _read_json(out / "improvement.json")
    answers = {}
    for h in horizons:
        st = _read_json(out / f"status_{h}d.json")
        try:
            answers[h] = {**_answers(st), **(_improvement_answers(st, imp, h) if st else {})}
        except (KeyError, TypeError, IndexError, ValueError) as exc:     # alter/unvollständiger Status: Spalte leer
            log.warning("option_learner: status_%sd.json unvollständig (%s) – Spalte bleibt leer", h, exc)
            answers[h] = {}
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
            or not release_gate(res.get("walk_forward"), res.get("forward"), res.get("forward_sequential"))["allowed"]):
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
                          "forward_sequential": (res.get("forward_sequential") or {}).get("status"),
                          "journaled": res["journaled"]}))
        if a.mail:
            log.info("option_learner mail: %s", mail(res))


if __name__ == "__main__":
    main()

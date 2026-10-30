"""modules/option_learner.py – lernt aus geprüften Netto-Options-Ergebnissen und leitet Empfehlungen ab (Research).

Kreislauf (autonom, kein LLM, keine Markt-API, keine Order):
  Kandidaten-Ledger (Merkmale + Kontrakt zum Entscheidungszeitpunkt eingefroren)
  -> Ausdrucksformen je Kandidat: Produktions-Strategie (single|spread), Einzel-Option (real_option) und die
     Research-Kontraktvariante (real_option_alt); derselbe Kontrakt zählt nur einmal
  -> Label: Ausführbare Rendite minus Kommission, NUR wenn research_evidence_discovery.provenance == VERIFIED
     (Entry und Exit aus dem Quote-Archiv nachgerechnet)
  -> Ridge-Regression mit festen Hyperparametern auf standardisierten Merkmalen
  -> Walk-Forward je Signaltag: trainiert nur auf Labels, deren Mark-Tag VOR dem Testtag liegt (Embargo)
  -> Prognose-Journal: jede Tagesprognose append-only, bevor ihr Label existiert (prospektiv)
  -> Forward-Auswertung des Journals, sobald die Labels geprüft vorliegen
  -> EMPFEHLUNG nur bei Walk-Forward-Status VALIDATED, nicht widerlegtem Forward-Test und Prognose > 0;
     je Kandidat die Ausdrucksform mit der höchsten Prognose; sonst Beobachtungsliste mit ehrlichem Status.
  -> Leistungszeugnis: Trainings-Manifest (welche geprüften Labels in welches Modell flossen), append-only
     Status-Historie (Lernkurve gegenüber der Vorversion), Vergleich mit "nie handeln" und "alles handeln".
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
REPORT_CARD = "report_card"
HORIZON = 20
COMMISSION_PER_LEG_SIDE = 0.65          # USD je Kontrakt und Seite (wie cost_model der Feedback-Outcomes)
RIDGE_ALPHA = 10.0
MIN_TRAIN_ROWS = 100
MIN_TRAIN_DATES = 8
MIN_OOS_DATES = 20                      # Walk-Forward-Verdikt erst ab so vielen Testtagen
MIN_SELECTED = 30                       # mindestens so viele OOS-Fälle mit Prognose > 0
MIN_FEATURE_COVERAGE = 0.5
BOOTSTRAP = 2000
WATCHLIST_N = 5
SPEC = {"name": "option_learner_ridge_v2", "alpha": RIDGE_ALPHA, "target": "executable return - commission",
        "expressions": ["strategy", "single", "single_alt"],
        "winsor": [0.01, 0.99], "selection": "pred > 0", "min_oos_dates": MIN_OOS_DATES,
        "min_selected": MIN_SELECTED}

LEDGER_FEATURES = (
    "impact", "surprise", "sentiment_score", "sentiment_confidence", "sentiment_drift", "news_count",
    "news_novelty", "rel_volume", "scan_day_ret", "short_mom", "log_market_cap", "short_pct_float",
    "options_skew_25d", "options_skew_ratio", "dealer_oi_ratio", "ml_exp_ret_60", "ml_q10_ret_60",
    "ml_q90_ret_60", "ml_exp_dd_60", "ml_p_gt10_60", "ml_disagreement_sd", "ml_hgb_xs20_v1", "ml_enet_xs20_v1",
)
CONTRACT_FEATURES = ("c_entry_spread", "c_is_spread", "c_is_call", "c_iv", "c_dte", "c_moneyness", "c_delta",
                     "c_log_oi", "c_iv_rank")
# (Ledger-Schlüssel, Outcome-Präfix, Name) – real_strategy heißt nach Struktur "single" oder "spread"
SOURCES = (("real_strategy", "real_strat", None), ("real_option", "real_opt", "single"),
           ("real_option_alt", "real_alt", "single_alt"))


def spec_hash() -> str:
    return hashlib.sha256(json.dumps(SPEC, sort_keys=True).encode()).hexdigest()[:12]


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
        pseudo = {**row, "real_strategy": {**rs, "iv_rank": rs.get("iv_rank", base_rs.get("iv_rank"))},
                  "real_option": contract, "outcomes": mapped}
        out.append({"expression": name or ("spread" if len(symbols) > 1 else "single"),
                    "contract": list(symbols), "row": pseudo})
    return out


def dataset(rows: list[dict], quotes: list[dict], as_of: date, horizon: int = HORIZON,
            counts: dict | None = None) -> list[dict]:
    """Geprüfte Labels mit Mark-Tag <= as_of je Ausdrucksform: {date, label_day, signal_id, ticker, expression,
    contract, x, y}. `counts` erhält die Prüfergebnisse aller fälligen Labels (auch der verworfenen)."""
    index = red.quote_index(quotes)
    seen, out = set(), []
    for r in rows:
        if not isinstance(r, dict) or not r.get("date") or not r.get("ticker"):
            continue
        key = (r["date"], r["ticker"], str(r.get("signal_id") or r.get("event_id") or ""))
        if key in seen:
            continue
        seen.add(key)
        for e in expressions(r):
            pr = e["row"]
            ld, y = _label_day(pr, horizon), net_label(pr, horizon)
            if ld is None or ld > as_of or y is None:
                continue
            status = red.provenance(pr, index, horizon)
            if counts is not None:
                counts[status] = counts.get(status, 0) + 1
            if status != "VERIFIED":
                continue
            out.append({"date": r["date"], "label_day": ld, "signal_id": r.get("signal_id"), "ticker": r["ticker"],
                        "expression": e["expression"], "contract": e["contract"], "x": features(pr), "y": y})
    return sorted(out, key=lambda d: (d["date"], d["ticker"], d["expression"]))


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
    lo, hi = np.quantile(y, SPEC["winsor"])
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


def _date_bootstrap_ci(by_date: dict[str, list[float]], seed: int = 7) -> tuple[float, float] | None:
    days = sorted(by_date)
    if len(days) < 2:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(BOOTSTRAP):
        vals = [v for d in (rng.choice(days) for _ in days) for v in by_date[d]]
        means.append(sum(vals) / len(vals))
    means.sort()
    return means[int(0.025 * BOOTSTRAP)], means[int(0.975 * BOOTSTRAP) - 1]


def score(preds: list[dict]) -> dict:
    """Kennzahlen für Prognosen [{date, pred, y}]: Rang-IC je Tag, Mittel der Auswahl (pred > 0) mit Datums-Bootstrap."""
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
    ci = _date_bootstrap_ci(sel) if n_sel else None
    return {"n": len(preds), "dates": len(by_date),
            "mean_all": round(float(np.mean([p["y"] for p in preds])), 6) if preds else None,
            "ic_mean": round(float(np.mean(ics)), 6) if ics else None,
            "ic_t": round(ic_t, 3) if ic_t is not None else None,
            "n_selected": n_sel, "selected_dates": len(sel),
            "mean_selected": round(sum(v for vs in sel.values() for v in vs) / n_sel, 6) if n_sel else None,
            "selected_ci95": [round(c, 6) for c in ci] if ci else None}


def _by_expression(preds: list[dict]) -> dict:
    """Strategievergleich: OOS-Mittel je Ausdrucksform, alle und nur die ausgewählten (Prognose > 0)."""
    groups = defaultdict(list)
    for p in preds:
        groups[p.get("expression") or "single"].append(p)
    out = {}
    for name, ps in sorted(groups.items()):
        sel = [p["y"] for p in ps if p["pred"] > 0]
        out[name] = {"n": len(ps), "mean_all": round(float(np.mean([p["y"] for p in ps])), 6),
                     "n_selected": len(sel), "mean_selected": round(float(np.mean(sel)), 6) if sel else None}
    return out


def walk_forward(data: list[dict]) -> dict:
    """Je Signaltag trainieren mit Labels, deren Mark-Tag vor dem Testtag liegt; Testtag vorhersagen."""
    preds = []
    for day in sorted({d["date"] for d in data}):
        test_day = date.fromisoformat(day)
        train = [d for d in data if d["label_day"] < test_day]
        if len({d["date"] for d in train}) < MIN_TRAIN_DATES:
            continue
        model = fit(train)
        if model is None:
            continue
        test = [d for d in data if d["date"] == day]
        for d, p in zip(test, model.predict([t["x"] for t in test])):
            preds.append({"date": day, "pred": float(p), "y": d["y"], "expression": d.get("expression")})
    s = score(preds)
    if s["dates"] < MIN_OOS_DATES or s["n_selected"] < MIN_SELECTED:
        verdict = "NEED_MORE_DATA"
    elif s["selected_ci95"] and s["selected_ci95"][0] > 0 and (s["ic_t"] or 0) > 2:
        verdict = "VALIDATED"
    else:
        verdict = "NOT_VALIDATED"
    return {**s, "by_expression": _by_expression(preds), "verdict": verdict}


def forward(journal: list[dict], data: list[dict], horizon: int) -> dict:
    """Prospektive Prognosen (vor dem Label geschrieben) gegen die inzwischen geprüften Labels; Zuordnung über
    Signal und Kontrakt."""
    labels = {(d["date"], d["signal_id"], tuple(d["contract"])): d for d in data}
    preds = []
    for j in journal:
        if j.get("horizon") != horizon:
            continue
        d = labels.get((j.get("date"), j.get("signal_id"), tuple(j.get("contract") or ())))
        if d is None or str(j.get("created_at", ""))[:10] > d["label_day"].isoformat():
            continue
        preds.append({"date": j["date"], "pred": float(j["pred_net"]), "y": d["y"], "expression": d["expression"]})
    s = score(preds)
    ci = s["selected_ci95"]
    if s["dates"] < MIN_OOS_DATES or s["n_selected"] < MIN_SELECTED:
        verdict = "NEED_MORE_DATA"
    elif ci and ci[1] < 0:
        verdict = "FORWARD_NEGATIVE"
    elif ci and ci[0] > 0:
        verdict = "FORWARD_POSITIVE"
    else:
        verdict = "INCONCLUSIVE"
    return {**s, "by_expression": _by_expression(preds), "verdict": verdict}


def _load(as_of: date, horizon: int, ledger: Path, quotes: Path, counts: dict | None = None):
    rows = red.load(ledger, as_of)
    labelled = [r for r in rows if isinstance(r, dict) and any(
        _num((r.get("outcomes") or {}).get(f"{prefix}_ret_{horizon}d")) is not None for _k, prefix, _n in SOURCES)]
    return rows, dataset(rows, red.load_quotes(quotes, labelled, as_of), as_of, horizon, counts)


def manifest(data: list[dict], as_of: date) -> dict:
    """Welche geprüften Labels in das Modell zum Stichtag flossen (Hash über Signal, Kontrakt, Mark-Tag)."""
    ids = sorted(f"{d['date']}|{d['signal_id']}|{'/'.join(d['contract'])}|{d['label_day']}" for d in data)
    by_expr = defaultdict(int)
    for d in data:
        by_expr[d["expression"]] += 1
    return {"cutoff": as_of.isoformat(), "n_labels": len(data),
            "labels_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16],
            "first_signal": ids[0][:10] if ids else None, "last_signal": max((d["date"] for d in data), default=None),
            "by_expression": dict(sorted(by_expr.items())), "all_labels_verified": True,
            "trains_model": len(data) >= MIN_TRAIN_ROWS}


def _baselines(wf: dict) -> dict:
    """Modell-Auswahl gegen "nie handeln" (0) und "alles handeln" (Mittel aller OOS-Fälle)."""
    sel, all_, ci = wf.get("mean_selected"), wf.get("mean_all"), wf.get("selected_ci95")
    return {"no_trade": 0.0, "trade_all": all_, "model_selected": sel,
            "beats_no_trade": bool(ci and ci[0] > 0),
            "beats_trade_all": bool(sel is not None and all_ is not None and sel > all_)}


def _vs_previous(history: list[dict], status: dict) -> dict | None:
    prev = next((h for h in reversed(history) if h.get("as_of") < status["as_of"]), None)
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
    rows, data = _load(as_of, horizon, ledger, quotes, counts)
    journal = read_jsonl(out / JOURNAL) if (out / JOURNAL).exists() else []
    model = fit(data)
    wf = walk_forward(data)
    week = (as_of.toordinal() - 7)
    recent = [r for r in rows if isinstance(r, dict) and r.get("date")
              and date.fromisoformat(str(r["date"])[:10]).toordinal() > week]
    status = {"as_of": as_of.isoformat(), "horizon": horizon, "spec": SPEC, "spec_hash": spec_hash(),
              "verified_labels": len(data), "label_dates": len({d["date"] for d in data}),
              "provenance_counts": dict(sorted(counts.items())),
              "ledger_rows_7d": len(recent),
              "option_snapshots_7d": sum(1 for r in recent if isinstance(r.get("real_option"), dict)),
              "manifest": manifest(data, as_of), "walk_forward": wf, "baselines": _baselines(wf),
              "forward": forward(journal, data, horizon), "model": model.describe() if model else None,
              "production_influence": "NONE", "orders_created": 0}
    hist_path = out / f"status_history_{horizon}d.jsonl"
    history = read_jsonl(hist_path) if hist_path.exists() else []
    status["vs_previous"] = _vs_previous(history, status)
    atomic_write_json(out / f"status_{horizon}d.json", status, indent=1)
    last = history[-1] if history else {}
    if (last.get("as_of"), (last.get("manifest") or {}).get("labels_sha256"), last.get("spec_hash")) != \
            (status["as_of"], status["manifest"]["labels_sha256"], status["spec_hash"]):
        append_jsonl(hist_path, [{k: status[k] for k in ("as_of", "horizon", "spec_hash", "verified_labels",
                                                          "manifest", "walk_forward", "baselines", "forward")}])
    report_card(out)
    return status


def recommend(day: date, *, horizon: int = HORIZON, ledger: Path = red.LEDGER, quotes: Path = red.QUOTES,
              out: Path = OUT, now: datetime | None = None) -> dict:
    """Prognosen für alle Ausdrucksformen der heutigen Kandidaten; append-only ins Journal; je Kandidat die beste
    Ausdrucksform; Empfehlung nur bei validiertem und nicht widerlegtem Modell, sonst Beobachtungsliste."""
    now = now or datetime.now(timezone.utc)
    rows, data = _load(day, horizon, ledger, quotes)
    journal = read_jsonl(out / JOURNAL) if (out / JOURNAL).exists() else []
    wf, fw = walk_forward(data), forward(journal, data, horizon)
    model = fit([d for d in data if d["label_day"] < day])
    done = {(j.get("date"), j.get("signal_id"), j.get("horizon"), tuple(j.get("contract") or ())) for j in journal}
    new, best = [], {}
    for r in rows:
        if not isinstance(r, dict) or r.get("date") != day.isoformat() or model is None:
            continue
        exprs = expressions(r)
        preds = model.predict([features(e["row"]) for e in exprs])
        for e, p in zip(exprs, preds):
            rs = e["row"]["real_strategy"]
            ask, mid = net_quote(_legs_at_decision(rs))
            item = {"date": r["date"], "signal_id": r.get("signal_id"), "ticker": r["ticker"],
                    "expression": e["expression"], "contract": e["contract"], "strategy": rs.get("strategy"),
                    "entry_ask_net": ask, "entry_mid_net": mid, "option_sample": r.get("option_sample"),
                    "horizon": horizon, "pred_net": round(float(p), 6), "model_spec": spec_hash(),
                    "walk_forward_verdict": wf["verdict"], "forward_verdict": fw["verdict"],
                    "created_at": now.isoformat(timespec="seconds")}
            if (item["date"], item["signal_id"], horizon, tuple(item["contract"])) not in done:
                new.append(item)
            if item["strategy"] in ("LONG_CALL", "BULL_CALL_SPREAD"):
                key = (item["signal_id"], item["ticker"])
                if key not in best or item["pred_net"] > best[key]["pred_net"]:
                    best[key] = item
    if new:
        append_jsonl(out / JOURNAL, new)
    validated = wf["verdict"] == "VALIDATED" and fw["verdict"] != "FORWARD_NEGATIVE"
    recs = sorted((i for i in best.values() if validated and i["pred_net"] > 0), key=lambda i: -i["pred_net"])
    watch = sorted(best.values(), key=lambda i: -i["pred_net"])[:WATCHLIST_N]
    result = {"date": day.isoformat(), "horizon": horizon, "model_available": model is not None,
              "verified_labels": len(data), "walk_forward": wf, "forward": fw,
              "recommendations": recs, "watchlist": [] if recs else watch,
              "journaled": len(new), "production_influence": "NONE", "orders_created": 0}
    target = out / "recommendations" / f"{day.isoformat()}_{horizon}d.json"
    atomic_write_json(target, result, indent=1)
    atomic_write_text(target.with_suffix(".md"), render(result))
    return result


def _fmt(v, pct=True):
    return "n/a" if v is None else (f"{v * 100:+.1f} %" if pct else f"{v:.2f}")


def render(res: dict) -> str:
    wf, fw = res["walk_forward"], res["forward"]
    lines = [f"# Options-Lernmodell – {res['date']} ({res['horizon']} T)", "",
             "_Research, keine Order, keine Anlageberatung._", "",
             f"- Geprüfte Options-Labels: {res['verified_labels']}",
             (f"- Walk-Forward: **{wf['verdict']}** – Testtage {wf['dates']}, Auswahl n={wf['n_selected']}, "
              f"Ø netto {_fmt(wf['mean_selected'])} (alle {_fmt(wf['mean_all'])}), 95-%-KI {wf['selected_ci95']}, "
              f"Rang-IC {wf['ic_mean']} (t {wf['ic_t']})"),
             (f"- Forward (prospektives Journal): **{fw['verdict']}** – Tage {fw['dates']}, "
              f"Auswahl n={fw['n_selected']}, Ø netto {_fmt(fw['mean_selected'])}"), ""]
    if res["recommendations"]:
        lines += ["## Empfehlungen (Modell validiert)", "",
                  "| Ticker | Strategie | Form | Kontrakt | Ask / Mid (Entscheidung) | erwartet netto |",
                  "|---|---|---|---|---|---|"]
        lines += [f"| {i['ticker']} | {i['strategy']} | {i['expression']} | {' / '.join(i['contract'])} | "
                  f"{_fmt(i['entry_ask_net'], False)} / {_fmt(i['entry_mid_net'], False)} | {_fmt(i['pred_net'])} |"
                  for i in res["recommendations"]]
        lines += ["", ("Erwartung gilt für Kauf zum Ask und Verkauf zum Bid; ein Limit zum Mid ist günstiger, "
                       "aber nicht garantiert ausgeführt.")]
    else:
        lines += ["## Keine Empfehlung", "",
                  "Das Modell ist nicht validiert oder sieht heute keinen positiven Netto-Erwartungswert.", ""]
        if res["watchlist"]:
            lines += ["Beobachtungsliste (unvalidiert, nicht handeln):", "",
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
    prev = st.get("vs_previous") or {}
    d_sel = prev.get("delta_mean_selected")
    expr = "; ".join(f"{k} {_fmt(v['mean_all'])} (n={v['n']})" for k, v in (wf.get("by_expression") or {}).items())
    proven = wf["verdict"] == "VALIDATED" and fw["verdict"] == "FORWARD_POSITIVE"
    active = wf["verdict"] == "VALIDATED" and fw["verdict"] != "FORWARD_NEGATIVE"
    return {
        "data": ("Ja" if st.get("option_snapshots_7d") else "Nein",
                 f"{st.get('ledger_rows_7d', 0)} Kandidaten, {st.get('option_snapshots_7d', 0)} mit Optionskontrakt (7 T)"),
        "provenance": ("offen" if not due else "Ja" if verified / due >= 0.9 else "Teilweise" if verified else "Nein",
                       f"{verified}/{due} fällige Labels geprüft"),
        "learns": ("Ja" if man["trains_model"] else "offen",
                   f"{man['n_labels']} geprüfte Labels, Manifest {man['labels_sha256']}"),
        "validated": ({"VALIDATED": "Ja", "NOT_VALIDATED": "Nein"}.get(wf["verdict"], "offen"),
                      f"{wf['verdict']}, {wf['dates']} Testtage"),
        "baselines": ("Ja" if base["beats_no_trade"] and base["beats_trade_all"] else
                      "offen" if wf["n_selected"] == 0 else "Nein",
                      f"Auswahl {_fmt(base['model_selected'])} vs. alles {_fmt(base['trade_all'])} vs. nie 0 %"),
        "forward": ({"FORWARD_POSITIVE": "Ja", "FORWARD_NEGATIVE": "Nein"}.get(fw["verdict"], "offen"),
                    f"{fw['verdict']}, {fw['n']} vorab festgehaltene Prognosen"),
        "improves": ("offen" if d_sel is None else "Ja" if d_sel > 0 else "Nein",
                     "keine Vorversion" if not prev else
                     f"Auswahl {_fmt(d_sel)} ggü. {prev['previous_as_of']}, Labels {prev['delta_labels']:+d}"),
        "expressions": ("offen" if not expr else "Ja", expr or "noch keine Testtage"),
        "proven": ("Ja" if proven else "Nein", "nur bei VALIDATED und FORWARD_POSITIVE"),
        "self_limit": ("Ja", f"Empfehlungen {'aktiv' if active else 'gesperrt'} (nur bei validiertem Modell)"),
    }


CARD_QUESTIONS = (
    ("data", "Werden Daten automatisch gesammelt?"),
    ("provenance", "Sind die Options-Ergebnisse gegen das Quote-Archiv geprüft?"),
    ("learns", "Lernt das System aus geprüften Options-Ergebnissen?"),
    ("validated", "Ist das Modell auf ungesehenen Tagen validiert?"),
    ("baselines", "Schlägt es 'nie handeln' und 'alles handeln'?"),
    ("forward", "Bestätigt es sich auf vorab festgehaltenen Prognosen?"),
    ("improves", "Wird es besser als die Vorversion?"),
    ("expressions", "Welche Ausdrucksform schneidet ab (Ø netto, ungesehen)?"),
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
    """Mail nur bei echten Empfehlungen (validiertes Modell) – keine tägliche Nullmeldung."""
    if not res["recommendations"]:
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

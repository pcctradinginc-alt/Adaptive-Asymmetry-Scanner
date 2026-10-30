"""modules/option_learner.py – lernt aus geprüften Netto-Options-Ergebnissen und leitet Empfehlungen ab (Research).

Kreislauf (autonom, kein LLM, keine Markt-API, keine Order):
  Kandidaten-Ledger (Merkmale + Kontrakt zum Entscheidungszeitpunkt eingefroren)
  -> Label: real_strat_ret_{h}d minus Kommission, NUR wenn research_evidence_discovery.provenance == VERIFIED
     (Entry und Exit aus dem Quote-Archiv nachgerechnet)
  -> Ridge-Regression mit festen Hyperparametern auf standardisierten Merkmalen
  -> Walk-Forward je Signaltag: trainiert nur auf Labels, deren Mark-Tag VOR dem Testtag liegt (Embargo)
  -> Prognose-Journal: jede Tagesprognose append-only, bevor ihr Label existiert (prospektiv)
  -> Forward-Auswertung des Journals, sobald die Labels geprüft vorliegen
  -> EMPFEHLUNG nur bei Walk-Forward-Status VALIDATED, nicht widerlegtem Forward-Test und Prognose > 0;
     sonst Beobachtungsliste mit ehrlichem Status.
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
SPEC = {"name": "option_learner_ridge_v1", "alpha": RIDGE_ALPHA, "target": "real_strat_ret_{h}d - commission",
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


def dataset(rows: list[dict], quotes: list[dict], as_of: date, horizon: int = HORIZON) -> list[dict]:
    """Geprüfte Labels mit Mark-Tag <= as_of: {date, label_day, signal_id, ticker, x, y}."""
    index = red.quote_index(quotes)
    seen, out = set(), []
    for r in rows:
        if not isinstance(r, dict) or not r.get("date") or not r.get("ticker"):
            continue
        key = (r["date"], r["ticker"], str(r.get("signal_id") or r.get("event_id") or ""))
        if key in seen:
            continue
        seen.add(key)
        ld = _label_day(r, horizon)
        y = net_label(r, horizon)
        if ld is None or ld > as_of or y is None or red.provenance(r, index, horizon) != "VERIFIED":
            continue
        out.append({"date": r["date"], "label_day": ld, "signal_id": r.get("signal_id"), "ticker": r["ticker"],
                    "x": features(r), "y": y})
    return sorted(out, key=lambda d: (d["date"], d["ticker"]))


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
            preds.append({"date": day, "pred": float(p), "y": d["y"]})
    s = score(preds)
    if s["dates"] < MIN_OOS_DATES or s["n_selected"] < MIN_SELECTED:
        verdict = "NEED_MORE_DATA"
    elif s["selected_ci95"] and s["selected_ci95"][0] > 0 and (s["ic_t"] or 0) > 2:
        verdict = "VALIDATED"
    else:
        verdict = "NOT_VALIDATED"
    return {**s, "verdict": verdict}


def forward(journal: list[dict], data: list[dict], horizon: int) -> dict:
    """Prospektive Prognosen (vor dem Label geschrieben) gegen die inzwischen geprüften Labels."""
    labels = {(d["date"], d["signal_id"]): d for d in data}
    preds = []
    for j in journal:
        if j.get("horizon") != horizon:
            continue
        d = labels.get((j.get("date"), j.get("signal_id")))
        if d is None or str(j.get("created_at", ""))[:10] > d["label_day"].isoformat():
            continue
        preds.append({"date": j["date"], "pred": float(j["pred_net"]), "y": d["y"]})
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
    return {**s, "verdict": verdict}


def _load(as_of: date, horizon: int, ledger: Path, quotes: Path) -> tuple[list[dict], list[dict]]:
    rows = red.load(ledger, as_of)
    labelled = [r for r in rows if isinstance(r, dict)
                and _num((r.get("outcomes") or {}).get(f"real_strat_ret_{horizon}d")) is not None]
    return rows, dataset(rows, red.load_quotes(quotes, labelled, as_of), as_of, horizon)


def evaluate(as_of: date, *, horizon: int = HORIZON, ledger: Path = red.LEDGER, quotes: Path = red.QUOTES,
             out: Path = OUT) -> dict:
    _rows, data = _load(as_of, horizon, ledger, quotes)
    journal = read_jsonl(out / JOURNAL) if (out / JOURNAL).exists() else []
    model = fit(data)
    status = {"as_of": as_of.isoformat(), "horizon": horizon, "spec": SPEC, "spec_hash": spec_hash(),
              "verified_labels": len(data), "label_dates": len({d["date"] for d in data}),
              "walk_forward": walk_forward(data), "forward": forward(journal, data, horizon),
              "model": model.describe() if model else None,
              "production_influence": "NONE", "orders_created": 0}
    atomic_write_json(out / f"status_{horizon}d.json", status, indent=1)
    return status


def recommend(day: date, *, horizon: int = HORIZON, ledger: Path = red.LEDGER, quotes: Path = red.QUOTES,
              out: Path = OUT, now: datetime | None = None) -> dict:
    """Prognosen für die heutigen Kandidaten mit Kontrakt; append-only ins Journal; Empfehlung nur bei
    validiertem und nicht widerlegtem Modell, sonst Beobachtungsliste."""
    now = now or datetime.now(timezone.utc)
    rows, data = _load(day, horizon, ledger, quotes)
    journal = read_jsonl(out / JOURNAL) if (out / JOURNAL).exists() else []
    wf, fw = walk_forward(data), forward(journal, data, horizon)
    model = fit([d for d in data if d["label_day"] < day])
    today = [r for r in rows if isinstance(r, dict) and r.get("date") == day.isoformat()
             and isinstance(r.get("real_strategy"), dict) and r["real_strategy"].get("legs")]
    preds = model.predict([features(r) for r in today]) if model else []
    done = {(j.get("date"), j.get("signal_id"), j.get("horizon")) for j in journal}
    new, items = [], []
    for r, p in zip(today, preds):
        rs = r["real_strategy"]
        legs = _legs_at_decision(rs)
        ask, mid = net_quote(legs)
        item = {"date": r["date"], "signal_id": r.get("signal_id"), "ticker": r["ticker"],
                "contract": [g.get("symbol") for g in legs], "strategy": rs.get("strategy"),
                "entry_ask_net": ask, "entry_mid_net": mid,
                "option_sample": r.get("option_sample"), "horizon": horizon, "pred_net": round(float(p), 6),
                "model_spec": spec_hash(), "walk_forward_verdict": wf["verdict"], "forward_verdict": fw["verdict"],
                "created_at": now.isoformat(timespec="seconds")}
        items.append(item)
        if (item["date"], item["signal_id"], horizon) not in done:
            new.append(item)
    if new:
        append_jsonl(out / JOURNAL, new)
    tradable = [i for i in items if i["strategy"] in ("LONG_CALL", "BULL_CALL_SPREAD")]
    validated = wf["verdict"] == "VALIDATED" and fw["verdict"] != "FORWARD_NEGATIVE"
    recs = sorted((i for i in tradable if validated and i["pred_net"] > 0), key=lambda i: -i["pred_net"])
    watch = sorted(tradable, key=lambda i: -i["pred_net"])[:WATCHLIST_N]
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
                  "| Ticker | Strategie | Kontrakt | Ask / Mid (Entscheidung) | erwartet netto |",
                  "|---|---|---|---|---|"]
        lines += [f"| {i['ticker']} | {i['strategy']} | {' / '.join(i['contract'])} | "
                  f"{_fmt(i['entry_ask_net'], False)} / {_fmt(i['entry_mid_net'], False)} | {_fmt(i['pred_net'])} |"
                  for i in res["recommendations"]]
        lines += ["", ("Erwartung gilt für Kauf zum Ask und Verkauf zum Bid; ein Limit zum Mid ist günstiger, "
                       "aber nicht garantiert ausgeführt.")]
    else:
        lines += ["## Keine Empfehlung", "",
                  "Das Modell ist nicht validiert oder sieht heute keinen positiven Netto-Erwartungswert.", ""]
        if res["watchlist"]:
            lines += ["Beobachtungsliste (unvalidiert, nicht handeln):", "",
                      "| Ticker | Strategie | Kontrakt | Prognose netto |", "|---|---|---|---|"]
            lines += [f"| {i['ticker']} | {i['strategy']} | {' / '.join(i['contract'])} | {_fmt(i['pred_net'])} |"
                      for i in res["watchlist"]]
    return "\n".join(lines) + "\n"


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

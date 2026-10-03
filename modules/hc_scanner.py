"""
modules/hc_scanner.py – High-Confidence-Kandidaten, Alerts, Deduplizierung

    python -m modules.hc_scanner [--send] [--dry-run]

KEINE ORDERAUSFÜHRUNG. Das Modul erzeugt ausschließlich Research-Signale
(Datei + optional E-Mail). Es gibt keinen Broker-Anschluss.

Ein Kandidat wird nur HIGH/VERY HIGH, wenn ALLE Bedingungen erfüllt sind:
  Global (sonst gar keine Alerts):
    * OOS-kalibrierte Regel aktiv (meta_learning.calibrate_hc_rule: Schwellen
      nur auf Kalibrierjahren gewählt, auf späteren Jahren validiert)
    * Wahrscheinlichkeits-Kalibrierung des aktiven Ensembles zuverlässig
      (ECE <= Schwelle) und Unsicherheitsintervalle kalibriert
    * kein Feature-Drift außerhalb des Trainingsbereichs
  Je Titel:
    * kalibrierte P(Überrendite 20d > 0) >= Regel-Schwelle
    * Modell-Uneinigkeit <= Regel-Schwelle (nur falls empirisch belegt)
    * Liquidität obere Hälfte
    * Datenqualität HIGH, erwartete 60d-Rendite > 0, Analogien >= k mit
      Gewinnanteil >= 50 %, Asymmetrie (Analog-MFE/|MAE|) >= 1
    * Earnings-Datum bekannt und nicht innerhalb der Sperrfrist
Konservative Plausibilitätsfilter (Analogien, Asymmetrie) können Alerts nur
verhindern, nie erzeugen. Keine Mindestanzahl: "keine Kandidaten" ist normal.

Deduplizierung (outputs/research/alerts_state.json, Log alerts_log.jsonl):
erneute Meldung nur bei materieller Verbesserung (Wahrscheinlichkeit,
erwartete Rendite), Regimewechsel oder Neuentstehung nach Pause.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from modules import ml_research as ml

log = logging.getLogger(__name__)

from modules.meta_learning import MP  # noqa: E402

HC = MP["high_confidence"]
AL = MP["alerts"]
OUT = ml.OUT_DIR
CANDIDATES_PATH = OUT / "hc_candidates.json"
STATE_PATH = OUT / "alerts_state.json"
LOG_PATH = OUT / "alerts_log.jsonl"
MAX_ECE = 0.05
MIN_ANALOG_SHARE = 0.5
MIN_ASYMMETRY = 1.0


def _interp(x: float, m: dict) -> float | None:
    if not m or not m.get("x"):
        return None
    return float(np.interp(x, m["x"], m["y"]))


def _load(path: Path, default):
    try:
        return json.loads(path.read_text()) if path.exists() else default
    except (OSError, json.JSONDecodeError) as e:
        log.warning(f"hc_scanner: {path} nicht lesbar ({e})")
        return default


def ensemble_scores(latest_ranks: dict[str, dict], models: list[str], weights: dict | None) -> dict[str, dict]:
    """Aktives Ensemble je Ticker: gewichteter Mittelwert (Rang − 0,5); ohne
    Gewichte gleichgewichtet (statischer Champion). Plus Uneinigkeit."""
    tick = set().union(*[set(latest_ranks.get(m, {})) for m in models]) if models else set()
    out = {}
    for t in tick:
        r = {m: latest_ranks[m][t] for m in models if t in latest_ranks.get(m, {})}
        if len(r) < max(2, len(models) // 2):
            continue
        if weights:
            w = {m: (weights.get(m) or {}).get("weight", 0.0) for m in r}
            tot = sum(w.values()) or 1.0
            s = sum(w[m] * (r[m] - 0.5) for m in r) / tot
        else:
            s = float(np.mean([v - 0.5 for v in r.values()]))
        vals = list(r.values())
        out[t] = {"score": s, "model_ranks": r, "agreement_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                  "bull_share": float(np.mean([v > 0.5 for v in vals]))}
    ranks = sorted(out, key=lambda k: out[k]["score"])
    n = len(ranks)
    for i, t in enumerate(ranks):
        out[t]["rank_pct"] = (i + 1) / n
    return out


def earnings_days(ticker: str, today: date) -> int | None:
    """Tage bis zum nächsten bekannten Earnings-Termin (yfinance, offiziell
    veröffentlichter Kalender). None = unbekannt."""
    try:
        import yfinance as yf
        cal = yf.Ticker(ticker).calendar
    except Exception as e:  # noqa: BLE001 – yfinance wirft uneinheitlich; unbekannt -> kein Alert
        log.warning(f"hc_scanner: Earnings-Kalender {ticker} nicht abrufbar: {e}")
        return None
    dates = (cal or {}).get("Earnings Date") if isinstance(cal, dict) else None
    if not dates:
        return None
    future = [d for d in dates if isinstance(d, date) and d >= today]
    return (min(future) - today).days if future else None


def company_name(ticker: str) -> str | None:
    try:
        import yfinance as yf
        return (yf.Ticker(ticker).info or {}).get("shortName")
    except Exception as e:  # noqa: BLE001 – rein kosmetisch
        log.warning(f"hc_scanner: Name {ticker} nicht abrufbar: {e}")
        return None


def global_gate(rule: dict, cards_meta: dict, calibration: dict | None) -> list[str]:
    reasons = []
    if not rule.get("enabled"):
        reasons.append(f"keine OOS-validierte Regel: {rule.get('disabled_reason')}")
    ece = rule.get("active_ece")
    if ece is None or ece > MAX_ECE:
        reasons.append(f"Wahrscheinlichkeits-Kalibrierung unzuverlässig (ECE {ece})")
    if not (calibration or {}).get("interval_calibrated"):
        reasons.append("Unsicherheitsintervalle nicht kalibriert")
    if rule.get("feature_drift_flag"):
        reasons.append("Feature-Drift außerhalb des Trainingsbereichs")
    if not cards_meta:
        reasons.append("keine aktuellen Unsicherheitskarten")
    if rule.get("probability_validated") is not True:               # Audit F12 / P1-4
        reasons.append("Wahrscheinlichkeiten nicht validiert: "
                       + ("; ".join(rule.get("probability_validation_reasons") or []) or "kein Bucket-Test"))
    return reasons


def regime_compatible(meta_json: dict, nextv: dict, regime: dict) -> tuple[bool, str]:
    """World-State-Kompatibilität: Nur handeln, wo das aktive Ensemble historisch
    OOS verdient hat. Bestätigte Abstinenz-Regel hat Vorrang (next_validation)."""
    vix, tr = regime.get("vix"), regime.get("spy_trend_200")
    if vix is None or tr is None:
        return False, "Regime unbekannt"
    ab = (nextv or {}).get("abstention_confirmation") or {}
    if ab.get("confirmed") or (ab.get("forward") or {}).get("confirmed"):   # kontaminierte Bestätigung zählt nicht (F03)
        ok = vix >= 20 or tr <= 0
        return ok, f"Abstinenz-Regel (bestätigt): {'aktiv' if ok else 'ruhiges Aufwärts-Regime -> nichts tun'}"
    act = (meta_json or {}).get("active_ensemble", "static_equal")
    reg = (((meta_json or {}).get("approaches") or {}).get(act) or {}).get("by_regime") or {}
    kv = "vix_ge_20" if vix >= 20 else "vix_lt_20"
    kt = "spy_uptrend" if tr > 0 else "spy_downtrend"
    ev, et = (reg.get(kv) or {}).get("expectancy"), (reg.get(kt) or {}).get("expectancy")
    if ev is None or et is None:
        return False, "keine Regime-Historie"
    ok = ev > 0 and et > 0
    return ok, f"historische Expectancy {kv}={ev}, {kt}={et}"


def _kg_edges(graph) -> list[dict]:
    edges = graph.get("edges") if isinstance(graph, dict) else getattr(graph, "edges", None)
    if isinstance(edges, dict):
        edges = list(edges.values())
    return [e for e in (edges or []) if isinstance(e, dict)]


def intelligence_checks(cands: list[dict], panel, specs: list[dict], out_dir: Path) -> tuple[list[dict], list[dict]]:
    """Phase 16: Ein Signal wird trotz hoher Wahrscheinlichkeit abgelehnt bei
    Counterfactual-Fragilität, KG-Widerspruch, Unknown-Risk (Blind-Spot-Segment)
    oder negativem Portfolio-Beitrag. -> (behalten, abgelehnt mit Grund)."""
    if not cands:
        return [], []
    from modules import blind_spots as bs
    from modules import counterfactual as cf
    from modules import decision_intel as dint
    nextv = _load(out_dir / "next_validation.json", {})
    kept, rejected = [], []
    cfx = cf.latest_counterfactuals(panel, specs) if specs else {}
    per = (cfx or {}).get("per_ticker") or {}
    try:
        from modules import knowledge_graph as kg
        graph = kg.build_graph(**kg.load_inputs())
    except (ImportError, OSError, ValueError, KeyError, TypeError) as e:
        log.warning(f"hc_scanner: Knowledge Graph nicht verfügbar ({e})")
        graph = None
    kg_measured = graph is not None and any(e.get("evidence_type") == "measured_oos" for e in _kg_edges(graph))
    clusters = [{"properties": c.get("common_properties") or {}} for c in nextv.get("blind_spot_clusters") or []]
    snap = panel[panel["date"] == panel["date"].max()].set_index("ticker")
    for c in cands:
        t = c["ticker"]
        why = []
        cft = per.get(t)
        if cft is None:
            why.append("keine Counterfactual-Auswertung")
        elif cft.get("fragile"):
            why.append(f"fragil: fällt unter '{cft.get('worst_case')}' aus dem Top-Dezil")
        if graph is not None:
            ev = kg.ticker_evidence(graph, t)
            # Widersprüche sind nur mit GEMESSENEN Kanten möglich; ohne sie ist der
            # Check wirkungslos und wird als inaktiv ausgewiesen (Audit P3-2).
            if kg_measured and ev.get("contradictions"):
                why.append(f"Knowledge Graph widersprüchlich: {ev['contradictions'][:2]}")
            c["kg_evidence"] = {k: ev.get(k) for k in ("leading_indicators", "exposures")}
            c["kg_check"] = "aktiv" if kg_measured else "inaktiv (0 gemessene Kanten)"
        if t in snap.index and clusters and bool(bs.match(snap.loc[[t]].reset_index(), clusters).iloc[0]):
            why.append("Unknown-Risk: Titel liegt in einem Blind-Spot-Segment")
        c["counterfactual"] = cft
        if why:
            rejected.append({"ticker": t, "reasons": why})
        else:
            kept.append(c)
    if kept:
        info = {c["ticker"]: {"exp": (c["card"] or {}).get("expected_return_60"), "prob": c["prob"],
                              "downside": ((c["card"] or {}).get("interval_80") or [None])[0],
                              "drawdown": (c["card"] or {}).get("expected_drawdown_60"), "asymmetry": c["asymmetry"]}
                for c in kept}
        prof = dint.candidate_profile(list(info), panel, info)
        final = []
        for c in kept:
            p = prof.get(c["ticker"]) or {}
            c["portfolio"] = p
            if (p.get("portfolio_utility") or 0) <= 0:
                rejected.append({"ticker": c["ticker"], "reasons": [f"Portfolio-Nutzen {p.get('portfolio_utility')} <= 0"]})
            else:
                final.append(c)
        kept = final
    return kept, rejected


def evaluate_candidates(ens: dict, cards: dict, rule: dict, liquidity: dict, today: date,
                        earnings_fn=earnings_days) -> list[dict]:
    r = rule.get("rule") or {}
    use_agree = rule.get("agreement_used", False)
    out = []
    for t, e in sorted(ens.items(), key=lambda kv: -kv[1]["score"]):
        prob = _interp(e["rank_pct"], (rule.get("prob_map") or {}).get("prob"))
        if prob is None or prob < r.get("prob", 1.1):
            continue
        if use_agree and (e["agreement_sd"] is None or e["agreement_sd"] > r.get("agreement_sd", 0)):
            continue
        if (liquidity.get(t) is None) or liquidity[t] <= HC["min_dollar_vol_rank"]:
            continue
        c = cards.get(t) or {}
        an = c.get("analogs") or {}
        mfe, mae = an.get("analog_median_mfe_60"), an.get("analog_median_mae_60")
        asym = (mfe / abs(mae)) if (mfe is not None and mae not in (None, 0)) else None
        checks = {
            "data_quality_high": c.get("data_quality") == "HIGH",
            "expected_return_60_positive": (c.get("expected_return_60") or -1) > 0,
            "analogs_enough": (an.get("analog_n") or 0) >= int(ml.UNC["analogs_k"]),
            "analogs_positive": (an.get("analog_share_positive") or 0) >= MIN_ANALOG_SHARE,
            "asymmetry": asym is not None and asym >= MIN_ASYMMETRY,
            "regime_confidence_ok": c.get("regime_confidence") in ("HIGH", "MEDIUM"),
        }
        if not all(checks.values()):
            continue
        ed = earnings_fn(t, today)
        if ed is None and HC["require_event_data"]:
            continue
        if ed is not None and ed <= HC["earnings_block_days"]:
            continue
        very = prob >= r.get("prob", 1) + HC["very_high_extra_prob"] and (an.get("analog_share_positive") or 0) >= 0.6
        out.append({"ticker": t, "prob": prob, "ens": e, "card": c, "asymmetry": asym, "earnings_in_days": ed,
                    "confidence": "VERY HIGH" if very else "HIGH"})
    return out


def build_alert(c: dict, rule: dict, signal_date: str, regime: dict, versions: dict, name: str | None) -> dict:
    card, e = c["card"], c["ens"]
    an = card.get("analogs") or {}
    cf = card.get("counterfactual") or {}
    lo, hi = (card.get("interval_80") or [None, None])[:2]
    exp20 = _interp(e["rank_pct"], (rule.get("prob_map") or {}).get("exp_xs20"))
    weights = rule.get("current_weights") or {}
    top_model = max(weights, key=lambda m: (weights[m] or {}).get("weight", 0)) if weights else None
    fp = (rule.get("failure_profiles") or {}).get(top_model, {}) if top_model else {}
    sig = hashlib.sha256(f"{c['ticker']}|{signal_date}|{versions}".encode()).hexdigest()[:16]
    supports = list((cf.get("supports") or {}).keys())
    against = list((cf.get("weighs_against") or {}).keys())
    return {
        "type": "HIGH-CONFIDENCE TRADE CANDIDATE", "no_order_execution": True,
        "ticker": c["ticker"], "company": name, "signal_date": signal_date, "signal_id": sig,
        "signal_version": versions,
        "calibrated_probability": round(c["prob"], 3),
        "probability_note": "P(20d-Überrendite ggü. S&P 500 > 0), isotonisch auf OOS-Daten kalibriert",
        "expected_return_20d": None if exp20 is None else round(exp20, 4),
        "expected_return_20d_note": "marktbereinigt, OOS-Mittel vergleichbarer Ränge",
        "expected_return_60d": card.get("expected_return_60"),
        "expected_return_120d": None, "expected_return_120d_note": "nicht modelliert – keine Scheingenauigkeit",
        "expected_downside": lo, "interval_80_return_60d": [lo, hi],
        "expected_max_adverse_excursion": card.get("expected_drawdown_60"),
        "expected_max_favorable_excursion": an.get("analog_median_mfe_60"),
        "asymmetry_ratio": None if c["asymmetry"] is None else round(c["asymmetry"], 2),
        "model_agreement": {"sd_of_ranks": e["agreement_sd"], "bull_share": e["bull_share"], "ranks": e["model_ranks"]},
        "regime_compatibility": {"regime": regime, "regime_confidence": card.get("regime_confidence"),
                                 "top_model": top_model, "top_model_fails_in": fp.get("fails", [])},
        "data_quality": card.get("data_quality"),
        "historical_analogue_confidence": an.get("analog_share_positive"),
        "number_historical_analogues": an.get("analog_n"),
        "historical_analogue_win_rate": an.get("analog_share_positive"),
        "historical_analogues_note": "Analogie-Engine: OOS-Nutzen nicht validiert (Audit P3-3) – nur Kontext",
        "historical_analogue_median_return": an.get("analog_median_ret_60"),
        "bull_case": [f"{f}: stützt die Prognose (Beitrag {v})" for f, v in (cf.get("supports") or {}).items()]
        + [f"Modell {top_model} funktioniert empirisch in: {', '.join(fp.get('works', [])[:3]) or '–'}"],
        "bear_case": [f"{f}: spricht dagegen (Beitrag {v})" for f, v in (cf.get("weighs_against") or {}).items()]
        + [f"Analog-Verlustanteil {round(1 - (an.get('analog_share_positive') or 0), 2)}"],
        "key_risk_factors": [x for x in [
            f"Earnings in {c['earnings_in_days']} Tagen" if c["earnings_in_days"] is not None else None,
            f"80-%-Band 60T reicht bis {lo}" if lo is not None else None,
            f"Top-Modell versagt empirisch in: {', '.join(fp.get('fails', [])[:3])}" if fp.get("fails") else None,
            f"Modell-Uneinigkeit sd {round(e['agreement_sd'], 3)}" if e["agreement_sd"] is not None else None] if x],
        "most_important_positive_features": supports, "most_important_negative_features": against,
        "invalidation_conditions": [x for x in [
            f"60T-Rendite fällt unter das untere 80-%-Band ({lo})" if lo is not None else None,
            f"Merkmal '{supports[0]}' fällt unter den Querschnittsmedian" if supports else None,
            "Ensemble-Rang fällt unter die Regel-Schwelle beim nächsten Wochenlauf",
            "Regimewechsel (VIX-Schwelle 20 / SPY unter 200-Tage-Linie)"] if x],
        "model_version": versions.get("models"), "meta_model_version": versions.get("meta"),
        "confidence": c["confidence"],
        "counterfactual": {k: (c.get("counterfactual") or {}).get(k) for k in
                           ("expected_return_60", "scenario_ranks", "worst_case", "fragile", "dominant_assumption")},
        "knowledge_graph_evidence": c.get("kg_evidence"),
        "portfolio": c.get("portfolio"),
        "uncertainty_note": "Kalibrierte Wahrscheinlichkeiten im Querschnitt liegen typischerweise nur wenige "
                            "Prozentpunkte über 50 %; alle Angaben sind Schätzungen mit breiten Bändern.",
    }


def dedup(alerts: list[dict], state: dict, today: date, regime_key: str) -> tuple[list[dict], dict]:
    """-> (zu meldende Alerts, neuer Zustand). Nie Daten überschreiben: Verlauf je Ticker."""
    to_send = []
    for a in alerts:
        t = a["ticker"]
        s = state.get(t)
        reason = None
        if s is None:
            reason = "neu"
        else:
            last = date.fromisoformat(s["last_seen"])
            if (today - last).days > AL["gap_days_new_signal"]:
                reason = "nach Pause neu entstanden"
            elif a["calibrated_probability"] >= s["last_alert_probability"] + AL["reprobability_delta"]:
                reason = "Wahrscheinlichkeit deutlich gestiegen"
            elif (a["expected_return_20d"] or 0) >= (s.get("last_alert_expected_return") or 0) + AL["reexpected_return_delta"]:
                reason = "erwartete Rendite deutlich gestiegen"
            elif s.get("last_regime") != regime_key:
                reason = "Regimewechsel"
            elif a["confidence"] == "VERY HIGH" and s.get("last_confidence") == "HIGH":
                reason = "Confidence gestiegen"
        hist = (s or {}).get("history", [])
        hist.append({"date": today.isoformat(), "p": a["calibrated_probability"], "exp20": a["expected_return_20d"],
                     "confidence": a["confidence"], "alerted": reason is not None})
        ns = {"signal_id": a["signal_id"] if reason in ("neu", "nach Pause neu entstanden") or s is None else s["signal_id"],
              "first_alert": (s or {}).get("first_alert") if s and reason != "nach Pause neu entstanden" else today.isoformat(),
              "last_seen": today.isoformat(), "history": hist[-52:],
              "last_alert": today.isoformat() if reason else (s or {}).get("last_alert"),
              "last_alert_probability": a["calibrated_probability"] if reason else s["last_alert_probability"],
              "last_alert_expected_return": a["expected_return_20d"] if reason else s.get("last_alert_expected_return"),
              "last_confidence": a["confidence"] if reason else s.get("last_confidence"),
              "last_regime": regime_key,
              "confidence_change": None if not s else round(a["calibrated_probability"] - s["last_alert_probability"], 4),
              "prediction_change": None if not s else round((a["expected_return_20d"] or 0) -
                                                            (s.get("last_alert_expected_return") or 0), 4)}
        state[t] = ns
        if reason:
            to_send.append({**a, "alert_reason": reason, "signal_id": ns["signal_id"]})
    return to_send, state


def render_alert(a: dict) -> tuple[str, str, str]:
    subj = f"[{a['confidence']} CONFIDENCE] {a['ticker']} – Adaptive Asymmetry Signal"
    keys = [("Ticker", "ticker"), ("Company", "company"), ("Signal date", "signal_date"), ("Signal-ID", "signal_id"),
            ("Anlass", "alert_reason"), ("Calibrated probability of positive outcome", "calibrated_probability"),
            ("Expected 20d return (marktbereinigt)", "expected_return_20d"), ("Expected 60d return", "expected_return_60d"),
            ("Expected 120d return", "expected_return_120d_note"), ("Expected downside (unteres 80-%-Band 60T)", "expected_downside"),
            ("Expected MAE", "expected_max_adverse_excursion"), ("Expected MFE", "expected_max_favorable_excursion"),
            ("Asymmetry ratio", "asymmetry_ratio"), ("Model agreement", "model_agreement"),
            ("Regime compatibility", "regime_compatibility"), ("Data quality", "data_quality"),
            ("Historical analogue confidence", "historical_analogue_confidence"),
            ("Number historical analogues", "number_historical_analogues"),
            ("Historical analogue win rate", "historical_analogue_win_rate"),
            ("Historical analogue median return", "historical_analogue_median_return"),
            ("Bull case", "bull_case"), ("Bear case", "bear_case"), ("Key risk factors", "key_risk_factors"),
            ("Most important positive features", "most_important_positive_features"),
            ("Most important negative features", "most_important_negative_features"),
            ("Invalidation conditions", "invalidation_conditions"), ("Model version", "model_version"),
            ("Meta-model version", "meta_model_version"), ("Confidence", "confidence"), ("Unsicherheit", "uncertainty_note")]
    lines = ["HIGH-CONFIDENCE TRADE CANDIDATE – Research-Signal, KEINE Orderausführung, keine Anlageberatung", ""]
    lines += [f"{k}: {a.get(v)}" for k, v in keys]
    text = "\n".join(lines)
    html = "<html><body style='font-family:sans-serif'><h3>" + subj + "</h3><p><b>Research-Signal, keine " \
        "Orderausführung.</b></p><table border='1' cellpadding='4' style='border-collapse:collapse'>" + \
        "".join(f"<tr><td>{k}</td><td>{a.get(v)}</td></tr>" for k, v in keys) + "</table></body></html>"
    return subj, html, text


def run(send: bool = False, dry_run: bool = True, today: date | None = None, panel=None,
        earnings_fn=earnings_days, name_fn=company_name, out_dir: Path = OUT,
        health_snapshot: Path | None = None) -> dict:
    today = today or datetime.now(timezone.utc).date()
    if health_snapshot is None:                     # Produktion: outputs/health; Tests: im out_dir
        from modules.source_health import SNAPSHOT
        health_snapshot = SNAPSHOT if Path(out_dir) == OUT else Path(out_dir) / SNAPSHOT.name
    rule = _load(out_dir / "hc_thresholds.json", {})
    cards_doc = _load(out_dir / "ml_cards.json", {})
    mlr = _load(out_dir / "ml_research.json", {})
    calibration = mlr.get("calibration")
    preds = ml._read_predictions(out_dir / "ml_predictions")
    latest = max((r["prediction_date"] for r in preds), default=None)
    ranks = {r["model_id"]: r["rank_pct"] for r in preds if r["prediction_date"] == latest}
    models = [m for m in (rule.get("models") or list(ranks)) if m in ranks]
    active = rule.get("active_ensemble", "static_equal")
    weights = rule.get("current_weights") if active != "static_equal" else None
    reasons = global_gate(rule, cards_doc.get("cards", {}), calibration)
    # Safe Mode = Modell/Drift (meta_cognition) ODER Daten (source_health); fehlt eins -> fail-closed
    from modules.source_health import effective_safe_mode
    sm = effective_safe_mode(out_dir, snapshot_path=health_snapshot,
                             now=datetime(today.year, today.month, today.day, 23, 59, tzinfo=timezone.utc))
    if sm["active"]:
        reasons.append(f"SAFE MODE aktiv: {'; '.join(sm['reasons'])}")
    mi = (_load(out_dir / "meta_learning.json", {}) or {}).get("model_intelligence") or {}
    if mi and sum(1 for m in mi.values() if m.get("trend") == "deteriorating") / len(mi) >= 0.5:
        reasons.append("Alpha Decay: Mehrheit der Modelle 'deteriorating'")
    if latest is None:
        reasons.append("keine aktuellen Modellprognosen")
    elif (today - date.fromisoformat(latest)).days > 8:
        reasons.append(f"Prognosen veraltet ({latest})")
    result = {"date": today.isoformat(), "signal_date": latest, "active_ensemble": active,
              "rule": rule.get("rule"), "enabled": not reasons, "disabled_reason": "; ".join(reasons) or None,
              "candidates": [], "sent": []}
    ens = ensemble_scores(ranks, models, weights) if ranks else {}
    if not reasons and panel is not None:
        snap = panel[panel["date"] == panel["date"].max()]
        liquidity = dict(zip(snap["ticker"], snap["log_dollar_vol"]))
        regime = {"vix": float(snap["vix"].iloc[0]) if len(snap) else None,
                  "spy_trend_200": float(snap["spy_trend_200"].iloc[0]) if len(snap) else None}
        ok_reg, why_reg = regime_compatible(_load(out_dir / "meta_learning.json", {}),
                                            _load(out_dir / "next_validation.json", {}), regime)
        result["regime_compatibility"] = why_reg
        cands = evaluate_candidates(ens, cards_doc.get("cards", {}), rule, liquidity, today, earnings_fn) if ok_reg else []
        if not ok_reg:
            result["disabled_reason"] = f"Regime nicht kompatibel – besser nichts tun ({why_reg})"
        reg = ml.load_registry()
        st_reg, _ = ml.check_registry(reg)
        specs = [x for x in reg.get("models") or [] if st_reg.get(x["id"]) == "valid"]
        cands, result["rejected_by_intelligence"] = intelligence_checks(cands, panel, specs, out_dir)
        versions = {"models": {m: r.get("spec_hash") for r in preds if r["prediction_date"] == latest
                               for m in [r["model_id"]]}, "meta": rule.get("meta_version"), "ensemble": active}
        alerts = [build_alert(c, rule, latest, regime, versions, name_fn(c["ticker"])) for c in cands]
        result["candidates"] = alerts
        regime_key = f"{'vix_ge_20' if (regime['vix'] or 0) >= 20 else 'vix_lt_20'}|" \
                     f"{'up' if (regime['spy_trend_200'] or 0) > 0 else 'down'}"
        prev_state = _load(out_dir / "alerts_state.json", {})
        to_send, state = dedup(alerts, json.loads(json.dumps(prev_state)), today, regime_key)
        if send or dry_run:
            from modules.mailer import send_mail
            for a in to_send:
                subj, html, text = render_alert(a)
                res = send_mail(subj, html, text, dry_run=dry_run or not send)
                result["sent"].append({"ticker": a["ticker"], "status": res.get("status"), "reason": a["alert_reason"]})
                if not send:
                    continue                                   # Dry-Run verändert keinen Zustand
                if res.get("status") != "sent":                # nicht zugestellt -> Zustand zurück, nächster Lauf versucht erneut (F10)
                    if a["ticker"] in prev_state:
                        state[a["ticker"]] = prev_state[a["ticker"]]
                    else:
                        state.pop(a["ticker"], None)
                with open(out_dir / "alerts_log.jsonl", "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"date": today.isoformat(), "ticker": a["ticker"], "signal_id": a["signal_id"],
                                         "reason": a["alert_reason"], "status": res.get("status"),
                                         "confidence": a["confidence"], "p": a["calibrated_probability"]}) + "\n")
        if send:
            (out_dir / "alerts_state.json").write_text(json.dumps(state, indent=1))
    # Gedächtnis: Top-Liste des aktiven Ensembles + Kandidaten (append-only)
    if ens and latest and send:
        from modules import prediction_memory as pm
        top = sorted(ens.items(), key=lambda kv: -kv[1]["score"])[:50]
        spec_by_model = {r["model_id"]: r.get("spec_hash") for r in preds if r["prediction_date"] == latest}
        cards = cards_doc.get("cards", {})
        cand_t = {a["ticker"] for a in result["candidates"]}
        rows = []
        for t, e in top:
            c = cards.get(t) or {}
            rows.append({"signal_date": latest, "ticker": t,
                         "model_versions": {m: spec_by_model.get(m) for m in e["model_ranks"]},   # spec_hash (Audit P2-4)
                         "meta_model_version": rule.get("meta_version"),
                         "ensemble": active, "raw_model_predictions": e["model_ranks"], "model_weights": weights or "equal",
                         "final_prediction": round(e["score"], 5),
                         "predicted_return": c.get("expected_return_60"), "predicted_drawdown": c.get("expected_drawdown_60"),
                         "predicted_probability": _interp(e["rank_pct"], (rule.get("prob_map") or {}).get("prob")),
                         "uncertainty": {"interval_80": c.get("interval_80"), "agreement_sd": e["agreement_sd"]},
                         "calibration_state": {"interval_calibrated": c.get("interval_calibrated"),
                                               "active_ece": rule.get("active_ece")},
                         "data_quality": c.get("data_quality"),
                         "historical_analogy_score": (c.get("analogs") or {}).get("analog_share_positive"),
                         "signal_reason": "high_confidence" if t in cand_t else "top50_active_ensemble"})
        result["memory_new"] = pm.record_predictions(rows, out_dir / "prediction_memory")
        if panel is not None:
            result["memory_outcomes"] = pm.record_outcomes(panel, out_dir / "prediction_memory")
    (out_dir / "hc_candidates.json").write_text(json.dumps(result, indent=1, default=str, ensure_ascii=False))
    return result


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--send", action="store_true")
    g.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    import os
    import pickle
    panel = None
    cache = os.environ.get("ML_PANEL_CACHE")
    if cache and Path(cache).exists():
        with open(cache, "rb") as fh:
            panel = pickle.load(fh)  # noqa: S301 – eigene, im selben Job erzeugte Datei
    res = run(send=a.send, dry_run=not a.send, panel=panel)
    print(json.dumps({k: v for k, v in res.items() if k != "candidates"}, indent=1, default=str))
    print(f"Kandidaten: {[c['ticker'] for c in res['candidates']] or 'No high-confidence candidates this week.'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

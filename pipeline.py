"""
pipeline.py v8.0 – Asymmetry Pro Flow

Neue Reihenfolge:
  1. Hard-Filter (Cap > 2B, Vol > 1M, RV > 0.8)
  2. Prescreening (Haiku)
  3. Options ROI Pre-Check (vor Deep Analysis → Fail Fast)
  4. Quick Monte Carlo (n=3.000, 30d)
  5. Deep Analysis Sonnet (MIT MC-Ergebnissen im Prompt)
  6. Mismatch-Score + Intraday-Delta
  7. Final Monte Carlo (n=10.000, adaptive DTE)
  8. RL-Scoring
  9. Options Design + ROI-Gate
 10. Email Report (immer)
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from modules.data_ingestion      import DataIngestion
from modules.prescreener         import Prescreener
from modules.deep_analysis       import DeepAnalysis
from modules.mismatch_scorer     import MismatchScorer
from modules.mirofish_simulation import MirofishSimulation, compute_time_value_efficiency
from modules.rl_agent            import RLScorer
from modules.options_designer    import OptionsDesigner
from modules.reporter            import Reporter
from modules.risk_gates          import RiskGates
from modules.email_reporter      import send_status_email
from modules.finbert_sentiment   import score_candidate
from modules.intraday_delta      import filter_by_intraday_delta
from modules.alpha_sources       import enrich_with_alpha_sources
from modules.data_validator      import validate_candidate_data, compute_option_roi
from modules.premium_signals     import enrich_top_candidates
from modules.sentiment_tracker   import enrich_with_sentiment_drift
from modules.macro_context       import get_macro_context
from modules.config              import cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

HISTORY_PATH = Path("outputs/history.json")
REPORTS_DIR  = Path("outputs/daily_reports")

# Quick MC Parameter (Vorfilter)
QUICK_MC_PATHS     = 3_000
QUICK_MC_DAYS      = 30
QUICK_MC_MIN_PROB  = 0.45   # < 45% Hit-Rate → kein weiterer Aufwand

# Final MC Parameter (Top-Kandidat)
FINAL_MC_PATHS     = 10_000


def load_history() -> dict:
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH) as f:
            return json.load(f)
    return {
        "feature_stats":  {},
        "active_trades":  [],
        "closed_trades":  [],
        "model_weights":  {"impact": 0.35, "mismatch": 0.45, "eps_drift": 0.20},
        "sentiment_history": {},
    }


def save_history(history: dict) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2, default=str)


def main() -> None:
    log.info("=== Adaptive Asymmetry-Scanner v8.0 gestartet ===")
    today   = datetime.utcnow().strftime("%Y-%m-%d")
    history = load_history()

    stats = {
        "vix": None, "candidates": 0, "prescreened": 0,
        "roi_precheck": 0, "quick_mc": 0, "analyzed": 0,
        "mismatch_ok": 0, "intraday_ok": 0, "final_mc": 0,
        "rl_scored": 0, "roi_ok": 0, "trades": 0,
        "stop_reason": "",
    }

    def send_email(proposals=None):
        try:
            send_status_email(stats, today)
        except Exception as e:
            log.error(f"Email-Fehler: {e}")

    # ── STUFE 0: Risk Gates ──────────────────────────────────────────────────
    gates = RiskGates()
    if not gates.global_ok():
        stats["stop_reason"] = f"VIX-Gate (VIX={gates.last_vix:.1f})"
        stats["vix"] = gates.last_vix
        send_email()
        return
    stats["vix"] = gates.last_vix

    # Makro-Kontext einmal laden
    macro = get_macro_context()
    log.info(f"Makro: {macro.get('macro_regime')} | YC={macro.get('yield_curve_desc', 'n/a')}")

    # ── STUFE 1: Hard-Filter ─────────────────────────────────────────────────
    log.info("Stufe 1: Hard-Filter (Cap>2B, Vol>1M, RV>0.8)")
    candidates = DataIngestion(history=history).run()
    stats["candidates"] = len(candidates)
    if not candidates:
        stats["stop_reason"] = "Keine Kandidaten nach Hard-Filter."
        send_email(); return

    # ── STUFE 1b: FinBERT + Sentiment-Drift ─────────────────────────────────
    log.info("Stufe 1b: FinBERT + Sentiment-Drift")
    enriched = []
    for c in candidates:
        try:
            sentiment = score_candidate(c)
            c.setdefault("features", {}).update(sentiment)
        except Exception:
            c.setdefault("features", {}).update({"sentiment_score": 0.0})
        c = enrich_with_sentiment_drift(c, history)
        enriched.append(c)
    candidates = enriched

    # ── STUFE 2: Prescreening (Haiku) ────────────────────────────────────────
    log.info("Stufe 2: Prescreening (Claude Haiku)")
    shortlist = Prescreener().run(candidates)
    stats["prescreened"] = len(shortlist)
    log.info(f"  → {len(shortlist)} nach Prescreening")
    if not shortlist:
        stats["stop_reason"] = f"Alle {len(candidates)} im Prescreening als 'kein Signal' bewertet."
        send_email(); return

    # Alpha-Sources + EPS-Validierung
    shortlist = [enrich_with_alpha_sources(c) for c in shortlist]
    shortlist = [validate_candidate_data(c) for c in shortlist]

    # ── STUFE 3: Options ROI Pre-Check (NEU: vor Deep Analysis) ─────────────
    log.info("Stufe 3: Options ROI Pre-Check (Fail Fast)")
    roi_viable = []
    sim_stub   = MirofishSimulation()
    for c in shortlist:
        ticker = c["ticker"]
        try:
            _, current, _ = sim_stub._get_market_params(ticker)
            if current <= 0:
                continue
            # Schneller IV-Rank und Kontrakt-Check (Options Designer light)
            designer   = OptionsDesigner(gates=gates)
            iv_rank    = designer._get_iv_rank(ticker)
            direction  = "BULLISH"   # Konservative Annahme für Pre-Check
            strategy   = designer._select_strategy(ticker, direction, iv_rank)
            option     = designer._find_option_for_dte(
                ticker, strategy, current, 21, 45
            )
            if option:
                sim_fake = {"current_price": current, "target_price": current * 1.08, "iv_rank": iv_rank}
                roi      = designer._compute_roi(
                    option, sim_fake, iv_rank,
                    {"label": "Short-Term", "dte_min": 21, "dte_max": 45, "min_roi": 0.15}
                )
                if roi["roi_net"] < -0.30:
                    log.info(f"  [{ticker}] ROI-PRECHECK: {roi['roi_net']:.1%} → hoffnungslos, übersprungen")
                    continue
            roi_viable.append(c)
            log.info(f"  [{ticker}] ROI-PRECHECK: viable")
        except Exception:
            roi_viable.append(c)   # Im Zweifel durchlassen

    stats["roi_precheck"] = len(roi_viable)
    log.info(f"  → {len(roi_viable)} nach ROI Pre-Check")
    if not roi_viable:
        stats["stop_reason"] = "Alle Optionsketten haben hoffnungslosen ROI."
        send_email(); return

    # ── STUFE 4: Quick Monte Carlo (n=3.000, 30d) ────────────────────────────
    log.info(f"Stufe 4: Quick Monte Carlo (n={QUICK_MC_PATHS}, {QUICK_MC_DAYS}d)")
    mc_viable = []
    for c in roi_viable:
        ticker = c["ticker"]
        # Quick MC: n=3000, 30d — reiner Plausibilitäts-Check
        qmc_result = sim_stub.run_for_dte(c, days_to_expiry=QUICK_MC_DAYS)
        if qmc_result is None:
            hit_rate = 0.0
        else:
            hit_rate = qmc_result.get("simulation", {}).get("hit_rate", 0.0)

        if hit_rate < QUICK_MC_MIN_PROB:
            log.info(
                f"  [{ticker}] Quick MC: {hit_rate:.1%} < {QUICK_MC_MIN_PROB:.0%} → verworfen"
            )
            continue

        c["quick_mc"] = {
            "hit_rate":  hit_rate,
            "n_paths":   QUICK_MC_PATHS,
            "n_days":    QUICK_MC_DAYS,
        }
        mc_viable.append(c)

    stats["quick_mc"] = len(mc_viable)
    log.info(f"  → {len(mc_viable)} nach Quick MC")
    if not mc_viable:
        stats["stop_reason"] = f"Alle Kandidaten unter Quick-MC-Schwelle ({QUICK_MC_MIN_PROB:.0%})."
        save_history(history); send_email(); return

    # ── STUFE 5: Deep Analysis (MIT MC-Daten im Prompt) ──────────────────────
    log.info("Stufe 5: Deep Analysis (Claude Sonnet + MC-Kontext)")
    # MC-Ergebnisse in Kandidaten injizieren damit DeepAnalysis sie im Prompt hat
    for c in mc_viable:
        c.setdefault("features", {})["quick_mc_hit_rate"] = (
            c.get("quick_mc", {}).get("hit_rate", 0.0)
        )
    analyses = DeepAnalysis().run(mc_viable)
    stats["analyzed"] = len(analyses)
    log.info(f"  → {len(analyses)} Analysen abgeschlossen")

    # ── STUFE 6: Mismatch-Score + Intraday-Delta ─────────────────────────────
    log.info("Stufe 6: Mismatch-Score + Intraday-Delta")
    scored = MismatchScorer().run(analyses)
    max_move = getattr(getattr(cfg, "pipeline", None), "max_intraday_move", 0.07)
    scored   = filter_by_intraday_delta(scored, max_move=max_move)
    stats["mismatch_ok"]  = len(scored)
    stats["intraday_ok"]  = len(scored)
    log.info(f"  → {len(scored)} nach Mismatch + Intraday")
    if not scored:
        stats["stop_reason"] = "Kein Signal hat Mismatch-Filter bestanden."
        save_history(history); send_email(); return

    # ── STUFE 7: Final Monte Carlo (n=10.000, adaptive DTE) ──────────────────
    log.info(f"Stufe 7: Final Monte Carlo (n={FINAL_MC_PATHS}, adaptive DTE)")
    sim_final   = MirofishSimulation()
    final_sims  = []
    for s in scored:
        ticker = s["ticker"]
        # DTE aus options_designer bestimmen
        designer = OptionsDesigner(gates=gates)
        iv_rank  = designer._get_iv_rank(ticker)
        sim_dte  = 120   # Default Mid-Term
        for tier in designer.DTE_TIERS if hasattr(designer, 'DTE_TIERS') else []:
            if tier.get("dte_min", 0) <= 120 <= tier.get("dte_max", 999):
                sim_dte = 120
                break

        result = sim_final.run_for_dte(s, days_to_expiry=sim_dte)
        if result:
            result["simulation"]["n_paths"] = FINAL_MC_PATHS
            final_sims.append(result)
            log.info(
                f"  [{ticker}] Final MC ({sim_dte}d): "
                f"{result['simulation']['hit_rate']:.1%} ✅"
            )
        else:
            log.info(f"  [{ticker}] Final MC: FAIL")

    stats["final_mc"] = len(final_sims)
    log.info(f"  → {len(final_sims)} nach Final MC")
    if not final_sims:
        stats["stop_reason"] = "Kein Kandidat besteht Final Monte Carlo."
        save_history(history); send_email(); return

    # ── STUFE 8: RL-Scoring ──────────────────────────────────────────────────
    log.info("Stufe 8: RL-Scoring")
    final_signals = RLScorer(history=history).run(final_sims)
    stats["rl_scored"] = len(final_signals)
    log.info(f"  → {len(final_signals)} nach RL-Scoring")
    if not final_signals:
        stats["stop_reason"] = "RL-Agent hat alle Signale als SKIP klassifiziert."
        save_history(history); send_email(); return

    # Premium Signals für Top-2
    final_signals = enrich_top_candidates(final_signals, top_n=2)

    # ── STUFE 9: Options Design + ROI-Gate ───────────────────────────────────
    log.info("Stufe 9: Options Design + adaptiver Laufzeit-Loop")
    designer         = OptionsDesigner(gates=gates)
    trade_proposals  = designer.run(final_signals)

    # TVE-Score berechnen
    for p in trade_proposals:
        roi = p.get("roi_analysis", {})
        dte = p.get("option", {}).get("dte", 90)
        p["time_value_efficiency"] = compute_time_value_efficiency(
            roi.get("roi_net", 0), dte
        )

    stats["roi_ok"]  = len(trade_proposals)
    stats["trades"]  = len(trade_proposals)

    if not trade_proposals:
        stats["stop_reason"] = "Alle Options-Kontrakte scheitern am ROI-Gate."

    # History + Report
    reporter = Reporter(reports_dir=REPORTS_DIR)
    reporter.save(today=today, proposals=trade_proposals, history=history)

    new_trades = []
    existing_keys = {(t["ticker"], t.get("entry_date", "")) for t in history["active_trades"]}
    for p in trade_proposals:
        key = (p["ticker"], today)
        if key not in existing_keys:
            history["active_trades"].append({
                "ticker":       p["ticker"],
                "entry_date":   today,
                "features":     p.get("features", {}),
                "strategy":     p.get("strategy", ""),
                "option":       p.get("option"),
                "simulation":   p.get("simulation"),
                "deep_analysis": p.get("deep_analysis"),
                "tve":          p.get("time_value_efficiency"),
                "outcome":      None,
            })
            new_trades.append(p["ticker"])

    log.info(f"  {len(new_trades)} neue Trade(s) in History: {new_trades}")
    save_history(history)

    # ── EMAIL (immer) ────────────────────────────────────────────────────────
    send_email(trade_proposals if trade_proposals else None)

    log.info(f"=== Pipeline beendet. {len(trade_proposals)} Trade-Vorschläge. ===")


if __name__ == "__main__":
    main()

"""
Adaptive Asymmetry-Scanner v3.5
Hauptpipeline – täglich ausgeführt via GitHub Actions (14:30 MEZ)
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from modules.data_ingestion import DataIngestion
from modules.prescreener import Prescreener
from modules.deep_analysis import DeepAnalysis
from modules.mismatch_scorer import MismatchScorer
from modules.mirofish_simulation import MirofishSimulation
from modules.quasi_ml import QuasiML
from modules.options_designer import OptionsDesigner
from modules.reporter import Reporter
from modules.risk_gates import RiskGates
from modules.email_reporter import send_email

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

HISTORY_PATH = Path("outputs/history.json")
REPORTS_DIR = Path("outputs/daily_reports")


def load_history() -> dict:
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH) as f:
            return json.load(f)
    return {
        "feature_stats": {
            "impact": {
                "low":  {"count": 0, "avg_return": 0.0},
                "mid":  {"count": 0, "avg_return": 0.0},
                "high": {"count": 0, "avg_return": 0.0},
            },
            "mismatch": {
                "weak":   {"count": 0, "avg_return": 0.0},
                "good":   {"count": 0, "avg_return": 0.0},
                "strong": {"count": 0, "avg_return": 0.0},
            },
            "eps_drift": {
                "noise":    {"count": 0, "avg_return": 0.0},
                "relevant": {"count": 0, "avg_return": 0.0},
                "massive":  {"count": 0, "avg_return": 0.0},
            },
        },
        "active_trades": [],
        "closed_trades": [],
        "model_weights": {"impact": 0.35, "mismatch": 0.45, "eps_drift": 0.20},
    }


def save_history(history: dict) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2, default=str)
    log.info("history.json gespeichert.")


def main() -> None:
    log.info("=== Adaptive Asymmetry-Scanner v3.5 gestartet ===")
    today = datetime.utcnow().strftime("%Y-%m-%d")
    history = load_history()

    # ── STUFE 0: Globale Risk-Gates ──────────────────────────────────────────
    gates = RiskGates()
    if not gates.global_ok():
        log.warning("Globales Risk-Gate ausgelöst (VIX > 35 o.ä.). Abbruch.")
        return

    # ── STUFE 1: Daten-Ingestion & Hard-Filter ───────────────────────────────
    log.info("Stufe 1: Daten-Ingestion")
    ingestion = DataIngestion(history=history)
    candidates = ingestion.run()
    log.info(f"  → {len(candidates)} Kandidaten nach Hard-Filter")
    if not candidates:
        log.info("Keine Kandidaten. Pipeline beendet.")
        return

    # ── STUFE 2: Vorselektion via Claude Haiku ───────────────────────────────
    log.info("Stufe 2: Prescreening (Claude Haiku)")
    prescreener = Prescreener()
    shortlist = prescreener.run(candidates)
    log.info(f"  → {len(shortlist)} Ticker nach Prescreening")
    if not shortlist:
        log.info("Shortlist leer. Pipeline beendet.")
        return

    # ── STUFE 3: Deep Analysis via Claude Sonnet ─────────────────────────────
    log.info("Stufe 3: Deep Analysis (Claude Sonnet)")
    analyzer = DeepAnalysis()
    analyses = analyzer.run(shortlist)
    log.info(f"  → {len(analyses)} Analysen abgeschlossen")

    # ── STUFE 4: Mismatch-Score (Quant-Validierung) ──────────────────────────
    log.info("Stufe 4: Mismatch-Score")
    scorer = MismatchScorer()
    scored = scorer.run(analyses)

    # ── STUFE 5: Pfad-Simulation (MiroFish) ──────────────────────────────────
    log.info("Stufe 5: MiroFish Monte-Carlo-Simulation")
    simulator = MirofishSimulation()
    simulated = simulator.run(scored)
    log.info(f"  → {len(simulated)} Ticker passieren die 70%-Schwelle")
    if not simulated:
        log.info("Keine Signale nach Simulation. Pipeline beendet.")
        return

    # ── STUFE 6: Adaptive Quasi-ML Scoring ───────────────────────────────────
    log.info("Stufe 6: Quasi-ML Final-Scoring")
    qml = QuasiML(history=history)
    final_signals = qml.run(simulated)

    # ── STUFE 7: Options-Design ───────────────────────────────────────────────
    log.info("Stufe 7: Options-Design (Tradier)")
    designer = OptionsDesigner(gates=gates)
    trade_proposals = designer.run(final_signals)

    # ── REPORT & HISTORY UPDATE ───────────────────────────────────────────────
    reporter = Reporter(reports_dir=REPORTS_DIR)
    reporter.save(today=today, proposals=trade_proposals, history=history)

    # Neue Trades in history schreiben
    for proposal in trade_proposals:
        history["active_trades"].append({
            "ticker":     proposal["ticker"],
            "entry_date": today,
            "features":   proposal["features"],
            "strategy":   proposal["strategy"],
            "option":     proposal.get("option"),
            "outcome":    None,
        })

    save_history(history)

    # ── EMAIL ─────────────────────────────────────────────────────────────
    send_email(trade_proposals, today)

    log.info(f"=== Pipeline beendet. {len(trade_proposals)} Trade-Vorschläge generiert. ===")


if __name__ == "__main__":
    main()

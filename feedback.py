"""
feedback.py – Adaptive Lern-Loop v4.0

Änderungen gegenüber v3.5:
  - Nach Trade-Close: PPO-Agent wird auf neuem closed_trade nachtrainiert
  - RL-Training: Inkrementelles Update (Continual Learning)
  - Bestehende Fixes M-04, M-05 bleiben erhalten
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yfinance as yf
from scipy import stats

from modules.config import cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

HISTORY_PATH     = Path("outputs/history.json")
MIN_TRADE_AGE_DAYS = 7


# ── History I/O ───────────────────────────────────────────────────────────────

def load_history() -> dict:
    if not HISTORY_PATH.exists():
        log.error("history.json nicht gefunden.")
        sys.exit(1)
    with open(HISTORY_PATH) as f:
        return json.load(f)


def save_history(history: dict) -> None:
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2, default=str)
    log.info("history.json aktualisiert.")


# ── Preis-Abruf ───────────────────────────────────────────────────────────────

def get_current_price(ticker: str) -> float:
    try:
        info = yf.Ticker(ticker).info
        return float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
    except Exception:
        return 0.0


def get_current_option_price(ticker: str, option: dict) -> float:
    if not option:
        return 0.0
    strike = option.get("strike")
    expiry = option.get("expiry")
    if not strike or not expiry:
        return 0.0
    try:
        t = yf.Ticker(ticker)
        if expiry not in t.options:
            return 0.0
        chain   = t.option_chain(expiry)
        matches = chain.calls[
            (chain.calls["strike"] == strike) & (chain.calls["ask"] > 0)
        ]
        if matches.empty:
            return 0.0
        row = matches.iloc[0]
        return float((row["bid"] + row["ask"]) / 2)
    except Exception as e:
        log.debug(f"Options-Preis Fehler für {ticker}: {e}")
        return 0.0


def compute_outcome(trade: dict, current_stock_price: float) -> float:
    ticker      = trade["ticker"]
    option      = trade.get("option", {})
    sim         = trade.get("simulation", {})
    entry_stock = sim.get("current_price", 0)

    stock_return = 0.0
    if entry_stock > 0 and current_stock_price > 0:
        stock_return = (current_stock_price - entry_stock) / entry_stock

    entry_last = option.get("last", 0) if option else 0
    if entry_last > 0:
        current_option = get_current_option_price(ticker, option)
        if current_option > 0:
            options_return = (current_option - entry_last) / entry_last
            log.info(f"    Options-P&L: entry=${entry_last:.2f} → current=${current_option:.2f} = {options_return:+.2%}")
            return options_return

    if entry_last > 0 and entry_stock > 0:
        leverage      = (entry_stock / entry_last) * 0.65
        approx_return = stock_return * leverage
        log.info(f"    Delta-approx: {stock_return:+.2%} × {leverage:.1f} = {approx_return:+.2%}")
        return approx_return

    log.info(f"    Stock-Return Fallback: {stock_return:+.2%}")
    return stock_return


# ── Bin-Updates (Legacy, für Backward-Kompatibilität) ─────────────────────────

def update_bin(stats_dict: dict, feature: str, bin_label: str, outcome: float) -> None:
    bin_data = stats_dict.setdefault(feature, {}).setdefault(
        bin_label, {"count": 0, "avg_return": 0.0}
    )
    old_avg = bin_data["avg_return"]
    old_cnt = bin_data["count"]
    new_cnt = old_cnt + 1
    new_avg = (old_avg * old_cnt + outcome) / new_cnt
    bin_data["count"]      = new_cnt
    bin_data["avg_return"] = round(new_avg, 6)


# ── RL-Training ───────────────────────────────────────────────────────────────

def retrain_rl_agent(history: dict) -> None:
    """
    NEU v4.0: Trainiert den PPO-Agenten inkrementell auf allen closed_trades.

    Wird nach jedem Feedback-Loop-Durchlauf aufgerufen, wenn mindestens
    1 neuer Trade abgeschlossen wurde.

    Strategie: Continual Learning
    - Existierendes Modell wird geladen und weiter-trainiert
    - Nur 2.000 Steps pro Feedback-Lauf (schnell, ~5s auf CPU)
    - Bei jedem neuen closed_trade verbessert sich das Modell graduell

    GitHub-Actions-Tauglichkeit:
    - 2.000 Steps auf CPU ≈ 3–8 Sekunden
    - Modell wird als .zip committed und bei nächstem Run weiterverwendet
    """
    try:
        from modules.rl_agent import train_agent
    except ImportError as e:
        log.warning(f"RL-Agent nicht importierbar: {e} → Training übersprungen")
        return

    closed = history.get("closed_trades", [])
    if len(closed) < 5:
        log.info(
            f"Nur {len(closed)} closed_trades → RL-Training übersprungen "
            f"(Minimum: 5, damit PPO stabile Gradienten berechnen kann)."
        )
        return

    log.info(f"Starte RL-Nachtraining auf {len(closed)} closed_trades...")

    # Inkrementelles Training: 2.000 Steps pro Feedback-Lauf
    # Nach 50 Trades ≈ 100.000 Gesamt-Steps → gut konvergiertes Modell
    success = train_agent(
        history         = history,
        total_timesteps = 2_000,
        force_retrain   = False,   # Weiter-Training statt Neu-Training
    )

    if success:
        log.info("RL-Agent erfolgreich nachtrainiert.")
    else:
        log.warning("RL-Nachtraining fehlgeschlagen (nicht kritisch).")


# ── Pearson-Gewichte (Legacy-Support) ────────────────────────────────────────

def compute_pearson_weights(history: dict) -> dict:
    closed = history.get("closed_trades", [])
    if len(closed) < 5:
        return history.get("model_weights", {"impact": 0.35, "mismatch": 0.45, "eps_drift": 0.20})

    outcomes, impacts, mismatches, drifts = [], [], [], []
    for t in closed:
        outcome = t.get("outcome")
        if outcome is None:
            continue
        feat = t.get("features", {})
        outcomes.append(outcome)
        impacts.append(_bin_to_num("impact",    feat.get("bin_impact",    "mid")))
        mismatches.append(_bin_to_num("mismatch", feat.get("bin_mismatch",  "good")))
        drifts.append(_bin_to_num("eps_drift", feat.get("bin_eps_drift", "noise")))

    if len(outcomes) < 5:
        return history.get("model_weights", {})

    outcomes_arr = np.array(outcomes)
    correlations = {}
    for name, arr in [("impact", np.array(impacts)),
                       ("mismatch", np.array(mismatches)),
                       ("eps_drift", np.array(drifts))]:
        r, _ = stats.pearsonr(arr, outcomes_arr)
        correlations[name] = max(r, 0)

    total = sum(correlations.values()) or 1.0
    old_w = history.get("model_weights", {})
    new_w = {}
    for feat, corr in correlations.items():
        raw_new     = corr / total
        old         = old_w.get(feat, 1/3)
        new_w[feat] = round(old + cfg.learning.learning_rate * (raw_new - old), 4)

    total_w = sum(new_w.values())
    return {k: round(v / total_w, 4) for k, v in new_w.items()}


def _bin_to_num(feature: str, bin_label: str) -> float:
    mapping = {
        "impact":    {"low": 0.0, "mid": 0.5, "high": 1.0},
        "mismatch":  {"weak": 0.0, "good": 0.5, "strong": 1.0},
        "eps_drift": {"noise": 0.0, "relevant": 0.5, "massive": 1.0},
    }
    return mapping.get(feature, {}).get(bin_label, 0.5)


# ── Haupt-Loop ────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=== Feedback-Loop v4.0 gestartet ===")
    history      = load_history()
    today        = datetime.utcnow()
    active       = history.get("active_trades", [])
    still_active = []
    newly_closed = 0

    for trade in active:
        ticker     = trade["ticker"]
        entry_date = datetime.strptime(trade["entry_date"][:10], "%Y-%m-%d")
        age_days   = (today - entry_date).days

        if age_days < MIN_TRADE_AGE_DAYS:
            log.info(f"  [{ticker}] Alter={age_days}d < {MIN_TRADE_AGE_DAYS} → zu jung.")
            still_active.append(trade)
            continue

        current = get_current_price(ticker)
        if current <= 0:
            still_active.append(trade)
            continue

        outcome = compute_outcome(trade, current)
        log.info(f"  [{ticker}] Alter={age_days}d Outcome={outcome:+.2%}")

        # Legacy Bin-Updates (für Backward-Kompatibilität mit QuasiML)
        feat = trade.get("features", {})
        for f_name, bin_key in [("impact", "bin_impact"),
                                  ("mismatch", "bin_mismatch"),
                                  ("eps_drift", "bin_eps_drift")]:
            bin_label = feat.get(bin_key)
            if bin_label:
                update_bin(history["feature_stats"], f_name, bin_label, outcome)

        if age_days >= cfg.learning.close_after_days:
            trade["outcome"]     = round(outcome, 4)
            trade["close_date"]  = today.strftime("%Y-%m-%d")
            trade["close_price"] = current
            history.setdefault("closed_trades", []).append(trade)
            log.info(f"  [{ticker}] Trade abgeschlossen (Return={outcome:+.2%})")
            newly_closed += 1
        else:
            trade["last_price"]    = current
            trade["current_return"] = round(outcome, 4)
            still_active.append(trade)

    history["active_trades"] = still_active
    history["model_weights"] = compute_pearson_weights(history)

    save_history(history)

    # NEU v4.0: RL-Nachtraining wenn neue Trades abgeschlossen wurden
    if newly_closed > 0:
        log.info(f"{newly_closed} neue closed_trades → starte RL-Nachtraining...")
        retrain_rl_agent(history)
    else:
        log.info("Keine neuen closed_trades → RL-Training übersprungen.")

    log.info("=== Feedback-Loop abgeschlossen ===")


if __name__ == "__main__":
    main()

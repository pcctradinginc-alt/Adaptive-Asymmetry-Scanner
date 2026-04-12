"""
feedback.py – Adaptive Lern-Loop
Läuft wöchentlich (oder täglich vor dem Scan).

Aufgaben:
1. Performance-Audit: entry_price vs. current_price aller Trades
2. Bin-Update: Aktualisiert avg_return pro Feature-Bin (laufender Durchschnitt)
3. Gewichts-Anpassung: Pearson-Korrelation → erfolgreiche Features werden stärker gewichtet
4. Handelt abgeschlossene Trades ab (> 120 Tage alt)
"""

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import yfinance as yf
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

HISTORY_PATH = Path("outputs/history.json")
LEARNING_RATE = 0.05
CLOSE_AFTER_DAYS = 130   # Trade gilt als abgeschlossen


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


def get_current_price(ticker: str) -> float:
    try:
        info = yf.Ticker(ticker).info
        return float(
            info.get("currentPrice") or info.get("regularMarketPrice") or 0
        )
    except Exception:
        return 0.0


def update_bin(stats_dict: dict, feature: str, bin_label: str, outcome: float) -> None:
    """Aktualisiert laufenden Durchschnitt für einen Bin."""
    bin_data = stats_dict.setdefault(feature, {}).setdefault(
        bin_label, {"count": 0, "avg_return": 0.0}
    )
    old_avg = bin_data["avg_return"]
    old_cnt = bin_data["count"]
    new_cnt = old_cnt + 1
    new_avg = (old_avg * old_cnt + outcome) / new_cnt
    bin_data["count"]      = new_cnt
    bin_data["avg_return"] = round(new_avg, 6)


def compute_pearson_weights(history: dict) -> dict:
    """
    Berechnet Pearson-Korrelation zwischen Feature-Bins und Outcomes.
    Passt Gewichtungen an erfolgreiche Features an.
    """
    closed = history.get("closed_trades", [])
    if len(closed) < 5:
        log.info("Zu wenig abgeschlossene Trades für Gewichts-Update.")
        return history["model_weights"]

    outcomes  = []
    impacts   = []
    mismatches = []
    drifts    = []

    for t in closed:
        outcome = t.get("outcome")
        if outcome is None:
            continue
        feat = t.get("features", {})
        outcomes.append(outcome)
        # Numerische Werte aus Bins
        impacts.append(_bin_to_num("impact",    feat.get("bin_impact",    "mid")))
        mismatches.append(_bin_to_num("mismatch", feat.get("bin_mismatch",  "good")))
        drifts.append(_bin_to_num("eps_drift", feat.get("bin_eps_drift", "noise")))

    if len(outcomes) < 5:
        return history["model_weights"]

    outcomes_arr  = np.array(outcomes)
    correlations = {}
    for name, arr in [
        ("impact",    np.array(impacts)),
        ("mismatch",  np.array(mismatches)),
        ("eps_drift", np.array(drifts)),
    ]:
        r, _ = stats.pearsonr(arr, outcomes_arr)
        correlations[name] = max(r, 0)   # Negative Korrelation → 0

    total = sum(correlations.values()) or 1.0
    old_w = history["model_weights"]
    new_w = {}
    for feat, corr in correlations.items():
        raw_new = corr / total
        old     = old_w.get(feat, 1/3)
        new_w[feat] = round(old + LEARNING_RATE * (raw_new - old), 4)

    # Normalisieren
    total_w = sum(new_w.values())
    new_w = {k: round(v / total_w, 4) for k, v in new_w.items()}

    log.info(f"Neue Gewichte: {new_w} (alt: {old_w})")
    return new_w


def _bin_to_num(feature: str, bin_label: str) -> float:
    mapping = {
        "impact":    {"low": 0.0, "mid": 0.5, "high": 1.0},
        "mismatch":  {"weak": 0.0, "good": 0.5, "strong": 1.0},
        "eps_drift": {"noise": 0.0, "relevant": 0.5, "massive": 1.0},
    }
    return mapping.get(feature, {}).get(bin_label, 0.5)


def main() -> None:
    log.info("=== Feedback-Loop gestartet ===")
    history = load_history()
    today   = datetime.utcnow()

    active      = history.get("active_trades", [])
    still_active = []

    for trade in active:
        ticker     = trade["ticker"]
        entry_date = datetime.strptime(trade["entry_date"][:10], "%Y-%m-%d")
        age_days   = (today - entry_date).days

        current = get_current_price(ticker)
        if current <= 0:
            still_active.append(trade)
            continue

        # Option-Preis als Proxy für Entry-Preis (vereinfacht: Stock-Return)
        entry_option = trade.get("option", {})
        entry_last   = entry_option.get("last", 0) if entry_option else 0

        # Stock-Return als Outcome
        option_proxy = trade.get("simulation", {}).get("current_price", 0)
        if option_proxy > 0 and current > 0:
            stock_return = (current - option_proxy) / option_proxy
        else:
            stock_return = 0.0

        log.info(
            f"  [{ticker}] Alter={age_days}d "
            f"Entry≈${option_proxy:.2f} → Aktuell=${current:.2f} "
            f"Return={stock_return:+.2%}"
        )

        # Bin-Updates
        feat = trade.get("features", {})
        for f_name, bin_key in [
            ("impact",    "bin_impact"),
            ("mismatch",  "bin_mismatch"),
            ("eps_drift", "bin_eps_drift"),
        ]:
            bin_label = feat.get(bin_key)
            if bin_label:
                update_bin(
                    history["feature_stats"], f_name, bin_label, stock_return
                )

        if age_days >= CLOSE_AFTER_DAYS:
            trade["outcome"]       = round(stock_return, 4)
            trade["close_date"]    = today.strftime("%Y-%m-%d")
            trade["close_price"]   = current
            history.setdefault("closed_trades", []).append(trade)
            log.info(f"  [{ticker}] Trade abgeschlossen (Return={stock_return:+.2%})")
        else:
            trade["last_price"]    = current
            trade["current_return"] = round(stock_return, 4)
            still_active.append(trade)

    history["active_trades"]  = still_active

    # Gewichts-Update
    history["model_weights"] = compute_pearson_weights(history)

    save_history(history)
    log.info("=== Feedback-Loop abgeschlossen ===")


if __name__ == "__main__":
    main()

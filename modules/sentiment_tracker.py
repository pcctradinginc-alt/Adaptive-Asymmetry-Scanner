"""
modules/sentiment_tracker.py – Sentiment-Drift über Zeit

Fix 4: Speichert täglich FinBERT-Scores pro Ticker in history.json.
       Nach 4 Wochen nutzbar für Akkumulations-Erkennung.

Akkumulationsphase-Erkennung:
    Wenn Sentiment-Trend über 4 Wochen steigt (+0.3) bei stagnierendem
    Preis (<3% Move) → klassische Akkumulationsphase → hochwertiges Signal.

Datenstruktur in history.json:
    "sentiment_history": {
        "AAPL": [
            {"date": "2026-04-14", "score": 0.42, "headline_count": 5},
            {"date": "2026-04-15", "score": 0.38, "headline_count": 3},
            ...
        ]
    }

Hinweis: Ab heute sammeln, nach 20+ Einträgen (4 Wochen) auswertbar.
"""

from __future__ import annotations
import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# Mindestanzahl Einträge für Drift-Analyse
MIN_ENTRIES_FOR_DRIFT  = 10   # ~2 Wochen
FULL_DRIFT_ENTRIES     = 20   # ~4 Wochen

# Akkumulations-Erkennung: Score-Anstieg über Periode
ACCUMULATION_THRESHOLD = 0.15  # +0.20 Sentiment-Anstieg über 4 Wochen
PRICE_STAGNATION_MAX   = 0.05  # Preis darf sich max. 5% bewegt haben


def update_sentiment_history(
    history: dict,
    ticker:  str,
    score:   float,
    headline_count: int = 0,
    today:   str = "",
) -> None:
    """
    Speichert den heutigen Sentiment-Score für einen Ticker.
    Wird täglich in pipeline.py nach dem FinBERT-Enrichment aufgerufen.

    Args:
        history:        history.json als Dict
        ticker:         Aktien-Ticker
        score:          FinBERT Sentiment-Score (-1.0 bis +1.0)
        headline_count: Anzahl Headlines die bewertet wurden
        today:          Datum (default: heute UTC)
    """
    if not today:
        today = datetime.utcnow().strftime("%Y-%m-%d")

    sent_history = history.setdefault("sentiment_history", {})
    ticker_hist  = sent_history.setdefault(ticker, [])

    # Duplikat-Check: Heute bereits eingetragen?
    if ticker_hist and ticker_hist[-1].get("date") == today:
        log.debug(f"  [{ticker}] Sentiment bereits für {today} gespeichert → überschreiben")
        ticker_hist[-1] = {
            "date":           today,
            "score":          round(score, 4),
            "headline_count": headline_count,
        }
    else:
        ticker_hist.append({
            "date":           today,
            "score":          round(score, 4),
            "headline_count": headline_count,
        })

    # Nur letzten 60 Einträge behalten (3 Monate)
    if len(ticker_hist) > 60:
        sent_history[ticker] = ticker_hist[-60:]

    log.debug(
        f"  [{ticker}] Sentiment gespeichert: {score:.3f} "
        f"({headline_count} Headlines, {len(ticker_hist)} Einträge total)"
    )


def get_sentiment_drift(
    history: dict,
    ticker:  str,
    days:    int = 28,
) -> dict:
    """
    Berechnet den Sentiment-Drift über die letzten N Tage.

    Returns:
        {
            "drift":              float,   # Veränderung des Sentiment-Scores
            "trend":              str,     # "rising" | "falling" | "flat"
            "entries_available":  int,
            "enough_data":        bool,
            "mean_score":         float,
            "recent_score":       float,   # Letzte 5 Tage
            "old_score":          float,   # Erste 5 Tage der Periode
            "accumulation_signal": bool,   # True wenn Akkumulationsphase erkannt
        }
    """
    sent_history = history.get("sentiment_history", {})
    ticker_hist  = sent_history.get(ticker, [])

    if len(ticker_hist) < MIN_ENTRIES_FOR_DRIFT:
        return {
            "drift":               0.0,
            "trend":               "insufficient_data",
            "entries_available":   len(ticker_hist),
            "enough_data":         False,
            "mean_score":          0.0,
            "recent_score":        0.0,
            "old_score":           0.0,
            "accumulation_signal": False,
            "days_until_ready":    MIN_ENTRIES_FOR_DRIFT - len(ticker_hist),
        }

    # Nur Einträge der letzten N Tage
    cutoff   = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    relevant = [e for e in ticker_hist if e.get("date", "") >= cutoff]

    if len(relevant) < MIN_ENTRIES_FOR_DRIFT:
        relevant = ticker_hist[-MIN_ENTRIES_FOR_DRIFT:]

    scores = [e["score"] for e in relevant]

    # Drift: Differenz zwischen letzten 5 und ersten 5 Einträgen
    n         = len(scores)
    old_score = float(np.mean(scores[:min(5, n//2)]))
    new_score = float(np.mean(scores[-min(5, n//2):]))
    drift     = new_score - old_score
    mean      = float(np.mean(scores))

    # Trend
    if drift > 0.10:
        trend = "rising"
    elif drift < -0.10:
        trend = "falling"
    else:
        trend = "flat"

    # Akkumulations-Signal: Steigendes Sentiment + kein starker Kursanstieg
    # (Preis-Check erfolgt in pipeline.py mit yfinance)
    accumulation = (
        drift >= ACCUMULATION_THRESHOLD and
        len(relevant) >= FULL_DRIFT_ENTRIES
    )

    result = {
        "drift":               round(drift, 4),
        "trend":               trend,
        "entries_available":   len(relevant),
        "enough_data":         len(relevant) >= MIN_ENTRIES_FOR_DRIFT,
        "mean_score":          round(mean, 4),
        "recent_score":        round(new_score, 4),
        "old_score":           round(old_score, 4),
        "accumulation_signal": accumulation,
    }

    if accumulation:
        log.info(
            f"  [{ticker}] AKKUMULATIONS-SIGNAL: Sentiment +{drift:.2f} "
            f"über {len(relevant)} Tage → mögliche Insider-Akkumulation"
        )

    return result


def enrich_with_sentiment_drift(
    candidate: dict,
    history:   dict,
) -> dict:
    """
    Reichert einen Kandidaten mit Sentiment-Drift-Daten an.
    Wird nach FinBERT-Enrichment in pipeline.py aufgerufen.

    Speichert heutigen Score UND analysiert historischen Drift.
    """
    ticker        = candidate.get("ticker", "")
    features      = candidate.get("features", {})
    today_score   = features.get("sentiment_score", 0.0)
    headline_cnt  = len(candidate.get("news", []))

    # Heutigen Score speichern
    update_sentiment_history(
        history        = history,
        ticker         = ticker,
        score          = today_score,
        headline_count = headline_cnt,
    )

    # Historischen Drift berechnen
    drift_data = get_sentiment_drift(history, ticker)
    candidate["sentiment_drift"] = drift_data

    if drift_data["enough_data"]:
        log.info(
            f"  [{ticker}] Sentiment-Drift ({drift_data['entries_available']}d): "
            f"trend={drift_data['trend']} drift={drift_data['drift']:+.3f}"
        )

        # Drift in Features aufnehmen (für RL-Agent Observation-Vektor)
        features["sentiment_drift"]  = drift_data["drift"]
        features["sentiment_trend"]  = 1.0 if drift_data["trend"] == "rising" else (
            -1.0 if drift_data["trend"] == "falling" else 0.0
        )
    else:
        days_left = drift_data.get("days_until_ready", MIN_ENTRIES_FOR_DRIFT)
        log.debug(
            f"  [{ticker}] Sentiment-Drift: noch {days_left} Tage bis auswertbar"
        )
        features["sentiment_drift"] = 0.0
        features["sentiment_trend"] = 0.0

    return candidate


def get_accumulation_candidates(history: dict) -> list[str]:
    """
    Gibt alle Ticker zurück die aktuell ein Akkumulations-Signal zeigen.
    Kann für täglichen Report oder Email-Hinweis genutzt werden.
    """
    accumulating = []
    for ticker in history.get("sentiment_history", {}):
        drift = get_sentiment_drift(history, ticker)
        if drift.get("accumulation_signal"):
            accumulating.append(ticker)
    return accumulating

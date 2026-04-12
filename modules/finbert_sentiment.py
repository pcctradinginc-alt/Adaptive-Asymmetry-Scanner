"""
modules/finbert_sentiment.py – FinBERT-Sentiment als Feature-Spalte

Quelle: ProsusAI/finbert (Standard-FinBERT, kein Fine-Tuning nötig)
        Alternativ: hab5510/finrl_project nutzt dasselbe Modell.

Output pro Ticker:
    {
        "sentiment_score": float,   # -1.0 (bearish) bis +1.0 (bullish)
        "sentiment_label": str,     # "positive" | "negative" | "neutral"
        "sentiment_confidence": float  # 0.0–1.0
    }

Design-Entscheidungen:
  - Lazy-Loading: Modell wird nur beim ersten Aufruf geladen (kein Cold-Start
    bei Import → GitHub-Actions-tauglich).
  - Batch-Inference: Alle Headlines eines Tickers in einem Forward-Pass.
  - CPU-only: Kein GPU nötig (GitHub-Actions free tier hat keine GPU).
    FinBERT-Inference auf CPU dauert ~0.1s pro Batch → akzeptabel für
    ~20 Ticker täglich.
  - Fallback: Gibt 0.0 zurück wenn Modell nicht ladbar (kein API-Key nötig,
    aber Netzwerkzugang für einmaliges Download nötig).
  - Cache: Modell wird in outputs/models/finbert/ gespeichert um
    Neudownload bei jedem GitHub-Actions-Run zu vermeiden.
"""

from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# Lokaler Cache-Pfad für GitHub-Actions
_CACHE_DIR = Path("outputs/models/finbert")
_MODEL_NAME = "ProsusAI/finbert"

# Lazy-Loaded Globals
_tokenizer = None
_model     = None


def _load_model():
    """Lazy-Load FinBERT einmalig. Thread-safe für Single-Process-Nutzung."""
    global _tokenizer, _model

    if _tokenizer is not None:
        return True

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        log.info(f"Lade FinBERT von '{_MODEL_NAME}' (Cache: {_CACHE_DIR})...")
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)

        _tokenizer = AutoTokenizer.from_pretrained(
            _MODEL_NAME,
            cache_dir=str(_CACHE_DIR),
        )
        _model = AutoModelForSequenceClassification.from_pretrained(
            _MODEL_NAME,
            cache_dir=str(_CACHE_DIR),
        )
        _model.eval()   # Inference-Modus, kein Dropout
        log.info("FinBERT erfolgreich geladen.")
        return True

    except Exception as e:
        log.warning(f"FinBERT konnte nicht geladen werden: {e} → Fallback 0.0")
        return False


def score_headlines(headlines: list[str]) -> dict:
    """
    Berechnet FinBERT-Sentiment für eine Liste von Headlines.

    Args:
        headlines: Liste von News-Titeln (max. 10 werden genutzt)

    Returns:
        Dict mit sentiment_score, sentiment_label, sentiment_confidence

    Beispiel:
        >>> score_headlines(["Apple beats earnings by 15%", "Market selloff"])
        {"sentiment_score": 0.42, "sentiment_label": "positive",
         "sentiment_confidence": 0.71}
    """
    if not headlines:
        return _neutral_result()

    if not _load_model():
        return _neutral_result()

    try:
        import torch

        # Max. 8 Headlines, max. 128 Tokens pro Headline (CPU-Performance)
        texts = [h[:256] for h in headlines[:8]]

        inputs = _tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = _model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).numpy()

        # FinBERT Label-Reihenfolge: [positive, negative, neutral]
        # (aus ProsusAI/finbert config.json)
        pos_probs  = probs[:, 0]   # positive
        neg_probs  = probs[:, 1]   # negative
        neut_probs = probs[:, 2]   # neutral

        # Gewichteter Score: +1 für positiv, -1 für negativ, 0 für neutral
        # Mittelwert über alle Headlines
        scores = pos_probs - neg_probs   # range: -1 to +1
        mean_score = float(np.mean(scores))

        # Dominante Klasse (Mehrheitsvotum über Headlines)
        mean_probs = np.mean(probs, axis=0)
        label_idx  = int(np.argmax(mean_probs))
        labels     = ["positive", "negative", "neutral"]
        label      = labels[label_idx]
        confidence = float(mean_probs[label_idx])

        log.debug(
            f"FinBERT: score={mean_score:.3f} "
            f"label={label} conf={confidence:.3f} "
            f"({len(texts)} headlines)"
        )

        return {
            "sentiment_score":      round(mean_score, 4),
            "sentiment_label":      label,
            "sentiment_confidence": round(confidence, 4),
        }

    except Exception as e:
        log.warning(f"FinBERT Inference-Fehler: {e}")
        return _neutral_result()


def _neutral_result() -> dict:
    return {
        "sentiment_score":      0.0,
        "sentiment_label":      "neutral",
        "sentiment_confidence": 0.0,
    }


def score_candidate(candidate: dict) -> dict:
    """
    Convenience-Wrapper: Nimmt einen Pipeline-Kandidaten-Dict und
    fügt FinBERT-Features direkt in candidate["features"] ein.

    Wird von data_ingestion.py aufgerufen, BEVOR der Kandidat
    in die Pipeline geht.
    """
    headlines = candidate.get("news", [])
    sentiment = score_headlines(headlines)
    return sentiment

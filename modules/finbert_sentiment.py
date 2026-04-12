"""
modules/finbert_sentiment.py – FinBERT-Sentiment als Feature-Spalte

FIX: Cache-Verzeichnis auf /tmp statt outputs/models/finbert/
     → verhindert dass das 417 MB Modell in Git landet.
     /tmp wird bei jedem GitHub-Actions-Run neu befüllt (Download ~30s).
"""

from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

_MODEL_NAME = "ProsusAI/finbert"

# FIX: /tmp statt outputs/models/finbert/
# /tmp ist in GitHub Actions verfügbar und wird NICHT committed
_CACHE_DIR = Path(os.environ.get("FINBERT_CACHE", "/tmp/finbert_cache"))

# Lazy-Loaded Globals
_tokenizer = None
_model     = None


def _load_model():
    """Lazy-Load FinBERT einmalig."""
    global _tokenizer, _model

    if _tokenizer is not None:
        return True

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        log.info(f"Lade FinBERT '{_MODEL_NAME}' (Cache: {_CACHE_DIR})...")
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)

        _tokenizer = AutoTokenizer.from_pretrained(
            _MODEL_NAME, cache_dir=str(_CACHE_DIR),
        )
        _model = AutoModelForSequenceClassification.from_pretrained(
            _MODEL_NAME, cache_dir=str(_CACHE_DIR),
        )
        _model.eval()
        log.info("FinBERT erfolgreich geladen.")
        return True

    except Exception as e:
        log.warning(f"FinBERT nicht ladbar: {e} → Fallback 0.0")
        return False


def score_headlines(headlines: list[str]) -> dict:
    """
    Berechnet FinBERT-Sentiment für eine Liste von Headlines.
    Gibt neutral (0.0) zurück wenn Modell nicht verfügbar.
    """
    if not headlines:
        return _neutral_result()

    if not _load_model():
        return _neutral_result()

    try:
        import torch

        texts = [h[:256] for h in headlines[:8]]
        inputs = _tokenizer(
            texts, padding=True, truncation=True,
            max_length=128, return_tensors="pt",
        )

        with torch.no_grad():
            outputs = _model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).numpy()

        # FinBERT Labels: [positive, negative, neutral]
        pos_probs  = probs[:, 0]
        neg_probs  = probs[:, 1]
        scores     = pos_probs - neg_probs
        mean_score = float(np.mean(scores))

        mean_probs = np.mean(probs, axis=0)
        label_idx  = int(np.argmax(mean_probs))
        labels     = ["positive", "negative", "neutral"]
        label      = labels[label_idx]
        confidence = float(mean_probs[label_idx])

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
    return score_headlines(candidate.get("news", []))

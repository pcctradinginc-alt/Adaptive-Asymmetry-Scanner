"""
tests/test_feature_stats_external.py – feedback.py: history["feature_stats_external"]

Prüft:
  - Update erfolgt AUSSCHLIESSLICH aus trade["external_context_entry"] (frozen).
  - Kein external_context_entry / kein Outcome → No-op, kein Fehler.
  - feature_stats_external fließt NIE in compute_pearson_weights()/model_weights.
  - Bucket-Extraktion (external_feature_buckets) ist tolerant gegenüber
    fehlenden Feldern.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import feedback


def _ext(**overrides) -> dict:
    base = {
        "states": {"global_freight_state": "EXPANSION", "global_maritime_state": "NEUTRAL"},
        "primitives": {
            "shipping_negative_breadth": 0.2,
            "road_shipping_divergence_z": 0.8,
            "road_shipping_agreement": "agreement",
            "weather_disruption_index": 0.3,
            "active_tropical_system": False,
        },
        "relation": {"relation": "SUPPORT", "materiality": 0.6},
    }
    base.update(overrides)
    return base


def test_external_feature_buckets_extraction():
    buckets = feedback.external_feature_buckets(_ext())
    assert buckets["freight_state"] == "EXPANSION"
    assert buckets["maritime_state"] == "NEUTRAL"
    assert buckets["external_relation"] == "SUPPORT"
    assert buckets["road_shipping_agreement"] == "agreement"
    assert buckets["shipping_breadth_bucket"] == "<0.33"
    assert buckets["divergence_bucket"] == "medium"
    assert buckets["weather_operational_risk"] == "normal"


def test_external_feature_buckets_tolerant_to_missing_fields():
    assert feedback.external_feature_buckets({}) == {
        "freight_state": None, "maritime_state": None,
        "weather_operational_risk": None, "external_relation": None,
        "road_shipping_agreement": None, "shipping_breadth_bucket": None,
        "divergence_bucket": None,
    }
    # None statt dict -> auch tolerant
    assert feedback.external_feature_buckets(None)["freight_state"] is None


def test_update_feature_stats_external_noop_without_entry():
    history = {}
    trade = {"outcome": 0.5}
    feedback.update_feature_stats_external(history, trade)
    assert "feature_stats_external" not in history


def test_update_feature_stats_external_noop_without_outcome():
    history = {}
    trade = {"external_context_entry": _ext()}
    feedback.update_feature_stats_external(history, trade)
    assert "feature_stats_external" not in history


def test_update_feature_stats_external_uses_frozen_entry_only():
    history = {}
    trade = {"outcome": 0.30, "external_context_entry": _ext()}
    feedback.update_feature_stats_external(history, trade)

    stats = history["feature_stats_external"]
    assert stats["freight_state"]["EXPANSION"]["count"] == 1
    assert stats["freight_state"]["EXPANSION"]["mean"] == 0.30
    assert stats["freight_state"]["EXPANSION"]["wins"] == 1
    assert stats["external_relation"]["SUPPORT"]["count"] == 1
    assert stats["shipping_breadth_bucket"]["<0.33"]["count"] == 1

    # Zweiter Trade, andere Werte -> running mean/win_rate korrekt
    trade2 = {"outcome": -0.50, "external_context_entry": _ext(
        states={"global_freight_state": "EXPANSION"},
    )}
    feedback.update_feature_stats_external(history, trade2)
    bucket = history["feature_stats_external"]["freight_state"]["EXPANSION"]
    assert bucket["count"] == 2
    assert bucket["wins"] == 1
    assert abs(bucket["mean"] - (0.30 - 0.50) / 2) < 1e-9


def test_update_feature_stats_external_never_recomputes_from_live_context():
    """Ein manipulierter (aber nicht 'gefrorener') externer Kontext auf dem
    Trade-Objekt selbst darf NICHT verwendet werden — nur external_context_entry."""
    history = {}
    trade = {
        "outcome": 0.1,
        "external_context_entry": _ext(),
        "external": _ext(states={"global_freight_state": "STRONG_CONTRACTION"}),  # "live" — ignorieren
    }
    feedback.update_feature_stats_external(history, trade)
    assert "STRONG_CONTRACTION" not in history["feature_stats_external"]["freight_state"]
    assert "EXPANSION" in history["feature_stats_external"]["freight_state"]


def test_feature_stats_external_never_feeds_pearson_weights():
    """HARTE GARANTIE: feature_stats_external darf compute_pearson_weights()
    nicht beeinflussen — Kontrollgruppe mit/ohne extremen Werten liefert
    identische model_weights."""
    closed_trades = [
        {
            "outcome": 0.4,
            "features": {"bin_impact": "high", "bin_mismatch": "strong", "bin_eps_drift": "massive"},
        }
        for _ in range(3)
    ] + [
        {
            "outcome": -0.3,
            "features": {"bin_impact": "low", "bin_mismatch": "weak", "bin_eps_drift": "noise"},
        }
        for _ in range(3)
    ]

    history_without = {"closed_trades": closed_trades}
    history_with = {
        "closed_trades": closed_trades,
        "feature_stats_external": {
            "freight_state": {
                "STRONG_EXPANSION": {
                    "count": 500, "mean": 5.0, "_m2": 0.0, "wins": 500,
                    "outcomes": [5.0] * 200, "strat_ret_sum": 2500.0, "strat_ret_count": 500,
                }
            }
        },
    }

    w_without = feedback.compute_pearson_weights(history_without)
    w_with = feedback.compute_pearson_weights(history_with)
    assert w_without == w_with


def test_outcomes_cap_streaming_approximation():
    history = {}
    for i in range(feedback.EXTERNAL_BUCKET_OUTCOMES_CAP + 50):
        outcome = 0.01 * i
        trade = {"outcome": outcome, "external_context_entry": _ext()}
        feedback.update_feature_stats_external(history, trade)
    bucket = history["feature_stats_external"]["freight_state"]["EXPANSION"]
    # count bleibt exakt über ALLE Updates (Welford), outcomes-Liste ist gecappt
    assert bucket["count"] == feedback.EXTERNAL_BUCKET_OUTCOMES_CAP + 50
    assert len(bucket["outcomes"]) == feedback.EXTERNAL_BUCKET_OUTCOMES_CAP

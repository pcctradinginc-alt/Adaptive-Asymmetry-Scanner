"""news_novelty als Zahl – nur in einer neuen, versionierten Variante (Audit 2026-11-06, IMP-4a0d2d31).

Live: Das Ledger speichert news_novelty als Objekt {novelty_max, novelty_mean, n_today, n_history, novelty_status};
option_learner.features() liest nur Zahlen -> 0 % Abdeckung, das Merkmal wird verworfen. Die Basis-Experimentversion
und ihre Prognosen bleiben unverändert (laufende Forward-Pläne). Die Zahl novelty_max steht ausschließlich der
freigegebenen Variante news_novelty_field=novelty_max zur Verfügung (eigene Experimentversion, eigener Test-Slot).
"""
import numpy as np
import pytest

from modules import option_learner as ol
from modules import research_director as rd

BASE_NAMES = [*ol.LEDGER_FEATURES, *ol.CONTRACT_FEATURES]
NOV = {"novelty_max": 0.6, "novelty_mean": 0.4, "n_today": 5, "n_history": 40,
       "novelty_version": "news-novelty-v1", "novelty_status": "ok"}


@pytest.mark.parametrize("raw, field, expected", [
    (NOV, "novelty_max", 0.6),
    (NOV, "novelty_mean", 0.4),
    ({**NOV, "novelty_status": "no_history"}, "novelty_max", None),   # nicht berechnet -> fehlt, nie 0
    ({**NOV, "novelty_max": None}, "novelty_max", None),
    ({**NOV, "novelty_max": 1.7}, "novelty_max", None),               # außerhalb [0, 1]: ungültig
    ({**NOV, "novelty_max": float("nan")}, "novelty_max", None),
    (0.6, "novelty_max", None),                                       # kein Objekt
    (None, "novelty_max", None),
    (NOV, "n_today", None),                                           # kein Neuheitsmaß
])
def test_novelty_value(raw, field, expected):
    assert ol.novelty_value(raw, field) == expected


def test_base_experiment_and_feature_set_unchanged():
    spec = ol.experiment()
    assert "news_novelty_field" not in spec["features"]
    assert ol.feature_names() == BASE_NAMES
    assert ol.experiment_id() != ol.experiment_id({"news_novelty_field": "novelty_max"})
    assert ol.experiment_id({"alpha": 30.0}) == ol.experiment_id({"alpha": 30.0})


def test_variant_replaces_the_empty_feature_only():
    names = ol.feature_names({"news_novelty_field": "novelty_max"})
    assert "news_novelty" not in names and ol.novelty_feature("novelty_max") in names
    assert len(names) == len(BASE_NAMES)


def test_features_expose_number_without_touching_base_keys():
    x = ol.features({"features": {"news_novelty": NOV, "rel_volume": 1.2}})
    assert "news_novelty" not in x                    # Basis: Objekt wird weiterhin verworfen
    assert x[ol.novelty_feature("novelty_max")] == 0.6 and x["rel_volume"] == 1.2


def _train(n=140, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        nov = float(rng.uniform(0, 1))
        rows.append({"x": {"rel_volume": float(rng.normal()), "scan_day_ret": float(rng.normal()),
                           ol.novelty_feature("novelty_max"): nov},
                     "y": 0.3 * nov + float(rng.normal(0, 0.05)), "date": f"2026-10-{1 + i % 20:02d}"})
    return rows


def test_base_fit_ignores_variant_key_and_variant_uses_it():
    train = _train()
    base = ol.fit(train)
    stripped = [{**d, "x": {k: v for k, v in d["x"].items() if not k.startswith("news_novelty__")}} for d in train]
    base_stripped = ol.fit(stripped)
    assert base.names == base_stripped.names
    assert np.allclose(base.coef, base_stripped.coef)               # Basis-Prognosen bitgleich
    var = ol.fit(train, {"news_novelty_field": "novelty_max"})
    assert ol.novelty_feature("novelty_max") in var.names
    hi, lo = var.predict([{"rel_volume": 0, "scan_day_ret": 0, ol.novelty_feature("novelty_max"): 0.9},
                          {"rel_volume": 0, "scan_day_ret": 0, ol.novelty_feature("novelty_max"): 0.1}])
    assert hi > lo


def test_allowlist_accepts_only_the_registered_value():
    cfg = ol.variant_config()
    assert ol.validate_overrides({"news_novelty_field": "novelty_max"}, cfg) == {"news_novelty_field": "novelty_max"}
    assert ol.variant_params({"news_novelty_field": "novelty_max"}, cfg) == {"news_novelty_field": "novelty_max"}
    for bad in ("novelty_mean", "n_today", "anything"):
        with pytest.raises(ValueError):
            ol.validate_overrides({"news_novelty_field": bad}, cfg)


def _gap(feat, variant_cov):
    return rd._gap(f"sparse_feature:{feat}", "data", 5, "leer", "feature_coverage", 0.0, ol.MIN_FEATURE_COVERAGE, 1.0,
                   feature=feat, variant_coverage={ol.novelty_feature("novelty_max"): variant_cov})


def _novelty_props(gaps):
    return [p for p in rd.option_proposals(gaps, ol.variant_config(), [])
            if "news_novelty_field" in (p.get("change") or {})]


def test_director_proposes_the_variant_only_for_news_novelty_and_only_once_it_can_differ():
    """Live 2026-11-06: novelty_max gültig in 48 % (5 T) bzw. 9 % (20 T) der geprüften Labels. Unter
    MIN_FEATURE_COVERAGE verwirft die Variante das Merkmal wie die Basis -> kein Vorschlag (kein verbrauchter Slot)."""
    assert [p["change"] for p in _novelty_props([_gap("news_novelty", 0.62)])] == [
        {"news_novelty_field": "novelty_max"}]
    assert _novelty_props([_gap("news_novelty", 0.48)]) == []
    assert _novelty_props([_gap("short_pct_float", 0.9)]) == []


def test_variant_feature_coverage_is_reported():
    data = [{"x": {ol.novelty_feature("novelty_max"): 0.5}}, {"x": {}}]
    cov = ol.variant_feature_coverage(data)
    assert cov[ol.novelty_feature("novelty_max")] == 0.5 and cov[ol.novelty_feature("novelty_mean")] == 0.0

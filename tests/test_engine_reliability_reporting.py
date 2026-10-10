"""Engine-Warnung (2026-10-09): Gewichte lernen nur aus RELIABLE (min. 5); Korrelationen über alle
Trades sind EXPLORATORY und dürfen nicht als Grund für unveränderte Gewichte erscheinen."""
from __future__ import annotations

from datetime import date

import pytest

import feedback
from modules.engine_monitor import build_health_report
from modules.outcomes import MIN_RELIABLE_FOR_WEIGHT_UPDATE, is_reliable_outcome

TODAY = date(2026, 10, 8)
START = {"impact": 0.35, "mismatch": 0.45, "eps_drift": 0.20}
BINS = [("high", "strong", "massive"), ("mid", "good", "relevant"), ("low", "weak", "noise")]


def _trade(i, reliable, rec=False):
    b = BINS[i % 3]
    t = {"ticker": f"T{i}", "entry_date": f"2026-0{1 + i % 9}-{10 + i % 18}", "close_date": "2026-09-30",
         "outcome": (-0.3, -0.1, 0.05)[i % 3],
         "features": {"bin_impact": b[0], "bin_mismatch": b[1], "bin_eps_drift": b[2]}}
    if reliable:
        t["outcome_method"] = "spread_quote"
    if rec:
        t["outcome_method_reconstructed"] = "delta_approx"
    return t


def _history(n_rel, n_unknown, n_rec=0):
    closed = ([_trade(i, True) for i in range(n_rel)] + [_trade(100 + i, False) for i in range(n_unknown)]
              + [_trade(300 + i, False, rec=True) for i in range(n_rec)])
    return {"closed_trades": closed, "active_trades": [], "model_weights": dict(START)}


def _report(h, tmp_path):
    return build_health_report(history=h, reports_dir=tmp_path, today=TODAY)


def test_three_reliable_weights_unchanged_reason_need_more_data(tmp_path):
    h = _history(3, 77, 40)
    assert feedback.compute_pearson_weights(h) == START                       # Gewichte unverändert
    r = _report(h, tmp_path)
    ll = r["metrics"]["learn_loop"]
    assert ll["weight_update_status"] == "NEED_MORE_DATA" and ll["n_reliable"] == 3 and ll["n_non_reliable"] == 117
    assert ll["min_reliable_for_weight_update"] == MIN_RELIABLE_FOR_WEIGHT_UPDATE == 5
    main = next(w for w in r["warnings"] if w.startswith("Gewichts-Update"))
    assert "NEED_MORE_DATA" in main and "3 RELIABLE" in main and "mindestens 5" in main
    # Hauptgrund steht vor der explorativen Diagnose
    assert main.index("NEED_MORE_DATA") < main.index("EXPLORATORY")


def test_exploratory_correlation_labeled_and_without_weight_effect(tmp_path):
    h = _history(3, 77, 40)
    r = _report(h, tmp_path)
    ll = r["metrics"]["learn_loop"]
    assert set(ll["feature_corr_exploratory"]) == {"impact", "mismatch", "eps_drift"}
    assert ll["feature_corr"] == {}                                           # keine RELIABLE-Korrelation
    main = next(w for w in r["warnings"] if w.startswith("Gewichts-Update"))
    assert "EXPLORATORY" in main and "n=120, davon 3 RELIABLE und 117 NON_RELIABLE" in main
    assert "not eligible for production learning / weight update" in main
    assert feedback.compute_pearson_weights(h) == START


def test_five_reliable_evaluates_existing_update_logic_unchanged(tmp_path):
    h = _history(5, 0)
    r = _report(h, tmp_path)
    assert r["metrics"]["learn_loop"]["weight_update_status"] == "ELIGIBLE"
    assert not any("NEED_MORE_DATA" in w for w in r["warnings"])
    w = feedback.compute_pearson_weights(h)                                   # bestehende Logik läuft
    assert set(w) == set(START) and abs(sum(w.values()) - 1) < 1e-3


@pytest.mark.parametrize("n_unknown,n_rec", [(200, 0), (0, 200)])
def test_unknown_and_reconstructed_never_count_productively(tmp_path, n_unknown, n_rec):
    h = _history(3, n_unknown, n_rec)
    assert sum(is_reliable_outcome(t) for t in h["closed_trades"]) == 3
    assert feedback.compute_pearson_weights(h) == START
    assert _report(h, tmp_path)["metrics"]["learn_loop"]["weight_update_status"] == "NEED_MORE_DATA"


def test_report_never_presents_all_trades_as_reliable(tmp_path):
    r = _report(_history(3, 117), tmp_path)
    txt = " ".join(r["warnings"])
    assert "120 RELIABLE" not in txt and "eingefroren" not in txt
    assert "keine positive Vorhersagekraft" not in txt                        # alte Kausalbehauptung entfernt


def test_production_output_unchanged(tmp_path):
    """Monitor ist reine Diagnose: verändert history (Gewichte) nicht."""
    import copy
    h = _history(3, 117)
    before = copy.deepcopy(h)
    _report(h, tmp_path)
    assert h == before

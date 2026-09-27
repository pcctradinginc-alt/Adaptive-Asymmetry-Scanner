"""Inkrementalanalyse: verlorene Gewinner, Erwartung je Signal, Cluster-Struktur."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import incremental_analysis as ia  # noqa: E402


def _rows():
    # Tag 1: zwei Verlierer + ein großer Gewinner, Tag 2..6: gemischt
    rows = [{"date": "2026-01-01", "outcome": -1.0}, {"date": "2026-01-01", "outcome": -0.9},
            {"date": "2026-01-01", "outcome": 5.0}]
    rows += [{"date": f"2026-01-0{d}", "outcome": (-1) ** d * 0.3} for d in range(2, 8)]
    return rows


def test_filter_counts_lost_winners_and_opportunity_cost():
    rows = _rows()
    excluded = [r["date"] == "2026-01-01" for r in rows]      # Filter trifft Tag 1
    res = ia.evaluate_filter(rows, excluded)
    assert res["avoided_losers"] == 2 and res["lost_winners"] == 1 and res["lost_big_winners"] == 1
    # Tag-1-Summe +3.1 geht verloren -> Erwartung je Signal sinkt, obwohl
    # die Trefferquote der verbleibenden Trades steigt
    assert res["delta_expected_per_signal"] < 0
    assert res["with_filter"]["win_rate"] > res["base"]["win_rate"]
    assert res["removed_dates"] == 1


def test_sample_structure_flags_regime_month_confounding():
    rows = [{"date": "2026-04-0%d" % i, "outcome": 0.1, "state": "A"} for i in range(1, 5)]
    rows += [{"date": "2026-05-0%d" % i, "outcome": 0.1, "state": "B"} for i in range(1, 5)]
    s = ia.sample_structure(rows, "state")
    assert s["n_regimes"] == 2 and s["n_independent_dates"] == 8
    assert s["regime_x_month"] == {"A": {"2026-04": 4}, "B": {"2026-05": 4}}


def test_ledger_family_masks_use_frozen_context_only():
    rows = [{"direction": "BULLISH", "external": {
                "ticker_exposure": {"road_freight_relevance": "HIGH", "maritime_relevance": "HIGH",
                                    "weather_relevance": "HIGH"},
                "states": {"us_freight_state": "CONTRACTION", "eu_freight_state": "NEUTRAL",
                           "asia_freight_state": None, "global_maritime_state": "CONTRACTION",
                           "global_freight_state": "STRONG_CONTRACTION"},
                "primitives": {"weather_disruption_index": 0.2}}},
            {"direction": "BEARISH", "external": {"states": {"us_freight_state": "CONTRACTION"}}}]
    m = ia.ledger_family_masks(rows)
    assert m["us_freight"] == [True, False] and m["eu_freight"] == [False, False]
    assert m["road_plus_shipping"] == [True, False] and m["weather"] == [False, False]

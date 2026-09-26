"""
Tests für modules/options_designer.py – reine Strategie-Auswahlfunktionen
(P0-2 Refactor: choose_strategy / pick_spread_leg_strike).

Diese Funktionen sind bewusst reine Funktionen (kein Netzwerk, kein
Ticker/Logging), damit sowohl die Produktion (OptionsDesigner._select_strategy
/ _find_spread_leg) als auch der Ledger-Counterfactual (real_strategy)
exakt denselben Entscheidungspfad durchlaufen. Diese Tests pinnen die
bestehenden Schwellen (IV_SPREAD_GATE=52, Dealer-Gamma-Anpassung
+13/-12, VIX-Backwardation -8, Spread-Fenster 1.05–1.20× / Ziel 1.10×).
"""

import math

import pandas as pd
import pytest

import modules.options_designer as od
from modules.options_designer import (
    choose_strategy,
    compute_iv_rank,
    compute_iv_rank_components,
    pick_spread_leg_strike,
    IV_SPREAD_GATE,
)


# ── choose_strategy: Parity-Tabelle ──────────────────────────────────────────

PARITY_CASES = [
    # (iv_rank, is_bullish, dealer_gamma_state, vix_structure) -> (strategy, gate)
    (60.0, True,  None,                                              None,           "BULL_CALL_SPREAD", 52.0),
    (40.0, True,  None,                                              None,           "LONG_CALL",         52.0),
    (60.0, False, None,                                              None,           "BEAR_PUT_SPREAD",   52.0),
    (40.0, False, None,                                              None,           "LONG_PUT",          52.0),
    # Dealer-Gamma negativ → Gate +13 (65 cap)
    (55.0, True,  {"data_available": True, "net_gamma_sign": "negative"}, None,      "LONG_CALL",         65.0),
    (66.0, True,  {"data_available": True, "net_gamma_sign": "negative"}, None,      "BULL_CALL_SPREAD",  65.0),
    # Dealer-Gamma positiv → Gate -12 (40 floor)
    (45.0, True,  {"data_available": True, "net_gamma_sign": "positive"}, None,      "BULL_CALL_SPREAD",  40.0),
    (39.0, True,  {"data_available": True, "net_gamma_sign": "positive"}, None,      "LONG_CALL",         40.0),
    # VIX Backwardation → weitere -8 auf den (ggf. bereits angepassten) Gate
    (45.0, True,  None,                                              "backwardation", "BULL_CALL_SPREAD", 44.0),
    (43.0, True,  None,                                              "backwardation", "LONG_CALL",        44.0),
    # Contango: kein zusätzliches Adjustment
    (52.0, True,  None,                                              "contango",     "BULL_CALL_SPREAD",  52.0),
    # gamma_ok False (data_available fehlt) → wie neutral
    (52.0, True,  {"net_gamma_sign": "negative"},                    None,           "BULL_CALL_SPREAD",  52.0),
]


@pytest.mark.parametrize("iv_rank,is_bullish,gamma,vix,expected_strategy,expected_gate", PARITY_CASES)
def test_choose_strategy_parity(iv_rank, is_bullish, gamma, vix, expected_strategy, expected_gate):
    strategy, gate, reason = choose_strategy(iv_rank, is_bullish, gamma, vix)
    assert strategy == expected_strategy
    assert gate == pytest.approx(expected_gate)
    assert isinstance(reason, str) and reason


def test_choose_strategy_default_gate_is_module_constant():
    strategy, gate, _ = choose_strategy(IV_SPREAD_GATE, True, None, None)
    assert gate == IV_SPREAD_GATE
    assert strategy == "BULL_CALL_SPREAD"  # >= Gate → Spread


# ── pick_spread_leg_strike ───────────────────────────────────────────────────

def test_pick_spread_leg_strike_nearest_to_110pct():
    strikes = [95, 100, 105, 108, 110, 112, 115, 118, 121, 125]
    # long_strike=100 → Fenster [105, 120], Ziel 110 → exakter Treffer
    assert pick_spread_leg_strike(strikes, 100.0) == 110


def test_pick_spread_leg_strike_nearest_when_no_exact_target():
    strikes = [95, 100, 106, 116, 125]
    # long_strike=100 → Fenster [105, 120] → Kandidaten 106, 116
    # Ziel 110 → |106-110|=4, |116-110|=6 → 106 gewinnt
    assert pick_spread_leg_strike(strikes, 100.0) == 106


def test_pick_spread_leg_strike_none_when_window_empty():
    strikes = [90, 95, 100, 102, 103]
    assert pick_spread_leg_strike(strikes, 100.0) is None


def test_pick_spread_leg_strike_empty_strikes():
    assert pick_spread_leg_strike([], 100.0) is None


# ── compute_iv_rank_components / compute_iv_rank ─────────────────────────────
#
# Review-Fix: reine Extraktion aus OptionsDesigner._get_iv_rank —
#   rv_score:   Percentile-Rank der aktuellen rollierenden 21d-RV innerhalb
#               der rollierenden 1y-RV-Verteilung (5%/95%-Quantile), braucht
#               >=60 Closes/>=20 RV-Punkte, sonst Default 50.0.
#   term_score: Slope zwischen kürzestem/längstem term_points-Punkt,
#               geclampt [0,80], braucht >=2 Punkte, sonst Default 20.0.
#   combined  = round(rv_score*0.80 + term_score*0.20, 1)

def _synthetic_closes(n=300):
    """Deterministische, nicht-triviale Kursreihe (keine echte Zufälligkeit,
    aber genug Schwankung für eine nicht-degenerierte rollierende
    RV-Verteilung) — reproduzierbar über Testläufe hinweg."""
    closes = []
    price = 100.0
    for i in range(n):
        drift = 0.02 * math.sin(i / 6.0) + 0.002 * math.sin(i / 1.7)
        price *= (1 + drift * 0.05)
        closes.append(round(price, 4))
    return closes


def test_compute_iv_rank_components_default_without_enough_closes():
    parts = compute_iv_rank_components([100.0] * 30, [])
    assert parts["rv_score"] == 50.0
    assert parts["term_score"] == 20.0
    assert parts["n_term_points"] == 0
    assert parts["combined"] == pytest.approx(50.0 * 0.80 + 20.0 * 0.20, abs=1e-6)


def test_compute_iv_rank_components_default_without_enough_term_points():
    closes = _synthetic_closes(300)
    parts = compute_iv_rank_components(closes, [(30, 0.4)])  # nur 1 Punkt
    assert parts["term_score"] == 20.0
    assert parts["n_term_points"] == 1


def test_compute_iv_rank_components_term_score_slope_direction():
    closes = _synthetic_closes(300)
    # iv_short(kurzeste DTE) > iv_long(längste DTE) → positive Slope → höherer term_score
    parts_high = compute_iv_rank_components(closes, [(10, 0.60), (90, 0.30)])
    parts_low  = compute_iv_rank_components(closes, [(10, 0.30), (90, 0.60)])
    assert parts_high["term_score"] > parts_low["term_score"]
    assert parts_high["n_term_points"] == 2 and parts_low["n_term_points"] == 2


def test_compute_iv_rank_components_term_score_clamped_0_80():
    closes = _synthetic_closes(300)
    parts = compute_iv_rank_components(closes, [(10, 5.0), (90, 0.01)])  # extreme Slope
    assert parts["term_score"] == 80.0
    parts2 = compute_iv_rank_components(closes, [(10, 0.01), (90, 5.0)])
    assert parts2["term_score"] == 0.0


def test_compute_iv_rank_matches_components_combined():
    closes = _synthetic_closes(300)
    term_points = [(10, 0.5), (90, 0.4)]
    parts = compute_iv_rank_components(closes, term_points)
    assert compute_iv_rank(closes, term_points) == parts["combined"]


def test_get_iv_rank_matches_compute_iv_rank_parity(monkeypatch):
    """End-to-End-Parität: OptionsDesigner._get_iv_rank (mit gemocktem
    yfinance-Ticker + Term-Structure) muss EXAKT denselben Wert liefern wie
    die extrahierte reine Funktion compute_iv_rank() mit denselben Inputs."""
    closes = _synthetic_closes(300)
    term_points = [(14, 0.55), (120, 0.35)]

    class _FakeTicker:
        info = {"currentPrice": 100.0}

        def history(self, period="1y"):
            return pd.DataFrame({"Close": closes})

    # Instanz ohne __init__ (kein Makro-Kontext-/Netzwerk-Overhead) — nur
    # _get_iv_rank + _get_term_structure_iv werden gebraucht.
    designer = od.OptionsDesigner.__new__(od.OptionsDesigner)
    monkeypatch.setattr(
        designer, "_get_term_structure_iv",
        lambda ticker, current, t=None: list(term_points),
    )

    got = designer._get_iv_rank("FAKE", t=_FakeTicker())
    expected = compute_iv_rank(closes, term_points)
    assert got == pytest.approx(expected)


def test_get_iv_rank_returns_rv_score_only_when_current_price_zero(monkeypatch):
    """Bisheriges Randverhalten (current<=0): NUR rv_score, kein Combine mit
    dem term_score-Default — muss durch die Extraktion erhalten bleiben."""
    closes = _synthetic_closes(300)

    class _FakeTickerNoPrice:
        info = {}  # keine currentPrice/regularMarketPrice → current=0

        def history(self, period="1y"):
            return pd.DataFrame({"Close": closes})

    designer = od.OptionsDesigner.__new__(od.OptionsDesigner)
    calls = []
    monkeypatch.setattr(
        designer, "_get_term_structure_iv",
        lambda ticker, current, t=None: calls.append(1) or [(10, 0.5), (90, 0.4)],
    )

    got = designer._get_iv_rank("FAKE", t=_FakeTickerNoPrice())
    expected_rv_only = compute_iv_rank_components(closes, [])["rv_score"]
    assert got == pytest.approx(expected_rv_only)
    assert calls == []  # Term-Structure wird bei current<=0 gar nicht erst abgefragt

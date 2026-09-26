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

import pytest

from modules.options_designer import choose_strategy, pick_spread_leg_strike, IV_SPREAD_GATE


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

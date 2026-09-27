"""Short-Leg-Auswahl richtungsabhängig (Review 2026-09-27): Bear-Put-Spread
verkauft den Put UNTER dem gekauften; net_debit muss positiv sein."""
from modules.options_designer import pick_spread_leg_strike

STRIKES = [80, 85, 90, 95, 100, 105, 110, 115, 120]


def test_call_spread_short_leg_above_long():
    assert pick_spread_leg_strike(STRIKES, 100, "call") == 110


def test_put_spread_short_leg_below_long():
    assert pick_spread_leg_strike(STRIKES, 100, "put") == 90


def test_put_spread_net_debit_positive_on_real_put_chain():
    # Put-Preise steigen mit dem Strike
    put_bid = {80: 0.5, 85: 1.0, 90: 1.8, 95: 2.9, 100: 4.1, 105: 6.0, 110: 8.5, 115: 11.5, 120: 15.0}
    long_ask = 4.3
    short = pick_spread_leg_strike(STRIKES, 100, "put")
    assert long_ask - put_bid[short] > 0          # vorher Short 110 -> 4.3 - 8.5 < 0


def test_default_stays_call_for_backward_compatibility():
    assert pick_spread_leg_strike(STRIKES, 100) == 110

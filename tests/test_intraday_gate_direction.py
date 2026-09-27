"""Intraday-'zu spät'-Gate ist richtungsabhängig (Review 2026-09-27)."""
from pipeline import directional_move

LIMIT = 0.07


def _too_late(move, direction):
    return directional_move(move, direction) > LIMIT


def test_bullish_rally_is_late_but_drop_is_not():
    assert _too_late(+0.09, "BULLISH")
    assert not _too_late(-0.09, "BULLISH")      # vorher abs() -> fälschlich verworfen


def test_bearish_drop_is_late_but_rally_is_not():
    assert _too_late(-0.09, "BEARISH")
    assert not _too_late(+0.09, "BEARISH")


def test_small_moves_pass():
    assert not _too_late(0.03, "BULLISH") and not _too_late(-0.03, "BEARISH")


def test_bearish_blocked_until_pricing_validated():
    from types import SimpleNamespace
    from pipeline import bearish_trading_allowed
    assert not bearish_trading_allowed(SimpleNamespace(allow_bearish=False))
    assert not bearish_trading_allowed(SimpleNamespace(allow_bearish=True))
    assert bearish_trading_allowed(SimpleNamespace(allow_bearish=True, bearish_pricing_validated=True))

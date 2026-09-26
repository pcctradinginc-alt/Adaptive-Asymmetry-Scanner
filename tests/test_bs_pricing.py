"""Tests für modules/bs_pricing.py – Black-Scholes Fair-Value-Pricer."""

import math

import pytest

from modules.bs_pricing import bs_price, norm_cdf


def test_norm_cdf_known_values():
    assert norm_cdf(0.0) == pytest.approx(0.5, abs=1e-9)
    assert norm_cdf(1.96) == pytest.approx(0.9750, abs=1e-3)


def test_call_price_matches_known_value():
    # S=K=100, T=1y, sigma=20%, r=4% → Referenzwert ≈ 9.93
    c = bs_price(100, 100, 1.0, 0.2, r=0.04, kind="call")
    assert c == pytest.approx(9.93, abs=0.02)


def test_put_price_via_parity():
    S, K, T, sigma, r = 100.0, 100.0, 1.0, 0.2, 0.04
    c = bs_price(S, K, T, sigma, r=r, kind="call")
    p = bs_price(S, K, T, sigma, r=r, kind="put")
    parity = c - S + K * math.exp(-r * T)
    assert p == pytest.approx(parity, abs=1e-9)


def test_zero_or_negative_time_returns_intrinsic():
    assert bs_price(110, 100, 0, 0.2, kind="call") == pytest.approx(10.0)
    assert bs_price(90, 100, 0, 0.2, kind="put") == pytest.approx(10.0)
    assert bs_price(90, 100, -5, 0.2, kind="call") == pytest.approx(0.0)


def test_zero_or_negative_sigma_returns_intrinsic():
    assert bs_price(110, 100, 1.0, 0.0, kind="call") == pytest.approx(10.0)
    assert bs_price(90, 100, 1.0, -0.1, kind="put") == pytest.approx(10.0)


def test_deep_otm_call_is_cheap_but_positive():
    c = bs_price(100, 200, 0.5, 0.3, kind="call")
    assert 0.0 <= c < 1.0


def test_invalid_inputs_never_raise():
    assert bs_price("nope", 100, 1, 0.2) == 0.0
    assert bs_price(-5, 100, 1, 0.2) == 0.0
    assert bs_price(100, -5, 1, 0.2) == 0.0

"""
modules/bs_pricing.py – Black-Scholes Fair-Value-Pricer (Observability-Hilfsmodul)

Zweck: reine Black-Scholes-Bewertung für hypothetische Options-Kontrakte, die
das candidate_ledger für Counterfactual-Analysen ("was hätte ein Options-Trade
gebracht, statt nur den Underlying-Return zu betrachten") benötigt.

Kein scipy nötig — die Normalverteilungs-CDF wird über math.erf berechnet.

WICHTIG: Dies ist ein reines Bewertungsmodell (konstante IV, kein IV-Crush,
kein American-Style-Exercise, keine Dividenden). Es dient ausschließlich der
Observability, nicht der echten Trade-Bewertung.
"""

import math

__all__ = ["norm_cdf", "bs_price", "bs_delta", "implied_vol"]


def norm_cdf(x: float) -> float:
    """Standardnormalverteilung – CDF via math.erf (kein scipy nötig)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(
    S: float,
    K: float,
    T_years: float,
    sigma: float,
    r: float = 0.04,
    kind: str = "call",
) -> float:
    """
    Black-Scholes Fair Value für einen europäischen Call/Put.

    S        Spot-Preis des Underlyings
    K        Strike
    T_years  Restlaufzeit in Jahren (<=0 → intrinsischer Wert)
    sigma    Annualisierte Volatilität (<=0 → intrinsischer Wert)
    r        Risikofreier Zins (Default 4%)
    kind     "call" oder "put"

    Referenzwert: S=K=100, T=1, sigma=0.2, r=0.04 → call ≈ 9.93
    (Put via Put-Call-Parität: put = call - S + K*exp(-rT))
    """
    kind = (kind or "call").lower()
    try:
        S = float(S)
        K = float(K)
        T_years = float(T_years)
        sigma = float(sigma)
        r = float(r)
    except (TypeError, ValueError):
        return 0.0

    if S < 0 or K < 0:
        return 0.0

    intrinsic_call = max(S - K, 0.0)
    intrinsic_put = max(K - S, 0.0)

    if T_years <= 0 or sigma <= 0:
        return intrinsic_put if kind == "put" else intrinsic_call

    sqrt_t = math.sqrt(T_years)
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T_years) / (sigma * sqrt_t)
    except (ValueError, ZeroDivisionError):
        # S<=0 oder K<=0 → intrinsischer Wert als Fallback
        return intrinsic_put if kind == "put" else intrinsic_call
    d2 = d1 - sigma * sqrt_t

    disc_k = K * math.exp(-r * T_years)

    if kind == "put":
        return disc_k * norm_cdf(-d2) - S * norm_cdf(-d1)
    return S * norm_cdf(d1) - disc_k * norm_cdf(d2)


def bs_delta(S: float, K: float, T_years: float, sigma: float, r: float = 0.04, kind: str = "call") -> float | None:
    """Black-Scholes-Delta (europäisch, ohne Dividenden); None bei ungültigen Eingaben."""
    try:
        S, K, T_years, sigma, r = float(S), float(K), float(T_years), float(sigma), float(r)
        if S <= 0 or K <= 0 or T_years <= 0 or sigma <= 0:
            return None
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T_years) / (sigma * math.sqrt(T_years))
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return norm_cdf(d1) - 1.0 if (kind or "call").lower() == "put" else norm_cdf(d1)


def implied_vol(price: float, S: float, K: float, T_years: float, r: float = 0.04, kind: str = "call",
                lo: float = 1e-4, hi: float = 5.0, tol: float = 1e-7) -> float | None:
    """Implizite Volatilität per Bisektion auf bs_price; None, wenn der Preis außerhalb der Arbitragegrenzen
    (unter dem diskontierten inneren Wert bzw. über dem Preis bei hi) liegt."""
    try:
        price, S, K, T_years = float(price), float(S), float(K), float(T_years)
    except (TypeError, ValueError):
        return None
    if not (price > 0 and S > 0 and K > 0 and T_years > 0):
        return None
    p_lo, p_hi = bs_price(S, K, T_years, lo, r, kind), bs_price(S, K, T_years, hi, r, kind)
    if not p_lo - 1e-9 <= price <= p_hi + 1e-9:
        return None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if bs_price(S, K, T_years, mid, r, kind) < price:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)

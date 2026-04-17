"""
modules/mirofish_simulation.py v8.0

Monte Carlo Simulation mit historischer yfinance-Kalibrierung.

Kernidee:
    sigma (Volatilität) kommt aus echten historischen Preisdaten.
    base_alpha (Signal-Drift) = historische Drift + Signal-Bonus.
    Das macht Hit-Rates realistisch und ticker-spezifisch.

Vorher:
    sigma = hardcoded 0.02 (für alle Ticker gleich)
    alpha = mismatch/100 * 1.3 = 0.078/Tag → immer 100%

Nachher:
    sigma = std(daily_returns, 6 Monate) ← ORCL hat andere Vola als AMZN
    alpha = hist_mu + signal_strength * 0.003 ← realistisch 0.1-0.5%/Tag
    → Hit-Rates 45-80% je nach Signal-Stärke
"""

from __future__ import annotations
import logging
import math
import numpy as np
import yfinance as yf
from functools import lru_cache

log = logging.getLogger(__name__)

NARRATIVE_DECAY = {
    "short":  0.015,  # Signal verpufft schnell
    "medium": 0.008,
    "long":   0.004,
}

DEFAULT_SIGMA   = 0.020   # Fallback wenn yfinance nicht erreichbar
QUICK_MC_PATHS  = 5_000
FINAL_MC_PATHS  = 10_000
HIT_RATE_CAP    = 0.97    # Nie über 97% — statistisch unglaubwürdig


@lru_cache(maxsize=128)
def _get_hist_params(ticker: str) -> tuple[float, float]:
    """
    Holt historische Volatilität und Drift von yfinance.
    Gecacht pro Ticker — wird nur einmal pro Run abgerufen.
    
    Returns:
        (sigma, mu) — tägliche Vola und täglicher Drift
    """
    try:
        hist   = yf.Ticker(ticker).history(period="6mo")
        if hist.empty or len(hist) < 30:
            log.debug(f"  [{ticker}] Hist-Daten zu wenig → Default sigma")
            return DEFAULT_SIGMA, 0.0

        returns = hist["Close"].pct_change().dropna()
        sigma   = float(returns.std())
        mu      = float(returns.mean())

        # Sanity checks
        if not (0.005 <= sigma <= 0.15):
            sigma = DEFAULT_SIGMA
        if not (-0.005 <= mu <= 0.005):
            mu = 0.0

        log.debug(f"  [{ticker}] Hist: σ={sigma:.4f} μ={mu:.5f} ({len(returns)} Tage)")
        return sigma, mu

    except Exception as e:
        log.debug(f"  [{ticker}] yfinance Hist-Fehler: {e} → Default")
        return DEFAULT_SIGMA, 0.0


class MirofishSimulation:

    def __init__(self):
        self.rng = np.random.default_rng()

    def run_for_dte(self, candidate: dict, days_to_expiry: int = 120) -> dict | None:
        ticker   = candidate.get("ticker", "")
        features = candidate.get("features", {}) or {}
        da       = candidate.get("deep_analysis", {}) or {}
        sim_data = candidate.get("simulation", {}) or {}

        current = float(
            sim_data.get("current_price") or
            candidate.get("current_price") or
            candidate.get("info", {}).get("currentPrice") or
            candidate.get("info", {}).get("regularMarketPrice") or 0
        )
        if current <= 0:
            try:
                info    = yf.Ticker(ticker).info
                current = float(
                    info.get("currentPrice") or
                    info.get("regularMarketPrice") or
                    info.get("previousClose") or 0
                )
            except Exception:
                pass
        if current <= 0:
            log.warning(f"  [{ticker}] Kein Preis verfügbar → MC übersprungen")
            return None

        # ── Historische Parameter von yfinance ───────────────────────────────
        sigma, hist_mu = _get_hist_params(ticker)

        # ── Signal-Alpha (News-Drift) ─────────────────────────────────────────
        impact   = float(da.get("impact", 5) or 5)
        surprise = float(da.get("surprise", 5) or 5)
        ttm      = da.get("time_to_materialization", "medium") or "medium"

        # Geometrisches Mittel: beide Dimensionen müssen stark sein
        signal_strength = math.sqrt((impact / 10.0) * (surprise / 10.0))

        # Signal-Bonus: max 0.4%/Tag bei perfektem Signal (10/10)
        # Das entspricht ~12% über 30 Tage bei vollem Signal
        max_signal_alpha = 0.004
        signal_alpha     = signal_strength * max_signal_alpha

        # Gesamt-Alpha: historischer Trend + Signal-Bonus
        base_alpha = hist_mu + signal_alpha
        decay_rate = NARRATIVE_DECAY.get(ttm, 0.008)

        # Kursziel: 8% über aktuellem Preis (konservativ)
        target      = current * 1.08
        n_paths     = QUICK_MC_PATHS if days_to_expiry <= 45 else FINAL_MC_PATHS
        threshold   = 0.45 if days_to_expiry <= 45 else 0.50

        log.info(
            f"  [{ticker}] MC-Kalibrierung: "
            f"σ={sigma:.3f} μ={hist_mu:.4f} "
            f"signal={signal_strength:.2f} α={base_alpha:.4f} "
            f"({days_to_expiry}d, {n_paths} Pfade)"
        )

        # ── Monte Carlo ───────────────────────────────────────────────────────
        paths_hit = 0
        for _ in range(n_paths):
            price = current
            hit   = False
            for d in range(days_to_expiry):
                # Alpha nimmt exponentiell ab (Narrative verblasst)
                daily_alpha = base_alpha * math.exp(-decay_rate * d)
                daily_ret   = daily_alpha + sigma * self.rng.standard_normal()
                price      *= (1.0 + daily_ret)
                if price >= target:
                    hit = True
                    break
            if hit:
                paths_hit += 1

        hit_rate = paths_hit / n_paths

        # Cap: >97% ist statistisch nicht glaubwürdig
        if hit_rate >= HIT_RATE_CAP:
            log.warning(
                f"  [{ticker}] Hit-Rate={hit_rate:.1%} → Cap auf {HIT_RATE_CAP:.0%} "
                f"(Signal möglicherweise zu stark: α={base_alpha:.4f})"
            )
            hit_rate = HIT_RATE_CAP

        stderr = math.sqrt(hit_rate * (1 - hit_rate) / n_paths)

        log.info(
            f"  [{ticker}] Simulation ({days_to_expiry}d, {n_paths} Pfade): "
            f"Hit={hit_rate:.1%} ±{stderr:.1%} "
            f"({'PASS' if hit_rate >= threshold else 'FAIL'})"
        )

        if hit_rate < threshold:
            log.info(f"  [{ticker}] Hit-Rate {hit_rate:.1%} < {threshold:.0%} → verworfen")
            return None

        return {
            **candidate,
            "simulation": {
                "current_price": round(current, 2),
                "target_price":  round(target, 2),
                "hit_rate":      round(hit_rate, 4),
                "n_paths":       n_paths,
                "days":          days_to_expiry,
                "sigma":         round(sigma, 4),
                "alpha":         round(base_alpha, 5),
            }
        }

    def _get_market_params(self, ticker: str) -> tuple[float, float, float]:
        """Für ROI Pre-Check: gibt (sigma, current, target) zurück."""
        sigma, _ = _get_hist_params(ticker)
        try:
            info    = yf.Ticker(ticker).info
            current = float(
                info.get("currentPrice") or
                info.get("regularMarketPrice") or 0
            )
        except Exception:
            current = 0.0
        target = current * 1.08
        return sigma, current, target


def compute_time_value_efficiency(roi_net: float, dte: int) -> dict:
    """ROI pro Tag und annualisierter ROI (nur wenn realistisch)."""
    dte = max(int(dte or 1), 1)
    roi_per_day = (roi_net / dte) * 100  # in %

    try:
        ann = float((1 + roi_net) ** (365 / dte) - 1)
        # Nur anzeigen wenn realistisch (<500% p.a.)
        ann_roi = round(ann, 4) if abs(ann) < 5.0 else None
    except Exception:
        ann_roi = None

    return {
        "roi_per_day_pct": round(roi_per_day, 4),
        "annualized_roi":  ann_roi,
        "dte":             dte,
    }

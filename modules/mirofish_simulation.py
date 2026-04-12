"""
Stufe 5: Pfad-Simulation (MiroFish)

Fix #3: Hit-Rate 100% war unrealistisch.
        Ursachen: (a) impact_multiplier pushte Drift zu stark bei niedriger Vola,
                  (b) kein Mindest-Mismatch-Schwellenwert vor Simulation.
        Lösung:   impact_multiplier auf max 1.3 gecappt (war unbegrenzt durch
                  Mismatch/100 * multiplier-Kombination).
                  Zusätzlich: explizite Warnung wenn Hit-Rate = 100%.

Fix #5: Min-Impact-Filter. Signale mit Impact < min_impact_threshold
        (aus config.yaml) werden NICHT simuliert → schlechte Signale raus.
"""

import logging
import time
import numpy as np
import yfinance as yf
from typing import Optional

from modules.config import cfg

log = logging.getLogger(__name__)

NARRATIVE_DECAY = {
    "4-8 Wochen":  0.015,
    "2-3 Monate":  0.008,
    "6 Monate":    0.004,
}

SECTOR_VOLATILITY_MULTIPLIER = {
    "Technology":            1.3,
    "Healthcare":            0.9,
    "Energy":                1.2,
    "Financial":             1.1,
    "Consumer Cyclical":     1.0,
    "Consumer Defensive":    0.8,
    "Real Estate":           0.9,
    "Utilities":             0.7,
    "Industrials":           1.0,
    "Basic Materials":       1.1,
    "Communication Services": 1.2,
    "default":               1.0,
}


class MirofishSimulation:

    def __init__(self):
        seed      = int(time.time_ns() % (2**32))
        self._rng = np.random.default_rng(seed=seed)
        log.debug(f"MirofishSimulation RNG seed={seed}")

    def run(self, scored: list[dict]) -> list[dict]:
        # FIX #5: Min-Impact-Filter VOR der Simulation
        min_impact = getattr(cfg, 'pipeline', None)
        min_impact = getattr(min_impact, 'min_impact_threshold', 4) if min_impact else 4

        passing = []
        for s in scored:
            impact = s.get("features", {}).get("impact", 0)

            # FIX #5: Impact zu niedrig → überspringen
            if impact < min_impact:
                log.info(
                    f"  [{s['ticker']}] Impact={impact} < {min_impact} "
                    f"(min_impact_threshold) → gefiltert."
                )
                continue

            result = self._simulate(s)
            if result:
                passing.append(result)
        return passing

    def _simulate(self, s: dict) -> Optional[dict]:
        ticker    = s["ticker"]
        features  = s["features"]
        da        = s.get("deep_analysis", {})
        direction = da.get("direction", "BULLISH")
        ttm       = da.get("time_to_materialization", "2-3 Monate")
        mismatch  = features.get("mismatch", 0)

        if mismatch <= 0:
            log.info(f"  [{ticker}] Mismatch={mismatch:.2f} ≤ 0 → gefiltert.")
            return None

        sigma, current_price, sector = self._get_market_params(ticker)
        if current_price <= 0:
            log.warning(f"  [{ticker}] Kein Preis verfügbar.")
            return None

        vol_mult  = SECTOR_VOLATILITY_MULTIPLIER.get(sector, 1.0)
        sigma_adj = sigma * vol_mult

        impact     = features.get("impact", 5)
        decay_rate = NARRATIVE_DECAY.get(ttm, 0.008)

        # FIX #3: impact_multiplier gecappt auf max 1.3
        # Vorher: 1.0 + (impact/20) → bei impact=10: 1.5
        # Problem: Kombiniert mit hohem Mismatch → übertriebener Drift
        # Jetzt: 1.0 + (impact/20) aber max 1.3 → realistischerer Drift
        impact_multiplier = min(1.0 + (impact / 20.0), 1.3)
        base_alpha        = (mismatch / 100.0) * impact_multiplier

        if direction == "BEARISH":
            base_alpha = -base_alpha

        target_move = cfg.options.target_move_pct
        if direction == "BULLISH":
            target_price = current_price * (1 + target_move)
        else:
            target_price = current_price * (1 - target_move)

        n_paths   = cfg.pipeline.n_simulation_paths
        n_days    = cfg.pipeline.simulation_days
        threshold = cfg.pipeline.confidence_gate

        paths_hit = 0
        for _ in range(n_paths):
            price = current_price
            hit   = False
            for day in range(n_days):
                alpha_today  = base_alpha * np.exp(-decay_rate * day)
                daily_return = alpha_today + sigma_adj * self._rng.standard_normal()
                price       *= (1 + daily_return)
                if direction == "BULLISH" and price >= target_price:
                    hit = True; break
                if direction == "BEARISH" and price <= target_price:
                    hit = True; break
            if hit:
                paths_hit += 1

        hit_rate = paths_hit / n_paths

        # FIX #3: Warnung bei Hit-Rate = 100% (deutet auf Drift-Problem hin)
        if hit_rate >= 0.999:
            log.warning(
                f"  [{ticker}] Hit-Rate=100% – Drift möglicherweise zu hoch. "
                f"base_alpha={base_alpha:.4f} sigma_adj={sigma_adj:.4f}"
            )

        log.info(
            f"  [{ticker}] Simulation: {hit_rate:.1%} "
            f"({'PASS' if hit_rate >= threshold else 'FAIL'}) "
            f"| alpha={base_alpha:.4f} mult={impact_multiplier:.2f}"
        )

        if hit_rate < threshold:
            return None

        return {
            **s,
            "simulation": {
                "hit_rate":      round(hit_rate, 4),
                "n_paths":       n_paths,
                "n_days":        n_days,
                "target_price":  round(target_price, 2),
                "current_price": round(current_price, 2),
                "sigma_adj":     round(sigma_adj, 4),
                "sector":        sector,
                "ttm":           ttm,
            },
        }

    def _get_market_params(self, ticker: str) -> tuple[float, float, str]:
        try:
            t    = yf.Ticker(ticker)
            info = t.info
            hist = t.history(period="35d")

            current_price = float(
                info.get("currentPrice") or
                info.get("regularMarketPrice") or 0
            )
            sector = info.get("sector", "default")

            if len(hist) >= 10:
                returns = hist["Close"].pct_change().dropna()
                sigma   = float(np.std(returns))
            else:
                sigma = 0.02

            return sigma, current_price, sector
        except Exception as e:
            log.debug(f"Marktdaten Fehler für {ticker}: {e}")
            return 0.02, 0.0, "default"

"""
Stufe 5: Pfad-Simulation (MiroFish-Integration)
- 10.000 Monte-Carlo-Pfade über 120 Tage
- Adoption Curve: Nachrichtenverbreitung unter Investoren-Agenten
- Variablen: Sektor-Volatilität, Narrative-Erosion, Time-to-Materialization
- Gate: > 70% der Pfade müssen Strike-Preis erreichen
"""

import logging
import numpy as np
import yfinance as yf
from typing import Optional

log = logging.getLogger(__name__)

# Narrative-Erosion: täglich verliert die News an Kraft
NARRATIVE_DECAY = {
    "4-8 Wochen":  0.015,   # schnelle Erosion
    "2-3 Monate":  0.008,
    "6 Monate":    0.004,   # langsame Erosion = länger gültig
}

# Sektor-Beta-Adjustments (vereinfacht)
SECTOR_VOLATILITY_MULTIPLIER = {
    "Technology":       1.3,
    "Healthcare":       0.9,
    "Energy":           1.2,
    "Financial":        1.1,
    "Consumer Cyclical": 1.0,
    "default":          1.0,
}

N_PATHS   = 10_000
N_DAYS    = 120
THRESHOLD = 0.70     # 70% der Pfade müssen Strike erreichen


class MirofishSimulation:
    def run(self, scored: list[dict]) -> list[dict]:
        passing = []
        for s in scored:
            result = self._simulate(s)
            if result:
                passing.append(result)
        return passing

    def _simulate(self, s: dict) -> Optional[dict]:
        ticker   = s["ticker"]
        features = s["features"]
        da       = s.get("deep_analysis", {})
        direction = da.get("direction", "BULLISH")
        ttm       = da.get("time_to_materialization", "2-3 Monate")

        # Basis-Parameter aus Marktdaten
        sigma, current_price, sector = self._get_market_params(ticker)
        if current_price <= 0:
            log.warning(f"  [{ticker}] Kein Preis verfügbar.")
            return None

        # Sektor-Adjustierung
        vol_mult = SECTOR_VOLATILITY_MULTIPLIER.get(sector, 1.0)
        sigma_adj = sigma * vol_mult

        # Adoption-Drift: Mismatch-getriebener Alpha-Drift
        # Je höher der Mismatch, desto stärker der erwartete tägliche Drift
        mismatch     = features.get("mismatch", 0)
        impact       = features.get("impact", 5)
        decay_rate   = NARRATIVE_DECAY.get(ttm, 0.008)

        # Alpha-Drift: starts at mismatch/100 per day, decays over time
        base_alpha = mismatch / 100.0
        if direction == "BEARISH":
            base_alpha = -base_alpha

        # Strike-Target: 10% Move als Basisziel (wird in OptionsDesigner verfeinert)
        target_move = 0.10
        if direction == "BULLISH":
            target_price = current_price * (1 + target_move)
        else:
            target_price = current_price * (1 - target_move)

        # Monte-Carlo
        rng = np.random.default_rng(seed=42)
        paths_hit = 0

        for _ in range(N_PATHS):
            price = current_price
            hit   = False
            for day in range(N_DAYS):
                # Adoption-Drift (abnehmend)
                alpha_today = base_alpha * np.exp(-decay_rate * day)
                # GBM-Step
                daily_return = alpha_today + sigma_adj * rng.standard_normal()
                price *= (1 + daily_return)

                # Pfad-Check
                if direction == "BULLISH" and price >= target_price:
                    hit = True
                    break
                if direction == "BEARISH" and price <= target_price:
                    hit = True
                    break
            if hit:
                paths_hit += 1

        hit_rate = paths_hit / N_PATHS

        log.info(
            f"  [{ticker}] Simulation: {hit_rate:.1%} Pfade treffen Strike "
            f"({'PASS' if hit_rate >= THRESHOLD else 'FAIL'})"
        )

        if hit_rate < THRESHOLD:
            return None

        return {
            **s,
            "simulation": {
                "hit_rate":      round(hit_rate, 4),
                "n_paths":       N_PATHS,
                "n_days":        N_DAYS,
                "target_price":  round(target_price, 2),
                "current_price": round(current_price, 2),
                "sigma_adj":     round(sigma_adj, 4),
                "sector":        sector,
                "ttm":           ttm,
            },
        }

    def _get_market_params(self, ticker: str) -> tuple[float, float, str]:
        """Returns (sigma_daily, current_price, sector)."""
        try:
            t    = yf.Ticker(ticker)
            info = t.info
            hist = t.history(period="35d")

            current_price = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
            sector        = info.get("sector", "default")

            if len(hist) >= 10:
                returns = hist["Close"].pct_change().dropna()
                sigma   = float(np.std(returns))
            else:
                sigma = 0.02   # Fallback

            return sigma, current_price, sector
        except Exception as e:
            log.debug(f"Marktdaten Fehler für {ticker}: {e}")
            return 0.02, 0.0, "default"

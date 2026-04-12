"""
modules/mirofish_simulation.py v7.0

Fix 1: EMA-200 Hard-Gate
    Aktie unter EMA-200 → struktureller Abwärtstrend → Long Calls VERBOTEN.
    Begründung: EMA-200 ist die institutionelle Trend-Grenze. Aktien unter
    EMA-200 haben statistisch ~40% schlechtere 6-Monats-Returns für Long-Calls.
    "Fallendes Messer" Effekt: schlechte News drücken weiter, gute News
    werden durch Overhead-Resistance absorbiert.

Fix 2: MACD-Trend-Check
    MACD (12,26,9) als Zusatz-Feature im Observation-Vektor.
    MACD > Signal-Linie: Momentum bullish → Signal stärker gewichten.
    MACD < Signal-Linie: Momentum bearish → Signal schwächer gewichten
    oder bei BEARISH-Signal verstärken.

    MACD wird NICHT als Hard-Gate implementiert (zu viele False-Positives),
    sondern als Multiplikator auf base_alpha in der Simulation.
"""

import logging
import time
from typing import Optional

import numpy as np
import yfinance as yf

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
        min_impact = getattr(
            getattr(cfg, "pipeline", None), "min_impact_threshold", 4
        )
        passing = []
        for s in scored:
            impact    = s.get("features", {}).get("impact", 0)
            direction = s.get("deep_analysis", {}).get("direction", "BULLISH")

            # Fix #5 (aus v4): Impact-Schwelle
            if impact < min_impact:
                log.info(
                    f"  [{s['ticker']}] Impact={impact} < {min_impact} → gefiltert."
                )
                continue

            # FIX 1: EMA-200 Hard-Gate
            ema_ok, ema_info = self._check_ema200(s["ticker"], direction)
            s["ema200_check"] = ema_info
            if not ema_ok:
                log.info(
                    f"  [{s['ticker']}] EMA-200-GATE: Preis={ema_info['current']:.2f} "
                    f"< EMA200={ema_info['ema200']:.2f} "
                    f"({ema_info['pct_vs_ema']:.1%} unter EMA) "
                    f"→ 'Fallendes Messer' — Long-Call verworfen."
                )
                continue

            result = self._simulate(s)
            if result:
                passing.append(result)

        return passing

    # ── FIX 1: EMA-200 Hard-Gate ──────────────────────────────────────────────

    def _check_ema200(
        self, ticker: str, direction: str
    ) -> tuple[bool, dict]:
        """
        FIX 1: Prüft ob Aktie über EMA-200 notiert.

        Für BULLISH-Signale: Aktie muss ÜBER EMA-200 sein.
        Für BEARISH-Signale: Aktie muss UNTER EMA-200 sein (Short-Momentum).

        Toleranz: -2% (leicht unter EMA OK, verhindert False-Negatives
        bei Aktien die gerade die EMA testen).

        Returns:
            (passes: bool, info: dict)
        """
        tolerance = getattr(
            getattr(cfg, "pipeline", None), "ema200_tolerance", -0.02
        )

        try:
            hist = yf.Ticker(ticker).history(period="1y")

            if hist.empty or len(hist) < 50:
                log.debug(
                    f"  [{ticker}] EMA-200: Zu wenig Daten → Gate übersprungen"
                )
                return True, {"data_available": False}

            closes   = hist["Close"]
            ema200   = float(closes.ewm(span=200, adjust=False).mean().iloc[-1])
            current  = float(closes.iloc[-1])
            pct_diff = (current - ema200) / ema200   # positiv = über EMA

            info = {
                "current":        round(current, 2),
                "ema200":         round(ema200, 2),
                "pct_vs_ema":     round(pct_diff, 4),
                "above_ema200":   current > ema200,
                "data_available": True,
            }

            if direction == "BULLISH":
                # Gate: Aktie muss über EMA-200 sein (mit Toleranz)
                passes = pct_diff >= tolerance
                if passes:
                    log.info(
                        f"  [{ticker}] EMA-200 OK: Preis={current:.2f} "
                        f"{'>' if current >= ema200 else '~'} "
                        f"EMA200={ema200:.2f} ({pct_diff:+.1%})"
                    )
            else:
                # BEARISH: Aktie soll unter EMA sein (Short-Momentum)
                passes = pct_diff <= -tolerance
                if not passes:
                    log.info(
                        f"  [{ticker}] EMA-200 BEARISH-Gate: "
                        f"Preis über EMA → Short-Signal schwächer"
                    )

            return passes, info

        except Exception as e:
            log.debug(f"  [{ticker}] EMA-200 Fehler: {e} → Gate übersprungen")
            return True, {"data_available": False, "error": str(e)}

    # ── FIX 2: MACD-Trend-Check ───────────────────────────────────────────────

    def _compute_macd(self, ticker: str) -> dict:
        """
        FIX 2: Berechnet MACD (12, 26, 9) für Momentum-Check.

        MACD = EMA(12) - EMA(26)
        Signal = EMA(9) von MACD
        Histogram = MACD - Signal

        Returns:
            {
                "macd":           float,
                "signal":         float,
                "histogram":      float,
                "momentum":       "bullish" | "bearish" | "neutral",
                "momentum_score": float,   # -1.0 bis +1.0
                "data_available": bool
            }
        """
        try:
            hist = yf.Ticker(ticker).history(period="1y")

            if hist.empty or len(hist) < 35:
                return {"data_available": False, "momentum_score": 0.0}

            closes = hist["Close"]
            ema12  = closes.ewm(span=12, adjust=False).mean()
            ema26  = closes.ewm(span=26, adjust=False).mean()
            macd   = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()
            histo  = macd - signal

            macd_val   = float(macd.iloc[-1])
            signal_val = float(signal.iloc[-1])
            histo_val  = float(histo.iloc[-1])

            # Momentum: positives Histogram = MACD über Signal = bullish
            if histo_val > 0 and histo.iloc[-2] <= 0:
                momentum = "bullish_crossover"   # Frischer Crossover — starkes Signal
            elif histo_val > 0:
                momentum = "bullish"
            elif histo_val < 0 and histo.iloc[-2] >= 0:
                momentum = "bearish_crossover"
            else:
                momentum = "bearish"

            # Normierter Score: Histogram relativ zu 30-Tage-Std
            histo_std   = float(histo.rolling(30).std().iloc[-1])
            if histo_std > 0:
                momentum_score = float(
                    np.clip(histo_val / histo_std, -1.0, 1.0)
                )
            else:
                momentum_score = 0.0

            log.debug(
                f"  [{ticker}] MACD: {macd_val:.3f} Signal: {signal_val:.3f} "
                f"Histo: {histo_val:.3f} → {momentum} (score={momentum_score:.2f})"
            )

            return {
                "macd":           round(macd_val, 4),
                "signal":         round(signal_val, 4),
                "histogram":      round(histo_val, 4),
                "momentum":       momentum,
                "momentum_score": round(momentum_score, 4),
                "data_available": True,
            }

        except Exception as e:
            log.debug(f"  [{ticker}] MACD Fehler: {e}")
            return {"data_available": False, "momentum_score": 0.0}

    # ── Simulation ────────────────────────────────────────────────────────────

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
            return None

        vol_mult   = SECTOR_VOLATILITY_MULTIPLIER.get(sector, 1.0)
        sigma_adj  = sigma * vol_mult

        impact      = features.get("impact", 5)
        decay_rate  = NARRATIVE_DECAY.get(ttm, 0.008)

        # FIX 3 (aus v4): impact_multiplier cap 1.3
        impact_multiplier = min(1.0 + (impact / 20.0), 1.3)
        base_alpha        = (mismatch / 100.0) * impact_multiplier

        # FIX 2: MACD-Momentum als Alpha-Multiplikator
        macd_data = self._compute_macd(ticker)
        s["macd"] = macd_data

        if macd_data["data_available"]:
            momentum_score = macd_data["momentum_score"]
            # MACD verstärkt oder dämpft Alpha um max ±20%
            if direction == "BULLISH":
                macd_multiplier = 1.0 + (momentum_score * 0.20)
            else:
                # Für BEARISH: negatives MACD = bullish für Short
                macd_multiplier = 1.0 + (-momentum_score * 0.20)

            macd_multiplier = max(0.70, min(1.30, macd_multiplier))
            base_alpha *= macd_multiplier

            log.info(
                f"  [{ticker}] MACD-Multiplier: {macd_multiplier:.2f} "
                f"(momentum={macd_data['momentum']}, "
                f"score={momentum_score:.2f})"
            )

        if direction == "BEARISH":
            base_alpha = -base_alpha

        target_move   = cfg.options.target_move_pct
        target_price  = (
            current_price * (1 + target_move)
            if direction == "BULLISH"
            else current_price * (1 - target_move)
        )

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

        if hit_rate >= 0.999:
            log.warning(
                f"  [{ticker}] Hit-Rate=100% — Drift möglicherweise zu hoch. "
                f"base_alpha={base_alpha:.4f}"
            )

        log.info(
            f"  [{ticker}] Simulation: {hit_rate:.1%} "
            f"({'PASS' if hit_rate >= threshold else 'FAIL'}) "
            f"alpha={base_alpha:.4f} macd={macd_data.get('momentum','n/a')}"
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
                "ema200":        s.get("ema200_check", {}),
                "macd":          macd_data,
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

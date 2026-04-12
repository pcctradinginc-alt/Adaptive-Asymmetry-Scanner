"""
Stufe 7: Options-Design & Strategie
- IV-Rank / IV-Percentile via Tradier API
- IV-Rank < 50: Long Call (Vega-Long, Delta-Long)
- IV-Rank >= 50: Bull Call Spread (Short Vega / Theta-Kompensation)
- Parameter: DTE 120-200, Delta 0.60-0.70, Bid-Ask-Spread < 10%
- Finales Skeptiker-Audit (Bear Case Check)
"""

import logging
import os
import requests
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional

log = logging.getLogger(__name__)

TRADIER_BASE  = "https://api.tradier.com/v1"
TRADIER_KEY   = os.getenv("TRADIER_API_KEY", "")

MIN_OPEN_INTEREST = 100
MAX_BID_ASK_RATIO = 0.10   # 10%
DTE_MIN, DTE_MAX  = 120, 200
DELTA_MIN, DELTA_MAX = 0.60, 0.70


class OptionsDesigner:
    def __init__(self, gates):
        self.gates = gates

    def run(self, signals: list[dict]) -> list[dict]:
        proposals = []
        for s in signals:
            # Finales Bear-Case Audit
            if not self._bear_case_ok(s):
                log.info(f"  [{s['ticker']}] Bear-Case Audit FAILED – übersprungen.")
                continue

            proposal = self._design_option(s)
            if proposal:
                proposals.append(proposal)
        return proposals

    # ── Bear-Case Audit ───────────────────────────────────────────────────────

    def _bear_case_ok(self, s: dict) -> bool:
        da       = s.get("deep_analysis", {})
        severity = da.get("bear_case_severity", 0)
        # Blockiere wenn Bear-Case-Schwere > 7 (zu riskant)
        if severity > 7:
            log.info(
                f"  [{s['ticker']}] Bear-Case-Severity={severity} > 7 → blockiert."
            )
            return False
        return True

    # ── Options-Design ────────────────────────────────────────────────────────

    def _design_option(self, s: dict) -> Optional[dict]:
        ticker    = s["ticker"]
        direction = s.get("deep_analysis", {}).get("direction", "BULLISH")
        sim       = s.get("simulation", {})
        current   = sim.get("current_price", 0)

        if current <= 0:
            return None

        # Earnings-Gate
        if self.gates.has_upcoming_earnings(ticker, days=7):
            log.info(f"  [{ticker}] Earnings < 7 Tage → blockiert.")
            return None

        # IV-Daten
        iv_rank = self._get_iv_rank(ticker)
        log.info(f"  [{ticker}] IV-Rank={iv_rank}")

        # Strategie-Wahl
        if iv_rank < 50:
            strategy = "LONG_CALL" if direction == "BULLISH" else "LONG_PUT"
        else:
            strategy = "BULL_CALL_SPREAD" if direction == "BULLISH" else "BEAR_PUT_SPREAD"

        # Bestes Options-Contract finden
        option = self._find_best_option(ticker, strategy, current)
        if not option:
            log.warning(f"  [{ticker}] Kein geeigneter Options-Kontrakt gefunden.")
            return None

        return {
            "ticker":    ticker,
            "strategy":  strategy,
            "iv_rank":   iv_rank,
            "direction": direction,
            "option":    option,
            "features":  s.get("features", {}),
            "simulation": s.get("simulation", {}),
            "deep_analysis": s.get("deep_analysis", {}),
            "final_score": s.get("final_score", 0),
        }

    # ── IV-Rank via Tradier ───────────────────────────────────────────────────

    def _get_iv_rank(self, ticker: str) -> float:
        if not TRADIER_KEY:
            return self._estimate_iv_rank_from_yfinance(ticker)
        try:
            headers = {
                "Authorization": f"Bearer {TRADIER_KEY}",
                "Accept": "application/json",
            }
            r = requests.get(
                f"{TRADIER_BASE}/markets/options/strikes",
                params={"symbol": ticker, "expiration": self._next_expiry()},
                headers=headers,
                timeout=10,
            )
            data = r.json()
            # Vereinfachter IV-Rank aus aktueller IV vs. 52W-Range
            iv_current = data.get("iv", 0.25)
            iv_52w_low = data.get("iv_52_week_low", 0.15)
            iv_52w_high = data.get("iv_52_week_high", 0.60)
            if iv_52w_high == iv_52w_low:
                return 50.0
            return ((iv_current - iv_52w_low) / (iv_52w_high - iv_52w_low)) * 100
        except Exception:
            return self._estimate_iv_rank_from_yfinance(ticker)

    def _estimate_iv_rank_from_yfinance(self, ticker: str) -> float:
        """Schätzt IV-Rank aus yfinance-Optionsdaten."""
        try:
            t = yf.Ticker(ticker)
            dates = t.options
            if not dates:
                return 30.0
            chain = t.option_chain(dates[0])
            calls = chain.calls
            if calls.empty or "impliedVolatility" not in calls.columns:
                return 30.0
            avg_iv = calls["impliedVolatility"].median() * 100
            # Grobe Heuristik: < 25 IV → niedriger Rank, > 40 → hoher Rank
            if avg_iv < 20:
                return 20.0
            if avg_iv > 50:
                return 70.0
            return avg_iv
        except Exception:
            return 30.0

    def _next_expiry(self) -> str:
        d = datetime.utcnow() + timedelta(days=150)
        return d.strftime("%Y-%m-%d")

    # ── Kontrakt-Suche ────────────────────────────────────────────────────────

    def _find_best_option(
        self, ticker: str, strategy: str, current_price: float
    ) -> Optional[dict]:
        try:
            t = yf.Ticker(ticker)
            expiry_dates = [
                d for d in t.options
                if DTE_MIN <= self._days_to(d) <= DTE_MAX
            ]
            if not expiry_dates:
                return None

            best_expiry = expiry_dates[0]
            chain = t.option_chain(best_expiry)
            options = chain.calls if "CALL" in strategy or "BULL" in strategy else chain.puts

            # Delta-Proxy: Strike nahe Delta 0.60-0.70 → ca. 5-15% OTM
            target_strike_low  = current_price * 1.00
            target_strike_high = current_price * 1.12

            filtered = options[
                (options["strike"] >= target_strike_low) &
                (options["strike"] <= target_strike_high) &
                (options["openInterest"] >= MIN_OPEN_INTEREST)
            ].copy()

            if filtered.empty:
                return None

            # Bid-Ask-Spread Filter
            filtered["spread_ratio"] = (filtered["ask"] - filtered["bid"]) / filtered["ask"]
            filtered = filtered[filtered["spread_ratio"] <= MAX_BID_ASK_RATIO]

            if filtered.empty:
                return None

            # Bester Kontrakt: höchstes Open Interest
            best = filtered.sort_values("openInterest", ascending=False).iloc[0]

            result = {
                "expiry":          best_expiry,
                "strike":          float(best["strike"]),
                "bid":             float(best["bid"]),
                "ask":             float(best["ask"]),
                "last":            float(best.get("lastPrice", 0)),
                "open_interest":   int(best["openInterest"]),
                "implied_vol":     float(best.get("impliedVolatility", 0)),
                "spread_ratio":    round(float(best["spread_ratio"]), 4),
                "dte":             self._days_to(best_expiry),
            }

            # Spread-Leg für Bull Call Spread
            if strategy == "BULL_CALL_SPREAD":
                spread_leg = self._find_spread_leg(
                    options, best["strike"], current_price
                )
                result["spread_leg"] = spread_leg

            return result

        except Exception as e:
            log.debug(f"Kontrakt-Suche Fehler für {ticker}: {e}")
            return None

    def _find_spread_leg(
        self, options, long_strike: float, current_price: float
    ) -> Optional[dict]:
        """Findet den Short-Strike für den Bull Call Spread (~10% über Long)."""
        short_target = long_strike * 1.10
        candidates = options[
            (options["strike"] >= long_strike * 1.05) &
            (options["strike"] <= long_strike * 1.20) &
            (options["openInterest"] >= MIN_OPEN_INTEREST)
        ]
        if candidates.empty:
            return None
        best = candidates.iloc[(candidates["strike"] - short_target).abs().argsort()].iloc[0]
        return {
            "strike": float(best["strike"]),
            "bid":    float(best["bid"]),
            "ask":    float(best["ask"]),
        }

    def _days_to(self, expiry_str: str) -> int:
        try:
            d = datetime.strptime(expiry_str, "%Y-%m-%d")
            return (d - datetime.utcnow()).days
        except Exception:
            return 0

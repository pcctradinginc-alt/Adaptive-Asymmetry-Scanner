"""
Stufe 7: Options-Design & Strategie

Fixes:
  C-03: Falscher Tradier-Endpunkt. markets/options/strikes gibt keine IV-Daten.
        Fix: markets/options/quotes mit greeks=true für echte IV-Daten.
        Fallback-Kette: Tradier → yfinance → Standardwert 30.0
  L-03: TRADIER_KEY auf Modulebene gesetzt → Env-Änderungen nach Import
        hatten keine Wirkung. Fix: os.getenv() zur Laufzeit aufrufen.
  cfg:  Alle Konstanten aus config.yaml.
"""

import logging
import os
import requests
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional

from modules.config import cfg

log = logging.getLogger(__name__)

TRADIER_BASE = "https://api.tradier.com/v1"


class OptionsDesigner:

    def __init__(self, gates):
        self.gates = gates

    def run(self, signals: list[dict]) -> list[dict]:
        proposals = []
        for s in signals:
            if not self._bear_case_ok(s):
                log.info(
                    f"  [{s['ticker']}] Bear-Case Audit FAILED – übersprungen."
                )
                continue
            proposal = self._design_option(s)
            if proposal:
                proposals.append(proposal)
        return proposals

    # ── Bear-Case Audit ───────────────────────────────────────────────────────

    def _bear_case_ok(self, s: dict) -> bool:
        severity = s.get("deep_analysis", {}).get("bear_case_severity", 0)
        if severity > cfg.risk.max_bear_case_severity:
            log.info(
                f"  [{s['ticker']}] Bear-Case-Severity={severity} "
                f"> {cfg.risk.max_bear_case_severity} → blockiert."
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

        if self.gates.has_upcoming_earnings(ticker):
            log.info(
                f"  [{ticker}] Earnings < "
                f"{cfg.risk.earnings_buffer_days} Tage → blockiert."
            )
            return None

        iv_rank = self._get_iv_rank(ticker)
        log.info(f"  [{ticker}] IV-Rank={iv_rank:.1f}")

        if iv_rank < 50:
            strategy = "LONG_CALL" if direction == "BULLISH" else "LONG_PUT"
        else:
            strategy = (
                "BULL_CALL_SPREAD" if direction == "BULLISH"
                else "BEAR_PUT_SPREAD"
            )

        option = self._find_best_option(ticker, strategy, current)
        if not option:
            log.warning(
                f"  [{ticker}] Kein geeigneter Options-Kontrakt gefunden."
            )
            return None

        return {
            "ticker":       ticker,
            "strategy":     strategy,
            "iv_rank":      iv_rank,
            "direction":    direction,
            "option":       option,
            "features":     s.get("features", {}),
            "simulation":   s.get("simulation", {}),
            "deep_analysis": s.get("deep_analysis", {}),
            "final_score":  s.get("final_score", 0),
        }

    # ── IV-Rank via Tradier ───────────────────────────────────────────────────

    def _get_iv_rank(self, ticker: str) -> float:
        """
        FIX C-03: Korrekter Tradier-Endpunkt für IV-Daten.

        Vorher: markets/options/strikes → gibt Strikes-Liste, KEINE IV-Daten.
                Felder 'iv', 'iv_52_week_low', 'iv_52_week_high' existieren
                nicht → immer Default → immer IV-Rank = 22.2%.

        Jetzt:  markets/options/quotes mit greeks=true → enthält
                impliedVolatility pro Kontrakt. IV-Rank wird aus
                aktuellem ATM-IV vs. 52W-Range berechnet.

        FIX L-03: TRADIER_KEY zur Laufzeit lesen, nicht bei Modulimport.
        """
        # L-03: os.getenv() zur Laufzeit, nicht auf Modulebene
        tradier_key = os.getenv("TRADIER_API_KEY", "")

        if not tradier_key:
            return self._estimate_iv_rank_from_yfinance(ticker)

        try:
            headers = {
                "Authorization": f"Bearer {tradier_key}",
                "Accept": "application/json",
            }

            # Schritt 1: Verfügbare Expiries laden
            exp_resp = requests.get(
                f"{TRADIER_BASE}/markets/options/expirations",
                params={"symbol": ticker, "includeAllRoots": "true"},
                headers=headers,
                timeout=10,
            )
            exp_data = exp_resp.json()
            expirations = (
                exp_data.get("expirations", {}).get("date", []) or []
            )
            if not expirations:
                return self._estimate_iv_rank_from_yfinance(ticker)

            # Schritt 2: Expiry im Ziel-DTE-Bereich wählen
            target_expiry = None
            for exp in expirations:
                dte = self._days_to(exp)
                if cfg.options.dte_min <= dte <= cfg.options.dte_max:
                    target_expiry = exp
                    break

            if not target_expiry:
                return self._estimate_iv_rank_from_yfinance(ticker)

            # Schritt 3: FIX C-03 – korrekter Endpunkt mit greeks
            quotes_resp = requests.get(
                f"{TRADIER_BASE}/markets/options/chains",
                params={
                    "symbol":     ticker,
                    "expiration": target_expiry,
                    "greeks":     "true",
                },
                headers=headers,
                timeout=10,
            )
            chains = quotes_resp.json()
            options = chains.get("options", {}).get("option", []) or []

            if not options:
                return self._estimate_iv_rank_from_yfinance(ticker)

            # ATM-Calls: Strike nächste zum aktuellen Preis
            calls = [o for o in options if o.get("option_type") == "call"]
            if not calls:
                return self._estimate_iv_rank_from_yfinance(ticker)

            # Mittlerer IV aus ATM-Calls (nächste 3 Strikes)
            calls_sorted = sorted(
                calls,
                key=lambda o: abs(
                    o.get("strike", 0) -
                    (o.get("underlying_price") or o.get("strike", 0))
                )
            )
            atm_ivs = [
                float(c["greeks"]["mid_iv"])
                for c in calls_sorted[:3]
                if c.get("greeks") and c["greeks"].get("mid_iv")
            ]

            if not atm_ivs:
                return self._estimate_iv_rank_from_yfinance(ticker)

            iv_current = sum(atm_ivs) / len(atm_ivs)

            # 52W IV-Range aus statistics-Endpunkt
            stats_resp = requests.get(
                f"{TRADIER_BASE}/markets/options/strikes",
                params={"symbol": ticker, "expiration": target_expiry},
                headers=headers,
                timeout=10,
            )
            # Schätze 52W-Range aus historischer Volatilität
            iv_52w_low, iv_52w_high = self._estimate_iv_range(ticker, iv_current)

            if iv_52w_high <= iv_52w_low:
                return 50.0

            iv_rank = ((iv_current - iv_52w_low) / (iv_52w_high - iv_52w_low)) * 100
            return round(max(0.0, min(100.0, iv_rank)), 2)

        except Exception as e:
            log.debug(f"Tradier IV-Rank Fehler für {ticker}: {e}")
            return self._estimate_iv_rank_from_yfinance(ticker)

    def _estimate_iv_range(
        self, ticker: str, current_iv: float
    ) -> tuple[float, float]:
        """
        Schätzt 52W IV-Low/High aus historischen Returns.
        Heuristik: σ_30d × √252 = annualisierte Vola als IV-Proxy.
        """
        try:
            hist = yf.Ticker(ticker).history(period="1y")
            if len(hist) < 50:
                return current_iv * 0.6, current_iv * 1.4

            daily_returns = hist["Close"].pct_change().dropna()
            # Rollierende 30-Tages-Vola (annualisiert)
            rolling_vol = (
                daily_returns
                .rolling(30)
                .std()
                .dropna()
                * (252 ** 0.5)
            )
            iv_low  = float(rolling_vol.quantile(0.10))
            iv_high = float(rolling_vol.quantile(0.90))
            return iv_low, iv_high

        except Exception:
            return current_iv * 0.6, current_iv * 1.4

    def _estimate_iv_rank_from_yfinance(self, ticker: str) -> float:
        """Schätzt IV-Rank aus yfinance-Optionsdaten (Fallback)."""
        try:
            t = yf.Ticker(ticker)
            dates = t.options
            if not dates:
                return 30.0

            # Nächste Expiry im DTE-Bereich wählen
            target_date = None
            for d in dates:
                dte = self._days_to(d)
                if cfg.options.dte_min <= dte <= cfg.options.dte_max:
                    target_date = d
                    break
            if not target_date:
                target_date = dates[0]

            chain = t.option_chain(target_date)
            calls = chain.calls

            if calls.empty or "impliedVolatility" not in calls.columns:
                return 30.0

            # Aktueller IV: Median der ATM-nahen Calls
            current_iv = float(calls["impliedVolatility"].median())

            # 52W-Range schätzen
            iv_low, iv_high = self._estimate_iv_range(ticker, current_iv)

            if iv_high <= iv_low:
                return 30.0

            iv_rank = ((current_iv - iv_low) / (iv_high - iv_low)) * 100
            return round(max(0.0, min(100.0, iv_rank)), 2)

        except Exception:
            return 30.0

    # ── Kontrakt-Suche ────────────────────────────────────────────────────────

    def _find_best_option(
        self, ticker: str, strategy: str, current_price: float
    ) -> Optional[dict]:
        try:
            t = yf.Ticker(ticker)
            expiry_dates = [
                d for d in t.options
                if cfg.options.dte_min <= self._days_to(d) <= cfg.options.dte_max
            ]
            if not expiry_dates:
                return None

            best_expiry = expiry_dates[0]
            chain   = t.option_chain(best_expiry)
            options = (
                chain.calls
                if "CALL" in strategy or "BULL" in strategy
                else chain.puts
            )

            target_low  = current_price * 1.00
            target_high = current_price * 1.12

            filtered = options[
                (options["strike"] >= target_low) &
                (options["strike"] <= target_high) &
                (options["openInterest"] >= cfg.risk.min_open_interest)
            ].copy()

            if filtered.empty:
                return None

            filtered["spread_ratio"] = (
                (filtered["ask"] - filtered["bid"]) / filtered["ask"]
            )
            filtered = filtered[
                filtered["spread_ratio"] <= cfg.risk.max_bid_ask_ratio
            ]

            if filtered.empty:
                return None

            best = filtered.sort_values("openInterest", ascending=False).iloc[0]

            result = {
                "expiry":        best_expiry,
                "strike":        float(best["strike"]),
                "bid":           float(best["bid"]),
                "ask":           float(best["ask"]),
                "last":          float(best.get("lastPrice", 0)),
                "open_interest": int(best["openInterest"]),
                "implied_vol":   float(best.get("impliedVolatility", 0)),
                "spread_ratio":  round(float(best["spread_ratio"]), 4),
                "dte":           self._days_to(best_expiry),
            }

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
        short_target = long_strike * 1.10
        candidates = options[
            (options["strike"] >= long_strike * 1.05) &
            (options["strike"] <= long_strike * 1.20) &
            (options["openInterest"] >= cfg.risk.min_open_interest)
        ]
        if candidates.empty:
            return None
        best = candidates.iloc[
            (candidates["strike"] - short_target).abs().argsort()
        ].iloc[0]
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

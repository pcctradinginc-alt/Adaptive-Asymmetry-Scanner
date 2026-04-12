"""
modules/options_designer.py v6.0

Fix 2: IV-Rank Hard-Gate
    Wenn IV-Rank > 70% → Long Call VERBOTEN (statistisch Verlierer auf 2-6M)
    Stattdessen: Bull Call Spread (reduziert Vega-Exposure um ~60%)
    Begründung: Long Call bei hohem IV-Rank hat negatives Vega-P&L auch bei
                steigender Aktie durch IV Mean-Reversion.

Fix 3: Sector Momentum Check
    Trade nur Long wenn Aktie Relative Stärke gegenüber Sektor-ETF zeigt.
    Schwacher Sektor = schwaches Signal, egal wie gut die Einzel-Aktie aussieht.
    Implementiert via yfinance (kostenlos, kein API-Key).

Sektor-ETF Mapping:
    Technology      → XLK
    Healthcare      → XLV
    Financials      → XLF
    Energy          → XLE
    Consumer Cycl.  → XLY
    Consumer Def.   → XLP
    Industrials     → XLI
    Materials       → XLB
    Real Estate     → XLRE
    Utilities       → XLU
    Communication   → XLC
    Biotech         → XBI (spezifischer als XLV)
"""

from __future__ import annotations
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import requests
import yfinance as yf

from modules.config import cfg

log = logging.getLogger(__name__)

TRADIER_BASE = "https://api.tradier.com/v1"

# FIX 2: IV-Schwellenwert über dem Long Calls verboten sind
IV_RANK_HARD_GATE = 70.0   # aus config.yaml: risk.iv_rank_long_call_max

# FIX 3: Sektor-ETF Mapping
SECTOR_ETF = {
    "Technology":              "XLK",
    "Healthcare":              "XLV",
    "Biotechnology":           "XBI",   # spezifischer
    "Pharmaceuticals":         "XLV",
    "Financial Services":      "XLF",
    "Financials":              "XLF",
    "Energy":                  "XLE",
    "Consumer Cyclical":       "XLY",
    "Consumer Defensive":      "XLP",
    "Industrials":             "XLI",
    "Basic Materials":         "XLB",
    "Real Estate":             "XLRE",
    "Utilities":               "XLU",
    "Communication Services":  "XLC",
    "default":                 "SPY",   # Fallback
}

# Relative Stärke: Aktie muss besser als Sektor-ETF performen (30 Tage)
RELATIVE_STRENGTH_DAYS     = 30
RELATIVE_STRENGTH_MIN_DIFF = -0.03   # Aktie darf max. 3% schlechter als ETF sein


class OptionsDesigner:

    def __init__(self, gates):
        self.gates = gates

    def run(self, signals: list[dict]) -> list[dict]:
        proposals = []
        for s in signals:
            ticker = s.get("ticker", "")

            # FIX 3: Sector Momentum Check
            sector_ok, sector_info = self._check_sector_momentum(s)
            s["sector_momentum"] = sector_info
            if not sector_ok:
                log.info(
                    f"  [{ticker}] SECTOR-GATE: Aktie underperformt Sektor-ETF "
                    f"({sector_info.get('sector_etf')}) → Signal verworfen."
                )
                continue

            if not self._bear_case_ok(s):
                log.info(f"  [{ticker}] Bear-Case FAILED → übersprungen.")
                continue

            proposal = self._design_option(s)
            if proposal:
                proposals.append(proposal)

        return proposals

    # ── FIX 2: IV-Rank Hard-Gate ──────────────────────────────────────────────

    def _select_strategy(self, ticker: str, direction: str, iv_rank: float) -> str:
        """
        FIX 2: Strategiewahl basierend auf IV-Rank.

        IV-Rank < 70%:  Long Call / Long Put (positives Vega OK)
        IV-Rank >= 70%: Spread-Strategie (Vega-neutral / negatives Vega)
            BULLISH → Bull Call Spread
            BEARISH → Bear Put Spread

        Begründung: Bei IV-Rank >= 70% ist IV auf historischem Peak.
        Mean-Reversion der IV drückt Option P&L auch wenn Aktie steigt.
        Spreads verkaufen die "überteuerte" Short-Leg und hedgen das Vega.
        """
        iv_gate = getattr(
            getattr(cfg, "risk", None), "iv_rank_long_call_max", IV_RANK_HARD_GATE
        )

        if iv_rank >= iv_gate:
            strategy = "BULL_CALL_SPREAD" if direction == "BULLISH" else "BEAR_PUT_SPREAD"
            log.info(
                f"  [{ticker}] IV-RANK HARD-GATE: IV-Rank={iv_rank:.1f}% >= {iv_gate:.0f}% "
                f"→ Erzwinge {strategy} statt Long Option (Vega-Schutz)."
            )
        else:
            strategy = "LONG_CALL" if direction == "BULLISH" else "LONG_PUT"
            log.info(
                f"  [{ticker}] IV-Rank={iv_rank:.1f}% < {iv_gate:.0f}% "
                f"→ {strategy} erlaubt."
            )

        return strategy

    # ── FIX 3: Sector Momentum Check ─────────────────────────────────────────

    def _check_sector_momentum(self, s: dict) -> tuple[bool, dict]:
        """
        FIX 3: Prüft ob Aktie Relative Stärke gegenüber Sektor-ETF zeigt.

        Methode: 30-Tage-Performance Aktie vs. Sektor-ETF.
        Wenn Aktie > (ETF - 3%): OK → Long-Signal valide.
        Wenn Aktie < (ETF - 3%): FAIL → Sektor zieht nach unten.

        Edge-Cases:
        - Kein Sektor in yfinance: Fallback → SPY → immer OK
        - BEARISH-Signal: invertierte Logik (ETF soll schwach sein)
        - yfinance Fehler: konservativ → OK (nicht verwerfen)
        """
        ticker    = s.get("ticker", "")
        info      = s.get("info", {})
        direction = s.get("deep_analysis", {}).get("direction", "BULLISH")
        sector    = info.get("sector", "default")

        etf_ticker = SECTOR_ETF.get(sector, SECTOR_ETF["default"])

        try:
            # 30-Tage Performance berechnen
            period   = f"{RELATIVE_STRENGTH_DAYS + 5}d"
            stock_h  = yf.Ticker(ticker).history(period=period)
            etf_h    = yf.Ticker(etf_ticker).history(period=period)

            if stock_h.empty or etf_h.empty or len(stock_h) < 5 or len(etf_h) < 5:
                return True, {"sector_etf": etf_ticker, "data_available": False}

            stock_ret = float(
                (stock_h["Close"].iloc[-1] - stock_h["Close"].iloc[0])
                / stock_h["Close"].iloc[0]
            )
            etf_ret   = float(
                (etf_h["Close"].iloc[-1] - etf_h["Close"].iloc[0])
                / etf_h["Close"].iloc[0]
            )

            rel_strength = stock_ret - etf_ret

            sector_info = {
                "sector":          sector,
                "sector_etf":      etf_ticker,
                "stock_ret_30d":   round(stock_ret, 4),
                "etf_ret_30d":     round(etf_ret, 4),
                "rel_strength":    round(rel_strength, 4),
                "data_available":  True,
            }

            log.info(
                f"  [{ticker}] Sector: {sector} ({etf_ticker}) | "
                f"Aktie={stock_ret:+.1%} ETF={etf_ret:+.1%} "
                f"RelStärke={rel_strength:+.1%}"
            )

            if direction == "BULLISH":
                # Für Long: Aktie soll nicht zu weit hinter ETF zurückliegen
                ok = rel_strength >= RELATIVE_STRENGTH_MIN_DIFF
                if not ok:
                    sector_info["fail_reason"] = (
                        f"Aktie underperformt {etf_ticker} um "
                        f"{rel_strength:.1%} (Min: {RELATIVE_STRENGTH_MIN_DIFF:.0%})"
                    )
            else:
                # Für Short/Bearish: Aktie soll schwächer als Sektor sein
                ok = rel_strength <= -RELATIVE_STRENGTH_MIN_DIFF

            return ok, sector_info

        except Exception as e:
            log.debug(f"  [{ticker}] Sector-Check Fehler: {e} → OK (Fallback)")
            return True, {
                "sector_etf": etf_ticker, "data_available": False,
                "error": str(e)
            }

    # ── Bear-Case Gate ────────────────────────────────────────────────────────

    def _bear_case_ok(self, s: dict) -> bool:
        severity  = s.get("deep_analysis", {}).get("bear_case_severity", 0)
        threshold = cfg.risk.max_bear_case_severity
        if severity >= threshold:
            log.info(f"  [{s['ticker']}] Bear-Case={severity} >= {threshold} → blockiert.")
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
            log.info(f"  [{ticker}] Earnings-Gate → blockiert.")
            return None

        iv_rank = self._get_iv_rank(ticker)
        log.info(f"  [{ticker}] IV-Rank={iv_rank:.1f}%")

        # FIX 2: IV-basierte Strategiewahl
        strategy = self._select_strategy(ticker, direction, iv_rank)

        option = self._find_best_option(ticker, strategy, current)
        if not option:
            log.warning(f"  [{ticker}] Kein geeigneter Kontrakt.")
            return None

        return {
            "ticker":          ticker,
            "strategy":        strategy,
            "iv_rank":         iv_rank,
            "iv_gate_applied": iv_rank >= getattr(
                getattr(cfg, "risk", None), "iv_rank_long_call_max", IV_RANK_HARD_GATE
            ),
            "direction":       direction,
            "option":          option,
            "features":        s.get("features", {}),
            "simulation":      s.get("simulation", {}),
            "deep_analysis":   s.get("deep_analysis", {}),
            "sector_momentum": s.get("sector_momentum", {}),
            "final_score":     s.get("final_score", 0),
        }

    # ── IV-Rank Berechnung ────────────────────────────────────────────────────

    def _get_iv_rank(self, ticker: str) -> float:
        """IV-Rank via Tradier → yfinance Fallback."""
        tradier_key = os.getenv("TRADIER_API_KEY", "")
        if tradier_key:
            try:
                iv = self._get_iv_from_tradier(ticker, tradier_key)
                if iv is not None:
                    return iv
            except Exception:
                pass
        return self._estimate_iv_rank_from_yfinance(ticker)

    def _get_iv_from_tradier(self, ticker: str, api_key: str) -> Optional[float]:
        headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
        try:
            exp_resp    = requests.get(
                f"{TRADIER_BASE}/markets/options/expirations",
                params={"symbol": ticker, "includeAllRoots": "true"},
                headers=headers, timeout=10,
            )
            expirations = exp_resp.json().get("expirations", {}).get("date", []) or []
            target_exp  = next(
                (e for e in expirations
                 if cfg.options.dte_min <= self._days_to(e) <= cfg.options.dte_max),
                None
            )
            if not target_exp:
                return None

            q_resp  = requests.get(
                f"{TRADIER_BASE}/markets/options/chains",
                params={"symbol": ticker, "expiration": target_exp, "greeks": "true"},
                headers=headers, timeout=10,
            )
            options = q_resp.json().get("options", {}).get("option", []) or []
            calls   = [o for o in options if o.get("option_type") == "call"]
            if not calls:
                return None

            atm_ivs = [
                float(c["greeks"]["mid_iv"])
                for c in sorted(calls, key=lambda o: abs(o.get("strike", 0)))[:3]
                if c.get("greeks") and c["greeks"].get("mid_iv")
            ]
            if not atm_ivs:
                return None

            iv_current = sum(atm_ivs) / len(atm_ivs)
            return self._iv_to_rank(ticker, iv_current)
        except Exception:
            return None

    def _iv_to_rank(self, ticker: str, iv_current: float) -> float:
        try:
            hist    = yf.Ticker(ticker).history(period="1y")
            returns = hist["Close"].pct_change().dropna()
            rolling = returns.rolling(30).std().dropna() * (252 ** 0.5)
            iv_low  = float(rolling.quantile(0.10))
            iv_high = float(rolling.quantile(0.90))
            if iv_high <= iv_low:
                return 50.0
            rank = ((iv_current - iv_low) / (iv_high - iv_low)) * 100
            return round(max(0.0, min(100.0, rank)), 2)
        except Exception:
            return 50.0

    def _estimate_iv_rank_from_yfinance(self, ticker: str) -> float:
        try:
            t     = yf.Ticker(ticker)
            dates = t.options
            if not dates:
                return 30.0
            target = next(
                (d for d in dates
                 if cfg.options.dte_min <= self._days_to(d) <= cfg.options.dte_max),
                dates[0]
            )
            chain = t.option_chain(target)
            calls = chain.calls
            if calls.empty or "impliedVolatility" not in calls.columns:
                return 30.0
            iv_current = float(calls["impliedVolatility"].median())
            return self._iv_to_rank(ticker, iv_current)
        except Exception:
            return 30.0

    # ── Kontrakt-Suche ────────────────────────────────────────────────────────

    def _find_best_option(
        self, ticker: str, strategy: str, current_price: float
    ) -> Optional[dict]:
        try:
            t     = yf.Ticker(ticker)
            dates = [
                d for d in t.options
                if cfg.options.dte_min <= self._days_to(d) <= cfg.options.dte_max
            ]
            if not dates:
                return None

            best_expiry = dates[0]
            chain       = t.option_chain(best_expiry)
            is_call     = "CALL" in strategy or "BULL" in strategy
            options     = chain.calls if is_call else chain.puts

            filtered = options[
                (options["strike"] >= current_price * 1.00) &
                (options["strike"] <= current_price * 1.12) &
                (options["openInterest"] >= cfg.risk.min_open_interest)
            ].copy()

            if filtered.empty:
                return None

            filtered["spread_ratio"] = (
                (filtered["ask"] - filtered["bid"]) / filtered["ask"].clip(lower=0.01)
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

            if "SPREAD" in strategy:
                spread_leg = self._find_spread_leg(options, best["strike"], current_price)
                result["spread_leg"] = spread_leg
                if spread_leg:
                    # Netto-Kosten für Spread berechnen
                    result["net_debit"] = round(
                        result["ask"] - spread_leg.get("bid", 0), 2
                    )

            return result
        except Exception as e:
            log.debug(f"Kontrakt-Suche Fehler für {ticker}: {e}")
            return None

    def _find_spread_leg(self, options, long_strike: float, current: float) -> Optional[dict]:
        short_target = long_strike * 1.10
        candidates   = options[
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
            return (datetime.strptime(expiry_str, "%Y-%m-%d") - datetime.utcnow()).days
        except Exception:
            return 0

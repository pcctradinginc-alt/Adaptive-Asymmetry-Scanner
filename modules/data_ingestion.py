"""
Stufe 1: Daten-Ingestion & Hard-Filter

Fixes:
  C-04: SP500_SAMPLE ignorierte universe aus config → jetzt universe.py
  H-01: RSS-Matching mit Kurztickern (A, V, MA) produzierte False-Positives.
        Fix: Word-Boundary-Matching mit Regex statt simplem substring-Check.
  H-02: Guard `< 2` in _get_48h_move() war falsch → IndexError bei len==2.
        Fix: Guard auf `< 3` korrigiert.
  M-01: relevant_threshold aus config.yaml war nicht genutzt → jetzt korrekt.
  cfg:  Hard-Filter-Werte und EPS-Drift-Thresholds aus config.yaml.
"""

import os
import re
import logging
import feedparser
import requests
import yfinance as yf
from typing import Any

from modules.config import cfg
from modules.universe import get_universe

log = logging.getLogger(__name__)

RSS_FEEDS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
]

# FIX H-01: Vorcompilierte Regex-Patterns pro Ticker (Word-Boundary-Matching).
_TICKER_PATTERNS: dict[str, re.Pattern] = {}


def _get_ticker_pattern(ticker: str) -> re.Pattern:
    """
    FIX H-01: Word-Boundary-Regex verhindert False-Positive-Matches.
    Beispiel: "MA" matcht nicht auf "market", "demand", "smart".
    """
    if ticker not in _TICKER_PATTERNS:
        escaped = re.escape(ticker)
        _TICKER_PATTERNS[ticker] = re.compile(
            rf"\b{escaped}\b", re.IGNORECASE
        )
    return _TICKER_PATTERNS[ticker]


class DataIngestion:

    def __init__(self, history: dict):
        self.history      = history
        self.news_api_key = os.getenv("NEWS_API_KEY", "")

    def run(self) -> list[dict]:
        universe = get_universe()
        log.info(
            f"Universum '{cfg.filters.universe}': "
            f"{len(universe)} Ticker geladen."
        )

        news_by_ticker = self._fetch_news(universe)
        candidates     = []

        for ticker in universe:
            info = self._get_ticker_info(ticker)
            if info is None:
                continue
            if not self._passes_hard_filter(info):
                continue

            eps_drift = self._compute_eps_drift(ticker, info)
            news      = news_by_ticker.get(ticker, [])
            if not news:
                continue

            candidates.append({
                "ticker":    ticker,
                "info":      info,
                "eps_drift": eps_drift,
                "news":      news,
            })

        return candidates

    # ── News-Fetching ─────────────────────────────────────────────────────────

    def _fetch_news(self, universe: list[str]) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {t: [] for t in universe}

        if self.news_api_key:
            for ticker in universe:
                try:
                    # Quoted query für exakten Ticker-Match
                    url = (
                        "https://newsapi.org/v2/everything"
                        f"?q=%22{ticker}%22"
                        f"&language=en&pageSize=5"
                        f"&apiKey={self.news_api_key}"
                    )
                    resp     = requests.get(url, timeout=10)
                    articles = resp.json().get("articles", [])
                    result[ticker] += [
                        a["title"] for a in articles if a.get("title")
                    ]
                except Exception as e:
                    log.debug(f"NewsAPI Fehler für {ticker}: {e}")

        for feed_url in RSS_FEEDS:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries:
                    title = entry.get("title", "")
                    for ticker in universe:
                        # FIX H-01: Word-Boundary-Regex
                        if _get_ticker_pattern(ticker).search(title):
                            result[ticker].append(title)
            except Exception as e:
                log.debug(f"RSS Fehler ({feed_url}): {e}")

        return result

    # ── Ticker-Info ───────────────────────────────────────────────────────────

    def _get_ticker_info(self, ticker: str) -> dict | None:
        try:
            t    = yf.Ticker(ticker)
            info = t.info
            if not info or "marketCap" not in info:
                return None
            return info
        except Exception as e:
            log.debug(f"yfinance Fehler für {ticker}: {e}")
            return None

    # ── Hard-Filter ───────────────────────────────────────────────────────────

    def _passes_hard_filter(self, info: dict) -> bool:
        market_cap = info.get("marketCap", 0) or 0
        avg_volume = info.get("averageVolume10days", 0) or 0
        if market_cap < cfg.filters.min_market_cap:
            return False
        if avg_volume < cfg.filters.min_avg_volume:
            return False
        return True

    # ── EPS-Drift ─────────────────────────────────────────────────────────────

    def _compute_eps_drift(self, ticker: str, info: dict) -> dict[str, Any]:
        current_eps = info.get("forwardEps") or 0.0
        rec_mean    = info.get("recommendationMean") or 0.0

        stored = self._get_stored_eps(ticker)
        if stored and stored != 0:
            drift = (current_eps - stored) / abs(stored)
        else:
            drift = 0.0

        # FIX M-01: relevant_threshold (0.05) jetzt korrekt genutzt
        abs_drift = abs(drift)
        if abs_drift > cfg.eps_drift.massive_threshold:
            weight = "massive"
        elif abs_drift > cfg.eps_drift.relevant_threshold:
            weight = "relevant"
        else:
            weight = "noise"

        return {
            "current_eps":  current_eps,
            "stored_eps":   stored,
            "drift":        round(drift, 4),
            "drift_weight": weight,
            "rec_mean":     rec_mean,
        }

    def _get_stored_eps(self, ticker: str) -> float | None:
        for trade in self.history.get("active_trades", []):
            if trade.get("ticker") == ticker:
                return trade.get("features", {}).get("eps", None)
        return None

    # FIX H-02: Guard war `< 2` → IndexError bei exakt 2 Datenpunkten
    # Methode wird von deep_analysis.py genutzt, hier als statische Hilfe
    @staticmethod
    def compute_48h_move(ticker: str) -> float:
        """
        Berechnet die 2-Tages-Rendite.
        FIX H-02: Guard `< 3` statt `< 2` verhindert IndexError bei len==2.
        """
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if len(hist) < 3:   # FIX: war < 2
                return 0.0
            close = hist["Close"]
            return float((close.iloc[-1] - close.iloc[-3]) / close.iloc[-3])
        except Exception:
            return 0.0

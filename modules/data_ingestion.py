"""
Stufe 1: Daten-Ingestion & Hard-Filter
- News von NewsAPI / GNews / RSS
- Universum: S&P 500 + Nasdaq 100
- Filter: Market Cap > 2 Mrd, Ø-Volumen > 1 Mio
- EPS-Drift Extraktion aus history.json
"""

import os
import logging
import feedparser
import requests
import yfinance as yf
from typing import Any

log = logging.getLogger(__name__)

# Repräsentative Auswahl (in Produktion: vollständige Indexlisten via API)
SP500_SAMPLE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "JPM",
    "UNH", "V", "XOM", "LLY", "JNJ", "MA", "AVGO", "HD", "MRK",
    "CVX", "ABBV", "COST", "PEP", "KO", "BAC", "PFE", "TMO",
]

RSS_FEEDS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
]


class DataIngestion:
    def __init__(self, history: dict):
        self.history = history
        self.news_api_key = os.getenv("NEWS_API_KEY", "")

    def run(self) -> list[dict]:
        news_by_ticker = self._fetch_news()
        candidates = []
        for ticker in SP500_SAMPLE:
            info = self._get_ticker_info(ticker)
            if info is None:
                continue
            if not self._passes_hard_filter(info):
                continue
            eps_drift = self._compute_eps_drift(ticker, info)
            news = news_by_ticker.get(ticker, [])
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

    def _fetch_news(self) -> dict[str, list[str]]:
        """Sammelt Headlines pro Ticker aus NewsAPI + RSS."""
        result: dict[str, list[str]] = {t: [] for t in SP500_SAMPLE}

        # NewsAPI
        if self.news_api_key:
            for ticker in SP500_SAMPLE:
                try:
                    url = (
                        "https://newsapi.org/v2/everything"
                        f"?q={ticker}&language=en&pageSize=5"
                        f"&apiKey={self.news_api_key}"
                    )
                    resp = requests.get(url, timeout=10)
                    articles = resp.json().get("articles", [])
                    result[ticker] += [a["title"] for a in articles if a.get("title")]
                except Exception as e:
                    log.debug(f"NewsAPI Fehler für {ticker}: {e}")

        # RSS-Feeds (Keyword-Matching)
        for feed_url in RSS_FEEDS:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries:
                    title = entry.get("title", "")
                    for ticker in SP500_SAMPLE:
                        if ticker.lower() in title.lower():
                            result[ticker].append(title)
            except Exception as e:
                log.debug(f"RSS Fehler ({feed_url}): {e}")

        return result

    # ── Ticker-Info ───────────────────────────────────────────────────────────

    def _get_ticker_info(self, ticker: str) -> dict | None:
        try:
            t = yf.Ticker(ticker)
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
        if market_cap < 2_000_000_000:
            return False
        if avg_volume < 1_000_000:
            return False
        return True

    # ── EPS-Drift ─────────────────────────────────────────────────────────────

    def _compute_eps_drift(self, ticker: str, info: dict) -> dict[str, Any]:
        current_eps = info.get("forwardEps") or 0.0
        rec_mean    = info.get("recommendationMean") or 0.0

        # Vergleich mit gespeichertem Wert
        stored = self._get_stored_eps(ticker)
        if stored and stored != 0:
            drift = (current_eps - stored) / abs(stored)
        else:
            drift = 0.0

        if abs(drift) > 0.10:
            weight = "massive"
        elif abs(drift) > 0.05:
            weight = "relevant"
        else:
            weight = "noise"

        return {
            "current_eps":    current_eps,
            "stored_eps":     stored,
            "drift":          round(drift, 4),
            "drift_weight":   weight,
            "rec_mean":       rec_mean,
        }

    def _get_stored_eps(self, ticker: str) -> float | None:
        for trade in self.history.get("active_trades", []):
            if trade.get("ticker") == ticker:
                return trade.get("features", {}).get("eps", None)
        return None

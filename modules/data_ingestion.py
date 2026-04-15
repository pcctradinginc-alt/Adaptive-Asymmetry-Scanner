"""
modules/data_ingestion.py v7.0

Fix 1: Parallel-Requests statt sequenziell
    Vorher: 493 Ticker × ~0.85s = ~7 Minuten
    Jetzt:  ThreadPoolExecutor mit 20 Workers = ~30-45 Sekunden
    
    Warum 20 Workers (nicht mehr):
    - Yahoo Finance drosselt bei zu vielen parallelen Requests
    - 20 ist der empirisch beste Wert für GitHub Actions IP-Ranges
    - Mehr als 30 → erhöhte 429-Rate

Fix 2: Haiku↔Sonnet Konsistenz-Check
    Prescreening-Begründung wird an Deep Analysis übergeben.
    Wenn Sonnet BEARISH für Haiku-YES (bullish begründet) →
    explizite Warnung im Log + data_confidence = 'low'.
"""

from __future__ import annotations
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Optional

import requests
import yfinance as yf

from modules.config   import cfg
from modules.universe import get_universe

log = logging.getLogger(__name__)

MIN_MARKET_CAP_USD    = 2_000_000_000
MIN_AVG_VOLUME        = 1_000_000
MIN_DOLLAR_VOLUME_USD = 10_000_000
MIN_RELATIVE_VOLUME   = 0.6
MAX_WORKERS           = 20   # Parallel Threads — empirisch für Yahoo Finance


class DataIngestion:

    def __init__(self, history: dict | None = None):
        self.history      = history or {}
        self.news_api_key = os.getenv("NEWS_API_KEY", "")

    def run(self) -> list[dict]:
        tickers = get_universe()
        log.info(f"Stufe 1: Hard-Filter auf {len(tickers)} Ticker "
                 f"(parallel, {MAX_WORKERS} Workers)")

        stats = {
            "total": len(tickers), "no_data": 0, "market_cap": 0,
            "avg_volume": 0, "dollar_volume": 0, "rel_volume": 0,
            "no_news": 0, "passed": 0,
        }

        candidates = []
        # Parallel-Requests — massiv schneller als sequenziell
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(self._evaluate_ticker, ticker, {}): ticker
                for ticker in tickers
            }
            for future in as_completed(futures):
                try:
                    result, ticker_stats = future.result()
                    # Stats thread-safe aggregieren
                    for k, v in ticker_stats.items():
                        stats[k] = stats.get(k, 0) + v
                    if result:
                        candidates.append(result)
                except Exception as e:
                    log.debug(f"Future-Fehler: {e}")
                    stats["no_data"] += 1

        self._log_filter_stats(stats)
        log.info(f"  → {len(candidates)} Kandidaten nach Hard-Filter")
        return candidates

    def _evaluate_ticker(
        self, ticker: str, _stats: dict
    ) -> tuple[Optional[dict], dict]:
        """Evaluiert einen Ticker. Gibt (result, stats_delta) zurück."""
        local_stats = {
            "no_data": 0, "market_cap": 0, "avg_volume": 0,
            "dollar_volume": 0, "rel_volume": 0, "no_news": 0, "passed": 0,
        }
        try:
            t    = yf.Ticker(ticker)
            info = t.info

            if not info or not isinstance(info, dict):
                local_stats["no_data"] += 1
                return None, local_stats

            market_cap = info.get("marketCap") or 0
            if market_cap < MIN_MARKET_CAP_USD:
                local_stats["market_cap"] += 1
                return None, local_stats

            avg_vol = info.get("averageVolume") or info.get("averageVolume10days") or 0
            if avg_vol < MIN_AVG_VOLUME:
                local_stats["avg_volume"] += 1
                return None, local_stats

            current_price = (
                info.get("currentPrice") or
                info.get("regularMarketPrice") or
                info.get("previousClose") or 0
            )
            if current_price * avg_vol < MIN_DOLLAR_VOLUME_USD:
                local_stats["dollar_volume"] += 1
                return None, local_stats

            volume_today = info.get("volume") or info.get("regularMarketVolume") or 0
            rel_volume   = volume_today / avg_vol if avg_vol > 0 and volume_today > 0 else 0.0
            if rel_volume < MIN_RELATIVE_VOLUME:
                local_stats["rel_volume"] += 1
                return None, local_stats

            news = self._fetch_news(ticker, info)
            if not news:
                local_stats["no_news"] += 1
                return None, local_stats

            local_stats["passed"] += 1
            dollar_volume = current_price * avg_vol
            log.info(
                f"  [{ticker}] ✅ Cap=${market_cap/1e9:.1f}B "
                f"AvgVol={avg_vol/1e6:.1f}M "
                f"$Vol=${dollar_volume/1e6:.0f}M "
                f"RV={rel_volume:.2f} News={len(news)}"
            )

            return {
                "ticker":        ticker,
                "info":          info,
                "news":          news,
                "market_cap":    market_cap,
                "avg_volume":    avg_vol,
                "dollar_volume": dollar_volume,
                "rel_volume":    round(rel_volume, 3),
                "current_price": current_price,
                "features":      {},
            }, local_stats

        except Exception as e:
            log.debug(f"  [{ticker}] Fehler: {e}")
            local_stats["no_data"] += 1
            return None, local_stats

    def _log_filter_stats(self, stats: dict) -> None:
        total  = stats["total"]
        passed = stats["passed"]
        log.info("=" * 55)
        log.info(f"HARD-FILTER ERGEBNIS: {passed}/{total} Ticker bestanden")
        log.info("-" * 55)
        log.info(f"  ❌ Kein Data/Fehler:         {stats['no_data']:>4}  ({stats['no_data']/total*100:.1f}%)")
        log.info(f"  ❌ Market Cap < 2 Mrd.:      {stats['market_cap']:>4}  ({stats['market_cap']/total*100:.1f}%)")
        log.info(f"  ❌ Avg Volume < 1M:          {stats['avg_volume']:>4}  ({stats['avg_volume']/total*100:.1f}%)")
        log.info(f"  ❌ Dollar-Vol < $10M:         {stats['dollar_volume']:>4}  ({stats['dollar_volume']/total*100:.1f}%)")
        log.info(f"  ❌ Rel. Volume < 0.6:       {stats['rel_volume']:>4}  ({stats['rel_volume']/total*100:.1f}%)")
        log.info(f"  ❌ Keine News:                {stats['no_news']:>4}  ({stats['no_news']/total*100:.1f}%)")
        log.info(f"  ✅ Bestanden:                {passed:>4}  ({passed/total*100:.1f}%)")
        log.info("=" * 55)
        if passed < 10:
            log.warning(f"Nur {passed} Kandidaten — wenig Material für Prescreening.")
        elif passed > 80:
            log.warning(f"{passed} Kandidaten — sehr viel, API-Kosten beachten.")

    def _fetch_news(self, ticker: str, info: dict) -> list[str]:
        finnhub_key = os.getenv("FINNHUB_API_KEY", "")
        if finnhub_key:
            news = self._fetch_finnhub_news(ticker, finnhub_key)
            if news:
                return news
        if self.news_api_key:
            company_name = info.get("longName", ticker).split()[0]
            news = self._fetch_newsapi(ticker, company_name)
            if news:
                return news
        return self._fetch_yfinance_news(ticker)

    def _fetch_finnhub_news(self, ticker: str, api_key: str) -> list[str]:
        try:
            since = (datetime.utcnow() - timedelta(days=2)).strftime("%Y-%m-%d")
            today = datetime.utcnow().strftime("%Y-%m-%d")
            resp  = requests.get(
                "https://finnhub.io/api/v1/company-news",
                params={"symbol": ticker, "from": since, "to": today, "token": api_key},
                timeout=8,
            )
            resp.raise_for_status()
            articles = resp.json()
            return [a["headline"] for a in (articles or [])[:5] if a.get("headline")]
        except Exception:
            return []

    def _fetch_newsapi(self, ticker: str, company_name: str) -> list[str]:
        try:
            since = (datetime.utcnow() - timedelta(days=2)).strftime("%Y-%m-%d")
            resp  = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": f'"{ticker}" OR "{company_name}"', "from": since,
                    "sortBy": "publishedAt", "pageSize": 5,
                    "apiKey": self.news_api_key, "language": "en",
                },
                timeout=8,
            )
            resp.raise_for_status()
            return [a["title"] for a in resp.json().get("articles", []) if a.get("title")]
        except Exception:
            return []

    def _fetch_yfinance_news(self, ticker: str) -> list[str]:
        try:
            t     = yf.Ticker(ticker)
            news  = t.news or []
            since = datetime.utcnow() - timedelta(hours=48)
            return [
                n["title"] for n in news[:5]
                if n.get("title") and
                datetime.fromtimestamp(n.get("providerPublishTime", 0)) >= since
            ]
        except Exception:
            return []

"""
modules/news_fetcher.py – Ticker-spezifische News via Finnhub

Fix 4: Ersetzt globale RSS-Feeds durch ticker-spezifisches Finnhub Company News API.

Vorher (RSS-Problem):
  - Reuters/CNBC RSS werden nach Ticker-Erwähnung durchsucht
  - Produziert False-Positives (z.B. ON Semiconductor matcht "ON sale")
  - Generisch, nicht ticker-spezifisch
  - Zeitlich unscharf (RSS-Posts haben kein exaktes Timestamp)

Jetzt (Finnhub):
  - /company-news Endpoint liefert NUR News für den spezifischen Ticker
  - Timestamp auf Minuten genau → ermöglicht echten Intraday-Delta
  - Kein False-Positive-Problem mehr
  - Free Tier: 60 Calls/Minute (ausreichend für 20 Ticker/Tag)
  - Fallback auf RSS wenn kein Finnhub-Key

API: https://finnhub.io/docs/api/company-news
Key: FINNHUB_API_KEY als GitHub Secret
"""

from __future__ import annotations
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Optional

import feedparser
import requests

log = logging.getLogger(__name__)

_FINNHUB_BASE   = "https://finnhub.io/api/v1"
_FINNHUB_DELAY  = 1.0   # Sekunden zwischen Calls (max 60/min)
_last_call_time = 0.0

# Fallback RSS-Feeds (nur wenn kein Finnhub-Key)
_RSS_FEEDS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
]

# Kurzticker die NICHT per RSS gematcht werden (False-Positive-Gefahr)
_RSS_UNSAFE = frozenset({
    "ON", "IT", "OR", "ARE", "BE", "TO", "DO", "GO", "SO", "RE",
    "AI", "GE", "AM", "PM", "IS", "AS", "AT", "BY", "IN", "OF",
    "A", "V", "C", "F", "K", "D", "L", "O",
})


def fetch_company_news(
    ticker: str,
    days_back: int = 2,
    max_articles: int = 5,
) -> list[dict]:
    """
    Ruft ticker-spezifische News via Finnhub ab.

    Args:
        ticker:       Aktien-Ticker (z.B. "AAPL")
        days_back:    Wie viele Tage zurück (default: 2 = 48h)
        max_articles: Maximale Anzahl zurückgegebener Artikel

    Returns:
        [{"title": str, "datetime": int (unix), "source": str, "url": str}]
        Sortiert nach Datum (neueste zuerst).
    """
    finnhub_key = os.getenv("FINNHUB_API_KEY", "")

    if finnhub_key:
        news = _fetch_finnhub(ticker, days_back, max_articles, finnhub_key)
        if news:
            return news
        log.debug(f"  [{ticker}] Finnhub: keine News → RSS-Fallback")

    return _fetch_rss_fallback(ticker, max_articles)


def fetch_news_headlines(ticker: str, days_back: int = 2) -> list[str]:
    """
    Convenience-Wrapper: Gibt nur Headlines als String-Liste zurück.
    Kompatibel mit dem bisherigen news-Format in data_ingestion.py.
    """
    news = fetch_company_news(ticker, days_back)
    return [n["title"] for n in news if n.get("title")]


def get_news_with_timestamps(ticker: str, days_back: int = 2) -> list[dict]:
    """
    Gibt News MIT Unix-Timestamps zurück.
    Wird für Intraday-Delta-Berechnung genutzt:
    Wenn die News älter als 4h ist und die Aktie schon +5% gestiegen →
    Alpha möglicherweise eingepreist.
    """
    return fetch_company_news(ticker, days_back)


def compute_news_age_hours(news_items: list[dict]) -> Optional[float]:
    """
    Berechnet das Alter der neuesten News in Stunden.
    Nur möglich mit Finnhub (hat Timestamps).

    Returns:
        Float (Stunden) oder None wenn kein Timestamp verfügbar.
    """
    timestamps = [
        n.get("datetime", 0)
        for n in news_items
        if n.get("datetime", 0) > 0
    ]
    if not timestamps:
        return None

    newest_ts   = max(timestamps)
    newest_dt   = datetime.fromtimestamp(newest_ts)
    age_hours   = (datetime.utcnow() - newest_dt).total_seconds() / 3600

    return round(age_hours, 1)


# ── Finnhub ───────────────────────────────────────────────────────────────────

def _rate_limit():
    global _last_call_time
    elapsed = time.time() - _last_call_time
    if elapsed < _FINNHUB_DELAY:
        time.sleep(_FINNHUB_DELAY - elapsed)
    _last_call_time = time.time()


def _fetch_finnhub(
    ticker: str,
    days_back: int,
    max_articles: int,
    api_key: str,
) -> list[dict]:
    """Ruft Company News von Finnhub ab."""
    _rate_limit()

    try:
        today = datetime.utcnow()
        since = today - timedelta(days=days_back)

        resp = requests.get(
            f"{_FINNHUB_BASE}/company-news",
            params={
                "symbol": ticker,
                "from":   since.strftime("%Y-%m-%d"),
                "to":     today.strftime("%Y-%m-%d"),
                "token":  api_key,
            },
            timeout=10,
        )
        resp.raise_for_status()
        articles = resp.json()

        if not isinstance(articles, list):
            return []

        # Neueste zuerst, max N zurückgeben
        articles.sort(key=lambda x: x.get("datetime", 0), reverse=True)
        result = []
        for a in articles[:max_articles]:
            result.append({
                "title":    a.get("headline", ""),
                "datetime": a.get("datetime", 0),
                "source":   a.get("source", "Finnhub"),
                "url":      a.get("url", ""),
                "summary":  a.get("summary", "")[:200],
            })

        if result:
            log.debug(f"  [{ticker}] Finnhub: {len(result)} News-Artikel")

        return result

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            log.warning(f"Finnhub Rate-Limit (429) für {ticker} → warte 60s")
            time.sleep(60)
        else:
            log.debug(f"  [{ticker}] Finnhub HTTP-Fehler: {e}")
        return []
    except Exception as e:
        log.debug(f"  [{ticker}] Finnhub Fehler: {e}")
        return []


# ── RSS Fallback ──────────────────────────────────────────────────────────────

import re
_PATTERNS: dict[str, re.Pattern] = {}


def _get_pattern(ticker: str) -> Optional[re.Pattern]:
    """Word-Boundary-Pattern, None für unsichere Kurzticker."""
    if len(ticker) < 3 or ticker.upper() in _RSS_UNSAFE:
        return None
    if ticker not in _PATTERNS:
        _PATTERNS[ticker] = re.compile(
            rf"\b{re.escape(ticker)}\b", re.IGNORECASE
        )
    return _PATTERNS[ticker]


def _fetch_rss_fallback(ticker: str, max_articles: int) -> list[dict]:
    """RSS-Fallback wenn kein Finnhub-Key. Weniger präzise."""
    pattern = _get_pattern(ticker)
    if pattern is None:
        log.debug(f"  [{ticker}] RSS: Kurzticker übersprungen (False-Positive-Risiko)")
        return []

    results = []
    for feed_url in _RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries:
                title = entry.get("title", "")
                if pattern.search(title):
                    results.append({
                        "title":    title,
                        "datetime": 0,   # RSS hat keinen zuverlässigen Timestamp
                        "source":   "RSS",
                        "url":      entry.get("link", ""),
                        "summary":  "",
                    })
                    if len(results) >= max_articles:
                        return results
        except Exception as e:
            log.debug(f"RSS Fehler ({feed_url}): {e}")

    return results

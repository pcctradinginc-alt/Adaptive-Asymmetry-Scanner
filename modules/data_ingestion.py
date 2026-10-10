"""
modules/data_ingestion.py v7.2

v7.2: Short-Interest-Bugfix + Momentum + Features-Anbindung
    Bug: shortRatio (Days-to-Cover, z.B. 2.28 Tage) wurde als Fallback für
    shortPercentOfFloat verwendet und fälschlich als Prozent-vom-Float
    interpretiert (Einheiten-Mix). Fix: shortPercentOfFloat ist jetzt die
    EINZIGE Quelle für short_float; shortRatio wird separat und ohne
    Umrechnung als short_ratio_days erfasst.
    Neu: short_mom (Short-Interest-Momentum ggü. Vormonat) — steigendes
    Short Interest vor positiver News ist die eigentliche Squeeze-Alpha-
    Hypothese.
    Neu: short_pct_float / short_mom fließen jetzt in candidate["features"]
    (nur wenn tatsächlich vorhanden, kein 0.0-Default) — damit landen sie
    in outputs/history.json bei Trades/Schatten-Trades und sind
    backtestbar.
    Die reine Berechnung steckt jetzt in der modulweiten Hilfsfunktion
    extract_short_interest(info) — netzwerkfrei, testbar ohne Mocking von
    yfinance.

v7.1: Short Interest als Feature
    yfinance liefert shortPercentOfFloat bereits im info-Dict.
    Kein zusätzlicher API-Call nötig.
    Short Float > 15% → asymmetrisches Upside bei positiver News (Squeeze-Potential).
    Wird als candidate["short_interest"] gespeichert.

v7.0:
Fix 1: Parallel-Requests statt sequenziell
    Vorher: 493 Ticker × ~0.85s = ~7 Minuten
    Jetzt:  ThreadPoolExecutor mit 20 Workers = ~30-45 Sekunden

Fix 2: Haiku↔Sonnet Konsistenz-Check
    Prescreening-Begründung wird an Deep Analysis übergeben.
"""

from __future__ import annotations
import logging
import os
import random
import threading
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
RV_BASE_THRESHOLD     = 0.6   # Basis-Schwelle (skaliert mit VIX dynamisch)
MAX_WORKERS           = 20   # Parallel Threads — empirisch für Yahoo Finance
RV_NEWS_OVERRIDE_MIN  = 0.25  # News-Override greift nur ab diesem RV (siehe _evaluate_ticker)

# Finnhub Free-Tier: 60 Aufrufe/Minute. 20 parallele Threads ohne Drossel
# liefen nach ~60 Tickern in HTTP 429, das still als "keine News" galt ->
# der Hard-Filter ließ fast nur Ticker A–E durch (Log 2026-09-25: 116/126).
FINNHUB_CALLS_PER_MIN = 55
FINNHUB_MAX_429_RETRIES = 3


class _RateLimiter:
    """Thread-sicherer Sliding-Window-Limiter (max_calls je period Sekunden)."""

    def __init__(self, max_calls: int, period: float = 60.0,
                 clock=time.monotonic, sleep=time.sleep):
        self.max_calls, self.period = max_calls, period
        self._clock, self._sleep = clock, sleep
        self._calls: list[float] = []
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = self._clock()
                self._calls = [t for t in self._calls if now - t < self.period]
                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return
                wait = self.period - (now - self._calls[0])
            self._sleep(max(wait, 0.05))


_FINNHUB_LIMITER = _RateLimiter(FINNHUB_CALLS_PER_MIN, 60.0)
_STATS_LOCK = threading.Lock()


_CORP_SUFFIXES = {"inc", "inc.", "corp", "corp.", "corporation", "co", "co.", "company", "ltd", "ltd.",
                  "plc", "n.v.", "s.a.", "ag", "se", "holdings", "group", "the", "&", "incorporated", "llc",
                  "l.p.", "lp"}


def newsapi_company_name(long_name: str) -> str | None:
    """Vollständiger Firmenname ohne Rechtsform-Suffixe für die NewsAPI-Phrase.
    Vorher: erstes Wort ("Bank" of America, "The" Home Depot) -> fremde Artikel
    (Lauf 2026-09-28: BAC-News handelten von IonQ, Red-Team-Veto)."""
    words = [w for w in str(long_name).replace(",", " ").split()]
    while words and words[-1].lower() in _CORP_SUFFIXES:
        words.pop()
    while words and words[0].lower() == "the":
        words.pop(0)
    name = " ".join(words).strip()
    return name if len(name) >= 4 else None


def parse_yfinance_news(items: list, since: datetime, limit: int = 5) -> list[str]:
    """Headlines der letzten Zeit aus yfinance .news -- altes Format
    (title/providerPublishTime) UND neues Format ab yfinance 0.2.5x
    (content.title/content.pubDate ISO-8601). Vorher las der Fallback nur das
    alte Format und lieferte mit aktuellem yfinance IMMER eine leere Liste."""
    out = []
    for n in (items or [])[:limit]:
        if not isinstance(n, dict):
            continue
        content = n.get("content") if isinstance(n.get("content"), dict) else {}
        title = n.get("title") or content.get("title")
        published = None
        if n.get("providerPublishTime"):
            try:
                published = datetime.utcfromtimestamp(int(n["providerPublishTime"]))
            except (TypeError, ValueError, OSError):
                published = None
        elif content.get("pubDate") or content.get("displayTime"):
            raw = str(content.get("pubDate") or content.get("displayTime"))
            try:
                dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                published = dt.replace(tzinfo=None) - (dt.utcoffset() or timedelta(0))
            except ValueError:
                published = None
        if title and published is not None and published >= since:
            out.append(title)
    return out

# v7.1: Short Interest Schwellenwerte
SHORT_INTEREST_HIGH   = 0.15   # 15% Short Float → "high" (Squeeze-Potential)
SHORT_INTEREST_MED    = 0.08   # 8% Short Float → "elevated"


def _safe_float(val) -> Optional[float]:
    """Wandelt val defensiv in float um — None/leer/Müll-Strings → None, nie Exception."""
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def extract_short_interest(info: dict | None) -> dict:
    """
    Reine, netzwerkfreie Hilfsfunktion — extrahiert Short-Interest-Kennzahlen
    aus dem yfinance info-Dict. Wird sowohl von _evaluate_ticker() als auch
    direkt von Tests aufgerufen (kein yfinance-Mocking nötig).

    Wichtig (v7.2-Fix): shortPercentOfFloat ist die EINZIGE Quelle für
    short_float. shortRatio ist "Days to Cover" (Tage, bis Shorts bei
    aktuellem Handelsvolumen gedeckt wären) — eine völlig andere Einheit
    als Prozent-vom-Float! Er wird separat und OHNE Umrechnung als
    short_ratio_days erfasst und fließt nicht in short_float ein.

    short_mom (Short-Interest-Momentum) = sharesShort / sharesShortPriorMonth - 1.
    Nur gesetzt, wenn beide Werte vorhanden und der Vormonatswert > 0 ist
    (sonst None). Steigendes Short Interest vor positiver News ist die
    Squeeze-Alpha-Hypothese dieses Features.

    Defensiv: yfinance-info-Felder können fehlen, None sein oder als
    Müll-String vorliegen — diese Funktion wirft niemals eine Exception.

    Rückgabe:
        {
            "short_float":      float,        # 0.0 wenn unbekannt
            "label":            str,           # "high" | "elevated" | "normal"
            "short_ratio_days": float | None,  # Days-to-Cover, unverändert
            "short_mom":        float | None,  # Momentum ggü. Vormonat
            "features": {                      # nur tatsächlich vorhandene Werte
                "short_pct_float": float,       # nur falls shortPercentOfFloat vorhanden
                "short_mom":       float,       # nur falls berechenbar
            },
        }
    """
    info = info or {}

    # ── short_float: NUR shortPercentOfFloat, keine Vermischung mit shortRatio ──
    raw_short_pct    = _safe_float(info.get("shortPercentOfFloat"))
    short_float      = 0.0
    have_short_float = raw_short_pct is not None
    if have_short_float:
        short_float = raw_short_pct
        # yfinance liefert manchmal als Ratio (0.15) oder als Prozent (15.0)
        if short_float > 1.0:
            short_float = short_float / 100.0   # 15.0 → 0.15
    short_float = round(short_float, 4)

    # ── short_ratio_days: shortRatio (Days-to-Cover) — OHNE Umrechnung ──
    short_ratio_days = _safe_float(info.get("shortRatio"))

    # ── short_mom: Short-Interest-Momentum ggü. Vormonat ──
    shares_short       = _safe_float(info.get("sharesShort"))
    shares_short_prior = _safe_float(info.get("sharesShortPriorMonth"))
    short_mom = None
    if shares_short is not None and shares_short_prior is not None and shares_short_prior > 0:
        short_mom = round(shares_short / shares_short_prior - 1.0, 4)

    # ── Label (unverändert: high/elevated/normal) ──
    if short_float >= SHORT_INTEREST_HIGH:
        label = "high"
    elif short_float >= SHORT_INTEREST_MED:
        label = "elevated"
    else:
        label = "normal"

    # ── Features fürs Backtesting: nur setzen, wenn Wert wirklich vorhanden ──
    # (kein 0.0-Default — sonst würde "unbekannt" später als "kein Short
    # Interest" fehlinterpretiert)
    features = {}
    if have_short_float:
        features["short_pct_float"] = short_float
    if short_mom is not None:
        features["short_mom"] = short_mom

    return {
        "short_float":      short_float,
        "label":            label,
        "short_ratio_days": short_ratio_days,
        "short_mom":        short_mom,
        "features":         features,
    }


class DataIngestion:

    def __init__(self, history: dict | None = None):
        self.history      = history or {}
        self.news_api_key = os.getenv("NEWS_API_KEY", "")

    def _get_current_vix(self) -> float:
        """VIX einmal holen — nicht pro Ticker wiederholen."""
        try:
            import yfinance as _yf
            val = _yf.Ticker("^VIX").fast_info.last_price
            if not val:
                log.warning("VIX nicht verfügbar → RV-Schwelle mit VIX=20 (neutral) berechnet")
            return float(val or 20.0)
        except Exception as e:
            log.warning(f"VIX-Abruf Fehler: {e} → RV-Schwelle mit VIX=20 (neutral) berechnet")
            return 20.0

    def run(self) -> list[dict]:
        tickers = list(get_universe())
        # Rohliste VOR dem Hard-Filter (pipeline: stats["universe"]); vorher fehlte das Attribut und
        # "Ticker im Universum" zeigte dieselbe Zahl wie "nach Hard-Filter" (Maintenance 2026-10-09).
        self.universe_size = len(tickers)
        # Reihenfolge je Tag deterministisch mischen: sollte eine Quelle doch
        # ins Limit laufen, trifft es nie systematisch dieselben (alphabetisch
        # späten) Ticker.
        random.Random(datetime.utcnow().strftime("%Y-%m-%d")).shuffle(tickers)
        log.info(f"Stufe 1: Hard-Filter auf {len(tickers)} Ticker "
                 f"(parallel, {MAX_WORKERS} Workers)")

        vix_current = self._get_current_vix()
        log.info(f"  RV-Filter: VIX={vix_current:.1f} → Schwelle={0.6 * max(0.5, min(1.5, vix_current/20.0)):.3f}")

        # Crumb-Fix: yfinance-Session vor parallelen Requests einmalig warmlaufen lassen,
        # damit alle Threads denselben gültigen Crumb verwenden.
        try:
            yf.Ticker("SPY").fast_info.last_price
        except Exception:
            pass

        stats = {
            "total": len(tickers), "no_data": 0, "market_cap": 0,
            "avg_volume": 0, "dollar_volume": 0, "rel_volume": 0,
            "no_news": 0, "passed": 0,
        }
        self.news_source_stats = {"finnhub_ok": 0, "finnhub_empty": 0, "finnhub_error": 0,
                                  "finnhub_429": 0, "newsapi_ok": 0, "newsapi_error": 0,
                                  "yfinance_ok": 0, "yfinance_empty": 0}

        candidates = []
        # Parallel-Requests — massiv schneller als sequenziell
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(self._evaluate_ticker, ticker, {}, vix_current): ticker
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
        ns = self.news_source_stats
        log.info(f"  News-Quellen: {ns}")
        if ns["finnhub_error"] or ns["finnhub_429"] or ns["newsapi_error"]:
            log.warning(f"  News-Abruf mit Fehlern (Finnhub 429={ns['finnhub_429']}, "
                        f"Finnhub Fehler={ns['finnhub_error']}, NewsAPI Fehler={ns['newsapi_error']}) "
                        f"-> 'Keine News'-Rejects können Datenlücken statt fehlender News sein")
        passed_initials = sorted({c["ticker"][0] for c in candidates if c.get("ticker")})
        log.info(f"  Anfangsbuchstaben der Kandidaten: {''.join(passed_initials)}")
        return candidates

    def _evaluate_ticker(
        self, ticker: str, _stats: dict, vix_current: float = 20.0
    ) -> tuple[Optional[dict], dict]:
        """Evaluiert einen Ticker. Gibt (result, stats_delta) zurück."""
        local_stats = {
            "no_data": 0, "market_cap": 0, "avg_volume": 0,
            "dollar_volume": 0, "rel_volume": 0, "no_news": 0, "passed": 0,
        }
        try:
            # Retry bei 401 (yfinance Crumb-Invalidierung durch parallele Requests)
            info = None
            for _attempt in range(3):
                try:
                    t    = yf.Ticker(ticker)
                    info = t.info
                    if info and isinstance(info, dict):
                        break
                except Exception as _e:
                    if _attempt < 2:
                        time.sleep(0.5 * (2 ** _attempt))   # 0.5s, 1.0s
                    else:
                        raise

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
            # Dynamischer RV-Schwellenwert: skaliert mit VIX (einmal in run() geholt)
            vix_avg = 20.0   # historischer Durchschnitt
            # Schwelle sinkt bei ruhigem Markt, steigt bei Panik
            rv_threshold = RV_BASE_THRESHOLD * max(0.5, min(1.5, vix_current / vix_avg))
            rv_threshold = round(rv_threshold, 2)

            # News-Override: Ticker mit starker News-Aktivität trotz niedrigem RV erlauben.
            # Unterhalb RV_NEWS_OVERRIDE_MIN kann auch der Override nicht greifen ->
            # verwerfen OHNE News-Abruf (spart das knappe Finnhub-Budget).
            if rel_volume < rv_threshold and rel_volume < RV_NEWS_OVERRIDE_MIN:
                local_stats["rel_volume"] += 1
                return None, local_stats
            news = self._fetch_news(ticker, info)
            news_count   = len(news) if news else 0
            news_override = (news_count >= 3 and rel_volume >= RV_NEWS_OVERRIDE_MIN)

            if rel_volume < rv_threshold and not news_override:
                local_stats["rel_volume"] += 1
                return None, local_stats

            if not news:
                local_stats["no_news"] += 1
                return None, local_stats

            # ── v7.1/v7.2: Short Interest extrahieren (kostenlos, bereits in info) ─
            short_info  = extract_short_interest(info)
            short_float = short_info["short_float"]
            short_label = short_info["label"]

            local_stats["passed"] += 1
            dollar_volume = current_price * avg_vol
            log.info(
                f"  [{ticker}] ✅ Cap=${market_cap/1e9:.1f}B "
                f"AvgVol={avg_vol/1e6:.1f}M "
                f"$Vol=${dollar_volume/1e6:.0f}M "
                f"RV={rel_volume:.2f} News={len(news)}"
                f"{f' Short={short_float:.0%}({short_label})' if short_float >= SHORT_INTEREST_MED else ''}"
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
                # v7.2: Short-Interest-Features fürs Backtesting (nur wenn vorhanden,
                # kein 0.0-Default — siehe extract_short_interest())
                "features":      dict(short_info["features"]),
                # v7.1/v7.2: Short Interest (Logging/Reports)
                "short_interest": {
                    "short_float_pct":  short_float,
                    "label":            short_label,
                    "short_ratio_days": short_info["short_ratio_days"],
                    "short_mom":        short_info["short_mom"],
                },
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
        log.info(f"  ❌ Rel. Volume < Schwelle:  {stats['rel_volume']:>4}  ({stats['rel_volume']/total*100:.1f}%)")
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
            company_name = newsapi_company_name(info.get("longName") or info.get("shortName") or "")
            news = self._fetch_newsapi(ticker, company_name)
            if news:
                return news
        return self._fetch_yfinance_news(ticker)

    def _count(self, key: str) -> None:
        st = getattr(self, "news_source_stats", None)
        if st is not None:
            with _STATS_LOCK:
                st[key] = st.get(key, 0) + 1

    def _fetch_finnhub_news(self, ticker: str, api_key: str) -> list[str]:
        since = (datetime.utcnow() - timedelta(days=2)).strftime("%Y-%m-%d")
        today = datetime.utcnow().strftime("%Y-%m-%d")
        for attempt in range(FINNHUB_MAX_429_RETRIES + 1):
            try:
                _FINNHUB_LIMITER.acquire()
                resp = requests.get(
                    "https://finnhub.io/api/v1/company-news",
                    params={"symbol": ticker, "from": since, "to": today, "token": api_key},
                    timeout=8,
                )
                if resp.status_code == 429:
                    self._count("finnhub_429")
                    if attempt < FINNHUB_MAX_429_RETRIES:
                        try:
                            wait = float(resp.headers.get("Retry-After") or 0)
                        except ValueError:
                            wait = 0.0
                        time.sleep(min(max(wait, 2.0 * (attempt + 1)), 20.0))
                        continue
                    self._count("finnhub_error")
                    return []
                resp.raise_for_status()
                articles = resp.json()
                out = [a["headline"] for a in (articles or [])[:5] if a.get("headline")]
                self._count("finnhub_ok" if out else "finnhub_empty")
                return out
            except Exception:
                self._count("finnhub_error")
                return []
        return []

    def _fetch_newsapi(self, ticker: str, company_name: str) -> list[str]:
        try:
            since = (datetime.utcnow() - timedelta(days=2)).strftime("%Y-%m-%d")
            resp  = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": f'"{company_name}"' if company_name else f'"{ticker}"', "from": since,
                    "sortBy": "publishedAt", "pageSize": 5,
                    "apiKey": self.news_api_key, "language": "en",
                },
                timeout=8,
            )
            resp.raise_for_status()
            out = [a["title"] for a in resp.json().get("articles", []) if a.get("title")]
            if out:
                self._count("newsapi_ok")
            return out
        except Exception:
            self._count("newsapi_error")
            return []

    def _fetch_yfinance_news(self, ticker: str) -> list[str]:
        try:
            news = yf.Ticker(ticker).news or []
            out = parse_yfinance_news(news, datetime.utcnow() - timedelta(hours=48))
            self._count("yfinance_ok" if out else "yfinance_empty")
            return out
        except Exception:
            self._count("yfinance_empty")
            return []

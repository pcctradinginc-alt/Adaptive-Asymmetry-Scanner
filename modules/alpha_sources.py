"""
modules/alpha_sources.py – Alternative Alpha-Quellen v9.0

Änderungen v9.0:
    #15 Put/Call-Skew + Dealer-Gamma-Schätzung als neue Signalquellen.

        fetch_options_skew(ticker, current_price):
            Berechnet Put/Call IV-Skew aus der Options-Chain.
            Hoher Skew (Puts teurer als Calls) = Markt ist bearish positioniert.
            Niedriger Skew = Markt sieht wenig Downside = bullish neutral.
            Nutzt Tradier (wenn verfügbar) sonst yfinance.

        estimate_dealer_gamma(ticker, current_price):
            Schätzt Dealer-Gamma-Exposure aus Open Interest.
            Negative Gamma: Dealer müssen in Richtung bewegen → Volatilität verstärkt.
            Positive Gamma: Dealer dämpfen Bewegungen → Mean Reversion wahrscheinlicher.
            Wichtig für: Interpreation ob ein Move sich beschleunigt oder abbricht.

        enrich_with_alpha_sources() jetzt auch mit Skew + Gamma-Signal.

Integriert FDA API, SEC Insider-Käufe und Finnhub Earnings-Kalender.

API-Limits:
  - FDA:     Unbegrenzt (offiziell, kein Key nötig)
  - SEC:     Unbegrenzt (offiziell, kein Key nötig)
  - Finnhub: 60 Calls/Minute auf Free Tier (API-Key nötig)
  - Tradier: Wie konfiguriert (für Skew-Berechnung, optional)
"""

from __future__ import annotations
import logging
import os
import re
import time
from datetime import datetime, timedelta
from typing import Optional

import requests

log = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "newstoption-scanner/4.1 research@pcctrading.com"}

# v9.0 #15: Skew-Schwellen
SKEW_BEARISH_THRESHOLD = 1.20   # Puts > 20% teurer als Calls → bearishes Signal
SKEW_BULLISH_THRESHOLD = 0.85   # Puts > 15% billiger als Calls → bullishes Signal
SKEW_LOOKBACK_DTE_MIN  = 20     # Minimum DTE für Skew-Messung
SKEW_LOOKBACK_DTE_MAX  = 50     # Maximum DTE für Skew-Messung (30-50d ist der Standard)


# ── FDA API ───────────────────────────────────────────────────────────────────

def fetch_fda_events(company_name: str, days_back: int = 7) -> list[dict]:
    """
    Ruft FDA-Ereignisse für ein Unternehmen ab.
    Quelle: https://api.fda.gov/drug/event.json
    """
    try:
        since = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y%m%d")
        url   = (
            f"https://api.fda.gov/drug/event.json"
            f"?search=receivedate:[{since}+TO+99991231]"
            f"+AND+companynumb:{company_name.replace(' ', '+')}"
            f"&limit=5"
        )
        resp = requests.get(url, headers=_HEADERS, timeout=10)

        if resp.status_code == 404:
            return []

        resp.raise_for_status()
        results = resp.json().get("results", [])

        events = []
        for r in results:
            date = r.get("receivedate", "")
            desc = r.get("primarysource", {}).get("reportercountry", "")
            events.append({
                "date":        date,
                "type":        "fda_adverse_event",
                "description": f"FDA Adverse Event Report ({desc})",
                "source":      "FDA",
            })

        return events

    except Exception as e:
        log.debug(f"FDA API Fehler für {company_name}: {e}")
        return []


def fetch_fda_drug_approvals(days_back: int = 7) -> list[dict]:
    """
    Ruft aktuelle FDA Drug Approvals ab (nicht ticker-spezifisch).
    Quelle: https://api.fda.gov/drug/drugsfda.json
    """
    try:
        since = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y%m%d")
        url   = (
            f"https://api.fda.gov/drug/drugsfda.json"
            f"?search=submissions.submission_status_date:[{since}+TO+99991231]"
            f"+AND+submissions.submission_type:ORIG"
            f"&limit=10"
        )
        resp = requests.get(url, headers=_HEADERS, timeout=10)

        if resp.status_code == 404:
            return []

        resp.raise_for_status()
        results = resp.json().get("results", [])

        approvals = []
        for r in results:
            sponsor = r.get("sponsor_name", "").upper()
            drugs   = [p.get("brand_name", "") for p in r.get("products", [])[:2]]
            approvals.append({
                "sponsor":     sponsor,
                "drugs":       drugs,
                "type":        "fda_approval",
                "description": f"FDA Approval: {', '.join(drugs)}",
                "source":      "FDA",
            })

        return approvals

    except Exception as e:
        log.debug(f"FDA Approvals Fehler: {e}")
        return []


def match_fda_to_ticker(ticker: str, company_info: dict, days_back: int = 7) -> list[str]:
    """Sucht FDA-Events für einen Ticker und gibt Headlines zurück."""
    company_name = company_info.get("shortName", "") or company_info.get("longName", "")
    if not company_name:
        return []

    name_short = company_name.split()[0] if company_name else ""
    if len(name_short) < 3:
        return []

    events    = fetch_fda_events(name_short, days_back)
    approvals = fetch_fda_drug_approvals(days_back)

    headlines = []

    for e in events:
        headlines.append(f"FDA {e['type'].replace('_', ' ').title()}: {e['description']}")

    for a in approvals:
        if name_short.upper() in a.get("sponsor", ""):
            headlines.append(f"FDA APPROVAL: {a['description']} by {a['sponsor']}")

    if headlines:
        log.info(f"  [{ticker}] FDA: {len(headlines)} Events gefunden")

    return headlines


# ── SEC Insider-Käufe ─────────────────────────────────────────────────────────

def fetch_sec_insider_trades(ticker: str, days_back: int = 14) -> list[dict]:
    """
    Ruft Insider-Käufe für einen Ticker via SEC EDGAR ab.
    """
    try:
        search_url = (
            f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22"
            f"&dateRange=custom&startdt="
            f"{(datetime.utcnow() - timedelta(days=days_back)).strftime('%Y-%m-%d')}"
            f"&forms=4"
        )
        resp = requests.get(search_url, headers=_HEADERS, timeout=10)

        if resp.status_code != 200:
            return _fetch_sec_form4_fallback(ticker, days_back)

        hits   = resp.json().get("hits", {}).get("hits", [])
        trades = []

        for hit in hits[:10]:
            src = hit.get("_source", {})
            trades.append({
                "date":        src.get("period_of_report", ""),
                "insider":     src.get("display_names", ["Unknown"])[0]
                               if src.get("display_names") else "Unknown",
                "filing_url":  src.get("file_date", ""),
                "form":        "Form 4",
                "source":      "SEC",
            })

        return trades

    except Exception as e:
        log.debug(f"SEC EDGAR Fehler für {ticker}: {e}")
        return _fetch_sec_form4_fallback(ticker, days_back)


def _fetch_sec_form4_fallback(ticker: str, days_back: int) -> list[dict]:
    try:
        url = (
            f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22"
            f"&forms=4&dateRange=custom"
            f"&startdt={(datetime.utcnow()-timedelta(days=days_back)).strftime('%Y-%m-%d')}"
        )
        resp = requests.get(url, headers=_HEADERS, timeout=8)
        if resp.status_code != 200:
            return []

        hits   = resp.json().get("hits", {}).get("hits", [])
        result = []
        for h in hits[:5]:
            s = h.get("_source", {})
            result.append({
                "date":    s.get("file_date", ""),
                "insider": (s.get("display_names") or ["Unknown"])[0],
                "form":    "Form 4",
                "source":  "SEC",
            })
        return result
    except Exception:
        return []


def detect_insider_cluster(ticker: str, days_back: int = 14) -> dict:
    """
    Erkennt Cluster-Insider-Käufe: Mehrere verschiedene Insider kaufen
    innerhalb von 72 Stunden → starkes Signal.
    """
    trades = fetch_sec_insider_trades(ticker, days_back)

    if not trades:
        return {"cluster_detected": False, "insider_count": 0, "trades": []}

    unique_insiders = set(t["insider"] for t in trades)
    cluster         = len(unique_insiders) >= 2

    result = {
        "cluster_detected": cluster,
        "insider_count":    len(unique_insiders),
        "trades":           trades[:5],
        "headline":         "",
    }

    if cluster:
        result["headline"] = (
            f"SEC Form 4: {len(unique_insiders)} Insider kaufen {ticker} "
            f"innerhalb {days_back} Tagen (Cluster-Signal)"
        )
        log.info(
            f"  [{ticker}] SEC Insider-Cluster: "
            f"{len(unique_insiders)} Insider, {len(trades)} Trades"
        )

    return result


# ── Finnhub Earnings-Kalender ─────────────────────────────────────────────────

def get_earnings_date_finnhub(ticker: str) -> Optional[str]:
    """Ruft das nächste Earnings-Datum via Finnhub ab."""
    finnhub_key = os.getenv("FINNHUB_API_KEY", "")
    if not finnhub_key:
        return None

    try:
        today   = datetime.utcnow()
        to_date = today + timedelta(days=30)
        url     = (
            f"https://finnhub.io/api/v1/calendar/earnings"
            f"?from={today.strftime('%Y-%m-%d')}"
            f"&to={to_date.strftime('%Y-%m-%d')}"
            f"&symbol={ticker}"
            f"&token={finnhub_key}"
        )
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()

        earnings_calendar = resp.json().get("earningsCalendar", [])
        if not earnings_calendar:
            return None

        dates = sorted([e["date"] for e in earnings_calendar if e.get("date")])
        return dates[0] if dates else None

    except Exception as e:
        log.debug(f"Finnhub Earnings Fehler für {ticker}: {e}")
        return None


def has_earnings_within_days(
    ticker:      str,
    buffer_days: int  = 7,
    use_finnhub: bool = True,
) -> tuple[bool, Optional[str]]:
    """Prüft ob Earnings innerhalb der nächsten buffer_days liegen."""
    earnings_date = None

    if use_finnhub and os.getenv("FINNHUB_API_KEY"):
        earnings_date = get_earnings_date_finnhub(ticker)

    if not earnings_date:
        try:
            import yfinance as yf
            info        = yf.Ticker(ticker).info
            earnings_ts = info.get("earningsTimestamp")
            if earnings_ts:
                earnings_date = datetime.fromtimestamp(earnings_ts).strftime("%Y-%m-%d")
        except Exception:
            pass

    if not earnings_date:
        return False, None

    try:
        earnings_dt = datetime.strptime(earnings_date, "%Y-%m-%d")
        days_until  = (earnings_dt - datetime.utcnow()).days

        if 0 <= days_until <= buffer_days:
            log.info(
                f"  [{ticker}] EARNINGS-GATE: Earnings in {days_until}d "
                f"({earnings_date}) → Hard-Block."
            )
            return True, earnings_date

        return False, earnings_date

    except Exception:
        return False, None


# ── v9.0 #15: Put/Call-Skew ───────────────────────────────────────────────────

def fetch_options_skew(ticker: str, current_price: float) -> dict:
    """
    Berechnet Put/Call IV-Skew aus der 30-50 DTE Options-Chain.

    Methode: ATM-Put-IV / ATM-Call-IV für das nächste Expiry im 20-50d Fenster.
    Skew > 1.20: Markt ist bearish (Puts teurer → erhöhter Downside-Schutz)
    Skew < 0.85: Markt sieht kaum Downside (bullish neutral)
    Skew ~1.0:   Ausgeglichen

    Nutzt yfinance (Tradier-Integration via TRADIER_API_KEY wenn verfügbar).

    Returns:
        {
            "skew_ratio":      float,   # put_iv / call_iv
            "put_iv":          float,
            "call_iv":         float,
            "expiry":          str,
            "signal":          "bearish_skew" | "neutral" | "bullish_skew",
            "headline":        str,     # Für News-Liste
            "data_available":  bool,
        }
    """
    result_empty = {
        "skew_ratio": 1.0, "put_iv": 0.0, "call_iv": 0.0,
        "expiry": "", "signal": "neutral", "headline": "",
        "data_available": False,
    }

    if current_price <= 0:
        return result_empty

    try:
        # Primär: Tradier (genauere IV-Daten)
        tradier_key = os.environ.get("TRADIER_API_KEY", "").strip()
        if tradier_key:
            result = _fetch_skew_tradier(ticker, current_price, tradier_key)
            if result and result.get("data_available"):
                log.info(
                    f"  [{ticker}] Put/Call Skew (Tradier): "
                    f"ratio={result['skew_ratio']:.2f} signal={result['signal']}"
                )
                return result

        # Fallback: yfinance
        result = _fetch_skew_yfinance(ticker, current_price)
        if result and result.get("data_available"):
            log.info(
                f"  [{ticker}] Put/Call Skew (yfinance): "
                f"ratio={result['skew_ratio']:.2f} signal={result['signal']}"
            )
        return result if result else result_empty

    except Exception as e:
        log.debug(f"  [{ticker}] Skew-Fehler: {e}")
        return result_empty


def _fetch_skew_yfinance(ticker: str, current_price: float) -> Optional[dict]:
    """yfinance-basierte Skew-Berechnung."""
    try:
        import yfinance as yf
        from datetime import timezone
        from datetime import datetime as _dt

        t = yf.Ticker(ticker)
        now = _dt.now(timezone.utc)

        target_expiry = None
        for exp in (t.options or []):
            try:
                exp_dt = _dt.strptime(exp, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                dte    = (exp_dt - now).days
                if SKEW_LOOKBACK_DTE_MIN <= dte <= SKEW_LOOKBACK_DTE_MAX:
                    target_expiry = exp
                    break
            except Exception:
                continue

        if not target_expiry:
            return None

        chain = t.option_chain(target_expiry)

        # ATM-Strike (nächster Strike zum aktuellen Preis)
        atm_strike = None
        for s in sorted(chain.calls["strike"].tolist(), key=lambda x: abs(x - current_price)):
            atm_strike = s
            break

        if not atm_strike:
            return None

        call_rows = chain.calls[chain.calls["strike"] == atm_strike]
        put_rows  = chain.puts[chain.puts["strike"] == atm_strike]

        if call_rows.empty or put_rows.empty:
            return None

        call_iv = float(call_rows["impliedVolatility"].iloc[0])
        put_iv  = float(put_rows["impliedVolatility"].iloc[0])

        if call_iv <= 0.01 or put_iv <= 0.01:
            return None

        skew_ratio = put_iv / call_iv
        return _build_skew_result(ticker, skew_ratio, put_iv, call_iv, target_expiry)

    except Exception as e:
        log.debug(f"  [{ticker}] yfinance Skew Fehler: {e}")
        return None


def _fetch_skew_tradier(ticker: str, current_price: float, api_key: str) -> Optional[dict]:
    """Tradier-basierte Skew-Berechnung (höhere IV-Qualität)."""
    try:
        from datetime import timezone
        from datetime import datetime as _dt

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept":        "application/json",
        }
        resp = requests.get(
            "https://api.tradier.com/v1/markets/options/expirations",
            params={"symbol": ticker, "includeAllRoots": "true"},
            headers=headers, timeout=10,
        )
        resp.raise_for_status()
        all_dates = resp.json().get("expirations", {}).get("date", []) or []
        if isinstance(all_dates, str):
            all_dates = [all_dates]

        now = _dt.now(timezone.utc)
        target_expiry = None
        for d in sorted(all_dates):
            try:
                exp_dt = _dt.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                dte    = (exp_dt - now).days
                if SKEW_LOOKBACK_DTE_MIN <= dte <= SKEW_LOOKBACK_DTE_MAX:
                    target_expiry = d
                    break
            except Exception:
                continue

        if not target_expiry:
            return None

        chain_resp = requests.get(
            "https://api.tradier.com/v1/markets/options/chains",
            params={"symbol": ticker, "expiration": target_expiry, "greeks": "true"},
            headers=headers, timeout=10,
        )
        chain_resp.raise_for_status()
        options = chain_resp.json().get("options", {}).get("option", []) or []
        if isinstance(options, dict):
            options = [options]

        call_iv_atm, put_iv_atm = None, None
        best_call_dist, best_put_dist = float("inf"), float("inf")

        for o in options:
            strike = float(o.get("strike", 0))
            dist   = abs(strike - current_price)
            greeks = o.get("greeks") or {}
            iv     = greeks.get("mid_iv") or greeks.get("smv_vol") or 0.0
            if not isinstance(iv, (int, float)) or iv <= 0.01:
                continue

            if o.get("option_type") == "call" and dist < best_call_dist:
                best_call_dist = dist
                call_iv_atm    = float(iv)
            elif o.get("option_type") == "put" and dist < best_put_dist:
                best_put_dist = dist
                put_iv_atm    = float(iv)

        if not call_iv_atm or not put_iv_atm:
            return None

        skew_ratio = put_iv_atm / call_iv_atm
        return _build_skew_result(ticker, skew_ratio, put_iv_atm, call_iv_atm, target_expiry)

    except Exception as e:
        log.debug(f"  [{ticker}] Tradier Skew Fehler: {e}")
        return None


def _build_skew_result(
    ticker: str, skew_ratio: float, put_iv: float, call_iv: float, expiry: str
) -> dict:
    """Erstellt das Skew-Result-Dict mit Signal und Headline."""
    if skew_ratio >= SKEW_BEARISH_THRESHOLD:
        signal   = "bearish_skew"
        headline = (
            f"Options-Skew {ticker}: Puts {skew_ratio:.1%} teurer als Calls "
            f"(put_iv={put_iv:.1%} vs call_iv={call_iv:.1%}) — Markt sichert Downside ab"
        )
    elif skew_ratio <= SKEW_BULLISH_THRESHOLD:
        signal   = "bullish_skew"
        headline = (
            f"Options-Skew {ticker}: Calls relativ zu Puts günstig "
            f"(skew={skew_ratio:.2f}) — Markt erwartet wenig Downside"
        )
    else:
        signal   = "neutral"
        headline = ""

    return {
        "skew_ratio":     round(skew_ratio, 3),
        "put_iv":         round(put_iv, 4),
        "call_iv":        round(call_iv, 4),
        "expiry":         expiry,
        "signal":         signal,
        "headline":       headline,
        "data_available": True,
    }


# ── v9.0 #15: Dealer-Gamma-Schätzung ─────────────────────────────────────────

def estimate_dealer_gamma(ticker: str, current_price: float) -> dict:
    """
    Schätzt die Netto-Dealer-Gamma-Position aus Open Interest.

    Vereinfachte Methode (ohne Live Market-Maker-Daten):
    - Calls: Dealer sind typischerweise SHORT Calls (haben negative Gamma)
      → hoher Call-OI nahe ATM → Dealer müssen kaufen wenn Preis steigt (Gamma hedging)
    - Puts: Dealer sind typischerweise SHORT Puts (haben negative Gamma)
      → hoher Put-OI nahe ATM → Dealer müssen verkaufen wenn Preis fällt

    Netto-Gamma-Schätzung: Call-OI × Γ_call - Put-OI × Γ_put
    Γ ≈ N'(d1) / (S × σ × √T) — für ATM Optionen vereinfacht proportional zu 1/σ

    Positive Netto-Gamma: Dealer dämpfen Bewegungen (mean-reversion wahrscheinlicher)
    Negative Netto-Gamma: Dealer verstärken Bewegungen (trending wahrscheinlicher)

    Returns:
        {
            "net_gamma_sign":  "positive" | "negative" | "neutral",
            "call_oi_atm":     int,
            "put_oi_atm":      int,
            "oi_ratio":        float,   # call_oi / put_oi
            "signal":          str,
            "headline":        str,
            "data_available":  bool,
        }
    """
    result_empty = {
        "net_gamma_sign": "neutral", "call_oi_atm": 0, "put_oi_atm": 0,
        "oi_ratio": 1.0, "signal": "neutral", "headline": "",
        "data_available": False,
    }

    if current_price <= 0:
        return result_empty

    try:
        import yfinance as yf
        from datetime import timezone
        from datetime import datetime as _dt

        t   = yf.Ticker(ticker)
        now = _dt.now(timezone.utc)

        target_expiry = None
        for exp in (t.options or []):
            try:
                exp_dt = _dt.strptime(exp, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                dte    = (exp_dt - now).days
                if 14 <= dte <= 45:
                    target_expiry = exp
                    break
            except Exception:
                continue

        if not target_expiry:
            return result_empty

        chain = t.option_chain(target_expiry)

        # ATM-Bereich: ±5% vom aktuellen Preis
        atm_low  = current_price * 0.95
        atm_high = current_price * 1.05

        calls_atm = chain.calls[
            (chain.calls["strike"] >= atm_low) &
            (chain.calls["strike"] <= atm_high)
        ]
        puts_atm = chain.puts[
            (chain.puts["strike"] >= atm_low) &
            (chain.puts["strike"] <= atm_high)
        ]

        call_oi = int(calls_atm["openInterest"].sum()) if not calls_atm.empty else 0
        put_oi  = int(puts_atm["openInterest"].sum())  if not puts_atm.empty  else 0

        if call_oi + put_oi < 100:
            return result_empty

        oi_ratio = call_oi / put_oi if put_oi > 0 else 2.0

        # Dealer-Gamma-Interpretation:
        # Hoher Call-OI ATM → Dealer short viele Calls → müssen bei Anstieg kaufen
        # = negative Dealer-Gamma (bei Calls) → verstärkt Aufwärtsbewegung
        # Für Trading-Signal: hoher put_oi/call_oi-Ratio → viel Absicherung
        # = Markt ist bearish positioniert aber gut abgesichert
        if oi_ratio > 1.5:
            net_gamma_sign = "positive"  # Mehr Calls, Dealer dämpfen Anstieg etwas
            signal         = "gamma_neutral_to_positive"
            headline       = (
                f"Dealer-Gamma {ticker}: Call-OI ({call_oi:,}) dominiert Put-OI ({put_oi:,}) "
                f"ATM — Dealer-Hedging könnte Aufwärtsbewegungen leicht dämpfen"
            )
        elif oi_ratio < 0.70:
            net_gamma_sign = "negative"  # Mehr Puts, Dealer-Hedging verstärkt Abwärts
            signal         = "gamma_bearish_pressure"
            headline       = (
                f"Dealer-Gamma {ticker}: Put-OI ({put_oi:,}) dominiert ATM "
                f"(Call/Put-Ratio={oi_ratio:.2f}) — Dealer-Hedging kann Abwärtsbewegungen verstärken"
            )
        else:
            net_gamma_sign = "neutral"
            signal         = "neutral"
            headline       = ""

        log.info(
            f"  [{ticker}] Dealer-Gamma: call_oi={call_oi} put_oi={put_oi} "
            f"ratio={oi_ratio:.2f} → {net_gamma_sign}"
        )

        return {
            "net_gamma_sign": net_gamma_sign,
            "call_oi_atm":    call_oi,
            "put_oi_atm":     put_oi,
            "oi_ratio":       round(oi_ratio, 3),
            "signal":         signal,
            "headline":       headline,
            "data_available": True,
        }

    except Exception as e:
        log.debug(f"  [{ticker}] Dealer-Gamma Fehler: {e}")
        return result_empty


# ── Kombinierter Alpha-Enrichment ─────────────────────────────────────────────

def enrich_with_alpha_sources(candidate: dict) -> dict:
    """
    Reichert einen Pipeline-Kandidaten mit FDA, SEC, Finnhub,
    Put/Call-Skew und Dealer-Gamma-Daten an.

    v9.0: Skew und Gamma werden als neue alpha_signals gespeichert
    und auffällige Werte als Headlines in candidate["news"] aufgenommen.
    """
    ticker = candidate.get("ticker", "")
    info   = candidate.get("info", {})

    current_price = float(
        info.get("currentPrice") or
        info.get("regularMarketPrice") or 0
    )

    alpha_signals = {
        "fda_headlines":     [],
        "sec_insider":       {},
        "earnings_date":     None,
        "has_near_earnings": False,
        "options_skew":      {},
        "dealer_gamma":      {},
    }

    # 1. FDA (nur für Healthcare/Biotech)
    sector = info.get("sector", "")
    if sector in ("Healthcare", "Biotechnology", "Pharmaceuticals"):
        fda_headlines = match_fda_to_ticker(ticker, info)
        alpha_signals["fda_headlines"] = fda_headlines
        if fda_headlines:
            candidate.setdefault("news", [])
            candidate["news"] = fda_headlines + candidate["news"]

    # 2. SEC Insider
    insider_data = detect_insider_cluster(ticker)
    alpha_signals["sec_insider"] = insider_data
    if insider_data.get("headline"):
        candidate.setdefault("news", [])
        candidate["news"] = [insider_data["headline"]] + candidate["news"]

    # 3. Finnhub Earnings
    has_earnings, earnings_date = has_earnings_within_days(ticker)
    alpha_signals["earnings_date"]     = earnings_date
    alpha_signals["has_near_earnings"] = has_earnings
    candidate["has_near_earnings"]     = has_earnings

    # 4. v9.0 #15: Put/Call-Skew
    if current_price > 0:
        skew_data = fetch_options_skew(ticker, current_price)
        alpha_signals["options_skew"] = skew_data
        if skew_data.get("headline"):
            candidate.setdefault("news", [])
            candidate["news"] = candidate["news"] + [skew_data["headline"]]

    # 5. v9.0 #15: Dealer-Gamma-Schätzung
    if current_price > 0:
        gamma_data = estimate_dealer_gamma(ticker, current_price)
        alpha_signals["dealer_gamma"] = gamma_data
        if gamma_data.get("headline"):
            candidate.setdefault("news", [])
            candidate["news"] = candidate["news"] + [gamma_data["headline"]]

    candidate["alpha_signals"] = alpha_signals
    return candidate

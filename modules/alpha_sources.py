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
from datetime import datetime, timedelta, date
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

# ── SEC Form 4: echte Insider-KÄUFE (offizielle EDGAR-API) ───────────────────
#
# Fix 2026-09-28: vorher Volltextsuche efts "q=%22TICKER%22&forms=4" -> traf
# jedes Form-4-Dokument, das den String enthielt ("ARE", "BE", "MMM" ...),
# zählte Verkäufe/Zuteilungen mit und erzeugte für fast JEDEN Kandidaten die
# Schlagzeile "N Insider kaufen X (Cluster-Signal)", die der Deep Analysis
# VORANGESTELLT wurde (falscher bullisher Katalysator). Jetzt: Ticker -> CIK
# (company_tickers.json), Emittenten-Filings (data.sec.gov/submissions),
# Form-4-XML parsen, nur Open-Market-Käufe (transactionCode P, Acquired) zählen.

SEC_TICKERS_URL     = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SEC_ARCHIVE_URL     = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"
SEC_MAX_FORM4_PER_TICKER = 25
SEC_MIN_INTERVAL_S  = 0.12          # SEC Fair Access: <= 10 Requests/s
CLUSTER_WINDOW_DAYS = 3             # "mehrere Insider innerhalb 72 h"

_sec_cik_map: dict | None = None
_sec_last_call = [0.0]


def _sec_get(url: str, timeout: int = 10):
    wait = SEC_MIN_INTERVAL_S - (time.monotonic() - _sec_last_call[0])
    if wait > 0:
        time.sleep(wait)
    _sec_last_call[0] = time.monotonic()
    return requests.get(url, headers=_HEADERS, timeout=timeout)


def sec_cik_for_ticker(ticker: str) -> Optional[int]:
    global _sec_cik_map
    if _sec_cik_map is None:
        try:
            resp = _sec_get(SEC_TICKERS_URL, timeout=15)
            resp.raise_for_status()
            _sec_cik_map = {str(v.get("ticker", "")).upper(): int(v["cik_str"])
                            for v in (resp.json() or {}).values() if v.get("cik_str")}
        except Exception as e:
            log.debug(f"SEC Ticker-CIK-Liste nicht abrufbar: {e}")
            return None
    t = ticker.upper()
    return _sec_cik_map.get(t) or _sec_cik_map.get(t.replace(".", "-"))


def _xml_text(node, path: str) -> Optional[str]:
    el = node.find(path) if node is not None else None
    return el.text.strip() if el is not None and el.text else None


def parse_form4_xml(xml_text: str) -> dict:
    """Form-4-XML -> Emittent, meldende Person(en), Transaktionen der
    nonDerivativeTable (Code, Datum, Stückzahl, Preis, A/D)."""
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml_text)
    owners = [_xml_text(o, "reportingOwnerId/rptOwnerName") for o in root.findall("reportingOwner")]
    txs = []
    for t in root.findall("nonDerivativeTable/nonDerivativeTransaction"):
        def num(path):
            v = _xml_text(t, path)
            try:
                return float(v) if v is not None else None
            except ValueError:
                return None
        txs.append({
            "date":   _xml_text(t, "transactionDate/value"),
            "code":   _xml_text(t, "transactionCoding/transactionCode"),
            "shares": num("transactionAmounts/transactionShares/value"),
            "price":  num("transactionAmounts/transactionPricePerShare/value"),
            "ad":     _xml_text(t, "transactionAmounts/transactionAcquiredDisposedCode/value"),
        })
    return {"issuer_cik": _xml_text(root, "issuer/issuerCik"),
            "issuer_symbol": _xml_text(root, "issuer/issuerTradingSymbol"),
            "owners": [o for o in owners if o], "transactions": txs}


def fetch_sec_insider_trades(ticker: str, days_back: int = 14) -> list[dict]:
    """Open-Market-Transaktionen (P = Kauf, S = Verkauf) aus den Form-4-
    Filings des Emittenten der letzten days_back Tage. Leere Liste, wenn
    keine Daten (fehlend != kein Insider-Kauf -> siehe data_available)."""
    cik = sec_cik_for_ticker(ticker)
    if cik is None:
        return []
    try:
        resp = _sec_get(SEC_SUBMISSIONS_URL.format(cik=cik))
        resp.raise_for_status()
        recent = (resp.json().get("filings") or {}).get("recent") or {}
    except Exception as e:
        log.debug(f"SEC Submissions Fehler für {ticker}: {e}")
        return []
    since = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    forms, dates = recent.get("form") or [], recent.get("filingDate") or []
    accs, docs = recent.get("accessionNumber") or [], recent.get("primaryDocument") or []
    trades, n = [], 0
    for form, fdate, acc, doc in zip(forms, dates, accs, docs):
        if form != "4" or fdate < since:
            continue
        n += 1
        if n > SEC_MAX_FORM4_PER_TICKER:
            break
        raw_doc = str(doc).split("/")[-1]           # "xslF345X05/x.xml" -> Roh-XML "x.xml"
        url = SEC_ARCHIVE_URL.format(cik=cik, acc=acc.replace("-", ""), doc=raw_doc)
        try:
            r = _sec_get(url)
            r.raise_for_status()
            parsed = parse_form4_xml(r.text)
        except Exception as e:
            log.debug(f"SEC Form 4 {acc} ({ticker}) nicht lesbar: {e}")
            continue
        if parsed["issuer_cik"] and int(parsed["issuer_cik"]) != cik:
            continue                                    # Filing eines anderen Emittenten
        insider = ", ".join(parsed["owners"]) or "Unknown"
        for tx in parsed["transactions"]:
            if tx["code"] not in ("P", "S"):
                continue                                # Zuteilung/Ausübung/Steuer != Marktsignal
            trades.append({"date": tx["date"] or fdate, "filing_date": fdate, "insider": insider,
                           "code": tx["code"], "shares": tx["shares"], "price": tx["price"],
                           "value": (tx["shares"] or 0) * (tx["price"] or 0),
                           "accession": acc, "form": "Form 4", "source": "SEC"})
    return trades


def _max_buyers_in_window(buys: list[dict], window_days: int) -> int:
    best = 0
    parsed = []
    for b in buys:
        try:
            parsed.append((datetime.strptime(str(b["date"])[:10], "%Y-%m-%d"), b["insider"]))
        except ValueError:
            continue
    for d0, _ in parsed:
        names = {n for d, n in parsed if timedelta(0) <= d - d0 <= timedelta(days=window_days)}
        best = max(best, len(names))
    return best


def detect_insider_cluster(ticker: str, days_back: int = 14) -> dict:
    """Cluster = >= 2 verschiedene Insider mit Open-Market-KÄUFEN (Code P)
    innerhalb von 72 h. Verkäufe werden separat gezählt, nie als Kauf."""
    cik = sec_cik_for_ticker(ticker)
    trades = fetch_sec_insider_trades(ticker, days_back) if cik is not None else []
    buys = [t for t in trades if t["code"] == "P"]
    sells = [t for t in trades if t["code"] == "S"]
    buyers = {t["insider"] for t in buys}
    window_buyers = _max_buyers_in_window(buys, CLUSTER_WINDOW_DAYS)
    cluster = window_buyers >= 2
    result = {
        "data_available": cik is not None,
        "cluster_detected": cluster,
        "insider_count": len(buyers),                  # Anzahl KÄUFER (nicht Filings)
        "buy_count": len(buys), "sell_count": len(sells),
        "buy_value": round(sum(t["value"] for t in buys), 2),
        "sell_value": round(sum(t["value"] for t in sells), 2),
        "trades": buys[:5],
        "headline": "",
    }
    if cluster:
        result["headline"] = (
            f"SEC Form 4: {window_buyers} Insider kaufen {ticker} am offenen Markt "
            f"innerhalb {CLUSTER_WINDOW_DAYS * 24} h (Cluster-Signal, "
            f"${result['buy_value']:,.0f} gesamt)"
        )
    if buys or sells:
        log.info(f"  [{ticker}] SEC Form 4 ({days_back}d): {len(buyers)} Käufer/{len(buys)} Käufe, "
                 f"{len(sells)} Verkäufe{' → CLUSTER' if cluster else ''}")
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


def _earnings_buffer_days() -> int:
    """risk.earnings_buffer_days aus config.yaml (vorher ignoriert, immer 7)."""
    try:
        from modules.config import cfg
        return int(getattr(cfg.risk, "earnings_buffer_days", 7) or 7)
    except Exception:
        return 7


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
        except Exception as e:
            log.warning(f"  [{ticker}] Earnings-Datum yfinance Fehler: {e}")

    if not earnings_date:
        # Unbekannt != keine Earnings: Gate kann nicht greifen -> sichtbar machen
        # (alpha_signals.earnings_date=None -> Ledger earnings_known=False).
        log.warning(f"  [{ticker}] Earnings-Termin unbekannt → Earnings-Gate ohne Datengrundlage")
        return False, None

    try:
        earnings_dt = datetime.strptime(earnings_date, "%Y-%m-%d")
        # date.today() statt datetime.utcnow() — konsistent mit risk_gates.py.
        # datetime.utcnow() hat UTC-Offset-Fehler (0d vs 1d je nach Tageszeit).
        days_until  = (earnings_dt.date() - date.today()).days

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
    ACHTUNG (Audit 2026-09-28): gleicher Strike -> Put-Call-Parität -> Quotient
    strukturell ~1.0, misst KEINE Schiefe. skew_ratio/signal bleiben aus
    Governance-Gründen unverändert (Produktionsverhalten); die echte Schiefe
    steht als Forschungsfeature in skew_25d (25-Delta-Put-IV / 25-Delta-Call-IV,
    yfinance-Fallback: Strikes +-7 %), ohne Schlagzeile/Score-Wirkung.
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
        res = _build_skew_result(ticker, skew_ratio, put_iv, call_iv, target_expiry)
        try:
            k_put = min(chain.puts["strike"].tolist(), key=lambda x: abs(x - 0.93 * current_price))
            k_call = min(chain.calls["strike"].tolist(), key=lambda x: abs(x - 1.07 * current_price))
            piv = float(chain.puts[chain.puts["strike"] == k_put]["impliedVolatility"].iloc[0])
            civ = float(chain.calls[chain.calls["strike"] == k_call]["impliedVolatility"].iloc[0])
            if piv > 0.01 and civ > 0.01:
                res["skew_25d"] = round(piv / civ, 3)
                res["skew_25d_method"] = "yfinance_moneyness_7pct"
        except Exception:
            pass
        return res

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
        c25 = p25 = None   # (|delta-0.25|, iv)

        for o in options:
            strike = float(o.get("strike", 0))
            dist   = abs(strike - current_price)
            greeks = o.get("greeks") or {}
            iv     = greeks.get("mid_iv") or greeks.get("smv_vol") or 0.0
            if not isinstance(iv, (int, float)) or iv <= 0.01:
                continue
            delta = greeks.get("delta")
            if isinstance(delta, (int, float)):
                if o.get("option_type") == "call" and 0.10 <= delta <= 0.40:
                    d = abs(delta - 0.25)
                    if c25 is None or d < c25[0]:
                        c25 = (d, float(iv))
                elif o.get("option_type") == "put" and -0.40 <= delta <= -0.10:
                    d = abs(delta + 0.25)
                    if p25 is None or d < p25[0]:
                        p25 = (d, float(iv))

            if o.get("option_type") == "call" and dist < best_call_dist:
                best_call_dist = dist
                call_iv_atm    = float(iv)
            elif o.get("option_type") == "put" and dist < best_put_dist:
                best_put_dist = dist
                put_iv_atm    = float(iv)

        if not call_iv_atm or not put_iv_atm:
            return None

        skew_ratio = put_iv_atm / call_iv_atm
        res = _build_skew_result(ticker, skew_ratio, put_iv_atm, call_iv_atm, target_expiry)
        if c25 and p25:
            res["skew_25d"] = round(p25[1] / c25[1], 3)
            res["skew_25d_method"] = "tradier_delta"
        return res

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
    has_earnings, earnings_date = has_earnings_within_days(ticker, buffer_days=_earnings_buffer_days())
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

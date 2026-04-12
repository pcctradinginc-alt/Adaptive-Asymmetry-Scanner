"""
modules/data_validator.py v6.0

Fix 5: SEC EDGAR XBRL für EPS (statt veraltetem yfinance forwardEps)
    EDGAR liefert geprüfte EPS-Daten direkt aus 10-Q/10-K Filings.
    Kostenlos, kein API-Key nötig.
    URL: https://data.sec.gov/api/xbrl/companyfacts/{CIK}.json

Fix 6: Echte Vega-Kalkulation im ROI-Gate
    Bisher: Vega ignoriert → ROI-Schätzung zu optimistisch bei hohem IV-Rank
    Jetzt:  Black-Scholes Vega berechnet → IV-Mean-Reversion-Verlust eingepreist
    Formel: Vega-P&L = Vega × ΔIV (erwartete IV-Reduktion nach Event)
"""

from __future__ import annotations
import logging
import math
import os
import time
from typing import Optional

import requests
import yfinance as yf

log = logging.getLogger(__name__)

# Alpha Vantage Rate-Limit
_AV_BASE       = "https://www.alphavantage.co/query"
_AV_DELAY      = 12.5
_last_av_call  = 0.0

# SEC EDGAR
_SEC_BASE      = "https://data.sec.gov"
_SEC_HEADERS   = {
    "User-Agent": "newstoption-scanner/5.0 research@pcctrading.com",
    "Accept-Encoding": "gzip, deflate",
}
_cik_cache: dict[str, str] = {}


# ── Fix 5: SEC EDGAR XBRL EPS ────────────────────────────────────────────────

def fetch_eps_sec_edgar(ticker: str) -> Optional[float]:
    """
    Ruft EPS (TTM) direkt aus SEC EDGAR XBRL-Daten ab.
    Quelle: 10-Q / 10-K Filings — geprüfte Zahlen, nicht veraltet.
    Kein API-Key nötig.

    Ablauf:
    1. CIK (Central Index Key) für Ticker lookup via EDGAR Tickers API
    2. EPS-Daten aus companyfacts API (XBRL Tag: us-gaap/EarningsPerShareBasic)
    3. TTM berechnen aus letzten 4 Quartals-Filings
    """
    try:
        cik = _get_cik(ticker)
        if not cik:
            return None

        url  = f"{_SEC_BASE}/api/xbrl/companyfacts/CIK{cik}.json"
        resp = requests.get(url, headers=_SEC_HEADERS, timeout=15)

        if resp.status_code != 200:
            return None

        facts = resp.json()
        us_gaap = facts.get("facts", {}).get("us-gaap", {})

        # EPS Basic (oder Diluted als Fallback)
        for tag in ("EarningsPerShareBasic", "EarningsPerShareDiluted"):
            eps_data = us_gaap.get(tag, {})
            if not eps_data:
                continue

            # Nur USD-Einheit, Quartals-Filings (10-Q)
            units = eps_data.get("units", {}).get("USD/shares", [])
            if not units:
                continue

            # Neueste 4 Quartale für TTM
            quarterly = [
                u for u in units
                if u.get("form") in ("10-Q", "10-K") and u.get("val") is not None
            ]

            if not quarterly:
                continue

            # Neueste 4 Einträge
            quarterly.sort(key=lambda x: x.get("end", ""), reverse=True)
            ttm_eps = sum(u["val"] for u in quarterly[:4])

            log.info(
                f"  [{ticker}] SEC EDGAR EPS: TTM={ttm_eps:.2f} "
                f"(aus {min(4, len(quarterly))} Quartalen, Tag={tag})"
            )
            return round(ttm_eps, 4)

        return None

    except Exception as e:
        log.debug(f"SEC EDGAR EPS Fehler für {ticker}: {e}")
        return None


def _get_cik(ticker: str) -> Optional[str]:
    """Lookup CIK (Central Index Key) für einen Ticker via EDGAR."""
    if ticker in _cik_cache:
        return _cik_cache[ticker]

    try:
        resp = requests.get(
            f"{_SEC_BASE}/files/company_tickers.json",
            headers=_SEC_HEADERS,
            timeout=10,
        )
        if resp.status_code != 200:
            return None

        data = resp.json()
        for _, company in data.items():
            if company.get("ticker", "").upper() == ticker.upper():
                cik = str(company["cik_str"]).zfill(10)
                _cik_cache[ticker] = cik
                return cik

        return None

    except Exception as e:
        log.debug(f"CIK-Lookup Fehler für {ticker}: {e}")
        return None


def cross_check_eps_edgar(
    ticker: str,
    yfinance_eps: float,
    tolerance: float = 0.10,
) -> dict:
    """
    Cross-Check: yfinance EPS vs. SEC EDGAR (offizielle Filings).

    Priorität:
    1. SEC EDGAR (geprüfte Zahlen) — primäre Quelle
    2. Alpha Vantage (Fallback wenn EDGAR fehlschlägt)
    3. yfinance allein (wenn beides fehlschlägt)

    Returns:
        {
            "sec_eps":         float or None,
            "yf_eps":          float,
            "deviation_pct":   float or None,
            "consistent":      bool,
            "confidence":      "high" | "medium" | "low",
            "source":          "SEC_EDGAR" | "ALPHA_VANTAGE" | "YFINANCE_ONLY",
            "data_anomaly":    bool,  # True wenn >20% Abweichung → Claude warnen
        }
    """
    sec_eps = fetch_eps_sec_edgar(ticker)
    source  = "YFINANCE_ONLY"

    if sec_eps is None:
        # Fallback: Alpha Vantage
        sec_eps = _fetch_eps_alphavantage(ticker)
        if sec_eps is not None:
            source = "ALPHA_VANTAGE"
    else:
        source = "SEC_EDGAR"

    if sec_eps is None:
        return {
            "sec_eps": None, "yf_eps": yfinance_eps,
            "deviation_pct": None, "consistent": True,
            "confidence": "medium", "source": "YFINANCE_ONLY",
            "data_anomaly": False,
        }

    if yfinance_eps == 0 or yfinance_eps is None:
        return {
            "sec_eps": sec_eps, "yf_eps": yfinance_eps,
            "deviation_pct": None, "consistent": False,
            "confidence": "low", "source": source,
            "data_anomaly": True,
        }

    deviation = abs(sec_eps - yfinance_eps) / abs(yfinance_eps)
    consistent  = deviation <= tolerance
    data_anomaly = deviation > 0.20   # >20% → Claude im Prompt warnen

    confidence = "high" if deviation < 0.02 else ("medium" if consistent else "low")

    if data_anomaly:
        log.warning(
            f"  [{ticker}] DATA ANOMALY: yfinance={yfinance_eps:.2f} vs "
            f"{source}={sec_eps:.2f} Abweichung={deviation:.1%} > 20% "
            f"→ Signal-Qualität reduziert."
        )
    elif not consistent:
        log.warning(
            f"  [{ticker}] EPS-Inkonsistenz: yf={yfinance_eps:.2f} "
            f"vs {source}={sec_eps:.2f} ({deviation:.1%})"
        )

    return {
        "sec_eps":       round(sec_eps, 4),
        "yf_eps":        round(yfinance_eps, 4),
        "deviation_pct": round(deviation, 4),
        "consistent":    consistent,
        "confidence":    confidence,
        "source":        source,
        "data_anomaly":  data_anomaly,
    }


def _fetch_eps_alphavantage(ticker: str) -> Optional[float]:
    global _last_av_call
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    if not api_key:
        return None
    elapsed = time.time() - _last_av_call
    if elapsed < _AV_DELAY:
        time.sleep(_AV_DELAY - elapsed)
    _last_av_call = time.time()
    try:
        resp = requests.get(
            _AV_BASE,
            params={"function": "OVERVIEW", "symbol": ticker, "apikey": api_key},
            timeout=10,
        )
        data    = resp.json()
        eps_str = data.get("EPS", "")
        if eps_str and eps_str not in ("None", "-", ""):
            return float(eps_str)
        return None
    except Exception:
        return None


# ── Fix 6: Echte Vega-Kalkulation im ROI-Gate ────────────────────────────────

def compute_option_roi_with_vega(option: dict, simulation: dict) -> dict:
    """
    Fix 6: ROI-Berechnung mit Vega-Adjustment.

    Bisher ignoriert: IV Mean-Reversion nach News-Event.
    Typisches Muster:
      - News → IV springt +30-50% (Fear/Greed)
      - 2-4 Wochen später: IV normalisiert sich -20-30%
      - Long Call verliert durch Vega auch wenn Aktie steigt

    Formel:
      roi_delta_approx = stock_move × delta × leverage
      vega_loss        = vega × expected_iv_drop
      roi_net          = roi_delta_approx - spread_cost - vega_loss

    Delta und Vega aus Black-Scholes (vereinfacht, ohne Zinsterm).
    """
    bid     = option.get("bid", 0.0) or 0.0
    ask     = option.get("ask", 0.0) or 0.0
    strike  = option.get("strike", 0.0) or 0.0
    iv      = option.get("implied_vol", 0.30) or 0.30
    dte     = option.get("dte", 120) or 120

    if ask <= 0:
        return _roi_empty()

    spread_pct = (ask - bid) / ask if ask > 0 else 0.0

    current = simulation.get("current_price", 0.0) or 0.0
    target  = simulation.get("target_price", 0.0) or 0.0
    iv_rank = simulation.get("iv_rank", 50.0) or 50.0

    if current <= 0 or target <= current:
        roi_gross = 0.0
        vega_loss = 0.0
    else:
        T = dte / 365.0   # Zeit in Jahren

        # Black-Scholes Delta und Vega
        delta_bs, vega_bs = _bs_delta_vega(current, strike, iv, T)

        # Erwarteter Kursgewinn
        expected_move = (target - current) / current
        leverage      = current / ask if ask > 0 else 1.0

        # Delta-approximierter ROI
        roi_delta = expected_move * delta_bs * leverage

        # Vega-Verlust durch IV Mean-Reversion
        # Faustregel: Bei IV-Rank > 70% → IV fällt nach Event ~20-30%
        # Bei IV-Rank 50-70% → IV fällt ~10-15%
        # Bei IV-Rank < 50% → IV-Crush unwahrscheinlich, ~0-5%
        if iv_rank >= 70:
            expected_iv_drop = 0.25   # 25% IV-Reduktion
        elif iv_rank >= 50:
            expected_iv_drop = 0.12
        else:
            expected_iv_drop = 0.05

        # Vega-P&L: Vega × ΔIV × leverage
        # Vega in $ pro 1 Punkt IV, normalisiert auf Option-Preis
        vega_normalized = vega_bs * iv * expected_iv_drop * leverage
        vega_loss       = max(vega_normalized, 0.0)

        roi_gross = roi_delta
        log.debug(
            f"    ROI-Detail: delta={delta_bs:.2f} vega={vega_bs:.4f} "
            f"iv_drop={expected_iv_drop:.0%} vega_loss={vega_loss:.3f}"
        )

    # Spread-Kosten (Round-trip)
    roi_net = roi_gross - (spread_pct * 2) - vega_loss

    min_roi   = float(os.getenv("MIN_OPTION_ROI", "0.15"))
    passes    = roi_net >= min_roi

    if not passes:
        log.info(
            f"    ROI-GATE: gross={roi_gross:.1%} "
            f"spread={spread_pct:.1%} vega_loss={vega_loss:.1%} "
            f"net={roi_net:.1%} < {min_roi:.0%} → abgelehnt."
        )

    return {
        "roi_gross":          round(roi_gross, 4),
        "roi_net":            round(roi_net, 4),
        "spread_pct":         round(spread_pct, 4),
        "vega_loss":          round(vega_loss, 4),
        "passes_roi_gate":    passes,
        "min_roi_threshold":  min_roi,
        "iv_rank_used":       iv_rank,
    }


def _bs_delta_vega(S: float, K: float, sigma: float, T: float) -> tuple[float, float]:
    """
    Black-Scholes Delta und Vega für eine Call-Option.
    Vereinfacht: r=0 (für kurze Berechnungszwecke ausreichend).

    Args:
        S:     Aktueller Kurs
        K:     Strike
        sigma: Implizite Volatilität (z.B. 0.30 = 30%)
        T:     Zeit in Jahren

    Returns:
        (delta, vega)
    """
    if sigma <= 0 or T <= 0 or S <= 0 or K <= 0:
        return 0.5, 0.0

    try:
        d1    = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
        delta = _norm_cdf(d1)
        vega  = S * _norm_pdf(d1) * math.sqrt(T)
        return round(delta, 4), round(vega, 4)
    except Exception:
        return 0.5, 0.0


def _norm_cdf(x: float) -> float:
    """Kumulative Normalverteilung."""
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def _norm_pdf(x: float) -> float:
    """Normalverteilungs-PDF."""
    return math.exp(-0.5 * x ** 2) / math.sqrt(2.0 * math.pi)


def _roi_empty() -> dict:
    return {
        "roi_gross": 0.0, "roi_net": 0.0,
        "spread_pct": 0.0, "vega_loss": 0.0,
        "passes_roi_gate": False,
        "min_roi_threshold": 0.15,
        "iv_rank_used": 50.0,
    }


def validate_candidate_data(candidate: dict) -> dict:
    """Kombinierte Daten-Validierung: SEC EDGAR EPS + Confidence-Score."""
    info   = candidate.get("info", {})
    ticker = candidate.get("ticker", "")
    yf_eps = info.get("trailingEps") or info.get("forwardEps") or 0.0

    eps_check = cross_check_eps_edgar(ticker, yf_eps)

    candidate["data_validation"] = {"eps_cross_check": eps_check}
    candidate["data_confidence"]  = eps_check["confidence"]
    candidate["data_anomaly"]     = eps_check.get("data_anomaly", False)

    return candidate

"""
modules/premium_signals.py – FLASH Alpha & Eulerpool (nur Top-Kandidaten)

Priorität 5: Dealer Positioning, GEX, Vol Surface — nur für finale Kandidaten.

Budget-Regeln (KRITISCH):
  FLASH Alpha: Max 5 API-Calls/Tag → NUR für Top-2 Signale nach RL-Scoring
  Eulerpool:   Free Tier begrenzt  → NUR für Top-2 Signale nach RL-Scoring

Diese Signale laufen NACH Stufe 6 (RL-Scoring), VOR Stufe 7 (Options-Design).
Sie können ein Signal final boosten oder verwerfen.
"""

from __future__ import annotations
import logging
import os
from typing import Optional

import requests

log = logging.getLogger(__name__)

# Globaler Call-Zähler für FLASH Alpha (max 5/Tag)
_flash_calls_today: int = 0
_FLASH_MAX_DAILY   = 5


# ── FLASH Alpha API ───────────────────────────────────────────────────────────

def fetch_flash_alpha(ticker: str) -> dict:
    """
    Ruft Dealer Positioning, GEX und Gamma-Levels via FLASH Alpha ab.

    FLASH Alpha Daten erklären:
      - GEX (Gamma Exposure): Positives GEX → Dealer kaufen Rücksetzer (stabilisiert)
                               Negatives GEX → Dealer verkaufen Rücksetzer (verstärkt)
      - Gamma Flip Level: Preis unter dem Markt instabil wird
      - Put/Call Walls: Preisbereiche mit starker Options-Konzentration

    Relevant für Options-Trading:
      - Wenn Aktie über Gamma-Flip: Calls profitieren von Dealer-Hedging
      - Put-Wall unter Strike: natürlicher Support, reduziert Downside-Risiko

    API: FLASH Alpha (kostenpflichtig, 5 Requests/Tag im getesteten Free-Tier)
    Benötigt: FLASH_ALPHA_API_KEY als GitHub Secret
    """
    global _flash_calls_today

    api_key = os.getenv("FLASH_ALPHA_API_KEY", "")
    if not api_key:
        log.debug("FLASH_ALPHA_API_KEY nicht gesetzt → übersprungen")
        return _flash_empty(ticker)

    if _flash_calls_today >= _FLASH_MAX_DAILY:
        log.warning(
            f"FLASH Alpha Tageslimit ({_FLASH_MAX_DAILY}) erreicht → "
            f"{ticker} übersprungen."
        )
        return _flash_empty(ticker)

    try:
        # FLASH Alpha API Endpoint (Struktur gemäß Dokumentation)
        url  = f"https://api.flashalpha.com/v1/options/positioning/{ticker}"
        resp = requests.get(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept":        "application/json",
            },
            timeout=10,
        )
        _flash_calls_today += 1
        log.info(
            f"  [{ticker}] FLASH Alpha Call "
            f"({_flash_calls_today}/{_FLASH_MAX_DAILY} heute)"
        )

        if resp.status_code == 404:
            return _flash_empty(ticker)

        resp.raise_for_status()
        data = resp.json()

        gex              = data.get("gex", 0.0)
        gamma_flip       = data.get("gamma_flip_level", None)
        put_wall         = data.get("put_wall", None)
        call_wall        = data.get("call_wall", None)
        net_positioning  = data.get("net_positioning", "neutral")

        # Bullish Score aus Dealer-Daten
        dealer_bullish = _compute_dealer_score(
            gex, gamma_flip, put_wall, call_wall, data
        )

        result = {
            "ticker":           ticker,
            "gex":              gex,
            "gamma_flip_level": gamma_flip,
            "put_wall":         put_wall,
            "call_wall":        call_wall,
            "net_positioning":  net_positioning,
            "dealer_bullish":   dealer_bullish,  # -1 (bearish) bis +1 (bullish)
            "data_available":   True,
            "source":           "FLASH Alpha",
        }

        log.info(
            f"  [{ticker}] FLASH Alpha: GEX={gex:.0f} "
            f"flip={gamma_flip} dealer_score={dealer_bullish:.2f}"
        )
        return result

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 402:
            log.warning("FLASH Alpha: Limit erreicht (402 Payment Required)")
            _flash_calls_today = _FLASH_MAX_DAILY  # Weiteres Aufrufen verhindern
        else:
            log.debug(f"FLASH Alpha HTTP Fehler für {ticker}: {e}")
        return _flash_empty(ticker)
    except Exception as e:
        log.debug(f"FLASH Alpha Fehler für {ticker}: {e}")
        return _flash_empty(ticker)


def _compute_dealer_score(
    gex: float,
    gamma_flip: Optional[float],
    put_wall: Optional[float],
    call_wall: Optional[float],
    raw_data: dict,
) -> float:
    """
    Berechnet einen normierten Dealer-Score (-1 bis +1).

    Positiver Score → Dealer-Dynamik begünstigt Calls
    Negativer Score → Dealer-Dynamik begünstigt Puts
    """
    score = 0.0

    # GEX: Positives GEX ist bullish (Dealer kaufen Rücksetzer)
    if gex > 0:
        score += min(gex / 1_000_000_000, 0.4)   # Cap bei 0.4
    elif gex < 0:
        score -= min(abs(gex) / 1_000_000_000, 0.4)

    # Positioning
    positioning = raw_data.get("net_positioning", "neutral")
    if positioning == "bullish":
        score += 0.3
    elif positioning == "bearish":
        score -= 0.3

    return round(max(-1.0, min(1.0, score)), 3)


def _flash_empty(ticker: str) -> dict:
    return {
        "ticker":           ticker,
        "gex":              0.0,
        "gamma_flip_level": None,
        "put_wall":         None,
        "call_wall":        None,
        "net_positioning":  "neutral",
        "dealer_bullish":   0.0,
        "data_available":   False,
        "source":           "FLASH Alpha",
    }


# ── Eulerpool Vol Surface ─────────────────────────────────────────────────────

def fetch_eulerpool_vol_surface(ticker: str) -> dict:
    """
    Ruft Vol Surface Anomalien, OI Dynamics und Flow-Daten via Eulerpool ab.

    Relevant für Options-Trading:
      - Vol Surface Skew: Wenn Puts deutlich teurer als Calls → Downside-Angst
      - OI Spike: Plötzlicher Open Interest Anstieg → Institutionelles Positioning
      - Call/Put Flow: Netto-Options-Volumen zeigt Smart-Money-Richtung

    Benötigt: EULERPOOL_API_KEY als GitHub Secret
    Free Tier: Begrenzte Calls/Tag (genaues Limit in Doku prüfen)
    """
    api_key = os.getenv("EULERPOOL_API_KEY", "")
    if not api_key:
        log.debug("EULERPOOL_API_KEY nicht gesetzt → übersprungen")
        return _eulerpool_empty(ticker)

    try:
        # Eulerpool Options Flow Endpoint
        url  = f"https://api.eulerpool.com/v1/options/flow/{ticker}"
        resp = requests.get(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept":        "application/json",
            },
            timeout=10,
        )

        if resp.status_code in (404, 403):
            return _eulerpool_empty(ticker)

        resp.raise_for_status()
        data = resp.json()

        # Vol Surface
        iv_skew           = data.get("iv_skew", 0.0)         # Put IV - Call IV
        iv_percentile     = data.get("iv_percentile", 50.0)  # 0-100
        vol_surface_trend = data.get("vol_surface_trend", "flat")

        # Flow & OI
        net_call_flow  = data.get("net_call_flow", 0.0)   # USD
        net_put_flow   = data.get("net_put_flow", 0.0)    # USD
        oi_change_pct  = data.get("oi_change_pct_24h", 0.0)

        # IV Crush Risiko berechnen
        iv_crush_risk = _assess_iv_crush_risk(iv_percentile, iv_skew)

        result = {
            "ticker":           ticker,
            "iv_skew":          iv_skew,
            "iv_percentile":    iv_percentile,
            "vol_surface_trend": vol_surface_trend,
            "net_call_flow":    net_call_flow,
            "net_put_flow":     net_put_flow,
            "oi_change_pct":    oi_change_pct,
            "iv_crush_risk":    iv_crush_risk,  # "low" | "medium" | "high"
            "smart_money_bias": _compute_flow_bias(net_call_flow, net_put_flow),
            "data_available":   True,
            "source":           "Eulerpool",
        }

        log.info(
            f"  [{ticker}] Eulerpool: IV-Rank={iv_percentile:.0f}% "
            f"skew={iv_skew:.2f} crush_risk={iv_crush_risk} "
            f"flow_bias={result['smart_money_bias']:.2f}"
        )
        return result

    except Exception as e:
        log.debug(f"Eulerpool Fehler für {ticker}: {e}")
        return _eulerpool_empty(ticker)


def _assess_iv_crush_risk(iv_percentile: float, iv_skew: float) -> str:
    """
    Bewertet das Risiko eines IV-Crush nach Event.

    IV-Crush tritt auf wenn:
    - IV sehr hoch vor Event (IV-Percentile > 80%)
    - Event löst auf (Earnings, FDA-Entscheid)
    - Realized Volatility ist niedriger als implizite

    Für Long-Call-Käufer: IV-Crush kann profitablen Trade in Verlust verwandeln.
    """
    if iv_percentile > 80:
        return "high"
    elif iv_percentile > 60:
        return "medium"
    else:
        return "low"


def _compute_flow_bias(net_call_flow: float, net_put_flow: float) -> float:
    """
    Berechnet Smart-Money-Bias aus Options-Flow.
    +1.0 = Starker Call-Flow (bullish)
    -1.0 = Starker Put-Flow (bearish)
    """
    total = abs(net_call_flow) + abs(net_put_flow)
    if total == 0:
        return 0.0
    return round((net_call_flow - net_put_flow) / total, 3)


def _eulerpool_empty(ticker: str) -> dict:
    return {
        "ticker":           ticker,
        "iv_skew":          0.0,
        "iv_percentile":    50.0,
        "vol_surface_trend": "flat",
        "net_call_flow":    0.0,
        "net_put_flow":     0.0,
        "oi_change_pct":    0.0,
        "iv_crush_risk":    "medium",
        "smart_money_bias": 0.0,
        "data_available":   False,
        "source":           "Eulerpool",
    }


# ── Top-2 Enrichment ──────────────────────────────────────────────────────────

def enrich_top_candidates(signals: list[dict], top_n: int = 2) -> list[dict]:
    """
    Reichert die Top-N Signale nach RL-Scoring mit FLASH Alpha und Eulerpool an.
    Wird nach Stufe 6 (RL-Scoring), vor Stufe 7 (Options-Design) aufgerufen.

    Argumente:
        signals: Nach final_score absteigend sortierte Liste
        top_n:   Nur die ersten N Signale bekommen Premium-Daten (Budget-Schutz)
    """
    enriched = []
    for i, s in enumerate(signals):
        if i < top_n:
            ticker = s.get("ticker", "")
            log.info(f"Premium-Enrichment für Top-{i+1}: {ticker}")

            s["flash_alpha"] = fetch_flash_alpha(ticker)
            s["eulerpool"]   = fetch_eulerpool_vol_surface(ticker)

            # IV-Crush Gate: Wenn hohes IV-Crush-Risiko → warnen
            crush_risk = s["eulerpool"].get("iv_crush_risk", "medium")
            if crush_risk == "high":
                log.warning(
                    f"  [{ticker}] ⚠️ IV-Crush-Risiko HOCH "
                    f"(IV-Percentile={s['eulerpool']['iv_percentile']:.0f}%) "
                    f"→ Long-Options riskanter."
                )
                s["iv_crush_warning"] = True
            else:
                s["iv_crush_warning"] = False

            # Dealer Score in final_score einpreisen (+/- 10% Adjustment)
            dealer_score = s["flash_alpha"].get("dealer_bullish", 0.0)
            if s["flash_alpha"]["data_available"]:
                old_score      = s.get("final_score", 0.0)
                adjustment     = dealer_score * 0.10   # Max ±10%
                s["final_score"] = round(old_score + adjustment, 4)
                log.info(
                    f"  [{ticker}] Dealer-Adjustment: "
                    f"{old_score:.4f} + {adjustment:+.4f} = {s['final_score']:.4f}"
                )
        else:
            # Nicht-Top-N: Leere Structs
            s["flash_alpha"]     = _flash_empty(s.get("ticker", ""))
            s["eulerpool"]       = _eulerpool_empty(s.get("ticker", ""))
            s["iv_crush_warning"] = False

        enriched.append(s)

    return enriched

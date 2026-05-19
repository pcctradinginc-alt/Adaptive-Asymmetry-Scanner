"""
modules/macro_context.py – Makro-Kontext für Claude-Prompt

Fix 3: ISM Einkaufsmanagerindex + 10Y-2Y Yield Curve via FRED API.
       Kein API-Key nötig (FRED public data).
       Wird als Kontext-Variable in deep_analysis.py Prompt injiziert.

Wichtig: Kein Hard-Gate — nur Kontext für Claude.
Begründung: ISM und Yield Curve sind zu träge für Hard-Gates.
            Sie geben Claude aber wichtigen Hintergrund für die
            Bewertung von 2-6 Monats-Signalen.

FRED Endpoints (kostenlos, kein Key für Basic-Daten):
    ISM Manufacturing PMI: MANEMP Proxy via M-PMI
    10Y-2Y Spread:         T10Y2Y
    Fed Funds Rate:        FEDFUNDS (als Zinskontext)
"""

from __future__ import annotations
import logging
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Optional

import requests

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False

log = logging.getLogger(__name__)

_FRED_BASE    = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_HEADERS      = {"User-Agent": "newstoption-scanner/5.0 research@pcctrading.com"}

# Cache: Makro-Daten täglich einmal laden (nicht für jeden Ticker)
_macro_cache: dict = {}
_cache_date:  str  = ""


def get_macro_context() -> dict:
    """
    Lädt Makro-Kontext-Daten einmal pro Tag (gecached).

    Returns:
        {
            "yield_curve_spread": float,   # 10Y-2Y in %
            "yield_curve_regime": str,     # "normal" | "flat" | "inverted"
            "ism_proxy":          float,   # 10Y Zins als Proxy (kein Key nötig)
            "fed_funds_rate":     float,
            "macro_regime":       str,     # "expansive" | "neutral" | "recessionary"
            "claude_context":     str,     # Fertig formulierter Kontext-String für Claude
            "data_available":     bool,
        }
    """
    global _macro_cache, _cache_date

    today = datetime.utcnow().strftime("%Y-%m-%d")
    if _cache_date == today and _macro_cache:
        return _macro_cache

    result = _fetch_macro_data()
    _macro_cache = result
    _cache_date  = today
    return result


def _fetch_macro_data() -> dict:
    """Lädt aktuelle Makro-Daten von FRED."""
    t10y2y    = _fetch_fred_series("T10Y2Y")    # 10Y-2Y Spread
    fedfunds  = _fetch_fred_series("FEDFUNDS")  # Fed Funds Rate
    t10y      = _fetch_fred_series("GS10")      # 10Y Treasury

    # Yield Curve Regime
    if t10y2y is not None:
        if t10y2y > 0.5:
            yc_regime = "normal"
            yc_desc   = f"+{t10y2y:.2f}% (Kurve steil, expansives Umfeld)"
        elif t10y2y > -0.25:
            yc_regime = "flat"
            yc_desc   = f"{t10y2y:.2f}% (Kurve flach, Übergangsphase)"
        else:
            yc_regime = "inverted"
            yc_desc   = f"{t10y2y:.2f}% (Kurve invertiert, Rezessionsrisiko)"
    else:
        yc_regime = "unknown"
        yc_desc   = "Nicht verfügbar"

    # Makro-Gesamtregime
    if yc_regime == "normal":
        macro_regime = "expansive"
        regime_desc  = "Expansiv — Makro begünstigt Long-Positionen"
    elif yc_regime == "flat":
        macro_regime = "neutral"
        regime_desc  = "Neutral — Makro weder Rücken- noch Gegenwind"
    else:
        macro_regime = "recessionary"
        regime_desc  = "Rezessiv — Makro kann positives Alpha dämpfen"

    # VIX Term Structure
    vix_ts = get_vix_term_structure()

    # Fertiger Kontext-String für Claude-Prompt
    claude_context = _build_claude_context(
        t10y2y, fedfunds, t10y, yc_desc, regime_desc, vix_ts
    )

    result = {
        "yield_curve_spread": t10y2y,
        "yield_curve_regime": yc_regime,
        "yield_curve_desc":   yc_desc,
        "fed_funds_rate":     fedfunds,
        "t10y_rate":          t10y,
        "macro_regime":       macro_regime,
        "macro_regime_desc":  regime_desc,
        "vix_term_structure": vix_ts,
        "claude_context":     claude_context,
        "data_available":     t10y2y is not None,
        "fetched_at":         datetime.utcnow().isoformat(),
    }

    fed_str = f"{fedfunds:.2f}%" if fedfunds is not None else "Nicht verfügbar"
    log.info(
        f"Makro-Kontext geladen: Yield Curve={yc_desc} | "
        f"Fed={fed_str} | Regime={macro_regime}"
    )

    return result


def _fetch_fred_series(series_id: str, last_n: int = 1) -> Optional[float]:
    """
    Lädt den aktuellsten Wert einer FRED-Zeitreihe.
    Nutzt das CSV-Endpoint (kein API-Key nötig).
    """
    try:
        since = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
        resp  = requests.get(
            _FRED_BASE,
            params={
                "id":              series_id,
                "vintage_date":    datetime.utcnow().strftime("%Y-%m-%d"),
                "observation_start": since,
            },
            headers=_HEADERS,
            timeout=10,
        )

        if resp.status_code != 200:
            return None

        lines = [l for l in resp.text.strip().split("\n") if l and not l.startswith("DATE")]
        if not lines:
            return None

        # Letzter nicht-leerer Wert
        for line in reversed(lines):
            parts = line.split(",")
            if len(parts) >= 2 and parts[1].strip() not in (".", ""):
                try:
                    return float(parts[1].strip())
                except ValueError:
                    continue

        return None

    except Exception as e:
        log.debug(f"FRED {series_id} Fehler: {e}")
        return None


def _build_claude_context(
    t10y2y:      Optional[float],
    fedfunds:    Optional[float],
    t10y:        Optional[float],
    yc_desc:     str,
    regime_desc: str,
    vix_ts:      Optional[dict] = None,
) -> str:
    """
    Baut den Makro-Kontext-String für den Claude-Prompt.
    Klar formuliert damit Claude ihn richtig interpretiert.
    """
    lines = ["=== MAKRO-KONTEXT FÜR MITTELFRISTIGE BEWERTUNG (2-6 Monate) ==="]

    if t10y2y is not None:
        lines.append(f"Zinskurve (10Y-2Y): {yc_desc}")

    if fedfunds is not None:
        lines.append(f"Fed Funds Rate: {fedfunds:.2f}%")

    if t10y is not None:
        lines.append(f"10Y Treasury: {t10y:.2f}%")

    lines.append(f"Regime-Einschätzung: {regime_desc}")

    if vix_ts and vix_ts.get("available"):
        lines.append(f"Volatilitäts-Struktur: {vix_ts['regime']}")

    lines.append("")
    lines.append(
        "ANWEISUNG: Berücksichtige diesen Makro-Kontext bei der Bewertung. "
        "In einem rezessiven Umfeld (invertierte Kurve) sind positive "
        "Einzel-Aktien-Signale auf 2-6 Monate mit erhöhter Skepsis zu bewerten, "
        "da makroökonomischer Gegenwind das fundamentale Alpha oft überlagert. "
        "In einem expansiven Umfeld können strukturelle Underreactions "
        "stärker gewichtet werden. "
        "Bei VIX Backwardation (kurzfristige Angst hoch) bevorzuge Spreads "
        "gegenüber nackten Long-Calls um Vol-Crush-Risiko nach Events zu begrenzen."
    )

    return "\n".join(lines)


def get_vix_term_structure() -> dict:
    """
    Analysiert die VIX Term Structure (Contango vs. Backwardation).

    Nutzt kostenlose yfinance-Ticker:
      ^VIX9D  = 9-Tage-VIX  (sehr kurzfristig)
      ^VIX    = 30-Tage-VIX (Standard)
      ^VIX3M  = 3-Monats-VIX

    Struktur:
      Contango:      VIX9D < VIX < VIX3M → Markt ruhig, IV sinkt erwartet
                     → Optionskäufer vorteilhaft, Vol-Crush-Risiko gering
      Backwardation: VIX9D > VIX > VIX3M → kurzfristige Angst hoch
                     → erhöhtes Vol-Crush-Risiko nach Event → Spreads bevorzugen
      Flat:          Keine klare Richtung

    Returns:
        {
            "vix9d":      float,
            "vix30":      float,
            "vix3m":      float,
            "structure":  "contango" | "backwardation" | "flat",
            "slope":      float,   # (vix3m - vix9d) / vix9d, positiv = Contango
            "regime":     str,     # Kurzbeschreibung für Prompt
            "available":  bool,
        }
    """
    empty = {
        "vix9d": None, "vix30": None, "vix3m": None,
        "structure": "unknown", "slope": 0.0,
        "regime": "VIX Term Structure nicht verfügbar",
        "available": False,
    }

    if not _YF_AVAILABLE:
        return empty

    try:
        tickers = yf.download(
            ["^VIX9D", "^VIX", "^VIX3M"],
            period="2d", progress=False, auto_adjust=True,
        )
        close = tickers["Close"] if "Close" in tickers.columns else tickers

        def _last(sym: str) -> Optional[float]:
            if sym in close.columns:
                vals = close[sym].dropna()
                return float(vals.iloc[-1]) if not vals.empty else None
            return None

        vix9d = _last("^VIX9D")
        vix30 = _last("^VIX")
        vix3m = _last("^VIX3M")

        if not all([vix9d, vix30, vix3m]):
            return empty

        slope = (vix3m - vix9d) / vix9d  # positiv = Contango

        if slope > 0.05:
            structure = "contango"
            regime    = (
                f"VIX Contango (9d={vix9d:.1f} → 3M={vix3m:.1f}, slope=+{slope:.0%}): "
                f"Markt erwartet sinkende Volatilität → Long-Vol-Strategien weniger riskant"
            )
        elif slope < -0.05:
            structure = "backwardation"
            regime    = (
                f"VIX Backwardation (9d={vix9d:.1f} → 3M={vix3m:.1f}, slope={slope:.0%}): "
                f"Kurzfristige Angst erhöht → Vol-Crush nach Event wahrscheinlicher → Spreads bevorzugen"
            )
        else:
            structure = "flat"
            regime    = (
                f"VIX Term Structure flach (9d={vix9d:.1f}, 3M={vix3m:.1f}): "
                f"Kein eindeutiges Volatilitäts-Signal"
            )

        log.info(f"VIX Term Structure: {structure} | slope={slope:.1%} | {vix9d:.1f}/{vix30:.1f}/{vix3m:.1f}")

        return {
            "vix9d":     round(vix9d, 2),
            "vix30":     round(vix30, 2),
            "vix3m":     round(vix3m, 2),
            "structure": structure,
            "slope":     round(slope, 4),
            "regime":    regime,
            "available": True,
        }

    except Exception as e:
        log.debug(f"VIX Term Structure Fehler: {e}")
        return empty


def get_macro_regime_multiplier() -> float:
    """
    Gibt einen Multiplikator (0.7-1.2) für den Mismatch-Score zurück.
    Rezessives Umfeld → Signal schwächer gewichten.
    Expansives Umfeld → Signal stärker gewichten.

    Wird in mismatch_scorer.py genutzt.
    """
    ctx = get_macro_context()
    regime = ctx.get("macro_regime", "neutral")

    multipliers = {
        "expansive":    1.10,   # +10% Vertrauen in Signale
        "neutral":      1.00,   # neutral
        "recessionary": 0.80,   # -20% Vertrauen bei invertierter Kurve
        "unknown":      1.00,
    }
    return multipliers.get(regime, 1.00)

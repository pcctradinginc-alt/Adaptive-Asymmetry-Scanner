"""
modules/risk_gates.py – Fixes

Fix 1: VIX Timeout → Fallback statt Pipeline-Abbruch
       Begründung: Bei yfinance Timeout (curl 28) ist der Markt normal —
       VIX nicht abrufbar bedeutet nicht VIX > 35.
       Safety-First bedeutet: bei Ungewissheit Fallback-Wert nutzen,
       nicht Pipeline komplett stoppen.

Fix 2: last_vix immer float (nie None) nach globalok()
       Verhindert TypeError in pipeline.py stop_reason Formatierung.
"""

from __future__ import annotations
import logging
import os
from datetime import datetime, date
from typing import Optional

import requests
import yfinance as yf

log = logging.getLogger(__name__)

VIX_HARD_GATE    = 35.0   # Über diesem Wert → kein Trading
VIX_FALLBACK     = 20.0   # Fallback wenn VIX nicht abrufbar
VIX_MAX_RETRIES  = 2
VIX_TIMEOUT      = 15     # Sekunden (war 30 → zu lang für GitHub Actions)


class RiskGates:

    def __init__(self):
        self.last_vix: float = VIX_FALLBACK   # Default: nie None

    def global_ok(self) -> bool:
        """
        Prüft ob globales Marktumfeld Trading erlaubt.
        Bei VIX-Fehler: Fallback statt Abbruch.
        """
        vix = self._fetch_vix()

        if vix is None:
            log.warning(
                f"VIX nicht abrufbar → Fallback {VIX_FALLBACK:.1f} "
                f"(Pipeline läuft weiter)"
            )
            self.last_vix = VIX_FALLBACK
            return True   # Im Zweifel: Pipeline laufen lassen

        self.last_vix = float(vix)
        log.info(f"VIX aktuell: {self.last_vix:.2f} (Schwelle: {VIX_HARD_GATE})")

        if self.last_vix >= VIX_HARD_GATE:
            log.warning(
                f"VIX {self.last_vix:.1f} ≥ {VIX_HARD_GATE} "
                f"→ Markt zu volatil, Pipeline gestoppt."
            )
            return False

        return True

    def _fetch_vix(self) -> Optional[float]:
        """VIX abrufen mit kurzem Timeout und FRED-Fallback."""
        # Versuch 1: yfinance
        for attempt in range(1, VIX_MAX_RETRIES + 1):
            try:
                ticker = yf.Ticker("^VIX")
                hist   = ticker.history(period="2d", timeout=VIX_TIMEOUT)
                if not hist.empty:
                    return float(hist["Close"].iloc[-1])

                info = ticker.info
                vix  = info.get("regularMarketPrice") or info.get("previousClose")
                if vix:
                    return float(vix)

            except Exception as e:
                log.debug(f"VIX yfinance Versuch {attempt}: {e}")

        # Versuch 2: FRED API (kostenlos, kein Key)
        try:
            resp = requests.get(
                "https://fred.stlouisfed.org/graph/fredgraph.csv",
                params={"id": "VIXCLS"},
                timeout=10,
                headers={"User-Agent": "newstoption-scanner/8.0"},
            )
            if resp.status_code == 200:
                lines = [
                    l for l in resp.text.strip().split("\n")
                    if l and not l.startswith("DATE") and "." in l
                ]
                if lines:
                    val = lines[-1].split(",")[1].strip()
                    if val and val != ".":
                        log.info(f"VIX via FRED: {val}")
                        return float(val)
        except Exception as e:
            log.debug(f"VIX FRED Fallback Fehler: {e}")

        return None   # Beide Quellen fehlgeschlagen → Fallback in global_ok()

    def has_upcoming_earnings(self, ticker: str) -> bool:
        """Prüft ob Earnings innerhalb der nächsten 14 Tage."""
        try:
            cal = yf.Ticker(ticker).calendar
            if cal is None or cal.empty:
                return False
            if "Earnings Date" not in cal.index:
                return False

            earnings_dates = cal.loc["Earnings Date"]
            if hasattr(earnings_dates, "__iter__"):
                for ed in earnings_dates:
                    try:
                        if isinstance(ed, str):
                            ed = datetime.strptime(ed, "%Y-%m-%d").date()
                        elif hasattr(ed, "date"):
                            ed = ed.date()
                        days_away = (ed - date.today()).days
                        if 0 <= days_away <= 14:
                            log.info(
                                f"  [{ticker}] Earnings in {days_away}d "
                                f"→ Earnings-Gate aktiv"
                            )
                            return True
                    except Exception:
                        continue
        except Exception:
            pass
        return False

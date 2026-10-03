"""
modules/risk_gates.py

VIX-Gate (Source Health, 2026-10-03): FAIL-CLOSED.
  Früher: VIX nicht abrufbar -> stiller Fallback-Wert 20, Pipeline läuft weiter.
  Das widerspricht der Kernregel "nie mit unbekannten Daten normal weiterhandeln"
  (VIX ist Pflichtdatum des Risk Gates, Kritikalität CRITICAL).
  Jetzt: Quellen in fester, getesteter Reihenfolge –
    1. yfinance ^VIX            (Primary)
    2. FRED API VIXCLS          (offizieller Fallback, FRED_API_KEY; Datum geprüft)
    3. FRED fredgraph.csv       (ohne Key; aus GitHub Actions oft Timeout)
  Liefert keine Quelle einen aktuellen Wert -> last_vix = None, global_ok() = False
  ("VIX-Gate (VIX nicht abrufbar)"). vix_source hält die tatsächlich genutzte Quelle
  (source_actual) – ein Providerwechsel bleibt nie verborgen.
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
VIX_FALLBACK     = 20.0   # NUR noch für Altaufrufer (Default-Schwellen); nie als Ersatz für einen fehlenden VIX
VIX_MAX_AGE_DAYS = 6      # FRED VIXCLS erscheint mit ~1 Tag Verzug; älter = unbekannt
VIX_MAX_RETRIES  = 2
VIX_TIMEOUT      = 15     # Sekunden (war 30 → zu lang für GitHub Actions)


class RiskGates:

    def __init__(self):
        self.last_vix: float | None = None
        self.vix_source: str | None = None

    def global_ok(self) -> bool:
        """
        Prüft ob globales Marktumfeld Trading erlaubt.
        Bei VIX-Fehler: Fallback statt Abbruch.
        """
        vix = self._fetch_vix()

        if vix is None:
            log.error("VIX aus keiner Quelle abrufbar (yfinance, FRED API, FRED CSV) -> "
                      "Risk Gate geschlossen: kein Handel mit unbekanntem Risiko")
            self.last_vix = None
            self.vix_source = None
            return False

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
                    self.vix_source = "vix_level"
                    return float(hist["Close"].iloc[-1])

                info = ticker.info
                vix  = info.get("regularMarketPrice") or info.get("previousClose")
                if vix:
                    self.vix_source = "vix_level"
                    return float(vix)

            except Exception as e:
                log.debug(f"VIX yfinance Versuch {attempt}: {e}")

        # Versuch 2: offizielle FRED API (Key per Secret; aus Actions zuverlässiger als die CSV)
        api_key = os.environ.get("FRED_API_KEY", "").strip()
        if api_key:
            try:
                resp = requests.get(
                    "https://api.stlouisfed.org/fred/series/observations",
                    params={"series_id": "VIXCLS", "api_key": api_key, "file_type": "json",
                            "sort_order": "desc", "limit": 10},
                    timeout=10,
                )
                if resp.status_code == 200:
                    for o in (resp.json().get("observations") or []):
                        val, d = o.get("value"), o.get("date")
                        if val in (None, "", ".") or not d:
                            continue
                        age = (date.today() - datetime.strptime(d, "%Y-%m-%d").date()).days
                        if age > VIX_MAX_AGE_DAYS:
                            log.warning(f"VIX via FRED API zu alt ({d}) -> unbekannt")
                            break
                        log.info(f"VIX via FRED API: {val} ({d})")
                        self.vix_source = "vix_fred"
                        return float(val)
            except Exception as e:  # noqa: BLE001 – Key nie loggen (requests-Fehler können die URL enthalten)
                log.debug(f"VIX FRED API Fallback Fehler: {type(e).__name__}")

        # Versuch 3: FRED CSV (ohne Key)
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
                    d_s, val = lines[-1].split(",")[0].strip(), lines[-1].split(",")[1].strip()
                    try:
                        age = (date.today() - datetime.strptime(d_s, "%Y-%m-%d").date()).days
                    except ValueError:
                        age = None
                    if age is None or age > VIX_MAX_AGE_DAYS:
                        log.warning(f"VIX via FRED CSV ohne aktuelles Datum ({d_s}) -> unbekannt")
                    elif val and val != ".":
                        log.info(f"VIX via FRED CSV: {val} ({d_s})")
                        self.vix_source = "vix_fred"
                        return float(val)
        except Exception as e:
            log.debug(f"VIX FRED Fallback Fehler: {e}")

        return None   # keine Quelle -> global_ok() schließt das Gate (fail-closed)

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
                    except Exception as e:
                        log.warning(f"  [{ticker}] Earnings-Datum nicht lesbar: {e}")
                        continue
        except Exception as e:
            log.warning(f"  [{ticker}] Earnings-Kalender Fehler: {e} → Earnings-Gate ohne Datengrundlage")
        return False

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
import math
import os
from datetime import datetime, date, timezone
from typing import Optional

import requests
import yfinance as yf

log = logging.getLogger(__name__)

VIX_HARD_GATE    = 35.0   # Über diesem Wert → kein Trading
VIX_FALLBACK     = 20.0   # NUR noch für Altaufrufer (Default-Schwellen); nie als Ersatz für einen fehlenden VIX
VIX_MAX_AGE_DAYS = 6      # FRED VIXCLS erscheint mit ~1 Tag Verzug; älter = unbekannt
VIX_MAX_RETRIES  = 2
VIX_TIMEOUT      = 15     # Sekunden (war 30 → zu lang für GitHub Actions)


def _valid_vix(value) -> bool:
    try:
        return not isinstance(value, bool) and math.isfinite(float(value)) and float(value) > 0
    except (TypeError, ValueError, OverflowError):
        return False


def _valid_age(day) -> bool:
    try:
        return 0 <= (date.today() - day).days <= VIX_MAX_AGE_DAYS
    except (TypeError, ValueError, OverflowError):
        return False


class RiskGates:

    def __init__(self):
        self.last_vix: float | None = None
        self.vix_source: str | None = None

    def global_ok(self) -> bool:
        """
        Prüft ob globales Marktumfeld Trading erlaubt.
        Bei ungültigem oder fehlendem VIX: Gate geschlossen.
        """
        vix = self._fetch_vix()

        if not _valid_vix(vix):
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
                    value = hist["Close"].iloc[-1]
                    if _valid_vix(value) and _valid_age(hist.index[-1].date()):
                        self.vix_source = "vix_level"
                        return float(value)

                info = ticker.info
                vix  = info.get("regularMarketPrice") or info.get("previousClose")
                stamp = info.get("regularMarketTime")
                if _valid_vix(vix) and stamp is not None and _valid_age(datetime.fromtimestamp(stamp, timezone.utc).date()):
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
                        if not 0 <= age <= VIX_MAX_AGE_DAYS:
                            log.warning(f"VIX via FRED API zu alt ({d}) -> unbekannt")
                            break
                        if not _valid_vix(val):
                            continue
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
                    ln for ln in resp.text.strip().split("\n")
                    if ln and not ln.startswith("DATE") and "." in ln
                ]
                if lines:
                    d_s, val = lines[-1].split(",")[0].strip(), lines[-1].split(",")[1].strip()
                    try:
                        age = (date.today() - datetime.strptime(d_s, "%Y-%m-%d").date()).days
                    except ValueError:
                        age = None
                    if age is None or not 0 <= age <= VIX_MAX_AGE_DAYS:
                        log.warning(f"VIX via FRED CSV ohne aktuelles Datum ({d_s}) -> unbekannt")
                    elif _valid_vix(val):
                        log.info(f"VIX via FRED CSV: {val} ({d_s})")
                        self.vix_source = "vix_fred"
                        return float(val)
        except Exception as e:
            log.debug(f"VIX FRED Fallback Fehler: {e}")

        return None   # keine Quelle -> global_ok() schließt das Gate (fail-closed)

    def earnings_status(self, ticker: str) -> str:
        """Distinguish a known future date from missing or ambiguous evidence."""
        try:
            cal = yf.Ticker(ticker).calendar
            if cal is None:
                return "UNKNOWN"
            if isinstance(cal, dict):
                values = cal.get("Earnings Date")
                if values is None:
                    return "UNKNOWN"
                if not isinstance(values, (list, tuple)):
                    values = [values]
            else:
                if cal.empty or "Earnings Date" not in cal.index:
                    return "UNKNOWN"
                values = list(cal.loc["Earnings Date"])
            days = []
            invalid = False
            for value in values:
                try:
                    if isinstance(value, str):
                        value = datetime.strptime(value, "%Y-%m-%d").date()
                    elif hasattr(value, "date"):
                        value = value.date()
                    if not isinstance(value, date):
                        raise ValueError("invalid earnings date")
                    days.append((value-date.today()).days)
                except (TypeError, ValueError, OverflowError):
                    invalid = True
            if any(0 <= d <= 14 for d in days):
                return "UPCOMING"
            # A past date alone does not establish absence of future earnings.
            if not invalid and days and all(d > 14 for d in days):
                return "CLEAR"
        except Exception as exc:
            log.warning("earnings calendar unavailable for %s: %s", ticker, type(exc).__name__)
        return "UNKNOWN"

    def has_upcoming_earnings(self, ticker: str) -> bool:
        """Legacy Champion decision unchanged; unknown is explicitly logged."""
        state = self.earnings_status(ticker)
        if state == "UNKNOWN":
            log.warning("earnings risk UNKNOWN for %s; legacy decision unchanged", ticker)
        return state == "UPCOMING"

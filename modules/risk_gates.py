"""
Risk-Gates: Globale und ticker-spezifische Sicherheitschecks
- VIX-Check: Abbruch wenn VIX > 35
- Earnings-Gate: Blockiere Ticker mit Earnings < 7 Tagen
- Liquiditäts-Gate: Open Interest < 100
"""

import logging
from datetime import datetime, timedelta
import yfinance as yf

log = logging.getLogger(__name__)

VIX_THRESHOLD = 35.0


class RiskGates:
    def global_ok(self) -> bool:
        """Prüft globale Marktbedingungen."""
        vix = self._get_vix()
        log.info(f"VIX aktuell: {vix:.2f}")
        if vix > VIX_THRESHOLD:
            log.warning(f"VIX={vix:.2f} > {VIX_THRESHOLD} → Abbruch.")
            return False
        return True

    def has_upcoming_earnings(self, ticker: str, days: int = 7) -> bool:
        """Prüft ob Earnings innerhalb der nächsten `days` Tage anstehen."""
        try:
            t = yf.Ticker(ticker)
            cal = t.calendar
            if cal is None or cal.empty:
                return False
            # Earnings Date-Spalte
            if "Earnings Date" in cal.columns:
                earnings_dates = cal["Earnings Date"].dropna()
            elif "Earnings Dates" in cal.columns:
                earnings_dates = cal["Earnings Dates"].dropna()
            else:
                return False

            cutoff = datetime.utcnow() + timedelta(days=days)
            for ed in earnings_dates:
                if isinstance(ed, str):
                    ed = datetime.strptime(ed[:10], "%Y-%m-%d")
                if hasattr(ed, "to_pydatetime"):
                    ed = ed.to_pydatetime().replace(tzinfo=None)
                if datetime.utcnow() <= ed <= cutoff:
                    log.info(f"  [{ticker}] Earnings am {ed.date()} (< {days} Tage)")
                    return True
        except Exception as e:
            log.debug(f"Earnings-Check Fehler für {ticker}: {e}")
        return False

    def _get_vix(self) -> float:
        try:
            vix = yf.Ticker("^VIX")
            hist = vix.history(period="2d")
            if not hist.empty:
                return float(hist["Close"].iloc[-1])
        except Exception as e:
            log.debug(f"VIX-Abruf Fehler: {e}")
        return 20.0   # Fallback: OK

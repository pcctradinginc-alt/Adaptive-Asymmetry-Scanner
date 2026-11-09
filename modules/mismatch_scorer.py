"""
Stufe 5: Normalisierter Mismatch-Score (Quant-Validierung)

Fixes:
  H-03: Negative Mismatch-Werte (Markt hat überreagiert) wurden als "weak"
        klassifiziert und weiterverarbeitet. Fix: expliziter Filter, der
        Überreaktionen vor der Simulation aussortiert.
  M-01: EPS-Drift-Bins nutzten hardcodierte Werte statt config.yaml.
        Fix: cfg.eps_drift.* Thresholds.

  v8.2 FIX-A: price_move_48h wurde NIRGENDS im Produktionscode gesetzt.
        r_2d war IMMER 0 → Z-Score IMMER 0 → Mismatch = Impact.
        Fix: Mismatch-Scorer berechnet 48h-Move selbst via yfinance.

  v8.2 FIX-B: EPS-Drift wurde unter falschem Key gelesen.
        a.get("eps_drift", {}).get("drift", 0.0) → Feld existiert nie.
        Korrekter Pfad: a["data_validation"]["eps_cross_check"]["deviation_pct"]

  2026-11-09 (SHADOW, Aktionsplan Punkt 4): Der Produktions-Mismatch nutzt |48h-Move| relativ zum Scan-Tag –
        richtungsblind (eine bullische These nach −8 % gilt als "eingepreist") und nicht an der Nachricht
        verankert. Zusätzlich protokolliert (für JEDE analysierte These, auch die gefilterten, ohne Gate-Wirkung):
          mismatch_dir_48h  = impact − 5·max(0, s·move_48h)/σ          (s = +1 BULLISH, −1 BEARISH)
          mismatch_dir_news = impact − 5·max(0, s·move_seit_news)/σ    (Basis: letzter Schluss VOR der
                              ersten verwendeten Schlagzeile, NY-Börsenschluss 16:00; Ende wie Produktion)
        Gleiche Skalierung wie die Produktion, damit nur Richtung und Anker getestet werden (Challenger
        mismatch_directional_news in challengers.yaml).
"""

import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import yfinance as yf

from modules.config import cfg

_NY = ZoneInfo("America/New_York")

log = logging.getLogger(__name__)


def _bin_impact(impact: float) -> str:
    if impact <= 4:
        return "low"
    if impact <= 7:
        return "mid"
    return "high"


def _bin_mismatch(mismatch: float) -> str:
    """
    FIX H-03: Negative Werte bedeuten Überreaktion (Markt hat mehr reagiert
    als die News rechtfertigen). Diese werden NICHT mehr als "weak" behandelt
    – sie werden durch den expliziten Filter in _score() bereits entfernt.
    Der verbleibende Wertebereich ist immer ≥ 0.
    """
    if mismatch < 3:
        return "weak"
    if mismatch <= 6:
        return "good"
    return "strong"


def _bin_eps_drift(drift: float) -> str:
    """FIX M-01: Thresholds aus config.yaml statt hartcodiert."""
    abs_drift = abs(drift)
    if abs_drift > cfg.eps_drift.massive_threshold:
        return "massive"
    if abs_drift > cfg.eps_drift.relevant_threshold:
        return "relevant"
    return "noise"


class MismatchScorer:

    def __init__(self):
        self.data_missing: list[str] = []
        self.shadow: dict[str, dict] = {}     # Ticker -> gerichtete/news-verankerte Varianten (ohne Gate-Wirkung)

    def run(self, analyses: list[dict]) -> list[dict]:
        scored = []
        for a in analyses:
            result = self._score(a)
            if result:
                scored.append(result)
        self._note_shadow()
        return scored

    def _note_shadow(self) -> None:
        """Schattenwerte für ALLE analysierten Thesen in den Kandidaten-Ledger (auch Mismatch ≤ 0), damit der
        Challenger beide Definitionen auf derselben Population vergleichen kann. Nie lauf-brechend."""
        try:
            from modules import candidate_ledger
            for ticker, sh in self.shadow.items():
                candidate_ledger.note(ticker, **sh)
        except Exception as e:  # noqa: BLE001 – reine Observability
            log.warning(f"Mismatch-Schattenwerte nicht notiert (ignoriert): {e}")

    def _shadow_variants(self, a: dict, impact: float, move_signed: float, sigma: float, mismatch: float) -> dict:
        direction = str((a.get("deep_analysis") or {}).get("direction") or "BULLISH").upper()
        s = -1.0 if direction == "BEARISH" else 1.0
        out = {"mismatch_raw": round(mismatch, 3),
               "mismatch_dir_48h": round(impact - 5.0 * max(0.0, s * move_signed) / sigma, 3)}
        first = (a.get("news_ts") or {}).get("first")
        if first:
            since = self._compute_move_since(a["ticker"], first)
            if since is not None:
                move, base_day, days = since
                out.update({"move_since_news": round(move, 4), "news_anchor_close": base_day,
                            "news_anchor_sessions": days,
                            "mismatch_dir_news": round(impact - 5.0 * max(0.0, s * move) / sigma, 3)})
        return out

    def _compute_move_since(self, ticker: str, first_iso: str):
        """(Rendite vom letzten Schluss VOR der Nachricht bis zum letzten abgeschlossenen Schluss, Basis-Tag,
        Anzahl Sitzungen) oder None. Eine Nachricht nach 16:00 New York zählt zum Folgetag."""
        try:
            ts = datetime.fromisoformat(str(first_iso).replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            local = ts.astimezone(_NY)
            hist = yf.Ticker(ticker).history(period="15d")
            close = hist["Close"]
            if hasattr(close, "squeeze"):
                close = close.squeeze()
            if len(close) < 3:
                return None
            days = [d.date() if hasattr(d, "date") else d for d in close.index]
            end = len(close) - 2                                   # wie Produktion: letzter abgeschlossener Tag
            cutoff = local.date() if local.hour >= 16 else None
            base = max((i for i, d in enumerate(days) if (d <= cutoff if cutoff else d < local.date())),
                       default=None)
            if base is None or base > end or not float(close.iloc[base]) > 0:
                return None
            move = float(close.iloc[end]) / float(close.iloc[base]) - 1.0
            return move, str(days[base]), end - base
        except Exception as e:  # noqa: BLE001 – Schattenwert, nie lauf-brechend
            log.warning(f"  [{ticker}] Bewegung seit Nachricht nicht berechenbar: {e}")
            return None

    def _score(self, a: dict) -> dict | None:
        ticker = a["ticker"]
        da     = a.get("deep_analysis", {})
        impact = da.get("impact", 0)

        # ── FIX v8.2-A: 48h-Move selbst berechnen ────────────────────────────
        # Vorher: r_2d = abs(a.get("price_move_48h", 0))
        # Problem: "price_move_48h" wurde NIRGENDS im Produktionscode gesetzt.
        #          Nur in Tests als Mock-Daten vorhanden.
        #          → r_2d war IMMER 0 → Z-Score IMMER 0 → Mismatch = Impact.
        # Jetzt:   Eigene Berechnung, gleiche Logik wie deep_analysis v8.2.
        # Fehlende Kursdaten NIE als 0-Move werten: 0 bedeutet "Markt hat noch
        # nicht reagiert" -> Mismatch = Impact = stärkstes Signal. Ein
        # Datenfehler würde so zum Top-Kandidaten (Audit 2026-09-29).
        move_signed = self._compute_48h_move(ticker)
        if move_signed is None:
            log.warning(f"  [{ticker}] Keine 48h-Kursdaten → verworfen (kein 0-Default)")
            self.data_missing.append(ticker)
            return None
        r_2d = abs(move_signed)

        sigma = self._compute_sigma(ticker)
        if sigma == 0:
            log.warning(f"  [{ticker}] σ30d = 0, übersprungen.")
            return None

        z_score  = r_2d / sigma
        mismatch = impact - (z_score * 5)
        self.shadow[ticker] = self._shadow_variants(a, impact, move_signed, sigma, mismatch)

        # FIX H-03: Negative Mismatch explizit filtern.
        # Negatives Mismatch = Markt hat MEHR reagiert als die News rechtfertigen
        # = Überreaktion. Das ist das Gegenteil des gesuchten Signals (Underreaction).
        # Solche Ticker werden nicht weiter verarbeitet.
        if mismatch <= 0:
            log.info(
                f"  [{ticker}] Mismatch={mismatch:.2f} ≤ 0 "
                f"(Impact={impact}, Z={z_score:.2f}, 48h-Move={r_2d:.3f}) "
                f"→ Markt hat überreagiert, gefiltert."
            )
            return None

        # ── FIX v8.2-B: EPS-Drift aus korrektem Pfad lesen ───────────────────
        # Vorher: eps_drift_val = a.get("eps_drift", {}).get("drift", 0.0)
        # Problem: Feld "eps_drift" mit Subkey "drift" existiert nie.
        #          data_validator.py schreibt unter:
        #          candidate["data_validation"]["eps_cross_check"]["deviation_pct"]
        # Jetzt:   Korrekter Zugriffspfad.
        eps_check     = a.get("data_validation", {}).get("eps_cross_check", {})
        eps_drift_val = eps_check.get("deviation_pct", 0.0) or 0.0

        features = {
            "impact":        impact,
            "surprise":      da.get("surprise", 0),
            "mismatch":      round(mismatch, 3),
            "z_score":       round(z_score, 3),
            "sigma_30d":     round(sigma, 4),
            "price_move_48h": round(r_2d, 4),   # NEU: für Reports & Debugging
            # Forschungsfeatures (Audit 2026-09-29, KEINE Gate-Wirkung):
            # z_score teilt einen 2-Tages-Move durch die TÄGLICHE σ (korrekt
            # wäre σ·√2), abs() ignoriert die Richtung. Beide Varianten werden
            # mitgeloggt, damit der Faktor-Monitor sie gegen Forward-Returns prüft.
            "price_move_48h_signed": round(move_signed, 4),
            "z_score_2d_scaled":     round(r_2d / (sigma * np.sqrt(2)), 3),
            "eps_drift":     round(eps_drift_val, 4),
            "bin_impact":    _bin_impact(impact),
            "bin_mismatch":  _bin_mismatch(mismatch),
            "bin_eps_drift": _bin_eps_drift(eps_drift_val),
        }

        log.info(
            f"  [{ticker}] Mismatch={mismatch:.2f} "
            f"Z={z_score:.2f} σ={sigma:.4f} "
            f"48h-Move={r_2d:.3f} EPS-Drift={eps_drift_val:.4f}"
        )

        return {**a, "features": features}

    def _compute_sigma(self, ticker: str) -> float:
        """30-Tages Standardabweichung der täglichen Returns."""
        try:
            hist = yf.Ticker(ticker).history(period="35d")
            if len(hist) < 10:
                return 0.0
            returns = hist["Close"].pct_change().dropna()
            return float(np.std(returns))
        except Exception as e:
            log.warning(f"Sigma-Berechnung Fehler für {ticker}: {e}")
            return 0.0

    def _compute_48h_move(self, ticker: str) -> float:
        """
        Berechnet die Preisbewegung der letzten 2 vollen Handelstage.

        Nutzt period='10d' und vergleicht volle Handelstage:
          close[-2] = gestriger Close (letzter abgeschlossener Tag)
          close[-4] = vor-3-Tage-Close (48h-Fenster)

        Gleiche Logik wie deep_analysis.py v8.2 _get_48h_move().
        """
        try:
            hist  = yf.Ticker(ticker).history(period="10d")
            close = hist["Close"]
            if hasattr(close, "iloc"):
                close = close.squeeze()
            if len(close) < 5:
                return None
            base = float(close.iloc[-4])
            if not base > 0:
                return None
            return float((close.iloc[-2] - base) / base)
        except Exception as e:
            log.warning(f"  [{ticker}] 48h-Move nicht berechenbar: {e}")
            return None

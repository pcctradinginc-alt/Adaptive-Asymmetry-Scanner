"""
modules/mirofish_simulation.py v8.2

Monte Carlo Simulation mit historischer yfinance-Kalibrierung.

Kernidee:
    sigma (Volatilität) kommt aus echten historischen Preisdaten.
    base_alpha (Signal-Drift) = historische Drift + Signal-Bonus.
    Das macht Hit-Rates realistisch und ticker-spezifisch.

v8.2 Kalibrierungs-Fix:
    Problem: Hit-Rates lagen bei 97% für praktisch jedes Signal.
             Der Cap (85%/95%) griff IMMER → alle Signale sahen gleich aus.

    Ursache: max_signal_alpha=0.004 → bei Impact=6/Surprise=5 akkumulierten
             sich ~19% Alpha-Drift über 120 Tage. Target war nur +8%.
             → Trivial erreichbar, auch bei schwachen Signalen.

    Fix (4 Stellschrauben):
      1. max_signal_alpha: 0.004 → 0.001 (realistischer Informationsvorsprung)
      2. NARRATIVE_DECAY: verdoppelt (News-Signal verblasst schneller)
      3. HIT_RATE_CAP: 85%/95% → 75% (kein einzelnes Signal > 75%)
      4. Target: volatilitäts-adaptiv statt pauschal +8%
         Formel: target = current * (1 + max(0.08, 0.5 * σ * √days))
         → Volatile Aktien brauchen grösseren Move für "Hit"

    Erwartete Hit-Rates nach Fix: 35-70% je nach Signal-Stärke und Vola.
"""

from __future__ import annotations
import logging
import math
import numpy as np
import yfinance as yf
from functools import lru_cache

log = logging.getLogger(__name__)

# ── v8.2: Verdoppelte Decay-Rates ────────────────────────────────────────────
# Vorher: short=0.015, medium=0.008, long=0.004
# Begründung: Empirisch zerfallen News-Signale schneller als v8.0 annahm.
# Typischer Nachrichtenzyklus: 3-5 Tage Peak, danach rapider Abfall.
NARRATIVE_DECAY = {
    "short":  0.030,   # Signal verpufft in ~30 Tagen (war: ~65 Tage)
    "medium": 0.016,   # Signal verpufft in ~60 Tagen (war: ~125 Tage)
    "long":   0.008,   # Signal verpufft in ~120 Tagen (war: ~250 Tage)
}

DEFAULT_SIGMA   = 0.020   # Fallback wenn yfinance nicht erreichbar
QUICK_MC_PATHS  = 5_000
FINAL_MC_PATHS  = 10_000

# ── v8.2: Einheitlicher Cap auf 75% ─────────────────────────────────────────
# Vorher: Short=85%, Long=95%
# Begründung: Kein einzelnes News-Signal rechtfertigt >75% Konfidenz.
# Selbst perfekte Insider-Information hat Execution-Risiko, Makro-Risiko,
# und Timing-Unsicherheit. 75% ist die Obergrenze für "sehr starkes Signal".
HIT_RATE_CAP_SHORT = 0.75
HIT_RATE_CAP_LONG  = 0.75

# ── v8.2: Reduzierter Signal-Alpha ──────────────────────────────────────────
# Vorher: 0.004 → bei Impact=6/Surprise=5 = 0.22%/Tag = ~19% über 120d
# Jetzt:  0.001 → bei Impact=6/Surprise=5 = 0.055%/Tag = ~4.5% über 120d
# Begründung: Ein News-Signal gibt keinen 0.4%/Tag-Vorsprung.
# Selbst der stärkste fundamentale Katalysator liefert ~0.1%/Tag Alpha.
MAX_SIGNAL_ALPHA = 0.001


@lru_cache(maxsize=128)
def _get_hist_params(ticker: str) -> tuple[float, float]:
    """
    Holt historische Volatilität und Drift von yfinance.
    Gecacht pro Ticker — wird nur einmal pro Run abgerufen.
    Nutzt yf.download() für effizientere Batch-Verarbeitung.
    
    Returns:
        (sigma, mu) — tägliche Vola und täglicher Drift
    """
    try:
        hist = yf.download(ticker, period="6mo", progress=False, auto_adjust=True)
        if hist.empty or len(hist) < 30:
            log.debug(f"  [{ticker}] Hist-Daten zu wenig → Default sigma")
            return DEFAULT_SIGMA, 0.0

        close = hist["Close"]
        if hasattr(close, "iloc"):
            close = close.squeeze()  # MultiIndex → Series

        returns = close.pct_change().dropna()
        sigma   = float(returns.std())
        mu      = float(returns.mean())

        # Sanity checks
        if not (0.005 <= sigma <= 0.15):
            sigma = DEFAULT_SIGMA
        if not (-0.005 <= mu <= 0.005):
            mu = 0.0

        log.debug(f"  [{ticker}] Hist: σ={sigma:.4f} μ={mu:.5f} ({len(returns)} Tage)")
        return sigma, mu

    except Exception as e:
        log.debug(f"  [{ticker}] yfinance Hist-Fehler: {e} → Default")
        return DEFAULT_SIGMA, 0.0


def preload_hist_params(tickers: list[str]) -> None:
    """
    Lädt historische Parameter für alle Ticker parallel vor.
    Spart Zeit weil MC-Runs sofort auf Cache treffen.
    Wird in pipeline.py nach Hard-Filter aufgerufen.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    log.info(f"Preload historische Daten für {len(tickers)} Ticker...")
    
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(_get_hist_params, t): t for t in tickers}
        done = 0
        for f in as_completed(futures):
            done += 1
            try:
                f.result()
            except Exception:
                pass
    log.info(f"  Historische Daten geladen ({done} Ticker gecacht)")


def _compute_dynamic_target(current: float, sigma: float, days: int) -> float:
    """
    v8.2: Volatilitäts-adaptives Kursziel.

    Vorher: pauschal current * 1.08 (+8% für alle)
    Problem: +8% ist für eine Low-Vol-Aktie (σ=0.01) ein grosser Move,
             aber für eine High-Vol-Aktie (σ=0.04) trivial.

    Jetzt: target = current * (1 + max(0.08, 0.5 * σ * √days))
      - Low-Vol  (σ=0.01, 120d): max(0.08, 0.055) = 8.0%  (unchanged)
      - Normal   (σ=0.02, 120d): max(0.08, 0.110) = 11.0%
      - High-Vol (σ=0.04, 120d): max(0.08, 0.219) = 21.9%

    Ergebnis: Volatile Aktien brauchen stärkere Signale um hohe Hit-Rates
    zu erreichen. Das verhindert falsch-positive "starke" Signale bei
    naturgemäss volatilen Tech/Biotech-Titeln.
    """
    sigma_move = 0.5 * sigma * math.sqrt(days)
    target_pct = max(0.08, sigma_move)
    return current * (1.0 + target_pct)


class MirofishSimulation:

    def __init__(self):
        self.rng = np.random.default_rng()

    def run_for_dte(self, candidate: dict, days_to_expiry: int = 120) -> dict | None:
        ticker   = candidate.get("ticker", "")
        features = candidate.get("features", {}) or {}
        da       = candidate.get("deep_analysis", {}) or {}
        sim_data = candidate.get("simulation", {}) or {}

        current = float(
            sim_data.get("current_price") or
            candidate.get("current_price") or
            candidate.get("info", {}).get("currentPrice") or
            candidate.get("info", {}).get("regularMarketPrice") or 0
        )
        if current <= 0:
            try:
                info    = yf.Ticker(ticker).info
                current = float(
                    info.get("currentPrice") or
                    info.get("regularMarketPrice") or
                    info.get("previousClose") or 0
                )
            except Exception:
                pass
        if current <= 0:
            log.warning(f"  [{ticker}] Kein Preis verfügbar → MC übersprungen")
            return None

        # ── Historische Parameter von yfinance ───────────────────────────────
        sigma, hist_mu = _get_hist_params(ticker)

        # ── Signal-Alpha (News-Drift) ─────────────────────────────────────────
        impact   = float(da.get("impact", 5) or 5)
        surprise = float(da.get("surprise", 5) or 5)
        ttm      = da.get("time_to_materialization", "medium") or "medium"

        # TTM-String normalisieren (Claude gibt manchmal verschiedene Formate)
        ttm_lower = ttm.lower()
        if "woche" in ttm_lower or "4-8" in ttm_lower:
            ttm_key = "short"
        elif "6 monat" in ttm_lower:
            ttm_key = "long"
        else:
            ttm_key = "medium"

        # Geometrisches Mittel: beide Dimensionen müssen stark sein
        signal_strength = math.sqrt((impact / 10.0) * (surprise / 10.0))

        # v8.2: Signal-Bonus reduziert — max 0.1%/Tag bei perfektem Signal
        signal_alpha = signal_strength * MAX_SIGNAL_ALPHA

        # Gesamt-Alpha: historischer Trend + Signal-Bonus
        base_alpha = hist_mu + signal_alpha
        decay_rate = NARRATIVE_DECAY.get(ttm_key, 0.016)

        # v8.2: Volatilitäts-adaptives Target
        target  = _compute_dynamic_target(current, sigma, days_to_expiry)
        target_pct = (target / current - 1.0) * 100

        n_paths   = QUICK_MC_PATHS if days_to_expiry <= 45 else FINAL_MC_PATHS
        threshold = 0.45 if days_to_expiry <= 45 else 0.50

        log.info(
            f"  [{ticker}] MC-Kalibrierung: "
            f"σ={sigma:.3f} μ={hist_mu:.4f} "
            f"signal={signal_strength:.2f} α={base_alpha:.5f} "
            f"target=+{target_pct:.1f}% decay={decay_rate:.3f} "
            f"({days_to_expiry}d, {n_paths} Pfade)"
        )

        # ── Monte Carlo ───────────────────────────────────────────────────────
        paths_hit = 0
        for _ in range(n_paths):
            price = current
            hit   = False
            for d in range(days_to_expiry):
                # Alpha nimmt exponentiell ab (Narrative verblasst)
                daily_alpha = base_alpha * math.exp(-decay_rate * d)
                daily_ret   = daily_alpha + sigma * self.rng.standard_normal()
                price      *= (1.0 + daily_ret)
                if price >= target:
                    hit = True
                    break
            if hit:
                paths_hit += 1

        hit_rate = paths_hit / n_paths

        # v8.2: Einheitlicher Cap auf 75%
        hit_rate_cap = HIT_RATE_CAP_SHORT if days_to_expiry <= 45 else HIT_RATE_CAP_LONG
        if hit_rate >= hit_rate_cap:
            log.warning(
                f"  [{ticker}] Hit-Rate={hit_rate:.1%} → Cap auf {hit_rate_cap:.0%}"
            )
            hit_rate = hit_rate_cap

        stderr = math.sqrt(hit_rate * (1 - hit_rate) / n_paths)

        log.info(
            f"  [{ticker}] Simulation ({days_to_expiry}d, {n_paths} Pfade): "
            f"Hit={hit_rate:.1%} ±{stderr:.1%} "
            f"({'PASS' if hit_rate >= threshold else 'FAIL'})"
        )

        if hit_rate < threshold:
            log.info(f"  [{ticker}] Hit-Rate {hit_rate:.1%} < {threshold:.0%} → verworfen")
            return None

        return {
            **candidate,
            "simulation": {
                "current_price": round(current, 2),
                "target_price":  round(target, 2),
                "hit_rate":      round(hit_rate, 4),
                "n_paths":       n_paths,
                "days":          days_to_expiry,
                "sigma":         round(sigma, 4),
                "alpha":         round(base_alpha, 5),
            }
        }

    def _get_market_params(self, ticker: str) -> tuple[float, float, float]:
        """Für ROI Pre-Check: gibt (sigma, current, target) zurück."""
        sigma, _ = _get_hist_params(ticker)
        try:
            info    = yf.Ticker(ticker).info
            current = float(
                info.get("currentPrice") or
                info.get("regularMarketPrice") or 0
            )
        except Exception:
            current = 0.0
        target = _compute_dynamic_target(current, sigma, 120)
        return sigma, current, target


def compute_time_value_efficiency(roi_net: float, dte: int) -> dict:
    """ROI pro Tag und annualisierter ROI (nur wenn realistisch)."""
    dte = max(int(dte or 1), 1)
    roi_per_day = (roi_net / dte) * 100  # in %

    try:
        ann = float((1 + roi_net) ** (365 / dte) - 1)
        # Nur anzeigen wenn realistisch (<500% p.a.)
        ann_roi = round(ann, 4) if abs(ann) < 5.0 else None
    except Exception:
        ann_roi = None

    return {
        "roi_per_day_pct": round(roi_per_day, 4),
        "annualized_roi":  ann_roi,
        "dte":             dte,
    }

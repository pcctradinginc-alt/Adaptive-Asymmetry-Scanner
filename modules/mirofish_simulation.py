"""
modules/mirofish_simulation.py v8.3

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

v8.3 Options-P&L Monte Carlo (Phase 1):
    Neue Methoden: _generate_gbm_paths, _black_scholes_call,
                   _compute_mismatch_drift, simulate_option_pnl.
    simulate_option_pnl berechnet expected_pnl_pct via Haltepunkt-Repricing
    (~45% der Laufzeit) mit festem IV-Crush. Theta und Vega wirken realistisch.
"""

from __future__ import annotations
import logging
import math
import numpy as np
from scipy.stats import norm as _scipy_norm
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

    def run_for_dte(
        self,
        candidate:    dict,
        days_to_expiry: int            = 120,
        min_hit_rate:   float | None   = None,  # Override interner Threshold (Pre-MC)
    ) -> dict | None:
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
        # min_hit_rate überschreibt den internen Threshold (genutzt für Pre-MC Gate
        # mit niedrigerer Schwelle, ohne das Quick/Final-MC zu beeinflussen)
        threshold = min_hit_rate if min_hit_rate is not None else (
            0.45 if days_to_expiry <= 45 else 0.50
        )

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

    # ── Phase 1: Options-P&L Monte Carlo ─────────────────────────────────────

    def _generate_gbm_paths(
        self,
        S0:      float,
        mu:      float,
        sigma:   float,
        T:       float,
        n_paths: int,
        n_steps: int,
    ) -> np.ndarray:
        """Vektorisierte GBM-Pfade. Gibt Array (n_paths, n_steps+1) zurück."""
        if n_steps <= 0:
            return np.full((n_paths, 1), S0)
        dt        = T / n_steps
        Z         = self.rng.standard_normal((n_paths, n_steps))
        drift     = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * math.sqrt(dt) * Z
        paths     = S0 * np.exp(np.cumsum(drift + diffusion, axis=1))
        return np.hstack([np.full((n_paths, 1), S0), paths])

    def _black_scholes_call(
        self,
        S:     np.ndarray,
        K:     float,
        T:     float,
        sigma: float,
    ) -> np.ndarray:
        """Vektorisierter Black-Scholes Call (r=0)."""
        if T <= 0.001:
            return np.maximum(S - K, 0.0)
        d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return S * _scipy_norm.cdf(d1) - K * _scipy_norm.cdf(d2)

    def _compute_mismatch_drift(self, candidate: dict) -> float:
        """
        Drift für GBM-Pfade: hist_mu + signal_alpha.
        Identische Formel wie run_for_dte() — keine Inkonsistenz.
        """
        ticker   = candidate.get("ticker", "")
        _, hist_mu = _get_hist_params(ticker) if ticker else (DEFAULT_SIGMA, 0.0)
        da       = candidate.get("deep_analysis", {}) or {}
        impact   = float(da.get("impact",   5) or 5)
        surprise = float(da.get("surprise", 5) or 5)
        strength = math.sqrt((impact / 10.0) * (surprise / 10.0))
        return hist_mu + strength * MAX_SIGNAL_ALPHA

    def simulate_option_pnl(
        self,
        candidate:       dict,
        option:          dict,
        days_to_expiry:  int,
        n_paths:         int   = 5000,
        iv_crush_factor: float = 0.22,
    ) -> dict:
        """
        Phase 1: Options-P&L via Haltepunkt-Repricing (~45% der Laufzeit).
        IV-Crush und Theta-Decay haben bei verbleibender Restlaufzeit vollen Effekt.

        Returns dict mit expected_pnl_pct, hit_rate, pnl_std, hold_days.
        Bei ungültigen Eingaben: dict mit 'error'-Key.
        """
        try:
            current_price = float(candidate["simulation"]["current_price"])
            strike        = float(option["strike"])
            entry_price   = float(option.get("ask") or option.get("last") or 0)
            entry_iv      = float(option.get("implied_vol") or 0.30)
        except (KeyError, TypeError, ValueError):
            return {"expected_pnl_pct": 0.0, "hit_rate": 0.0, "error": "invalid_input"}

        if entry_price <= 0 or days_to_expiry < 3 or current_price <= 0:
            return {"expected_pnl_pct": 0.0, "hit_rate": 0.0, "error": "invalid_input"}

        # sigma_30d und hist_mu aus _get_hist_params() sind TÄGLICH.
        # _generate_gbm_paths erwartet annualisierte Werte (dt = 1/365 year).
        mu_daily    = self._compute_mismatch_drift(candidate)
        sigma_daily = float(candidate.get("features", {}).get("sigma_30d") or DEFAULT_SIGMA)
        mu          = mu_daily * 252
        sigma       = sigma_daily * math.sqrt(252)

        # Haltepunkt: 45% der Laufzeit — Theta und IV-Crush sind noch voll wirksam
        hold_days   = max(3, int(days_to_expiry * 0.45))
        T_total     = days_to_expiry / 365.0
        paths       = self._generate_gbm_paths(
            current_price, mu, sigma, T_total, n_paths, days_to_expiry
        )

        mid_S       = paths[:, hold_days]
        remaining_T = (days_to_expiry - hold_days) / 365.0
        final_iv    = max(entry_iv * (1.0 - iv_crush_factor), 0.05)

        repriced     = self._black_scholes_call(mid_S, strike, remaining_T, final_iv)
        pnl_per_path = (repriced - entry_price) / entry_price

        log.debug(
            f"  [{candidate.get('ticker', '?')}] MC-P&L: "
            f"hold={hold_days}d remaining_T={remaining_T:.3f}y "
            f"IV {entry_iv:.1%}→{final_iv:.1%} "
            f"median={float(np.median(pnl_per_path)):.1%}"
        )

        return {
            "expected_pnl_pct": float(np.median(pnl_per_path)),
            "mean_pnl_pct":     float(np.mean(pnl_per_path)),
            "hit_rate":         float(np.mean(paths[:, -1] > strike)),
            "pnl_std":          float(np.std(pnl_per_path)),
            "iv_crush_used":    round(iv_crush_factor, 2),
            "hold_days":        hold_days,
            "paths":            n_paths,
            "entry_price":      round(entry_price, 4),
        }


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

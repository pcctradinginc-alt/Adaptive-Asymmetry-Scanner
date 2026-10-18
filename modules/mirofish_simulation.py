"""
modules/mirofish_simulation.py v8.4

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

v8.4 Stochastische IV – Phase 2:
    Neue Methoden: _generate_iv_paths (OU-Prozess), _calibrate_ou_from_history
                   (lineare Regression), _get_ou_parameters (2-Layer Hybrid),
                   _log_iv_today (IV-Logger).
    simulate_option_pnl: IV-Crush jetzt stochastisch via OU statt festem Faktor.
    Hybrid: ≥30 eigene IV-Tage → Regression, sonst Heuristik (IV-Rank basiert).

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


# Herkunft der MC-Parameter je Ticker (Audit 2026-09-29): "measured",
# "clamped" (Messwert außerhalb 0.5–15 %/Tag), "default_no_data",
# "default_error". Nur bei "measured" simuliert run_for_dte (keine
# Trefferquote aus erfundener Volatilität für ein Gate; clamped seit 2026-10-17:
# MRNA σ=16,6 %/Tag gemessen, simuliert wurde mit 2 % -> Pre-MC bestanden).
PARAM_SOURCE: dict[str, str] = {}


def bs_call_r0(S: float, K: float, T: float, sigma: float) -> float:
    """Black-Scholes-Call (r=0, ohne Dividende) – dieselbe Konvention wie die MC-Bewertung."""
    if T <= 0.001 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    return float(S * _scipy_norm.cdf(d1) - K * _scipy_norm.cdf(d1 - sigma * math.sqrt(T)))


def implied_vol_r0(price: float, S: float, K: float, T: float) -> float | None:
    """Implizite Vola eines Calls in der r=0-Konvention (Bisektion); None, wenn der Preis außerhalb der
    arbitragefreien Grenzen liegt (unter innerem Wert oder über dem Kurs)."""
    if not (price > 0 and S > 0 and K > 0 and T > 0) or price <= max(S - K, 0.0) or price >= S:
        return None
    lo, hi = 0.01, 4.0
    if not (bs_call_r0(S, K, T, lo) <= price <= bs_call_r0(S, K, T, hi)):
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        if bs_call_r0(S, K, T, mid) > price:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def bull_call_spread_terms(option: dict, S: float, T: float) -> dict | None:
    """Kennzahlen eines Bull Call Spreads aus dem option-Dict (Long-Bein + spread_leg): Debit (Long-Ask − Short-Bid),
    Breite, IV beider Beine aus ihren Mids (r=0, wie die Bewertung; sonst Chain-IV) und die Spanne der Kombination.
    None, wenn kein gültiges Short-Bein darüber liegt (dann ist die Position ein Long Call)."""
    leg = option.get("spread_leg") or {}
    try:
        k1, k2 = float(option.get("strike") or 0), float(leg.get("strike") or 0)
        debit = float(option.get("net_debit") or 0)
        lb, la = float(option.get("bid") or 0), float(option.get("ask") or 0)
        sb, sa = float(leg.get("bid") or 0), float(leg.get("ask") or 0)
    except (TypeError, ValueError):
        return None
    if not (k2 > k1 > 0 and debit > 0 and la > 0 and sa > 0):
        return None
    chain_iv = float(option.get("implied_vol") or 0.30)
    iv_long = implied_vol_r0((lb + la) / 2, S, k1, T) or chain_iv
    iv_short = implied_vol_r0((sb + sa) / 2, S, k2, T) or iv_long
    return {"k1": k1, "k2": k2, "width": k2 - k1, "debit": debit, "iv_long": iv_long, "iv_short": iv_short,
            "combo_spread": max((la - sb) - (lb - sa), 0.0)}


def _signal_terms(deep_analysis: dict | None) -> tuple[float, float, float]:
    """(signal_strength, signal_alpha je Tag, decay_rate) aus der Deep Analysis – eine Regel für run_for_dte und
    die V3-Shadow-Drift der Options-P&L."""
    da       = deep_analysis or {}
    impact   = float(da.get("impact", 5) or 5)
    surprise = float(da.get("surprise", 5) or 5)
    ttm      = da.get("time_to_materialization", "medium") or "medium"

    # TTM-String normalisieren (Claude gibt manchmal verschiedene Formate)
    ttm_lower = str(ttm).lower()
    if "woche" in ttm_lower or "4-8" in ttm_lower:
        ttm_key = "short"
    elif "6 monat" in ttm_lower:
        ttm_key = "long"
    else:
        ttm_key = "medium"

    # Geometrisches Mittel: beide Dimensionen müssen stark sein
    signal_strength = math.sqrt((impact / 10.0) * (surprise / 10.0))
    # v8.2: Signal-Bonus reduziert — max 0.1%/Tag bei perfektem Signal
    return signal_strength, signal_strength * MAX_SIGNAL_ALPHA, NARRATIVE_DECAY.get(ttm_key, 0.016)


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
            log.warning(f"  [{ticker}] Hist-Daten zu wenig ({len(hist)}) → keine MC-Parameter")
            PARAM_SOURCE[ticker] = "default_no_data"
            return DEFAULT_SIGMA, 0.0

        close = hist["Close"]
        if hasattr(close, "iloc"):
            close = close.squeeze()  # MultiIndex → Series

        returns = close.pct_change().dropna()
        sigma   = float(returns.std())
        mu      = float(returns.mean())

        # Sanity checks
        PARAM_SOURCE[ticker] = "measured"
        if not (0.005 <= sigma <= 0.15):
            log.warning(f"  [{ticker}] σ={sigma:.4f} außerhalb 0.5–15 %/Tag → keine MC-Trefferquote")
            PARAM_SOURCE[ticker] = "clamped"
            sigma = DEFAULT_SIGMA
        if not (-0.005 <= mu <= 0.005):
            mu = 0.0

        log.debug(f"  [{ticker}] Hist: σ={sigma:.4f} μ={mu:.5f} ({len(returns)} Tage)")
        return sigma, mu

    except Exception as e:
        log.warning(f"  [{ticker}] yfinance Hist-Fehler: {e} → keine MC-Parameter")
        PARAM_SOURCE[ticker] = "default_error"
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
        self.shadow_rng = np.random.default_rng()   # eigener Strom: Shadow-Rechnungen verschieben keine Produktionsziehung

    def run_for_dte(
        self,
        candidate:    dict,
        days_to_expiry: int            = 120,
        min_hit_rate:   float | None   = None,  # Override interner Threshold (Pre-MC)
    ) -> dict | None:
        ticker   = candidate.get("ticker", "")
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
        sigma_source = PARAM_SOURCE.get(ticker, "measured")
        if sigma_source != "measured":
            log.warning(f"  [{ticker}] MC übersprungen: σ nicht gemessen/plausibel ({sigma_source}) "
                        f"-> keine Trefferquote aus Default-σ")
            return None

        # ── Signal-Alpha (News-Drift) ─────────────────────────────────────────
        signal_strength, signal_alpha, decay_rate = _signal_terms(da)

        # Gesamt-Alpha: historischer Trend + Signal-Bonus
        base_alpha = hist_mu + signal_alpha

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

        # V4 (SHADOW, 2026-10-18): Tages-σ und -α stammen aus Handelstagen, simuliert wurden aber `days_to_expiry`
        # Schritte (Kalendertage, z. B. Options-DTE) -> Streuung um √(365/252) zu hoch. Hier dieselbe
        # Berührwahrscheinlichkeit desselben Ziels mit Handelstagen; entscheidet nichts, nur zum Vergleich.
        try:
            shadow_v4 = self.shadow_hit_rate_trading_days(current, target, sigma, base_alpha, decay_rate,
                                                          days_to_expiry, n_paths, hit_rate_cap)
        except Exception as e:  # noqa: BLE001 – Shadow darf die Produktion nie stören
            log.debug(f"  [{ticker}] V4-Shadow Fehler (ignoriert): {e}")
            shadow_v4 = None

        log.info(
            f"  [{ticker}] Simulation ({days_to_expiry}d, {n_paths} Pfade): "
            f"Hit={hit_rate:.1%} ±{stderr:.1%} "
            f"({'PASS' if hit_rate >= threshold else 'FAIL'})"
        )

        # run_for_dte ist ein reiner Simulator und entscheidet NICHT mehr selbst über
        # PASS/FAIL. Der Caller wendet seine eigene (ggf. VIX-adaptive) Schwelle an und
        # labelt korrekt. Frueher fuehrte das interne `return None` bei niedriger Hit-Rate
        # dazu, dass die Pipeline es als "mc_no_result" (= kein Preis) fehl-labelte und in
        # High-VIX-Phasen die interne 0.45 strenger war als die Pipeline-Schwelle (0.42).
        # Einziges legitimes None: kein Preis verfuegbar (oben).
        return {
            **candidate,
            "simulation": {
                "current_price": round(current, 2),
                "target_price":  round(target, 2),
                "hit_rate":      round(hit_rate, 4),
                "n_paths":       n_paths,
                "days":          days_to_expiry,
                "sigma":         round(sigma, 4),
                "sigma_source":  sigma_source,
                "alpha":         round(base_alpha, 5),
                "shadow_v4":     shadow_v4,
            }
        }

    def shadow_hit_rate_trading_days(self, current: float, target: float, sigma: float, base_alpha: float,
                                     decay_rate: float, calendar_days: int, n_paths: int,
                                     cap: float | None = None) -> dict:
        """V4-Shadow: Berührwahrscheinlichkeit des Ziels in calendar_days Kalendertagen, simuliert mit
        round(calendar_days × 252/365) Handelstagen (gleiche arithmetische Tagesrendite und Alpha-Decay je
        Handelstag wie run_for_dte). Vektorisiert, eigener Zufallsstrom."""
        steps = max(1, int(round(calendar_days * 252 / 365)))
        alpha = base_alpha * np.exp(-decay_rate * np.arange(steps))
        rets = alpha + sigma * self.shadow_rng.standard_normal((n_paths, steps))
        hit = float(np.mean((current * np.cumprod(1.0 + rets, axis=1) >= target).any(axis=1)))
        return {"hit_rate": round(min(hit, cap) if cap is not None else hit, 4),
                "hit_rate_uncapped": round(hit, 4), "steps": steps, "calendar_days": int(calendar_days),
                "unit": "trading_days", "version": "v4-shadow-1"}

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

    # ── Phase 2: Stochastische IV (OU) + Hybrid-Kalibrierung ────────────────────

    def _generate_iv_paths(
        self,
        n_paths:    int,
        n_steps:    int,
        current_iv: float,
        ou_params:  dict,
    ) -> np.ndarray:
        """Ornstein-Uhlenbeck Prozess für stochastische IV-Pfade. Shape: (n_paths, n_steps+1)."""
        dt      = 1.0 / 365.0
        kappa   = float(ou_params["kappa"])
        theta   = float(ou_params["theta"])
        sigma_v = float(ou_params["sigma_v"])

        iv_paths       = np.zeros((n_paths, n_steps + 1))
        iv_paths[:, 0] = current_iv

        for t in range(1, n_steps + 1):
            dW             = self.rng.standard_normal(n_paths) * math.sqrt(dt)
            drift          = kappa * (theta - iv_paths[:, t - 1]) * dt
            iv_paths[:, t] = iv_paths[:, t - 1] + drift + sigma_v * dW
            iv_paths[:, t] = np.maximum(iv_paths[:, t], 0.05)

        return iv_paths

    def _calibrate_ou_from_history(
        self, iv_series: list, dt: float = 1.0 / 365.0
    ) -> dict | None:
        """Lineare Regression für OU-Parameter (κ, θ, σ_v). Mindestens 25 Datenpunkte."""
        if len(iv_series) < 25:
            return None

        iv = np.array(iv_series, dtype=float)
        x  = iv[:-1]
        dx = np.diff(iv)

        A              = np.vstack([np.ones_like(x), x]).T
        coeffs, _, _, _ = np.linalg.lstsq(A, dx, rcond=None)
        a, b           = float(coeffs[0]), float(coeffs[1])

        kappa = -b / dt
        theta = a / (-b) if abs(b) > 1e-8 else float(np.mean(iv))

        # R² aus direkt berechneten Residuen (lstsq-Rückgabe ist bei 2D nicht zuverlässig)
        fitted = a + b * x
        ss_res = float(np.sum((dx - fitted) ** 2))
        ss_tot = float(np.sum((dx - dx.mean()) ** 2))
        r2     = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

        sigma_v = float(np.std(dx - fitted)) / math.sqrt(dt)

        kappa   = max(0.3, min(kappa,   5.0))
        theta   = max(0.08, min(theta,  0.85))
        sigma_v = max(0.04, min(sigma_v, 0.60))

        return {
            "kappa":   round(kappa,   3),
            "theta":   round(theta,   4),
            "sigma_v": round(sigma_v, 4),
            "method":  "regression",
            "n_days":  len(iv_series),
            "r2":      round(r2, 3),
        }

    def _get_ou_parameters(
        self,
        ticker:     str,
        history:    dict,
        current_iv: float,
        iv_rank:    float,
    ) -> dict:
        """2-Layer Hybrid: Regression (≥30 Tage eigene Daten) oder Heuristik."""
        iv_history = history.get("iv_history", {}).get(ticker, [])
        iv_values  = [e["atm_iv"] for e in iv_history[-120:]]

        if len(iv_values) >= 30:
            ou = self._calibrate_ou_from_history(iv_values)
            if ou is not None:
                log.info(
                    f"  [{ticker}] OU-Regression ({ou['n_days']}d, R²={ou['r2']:.2f}) → "
                    f"κ={ou['kappa']} θ={ou['theta']:.1%} σv={ou['sigma_v']:.1%}"
                )
                return ou

        # Heuristik-Fallback
        theta   = current_iv
        kappa   = 1.8 if iv_rank >= 70 else (1.2 if iv_rank >= 40 else 0.8)
        sigma_v = round(theta * 0.28, 4)

        log.debug(f"  [{ticker}] OU-Heuristik ({len(iv_values)} Tage) → κ={kappa} θ={theta:.1%}")
        return {
            "kappa":   round(kappa,   3),
            "theta":   round(theta,   4),
            "sigma_v": sigma_v,
            "method":  "heuristic",
            "n_days":  len(iv_values),
        }

    def _log_iv_today(self, ticker: str, atm_iv: float, history: dict) -> None:
        """Speichert die heutige ATM-IV in history['iv_history'][ticker]. Kein Duplikat pro Tag."""
        from datetime import datetime as _dt
        today   = _dt.utcnow().strftime("%Y-%m-%d")
        iv_hist = history.setdefault("iv_history", {})
        entries = iv_hist.setdefault(ticker, [])

        if entries and entries[-1].get("date") == today:
            return  # bereits für heute geloggt

        entries.append({"date": today, "atm_iv": round(float(atm_iv), 4)})
        iv_hist[ticker] = entries[-120:]   # max. 120 Tage behalten

    def simulate_option_pnl(
        self,
        candidate:       dict,
        option:          dict,
        days_to_expiry:  int,
        history:         dict  = None,
        n_paths:         int   = 5000,
        iv_crush_factor: float = 0.22,   # Phase-1-Kompatibilität, in Phase 2 ignoriert
        iv_rank:         float = 50.0,
    ) -> dict:
        """
        Phase 2: Stochastische IV via OU-Prozess + 2-Layer Hybrid-Kalibrierung.
        iv_crush_factor wird ignoriert (OU übernimmt), bleibt für Rückwärtskompatibilität.
        """
        try:
            current_price = float(candidate["simulation"]["current_price"])
            strike        = float(option["strike"])
            entry_price   = float(option.get("ask") or option.get("last") or 0)
            entry_iv      = float(option.get("implied_vol") or 0.30)
        except (KeyError, TypeError, ValueError):
            return {"expected_pnl_pct": 0.0, "hit_rate": 0.0, "error": "invalid_input"}

        if entry_price <= 0 or days_to_expiry < 5 or current_price <= 0:
            return {"expected_pnl_pct": 0.0, "hit_rate": 0.0, "error": "invalid_input"}

        # V1 (2026-10-18): Bull Call Spread als Spread bewerten – Einstieg zum Debit, Long- minus Short-Call auf
        # denselben Pfaden, Wert auf die Breite begrenzt. Vorher Long-Call ab Long-Ask: Gate-ROI über dem Maximalertrag
        # (BBY 171 % > 138 %, ADM 56 % > 53 %, KDP 96 % > 82 %).
        spread = bull_call_spread_terms(option, current_price, days_to_expiry / 365.0)
        if spread:
            entry_price = spread["debit"]

        # sigma_30d / hist_mu sind TÄGLICH — annualisieren für GBM (dt = 1/365 year)
        mu_daily    = self._compute_mismatch_drift(candidate)
        sigma_daily = float(candidate.get("features", {}).get("sigma_30d") or DEFAULT_SIGMA)
        mu          = mu_daily * 252
        sigma       = sigma_daily * math.sqrt(252)

        # OU-Parameter (2-Layer Hybrid)
        ticker = candidate.get("ticker", "")
        ou     = self._get_ou_parameters(ticker, history or {}, entry_iv, iv_rank)

        hold_days   = max(5, days_to_expiry // 2)
        T_total     = days_to_expiry / 365.0

        price_paths = self._generate_gbm_paths(current_price, mu, sigma, T_total, n_paths, days_to_expiry)
        iv_paths    = self._generate_iv_paths(n_paths, days_to_expiry, entry_iv, ou)

        mid_S       = price_paths[:, hold_days]
        mid_iv      = iv_paths[:, hold_days]     # per-Pfad IV am Haltepunkt
        remaining_T = (days_to_expiry - hold_days) / 365.0

        def _reprice(S_hold: np.ndarray) -> np.ndarray:
            if spread:
                # IV beider Beine folgt dem OU-Pfad mit ihrem Abstand zur Chain-IV beim Einstieg (Skew bleibt)
                iv_l = np.maximum(mid_iv + (spread["iv_long"] - entry_iv), 0.05)
                iv_s = np.maximum(mid_iv + (spread["iv_short"] - entry_iv), 0.05)
                return np.clip(self._black_scholes_call(S_hold, strike, remaining_T, iv_l)
                               - self._black_scholes_call(S_hold, spread["k2"], remaining_T, iv_s),
                               0.0, spread["width"])
            return self._black_scholes_call(S_hold, strike, remaining_T, mid_iv)

        repriced     = _reprice(mid_S)
        pnl_per_path = (repriced - entry_price) / entry_price

        # V3 (SHADOW, 2026-10-18): dieselben Pfade mit der Drift-Regel von run_for_dte ohne 6-Monats-Trend –
        # nur Signal-Alpha, abklingend je Handelstag. Produktiv: konstant hist_mu + Signal-Alpha (kein Decay;
        # hist_mu erklärt 73–90 % der Drift, t ≈ 0,6). Entscheidet nichts, nur zum Vergleich.
        shadow_v3 = None
        try:
            _, sig_alpha, decay = _signal_terms(candidate.get("deep_analysis"))
            dt = T_total / days_to_expiry
            mu_shadow = sig_alpha * 252 * np.exp(-decay * np.arange(hold_days) * 252 / 365)
            cum_prod, cum_shadow = mu * dt * hold_days, float(np.sum(mu_shadow) * dt)
            pnl_v3 = (_reprice(mid_S * math.exp(cum_shadow - cum_prod)) - entry_price) / entry_price
            shadow_v3 = {"mean_pnl_pct": round(float(np.mean(pnl_v3)), 4),
                         "median_pnl_pct": round(float(np.median(pnl_v3)), 4),
                         "cum_drift_prod": round(cum_prod, 4), "cum_drift_shadow": round(cum_shadow, 4),
                         "version": "v3-shadow-1"}
        except Exception as e:  # noqa: BLE001 – Shadow darf die Produktion nie stören
            log.debug(f"  [{ticker}] V3-Shadow Fehler (ignoriert): {e}")

        iv_change_pct = float(np.mean(mid_iv) / entry_iv - 1) if entry_iv > 0 else 0.0

        log.debug(
            f"  [{ticker}] MC-P&L P2 ({ou['method']}): "
            f"hold={hold_days}d IV Ø {entry_iv:.1%}→{float(np.mean(mid_iv)):.1%} "
            f"median={float(np.median(pnl_per_path)):.1%}"
        )

        return {
            "expected_pnl_pct": float(np.median(pnl_per_path)),
            "mean_pnl_pct":     float(np.mean(pnl_per_path)),
            "hit_rate":         float(np.mean(price_paths[:, -1] > strike)),
            "pnl_std":          float(np.std(pnl_per_path)),
            "iv_crush_used":    round(iv_change_pct, 4),
            "ou_method":        ou["method"],
            "ou_n_days":        ou["n_days"],
            "hold_days":        hold_days,
            "paths":            n_paths,
            "entry_price":      round(entry_price, 4),
            "instrument":       "BULL_CALL_SPREAD" if spread else "LONG_CALL",
            "shadow_v3":        shadow_v3,
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

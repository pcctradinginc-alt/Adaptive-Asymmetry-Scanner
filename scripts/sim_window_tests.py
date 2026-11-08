"""Simulation: Testgröße und Power der Fenstertests unter überlappenden Labels (docs/STATISTICAL_POWER_2026-11-08.md).

A Alternierung (bisher, bleibt) | B HAC Lag 1, df n-1 | C Schranke 2s²/n | D HAC, df n/2-1 | E Schranke, df n/2-1.
Aufruf: python scripts/sim_window_tests.py  (Laufzeit einige Minuten; nur Forschung, schreibt nichts)
"""
import numpy as np
from scipy import stats

rng = np.random.default_rng(1)

def blocks(n_blocks, h, mu, a, b, reps):
    D = n_blocks * h
    u = rng.standard_normal((reps, D + h))
    # IC_t = mu + a * mean of next h market shocks (overlapping labels) + b * own noise
    cs = np.cumsum(u, axis=1)
    over = (cs[:, h:D + h] - cs[:, 0:D]) / np.sqrt(h)
    ic = mu + a * over + b * rng.standard_normal((reps, D))
    return ic.reshape(reps, n_blocks, h).mean(axis=2)          # block means, consecutive (1-dependent)

def tests(B, alpha):
    n = B.shape[1]
    out = {}
    # A: current - every second block
    A = B[:, ::2]; nA = A.shape[1]
    t = A.mean(1) / (A.std(1, ddof=1) / np.sqrt(nA)); out["A alternierend"] = stats.t.sf(t, nA - 1) < alpha
    m = B.mean(1); s2 = B.var(1, ddof=1)
    Bc = B - m[:, None]; c1 = (Bc[:, 1:] * Bc[:, :-1]).sum(1) / n
    v_hac = (s2 + 2 * np.maximum(c1, 0)) / n
    t = m / np.sqrt(v_hac); out["B HAC1 df=n-1"] = stats.t.sf(t, n - 1) < alpha
    out["D HAC1 df=n/2-1"] = stats.t.sf(t, n // 2 - 1) < alpha
    t = m / np.sqrt(2 * s2 / n); out["C Schranke 2s2 df=n-1"] = stats.t.sf(t, n - 1) < alpha
    out["E Schranke 2s2 df=n/2-1"] = stats.t.sf(t, n // 2 - 1) < alpha
    return {k: v.mean() for k, v in out.items()}

for alpha in (0.05, 0.01, 0.002):
    for n in (8, 16, 24):
        for a, b in ((1.0, 0.0), (1.0, 1.0), (0.3, 1.0)):
            B = blocks(n, 5, 0.0, a, b, 200_000)
            r = tests(B, alpha)
            print(f"SIZE a={alpha} n={n:2d} overlap={a},{b}: " + " | ".join(f"{k}: {v/alpha:.2f}x" for k, v in r.items()))

print("--- POWER ---")
for alpha in (0.01, 0.002):
    for n in (16, 24, 48):
        for mu in (0.1, 0.2, 0.3):
            B = blocks(n, 5, mu, 1.0, 1.0, 60_000)
            r = tests(B, alpha)
            print(f"POWER a={alpha} n={n} mu={mu}: " + " | ".join(f"{k}: {v:.2f}" for k, v in r.items()))

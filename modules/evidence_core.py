"""modules/evidence_core.py – gemeinsame Validierungs-Statistik der Evidence Engine.

Eine Implementierung statt vieler Varianten (docs/ARCHITECTURE_MAP.md, Befund 9) für Research-/Lernpfade:
* group_bootstrap_ci   – CI des Mittelwerts mit Block-Bootstrap über Gruppen (Signaltage, Ereignis-Cluster).
  Abhängige Beobachtungen desselben Tages/Ereignisses werden nie einzeln gezogen.
* purged_walk_forward  – zeitlich saubere Folds: Training nur aus Beobachtungen, deren Label VOR dem Testbeginn
  (minus Embargo) feststand (Purging überlappender Labels + Embargo).
* bonferroni_alpha     – Mehrfachtest-Kontrolle über alle geprüften Challenger.

Die Statistik des PromotionControllers bleibt bewusst unverändert. Sie ist ein eingefrorenes Produktions-Gate,
und jede Änderung würde die Evidenz registrierter Verträge verschieben.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np


def group_bootstrap_ci(values, groups, *, n: int = 2000, seed: int = 47, alpha: float = 0.05,
                       two_sided: bool = True) -> tuple[float | None, float | None]:
    """(lo, hi) des Mittelwerts aller Beobachtungen; Ziehung ganzer Gruppen mit Zurücklegen (vektorisiert).
    < 2 Gruppen -> (None, None). two_sided=False -> einseitige Grenzen bei alpha bzw. 1−alpha."""
    by: dict = {}
    for v, g in zip(values, groups):
        if v is None or not np.isfinite(float(v)):
            continue
        by.setdefault(g, []).append(float(v))
    if len(by) < 2:
        return None, None
    sums = np.array([sum(v) for v in by.values()])
    cnts = np.array([len(v) for v in by.values()], dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(by), size=(int(n), len(by)))
    means = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
    q = (alpha / 2, 1 - alpha / 2) if two_sided else (alpha, 1 - alpha)
    lo, hi = np.quantile(means, q)
    return round(float(lo), 6), round(float(hi), 6)


def bonferroni_alpha(alpha: float, n_tests: int) -> float:
    return float(alpha) / max(1, int(n_tests))


def _d(x) -> date:
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    return date.fromisoformat(str(x)[:10])


def purged_walk_forward(decision_times, label_ends, *, n_folds: int = 4, embargo_days: int = 5,
                        min_train: int = 30, min_test: int = 10) -> list[dict]:
    """Expandierende Walk-Forward-Folds nach Zeit.
    Test-Fold k = k-tes Zeitsegment der (sortierten) Entscheidungszeitpunkte. Training = alle Beobachtungen, deren
    Label-Ende < Testbeginn − embargo_days (Purging: kein Label überlappt den Testzeitraum; Embargo: Abstand gegen
    serielle Abhängigkeit). Folds mit zu wenig Training/Test werden ausgelassen (nie aufgefüllt).
    -> [{"fold", "test_start", "test_end", "train_idx", "test_idx"}]"""
    t = [_d(x) for x in decision_times]
    e = [_d(x) for x in label_ends]
    order = sorted(range(len(t)), key=lambda i: t[i])
    uniq = sorted(set(t))
    if len(uniq) < n_folds + 1:
        return []
    bounds = np.array_split(np.array(uniq[len(uniq) // (n_folds + 1):], dtype=object), n_folds)
    out = []
    for k, seg in enumerate(bounds):
        if not len(seg):
            continue
        ts, te = seg[0], seg[-1]
        test = [i for i in order if ts <= t[i] <= te]
        cut = ts - timedelta(days=int(embargo_days))
        train = [i for i in order if e[i] < cut and t[i] < ts]
        if len(train) >= min_train and len(test) >= min_test:
            out.append({"fold": k, "test_start": ts.isoformat(), "test_end": te.isoformat(),
                        "train_idx": train, "test_idx": test})
    return out

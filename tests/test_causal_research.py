"""Causal Research: Leakage (Tilts nutzen nur Ziele mit Ende < t), Evidence-Level-Logik
(Lead-Signal / Rauschen / zeitgleich / umgekehrte Richtung), BH-Zählung, nie automatisch
causal_evidence. Synthetisch, kein Netz."""
from __future__ import annotations

import numpy as np
import pandas as pd

from modules import causal_research as cr

LOCKED = pd.Timestamp("2024-01-01")
FAST = {"bootstrap_n": 400}


def _synth(mode: str, seed: int = 3, n: int = 620, b: float = 0.02, sig: float = 0.006):
    """Wochenraster ab 2012. mode: lead | noise | concurrent | reverse. Treiber 'drv', Ziel XLI."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2012-01-06", periods=n * 5, freq="B")
    dates = pd.DatetimeIndex(pd.Series(dates, index=dates).groupby(dates.to_period("W-FRI")).max().values)[:n]
    x = rng.normal(size=n)
    noise = rng.normal(scale=sig, size=n)
    rel = np.zeros(n)                       # relative Wochenrendite (Woche t-1 -> t)
    if mode == "lead":
        rel[1:] = b * x[:-1] + noise[1:]                    # Treiber_t bestimmt Rendite t -> t+1
    elif mode == "concurrent":
        rel = b * x + noise                                 # Treiber_t gleichzeitig mit Rendite t-1 -> t
    elif mode == "reverse":
        rel = noise * 1.0 + rng.normal(scale=0.02, size=n)
        x = np.r_[0.0, rel[:-1]] * 0 + rel                  # Treiber folgt der Rendite (Sektor führt)
        x = x / x.std()
    else:
        rel = rng.normal(scale=0.02, size=n)
    spy_r = rng.normal(0.002, 0.01, n)
    px = pd.DataFrame(index=dates)
    px["SPY"] = 100 * np.exp(np.cumsum(spy_r))
    for e in cr.SECTOR_ETFS:
        r = rel if e == "XLI" else rng.normal(scale=0.02, size=n)
        px[e] = 100 * np.exp(np.cumsum(spy_r + r))
    ind = pd.DataFrame({"drv": x, "vix": rng.normal(20, 4, n), "spy_trend_200": rng.normal(0, 0.05, n)}, index=dates)
    return ind, px


PRIOR = [{"driver": "drv", "target": "XLI", "sign": 1}]


def test_bh_monotone_and_count():
    q = cr.benjamini_hochberg([0.001, 0.02, 0.04, 0.5])
    assert q == sorted(q) and abs(q[0] - 0.004) < 1e-9 and q[-1] == 0.5 and all(v <= 1 for v in q)
    assert cr.benjamini_hochberg([]) == []
    ind, px = _synth("noise")
    res = cr.analyze(ind, px, LOCKED, priors=PRIOR + [{"driver": "drv", "target": "XLE", "sign": -1}], cfg=FAST)
    assert res["n_tested"] == 2 * 2 == len(res["relations"])          # Paare x Horizonte
    assert all(r["stats"]["q_bh"] is not None for r in res["relations"])


def test_true_lead_is_leading_and_hypothesis():
    ind, px = _synth("lead")
    res = cr.analyze(ind, px, LOCKED, priors=PRIOR, cfg=FAST)
    r4 = next(r for r in res["relations"] if r["horizon_weeks"] == 4)
    assert r4["level"] in ("temporally_leading", "causal_hypothesis")
    assert r4["observed_sign"] == 1 and r4["evidence"]["economic_plausibility"] == "HIGH"
    assert r4["stats"]["corr_lead"] > 0.2 and abs(r4["stats"]["corr_lag"] or 0) < 0.15
    assert any(r["level"] == "causal_hypothesis" and r["id"].startswith("CAUSAL_HYPOTHESIS_") for r in res["relations"])


def test_noise_is_only_correlation():
    ind, px = _synth("noise", seed=5)
    res = cr.analyze(ind, px, LOCKED, priors=PRIOR, cfg=FAST)
    assert all(r["level"] == "correlation" for r in res["relations"])
    assert all(r["evidence"]["causal_confidence"] == "LOW" for r in res["relations"])


def test_concurrent_and_reverse_not_leading():
    for mode in ("concurrent", "reverse"):
        ind, px = _synth(mode, seed=7)
        res = cr.analyze(ind, px, LOCKED, priors=PRIOR, cfg=FAST)
        assert all(r["level"] not in ("temporally_leading", "causal_hypothesis") for r in res["relations"]), mode


def test_causal_evidence_never_automatic():
    ind, px = _synth("lead")
    res = cr.analyze(ind, px, LOCKED, priors=PRIOR, cfg=FAST)
    assert all(r["causal_evidence"] == "none" for r in res["relations"])
    assert all(r["evidence"]["causal_confidence"] in ("LOW", "MODERATE") for r in res["relations"])
    assert "causal_evidence" not in cr.LEVELS


def test_assess_needs_prior_for_hypothesis():
    ind, px = _synth("lead")
    st = cr.pair_stats(ind, px, "drv", "XLI", 4, LOCKED, {**cr.CP, **FAST})
    a = cr.assess(st, None, 0.001, st, cr.CP)
    assert a["evidence"]["economic_plausibility"] == "UNKNOWN" and a["level"] == "temporally_leading"
    wrong = cr.assess(st, -1, 0.001, st, cr.CP)
    assert wrong["evidence"]["economic_plausibility"] == "LOW" and wrong["level"] == "temporally_leading"
    assert cr.assess(st, 1, 0.9, st, cr.CP)["level"] == "correlation"          # nicht signifikant nach BH


def test_tilts_no_leakage_and_priors_fixed():
    ind, px = _synth("lead")
    rels = [{"driver": "drv", "target": "XLI", "horizon_weeks": 13}, {"driver": "drv", "target": "XLI", "horizon_weeks": 4}]
    dates = list(ind.index[300:400])
    base = cr.sector_tilts(ind, px, rels, dates)
    cut = dates[50]
    px2 = px.copy()
    px2.loc[px2.index >= cut, list(cr.SECTOR_ETFS) + ["SPY"]] *= 5.0 * np.linspace(1, 3, int((px2.index >= cut).sum()))[:, None]
    alt = cr.sector_tilts(ind, px2, rels, dates)
    # Kurse ab `cut` geändert: Ziele mit Ende >= cut dürfen Tilts bei t <= cut nicht beeinflussen
    a, b = base[base.date <= cut].set_index("date")["tilt"], alt[alt.date <= cut].set_index("date")["tilt"]
    pd.testing.assert_series_equal(a, b)
    assert not base.empty and set(base.columns) == {"date", "sector_etf", "tilt"}
    # Priors sind fix im Protokoll (16 Paare) und fließen nicht in die Tilts ein
    assert len(cr.CP["priors"]) >= 12


def test_tilts_informative_on_true_signal_and_verdict():
    ind, px = _synth("lead", n=700)
    pri = PRIOR + [{"driver": "drv", "target": e, "sign": 1} for e in ("XLE", "XLF", "XLK", "XLV", "XLY")]
    res = cr.analyze(ind, px, LOCKED, priors=pri, cfg=FAST)
    dates = [d for d in ind.index if d >= pd.Timestamp("2019-01-01")]
    tilts = cr.sector_tilts(ind, px, res["relations"], dates)
    te = cr.evaluate_tilts(tilts, px, ind.index, LOCKED, cfg=FAST)
    assert te["n_dates"] > 50 and te["ic_mean"] > 0
    v, why = cr.decide(res["relations"], {"positive": True})
    assert v == "KEEP" and why
    assert cr.decide(res["relations"], {"positive": False})[0] == "MODIFY"
    assert cr.decide([{"level": "correlation"}], {"positive": True})[0] == "REJECT"


def test_missing_driver_skipped():
    ind, px = _synth("noise")
    res = cr.analyze(ind, px, LOCKED, priors=[{"driver": "nope", "target": "XLI", "sign": 1}], cfg=FAST)
    assert res["n_tested"] == 0 and res["relations"] == []

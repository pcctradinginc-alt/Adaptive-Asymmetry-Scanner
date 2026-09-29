"""Volumen-Event-Studie: Event-Erkennung, keine Zukunftsdaten in Features,
Entry Open_{t+1}, Training nur mit abgeschlossenen Outcomes, Kennzahlen,
Entscheidungsregel. Synthetische Daten, kein Netzwerk."""
from __future__ import annotations

import numpy as np
import pandas as pd

from modules import price_event_study as pes


def _frame(n=400, seed=1, spike_at=(), drift_after=0.0):
    rnd = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    r = rnd.normal(0, 0.01, n)
    for i in spike_at:
        r[i] = 0.05
        for k in range(1, 21):
            if i + k < n:
                r[i + k] += drift_after
    close = 50 * np.cumprod(1 + r)
    open_ = close / (1 + rnd.normal(0, 0.002, n))
    vol = np.full(n, 1_000_000.0)
    for i in spike_at:
        vol[i] = 5_000_000.0
    return pd.DataFrame({"Open": open_, "Close": close, "Volume": vol}, index=idx)


def _spy(n=400):
    idx = pd.bdate_range("2015-01-01", periods=n)
    c = pd.Series(np.linspace(200, 220, n), index=idx)
    return pd.DataFrame({"Open": c, "Close": c})


def test_events_detected_and_deduplicated():
    f = _frame(spike_at=(100, 102, 200))
    ev = pes.build_events({"AAA": f}, _spy())
    assert list(ev["date"]) == [f.index[100], f.index[200]]          # 102 innerhalb 5 Tage entdoppelt
    assert (ev["relvol"] >= pes.RELVOL_MIN).all()


def test_features_do_not_use_future_data():
    f = _frame(spike_at=(150,))
    ev1 = pes.build_events({"AAA": f}, _spy())
    g = f.copy()
    g.iloc[160:, :] = g.iloc[160:, :] * 3.0                           # Zukunft (ab t+10) massiv ändern
    ev2 = pes.build_events({"AAA": g}, _spy())
    for col in ("ev_ret_adj", "z2", "rs35", "relvol", "sigma20"):
        assert np.isclose(ev1[col].iloc[0], ev2[col].iloc[0]), col
    assert not np.isclose(ev1["fwd_20"].iloc[0], ev2["fwd_20"].iloc[0])


def test_entry_is_next_open_and_market_adjusted():
    f = _frame(spike_at=(150,))
    spy = _spy()
    ev = pes.build_events({"AAA": f}, spy, horizons=(5,))
    t = 150
    exp = (f["Close"].iloc[t + 5] / f["Open"].iloc[t + 1] - 1) - \
          (spy["Close"].iloc[t + 5] / spy["Open"].iloc[t + 1] - 1)
    assert np.isclose(ev["fwd_5"].iloc[0], exp)
    assert ev["exit_5"].iloc[0] == f.index[t + 5]


def _synthetic_events(drift, years=range(2015, 2025), per_year=300, seed=3):
    rnd = np.random.default_rng(seed)
    rows = []
    for y in years:
        dates = pd.bdate_range(f"{y}-01-02", f"{y}-12-20")
        for i in range(per_year):
            d = dates[rnd.integers(len(dates))]
            s = rnd.choice([-1, 1])
            rows.append({"date": d, "ticker": f"T{i%50}", "ev_ret_adj": s * 0.03,
                         "z2": rnd.normal(0, 2), "rs35": rnd.normal(0, 0.05), "relvol": 2 + rnd.exponential(1),
                         "sigma20": 0.02, "vix": rnd.choice([15, 25]), "trend_up": bool(rnd.integers(2)),
                         "fwd_20": s * drift + rnd.normal(0, 0.05), "exit_20": d + pd.Timedelta(days=28),
                         **{f"fwd_{h}": s * drift + rnd.normal(0, 0.05) for h in (1, 5, 60)},
                         **{f"exit_{h}": d + pd.Timedelta(days=int(h * 1.4) + 1) for h in (1, 5, 60)}})
    ev = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    ev["year"] = pd.to_datetime(ev["date"]).dt.year
    return ev


def test_real_drift_is_supported_and_noise_is_not():
    strong = pes.evaluate(_synthetic_events(drift=0.02))
    assert strong["hypotheses"]["H1_drift"]["decision"]["supported"]
    noise = pes.evaluate(_synthetic_events(drift=0.0, seed=9))
    assert not noise["hypotheses"]["H1_drift"]["decision"]["supported"]


def test_costs_can_kill_small_edge():
    small = pes.evaluate(_synthetic_events(drift=0.003, per_year=3000))
    d = small["hypotheses"]["H1_drift"]
    assert d["h20"]["stress_cost"]["mean"] < d["h20"]["base_cost"]["mean"]


def test_walk_forward_training_excludes_unfinished_outcomes():
    ev = _synthetic_events(drift=0.01)
    ev.loc[ev["year"] == 2018, "exit_20"] = pd.Timestamp("2030-01-01")   # nicht abgeschlossen
    seen = {}
    orig = pes.hypothesis_trades

    def spy(name, train, test, h):
        seen[int(test["year"].iloc[0])] = train
        return orig(name, train, test, h)
    pes.hypothesis_trades = spy
    try:
        pes.walk_forward(ev, "H2_small_move_gate", 20)
    finally:
        pes.hypothesis_trades = orig
    for y, train in seen.items():
        assert (pd.to_datetime(train["exit_20"]) < pd.Timestamp(f"{y}-01-01")).all()
        assert (train["year"] < y).all()

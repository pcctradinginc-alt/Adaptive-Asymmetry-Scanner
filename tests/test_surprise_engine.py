"""Surprise Engine: PIT-Reaktionsfenster, Training-only-Schwellen, Robustheit, Verdikte (synthetisch)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from modules import surprise_engine as se

NY = "America/New_York"


def test_fund_surprise_floor_clip_and_missing():
    assert se.fund_surprise(1.10, 1.00) == pytest.approx(0.10)
    assert se.fund_surprise(0.03, 0.01) == pytest.approx(0.4)          # Nenner-Untergrenze 0.05
    assert se.fund_surprise(10, 1) == 2.0 and se.fund_surprise(-10, 1) == -2.0
    assert se.fund_surprise(float("nan"), 1) is None and se.fund_surprise(None, 1) is None


CAL = pd.bdate_range("2024-01-01", "2024-01-31")                  # Mo 1.1. ... (keine Feiertage)


def _ts(s):
    return pd.Timestamp(s, tz=NY)


def test_reaction_window_timing():
    i = lambda d: int(CAL.get_loc(pd.Timestamp(d)))
    # nach Börsenschluss (Di 16:05) -> Reaktion Mi, Start = Close Di
    assert se.reaction_window(_ts("2024-01-09 16:05"), CAL) == (i("2024-01-09"), i("2024-01-10"))
    # vor Börsenöffnung (Di 07:00) -> Reaktion Di, Start = Close Mo
    assert se.reaction_window(_ts("2024-01-09 07:00"), CAL) == (i("2024-01-08"), i("2024-01-09"))
    # Uhrzeit unbekannt -> konservativ Meldetag + Folgetag
    assert se.reaction_window(_ts("2024-01-09 00:00"), CAL) == (i("2024-01-08"), i("2024-01-10"))
    # Wochenende -> erster Handelstag danach
    assert se.reaction_window(_ts("2024-01-13 08:00"), CAL) == (i("2024-01-12"), i("2024-01-15"))
    assert se.reaction_window(_ts("2023-12-01 08:00"), CAL) is None


def _synthetic(effect: float, n_tickers=40, seed=1):
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range("2014-01-01", "2023-12-29")
    spy = pd.DataFrame({"Open": 100.0, "Close": 100.0}, index=cal)
    frames, earn = {}, []
    for k in range(n_tickers):
        t = f"T{k:02d}"
        ret = rng.normal(0, 0.01, len(cal))
        days = list(range(30 + k % 20, len(cal) - 80, 63))
        for d in days:
            s = rng.normal(0, 0.1)
            earn.append({"ticker": t, "ann_ts": pd.Timestamp(cal[d]).tz_localize(NY) + pd.Timedelta(hours=17),
                         "eps_estimate": 1.0, "eps_actual": 1.0 + s})
            ret[d + 2: d + 22] += effect * np.sign(s)              # Drift NACH dem Reaktionstag
        close = 50 * np.cumprod(1 + ret)
        frames[t] = pd.DataFrame({"Open": close, "Close": close}, index=cal)
    return pd.DataFrame(earn), frames, spy


def test_events_are_point_in_time():
    earn, frames, spy = _synthetic(0.0, n_tickers=2)
    ev = se.build_events(earn, frames, spy)
    assert len(ev) > 10
    assert (pd.to_datetime(ev["exit_20"]) > pd.to_datetime(ev["date"])).all()
    assert (pd.to_datetime(ev["lag_exit_20"]) > pd.to_datetime(ev["exit_20"])).all()
    row = ev.iloc[0]
    c = frames[row["ticker"]]["Close"]
    t = c.index.get_loc(row["date"])
    assert row["fwd_20"] == pytest.approx(c.iloc[t + 20] / frames[row["ticker"]]["Open"].iloc[t + 1] - 1.0)


def test_thresholds_only_from_training():
    train = pd.DataFrame({"fund_surprise": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], "react_z": [0.0] * 6})
    test = pd.DataFrame({"fund_surprise": [0.35, 5.0], "react_z": [0.0, 0.0], "fwd_20": [0.01, 0.02],
                         "date": pd.to_datetime(["2020-01-02"] * 2), "vix": 15, "trend_up": True, "ticker": "X"})
    tr = se.hypothesis_trades("S1L_pead_long", train, test, 20)
    assert len(tr) == 2                                   # Schwelle = Train-q(2/3) = 0.333, nicht aus Test
    test2 = test.assign(fund_surprise=[0.30, 0.31])
    assert se.hypothesis_trades("S1L_pead_long", train, test2, 20).empty


def test_benjamini_hochberg():
    r = se.benjamini_hochberg({"a": 0.001, "b": 0.02, "c": 0.5, "d": None}, q=0.10)
    assert r == {"a": True, "b": True, "c": False, "d": False}


def test_detects_real_effect_and_rejects_noise():
    earn, frames, spy = _synthetic(0.002, seed=3)
    res = se.evaluate(se.build_events(earn, frames, spy), placebo_n=19)
    s1 = res["hypotheses"]["S1_pead"]
    assert s1["h20"]["base_cost"]["mean"] > 0 and s1["h20"]["bh_significant"]
    assert s1["h20"]["placebo_p"] <= 0.05 and s1["h20"]["lag"]["mean"] is not None
    assert s1["decision"]["verdict"] in ("KEEP", "MODIFY")
    earn0, frames0, spy0 = _synthetic(0.0, seed=4)
    res0 = se.evaluate(se.build_events(earn0, frames0, spy0), placebo_n=9)
    assert res0["hypotheses"]["S1_pead"]["decision"]["verdict"] != "KEEP"
    assert all(h["decision"]["verdict"] != "KEEP" for h in res0["hypotheses"].values())


def test_decide_requires_all_robustness_checks():
    base = {"mean": 0.01, "t_months": 3.0, "years_positive_share": 0.8}
    rep = {k: {"mean": 0.01} for k in ("universe_A", "universe_B", "period_early", "period_late")}
    ok = se.decide(base, {"mean": 0.005}, {"vix_low": {"n": 5, "mean": 0.01}}, {"mean": 0.004}, rep, True, 0.01, True)
    assert ok["verdict"] == "KEEP"
    no_placebo = se.decide(base, {"mean": 0.005}, {}, {"mean": 0.004}, rep, True, 0.2, True)
    assert no_placebo["verdict"] == "MODIFY" and any("Placebo" in r for r in no_placebo["reasons"])
    bad = se.decide({"mean": -0.01, "t_months": -1}, {"mean": -0.02}, {}, {"mean": -0.01}, rep, False, None, True)
    assert bad["verdict"] == "REJECT"


def test_counter_hypothesis_only_on_holdout_years():
    earn, frames, spy = _synthetic(0.0, seed=5)
    ev = se.build_events(earn, frames, spy)
    c = se.holdout_reversal(ev)
    assert c["holdout_years"] and max(c["holdout_years"]) < se.TEST_START_YEAR
    assert c["decision"]["verdict"] in ("KEEP", "MODIFY", "REJECT")
    # Testjahre verändern das Holdout-Ergebnis nicht
    ev2 = ev.copy()
    ev2.loc[ev2["year"] >= se.TEST_START_YEAR, "fwd_20"] = 9.9
    assert se.holdout_reversal(ev2)["h20"]["base_cost"]["mean"] == c["h20"]["base_cost"]["mean"]


def test_counter_hypothesis_detects_real_reversal():
    rng = np.random.default_rng(7)
    cal = pd.bdate_range("2014-01-01", "2018-12-31")
    spy = pd.DataFrame({"Open": 100.0, "Close": 100.0}, index=cal)
    frames, earn = {}, []
    for k in range(40):
        ret = rng.normal(0, 0.01, len(cal))
        for d in range(30 + k % 20, len(cal) - 80, 63):
            jump = rng.choice([-0.05, 0.05])
            ret[d + 1] += jump                          # Reaktionstag (Meldung nach Schluss)
            ret[d + 2: d + 22] += -jump / 20 * 1.5      # Umkehr danach
            earn.append({"ticker": f"T{k}", "ann_ts": pd.Timestamp(cal[d]).tz_localize(NY) + pd.Timedelta(hours=17),
                         "eps_estimate": 1.0, "eps_actual": 1.1})
        c = 50 * np.cumprod(1 + ret)
        frames[f"T{k}"] = pd.DataFrame({"Open": c, "Close": c}, index=cal)
    res = se.holdout_reversal(se.build_events(pd.DataFrame(earn), frames, spy))
    assert res["h20"]["base_cost"]["mean"] > 0 and res["h20"]["base_cost"]["t_months"] > 2

"""Expectation Alpha – Pflicht-Leakage-Tests (PIT). Synthetische Daten (tests/ea_fixtures.py).

Geprüft wird: Publikationsverzug, Vintages, keine Zukunftswerte in Kontext/Gaps/Regime/Signalen, keine
Zukunftsdaten im Trigger-Replay, kein Outcome im Entscheidungsdatensatz, z/Perzentil/Wechselwahrscheinlichkeit
nur aus der Vergangenheit.
"""
from __future__ import annotations

import copy
import json
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

import tests.ea_fixtures as fx
from modules.expectation_alpha import build_context
from modules.expectation_alpha import config as eacfg
from modules.expectation_alpha import data as eadata
from modules.expectation_alpha import ledger as eal
from modules.expectation_alpha import timing as tm
from modules.expectation_alpha.future_state import expanding_percentile, expanding_z
from modules.expectation_alpha.regime_change import transition_probability

UTC = timezone.utc
DT = datetime(2026, 10, 9, 15, 0, tzinfo=UTC)          # Entscheidung vor Börsenschluss


@pytest.fixture(scope="module")
def base():
    return {"px": fx.make_px(end="2027-06-30", years=7), "ar": fx.make_archive(), "cm": fx.make_commodity(),
            "cfg": eacfg.load()}


def _ctx(px, ar, cm, cfg, t=DT):
    return build_context(t, cfg, loaders=fx.loaders(px, ar, cm))


def test_prices_after_decision_and_unfinished_bar_never_used(base):
    px = base["px"]
    cut = pd.Timestamp("2026-10-08")                          # letzter abgeschlossener Tag vor 15:00 UTC am 9.10.
    ctx_full = _ctx(px, base["ar"], base["cm"], base["cfg"])
    ctx_cut = _ctx(px[px.index <= cut], base["ar"], base["cm"], base["cfg"])
    poisoned = px.copy()
    poisoned.loc[poisoned.index > cut] *= 50.0               # Zukunft massiv verändert
    ctx_poison = _ctx(poisoned, base["ar"], base["cm"], base["cfg"])
    assert ctx_full["signals"]["date"] == "2026-10-08"
    assert ctx_full["context_hash"] == ctx_cut["context_hash"] == ctx_poison["context_hash"]


def test_after_close_decision_uses_same_day_bar(base):
    late = datetime(2026, 10, 9, 21, 30, tzinfo=UTC)
    assert eadata.last_complete_close(late) == date(2026, 10, 9)
    assert eadata.last_complete_close(DT) == date(2026, 10, 8)
    p = eadata.pit_prices(base["px"], late)
    assert p.index[-1] == pd.Timestamp("2026-10-09")


def test_archive_values_published_after_decision_are_invisible(base):
    ar = copy.deepcopy(base["ar"])
    ctx0 = _ctx(base["px"], ar, base["cm"], base["cfg"])
    # Zusätzliche (gefälschte) Vintages, die erst NACH dem Stichtag verfügbar werden
    for src, obs in ar.items():
        for o in list(obs)[-50:]:
            o2 = copy.deepcopy(o)
            o2.value = (o.value or 1.0) * 100.0
            o2.available_at = datetime(2026, 10, 9, 0, 0, tzinfo=UTC)   # = Stichtag 00:00 -> nicht < cut
            obs.append(o2)
    ctx1 = _ctx(base["px"], ar, base["cm"], base["cfg"])
    assert ctx0["context_hash"] == ctx1["context_hash"]


def test_publication_lag_and_vintage_selection():
    per = [datetime(2026, m, 1, tzinfo=UTC) for m in (5, 6, 7, 8)]
    obs = fx.make_obs("s", "m", per, [100.0, 101.0, 102.0, 103.0], unit="x", release_lag_days=45,
                      revisions={3: (110.0, 10)})                 # August: Erstwert 15.09., Revision 25.09.
    d_before = [pd.Timestamp("2026-09-15")]                      # verfügbar 15.09. 00:00 -> erst ab 16.09. (strikt <)
    d_first = [pd.Timestamp("2026-09-16")]                       # August-Erstwert sichtbar, Revision noch nicht
    d_rev = [pd.Timestamp("2026-09-26")]                         # Revision sichtbar
    assert eadata.pit_weekly(obs, "m", d_before, 120).iloc[0] == 102.0
    assert eadata.pit_weekly(obs, "m", d_first, 120).iloc[0] == 103.0
    assert eadata.pit_weekly(obs, "m", d_rev, 120).iloc[0] == 110.0
    o = eadata.latest_vintage(obs, "m", datetime(2026, 9, 20, tzinfo=UTC))
    assert o.value == 103.0 and o.available_at < datetime(2026, 9, 20, tzinfo=UTC)
    o = eadata.latest_vintage(obs, "m", datetime(2026, 9, 26, tzinfo=UTC))
    assert o.value == 110.0                                      # jüngste Vintage derselben Periode


def test_stale_series_is_missing_not_last_value():
    per = [datetime(2025, m, 1, tzinfo=UTC) for m in (1, 2, 3)]
    obs = fx.make_obs("s", "m", per, [1.0, 2.0, 3.0], unit="x", release_lag_days=30)
    v = eadata.pit_weekly(obs, "m", [pd.Timestamp("2026-10-01")], max_age_days=75)
    assert np.isnan(v.iloc[0])


def test_commodity_rows_after_decision_dropped(base):
    cm = fx.make_commodity(end="2027-06-30", years=5)
    inp = eadata.load_inputs(DT, base["cfg"], archive_fn=lambda c: {}, prices_fn=lambda c, t: base["px"],
                             commodity_fn=lambda c, t: cm)
    assert inp.commodity.index.max() <= pd.Timestamp("2026-10-09")


def test_expanding_z_and_percentile_use_only_past():
    rng = np.random.default_rng(3)
    s = pd.Series(rng.normal(size=300), index=pd.date_range("2020-01-03", periods=300, freq="W-FRI"))
    s2 = s.copy()
    s2.iloc[200:] = 1e6                                          # Zukunft verändert
    pd.testing.assert_series_equal(expanding_z(s, 52).iloc[:200], expanding_z(s2, 52).iloc[:200])
    pd.testing.assert_series_equal(expanding_percentile(s, 52).iloc[:200], expanding_percentile(s2, 52).iloc[:200])
    # der aktuelle Wert zählt nie für seine eigene Normierung
    s3 = pd.Series([0.0, 1.0] * 30 + [100.0])
    assert expanding_percentile(s3, 52).iloc[-1] == 1.0 and expanding_z(s3, 52).iloc[-1] == 4.0
    flat = pd.Series([1.0] * 60 + [2.0])                        # Std 0 -> z fehlend, nie Division durch 0
    assert np.isnan(expanding_z(flat, 52).iloc[-1])


def test_transition_probability_ignores_anchors_with_unknown_outcome():
    rng = np.random.default_rng(4)
    z = pd.Series(rng.normal(size=400))
    at = 300
    a = transition_probability(z, 0.5, 4, 104, at=at)
    z2 = z.copy()
    z2.iloc[at + 1:] = -z2.iloc[at + 1:]                         # Zukunft nach `at` umgedreht
    b = transition_probability(z2, 0.5, 4, 104, at=at)
    assert a == b and a["n_anchors"] == at - 4 + 1


def test_decision_rows_contain_no_outcomes_and_stay_unchanged_after_resolution(tmp_path, base):
    from modules import expectation_alpha as ea
    px = base["px"].copy()
    px["AAPL"] = px["XLK"] * 1.3
    ea.enrich_candidates([fx.make_analysis("AAPL")], decision_time=DT, cfg=base["cfg"],
                         loaders=fx.loaders(px, base["ar"], base["cm"]), root=tmp_path, contracts=[],
                         registry=tmp_path / "r.jsonl")
    f = next((tmp_path / "candidates").glob("*.jsonl"))
    before = f.read_bytes()
    row = json.loads(before.decode().splitlines()[0])
    for k in ("outcome", "outcome_net", "mfe", "mae", "exit_date"):
        assert k not in row
    st = eal.resolve_outcomes(today=date(2027, 6, 30), root=tmp_path, bars_fn=fx.bars_fn_from(px), cfg=base["cfg"])
    assert st["resolved"] > 0
    assert f.read_bytes() == before                              # Entscheidungsdatensatz unverändert
    assert (tmp_path / "outcomes.jsonl").exists()


def test_wait_trigger_replay_uses_only_closes_up_to_each_day(base):
    px = base["px"].copy()
    px["AAPL"] = px["XLK"] * 1.3
    row = {"direction_sign": 1, "sector_etf": "XLK", "wait_trigger": tm.wait_trigger(base["cfg"]),
           "cross_asset_confirmation": {"signals": [
               {"signal": "spy_ret_20d", "expected": 1, "deadband": 0.0, "weight": 1.0},
               {"signal": "hyg_lqd_20d", "expected": 1, "deadband": 0.0, "weight": 1.0},
               {"signal": "sector_rs_20d:XLK", "expected": 1, "deadband": 0.0, "weight": 1.0}]}}
    from modules.expectation_alpha import cross_asset_confirmation as cac
    bars = fx.make_bars(px["AAPL"])
    i0 = next(i for i, b in enumerate(bars) if b[0] >= date(2026, 10, 9))
    j = eal.trigger_day(row, cac.signal_frame(px), bars, i0)
    poisoned = px.copy()
    if j is not None:
        poisoned.loc[poisoned.index > pd.Timestamp(bars[j][0])] *= 3.0
    else:
        poisoned.loc[poisoned.index > pd.Timestamp(bars[i0 + 10][0])] *= 3.0
    assert eal.trigger_day(row, cac.signal_frame(poisoned), bars, i0) == j


def test_economic_invalidation_only_from_later_contexts():
    bars = [(date(2026, 10, d), 101.0, 99.0, 100.0) for d in range(1, 31)]
    kill = tm.kill_conditions(ttm=None, primary_domain="growth", gap_sign=1, cfg=eacfg.load())
    row = {"direction_sign": 1, "kill_conditions": kill, "cross_asset_confirmation": {"signals": []}}
    ctxs = [{"date": "2026-10-01", "gaps": {"growth": {"status": "OK", "sign": -1}}},   # Entscheidungstag: zählt nicht
            {"date": "2026-10-12", "gaps": {"growth": {"status": "OK", "sign": -1}}}]
    ev = eal.kill_events(row, bars, 0, 20, None, ctxs)
    assert ev.get("economic_invalidation") == 11


def test_entry_is_first_close_after_decision():
    from modules.expectation_alpha.thesis import entry_not_before
    assert entry_not_before("2026-10-09T15:00:00+00:00") == "2026-10-09"   # 11:00 NY, vor Schluss
    assert entry_not_before("2026-10-09T20:30:00+00:00") == "2026-10-10"   # 16:30 NY (Sommerzeit) -> Folgetag
    assert entry_not_before("2026-12-09T20:30:00+00:00") == "2026-12-09"   # 15:30 NY (Winterzeit) -> noch heute
    assert entry_not_before("2026-12-09T21:00:00+00:00") == "2026-12-10"

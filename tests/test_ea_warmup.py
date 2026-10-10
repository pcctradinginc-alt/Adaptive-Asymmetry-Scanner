"""Expectation Alpha – Warm-up-Guards: zu wenig Historie -> INSUFFICIENT_HISTORY (nie 0, neutral,
künstlicher Extrem-z oder langer Forward-Fill). Startup, kurze Historie, Grenzen, volle Historie."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

import tests.ea_fixtures as fx
from modules import expectation_alpha as ea
from modules.expectation_alpha import config as eacfg
from modules.expectation_alpha import cross_asset_confirmation as cac
from modules.expectation_alpha import data as eadata
from modules.expectation_alpha.future_state import expanding_z, min_points, roc_set
from modules.expectation_alpha.schemas import INSUFFICIENT_HISTORY, OK

CFG = eacfg.load()
DT = datetime(2026, 10, 9, 15, 0, tzinfo=timezone.utc)
D1, D3 = CFG["future_state"]["delta_1m_weeks"], CFG["future_state"]["delta_3m_weeks"]
MH = CFG["data"]["min_history_weeks"]


def _s(n, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(np.cumsum(rng.normal(size=n)), index=pd.date_range("2020-01-03", periods=n, freq="W-FRI"))


def test_min_points_derived_from_config():
    need = min_points(D1, D3, MH)
    assert need == {"delta_1m": D1 + 1, "delta_3m": D3 + 1, "velocity": D3 + 1, "acceleration": D3 + 1,
                    "change_of_change": D3 + D1 + 1, "z": MH + 1, "percentile": MH + 1}


@pytest.mark.parametrize("field", ["delta_1m", "delta_3m", "velocity", "acceleration", "change_of_change", "z",
                                   "percentile"])
def test_boundary_each_field(field):
    need = min_points(D1, D3, MH)[field]
    below = roc_set(_s(need - 1), CFG, unit="z", min_hist=MH)
    at = roc_set(_s(need), CFG, unit="z", min_hist=MH)
    assert below[field] is None and below["history_status"][field] == INSUFFICIENT_HISTORY
    assert at[field] is not None and at["history_status"][field] == OK
    assert field in below["insufficient_history_fields"]


def test_startup_short_series_no_zero_no_neutral_no_extreme():
    r = roc_set(_s(3), CFG, unit="z", min_hist=MH)
    assert r["level"] is not None and r["status"] == INSUFFICIENT_HISTORY
    for f in ("delta_3m", "velocity", "acceleration", "change_of_change", "z", "percentile", "uncertainty"):
        assert r[f] is None                                       # nie 0
    assert r["regime_state"] is None                              # nie "neutral"
    assert r["regime_transition_probability"] is None and r["history_status"]["regime_transition_probability"] == INSUFFICIENT_HISTORY


def test_zscore_no_artificial_extreme_with_short_history():
    s = pd.Series([0.0, 0.1] * 10 + [1e6])                         # 20 Punkte Historie, dann Ausreißer
    assert np.isnan(expanding_z(s, MH).iloc[-1])                   # kein künstliches ±4
    long = pd.Series([0.0, 0.1] * 40 + [1e6])
    assert expanding_z(long, MH).iloc[-1] == 4.0                   # erst mit Historie: geclippt, nicht unendlich


def test_full_history_all_ok():
    r = roc_set(_s(400), CFG, unit="z", min_hist=MH)
    assert r["status"] == OK and r["insufficient_history_fields"] == []
    assert all(r[f] is not None for f in ("delta_1m", "delta_3m", "velocity", "acceleration", "change_of_change",
                                          "z", "percentile", "regime_transition_probability"))


def test_gap_inside_history_is_unavailable_not_insufficient():
    s = _s(200)
    s.iloc[-1 - D3] = np.nan                                       # Lag-Punkt fehlt, Historie lang genug
    r = roc_set(s, CFG, unit="z", min_hist=MH)
    assert r["delta_3m"] is None and r["history_status"]["delta_3m"] == "UNAVAILABLE"


def test_stale_series_not_forward_filled():
    per = [datetime(2025, m, 1, tzinfo=timezone.utc) for m in (1, 2, 3)]
    obs = fx.make_obs("s", "m", per, [1.0, 2.0, 3.0], unit="x", release_lag_days=30)
    v = eadata.pit_weekly(obs, "m", [pd.Timestamp("2026-10-01")], max_age_days=75)
    assert np.isnan(v.iloc[0])                                     # kein langer stiller Forward-Fill


def test_cross_asset_signals_warmup_reason():
    px = fx.make_px(end="2026-10-08", years=1)
    short = px.iloc[-40:]                                          # 40 Handelstage: 20T ok, 63T nicht
    sig = cac.market_signals(short, None, 20, CFG["warmup"]["signal_min_rows"])
    assert sig["values"]["spy_ret_20d"] is not None
    assert sig["values"]["sector_rs_63d:XLK"] is None
    assert sig["missing_reason"]["sector_rs_63d:XLK"] == INSUFFICIENT_HISTORY
    tiny = cac.market_signals(px.iloc[-10:], None, 20, CFG["warmup"]["signal_min_rows"])
    assert tiny["missing_reason"]["spy_ret_20d"] == INSUFFICIENT_HISTORY


def test_startup_run_end_to_end_is_explicit(tmp_path):
    """Kaltstart mit wenigen Wochen Daten: kein Absturz, keine Nullen, Warm-up gezählt, nie TRADE aus Artefakten."""
    px = fx.make_px(end="2026-10-08", years=1).iloc[-60:]
    ar = fx.make_archive(years=1)
    s = ea.enrich_candidates([fx.make_analysis("AAPL")], decision_time=DT, cfg=CFG,
                             loaders=fx.loaders(px, ar, fx.make_commodity(years=1)), root=tmp_path, contracts=[],
                             registry=tmp_path / "r.jsonl")
    assert s["insufficient_history_count"]["total"] > 0
    from modules.expectation_alpha import ledger as eal
    ctx = eal.read_contexts(tmp_path)[0]
    row = eal.read_rows(tmp_path)[0]
    # Kontext fehlt (nicht negativ) -> Gruppe X; Warm-up-Signale (63T) zählen weder bestätigend noch widersprechend
    assert row["context_status"] is None and row["group"] == "X" and row["macro_alignment"] is None
    warm = {k for k, v in ctx["signals"]["missing_reason"].items() if v == INSUFFICIENT_HISTORY}
    used = {x["signal"] for x in row["cross_asset_confirmation"]["signals"] if x["state"] != "MISSING"}
    assert warm and not (warm & used)
    assert ctx["regime"].get("regime_uncertainty") is None
    for g in ctx["gaps"].values():
        assert g["status"] in ("INSUFFICIENT_HISTORY", "UNAVAILABLE")
        assert g.get("gap_z") is None                              # nie 0 als Ersatz

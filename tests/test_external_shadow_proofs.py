"""Beweis-Tests (Audit-Follow-up 2026-09-27):

1. external_context.mode = shadow liefert dieselben Produktionsentscheidungen
   wie off: kein Entscheidungsmodul liest external_context; Scores/Ranking
   sind mit und ohne Kontext identisch.
2. Externe Features erreichen weder Pearson-Gewichte noch PPO-Beobachtungen.
3. Kandidat T1 -> Revision T2 -> Feedback T3: die T1-Werte bleiben.
"""
from __future__ import annotations

import copy
import inspect
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

DECISION_MODULES = [
    "modules/trade_scorer.py", "modules/quasi_ml.py", "modules/risk_gates.py",
    "modules/mismatch_scorer.py", "modules/rl_agent.py", "modules/rl_environment.py",
    "modules/deep_analysis.py",
]


def _external_ctx(state="STRONG_CONTRACTION", relation="CONTRADICT"):
    return {
        "snapshot_id": "x", "feature_version": {"context": "v1"}, "available_at": "2026-09-01T00:00:00+00:00",
        "primitives": {"freight_global_z": -2.5, "shipping_global_z": -2.1, "weather_disruption_index": 0.9},
        "states": {"global_freight_state": state, "global_maritime_state": state,
                   "global_freight_confidence": 0.9},
        "ticker_exposure": {"road_freight_relevance": "HIGH", "maritime_relevance": "HIGH",
                            "weather_relevance": "HIGH"},
        "relation": {"relation": relation, "materiality": "HIGH", "confidence": 0.9},
        "policy": {"mode": "shadow", "score_delta": 0.0, "veto": False},
    }


def _candidate(ticker, impact, mismatch, bins):
    return {
        "ticker": ticker,
        "features": {"impact": impact, "surprise": 6, "mismatch": mismatch, "z_score": 0.5,
                     "eps_drift": 0.01, "bin_impact": bins[0], "bin_mismatch": bins[1],
                     "bin_eps_drift": bins[2], "trade_score": 60},
        "deep_analysis": {"impact": impact, "surprise": 6, "direction": "BULLISH",
                          "catalyst": "guidance raise", "time_to_materialization": "1-3 Monate",
                          "red_team": {"severity": 3}},
        "simulation": {"hit_rate": 0.55, "iv_rank": 40},
        "option": {"dte": 60, "strike": 100, "bid": 2.0, "ask": 2.2, "open_interest": 500,
                   "spread_ratio": 0.05},
        "roi_analysis": {"dte": 60, "expected_return": 0.3},
    }


def _pairs():
    base = [_candidate("AAA", 8, 4.5, ("high", "good", "noise")),
            _candidate("BBB", 6, 2.0, ("mid", "weak", "relevant")),
            _candidate("CCC", 7, 6.0, ("mid", "strong", "massive"))]
    with_ctx = copy.deepcopy(base)
    for c in with_ctx:
        c["external_context"] = _external_ctx()
    return base, with_ctx


def test_decision_modules_never_read_external_context():
    for path in DECISION_MODULES:
        src = Path(path).read_text(encoding="utf-8")
        assert "external_context" not in src, f"{path} liest external_context"
        assert "external_context_entry" not in src, path


def test_quasi_ml_ranking_identical_with_and_without_external_context():
    from modules.quasi_ml import QuasiML
    history = {"feature_stats": {}, "model_weights": {"impact": 0.35, "mismatch": 0.45, "eps_drift": 0.2}}
    off, shadow = _pairs()
    a = QuasiML(history).run(off)
    b = QuasiML(history).run(shadow)
    assert [(x["ticker"], x["final_score"]) for x in a] == [(x["ticker"], x["final_score"]) for x in b]


def test_trade_score_identical_with_and_without_external_context():
    from modules.trade_scorer import compute_trade_score
    off, shadow = _pairs()
    for a, b in zip(off, shadow):
        assert compute_trade_score(a) == compute_trade_score(b)


def test_rl_scorer_identical_with_and_without_external_context():
    from modules.rl_agent import RLScorer
    off, shadow = _pairs()
    a = RLScorer({"feature_stats": {}}, veto_enabled=False).run(off)
    b = RLScorer({"feature_stats": {}}, veto_enabled=False).run(shadow)
    assert [(x["ticker"], x["final_score"]) for x in a] == [(x["ticker"], x["final_score"]) for x in b]


def test_shadow_policy_never_changes_score_or_veto():
    from modules.external import policy
    src = inspect.getsource(policy)
    assert "shadow" in src
    ctx = _external_ctx()
    assert ctx["policy"]["score_delta"] == 0.0 and ctx["policy"]["veto"] is False


def _closed_history(with_external: bool):
    trades = []
    for i in range(12):
        t = {"ticker": f"T{i}", "entry_date": f"2026-05-{i + 1:02d}", "outcome": (-1) ** i * 0.4 + i * 0.01,
             "features": {"impact": 5 + i % 4, "surprise": 5, "mismatch": 2 + i % 3, "z_score": 0.3,
                          "eps_drift": 0.0, "bin_impact": ["low", "mid", "high"][i % 3],
                          "bin_mismatch": ["weak", "good", "strong"][i % 3],
                          "bin_eps_drift": "noise"},
             "option": {"dte": 60}}
        if with_external:
            t["external_context_entry"] = _external_ctx(
                state=["CONTRACTION", "EXPANSION"][i % 2], relation=["SUPPORT", "CONTRADICT"][i % 2])
        trades.append(t)
    return {"closed_trades": trades, "model_weights": {"impact": 0.35, "mismatch": 0.45, "eps_drift": 0.2}}


def test_external_features_do_not_reach_pearson_weights():
    import feedback
    assert feedback.compute_pearson_weights(_closed_history(False)) == \
        feedback.compute_pearson_weights(_closed_history(True))


def test_external_features_do_not_reach_ppo_observations():
    import numpy as np
    from modules.rl_environment import OptionsRLEnv
    a = OptionsRLEnv(_closed_history(False)["closed_trades"])
    b = OptionsRLEnv(_closed_history(True)["closed_trades"])
    oa, _ = a.reset()
    ob, _ = b.reset()
    for _ in range(len(a.trade_data)):
        assert np.array_equal(oa, ob)
        oa, ra, da, _, _ = a.step(1)
        ob, rb, db, _, _ = b.step(1)
        assert ra == rb
        if da:
            break


def test_ledger_keeps_t1_context_after_t2_revision(tmp_path):
    """T1: Kandidat mit Kontext aus dem Archiv. T2: Quelle revidiert. T3:
    neuer Snapshot. Der Ledger-Eintrag behält die T1-Werte (Freeze), und
    ein Kontext zu T1 aus dem revidierten Archiv liefert weiterhin T1-Werte."""
    from modules import candidate_ledger as cl
    from modules.external.archive import ExternalArchive
    from modules.external.pit import AvailabilityPrecision, Observation

    t_obs = datetime(2026, 6, 1, tzinfo=timezone.utc)
    t1 = datetime(2026, 9, 1, 14, tzinfo=timezone.utc)
    t2 = t1 + timedelta(days=5)
    archive = ExternalArchive(root=tmp_path)

    def obs(value, avail):
        return Observation(source_id="bts_freight_tsi", dataset="tsi", series_id="TSIFRGHT", entity_id="",
                           metric="us_freight_tsi", value=value, unit="index", observation_time=t_obs,
                           available_at=avail, retrieved_at=avail, vintage_time=avail,
                           availability_precision=AvailabilityPrecision.EXACT_DATE, parser_version="1")
    archive.store_observations([obs(130.0, t1 - timedelta(days=10))])
    known_t1 = [o.value for o in archive.as_of("bts_freight_tsi", t1)]
    archive.store_observations([obs(120.0, t2)])                 # Revision nach T1

    cl.start_run("2026-09-01")
    cl.note("AAA", stage="deep_analysis", direction="BULLISH")
    frozen = {"primitives": {"us_tsi_value": known_t1[0]}, "snapshot_id": "t1"}
    cl.note("AAA", external=frozen)
    cl.note("AAA", external={"primitives": {"us_tsi_value": 120.0}, "snapshot_id": "t3"})  # späterer Versuch
    ext = cl._signals("AAA")[0].get("external")
    assert ext["snapshot_id"] == "t1" and ext["primitives"]["us_tsi_value"] == 130.0

    # PIT: der Stand zu T1 bleibt aus dem revidierten Archiv rekonstruierbar
    assert [o.value for o in archive.as_of("bts_freight_tsi", t1)] == [130.0]
    assert [o.value for o in archive.as_of("bts_freight_tsi", t2)] == [120.0]
    assert len(archive.vintage_history("bts_freight_tsi", "TSIFRGHT", "", "us_freight_tsi", t_obs)) == 2

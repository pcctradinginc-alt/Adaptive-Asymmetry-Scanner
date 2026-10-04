"""Tests für modules/trade_memory.py (synthetisch, offline)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from modules import trade_memory as tm

REAL_HISTORY = Path(__file__).resolve().parent.parent / "outputs" / "history.json"


def _trade(**kw):
    t = {
        "ticker": "AAA", "entry_date": "2026-04-01", "close_date": "2026-05-01",
        "strategy": "LONG_CALL",
        "features": {"impact": 5, "surprise": 4, "mismatch": 3.0, "z_score": 0.3,
                     "sigma_30d": 0.02, "eps_drift": 0.0},
        "simulation": {"current_price": 100.0, "hit_rate": 0.7, "sigma": 0.02},
        "deep_analysis": {"direction": "BULLISH", "data_confidence": "high"},
        "close_price": 100.0, "outcome": -0.5, "outcome_method": "option_quote",
    }
    t.update(kw)
    return t


def _case(**kw):
    """Fall über build_cases, damit der Weg Rohdaten -> Klassifikation mitgetestet wird."""
    return tm.build_cases({"closed_trades": [_trade(**kw)]}, kw.pop("_reg", None))[0]


def test_case_fields_and_id_stable():
    c = _case(close_price=110.0, outcome=0.4, catalyst_type="EARNINGS", peak_return=0.6)
    assert c["case_id"] == tm.make_case_id("AAA", "2026-04-01", "LONG_CALL")
    assert c["holding_days"] == 30 and c["direction"] == 1
    assert c["underlying_return_signed"] == pytest.approx(0.10)
    assert c["max_upside"] == 0.6 and c["outcome_reliable"] is True
    assert c["failure"] is None and c["what_went_wrong"] == []
    assert c["reason_for_trade"]["catalyst_type"] == "EARNINGS"


def test_direction_bearish_signed_return():
    c = _case(strategy="LONG_PUT", deep_analysis={"direction": "BEARISH"}, close_price=90.0, outcome=0.3)
    assert c["direction"] == -1 and c["underlying_return_signed"] == pytest.approx(0.10)


def test_direction_fallback_strategy():
    c = _case(strategy="BEAR_PUT_SPREAD", deep_analysis={}, close_price=90.0, outcome=0.1)
    assert c["direction"] == -1


def test_unreliable_excluded():
    c = _case(outcome_method_reconstructed="delta_approx", close_price=80.0)
    assert c["outcome_reliable"] is False
    assert c["failure"]["primary"] == "unreliable_outcome"
    assert c["failure"]["labels"] == ["unreliable_outcome"]


def test_signal_wrong():
    # sigma 2 %/Tag, 30 Kalendertage -> ca. 21 Handelstage -> Schwelle ca. -4.6 %
    c = _case(close_price=90.0)
    assert c["failure"]["primary"] == "signal_wrong"
    assert c["failure"]["details"]["signal_threshold"] == pytest.approx(-0.5 * 0.02 * (30 * 5 / 7) ** 0.5)


def test_weak_adverse_move_not_signal_wrong():
    c = _case(close_price=99.0)
    assert c["failure"]["primary"] == "weak_adverse_move"


def test_signal_wrong_fallback_without_sigma():
    t = _trade(close_price=97.0)
    t["simulation"] = {"current_price": 100.0}
    t["features"] = dict(t["features"], sigma_30d=None)
    c = tm.build_cases({"closed_trades": [t]})[0]
    assert "signal_wrong" in c["failure"]["labels"]  # -3 % < -2 %


def test_structure_decay():
    c = _case(close_price=103.0)
    assert c["failure"]["primary"] == "structure_decay"


def test_exit_gave_back_has_priority():
    c = _case(close_price=90.0, peak_return=0.35)
    assert set(c["failure"]["labels"]) >= {"exit_gave_back", "signal_wrong"}
    assert c["failure"]["primary"] == "exit_gave_back"


def test_timing_too_early():
    da = {"direction": "BULLISH", "data_confidence": "high", "time_to_materialization": "2-3 Monate"}
    c = _case(close_price=99.0, deep_analysis=da)             # schwache Gegenbewegung
    assert "timing_too_early" in c["failure"]["labels"]
    assert c["failure"]["primary"] == "timing_too_early"
    strong = _case(close_price=90.0, deep_analysis=da)        # deutliche Gegenbewegung
    assert strong["failure"]["primary"] == "signal_wrong"
    assert "timing_too_early" not in strong["failure"]["labels"]


def test_regime_changed_only_core_keys():
    assert not tm._regime_changed({"oil": "oil_up", "credit": "credit_loose"}, {"oil": "oil_down", "credit": "credit_loose"})
    assert tm._regime_changed({"fin_conditions": "loose"}, {"fin_conditions": "tight"})


def test_timing_not_assigned_without_field_or_if_long_enough():
    c = _case(close_price=90.0)
    assert "timing_too_early" not in c["failure"]["labels"]
    da = {"direction": "BULLISH", "data_confidence": "high", "time_to_materialization": "4-8 Wochen"}
    c = _case(close_price=90.0, deep_analysis=da)  # 28 Tage <= 30 Tage Haltedauer
    assert "timing_too_early" not in c["failure"]["labels"]


def test_ttm_parsing():
    assert tm._ttm_days("4-8 Wochen") == 28
    assert tm._ttm_days("2-3 Monate") == 60
    assert tm._ttm_days("6 Monate") == 180
    assert tm._ttm_days(None) is None and tm._ttm_days("bald") is None


def test_regime_changed_and_lag():
    reg = {"2026-04-01": {"vix": "low_vol", "credit": "loose"},
           "2026-05-01": {"vix": "high_vol", "credit": "loose"}}
    t = _trade(close_price=103.0)
    c = tm.build_cases({"closed_trades": [t]}, reg)[0]
    assert "regime_changed" in c["failure"]["labels"]
    # Entry-Regime 2 Tage alt wird noch verwendet, 10 Tage alt nicht
    assert tm._regime_on({"2026-03-30": {"a": 1}}, "2026-04-01") == {"a": 1}
    assert tm._regime_on({"2026-03-20": {"a": 1}}, "2026-04-01") is None
    # gleiche Regime -> kein Label
    same = {"2026-04-01": {"vix": "low_vol"}, "2026-05-01": {"vix": "low_vol"}}
    c = tm.build_cases({"closed_trades": [t]}, same)[0]
    assert "regime_changed" not in c["failure"]["labels"]


def test_entry_cost_high():
    c = _case(close_price=103.0, entry_quote={"spread_pct": 0.12})
    assert "entry_cost_high" in c["failure"]["labels"]
    c = _case(close_price=103.0, entry_quote={"spread_pct": 0.05})
    assert "entry_cost_high" not in c["failure"]["labels"]


def test_low_data_confidence():
    da = {"direction": "BULLISH", "data_confidence": "low"}
    c = _case(close_price=103.0, deep_analysis=da)
    assert "low_data_confidence" in c["failure"]["labels"]
    t = _trade(close_price=103.0)
    t["features"] = {"impact": 5}   # Kernfeatures fehlen
    c = tm.build_cases({"closed_trades": [t]})[0]
    assert "low_data_confidence" in c["failure"]["labels"]


def test_underlying_unknown_and_other():
    t = _trade()
    del t["close_price"]
    c = tm.build_cases({"shadow_trades": [t]})[0]
    assert c["failure"]["primary"] == "underlying_unknown"
    # other: nur direkt über classify_failure erreichbar (nichts trifft zu)
    r = tm.classify_failure({"outcome": -0.2, "outcome_reliable": True, "underlying_return_signed": None,
                             "features": {k: 1.0 for k in tm.CORE_FEATURES}})
    assert r["primary"] == "underlying_unknown"
    assert tm.classify_failure({"outcome": 0.2})["primary"] is None


def test_priority_order_is_total():
    assert len(set(tm.PRIMARY_PRIORITY)) == len(tm.PRIMARY_PRIORITY)
    assert tm.PRIMARY_PRIORITY[0] == "unreliable_outcome" and tm.PRIMARY_PRIORITY[-1] == "other"


def test_open_and_missing_outcome_skipped_and_dedup():
    hist = {"closed_trades": [_trade(), _trade()],
            "shadow_trades": [_trade(close_date=None), _trade(ticker="B", outcome=None)]}
    assert len(tm.build_cases(hist)) == 1


# ── Aggregation ──────────────────────────────────────────────────────────────

def _mk(n, outcome, close_price, reliable=True, mismatch=3.0, day=1):
    return _trade(ticker=f"T{n}", outcome=outcome, close_price=close_price, close_date=f"2026-05-{day:02d}",
                  features={"impact": 5, "surprise": 4, "mismatch": mismatch, "z_score": 0.3,
                            "sigma_30d": 0.02, "eps_drift": 0.0},
                  **({} if reliable else {"outcome_method_reconstructed": "delta_approx"}))


def test_aggregation_split_and_shares():
    trades = ([_mk(i, -0.5, 90.0, day=1 + i) for i in range(3)]        # signal_wrong
              + [_mk(10, -0.4, 103.0, day=5)]                          # structure_decay
              + [_mk(20, -0.9, 90.0, reliable=False, day=6)]           # unreliable
              + [_mk(30, 0.5, 110.0, mismatch=5.0, day=7)])            # Gewinner
    cases = tm.build_cases({"closed_trades": trades})
    agg = tm.aggregate_failures(cases)
    rel, unr = agg["reliable"], agg["unreliable"]
    assert rel["n"] == 4 and unr["n"] == 1
    assert rel["primary"]["signal_wrong"]["share"] == pytest.approx(0.75)
    assert rel["primary"]["signal_wrong"]["mean_outcome"] == pytest.approx(-0.5)
    assert rel["primary"]["structure_decay"]["n"] == 1
    assert "unreliable_outcome" not in rel["primary"]
    assert unr["primary"]["unreliable_outcome"]["n"] == 1
    m = agg["winner_vs_loser"]["mismatch"]
    assert m["mean_winners"] == 5.0 and m["mean_losers"] == 3.0 and m["diff"] == 2.0


def test_aggregation_last_n_uses_most_recent():
    trades = [_mk(i, -0.5, 90.0, day=1 + i) for i in range(6)]
    agg = tm.aggregate_failures(tm.build_cases({"closed_trades": trades}), last_n=2)
    assert agg["reliable"]["n"] == 2 and agg["n_losses_total"] == 6


# ── similar_cases ────────────────────────────────────────────────────────────

def _mem_case(n, entry, close, mism, outcome=0.2, direction=1, cause=None):
    return {"case_id": f"c{n}", "entry_date": entry, "close_date": close, "direction": direction,
            "features": {"impact": 5.0 + n * 0.1, "surprise": 4.0 + n * 0.2, "mismatch": mism, "z_score": 0.3 + n * 0.01},
            "outcome": outcome, "outcome_reliable": True,
            "failure": {"primary": cause, "labels": [cause]} if cause else None}


def test_similar_cases_no_lookahead():
    cases = [_mem_case(1, "2026-04-01", "2026-04-20", 3.0),
             _mem_case(2, "2026-05-10", "2026-05-20", 3.0),          # Entry nach Query
             _mem_case(3, "2026-04-25", "2026-06-01", 3.0),          # Entry davor, Outcome erst später
             _mem_case(4, "2026-03-01", "2026-03-20", 9.0)]
    q = {"impact": 5, "surprise": 4, "mismatch": 3.0, "z_score": 0.3}
    res = tm.similar_cases(q, cases, k=10, query_date="2026-05-01")
    ids = {c["case_id"] for c in res["cases"]}
    assert ids == {"c1", "c4"}
    assert res["cases"][0]["case_id"] == "c1"
    assert res["summary"]["n"] == 2
    # ohne query_date sind alle da
    assert tm.similar_cases(q, cases)["summary"]["n"] == 4


def test_similar_cases_direction_pref_and_summary():
    cases = [_mem_case(1, "2026-04-01", "2026-04-10", 3.0, outcome=-0.5, direction=-1, cause="signal_wrong"),
             _mem_case(2, "2026-04-01", "2026-04-10", 3.2, outcome=-0.2, direction=1, cause="structure_decay"),
             _mem_case(3, "2026-04-01", "2026-04-10", 4.0, outcome=0.6, direction=1),
             _mem_case(4, "2026-04-01", "2026-04-10", 6.0, outcome=0.4, direction=1)]
    q = {"impact": 5, "surprise": 4, "mismatch": 3.0, "z_score": 0.3}
    res = tm.similar_cases(q, cases, k=3, query_direction=1)
    assert res["cases"][0]["direction"] == 1     # Gegenrichtung wird zurückgesetzt
    s = res["summary"]
    assert s["n"] == 3 and s["win_share"] == pytest.approx(1 / 3)
    assert s["median_outcome"] == pytest.approx(sorted(c["outcome"] for c in res["cases"])[1])
    assert s["top_failure_cause"] in ("signal_wrong", "structure_decay")


def test_similar_cases_needs_shared_features():
    cases = [_mem_case(1, "2026-04-01", "2026-04-10", 3.0)]
    res = tm.similar_cases({"mismatch": 3.0}, cases)
    assert res["cases"] == [] and res["summary"]["n"] == 0
    assert res["summary"]["median_outcome"] is None


# ── run() ────────────────────────────────────────────────────────────────────

def test_run_writes_files(tmp_path, monkeypatch):
    monkeypatch.setattr(tm, "_load_regimes_for", lambda dates: {})
    trades = [_mk(i, -0.5, 90.0, day=1 + i) for i in range(3)] + [_mk(9, 0.5, 110.0, day=9)]
    hp = tmp_path / "history.json"
    hp.write_text(json.dumps({"closed_trades": trades, "shadow_trades": []}), encoding="utf-8")
    out = tmp_path / "res"
    res = tm.run(hp, out)
    assert res["n_cases"] == 4
    lines = (out / "trade_memory.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4 and json.loads(lines[0])["case_id"]
    assert json.loads((out / "failure_analysis.json").read_text(encoding="utf-8"))["n_losses_total"] == 3
    md = (out / "failure_analysis.md").read_text(encoding="utf-8")
    assert "signal_wrong" in md and "Stichprobe" in md and "delta_approx" in md


def test_run_missing_history_raises(tmp_path):
    with pytest.raises(OSError):
        tm.run(tmp_path / "nope.json", tmp_path / "o")


@pytest.mark.skipif(not REAL_HISTORY.exists(), reason="keine echte history.json")
def test_run_on_real_history(tmp_path):
    res = tm.run(REAL_HISTORY, tmp_path / "out")
    assert res["n_cases"] > 0
    assert (tmp_path / "out" / "trade_memory.jsonl").exists()

"""Research Director + Active Learning: Kandidaten nur aus gemessenen Befunden,
PIT-Ausdrücke gültig, Begrenzung je Lauf, keine Wiederholung getesteter
Signale, Datenquellen nie automatisch vertrauenswürdig. Kein Netz."""
from __future__ import annotations

from modules import research_director as rd
from modules import research_lab as lab

META = {"reference": "static_equal", "approaches": {"static_equal": {"by_regime": {
    "vix_ge_20": {"expectancy": 0.012}, "vix_lt_20": {"expectancy": -0.002}}}},
    "model_intelligence": {"momentum_12_1": {"trend": "deteriorating", "trend_t": -2.6, "prior_ic": 0.07, "recent_ic": -0.1}}}
NEXTV = {"blind_spot_clusters": [
    {"id": "UNKNOWN_CLUSTER_001", "lift": 1.9, "typical_error": -0.12,
     "common_properties": {"volatility": "high_vol", "trend": "downtrend"}},
    {"id": "UNKNOWN_CLUSTER_002", "lift": 1.7, "typical_error": -0.1, "common_properties": {"sector": "Energy"}}]}
HYP_DB = {"hypotheses": {"HYP-0101": {"title": "Kurzfrist-Umkehr (1 Monat)", "signal": "rev_1m", "direction": -1,
                                      "canonical_status": "INCONCLUSIVE", "walk_forward": {"base": {"t_months": 2.11}}},
                         "HYP-0102": {"title": "Niedrig-Vola", "signal": "vol_60", "canonical_status": "REJECTED"}}}
WORLD = {"current": {"earnings_momentum_state": "unavailable", "labour_market_state": "neutral",
                     "labour_market_uncertainty": 0.04, "freight_supply_chain_state": "low",
                     "freight_supply_chain_uncertainty": 0.2}}
CATALOG = [{"id": "earnings_calendar_history", "dimension": ["event_risk", "earnings_momentum"], "provider": "SEC",
            "license": "gemeinfrei", "pit": "filing_timestamp", "history_from": 2004, "acquisition_cost": 3, "status": "candidate"},
           {"id": "portwatch_history_pit", "dimension": ["freight_supply_chain"], "provider": "IMF", "license": "frei",
            "pit": "forward_archive_only", "history_from": 2026, "acquisition_cost": 1, "status": "accumulating"}]


def test_plan_generates_valid_bounded_testable_hypotheses():
    rep = rd.plan(META, NEXTV, WORLD, HYP_DB, CATALOG, max_new=3)
    assert 0 < rep["n_selected_for_testing"] <= 3
    for h in rep["hypotheses"]:
        lab.validate_expr(h["signal"])                                  # nur PIT-Merkmale
        assert h["id"].startswith("RD-") and h["direction"] in (1, -1)
    kinds = {c["kind"] for c in rep["candidates"]}
    assert {"testable", "untestable", "diagnostic"} <= kinds             # Sektor-Cluster: nicht in Signal-Sprache
    for c in rep["candidates"]:
        for k in ("research_id", "question", "hypothesis", "economic_rationale", "required_data", "available_data",
                  "expected_information_gain", "novelty", "risk_of_overfitting", "research_cost", "priority"):
            assert k in c, k
    assert any("step(vix - 20)" in h["signal"] and "rev_1m" in h["signal"] for h in rep["hypotheses"])


def test_already_tested_signal_has_zero_novelty():
    tested = {"X": {"title": "a", "signal": "-(step(vol_60 - 0.17) * step(-spy_trend_200))"}}
    c = {"title": "b", "signal": "-(step(vol_60 - 0.17) * step(-spy_trend_200))"}
    assert rd._novelty(c, tested) == 0.0


def test_active_learning_never_trusts_new_sources():
    al = rd.active_learning(WORLD, NEXTV["blind_spot_clusters"], CATALOG)
    by = {a["source"]: a for a in al}
    assert by["earnings_calendar_history"]["expected_information_gain"] > by["portwatch_history_pit"]["expected_information_gain"]
    for a in al:
        assert "pending" in a["onboarding_checklist"].values() or a["onboarding_checklist"]["incremental_oos_test"] != "done"
        assert "nicht vertrauenswürdig" in a["note"]
    assert by["portwatch_history_pit"]["onboarding_checklist"]["timestamp_integrity"] == "forward_only"


def test_director_hypotheses_flow_into_lab(tmp_path, monkeypatch):
    import json
    p = tmp_path / "d.json"
    p.write_text(json.dumps({"hypotheses": [{"id": "RD-x", "title": "t", "signal": "rev_1m", "direction": -1}]}))
    hs = lab.load_director(p)
    assert hs and hs[0]["source"] == "director"

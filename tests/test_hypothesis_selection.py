"""Alpha Discovery 4.0: selection intelligence, NOT a new predictor or proof of alpha.

Research candidates are synthetic; no external sources, options calls or orders.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone, timedelta

import pytest

from modules import hypothesis_selection as hs
from modules import hypothesis_factory as hf
from modules import research_memory as rm
from modules import research_director as rd


NOW = datetime(2026, 11, 8, 9, 0, tzinfo=timezone.utc)


def idea(key, family, signal, p=1, exploratory=False, source="cross_domain", sd=0.17):
    return {"id": key, "family": family, "signal": signal, "priority": p,
            "idea_source": source, "exploratory": exploratory,
            "priority_components": {"posterior_sd": sd}, "direction": 1}


def test_capacity_exploratory_reserve_source_diversity_and_family_cap():
    candidates = [
        idea("A", "macro", "rank(tnx)", 2.0),
        idea("B", "macro", "rank(curve_10y_3m)", 1.99),
        idea("C", "sentiment", "rank(sentiment_score)", 1.1, exploratory=True, source="anomaly"),
        idea("D", "risk", "rank(vix)", 1.05, exploratory=True, source="drift"),
        idea("E", "stocks", "rank(mom_3m)", 1.9),
        idea("F", "value", "rank(rev_1m)", 1.8),
    ]
    chosen, report = hs.select(candidates, slots=4, exploratory_slots=2)
    assert len(chosen) == 4
    assert set(report["selection_order"]) >= {"C", "D"}
    assert len({c["family"] for c in chosen}) == 4
    assert "B" not in report["selection_order"]  # same-family duplicate, not twice paid
    assert report["distinct_families"] == 4
    assert report["slots"] == 4 and report["exploratory_slots"] == 2
    assert report["no_locked_holdout_used"] and report["no_new_model"]
    assert all(c["selection_audit"]["rank"] > 0 for c in chosen)
    assert all(c["selection_audit"]["basis"] == "pre-test evidence and Research Memory only" for c in chosen)


def test_duplicate_signal_under_new_name_cannot_consume_two_tests():
    a = idea("A", "one", "rank(vol_20)", 1.4)
    b = idea("B", "two", "rank( vol_20 )", 1.39)
    c = idea("C", "three", "rank(mom_3m)", 1.1)
    selected, report = hs.select([a, b, c], slots=3)
    assert [x["id"] for x in selected] == ["A", "C"]
    assert b["selection_audit"]["reason"] == "NEAR_DUPLICATE_EXPRESSION_IN_BATCH"
    assert report["unfilled"] == 1  # never use a fake/redundant experiment to fill budget


def test_expected_research_information_breaks_equal_priority_not_future_profit():
    high = idea("HIGH", "novel", "rank(vix)", p=1, sd=0.17)
    low = idea("LOW", "known", "rank(vol_20)", p=1, sd=0.01)
    high["future_pnl"] = -1000000
    low["future_pnl"] = 1000000
    chosen, _ = hs.select([low, high], slots=1)
    assert [h["id"] for h in chosen] == ["HIGH"]
    assert high["selection_audit"]["uncertainty_source"] == "existing_research_memory"
    assert high["selection_audit"]["marginal_score"] > low["selection_audit"]["marginal_score"]


def test_ranking_is_deterministic_even_if_input_order_changes():
    candidates = [idea("X", "f1", "rank(vix)"), idea("B", "f2", "rank(mom_3m)"),
                  idea("A", "f3", "rank(rev_1m)")]
    a, _ = hs.select(copy.deepcopy(candidates), slots=2)
    b, _ = hs.select(copy.deepcopy(list(reversed(candidates))), slots=2)
    assert [c["id"] for c in a] == [c["id"] for c in b] == ["A", "B"]


def test_cutoff_excludes_future_memory_without_rewriting_history():
    current = rm._entry("research", "NOW", "REJECTED", {"signal": "rank(vix)"})
    current["recorded_at"] = (NOW - timedelta(days=1)).isoformat()
    future = rm._entry("research", "FUTURE", "FORWARD_VALIDATED", {"signal": "rank(vix)"})
    future["recorded_at"] = (NOW + timedelta(days=1)).isoformat()
    naive = rm._entry("research", "NAIVE", "FORWARD_VALIDATED", {"signal": "rank(vol_20)"})
    naive["recorded_at"] = "2027-01-01T10:00:00"
    old = rm._entry("research", "LEGACY", "EXPLORATORY", {"signal": "rank(mom_3m)"})
    kept = hs.memory_as_of([future, current, naive, old], NOW)
    assert [x["hypothesis_id"] for x in kept] == ["NOW", "LEGACY"]
    with pytest.raises(ValueError, match="timezone-aware"):
        hs.memory_as_of([], datetime(2026, 11, 8))


def test_preexisting_factory_protocol_and_lab_are_unchanged():
    from pathlib import Path
    import hashlib
    digest = hashlib.sha256(Path("config/factory_protocol.yaml").read_bytes()).hexdigest()
    assert digest == "d1d9d16412d7656687755130f612ba4019bd9eaab6ef57f79b53f63091e3ede2"
    p = hf.plan(panel=None, memory=[], now=NOW, conditions={"sources": {}})
    assert p["selection"]["version"] == hs.VERSION
    assert p["selection"]["allocated"] <= p["budget"]["slots"] == 6
    assert p["selection"]["exploratory_allocated"] >= p["budget"]["exploratory_slots"]
    assert len({h["family"] for h in p["ideas"] if h["plan_status"] == "SELECTED"}) == len(p["selected"])
    assert all(h["selection_audit"]["decision"] in ("SELECTED", "NOT_SELECTED", "NOT_ELIGIBLE")
               for h in p["ideas"])
    assert all(h["spec_hash"] == hf.spec_hash(h) for h in p["ideas"])


def test_research_director_reuses_selector_without_extra_tests():
    from tests.test_research_director import META, NEXTV, WORLD, HYP_DB, CATALOG
    rep = rd.plan(META, NEXTV, WORLD, HYP_DB, CATALOG, max_new=3)
    assert rep["selection"]["version"] == hs.VERSION
    assert rep["selection"]["allocated"] == rep["n_selected_for_testing"]
    assert 0 < rep["n_selected_for_testing"] <= 3
    assert all(h["research_dossier"]["selection_audit"]["decision"] == "SELECTED"
               for h in rep["hypotheses"])
    assert all(h["research_dossier"]["promotion"] == "NEVER_AUTOMATIC"
               for h in rep["hypotheses"])



def test_director_does_not_use_future_memory_in_historical_replay():
    from tests.test_research_director import META, NEXTV, WORLD, HYP_DB, CATALOG
    past = rd.plan(META, NEXTV, WORLD, HYP_DB, CATALOG, max_new=3, now=NOW)
    assert past["selected"]
    first = next(h for h in past["hypotheses"] if h["signal"])
    record = rm._entry("research", "FORWARD-FUTURE", "REJECTED",
                       {"signal": first["signal"], "direction": first["direction"]})
    record["recorded_at"] = (NOW + timedelta(days=2)).isoformat()
    replay = rd.plan(META, NEXTV, WORLD, HYP_DB, CATALOG, max_new=3,
                     now=NOW, memory=[record])
    assert replay["selected"] == past["selected"]
    assert replay["memory_entries_considered"] == 0
    record["recorded_at"] = (NOW - timedelta(days=2)).isoformat()
    historical = rd.plan(META, NEXTV, WORLD, HYP_DB, CATALOG, max_new=3,
                         now=NOW, memory=[record])
    assert historical["memory_entries_considered"] == 1
    assert first["id"] not in historical["selected"]

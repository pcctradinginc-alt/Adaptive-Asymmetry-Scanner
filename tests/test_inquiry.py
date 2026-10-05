"""Kernfrage-Ketten, Forschungs-Lernkurve, hierarchischer Ideentyp-Prior, offene Fragen -> Hypothesen."""
from __future__ import annotations

from modules import hypothesis_factory as hf
from modules import inquiry as iq
from modules import research_memory as rm

SYS = {"drift_state": {"components": {
    "model": {"deteriorating": ["enet_xs20_v1"]},
    "feature": {"features_out_of_range": ["tnx"], "per_feature": {"tnx": {"p99": 4.78}}}}}}


def test_anomalies_from_measured_findings():
    nv = {"blind_spot_clusters": [{"id": "C1", "typical_error": -0.2, "n": 50, "lift": 2,
                                   "common_properties": {"sector": "Energy"}},
                                  {"id": "C2", "typical_error": -0.1, "n": 40, "lift": 1.5,
                                   "common_properties": {"lottery": "x"}}]}
    from modules.outcomes import RELIABILITY_DEFINITION
    fail = {"reliable": {"primary": {"timing_too_early": {"share": 0.2, "mean_outcome": -0.5}}},
            "reliability_definition": RELIABILITY_DEFINITION}
    stale = {k: v for k, v in fail.items() if k != "reliability_definition"}     # alte Definition: kein Befund
    assert not any(x["id"].startswith("failure:") for x in iq.anomalies(nv, None, stale, SYS))
    a = {x["id"]: x for x in iq.anomalies(nv, None, fail, SYS)}
    assert set(a) == {"C1", "C2", "model_drift:enet_xs20_v1", "feature_drift:tnx", "failure:timing_too_early"}
    assert "blind_spot_sector_match" in a["C1"]["keys"] and "blind_spot_sector_match" not in a["C2"]["keys"]


def test_chain_status_progression():
    assert iq.chain_status([]) == "OPEN_QUESTION"
    assert iq.chain_status([{"status": "DATA_GAP"}]) == "NEEDS_DATA"
    assert iq.chain_status([{"status": "REJECTED"}, {"status": "DATA_GAP"}]) == "HISTORICALLY_REJECTED"
    assert iq.chain_status([{"status": "NOT_YET_TESTED"}]) == "UNDER_TEST"
    assert iq.chain_status([{"status": "REJECTED"}, {"status": "PROSPECTIVE_CHALLENGER"}]) == "FORWARD_TEST"
    assert iq.chain_status([{"status": "LIMITED_PRODUCTION", "influence_level": "WEIGHT_10"}]) == "BEHAVIOUR_CHANGED"


def test_explanations_link_contract_only_to_matching_feature():
    contract = {"hypothesis_id": "PROM-ABST-003", "version": 1, "features": ["blind_spot_sector_match"],
                "title": "Blind-Spot-Sektor"}
    promo = {"hypotheses": {"PROM-ABST-003@v1": {"state": "PROSPECTIVE_CHALLENGER", "evidence": {"n_observations": 4}}}}
    sector = {"id": "C1", "kind": "blind_spot", "keys": ["C1", "blind_spot_sector_match"]}
    other = {"id": "C2", "kind": "blind_spot", "keys": ["C2"]}
    assert [e["id"] for e in iq.explanations(sector, None, None, None, None, promo, [contract])] == ["PROM-ABST-003@v1"]
    assert iq.explanations(other, None, None, None, None, promo, [contract]) == []


def test_failure_chains_use_tested_champion_rules():
    a = {"id": "failure:x", "kind": "prediction_error", "keys": ["x"]}
    props = {"rejected": [{"rule": "impact > 6", "reason": "Test-Δ nicht negativ"}]}
    ex = iq.explanations(a, None, None, None, None, None, [], props)
    assert ex[0]["status"] == "REJECTED" and iq.chain_status(ex) == "HISTORICALLY_REJECTED"


def _mem(status, prio, src="cross_domain", fam="f1", ts="2026-10-03T00:00:00+00:00", hid=None):
    return {"kind": "factory", "hypothesis_id": hid or f"H{prio}{status}", "status": status, "recorded_at": ts,
            "spec": {"family": fam, "idea_source": src, "priority": prio}}


def test_learning_curve_and_priority_calibration():
    ent = [_mem("ACCEPTED", 0.9 - i * 0.01, hid=f"A{i}") for i in range(5)] + \
          [_mem("REJECTED", 0.1 + i * 0.01, hid=f"R{i}") for i in range(6)]
    lc = rm.learning_curve(ent)
    assert lc["by_quarter"]["2026-Q4"] == {"tested": 11, "success_rate": round(5 / 11, 3)}
    assert lc["priority_calibration"]["spearman"] > 0.8
    assert lc["by_idea_source"]["cross_domain"]["tested"] == 11
    assert rm.learning_curve(ent[:3])["priority_calibration"]["spearman"] is None


def test_new_family_borrows_idea_source_experience():
    protocol = {"priority": {"prior_success": [1.0, 4.0]}}
    dirs = rm.directions([_mem("REJECTED", 0.5, src="drift", fam=f"f{i}", hid=f"D{i}") for i in range(8)],
                         (1.0, 4.0))
    h = {"family": "brand_new", "idea_source": "drift", "relevance": 1.0,
         "readiness": {"status": "READY", "cost": 1.0, "data_quality": 1.0}}
    h_unknown = {**h, "idea_source": "never_seen"}
    assert hf.priority(h, dirs, 0.0, protocol) < hf.priority(h_unknown, dirs, 0.0, protocol)


def test_open_question_becomes_falsifiable_hypothesis():
    ideas = hf.ideas_open_questions(SYS, "2026-10-03T00:00:00+00:00")
    assert len(ideas) == 1
    h = ideas[0]
    assert h["signal"] == "rank(mom_12_1) * step(tnx - 4.78)" and h["idea_source"] == "open_question"
    for k in ("population", "exposure", "lag", "horizon", "control_group", "primary_metric", "H0", "H_alt",
              "failure_condition"):
        assert h.get(k), k
    assert hf.ideas_open_questions({}, "x") == []

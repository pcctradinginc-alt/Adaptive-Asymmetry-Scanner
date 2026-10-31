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


# ── Options-Lernmodell: messbare Lücken ─────────────────────────────────────

def _opt_status(n=359, features=34, coverage=None, wf=None, fw=None):
    from modules import option_learner as ol
    cov = {f: 1.0 for f in (*ol.LEDGER_FEATURES, *ol.CONTRACT_FEATURES)} | (coverage or {})
    empty = {"verdict": "NEED_MORE_DATA", "dates": 0, "independent_periods": 0, "n_selected": 0, "trades": 0,
             "coverage": None, "representativeness": {"warnings": []}}
    return {"verified_labels": n, "feature_coverage": cov, "model": {"features": features} if features else None,
            "walk_forward": {**empty, **(wf or {})}, "forward": {**empty, **(fw or {})}}


def test_option_gaps_are_measured_and_prioritised():
    st = _opt_status(coverage={"news_novelty": 0.0},
                     wf={"representativeness": {"warnings": ["überwiegend Forschungsstichprobe"]}})
    gaps = rd.option_gaps({5: st, 20: {}})
    ids = [g["gap_id"] for g in gaps]
    assert "evidence:status_missing:20d" in ids[:2]                    # fehlender Status ist selbst eine Lücke
    assert {"data:sparse_feature:news_novelty:5d", "model:overfit_risk:5d", "model:untested_alternatives:5d",
            "evidence:insufficient:5d"} <= set(ids)
    assert any(i.startswith("representativeness:") for i in ids)
    assert all(0 <= g["severity"] <= 1 and g["metric"] and g["finding"] for g in gaps)
    assert [g["severity"] for g in gaps] == sorted((g["severity"] for g in gaps), reverse=True)
    sparse = next(g for g in gaps if g["gap_id"] == "data:sparse_feature:news_novelty:5d")
    assert sparse["value"] == 0.0 and sparse["feature"] == "news_novelty"


def test_option_gaps_need_a_measurement():
    few = rd.option_gaps({5: _opt_status(n=30, features=0, coverage={"news_novelty": 0.0})})
    assert not any(g["kind"] in ("data", "model") for g in few)        # zu wenige Labels: keine Behauptung
    tested = rd.option_gaps({5: _opt_status()}, variants=[{"horizon": 5}])
    assert "model:untested_alternatives:5d" not in {g["gap_id"] for g in tested}
    weak = rd.option_gaps({5: _opt_status(n=2000, wf={
        "verdict": "NOT_VALIDATED", "trades": 40, "coverage": 0.5, "unevaluable": {"EXIT_NOT_ARCHIVED": 20},
        "mean_selected": -0.02, "mean_all": 0.01, "vs_default": {"mean_diff": -0.05, "ci95": [-0.09, -0.01]}})})
    ids = {g["gap_id"] for g in weak}
    assert {"model:no_skill:5d", "data:outcome_coverage:walk_forward:5d", "model:selection_no_value:5d",
            "model:choice_worse_than_default:5d"} <= ids
    assert "model:overfit_risk:5d" not in ids                           # 2000 Labels / 34 Merkmale reichen


def test_run_options_writes_improvement_and_meta_cognition_shows_it(tmp_path, monkeypatch):
    import json

    from modules import meta_cognition as mc
    (tmp_path / "status_5d.json").write_text(json.dumps(_opt_status(coverage={"news_novelty": 0.0})))
    rep = rd.run_options(tmp_path)
    saved = json.loads((tmp_path / "improvement.json").read_text())
    assert saved["gaps"] == json.loads(json.dumps(rep["gaps"]))
    st = mc.machine_state({"option_improvement": saved}, {}, {}, {}, {"active": False, "reasons": []})
    assert any("news_novelty" in g for g in st["option_learner_gaps"])
    assert "Options-Lernmodell: erkannte Lücken" in mc.render_md(st)


PROPOSAL_FIELDS = {"proposal_id", "kind", "horizon", "gap_ids", "title", "change", "hypothesis", "expected_benefit",
                   "cost", "data_availability", "falsification", "auto_experiment", "approval_required", "priority"}


def test_option_proposals_are_structured_and_only_allowlisted_variants_run_automatically():
    from modules import option_learner as ol
    cfg = ol.variant_config()
    gaps = rd.option_gaps({5: _opt_status(coverage={"news_novelty": 0.0},
                                          wf={"representativeness": {"warnings": ["einseitig"]}})})
    props = rd.option_proposals(gaps, cfg, CATALOG)
    assert props and all(PROPOSAL_FIELDS <= set(p) for p in props)
    assert len({p["proposal_id"] for p in props}) == len(props)
    auto = [p for p in props if p["auto_experiment"]]
    assert auto and all(p["kind"] == "variant" and not p["approval_required"] for p in auto)
    assert all(p["cost"]["api_calls"] == 0 and p["cost"]["llm_calls"] == 0 and not p["cost"]["paid_data"] for p in auto)
    for p in auto:
        assert ol.validate_overrides(p["change"], cfg) == p["change"]      # nur freigegebene Methode und Wert
        assert p["falsification"]["max_days"] == 90 and "reject_if" in p["falsification"]
    alphas = [p["change"]["ridge_alpha"] for p in auto if "ridge_alpha" in p["change"]]
    assert alphas and all(a > ol.RIDGE_ALPHA for a in alphas)           # Überanpassung: nur stärker regularisieren
    assert {p["kind"] for p in props if not p["auto_experiment"]} == {"data_pipeline", "sampling"}
    merged = [p for p in auto if len(p["gap_ids"]) > 1]                    # gleiche Variante aus zwei Lücken
    assert merged and all(len(set(p["gap_ids"])) == len(p["gap_ids"]) for p in merged)


def test_data_sources_are_only_proposed_for_missing_skill_and_never_bought_automatically():
    from modules import option_learner as ol
    cat = CATALOG + [{"id": "ibes_estimate_revisions", "dimension": ["earnings_momentum"], "provider": "Refinitiv",
                      "license": "kommerziell, REVIEW_REQUIRED", "pit": "vintages", "history_from": 1980,
                      "acquisition_cost": 5, "status": "candidate"}]
    calm = rd.option_proposals(rd.option_gaps({5: _opt_status()}), ol.variant_config(), cat)
    assert not any(p["kind"] == "data_source" for p in calm)                # keine Daten ohne gemessenen Bedarf
    weak = rd.option_gaps({20: _opt_status(n=2000, wf={"verdict": "NOT_VALIDATED"})})
    ds = {p["change"]["source"]: p for p in rd.option_proposals(weak, ol.variant_config(), cat)
          if p["kind"] == "data_source"}
    assert set(ds) == {"earnings_calendar_history", "portwatch_history_pit", "ibes_estimate_revisions"}
    assert ds["ibes_estimate_revisions"]["cost"]["paid_data"] is True
    assert all(p["approval_required"] and not p["auto_experiment"] for p in ds.values())
    assert ds["earnings_calendar_history"]["priority"] > ds["ibes_estimate_revisions"]["priority"]


def test_run_options_registers_only_free_allowlisted_variants_once(tmp_path):
    import json
    from datetime import datetime, timezone

    from modules import option_learner as ol
    (tmp_path / "status_5d.json").write_text(json.dumps(_opt_status(coverage={"news_novelty": 0.0})))
    now = datetime(2026, 11, 7, 9, tzinfo=timezone.utc)
    rep = rd.run_options(tmp_path, now)
    reg = ol.variant_registry(tmp_path)
    assert len(rep["registered_variants"]) == 3 and set(rep["registered_variants"]) == set(reg)
    assert {next(iter(v["overrides"])) for v in reg.values()} == {"ridge_alpha", "drop_feature_group", "min_pred"}
    auto_ids = {p["proposal_id"] for p in rep["proposals"] if p["auto_experiment"]}
    assert all(v["proposal_id"] in auto_ids for v in reg.values())
    again = rd.run_options(tmp_path, now)
    assert again["registered_variants"] == []
    assert "model:untested_alternatives:5d" not in {g["gap_id"] for g in again["gaps"]}

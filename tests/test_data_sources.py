"""Datenquellen-Programm: SourceContracts vollständig, Meta-Learning über Quellen
(Quelle × Sektor × Regime × Horizont, Alpha Decay), Scoreboard-Status, Forward-Wert,
Hypothesen aus Quellen-Bedingungen nur aus Auswahljahren, Bericht."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

import tests.test_ml_research as T
from modules.alt_data import evaluate as ev
from modules.alt_data.registry import ALT_CONTRACT_FIELDS, SOURCES, contract_gaps, load_contracts


def test_every_alt_source_has_complete_contract_and_secret_free_auth():
    assert contract_gaps() == {}
    cs = load_contracts()
    for s in SOURCES.values():
        for cid in s["contracts"]:
            c = cs[cid]
            assert set(ALT_CONTRACT_FIELDS) <= set(c) and c["semantics"].get("mechanisms")
            assert c["revision_risk"] in ("low", "medium", "high") and c["maintenance_cost"] in ("low", "medium", "high")
            assert not c.get("requires_auth") or c.get("auth_env_variable")        # Keys nur per Env/Secret
    text = "".join(p.read_text() for p in __import__("pathlib").Path("config/external_sources").glob("*.yaml"))
    assert "api_key=" not in text.lower() and "token=" not in text.lower()


def test_alpha_decay_classification():
    assert ev.alpha_decay({"2019": 0.04, "2020": 0.035, "2021": 0.01, "2022": 0.005})["status"] == "decaying"
    assert ev.alpha_decay({"2019": 0.03, "2020": 0.03, "2021": 0.03, "2022": 0.028})["status"] == "stable"
    assert ev.alpha_decay({"2019": 0.03, "2020": 0.02, "2021": -0.02, "2022": -0.03})["status"] == "reversed"
    assert ev.alpha_decay({"2019": 0.001, "2020": 0.002, "2021": 0.0, "2022": 0.003})["status"] == "no_early_effect"
    assert ev.alpha_decay({"2019": 0.03})["status"] == "insufficient_years"


@pytest.fixture(scope="module")
def cond_panel():
    from modules import ml_research as ml
    old = ml.MIN_CROSS_SECTION
    ml.MIN_CROSS_SECTION = 20
    p = T._panel(n_days=1400, n_stocks=60, signal=False)
    rnd = np.random.default_rng(4)
    p["sector"] = np.where(p["ticker"].str[1:].astype(int) % 2 == 0, "Technology", "Energy")
    p["vix"] = 15.0 + 10 * (p["date"].dt.month % 2)
    p["xbrl_sue"] = rnd.normal(0, 1, len(p))
    tech = (p["sector"] == "Technology").astype(float)
    p["fwd_xs_20"] = 0.03 * tech * p["xbrl_sue"] + rnd.normal(0, 0.02, len(p))     # wirkt nur in Technology
    p.loc[p["label_end_20"].isna(), "fwd_xs_20"] = np.nan
    yield p
    ml.MIN_CROSS_SECTION = old


def test_conditional_value_finds_sector_cell_and_horizons(cond_panel):
    c = ev.conditional_value(cond_panel, ["xbrl_sue", "missing_feature"], [2016, 2017, 2018])
    assert "missing_feature" not in c
    sec = c["xbrl_sue"]["sector"]
    assert sec["Technology"]["20"]["t_months"] > 3 and abs(sec["Energy"]["20"]["t_months"] or 0) < 3
    assert set(c["xbrl_sue"]["all"]) == {"20", "60"} and "vol:vix_ge_20" in c["xbrl_sue"]["regime"]
    assert c["xbrl_sue"]["alpha_decay"]["status"] in ("stable", "insufficient_years", "decaying", "reversed")


def test_factory_turns_selection_cells_into_scoped_hypotheses(cond_panel):
    from modules import hypothesis_factory as hf
    cond = {"selection_years": [2016, 2017, 2018], "sources": {"sec_xbrl_fundamentals": {
        "verdict": "REJECT", "selection": ev.conditional_value(cond_panel, ["xbrl_sue"], [2016, 2017, 2018]),
        "dev": {"xbrl_sue": {"sector": {"Energy": {"20": {"t_months": 9.0, "mean_ic": 0.2}}}}}}}}
    ideas = hf.ideas_source_conditions(cond, "2026-10-03")
    sigs = [h["signal"] for h in ideas]
    assert "exp_technology * rank(xbrl_sue)" in sigs
    assert not any("exp_energy" in s for s in sigs)                                # dev-Zellen nie als Quelle
    h = next(h for h in ideas if h["signal"] == "exp_technology * rank(xbrl_sue)")
    assert h["direction"] == 1 and h["source_id"] == "sec_xbrl_fundamentals" and h["H0"] and h["failure_condition"]
    from modules.research_lab import validate_expr
    for h in ideas:
        validate_expr(h["signal"])
    assert len(ideas) <= hf.MAX_CONDITION_IDEAS
    res = hf.plan(None, memory=[], conditions=cond, now=datetime(2026, 10, 3, tzinfo=timezone.utc))
    assert any(i["idea_source"] == "source_condition" for i in res["ideas"])


def test_factory_priority_falls_back_to_source_experience():
    from modules import hypothesis_factory as hf
    from modules import research_memory as rm
    proto = hf.load_protocol()
    h = {"family": "cond_new", "source_id": "sec_deep_events", "relevance": 1.0,
         "readiness": {"status": "READY", "data_quality": 1.0, "cost": 1.0}}
    bad = rm.directions([rm._entry("research", f"A{i}", "REJECTED", {"signal": f"rank(vol_20)*{i}"}, source="alt_data:sec_deep_events")
                         for i in range(8)])
    assert hf.priority(h, bad, 0.0, proto) < hf.priority(h, {}, 0.0, proto)          # verworfene Quelle -> weniger Budget


def test_board_status_never_promotes_itself():
    assert ev.board_status("REJECT", {}, True) == "REJECTED"
    assert ev.board_status(None, {}, False) == "RESEARCH"
    assert ev.board_status("KEEP", {"n_cohorts": 99, "value": 0.05}, True) == "CHALLENGER"
    assert ev.board_status("KEEP", {}, False) == "SHADOW"
    assert ev.board_status("MODIFY", {}, True) == "SHADOW"


def test_forward_value_uses_only_labelled_cohorts_of_source(tmp_path, cond_panel):
    led = tmp_path / "f.jsonl"
    d1, d2 = sorted(cond_panel["date"].unique())[100], sorted(cond_panel["date"].unique())[-1]
    top = cond_panel[cond_panel["date"] == d1].nlargest(6, "fwd_xs_20")["ticker"].tolist()
    led.write_text("\n".join(json.dumps(r) for r in [
        {"hypothesis_id": "ALT-XBRL-001", "date": str(pd.Timestamp(d1).date()), "top_decile": top},
        {"hypothesis_id": "ALT-XBRL-001", "date": str(pd.Timestamp(d2).date()), "top_decile": top},   # kein Label
        {"hypothesis_id": "ALT-SEC-001", "date": str(pd.Timestamp(d1).date()), "top_decile": top}]) + "\n")
    fv = ev.forward_value(cond_panel, "sec_xbrl_fundamentals", led)
    assert fv["n_cohorts"] == 1 and fv["value"] > 0
    assert ev.forward_value(cond_panel, "ted_procurement", led)["n_cohorts"] == 0


def test_weekly_alt_health_blocks():
    from reports import weekly
    alt = {"health": {"sec_xbrl_fundamentals": {"last_observation": "2026-10-02", "coverage": 0.8, "error_rate": 0.01,
                                                "schema_errors": 0, "checked_at": "2026-10-03T08:00"}},
           "entity": {"gleif": {"HIGH": 90, "MEDIUM": 3, "LOW": 40, "remaining": 100}, "gleif_children": {"children": 55}},
           "conditions": {"sources": {"s": {"dev": {"f": {"sector": {"Tech": {"20": {"t_months": 3.1, "mean_ic": 0.02}}}}}}}}}
    board = {"a": {"verdict": "REJECT"}, "b": {"verdict": "KEEP", "forward_value": 0.004, "forward_cohorts": 30,
                                               "alpha_decay": {"f": "decaying"}}}
    text = json.dumps(weekly.alt_health_blocks(alt, board), ensure_ascii=False)
    for s in ("DATA SOURCE HEALTH", "sec_xbrl_fundamentals", "90 HIGH", "Töchter: 55", "b: Forward",
              "a: REJECT", "b/f: decaying", "s/f × Tech × 20 T"):
        assert s in text, s

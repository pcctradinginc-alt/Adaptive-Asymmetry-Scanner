"""Alpha Discovery 2.0: a reused historical holdout can never manufacture confirmation.

Synthetic evidence is used only to test classifications and the complete
historical-screen -> preregistration boundary. Not a market backtest.
"""
from __future__ import annotations

from copy import deepcopy
import json

import pandas as pd
import yaml

from modules import research_lab as lab
from modules import research_memory as memory
from modules.autonomous_alpha import registry


def test_cached_historical_success_is_reinterpreted_without_deleting_source():
    cached = {"status": "accepted", "canonical_status": "ACCEPTED",
              "locked_holdout_status": "CONTAMINATED", "p_value": 1e-30,
              "walk_forward": {"base": {"mean": 100.0}}}
    before = deepcopy(cached)
    assert memory.hypothesis_status(cached) == "EXPLORATORY"
    assert lab.canonical_status(cached) == "EXPLORATORY"
    assert cached == before  # preservation: no historical file is mutated
    assert "EXPLORATORY" not in memory.SUCCESS
    assert "ACCEPTED" not in memory.SUCCESS
    assert memory.hypothesis_status({"status": "ROBUST", "canonical_status": "ROBUST",
                                     "evidence_class": "EXPLORATORY_HISTORICAL"}) == "EXPLORATORY"
    assert memory.hypothesis_status({"status": "VALIDATED", "canonical_status": "VALIDATED",
                                     "locked_holdout_status": "CONTAMINATED"}) == "EXPLORATORY"


def test_all_sources_use_contaminated_locked_only_as_historical_screen(tmp_path, monkeypatch):
    settings = yaml.safe_load(open("config/research_protocol.yaml", encoding="utf-8"))
    assert settings["periods"]["locked_status"] == "CONTAMINATED"
    hyps = tmp_path / "hyps.yaml"
    hyps.write_text(yaml.safe_dump({"hypotheses": [
        {"id": "H-SYN", "source": "config", "title": "Synthetic enormous past return",
         "signal": "mom_3m", "direction": 1}]}))
    db = tmp_path / "db.json"
    def fake_historical_screen(panel, hyp, tested, protocol):
        return {"id": hyp["id"], "source": hyp["source"], "status": "passed_pre_fdr",
                "signal": hyp["signal"], "p_value": 1e-30,
                "walk_forward": {"base": {"mean": 10000.0, "t_months": 500.0}},
                "stages": {"coverage": "pass"}}
    monkeypatch.setattr(lab, "evaluate_hypothesis", fake_historical_screen)
    monkeypatch.setattr(lab, "locked_check", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("contaminated holdout must not be retested")))
    panel = pd.DataFrame({"date": pd.to_datetime(["2026-10-01"]), "mom_3m": [0.3]})
    out = lab.run(panel, protocol=settings, hyp_path=hyps, db_path=db, with_discovery=False)
    rec = out["hypotheses"]["H-SYN"]
    assert rec["status"] == "historical_exploratory_pending_forward"
    assert rec["canonical_status"] == "EXPLORATORY"
    assert rec["lab_screen_passed"] is True
    assert rec["locked_holdout_status"] == "CONTAMINATED"
    assert "H-SYN" not in [x for x, r in out["hypotheses"].items()
                           if r.get("canonical_status") in memory.SUCCESS]
    assert json.loads(db.read_text())["hypotheses"]["H-SYN"] == rec


def test_registry_does_not_rebrand_old_accepted_as_validated():
    rec = {"id": "H-SYN", "status": "accepted", "canonical_status": "ACCEPTED",
           "source": "factory", "signal": "mom_3m",
           "walk_forward": {"base": {"mean": 123.0}}}
    snap = registry._snapshot("research_lab", "H-SYN", rec, "2026-11-01",
                              "outputs/research/hypothesis_db.json", "CONTAMINATED")
    assert snap["status"] == "EXPLORATORY"
    assert snap["source_status"] == "ACCEPTED"
    assert snap["locked_holdout_status"] == "CONTAMINATED"
    assert snap["promotion_eligible"] is False
    assert snap["metrics"]["oos_net_mean"] == 123.0 or "metrics" in snap

"""Autonomous factor exploration: offline, bounded, immutable, prospectively honest."""
import json
from datetime import datetime, timezone

import pandas as pd
import pytest
import yaml

from modules.autonomous_alpha import explorer
from modules.autonomous_alpha import registry as reg
from modules import research_lab as lab


def _now(day):
    return datetime.fromisoformat(day).replace(tzinfo=timezone.utc)


def _setup(root, *, cap=4):
    cfg = {
        "version": "explore-v1", "enabled": True,
        "monthly_new_hypotheses": 2, "lifetime_hypotheses": cap,
        "max_variants_per_family": 2, "exploratory_fraction": 0.5, "seed": 41,
        "parents": [
            {"name": "momentum", "feature": "mom_12_1", "direction": 1},
            {"name": "reversal", "feature": "rev_1m", "direction": -1},
            {"name": "volatility", "feature": "vol_60", "direction": -1},
            {"name": "liquidity", "feature": "relvol_5_60", "direction": 1}
        ],
    }
    config = root / explorer.POLICY
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    db = root / explorer.HISTORICAL_DB
    db.parent.mkdir(parents=True, exist_ok=True)
    db.write_text(json.dumps({"hypotheses": {
        "HYP-PREV": {"signal": "mom_12_1", "direction": 1,
                     "canonical_status": "REJECTED"}}}), encoding="utf-8")
    return config


def test_monthly_registration_is_automatic_but_idempotent(tmp_path):
    _setup(tmp_path)
    a = explorer.register(tmp_path, _now("2026-10-28"))
    assert a["n_registered_new"] == 2
    assert a["production_effect"] == "NONE"
    assert len(explorer.proposals(tmp_path)) == 2
    again = explorer.register(tmp_path, _now("2026-10-29"))
    assert again["status"] == "already_registered_this_month"
    assert again["n_registered_new"] == 0
    b = explorer.register(tmp_path, _now("2026-11-04"))
    assert b["n_registered_new"] == 2
    assert len(explorer.proposals(tmp_path)) == 4
    done = explorer.register(tmp_path, _now("2026-12-04"))
    assert done["status"] == "budget_exhausted"
    assert len(explorer.proposals(tmp_path)) == 4
    assert len({h["id"] for h in explorer.proposals(tmp_path)}) == 4


def test_registered_genomes_only_use_point_in_time_whitelist(tmp_path):
    _setup(tmp_path)
    explorer.register(tmp_path, _now("2026-10-28"))
    for p in explorer.proposals(tmp_path):
        assert p["source"] == "autonomous_explorer"
        assert p["direction"] == 1
        assert p["id"].startswith("AUTO-")
        assert not any(x in p["signal"] for x in ("fwd_", "label_end", "mfe_"))
        assert lab.validate_expr(p["signal"]) is not None
        assert reg.digest(p["contract"]) == p["spec_hash"]


def test_same_seed_same_unseen_candidate_choice(tmp_path):
    _setup(tmp_path)
    x = explorer.select(tmp_path, _now("2026-10-28"))
    y = explorer.select(tmp_path, _now("2026-10-28"))
    assert [e["id"] for e in x["new"]] == [e["id"] for e in y["new"]]


def test_broken_archive_is_fail_closed(tmp_path):
    _setup(tmp_path)
    explorer.register(tmp_path, _now("2026-10-28"))
    p = tmp_path / explorer.PROPOSALS
    lines = p.read_text().splitlines()
    entry = json.loads(lines[0])
    entry["hypothesis"]["contract"]["signal"] = "fwd_xs_20"
    lines[0] = json.dumps(entry)
    p.write_text("\n".join(lines) + "\n")
    with pytest.raises(reg.EvidenceIntegrityError, match="content hash mismatch"):
        explorer.register(tmp_path, _now("2026-11-04"))


def test_future_features_rejected_by_policy(tmp_path):
    path = _setup(tmp_path)
    cfg = yaml.safe_load(path.read_text())
    cfg["parents"][0]["feature"] = "fwd_xs_20"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(reg.EvidenceIntegrityError, match="unsafe"):
        explorer.select(tmp_path, _now("2026-10-28"))


def test_global_budget_hard_caps_cannot_be_relaxed(tmp_path):
    path = _setup(tmp_path)
    cfg = yaml.safe_load(path.read_text())
    cfg["monthly_new_hypotheses"] = 12
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(reg.EvidenceIntegrityError, match="hard cap"):
        explorer.select(tmp_path, _now("2026-10-28"))


def test_explorer_historical_success_is_only_research_allocation(tmp_path):
    _setup(tmp_path)
    succ, rej = explorer._evidence_counts({
        "A": {"signal": "mom_12_1", "canonical_status": "ACCEPTED"},
        "B": {"signal": "rev_1m", "canonical_status": "REJECTED"},
    }, [{"feature": "mom_12_1"}, {"feature": "rev_1m"}])
    assert (succ, rej) == (1, 1)
    report = explorer.register(tmp_path, _now("2026-10-28"))
    assert report["requires_independent_forward_validation"]


def test_adaptive_hypothesis_cannot_be_accepted_using_contaminated_holdout(tmp_path, monkeypatch):
    _setup(tmp_path)
    explorer.register(tmp_path, _now("2026-10-28"))
    hyps = explorer.proposals(tmp_path)
    monkeypatch.setattr(lab, "load_hypotheses", lambda *a,**k: [])
    monkeypatch.setattr(lab, "load_director", lambda *a,**k: [])
    monkeypatch.setattr(lab, "load_factory", lambda *a,**k: [])
    monkeypatch.setattr(lab, "load_autonomous", lambda *a,**k: [hyps[0]])
    monkeypatch.setattr(lab, "add_exposures", lambda df: df)
    monkeypatch.setattr(lab, "memory_fields", lambda *a,**k: {})
    def evaluated(panel,hyp,seen,policy):
        return {"id": hyp["id"], "source": "autonomous_explorer",
                "signal": hyp["signal"], "status": "passed_pre_fdr",
                "p_value": 0.000001, "reasons": [], "stages": {}}
    monkeypatch.setattr(lab, "evaluate_hypothesis", evaluated)
    def forbidden(*a, **k):
        raise AssertionError("contaminated locked holdout must never be evaluated")
    monkeypatch.setattr(lab, "locked_check", forbidden)
    from modules.alt_data import contracts as ac
    monkeypatch.setattr(ac, "load", lambda: [])
    monkeypatch.setattr(ac, "register", lambda _: {})
    monkeypatch.setattr(ac, "lab_hypotheses", lambda *a: [])
    policy={"periods":{"locked_status":"CONTAMINATED"},
            "multiple_testing":{"fdr_q":0.1},
            "hypothesis_acceptance":{"locked_must_be_positive":True}}
    result=lab.run(pd.DataFrame(),protocol=policy,
                   db_path=tmp_path/"db.json",with_discovery=False)
    row=result["hypotheses"][hyps[0]["id"]]
    assert row["status"]=="historical_exploratory_pending_forward"
    assert row["canonical_status"]=="INCONCLUSIVE"


def test_existing_registered_proposals_all_remain_in_lab_testing_cohort(tmp_path,monkeypatch):
    _setup(tmp_path)
    explorer.register(tmp_path, _now("2026-10-28"))
    explorer.register(tmp_path, _now("2026-11-04"))
    monkeypatch.chdir(tmp_path)
    found=lab.load_autonomous()
    assert len(found)==4
    assert len(set(h["id"] for h in found))==4


def test_tampering_execution_signal_without_changing_contract_is_rejected(tmp_path):
    _setup(tmp_path)
    explorer.register(tmp_path, _now("2026-10-28"))
    path = tmp_path / explorer.PROPOSALS
    lines = path.read_text().splitlines()
    entry = json.loads(lines[0])
    entry["hypothesis"]["signal"] = "mom_12_1"
    lines[0] = json.dumps(entry)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(reg.EvidenceIntegrityError, match="mutable fields"):
        explorer.proposals(tmp_path)

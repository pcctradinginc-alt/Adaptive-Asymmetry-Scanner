"""Phase 3: bounded cost-aware research proposals, with no alpha promotion."""
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from modules import research_feature_director as d
from modules import research_feature_discovery as fd


def sources(root, *, dates=5, coverage=.5, windows=0, positive=False):
    proto, digest = fd.load_protocol()
    features = {h["feature"]: {"coverage": coverage, "dates_available": dates}
                for h in proto["hypotheses"]}
    status = {"protocol_sha256": digest, "protocol_version": proto["version"],
              "features": features}
    validation = {"protocol_sha256": digest, "global_alpha_ledger_slot_allocated": False,
                  "hypotheses": {}}
    for h in proto["hypotheses"]:
        validation["hypotheses"][h["id"]] = {
            "independent_forward_windows": windows,
            "looks": [{"paired_stock_mean": .07 if positive else -.01,
                       "local_bonferroni_diagnostic_p": .01 if positive else .98,
                       "paired_stock_95pct_bootstrap": [.01, .12] if positive else [-.05, .04]}
                      ] if windows >= 8 else [],
        }
    root.mkdir(parents=True, exist_ok=True)
    (root / fd.REPORT).write_text(json.dumps(status))
    (root / "paired_validation.json").write_text(json.dumps(validation))
    return status, validation


def test_sparse_data_causes_research_data_action_only(tmp_path):
    status, validation = sources(tmp_path)
    plan = d.propose(status, validation, [])
    assert len(plan["selected"]) == d.MAX_ACTIONS
    assert plan["global_alpha_ledger_slot_allocated"] is False
    assert all(row["action"] == "COLLECT_PIT" for row in plan["selected"])
    assert all(row["proposed_next_study"]["require_owner_approval"] for row in plan["selected"])
    assert all(row["proposed_next_study"]["production_influence"] == "NONE"
               for row in plan["all_candidates"])


def test_positive_forward_stock_diagnostic_is_never_a_trade_authorization(tmp_path):
    status, validation = sources(tmp_path, dates=60, coverage=.98, windows=8, positive=True)
    plan = d.propose(status, validation, [])
    assert all(p["action"] == "DESIGN_OPTION_TRIAL" for p in plan["all_candidates"])
    assert all(x["proposed_next_study"]["require_global_alpha_ledger_slot"] for x in plan["selected"])
    assert plan["new_hypotheses_registered"] == 0
    assert plan["champion_or_trade_modified"] is False


def test_result_changes_do_not_register_second_research_decision(tmp_path):
    sources(tmp_path, dates=60, coverage=1, windows=0)
    now = datetime(2026, 11, 8, tzinfo=timezone.utc)
    plan = d.run(root=tmp_path, memory_path=tmp_path / "none.jsonl", now=now)
    assert plan["new_decisions"] == 2
    repeat = d.run(root=tmp_path, memory_path=tmp_path / "none.jsonl", now=now)
    assert repeat["new_decisions"] == 0
    history, _ = d.read_journal(tmp_path / "director_decisions.jsonl")
    assert len(history) == 2
    assert all(r["no_auto_registration"] for r in history)


def test_tampered_research_prioritization_journal_fails_closed(tmp_path):
    sources(tmp_path, dates=60, coverage=1, windows=0)
    now = datetime(2026, 11, 8, tzinfo=timezone.utc)
    d.run(root=tmp_path, memory_path=tmp_path / "none", now=now)
    path = tmp_path / "director_decisions.jsonl"
    path.write_text(path.read_text().replace("CONTINUE_FORWARD", "PLACE_ORDER", 1))
    with pytest.raises(ValueError, match="chain broken"):
        d.run(root=tmp_path, memory_path=tmp_path / "none", now=now)


def test_source_change_invalidates_protocol_hash(tmp_path):
    status, validation = sources(tmp_path)
    validation["protocol_sha256"] = "x" * 64
    with pytest.raises(ValueError, match="mismatch"):
        d.propose(status, validation, [])


def test_prior_falsifications_reduce_priority_not_alpha():
    items = [{"kind": "feature_director_v1",
              "spec": {"signal": "fd3_liquidity_x_beta"}, "status": "REJECTED"}]
    m = d._memory_for_feature(items, "fd3_liquidity_x_beta")
    assert m["previous_falsifications"] == 1
    assert m["prior_forward_confirmed"] is False


def test_main_scanner_still_has_no_fd3_director_dependency():
    source = (Path(__file__).resolve().parents[1] / "pipeline.py").read_text()
    assert "research_feature_director" not in source

"""P6 synthetic end-to-end boundary: measured idea -> historical screen ->
immutable pre-forward registration -> no premature alpha confirmation.

No generated code execution, real API, broker call, production modifications or
synthetic price disguised as authentic market data.
"""
import json
from datetime import datetime, timezone

import pandas as pd
import yaml

from modules import hypothesis_factory as factory
from modules import research_director as director
from modules import research_lab as lab
from modules import research_memory as memory
from modules.autonomous_alpha import option_economics as economics


def test_synthetic_full_research_registration_memory_and_followup(tmp_path, monkeypatch):
    now = datetime(2026, 11, 9, 17, 0, tzinfo=timezone.utc)
    h = {"id": "FAC-SYN-E2E", "title": "Verzögerte Kursreaktion",
         "signal": "mom_3m", "direction": 1, "family": "price_action",
         "horizon": 20, "population": "SYNTHETIC", "exposure": "US_EQ",
         "control_group": "sector_baseline", "lag": 1, "primary_metric": "net",
         "failure_condition": "net <= 0", "mechanism": "Synthetic test of lagged pricing"}
    h["spec_hash"] = factory.spec_hash(h)
    lab.validate_expr(h["signal"])
    panel = pd.DataFrame({"date": [pd.Timestamp("2026-10-01")], "mom_3m": [0.5]})
    inp = tmp_path / "hypotheses.yaml"
    inp.write_text(yaml.safe_dump({"hypotheses": [
        {"id": h["id"], "source": "factory", "signal": h["signal"], "direction": 1,
         "spec_hash": h["spec_hash"], "family": h["family"]}]}))
    hyps = tmp_path / "factory_hyps.json"
    hyps.write_text(json.dumps({"hypotheses": [h]}))
    db = tmp_path / "hypothesis_db.json"
    # Stub only the historical statistical screener (this is NOT a claim
    # of a real economic effect). Keep real FDR and real contaminated-holdout gate.
    def fake_screen(panel, hyp, tested, protocol):
        return {"status": "passed_pre_fdr", "source": "factory", "signal": hyp["signal"],
                "p_value": 1e-30, "historical_net_mean": 10.0,
                "stages": {"coverage": "pass"}}
    monkeypatch.setattr(lab, "evaluate_hypothesis", fake_screen)
    monkeypatch.setattr(lab, "locked_check", lambda *a, **k:
                        (_ for _ in ()).throw(AssertionError("historical locked holdout reused")))
    report = lab.run(panel, hyp_path=inp, db_path=db, with_discovery=False)
    screened = report["hypotheses"][h["id"]]
    assert screened["lab_screen_passed"] is True
    assert screened["canonical_status"] == "EXPLORATORY"
    assert not screened.get("locked")

    challengers = tmp_path / "factory_challengers.jsonl"
    forward = tmp_path / "factory_forward_ledger.jsonl"
    result_path = tmp_path / "factory_results.json"
    monkeypatch.setattr(factory, "CHALLENGERS", challengers)
    monkeypatch.setattr(factory, "FORWARD_LEDGER", forward)
    monkeypatch.setattr(factory, "RESULTS_OUT", result_path)
    monkeypatch.setattr(factory, "battery", lambda *a, **kw:
                        {"robust": True, "tests": {"synthetic_robustness": "PASSED"}, "reasons": []})
    monkeypatch.setattr(factory, "record_forward", lambda panel: 0)
    result = factory.evaluate(panel, now=now, db_path=db, hyp_path=hyps)
    assert result["results"][h["id"]]["status"] == "PROSPECTIVE_CHALLENGER"
    frozen = json.loads(challengers.read_text().strip())
    assert frozen["hypothesis_id"] == h["id"]
    assert frozen["spec_hash"] == h["spec_hash"]
    assert frozen["forward_start"] > now.date().isoformat()
    assert not forward.exists()  # no future labels and NO artificial win

    memory_file = tmp_path / "research_memory.jsonl"
    n = memory.sync(path=memory_file, sources={
        "hypothesis_db": db, "factory_results": result_path}, now=now.isoformat())
    assert n >= 2
    entries = memory.load(memory_file)
    assert any(e["status"] == "EXPLORATORY" for e in entries)
    assert not any(e.get("prospective") for e in entries)
    assert not any(e["status"] in memory.SUCCESS for e in entries)

    # A copy/paste reformulation of this exact idea must not consume a new
    # hypothesis slot. Research Director sees the existing memory first.
    same = {"kind": "testable", "title": "Same historical idea",
            "question": "Same hypothesis, different wording",
            "statement": "Same idea", "signal": h["signal"],
            "direction": 1, "rationale": "synthetic",
            "evidence_t": 3, "value": 0.2, "available_data": True}
    monkeypatch.setattr(director, "candidates_from_blind_spots", lambda *a: [same])
    monkeypatch.setattr(director, "candidates_from_regimes", lambda *a: [])
    monkeypatch.setattr(director, "candidates_from_drift", lambda *a: [])
    choice = director.plan({}, {}, {}, {"hypotheses": {}}, [], memory=entries)
    assert choice["n_selected_for_testing"] == 0
    assert choice["deferred"][0]["reason"] == "RESEARCH_MEMORY_SIMILAR_PREVIOUSLY_TESTED"

    # The existing price economics can still be evaluated, but mock prices
    # here are a unit-test ONLY, never a verified trade or an alpha result.
    rule = {"contracts_per_position": 1, "commission_usd_per_contract_per_side": 0.65,
            "adverse_slippage_fraction_of_spread": 0.25}
    call = [{"occ_symbol": "TEST", "side": "long", "bid": 1.9, "ask": 2.1}]
    exit_call = [{"occ_symbol": "TEST", "bid": 2.9, "ask": 3.1}]
    econ = economics.position_economics(call, exit_call, rule, "ADVERSE")
    assert econ["pnl_usd"] < economics.position_economics(call, exit_call, rule, "BASE")["pnl_usd"]

"""Zero-network phase-C tests: no fabricated options alpha or future labels."""
import json
from datetime import date, datetime, timezone

import pandas as pd
import pytest
import yaml

from modules.autonomous_alpha import explorer, factor_forward, forward_options, reflection
from modules.autonomous_alpha.registry import EvidenceIntegrityError


def _write(root, path, content):
    path = root / path
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, (dict, list)):
        path.write_text(json.dumps(content) + "\n")
    else:
        path.write_text(content)
    return path


def _cfg(root):
    _write(root, forward_options.POLICY, """version: phase-c-v1
enabled: true
forward_start: "2026-10-29"
minimum_verified_trades: 26
minimum_distinct_entry_dates: 20
max_trade_records: 100000
max_quote_records_per_file: 50000
production_influence: NONE
""")


def _quote(root, date_iso, rid, symbol="XYZ270319C00100000"):
    p = root / forward_options.QUOTE_ROOT / (date_iso + ".jsonl")
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps({"record_id": rid, "occ_symbol":symbol,
                            "price_type":"OBSERVED_QUOTE",
                            "quality_status":"OK"})+"\n")


def _trade(**extra):
    row = {"schema":"trade-outcome-v1","trade_id":"T-1",
           "entry_date":"2026-11-04", "close_date":"2026-11-20",
           "entry_debit":3.1,"entry_quote_record_ids":["entry"],
           "exit_quote_record_ids":["exit"],"occ_symbols":["XYZ270319C00100000"],
           "price_type":"OBSERVED_QUOTE", "reliable":True,
           "outcome":0.12,"outcome_net":0.10,"strategy":"LONG_CALL",
           "ticker":"XYZ",
           "evaluation":{"status":"OK","tier":"REALIZED_RTH_REPRODUCIBLE",
                         "reproducible":True,"rth_executable":True,"flags":[]}}
    row.update(extra)
    return row


def test_options_real_quote_two_dates_and_no_alpha_attribution(tmp_path):
    _cfg(tmp_path)
    _quote(tmp_path,"2026-11-04","entry")
    _quote(tmp_path,"2026-11-20","exit")
    _write(tmp_path, forward_options.SOURCE, _trade())
    out=forward_options.evaluate(tmp_path,today=date(2026,11,21))
    assert out["n_verified_post_registration"]==1
    assert out["net_mean_verified"]==0.1
    assert out["prospective_alpha_validated"] is False
    assert out["strategy_alpha_status"]=="NOT_ATTRIBUTABLE"
    assert out["research_eligible"] is False
    assert out["production_influence"]=="NONE"


def test_pre_registered_old_trade_and_missing_entry_quote_not_evidence(tmp_path):
    _cfg(tmp_path)
    _quote(tmp_path,"2026-11-20","exit")
    row=_trade(entry_date="2026-10-15",
               entry_quote_record_ids=[],
               evaluation={**_trade()["evaluation"],"reproducible":False,"tier":"OBSERVED"})
    _write(tmp_path,forward_options.SOURCE,row)
    out=forward_options.evaluate(tmp_path,today=date(2026,11,21))
    assert out["n_verified_post_registration"]==0
    assert "pre_registration" in out["excluded_by_reason"]
    assert "entry_or_exit_quote_missing" in out["excluded_by_reason"]


def test_wrong_date_or_wrong_contract_quote_rejected(tmp_path):
    _cfg(tmp_path)
    _quote(tmp_path,"2026-11-04","entry",symbol="OTHER270319C00100000")
    _quote(tmp_path,"2026-11-20","exit")
    _write(tmp_path,forward_options.SOURCE,_trade())
    out=forward_options.evaluate(tmp_path,today=date(2026,11,21))
    assert out["n_verified_post_registration"]==0
    assert "quote_contract_mismatch" in out["excluded_by_reason"]


def test_options_unmatured_and_corrupt_records_never_succeed(tmp_path):
    _cfg(tmp_path)
    _write(tmp_path, forward_options.SOURCE, _trade())
    pending=forward_options.evaluate(tmp_path,today=date(2026,11,8))
    assert pending["n_verified_post_registration"]==0
    assert "invalid_or_unmatured_dates" in pending["excluded_by_reason"]
    (tmp_path / forward_options.SOURCE).write_text("{bad\n")
    with pytest.raises(EvidenceIntegrityError,match="malformed"):
        forward_options.evaluate(tmp_path,today=date(2026,11,21))


def _explore(root):
    config={"version":"explore-v1","enabled":True,
            "monthly_new_hypotheses":2,"lifetime_hypotheses":4,
            "max_variants_per_family":2,"exploratory_fraction":0.5,"seed":41,
            "parents":[{"name":"trend","feature":"mom_12_1","direction":1},
                       {"name":"reversal","feature":"rev_1m","direction":-1},
                       {"name":"lowvol","feature":"vol_60","direction":-1}]}
    _write(root,explorer.POLICY,yaml.safe_dump(config))
    explorer.register(root,datetime(2026,10,29,12,tzinfo=timezone.utc))


def _panel():
    # 60 PIT tickers with distinct signals and synthetic *future* labels.
    rows=[]
    for i in range(60):
        rows.append({"date":pd.Timestamp("2026-11-06"),"ticker":f"T{i:03}",
                     "mom_12_1":i/60.0,
                     "rev_1m":((i*11)%61)/61.0,
                     "vol_60":1+i/100,
                     "vix":25.0,
                     "fwd_xs_20":(i-30)/1000,
                     "label_end_20":pd.Timestamp("2026-12-04")})
    return pd.DataFrame(rows)


def test_stock_factor_prediction_is_frozen_until_labels_mature(tmp_path):
    _explore(tmp_path)
    p=_panel()
    before=factor_forward.process(tmp_path,panel=p,now=datetime(2026,11,7,tzinfo=timezone.utc))
    assert before["n_new_predictions"] > 0
    assert before["n_new_matured_labels"]==0
    early=factor_forward.process(tmp_path,panel=p,now=datetime(2026,11,20,tzinfo=timezone.utc))
    assert early["n_new_predictions"]==0
    assert early["n_new_matured_labels"]==0
    matured=factor_forward.process(tmp_path,panel=p,now=datetime(2026,12,10,tzinfo=timezone.utc))
    assert matured["n_new_matured_labels"]==before["n_new_predictions"]
    again=factor_forward.process(tmp_path,panel=p,now=datetime(2026,12,10,tzinfo=timezone.utc))
    assert again["n_new_matured_labels"]==0
    out=factor_forward._read(tmp_path / factor_forward.OUTCOMES)
    assert all(x["payload"]["net_options_pnl"] is None for x in out)
    assert all(x["payload"]["promotion_eligible"] is False for x in out)


def test_no_backfilled_predictions_before_registration(tmp_path):
    _explore(tmp_path)
    p=_panel()
    p["date"]=pd.Timestamp("2026-10-23")
    res=factor_forward.process(tmp_path,panel=p,now=datetime(2026,11,7,tzinfo=timezone.utc))
    assert res["n_new_predictions"]==0


def test_invalid_forward_source_fails_closed(tmp_path):
    _explore(tmp_path)
    pred_dir=tmp_path / factor_forward.PREDICTIONS
    _write(tmp_path,factor_forward.PREDICTIONS / "2026-11.jsonl",
           '{"sha256":"incorrect","payload":{"prediction_date":"2026-11-06"}}\n')
    with pytest.raises(EvidenceIntegrityError,match="content hash"):
        factor_forward.process(tmp_path,panel=_panel(),
                               now=datetime(2026,11,7,tzinfo=timezone.utc))


def test_reflection_never_authorizes_orders_or_llms(tmp_path):
    _cfg(tmp_path)
    _explore(tmp_path)
    _write(tmp_path,"outputs/research/hypothesis_db.json",
           {"hypotheses":{}})
    state=reflection.reflect(tmp_path,today=date(2026,11,15))
    assert state["alpha_validated"] is False
    assert state["promotion_authorized"] is False
    assert state["code_changes_authorized"] is False
    assert state["llm_calls"]==0
    assert state["production_influence"]=="NONE"
    assert state["exploration_slots_recommended"]==1


def test_existing_research_cron_disables_llm_and_retains_existing_schedules():
    from pathlib import Path
    workflow=(Path(__file__).resolve().parents[1]/".github/workflows/ml_research.yml").read_text()
    assert "cron: '41 7 3 * *'" in workflow
    assert "cron: '23 8 * * 6'" in workflow
    assert "python -m modules.autonomous_alpha factor-forward" in workflow
    assert "python -m modules.autonomous_alpha reflect" in workflow
    assert "python -m modules.hypothesis_factory plan --llm" not in workflow
    assert "ANTHROPIC_API_KEY:" not in workflow


def test_stale_pit_panel_cannot_backdate_a_new_prospective_signal(tmp_path):
    _explore(tmp_path)
    panel = _panel()  # last available observation 2026-11-06
    result = factor_forward.process(
        tmp_path, panel=panel, now=datetime(2026, 12, 10, tzinfo=timezone.utc))
    assert result["n_new_predictions"] == 0
    assert result["prospective_snapshot_status"] == "STALE_NO_NEW_PREDICTIONS"
    assert result["pit_snapshot_age_calendar_days"] > 4
    # Also make sure no synthetic forward records were written.
    assert factor_forward._read(tmp_path / factor_forward.PREDICTIONS) == []


def test_fresh_pit_signal_records_actual_registration_timestamp(tmp_path):
    _explore(tmp_path)
    timestamp = datetime(2026, 11, 7, 8, 23, tzinfo=timezone.utc)
    result = factor_forward.process(tmp_path, panel=_panel(), now=timestamp)
    assert result["prospective_snapshot_fresh"] is True
    snapshots = factor_forward._read(tmp_path / factor_forward.PREDICTIONS)
    assert snapshots
    assert all(x["payload"]["prediction_recorded_at"] == timestamp.isoformat()
               for x in snapshots)

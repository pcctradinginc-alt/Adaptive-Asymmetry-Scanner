"""FD3 Phase 4: real RTH quote ledger pairing, executable prices and source safety."""
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import hashlib
import json

import pytest

from modules.research_feature_option_signals import build_batches
from modules.research_feature_option_economics import trade_return, evaluate
from modules import research_feature_discovery as fd
from modules.autonomous_alpha import option_link as link

ROOT = Path(__file__).resolve().parents[1]


def _prepare_source(tmp_path, *, created="2026-11-14T10:00:00+00:00", n=35):
    raw = (ROOT / "config/research_feature_discovery_v3.yaml").read_bytes()
    proto, sha = fd.load_protocol()
    conf = tmp_path / "config/research_feature_discovery_v3.yaml"
    conf.parent.mkdir(parents=True)
    conf.write_bytes(raw)
    status = tmp_path / "outputs/research/feature_discovery_v3/status.json"
    status.parent.mkdir(parents=True)
    status.write_text(json.dumps({"protocol_sha256": sha}))
    pred = status.parent / "predictions.jsonl"
    with pred.open("w") as f:
        for h in proto["hypotheses"]:
            for i in range(n):
                f.write(json.dumps({
                    "protocol_sha256": sha, "hypothesis_id": h["id"],
                    "date": "2026-11-13", "ticker": f"S{i:03d}",
                    "frozen_at": created, "research_only": True,
                    "label": "FUTURE_UNOBSERVED",
                    "baseline_pred": i / n,
                    "challenger_pred": (n-i) / n,
                }) + "\n")
    return proto, sha


def _opts():
    return {"registration_start_utc": "2026-11-14T00:00:00+00:00",
            "max_hypotheses_per_signal_date": 2,
            "top_k": 1, "min_issuers": 30}


def test_fd3_pre_registered_source_provides_equal_baseline_challenger(tmp_path):
    _, sha = _prepare_source(tmp_path)
    stats = Counter()
    batches = build_batches(tmp_path, _opts(), date(2026, 11, 2),
                            timedelta(days=4),
                            datetime(2026, 11, 14, 12, tzinfo=timezone.utc), stats)
    assert len(batches) == 4
    assert len([b for b in batches if b["comparison_group"] == "REFERENCE"]) == 2
    assert len([b for b in batches if b["comparison_group"] == "TREATMENT"]) == 2
    for b in batches:
        assert b["hypothesis_version"] == sha
        assert b["hypothesis_source"] == "fd3_feature"
        assert len(b["top"]) == 1
        assert b["prediction_ref"]["n_universe"] == 35
        assert b["expected"] is None
    assert stats["fd3_paired_batches"] == 2


def test_retroactive_option_registration_is_rejected(tmp_path):
    _prepare_source(tmp_path, created="2026-11-13T10:00:00+00:00")
    stats = Counter()
    assert not build_batches(tmp_path, _opts(), date(2026, 11, 2),
                             timedelta(days=4), datetime(2026, 11, 14, 12,
                                                        tzinfo=timezone.utc), stats)
    assert stats["fd3_before_registration"] > 0


def test_source_protocol_mutation_fails_closed(tmp_path):
    _prepare_source(tmp_path)
    path = tmp_path / "outputs/research/feature_discovery_v3/status.json"
    path.write_text(json.dumps({"protocol_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="mismatches"):
        build_batches(tmp_path, _opts(), date(2026, 11, 2), timedelta(days=4),
                      datetime(2026, 11, 14, 12, tzinfo=timezone.utc), Counter())


def _trade(sid, did, ticker, ask, bid_exit, *, structure="LONG_CALL",
           verified=True):
    symbol = ticker + "261218C00100000"
    entered = "2026-11-16T16:01:00+00:00"
    entry_leg = {"occ_symbol": symbol, "side": "long", "bid": ask-.1, "ask": ask,
                 "usable_rth": True, "quote_record_id": f"ENT-{did}",
                 "quote_time": "2026-11-16T16:00:00+00:00"}
    d = {"decision_id": did, "signal_id": sid, "ticker": ticker, "structure": structure,
         "decision": "TRADE", "decision_day": "2026-11-16",
         "decided_at": entered, "net_debit_ask": ask,
         "evidence_class_at_entry": "QUOTE_VERIFIED_PAPER", "legs": [entry_leg]}
    o = {"decision_id": did, "signal_id": sid, "ticker": ticker, "structure": structure,
         "recorded_at": "2026-12-16T00:00:00+00:00", "exit_day": "2026-12-15",
         "status": "EXIT_OBSERVED",
         "evidence_class": "QUOTE_VERIFIED_PAPER" if verified else "UNVERIFIABLE",
         "exit_legs": [{"occ_symbol": symbol, "bid": bid_exit, "ask": bid_exit+.1,
                        "quote_time": "2026-12-15T16:00:00+00:00",
                        "quote_record_id": f"EXIT-{did}"}]}
    return d, o


def test_paper_economics_uses_ask_bid_fee_and_adverse_execution():
    d, o = _trade("S", "D", "AAA", 2.10, 3.00)
    result = trade_return(d, o, fee=.65, slippage_fraction=.25)
    assert result is not None
    assert result["net_roi"] == pytest.approx((300 - .65 - 210 - .65) / (210 + .65))
    assert result["adverse_net_roi"] < result["net_roi"]
    assert result["net_pnl_usd"] == pytest.approx(88.70)
    assert result["evidence"][0]["entry_record_id"] == "ENT-D"


def test_quote_missing_is_never_replaced_by_zero():
    d, o = _trade("S", "D", "AAA", 2.10, 3.00, verified=False)
    assert trade_return(d, o, fee=.65) is None
    o["evidence_class"] = "QUOTE_VERIFIED_PAPER"
    o["exit_legs"][0]["quote_record_id"] = None
    assert trade_return(d, o, fee=.65) is None
    o["exit_legs"][0]["quote_record_id"] = "EXIT-D"
    o["exit_legs"][0]["ask"] = 2.00  # crossed bid/ask
    assert trade_return(d, o, fee=.65) is None


def test_fd3_option_policies_are_paired_and_still_never_confirmed():
    proto, digest = fd.load_protocol()
    hid = proto["hypotheses"][0]["id"]
    entries = []
    decs, outs = [], []
    for arm, ticker, entry, exit_bid in (
        ("BASE", "AAA", 2.10, 2.10),
        ("CHALLENGER", "BBB", 2.10, 3.00),
    ):
        sid, did = f"SID-{arm}", f"DID-{arm}"
        entries.append({
            "signal_id": sid, "hypothesis_id": f"FD3:{hid}:{arm}",
            "hypothesis_source": "fd3_feature",
            "hypothesis_version": digest,
            "production_influence": "NONE", "order_placed": False,
            "role": "TOP", "pit_data_date": "2026-11-13",
            "registered_at": "2026-11-14T12:00:00+00:00",
            "hypothesis_registered_at": "2026-11-14T00:00:00+00:00",
            "prediction_ref": {"fd3_arm": arm, "fd3_hypothesis": hid},
        })
        d, o = _trade(sid, did, ticker, entry, exit_bid)
        decs.append(d)
        outs.append(o)
    data = {"signals": entries, "decisions": decs, "outcomes": outs}
    cfg = link.config()
    rep = evaluate(data, cfg, today=date(2026, 12, 20), protocol_hash=digest)
    ss = rep["hypotheses"][hid]["structures"]["LONG_CALL"]
    assert ss["both_arms_quote_verified_dates"] == 1
    assert ss["independent_quote_verified_windows"] == 1
    assert ss["mean_net_roi_delta"] > 0
    assert ss["status"] == "NEED_MORE_PAIRED_RTH_QUOTES"
    assert rep["option_edge_confirmed"] is False
    assert rep["global_alpha_ledger_slot_allocated"] is False
    assert rep["orders_created"] == 0


def test_future_recorded_outcome_not_allowed_in_asof_replay():
    proto, digest = fd.load_protocol()
    hid = proto["hypotheses"][0]["id"]
    s, ds, os = [], [], []
    for arm in ("BASE", "CHALLENGER"):
        sid, did = f"SID-{arm}", f"DID-{arm}"
        s.append({"signal_id": sid, "hypothesis_id": f"FD3:{hid}:{arm}",
                  "hypothesis_source": "fd3_feature", "hypothesis_version": digest,
                  "role": "TOP", "pit_data_date": "2026-11-13",
                  "prediction_ref": {"fd3_arm": arm, "fd3_hypothesis": hid},
                  "registered_at": "2026-11-14T12:00:00+00:00",
                  "hypothesis_registered_at": "2026-11-14T00:00:00+00:00",
                  "production_influence": "NONE", "order_placed": False})
        d, o = _trade(sid, did, "ABC", 2.10, 3.00)
        ds.append(d); os.append(o)
    rep = evaluate({"signals": s, "decisions": ds, "outcomes": os},
                   link.config(), today=date(2026, 11, 30), protocol_hash=digest)
    assert rep["hypotheses"][hid]["structures"]["LONG_CALL"]["independent_quote_verified_windows"] == 0


def test_frozen_source_not_imported_by_production_scanner():
    assert "research_feature_option_economics" not in (ROOT / "pipeline.py").read_text()
    assert link.config()["sources"]["fd3_shadow"]["research_only"] is True

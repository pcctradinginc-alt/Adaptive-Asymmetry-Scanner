"""Contract tests: broker-free account bookings, idempotence and honest quote gaps.

All dates/quotes are synthetic. No test writes to the real repository outputs.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from modules.autonomous_alpha import paper_account as pa
from modules.autonomous_alpha import option_economics as oe


START = datetime(2026, 11, 9, 14, 40, tzinfo=timezone.utc)
ENTRY = datetime(2026, 11, 9, 14, 50, tzinfo=timezone.utc)
NOW = datetime(2026, 11, 9, 14, 51, tzinfo=timezone.utc)
LATER = datetime(2026, 12, 9, 18, 0, tzinfo=timezone.utc)


@pytest.fixture
def account(tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    (tmp_path / pa.CONFIG).write_text(
        Path("config/paper_account.yaml").read_text(encoding="utf-8")
        .replace("approved_quote_sources: [tradier]", "approved_quote_sources: [SYNTHETIC_TEST_ONLY]"), encoding="utf-8")
    alpha_cfg = {"rule": {"contracts_per_position": 1, "commission_usd_per_contract_per_side": 0.65,
                          "adverse_slippage_fraction_of_spread": 0.25}}
    sig = {"signal_id": "ALS-1", "hypothesis_id": "AUTO-one",
           "hypothesis_version": "frozen-001", "model_id": "model-before-entry",
           "experiment_version": "experiment-001",
           "comparison_group": "TREATMENT", "tradable": True,
           "registered_at": "2026-11-09T14:42:00+00:00"}
    def decision(name, structure, cost=101.0, risk=101.65):
        return {"decision_id": name, "signal_id": sig["signal_id"], "hypothesis_id": sig["hypothesis_id"],
                "hypothesis_version": sig["hypothesis_version"], "comparison_group": "TREATMENT",
                "structure": structure, "decision": "TRADE", "ticker": "XYZ",
                "decided_at": ENTRY.isoformat(), "decision_day": "2026-11-09",
                "rule_hash": pa.al.rule_hash(alpha_cfg),
                "evidence_class_at_entry": "QUOTE_VERIFIED_PAPER",
                "capital_usd": cost, "max_loss_usd": risk,
                "legs": [{"occ_symbol": "XYZ270116C00100000",
                          "side": "long", "quote_record_id": "quote-entry-1"}]}
    data = {"signals": [sig],
            "decisions": [decision("ALD-1", "LONG_CALL"), decision("ALD-2", "BULL_CALL_SPREAD")],
            "outcomes": [], "integrity": {}}
    monkeypatch.setattr(pa.al, "config", lambda root: alpha_cfg)
    monkeypatch.setattr(pa.al, "read_all", lambda root: data)
    monkeypatch.setattr(pa.al, "_check_rule", lambda cfg, decisions: None)
    monkeypatch.setattr(pa, "_entry_archived", lambda root, d: True)
    return tmp_path, data, sig, decision


def activate(root):
    result = pa.sync(root, now=START, capture=True, persist_report=False)
    assert result["note"] == "ACTIVATED_NO_HISTORICAL_BACKFILL"
    assert result["starting_cash_usd"] == 100000
    assert len(pa.events(root)) == 1


def test_activation_never_backfills_existing_research_positions(account):
    root, data, _, _ = account
    activate(root)
    assert not any(e["type"] == "OPEN" for e in pa.events(root))


def test_live_entry_uses_only_one_structure_per_signal(account):
    root, _, _, _ = account
    activate(root)
    out = pa.sync(root, now=NOW, capture=True, persist_report=False)
    assert out["open_trades"] == 1
    assert out["cash_usd"] == 99899
    assert out["nav_at_cost_usd"] == 100000
    assert out["nav_market_usd"] is None
    opens = [e for e in pa.events(root) if e["type"] == "OPEN"]
    assert len(opens) == 1
    assert opens[0]["structure"] == "LONG_CALL"
    assert opens[0]["model_id"] == "model-before-entry"
    assert opens[0]["hypothesis_version"] == "frozen-001"
    assert opens[0]["broker_order"] is False


def test_duplicate_run_does_not_double_book(account):
    root, _, _, _ = account
    activate(root)
    first = pa.sync(root, now=NOW, capture=True, persist_report=False)
    second = pa.sync(root, now=NOW, capture=True, persist_report=False)
    assert first["new_open_or_skip"] == 1
    assert second["new_open_or_skip"] == 0
    assert second["cash_usd"] == first["cash_usd"]
    assert second["open_trades"] == 1


def test_weekly_never_opens_old_decisions(account):
    root, _, _, _ = account
    activate(root)
    result = pa.sync(root, now=LATER, capture=False, persist_report=False)
    assert result["open_trades"] == result["closed_trades"] == 0


def test_historical_decision_after_activation_never_enters_late(account):
    root, _, _, _ = account
    activate(root)
    old = pa.sync(root, now=LATER, capture=True, persist_report=False)
    assert old["open_trades"] == 0
    assert old["new_open_or_skip"] == 0


def test_capital_limit_excludes_both_structures(account):
    root, data, _, _ = account
    for d in data["decisions"]:
        d["capital_usd"], d["max_loss_usd"] = 3000.0, 3001.3
    activate(root)
    out = pa.sync(root, now=NOW, capture=True, persist_report=False)
    assert out["open_trades"] == 0 and out["account_skips"] == 1
    skip = next(e for e in pa.events(root) if e["type"] == "SKIP")
    assert "LONG_CALL:MAX_POSITION_RISK" in skip["reason"]


def test_control_signals_never_book_in_main_account(account):
    root, _, sig, _ = account
    sig["comparison_group"] = "CONTROL"
    activate(root)
    out = pa.sync(root, now=NOW, capture=True, persist_report=False)
    assert out["open_trades"] == 0
    assert out["account_skips"] == 0


def test_missing_entry_quote_means_skip_not_fictional_fill(account, monkeypatch):
    root, _, _, _ = account
    activate(root)
    monkeypatch.setattr(pa, "_entry_archived", lambda root, d: False)
    result = pa.sync(root, now=NOW, capture=True, persist_report=False)
    assert result["open_trades"] == 0
    assert "ENTRY_QUOTE_NOT_VERIFIED" in str(pa.events(root))


def test_nonexistent_quote_archive_never_counts_as_verified(account, monkeypatch):
    root, data, _, _ = account
    monkeypatch.undo()
    assert pa._entry_archived(root, data["decisions"][0]) is False


def test_journal_event_id_is_idempotent_and_conflicts_fail(account):
    root, _, _, _ = account
    activate(root)
    event = pa.events(root)[0]
    assert pa._save(root, event) is False
    modified = dict(event)
    modified["starting_cash_usd"] = 999999
    with pytest.raises(ValueError, match="conflicting immutable"):
        pa._save(root, modified)


def test_altered_hash_fails_closed(account):
    root, _, _, _ = account
    activate(root)
    (root / pa.JOURNAL / "PA-INIT.json").write_text('{"sha256":"bad","payload":{}}')
    with pytest.raises(ValueError, match="integrity"):
        pa.events(root)


def test_policy_cannot_silently_change_after_activation(account):
    root, _, _, _ = account
    activate(root)
    p = root / pa.CONFIG
    p.write_text(p.read_text().replace("starting_cash_usd: 100000", "starting_cash_usd: 120000"))
    with pytest.raises(ValueError, match="reviewed migration"):
        pa.sync(root, now=NOW, capture=True, persist_report=False)


def test_unverified_exit_remains_open(account, monkeypatch):
    root, data, _, _ = account
    activate(root)
    pa.sync(root, now=NOW, capture=True, persist_report=False)
    data["outcomes"] = [{"decision_id": "ALD-1", "status": "EXIT_MISSING",
                         "outcome_id": "ALO-1", "evidence_class": "UNVERIFIABLE"}]
    out = pa.sync(root, now=LATER, persist_report=False)
    assert out["open_trades"] == 1 and out["closed_trades"] == 0
    assert out["nav_market_status"] == "INCOMPLETE_MISSING_QUOTES"


def test_verified_exit_books_only_once_and_balances(account, monkeypatch):
    root, data, _, _ = account
    activate(root)
    pa.sync(root, now=NOW, capture=True, persist_report=False)
    data["outcomes"] = [{
        "decision_id": "ALD-1", "outcome_id": "ALO-1", "status": "EXIT_OBSERVED",
        "recorded_at": LATER.isoformat(), "exit_day": "2026-12-09",
        "evidence_class": "QUOTE_VERIFIED_PAPER",
        "exit_legs": [{"occ_symbol": "XYZ270116C00100000", "quote_record_id": "quote-exit-1"}],
    }]
    monkeypatch.setattr(pa.oe, "_archive", lambda *args: {
        "quote-entry-1": {"source": "SYNTHETIC_TEST_ONLY"},
        "quote-exit-1": {"source": "SYNTHETIC_TEST_ONLY"},
    })
    monkeypatch.setattr(pa.oe, "_verify", lambda *args: ["MISSING_EXIT_QUOTE"])
    first = pa.sync(root, now=LATER, persist_report=False)
    assert first["open_trades"] == 1 and first["closed_trades"] == 0
    monkeypatch.setattr(pa.oe, "_verify", lambda *args: [])
    monkeypatch.setattr(pa.oe, "position_economics",
        lambda entry, exit, rule, scenario: {
            "capital_usd": 101.0, "exit_value_usd": 120.0 if scenario == "BASE" else 114.0,
            "pnl_usd": 19.0 if scenario == "BASE" else 13.0})
    settled = pa.sync(root, now=LATER, persist_report=False)
    assert settled["closed_trades"] == 1 and settled["open_trades"] == 0
    assert settled["cash_usd"] == 100019
    assert settled["realized_pnl_usd"] == 19
    assert settled["net_pnl_adverse_usd"] == 13
    again = pa.sync(root, now=LATER, persist_report=False)
    assert again["new_close"] == 0 and again["cash_usd"] == 100019


def test_replay_rejects_two_different_bookings_for_one_signal(account):
    root, _, _, _ = account
    activate(root)
    pa.sync(root, now=NOW, capture=True, persist_report=False)
    original = next(e for e in pa.events(root) if e["type"] == "OPEN")
    duplicate = {**original, "event_id": pa._event_id("OPEN", "ALD-new"),
                 "decision_id": "ALD-new"}
    pa._save(root, duplicate)
    with pytest.raises(ValueError, match="double account decision"):
        pa._replay(pa.events(root), pa.policy(root))


def test_no_broker_or_real_order_configuration_is_accepted(account):
    root, _, _, _ = account
    p = root / pa.CONFIG
    p.write_text(p.read_text().replace("broker_connection: NONE", "broker_connection: TRADIER"))
    with pytest.raises(ValueError, match="broker-free"):
        pa.policy(root)


def test_tradier_is_read_only_market_data_source_no_tradier_paper_account(account):
    root, _, _, _ = account
    cfg = pa.policy(root)
    assert cfg["market_data_only"] is True
    assert cfg["broker_connection"] == "NONE"
    assert cfg["real_orders"] is False
    assert pa.policy(Path(".")).get("approved_quote_sources") == ["tradier"]
    assert pa.policy(Path(".")).get("enabled") is True


def test_market_data_only_guard_refuses_order_mode(account):
    root, _, _, _ = account
    p = root / pa.CONFIG
    p.write_text(p.read_text().replace("market_data_only: true", "market_data_only: false"))
    with pytest.raises(ValueError, match="read-only market data"):
        pa.policy(root)


def test_existing_bid_ask_economics_is_used_not_theoretical_prices():
    rule = {"contracts_per_position": 1, "commission_usd_per_contract_per_side": 0.65,
            "adverse_slippage_fraction_of_spread": 0.25}
    long = [{"occ_symbol": "L", "side": "long", "bid": 1.9, "ask": 2.1}]
    short = {"occ_symbol": "S", "side": "short", "bid": 0.8, "ask": 1.0}
    exit_long = {"occ_symbol": "L", "bid": 2.9, "ask": 3.1}
    exit_short = {"occ_symbol": "S", "bid": 0.3, "ask": 0.5}
    lc = oe.position_economics(long, [exit_long], rule, "BASE")
    sp = oe.position_economics(long + [short], [exit_long, exit_short], rule, "BASE")
    assert lc["pnl_usd"] == 78.7
    assert sp["pnl_usd"] == 107.4
    adverse = oe.position_economics(long + [short], [exit_long, exit_short], rule, "ADVERSE")
    assert adverse["pnl_usd"] < sp["pnl_usd"]

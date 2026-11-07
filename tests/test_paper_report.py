"""Synthetic weekly reporting tests: no SMTP traffic or secrets on CI."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from modules.autonomous_alpha import paper_report as pr


def test_weekly_report_is_explicit_about_missing_nav_and_alpha():
    account = {"status": "ACTIVE", "starting_cash_usd": 100000,
               "cash_usd": 99750, "open_risk_usd": 260,
               "open_trades": 1, "closed_trades": 0, "nav_market_usd": None,
               "nav_at_cost_usd": 100000, "nav_market_status": "INCOMPLETE_MISSING_QUOTES"}
    report = pr.render(account, {"hypotheses": {}, "status": "NEED_MORE_DATA"},
                       {"status": "NEED_MORE_DATA", "learning": {"missing_data": {"NO_QUOTE": 9}}},
                       {}, {}, week="2026-W45")
    assert "100,000.00 USD" in report
    assert "INCOMPLETE_MISSING_QUOTES" in report
    assert "nicht bewertbar" in report
    assert "NO_QUOTE: 9" in report
    assert "Ein verbesserter Depotwert allein beweist keinen Lernfortschritt" in report


def test_send_fails_closed_when_email_secrets_unavailable(tmp_path, monkeypatch):
    for key in ("GMAIL_SENDER", "GMAIL_APP_PW", "NOTIFY_EMAIL"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ValueError, match="missing configured Gmail"):
        pr.send(tmp_path, now=datetime(2026, 11, 7, tzinfo=timezone.utc))


def test_successful_weekly_receipt_deduplicates_send(tmp_path, monkeypatch):
    sent = []
    class FakeSMTP:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def login(self, user, pw):
            assert user == "sender@example.com" and pw == "test-only"
        def send_message(self, message):
            sent.append(message)
    monkeypatch.setenv("GMAIL_SENDER", "sender@example.com")
    monkeypatch.setenv("GMAIL_APP_PW", "test-only")
    monkeypatch.setenv("NOTIFY_EMAIL", "owner@example.com")
    monkeypatch.setattr(pr.smtplib, "SMTP_SSL", FakeSMTP)
    now = datetime(2026, 11, 7, tzinfo=timezone.utc)
    first = pr.send(tmp_path, now=now)
    assert first["status"] == "SENT" and len(sent) == 1
    second = pr.send(tmp_path, now=now)
    assert second["status"] == "ALREADY_SENT" and len(sent) == 1
    record = json.loads((tmp_path / pr.ROOT / "mail" / (first["week"] + ".json")).read_text())
    assert record["payload"]["status"] == "SENT"
    assert "test-only" not in json.dumps(record)


def test_e2e_owner_digest_carries_verified_feedback_and_actual_defects():
    audit = {"status": "TECHNICAL_DEFECT", "n_technical_defects": 1,
             "paper_account": {"activation_verified_from_events": True},
             "source_chain": {"verified_settled_economics": 3},
             "existing_learning": {"validation_plans": 2,
                                   "verdicts_reconciled_to_memory": 1,
                                   "forward_confirmed_and_memorized": 0,
                                   "alpha_improvement_proven": False},
             "issues": [{"severity": "TECHNICAL_DEFECT",
                         "id": "FORWARD_VERDICT_NOT_IN_RESEARCH_MEMORY",
                         "observed_fact": "FROZEN-12 is missing"}],
             "next_action": "Synchronize the existing research memory"}
    msg = pr.render({}, {}, {}, {}, {}, week="2026-W45", e2e_audit=audit)
    assert "End-to-End-Betriebsprüfung" in msg
    assert "TECHNICAL_DEFECT" in msg
    assert "1 / 2" in msg
    assert "NEIN / NOCH NICHT NACHGEWIESEN" in msg
    assert "FROZEN-12 is missing" in msg
    assert "Keine Doppel-Labels" in msg

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

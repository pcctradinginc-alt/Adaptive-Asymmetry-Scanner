"""Tests für modules/mailer.py – SMTP komplett gemockt, keine echten Mails."""
from __future__ import annotations

import logging
import smtplib

import pytest

from modules import mailer

PW = "geheimes-passwort-xyz"
SENDER = "absender@example.org"
RCPT = "empfaenger@example.net"
ENV = {"GMAIL_SENDER": SENDER, "GMAIL_APP_PW": PW, "NOTIFY_EMAIL": RCPT}


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(mailer, "_sleep", lambda s: None)


class FakeSMTP:
    """Ersetzt smtplib.SMTP_SSL; failures = Anzahl anfänglicher Fehlversuche."""
    calls = 0
    failures = 0
    sent = []

    def __init__(self, *a, **k):
        type(self).calls += 1

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def login(self, user, pw):
        if type(self).calls <= type(self).failures:
            raise smtplib.SMTPAuthenticationError(535, f"bad login {pw}".encode())

    def sendmail(self, frm, to, body):
        type(self).sent.append((frm, to, body))


@pytest.fixture
def fake(monkeypatch):
    FakeSMTP.calls, FakeSMTP.failures, FakeSMTP.sent = 0, 0, []
    monkeypatch.setattr(mailer.smtplib, "SMTP_SSL", FakeSMTP)
    return FakeSMTP


def test_not_configured(caplog):
    with caplog.at_level(logging.WARNING):
        r = mailer.send_mail("s", "<b>h</b>", "t", env={})
    assert r["status"] == "not_configured" and r["attempts"] == 0
    assert "nicht konfiguriert" in caplog.text


def test_dry_run_sends_nothing(fake):
    r = mailer.send_mail("Betreff", "<b>h</b>", "text", dry_run=True, env=ENV)
    assert r["status"] == "dry_run" and fake.calls == 0
    assert r["preview"]["subject"] == "Betreff"


def test_retry_then_success(fake):
    fake.failures = 2
    r = mailer.send_mail("s", "<b>h</b>", "t", env=ENV, max_retries=3)
    assert r["status"] == "sent" and r["attempts"] == 3
    assert len(fake.sent) == 1


def test_all_fail_is_bounded(fake):
    fake.failures = 999
    r = mailer.send_mail("s", "<b>h</b>", "t", env=ENV, max_retries=3)
    assert r["status"] == "failed" and r["attempts"] == 3 and fake.calls == 3
    assert PW not in (r["error"] or "")


def test_no_secrets_in_logs(fake, caplog):
    fake.failures = 999
    with caplog.at_level(logging.DEBUG):
        r = mailer.send_mail("s", "<b>h</b>", "t", env=ENV)
        mailer.send_mail("s", "<b>h</b>", "t", env=ENV, dry_run=True)
    assert PW not in caplog.text
    assert RCPT not in caplog.text and SENDER not in caplog.text
    assert "e***@example.net" in caplog.text
    assert PW not in str(r)


def test_multipart_has_text_and_html(fake):
    mailer.send_mail("s", "<b>HTML</b>", "PLAIN", env=ENV)
    body = fake.sent[0][2]
    assert "text/plain" in body and "text/html" in body
    msg = mailer.build_message("s", "<b>HTML</b>", "PLAIN", SENDER, [RCPT])
    assert [p.get_content_type() for p in msg.get_payload()] == ["text/plain", "text/html"]


def test_multiple_recipients_and_fallback(fake):
    env = {**ENV, "MAIL_TO": "a@x.org, b@y.org"}
    mailer.send_mail("s", "h", "t", env=env)
    assert fake.sent[0][1] == ["a@x.org", "b@y.org"]


def test_unknown_provider_no_crash():
    r = mailer.send_mail("s", "h", "t", env={**ENV, "MAIL_PROVIDER": "nope"})
    assert r["status"] == "not_configured"


def test_mask_address():
    assert mailer.mask_address("alice@example.com") == "a***@example.com"
    assert mailer.mask_address("kaputt") == "***"

"""Audit P1-8: tägliche Mail (email_reporter) – Inhalt, Filter, Versandpfad, ohne SMTP."""
from __future__ import annotations

import pytest

from modules import email_reporter as er
from modules.reporter import compute_exit_rules


@pytest.fixture
def outbox(monkeypatch):
    sent = []
    monkeypatch.setattr(er, "_send_smtp", lambda subject, html: sent.append((subject, html)))
    monkeypatch.setattr(er, "_external_context_html", lambda: "<!--ext-->")
    return sent


def _proposal(score=70, strategy="LONG_CALL", hr=0.62):
    p = {"ticker": "ABC", "strategy": strategy, "trade_score": {"total": score, "grade": "B",
                                                                  "best_argument_for": "x" * 900,
                                                                  "best_argument_against": "gegen"},
            "deep_analysis": {"catalyst_confidence": 7, "time_to_materialization": "4-8 Wochen"},
            "simulation": {"hit_rate": hr}, "mc_hit_rate": hr,
            "option": {"strike": 105, "expiry": "2026-12-18", "dte": 77, "bid": 4.0, "ask": 4.2,
                       "spread_leg": {"strike": 115, "bid": 1.0, "ask": 1.1}},
            "roi_analysis": {"delta": 0.55, "theta_daily_pct": 0.02, "vega_loss": 0.05, "breakeven": 109.2,
                             "breakeven_pct": 0.04},
            "implied_move_pct": 8.0, "model_move_pct": 14.0, "edge_vs_implied": 6.0}
    if strategy == "BULL_CALL_SPREAD":
        p["option"]["net_debit"] = 3.1
    p["exit_rules"] = compute_exit_rules(p)                       # echte Exit-Regeln der Produktion
    return p


def test_trade_mail_only_above_score_threshold(outbox):
    er.send_email([_proposal(score=10)], "2026-10-02", {"universe": 500, "vix": 16.4})
    subj, html = outbox[-1]
    assert "Kein Trade" in subj and "500 Ticker im Universum" in html
    er.send_email([_proposal(score=90)], "2026-10-02", {})
    subj, html = outbox[-1]
    assert "Trade Empfehlung" in subj and "ABC" in html


def test_trade_mail_marks_mc_hit_rate_as_uncalibrated(outbox):
    er.send_email([_proposal(score=90)], "2026-10-02", {})
    html = outbox[-1][1]
    assert "NICHT kalibriert" in html and "keine Gewinnwahrscheinlichkeit" in html


def test_trade_mail_truncates_arguments_and_shows_greeks(outbox):
    er.send_email([_proposal(score=90, strategy="BULL_CALL_SPREAD")], "2026-10-02", {})
    html = outbox[-1][1]
    assert "x" * 700 in html and "x" * 701 not in html
    assert "Delta:" in html and "Breakeven" in html and "Market-Implied" in html


def test_status_mail_funnel_and_engine_warnings(outbox):
    er.send_status_email({"trades": 0, "universe": 405, "candidates": 405, "prescreened": 129, "analyzed": 0,
                          "vix": 16.44}, "2026-10-01", {"warnings": ["Dürre-Streak: 5 Tage"]})
    subj, html = outbox[-1]
    assert "Kein Trade" in subj and "VIX 16.44" in html and "Dürre-Streak" in html and "Engine-Status: WARN" in html
    er.send_status_email({"trades": 1, "vix": None}, "2026-10-01")
    subj, html = outbox[-1]
    assert "Trade Empfehlung" in subj and "VIX –" in html and "Engine-Status" not in html


def test_exit_alert_mail(outbox):
    er.send_exit_alert_email([{"reason": "stop_loss", "ticker": "ABC", "strategy": "LONG_CALL", "outcome": -0.5,
                               "age_days": 12, "option": {"strike": 100, "expiry": "2026-12-18"}},
                              {"reason": "take_profit", "ticker": "XYZ", "outcome": 0.9, "age_days": 30}],
                             "2026-10-02")
    subj, html = outbox[-1]
    assert "1× TP, 1× SL" in subj and "STOP-LOSS" in html and "-50.0%" in html


def test_rl_arming_mail_escapes_prompt(outbox):
    er.send_rl_arming_email(120, 100, 0.41, "2026-04-01")
    subj, html = outbox[-1]
    assert "120 Trades" in subj and "41%" in html and "&gt;90%" in html


def test_smtp_not_called_without_credentials(monkeypatch):
    monkeypatch.delenv("GMAIL_SENDER", raising=False)
    monkeypatch.delenv("GMAIL_APP_PW", raising=False)
    import smtplib

    def boom(*a, **k):
        raise AssertionError("SMTP darf ohne Zugangsdaten nicht verbunden werden")
    monkeypatch.setattr(smtplib, "SMTP_SSL", boom)
    er._send_smtp("s", "<p>x</p>")                              # still, kein Fehler


def test_smtp_errors_are_logged_not_raised(monkeypatch):
    monkeypatch.setenv("GMAIL_SENDER", "a@example.org")
    monkeypatch.setenv("GMAIL_APP_PW", "x")
    import smtplib

    def boom(*a, **k):
        raise OSError("network")
    monkeypatch.setattr(smtplib, "SMTP_SSL", boom)
    er._send_smtp("s", "<p>x</p>")

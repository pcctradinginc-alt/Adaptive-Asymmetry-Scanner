"""Härtungsdurchgang: Gate-Audit-Evidenz, Kalibrierung in Berichten, kanonischer Safe Mode."""
from __future__ import annotations

import json

import pytest


def test_shadow_stats_split_by_reject_reason():
    from monthly_report import shadow_stats
    h = {"shadow_trades": [
        {"reject_reason": "roi_gate", "outcome": -0.5, "close_date": "2026-08-17"},
        {"reject_reason": "roi_gate", "outcome": 0.2, "close_date": "2026-08-17"},
        {"reject_reason": "final_mc_survivor", "outcome": 0.1, "close_date": "2026-08-17"},
        {"reject_reason": "score_48", "outcome": -0.1, "close_date": "2026-08-17"},
        {"reject_reason": "roi_gate", "outcome": None, "close_date": "2026-08-17"},
        {"reject_reason": "roi_gate", "outcome": 0.9, "close_date": "2026-07-17"},
    ]}
    s = shadow_stats(h, "2026-08")
    assert s["n"] == 4 and set(s["by_reason"]) == {"roi_gate", "final_mc_survivor", "score_gate"}
    assert s["by_reason"]["roi_gate"]["n"] == 2 and s["by_reason"]["roi_gate"]["mean"] == pytest.approx(-0.15)


def test_shadow_html_never_mixes_gates_and_respects_small_n():
    from monthly_report import build_html
    shadow = {"n": 3, "win_rate": 1.0, "wins": 3, "mean": 0.5,
              "by_reason": {"roi_gate": {"n": 3, "win_rate": 1.0, "wins": 3, "mean": 0.5}}}
    cur = {"n": 5, "win_rate": 0.2, "wins": 1, "mean": -0.1, "median": -0.1, "total_losses": 0}
    html = build_html("2026-08", cur, None, None, {"days": 0}, [], None, shadow, None, None)
    assert "roi_gate" in html and "n<10 – keine Aussage" in html and "Gates arbeiten korrekt" not in html


def test_truncated_shadow_trades_are_archived_not_lost(tmp_path, monkeypatch):
    import feedback
    monkeypatch.setattr(feedback, "SHADOW_ARCHIVE", tmp_path / "arch.jsonl")
    monkeypatch.setattr(feedback, "get_current_price", lambda t: 0.0)
    h = {"shadow_trades": [{"ticker": f"T{i}", "entry_date": "2026-10-01", "outcome": i} for i in range(305)]}
    feedback.evaluate_shadow_trades(h, feedback.datetime(2026, 10, 3))
    assert len(h["shadow_trades"]) == 300 and h["shadow_trades"][0]["ticker"] == "T5"
    rows = [json.loads(x) for x in (tmp_path / "arch.jsonl").read_text().splitlines()]
    assert [r["ticker"] for r in rows] == [f"T{i}" for i in range(5)]


def test_daily_email_shows_measured_calibration_not_raw_probability(tmp_path, monkeypatch):
    from modules import email_reporter as er
    p = tmp_path / "ppa.json"
    p.write_text(json.dumps({"mc_hit_rate_calibration": {"0.65-0.75": {"n": 27, "win_rate": 0.222}}}))
    monkeypatch.setattr(er, "CALIBRATION_PATH", p)
    assert er._calibrated_line(0.71).startswith("22% (Band 0.65-0.75, n=27")
    monkeypatch.setattr(er, "CALIBRATION_PATH", tmp_path / "missing.json")
    assert er._calibrated_line(0.71) == "n/a"                 # fehlend -> nie geraten


def test_pipeline_logs_safe_mode_only_from_canonical_state():
    src = open("pipeline.py").read()
    assert 'stats["data_health"]["data_safe_mode"]' in src
    assert 'if stats["system_state"].get("active")' in src
    assert 'if _dh.get("safe_mode")' not in src


def test_cost_section_labels_measured_vs_estimated():
    from monthly_report import build_cost_html
    html = build_cost_html("2026-10", {"telemetry_calls": 0, "complete": False})
    for lbl in ("[MEASURED: API-usage × Listenpreis]", "[ESTIMATED: Planpreis anteilig]",
                "[ESTIMATED vs. ohne Cache]", "[MEASURED: Requests]"):
        assert lbl in html

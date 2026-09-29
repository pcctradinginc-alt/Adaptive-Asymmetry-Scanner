"""Tests für reports/weekly.py – synthetische Eingaben in tmp_path, keine Netzwerkzugriffe."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from reports import weekly

REPO = Path(__file__).resolve().parent.parent
TODAY = date(2026, 9, 29)


def _w(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj), encoding="utf-8")


def _trade(tk, entry, close, out, approx=False):
    t = {"ticker": tk, "entry_date": entry, "close_date": close, "outcome": out}
    if approx:
        t["outcome_method_reconstructed"] = "delta_approx"
    return t


def make_full(root: Path, weight=0.12, coverage=0.65, status="rejected"):
    rs = root / "outputs" / "research"
    _w(rs / "meta_learning.json", {
        "generated": "2026-09-28T10:00:00+00:00", "meta_version": "v1", "primary_meta": "meta_ridge",
        "reference": "static_eq", "current_regime": {"name": "risk_on", "confidence": "LOW"},
        "drift": {"model_drift": {"flag": False}, "feature_drift": {"flag": True}},
        "disagreement": {"current_level": "HIGH"},
        "decision": {"verdict": "not_proven", "reasons": ["Delta-CI schließt 0 ein"]},
        "approaches": {"meta_ridge": {"metrics": {"sharpe": 0.9, "cagr": 0.08, "brier": 0.21}},
                       "static_eq": {"metrics": {"sharpe": 0.7, "cagr": 0.06, "brier": 0.22}}},
        "deltas": {"sharpe": {"delta": 0.2, "ci_low": -0.1, "ci_high": 0.5}},
        "calibration_buckets": {"meta_ridge": [
            {"bucket": "0.6-0.7", "n": 40, "win_rate": 0.5, "avg_return": 0.01, "expected_return": 0.03,
             "calibration_error": -0.1, "flag": "overconfident"},
            {"bucket": "0.7-0.8", "n": 3, "win_rate": 0.6, "avg_return": 0.02, "expected_return": 0.03,
             "calibration_error": 0.0, "flag": "low_n"}]},
        "model_intelligence": {"mA": {"oos_ic": 0.03, "recent_ic": 0.05, "trend": "improving", "trend_t": 2.1,
                                      "calibration": "ok", "contribution": 0.4, "meta_weight": weight}},
    })
    _w(rs / "meta_state.json", {"safe_mode": True, "active_ensemble": "static", "reasons": ["drift"], "updated": "x"})
    _w(rs / "hc_candidates.json", {"date": "2026-09-29", "enabled": True, "disabled_reason": None, "rule": {},
                                   "candidates": [{"ticker": "ZZZ", "confidence": "HIGH", "calibrated_probability": 0.71,
                                                   "expected_return_20d": 0.06, "expected_downside": -0.03,
                                                   "asymmetry_ratio": 2.0, "model_agreement": 0.8,
                                                   "regime_compatibility": 0.9}]})
    _w(rs / "ml_research.json", {
        "generated": "2026-09-27T00:00:00+00:00", "mode": "full", "champion": None,
        "models": {"mA": {"role": "challenger", "wf": {"base": {"mean": 0.01, "t_months": 1.5, "sharpe_ann": 0.5, "max_dd": -0.2},
                                                         "ic": {"mean_ic": 0.02, "t_months": 1.0}},
                          "decision": {"verdict": status, "reasons": []}}},
        "calibration": {"coverage": coverage, "target": 0.8, "interval_calibrated": False, "brier": 0.23,
                        "brier_base": 0.22, "p_up_skill": -0.05}})
    _w(rs / "ml_forward.json", {"forward": {"mA": {"base": {"n_months": 3, "mean": 0.01, "t_months": 0.5}, "ic": {"mean_ic": 0.01}}}})
    _w(rs / "hypothesis_db.json", {"n_tested_total": 4, "discovery": {"n_tested": 10, "n_survivors": 0, "window": "w"},
                                   "hypotheses": {
                                       "H1": {"id": "H1", "title": "A", "status": "rejected", "created_at": "2026-09-01"},
                                       "H2": {"id": "H2", "title": "B", "status": "accepted", "created_at": "2026-09-20"},
                                       "H3": {"id": "H3", "title": "C", "status": "passed_pending_locked", "created_at": "2026-09-10"},
                                       "H4": {"id": "H4", "title": "D", "status": "testing", "created_at": "2026-09-25"}}})
    _w(rs / "failure_analysis.json", {"reliable": {"primary": {"signal_wrong": {"n": 2, "share": 1.0}}}})
    (rs / "trade_memory.jsonl").write_text(
        json.dumps({"ticker": "LOS", "entry_date": "2026-09-01", "failure": {"primary": "signal_wrong"}, "what_went_wrong": ["signal_wrong"]}) + "\n{kaputt\n",
        encoding="utf-8")
    _w(root / "outputs" / "history.json", {
        "active_trades": [{"ticker": "OPN", "entry_date": "2026-09-20"}],
        "closed_trades": [_trade("WIN", "2026-09-01", "2026-09-15", 1.0),
                          _trade("LOS", "2026-09-01", "2026-09-20", -0.5),
                          _trade("LO2", "2026-05-01", "2026-06-20", -0.5, approx=True),
                          _trade("WI2", "2026-04-01", "2026-05-20", 2.0)]})
    _w(root / "outputs" / "external_data" / "health" / "source_health.json", {
        "a": {"status": "PASS", "staleness": "FRESH", "last_success": "2026-09-28T00:00:00+00:00"},
        "b": {"status": "FAIL", "staleness": "STALE", "last_success": "2026-09-01T00:00:00+00:00"},
        "c": {"status": "WARN"}, "d": {"status": "DEFERRED"}})
    (root / "outputs" / "candidate_ledger").mkdir(parents=True, exist_ok=True)
    (root / "outputs" / "candidate_ledger" / "2026-09.jsonl").write_text(
        json.dumps({"status": "rejected"}) + "\n" + json.dumps({"status": "open"}) + "\n", encoding="utf-8")


def text_for(root, state=None):
    return weekly.render_text(weekly.collect(root, TODAY, state_path=state))


def test_full_data_all_sections(tmp_path):
    make_full(tmp_path)
    t = text_for(tmp_path)
    for n, title in weekly.SECTION_TITLES.items():
        assert f"{n}. {title}" in t
    assert "↑ improving" in t and "2.10" in t          # Pfeil nur aus trend, trend_t sichtbar
    assert "ZZZ" in t and "Erster Bericht" in t
    assert "signal_wrong" in t                          # Verlierer-Ursache aus trade_memory
    assert "Delta [CI]" in t and "meta_ridge (meta)" in t
    for code in ("DATA PIPELINE FAILURE", "DATA DRIFT", "CALIBRATION FAILURE", "REGIME UNCERTAINTY",
                 "EXCESSIVE MODEL DISAGREEMENT", "SAFE MODE", "LOW SAMPLE SIZE"):
        assert code in t
    assert "MODEL DRIFT" not in t                       # nur wenn Daten es zeigen
    assert "accepted: 1" in t and "inconclusive: 1" in t and "currently testing: 1" in t


def test_missing_files_no_crash(tmp_path):
    t = text_for(tmp_path)
    assert "keine Daten" in t and "n/a" in t
    assert weekly.NO_HC_TEXT in t
    assert "Erster Bericht – keine Vergleichsbasis." in t
    assert weekly.DISCLAIMER in t
    weekly.render_html(weekly.collect(tmp_path, TODAY))
    weekly.render_md(weekly.collect(tmp_path, TODAY))


def test_empty_candidates_exact_text(tmp_path):
    _w(tmp_path / "outputs/research/hc_candidates.json",
       {"enabled": True, "candidates": []})
    assert weekly.NO_HC_TEXT in text_for(tmp_path)
    _w(tmp_path / "outputs/research/hc_candidates.json",
       {"enabled": False, "disabled_reason": "weil", "candidates": [{"ticker": "X"}]})
    t = text_for(tmp_path)
    assert weekly.NO_HC_TEXT in t and "weil" in t and "X " not in t.split("6.")[1].split("7.")[0]


def test_forward_metrics_arithmetic():
    m = weekly.forward_metrics([1.0, -0.5, -0.5, 2.0])
    assert m["closed"] == 4 and m["low_sample"]
    assert m["win_rate"] == pytest.approx(0.5)
    assert m["avg_return"] == pytest.approx(0.5) == pytest.approx(m["expectancy"])
    assert m["median"] == pytest.approx(0.25)
    assert m["avg_winner"] == pytest.approx(1.5) and m["avg_loser"] == pytest.approx(-0.5)
    assert m["payoff_ratio"] == pytest.approx(3.0)
    assert m["profit_factor"] == pytest.approx(3.0)
    assert m["max_dd"] == pytest.approx(-1.0)            # cum 1, .5, 0, 2 ; Peak 1 -> -1.0
    assert weekly.forward_metrics([])["closed"] == 0
    assert weekly.forward_metrics([0.5, 0.5])["profit_factor"] is None   # keine Verluste


def test_forward_windows_and_reliable_split(tmp_path):
    make_full(tmp_path)
    fw = weekly.collect(tmp_path, TODAY)["forward"]
    since = fw["windows"][0]
    assert since["all"]["closed"] == 4 and since["reliable"]["closed"] == 3   # delta_approx raus
    w4 = fw["windows"][3]
    assert w4["all"]["closed"] == 2                       # WIN (09-15) + LOS (09-20)
    assert since["signals"] == 5                          # 4 closed + 1 aktiv
    t = weekly.render_text(weekly.collect(tmp_path, TODAY))
    assert "Paper-Trades, echt – kein Backtest" in t and "BACKTEST" in t and "FORWARD-SHADOW" in t
    assert "LOW SAMPLE" in t


def test_learning_diff_with_snapshot(tmp_path):
    make_full(tmp_path, weight=0.12, coverage=0.65, status="rejected")
    state = tmp_path / "state.json"
    snap = weekly.collect(tmp_path, TODAY)["snapshot"]
    assert weekly.save_state(state, snap)
    make_full(tmp_path, weight=0.30, coverage=0.71, status="promising")
    hyp = json.loads((tmp_path / "outputs/research/hypothesis_db.json").read_text())
    hyp["hypotheses"]["H1"]["status"] = "accepted"
    _w(tmp_path / "outputs/research/hypothesis_db.json", hyp)
    t = text_for(tmp_path, state)
    assert "Meta-Gewicht Modell mA 0.12->0.3" in t
    assert "Kalibrierung coverage 0.65->0.71" in t
    assert "Hypothese H1: rejected -> accepted" in t
    assert "Modell mA: Verdikt rejected -> promising" in t
    assert "Erster Bericht" not in t


def test_dry_run_writes_no_snapshot_send_does(tmp_path, monkeypatch):
    root, out = tmp_path / "root", tmp_path / "out"
    make_full(root)
    assert weekly.main(["--dry-run", "--root", str(root), "--out-dir", str(out), "--date", "2026-09-29"]) == 0
    assert (out / "weekly_2026-09-29.html").exists() and (out / "weekly_2026-09-29.txt").exists()
    assert (out / "weekly_2026-09-29.md").exists()
    assert not (out / "weekly_state.json").exists()

    calls = []
    import modules.mailer as mailer
    monkeypatch.setattr(mailer, "send_mail", lambda s, h, t, **k: calls.append(s) or {"status": "sent", "attempts": 1})
    assert weekly.main(["--send", "--root", str(root), "--out-dir", str(out), "--date", "2026-09-29"]) == 0
    assert calls == ["Adaptive Asymmetry Scanner – Weekly Intelligence Report – 2026-09-29"]
    assert (out / "weekly_state.json").exists()


def test_send_failed_keeps_state_and_rc1(tmp_path, monkeypatch):
    make_full(tmp_path / "r")
    import modules.mailer as mailer
    monkeypatch.setattr(mailer, "send_mail", lambda *a, **k: {"status": "failed", "attempts": 3})
    assert weekly.main(["--send", "--root", str(tmp_path / "r"), "--out-dir", str(tmp_path / "o")]) == 1
    assert not (tmp_path / "o" / "weekly_state.json").exists()


def test_subject_format():
    assert weekly.subject_for("2026-09-29") == "Adaptive Asymmetry Scanner – Weekly Intelligence Report – 2026-09-29"


def test_html_escapes(tmp_path):
    _w(tmp_path / "outputs/research/hc_candidates.json",
       {"enabled": True, "candidates": [{"ticker": "<script>"}]})
    h = weekly.render_html(weekly.collect(tmp_path, TODAY))
    assert "<script>" not in h and "&lt;script&gt;" in h


def test_real_repo_dry_run_does_not_crash(tmp_path):
    rc = weekly.main(["--dry-run", "--root", str(REPO), "--out-dir", str(tmp_path), "--date", "2026-09-29"])
    assert rc == 0
    assert (tmp_path / "weekly_2026-09-29.txt").exists()

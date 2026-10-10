"""Maintenance 2026-10-09: WATCH/Paper-Trade-Semantik, Duplikate, Alt-Data-Timeout-Safety, kanonische
Version, Universe-Zählung, Shadow-Score-Gleichstände."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _p(ticker, score, grade):
    return {"ticker": ticker, "strategy": "LONG_CALL", "trade_score": {"total": score, "grade": grade},
            "option": {"ask": 2.0, "bid": 1.9, "strike": 100, "expiry": "2027-01-15"}, "simulation": {}, "features": {}}


def test_watch_above_trade_gate_is_booked_per_policy():
    import pipeline
    from modules.booking_labels import booking_text
    h = {"active_trades": []}
    n = pipeline.book_proposals([_p("SNPS", 59, "WATCH 🟠")], h, "2026-10-06", 55.0)
    assert n == 1 and len(h["active_trades"]) == 1
    txt = booking_text({"trade_score": {"grade": "WATCH 🟠"}, "booking": {"status": "BOOKED", "trade_score_min": 55.0}})
    assert "neu gebucht" in txt and "Label WATCH" in txt and "≥ 55" in txt and "gemäß Policy" in txt


def test_watch_below_gate_never_reaches_booking():
    """Score < trade_score_min wird in der Pipeline vorher zum Schatten-Trade (score_<n>), nie gebucht."""
    src = (ROOT / "pipeline.py").read_text()
    assert "if score >= trade_score_min:" in src and '_shadow.append((p, f"score_{score}"))' in src
    cfg = (ROOT / "config.yaml").read_text()
    assert "trade_score_min: 55" in cfg


def test_open_position_not_booked_again_and_labelled():
    import pipeline
    from modules.booking_labels import booking_text
    h = {"active_trades": []}
    pipeline.book_proposals([_p("SNPS", 59, "WATCH 🟠")], h, "2026-10-06", 55.0)
    nxt = [_p("SNPS", 61, "BUY 🟡")]
    n = pipeline.book_proposals(nxt, h, "2026-10-07", 55.0)
    assert n == 0 and len(h["active_trades"]) == 1                                  # kein Duplikat
    assert nxt[0]["booking"]["status"] == "NOT_BOOKED" and nxt[0]["booking"]["open_since"] == "2026-10-06"
    assert "NICHT neu gebucht" in booking_text(nxt[0]) and "kein neuer, unabhängiger Trade" in booking_text(nxt[0])
    again = [_p("SNPS", 61, "BUY 🟡")]
    pipeline.book_proposals(again, h, "2026-10-06", 55.0)
    assert again[0]["booking"]["status"] == "ALREADY_BOOKED_TODAY" and len(h["active_trades"]) == 1


def test_daily_report_shows_booking_and_canonical_version(tmp_path):
    from modules.reporter import Reporter, compute_exit_rules
    from modules.version import APP_VERSION
    p = _p("SNPS", 61, "BUY 🟡")
    p.update(deep_analysis={}, exit_rules=compute_exit_rules(p), booking={"status": "NOT_BOOKED", "open_since": "2026-10-06"})
    Reporter(tmp_path).save("2026-10-07", [p], {"model_weights": {}})
    md = (tmp_path / "2026-10-07.md").read_text()
    assert "NICHT neu gebucht" in md and f"Scanner {APP_VERSION}" in md and "v8.2" not in md
    assert "Gleichstände möglich" in md                                              # Shadow-Score-Ties dokumentiert


def test_mail_uses_canonical_version_and_labels_universe(monkeypatch):
    from modules import email_reporter as er
    from modules.version import APP_VERSION
    sent = []
    monkeypatch.setattr(er, "_send_smtp", lambda s, h: sent.append(h))
    monkeypatch.setattr(er, "_external_context_html", lambda: "")
    er.send_email([], "2026-10-08", {"universe": 514, "candidates": 418, "vix": 15.7})
    html = sent[-1]
    assert APP_VERSION in html and "v8.2" not in html
    assert "514 Ticker im Universum (Rohliste vor Filtern)" in html and "418 nach Hard-Filter" in html
    for f in ("modules/email_reporter.py", "modules/reporter.py"):
        src = (ROOT / f).read_text()
        assert "· v8.3<" not in src and "Scanner v8.2_" not in src and "Scanner v8.3 &nbsp;" not in src


def test_ingestion_sets_raw_universe_size():
    src = (ROOT / "modules/data_ingestion.py").read_text()
    assert "self.universe_size = len(tickers)" in src
    pipe = (ROOT / "pipeline.py").read_text()
    assert 'getattr(ingestion, "universe_size", None)' in pipe and 'stats["candidates"] = len(candidates)' in pipe


def test_shadow_score_ties_have_no_production_effect():
    """FinalScore (QuasiML) ist grob quantisiert (3 Features × 3 Bins); Trade-Ranking nutzt trade_score."""
    src = (ROOT / "modules/quasi_ml.py").read_text()
    assert "das Trade-Ranking nutzt trade_score" in src
    from modules.quasi_ml import QuasiML
    q = QuasiML({"model_weights": {"impact": 0.35, "mismatch": 0.45, "eps_drift": 0.2}, "feature_stats": {}})
    a = q._compute_final_score({"features": {"bin_impact": "mid", "bin_mismatch": "good", "bin_eps_drift": "noise", "mismatch": 2.287}})
    b = q._compute_final_score({"features": {"bin_impact": "mid", "bin_mismatch": "good", "bin_eps_drift": "noise", "mismatch": 1.068}})
    assert a == b                                                                     # echte Gleichheit, keine Rundung


def test_entity_build_budget_stops_cleanly_and_marks_partial(tmp_path):
    from modules.entity_resolution.build import build
    from modules.entity_resolution.store import EntityStore
    t = {"now": 0.0}

    def clock():
        t["now"] += 10
        return t["now"]
    rows = [{"ticker": f"T{i}", "cik": str(1000 + i), "title": f"Co {i}"} for i in range(50)]
    rep = build([r["ticker"] for r in rows], EntityStore(tmp_path / "m.jsonl"), "2026-10-09",
                fetch_tickers=lambda: rows, fetch_submissions=lambda cik: {"name": "X", "formerNames": []},
                fetch_gleif=lambda n: [], sleep=lambda s: None, deadline=100.0, clock=clock)
    assert rep["status"] == "PARTIAL_BUDGET_EXHAUSTED" and rep["stopped_in"] == "sec_submissions"
    assert rep["sec"]["submissions_skipped"] > 0


def test_entity_build_stops_source_after_error_series(tmp_path):
    from modules.entity_resolution.build import MAX_CONSECUTIVE_ERRORS, build
    from modules.entity_resolution.store import EntityStore
    rows = [{"ticker": f"T{i}", "cik": str(1000 + i), "title": f"Co {i}"} for i in range(80)]
    calls = []

    def boom(cik):
        calls.append(cik)
        raise ConnectionError("429")
    rep = build([r["ticker"] for r in rows], EntityStore(tmp_path / "m.jsonl"), "2026-10-09",
                fetch_tickers=lambda: rows, fetch_submissions=boom, fetch_gleif=lambda n: [], sleep=lambda s: None)
    assert rep["status"] == "PARTIAL_SOURCE_ERRORS" and len(calls) == MAX_CONSECUTIVE_ERRORS


def test_entity_store_survives_truncated_last_line(tmp_path):
    from modules.entity_resolution.store import EntityStore
    p = tmp_path / "m.jsonl"
    p.write_text('{"op": "open", "record": {"entity_id"')                            # Abbruch mitten in der Zeile
    s = EntityStore(p)
    assert s.records == []


def test_alt_data_workflow_timeout_safety():
    import yaml
    wf = yaml.safe_load((ROOT / ".github/workflows/alt_data.yml").read_text())
    steps = wf["jobs"]["ingest"]["steps"]
    ent = next(s for s in steps if s.get("id") == "entity")
    assert ent["timeout-minutes"] <= 90 and "--max-minutes" in ent["run"] and ent.get("continue-on-error") is True
    last = steps[-1]
    assert last.get("if") == "always()" and "ci_stage_check" in last["run"] and "steps.entity.outcome" in last["env"]["DEGRADED"]

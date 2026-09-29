"""Regression Lauf 2026-09-28: 28/153 Ledger-Zeilen ohne Ablehnungsgrund
('unlabeled_after_*') und Prescreen-Absagen ohne Basismerkmale."""
from __future__ import annotations

import json

import pipeline
from modules import candidate_ledger as cl


def _fresh():
    cl.start_run("2026-09-28")
    pipeline.reject_stats.clear()


def test_label_dropped_marks_only_unlabeled_dropouts(tmp_path, monkeypatch):
    _fresh()
    monkeypatch.setattr(cl, "_fetch_prices_batch", lambda t: {})
    monkeypatch.setattr(cl.market_snapshot, "fetch_underlying_quotes", lambda t: {})
    for t in ("AAA", "BBB", "CCC"):
        cl.note(t, stage="deep_analysis")
    pipeline.reject("bearish_disabled", "BBB")          # hat schon einen Grund
    n = pipeline.label_dropped([{"ticker": "AAA"}, {"ticker": "BBB"}, {"ticker": "CCC"}],
                               [{"ticker": "CCC"}], "mismatch_below_min")
    assert n == 1
    assert pipeline.reject_stats["mismatch_below_min"]["tickers"] == ["AAA"]
    cl.flush(tmp_path)
    rows = {r["ticker"]: r for r in map(json.loads, (tmp_path / "2026-09.jsonl").read_text().splitlines())}
    assert rows["AAA"]["reject_reason"] == "mismatch_below_min"
    assert rows["BBB"]["reject_reason"] == "bearish_disabled"
    assert not any(str(r["reject_reason"]).startswith("unlabeled") for r in rows.values()
                   if r["ticker"] != "CCC")


def test_base_features_recorded_for_every_candidate(tmp_path, monkeypatch):
    _fresh()
    monkeypatch.setattr(cl, "_fetch_prices_batch", lambda t: {})
    monkeypatch.setattr(cl.market_snapshot, "fetch_underlying_quotes", lambda t: {})
    c = {"ticker": "ZZZ", "rel_volume": 0.8, "market_cap": 5e10, "news": ["a", "b"],
         "info": {"sector": "Industrials", "industry": "Railroads"},
         "features": {"sentiment_score": 0.4, "sentiment_confidence": 0.9, "short_pct_float": 0.05}}
    pipeline.note_base_features(c)
    pipeline.reject("prescreen_no", "ZZZ")
    cl.flush(tmp_path)
    row = json.loads((tmp_path / "2026-09.jsonl").read_text().splitlines()[0])
    f = row["features"]
    assert f["sector"] == "Industrials" and f["rel_volume"] == 0.8 and f["news_count"] == 2
    assert f["log_market_cap"] == 10.699 and f["sentiment_score"] == 0.4
    assert row["reject_reason"] == "prescreen_no"

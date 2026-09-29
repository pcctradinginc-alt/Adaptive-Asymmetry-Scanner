"""Regression Testlauf 2026-09-29: Job-Timeout ohne jede Ausgabe. Das
Laufzeitbudget stoppt teure Stufen rechtzeitig und labelt die Übersprungenen."""
from __future__ import annotations

import time

import pipeline
from modules.deep_analysis import DeepAnalysis


def test_deadline_computed_from_config():
    d = pipeline._set_run_deadline(start=1000.0)
    total = float(pipeline.cfg.pipeline.max_runtime_minutes)
    reserve = float(pipeline.cfg.pipeline.finalize_reserve_minutes)
    assert d == 1000.0 + (total - reserve) * 60


def test_over_budget_flag():
    pipeline._RUN_DEADLINE[:] = [time.monotonic() - 1]
    assert pipeline.over_budget()
    pipeline._RUN_DEADLINE[:] = [time.monotonic() + 3600]
    assert not pipeline.over_budget()
    pipeline._RUN_DEADLINE[:] = []


def test_deep_analysis_stops_at_deadline_and_reports_skipped():
    da = DeepAnalysis.__new__(DeepAnalysis)
    calls = []
    da._analyze = lambda c: calls.append(c["ticker"]) or None
    shortlist = [{"ticker": t} for t in ("A", "B", "C")]
    assert da.run(shortlist, deadline=time.monotonic() - 1) == []
    assert calls == [] and da.skipped_for_time == ["A", "B", "C"]
    da.run(shortlist, deadline=time.monotonic() + 60)
    assert calls == ["A", "B", "C"] and da.skipped_for_time == []

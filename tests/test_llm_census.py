"""LLM-Zensus (Aktionsplan 2026-11-09, Punkt 6): Zufallsstichprobe ohne Gates, π protokolliert, nie im Trade-Pfad.

Geprüft: Auswahl reproduzierbar, gleichverteilt (empirisch), π = k/N; Ergebnisse nur als census_*-Felder (nie das
Feld direction, das den Snapshot- und Score-Pfad steuert); nicht analysierte Ziehungen werden mit Grund notiert;
der Bericht wertet nur analysierte Zeilen und die richtungsbereinigte Rendite aus; ohne API-Schlüssel kein Lauf.
"""
from __future__ import annotations

from collections import Counter

import pytest

from modules import llm_census as lc


def _cands(n):
    return [{"ticker": f"T{i:03d}", "news": ["x"]} for i in range(n)]


def test_selection_is_reproducible_uniform_and_logs_pi():
    picks, design = lc.select(_cands(50), "2026-11-10", 2)
    assert [p["ticker"] for p in picks] == [p["ticker"] for p in lc.select(_cands(50), "2026-11-10", 2)[0]]
    assert design == {"policy": lc.POLICY, "day": "2026-11-10", "n": 50, "k": 2, "pi": pytest.approx(0.04)}
    counts = Counter(p["ticker"] for d in range(2000) for p in lc.select(_cands(20), f"day{d}", 2)[0])
    assert max(abs(v / 2000 - 0.1) for v in counts.values()) < 0.03            # P(gezogen) = k/N
    assert lc.select([], "d", 2) == ([], {"policy": lc.POLICY, "day": "d", "n": 0, "k": 0, "pi": 0.0})
    c = _cands(3)
    picks, _ = lc.select(c, "d", 5)
    picks[0]["ticker"] = "CHANGED"
    assert all(x["ticker"] != "CHANGED" for x in c)                            # tiefe Kopie, Produktion unberührt


class _Analyzer:
    def __init__(self, skip=()):
        self.skipped_for_time = list(skip)

    def run(self, picks, deadline=None):
        return [{**p, "deep_analysis": {"impact": 7, "surprise": 5, "direction": "BULLISH", "bear_case_severity": 3}}
                for p in picks if p["ticker"] not in self.skipped_for_time]


class _Scorer:
    def __init__(self):
        self.shadow = {}

    def _score(self, a):
        self.shadow[a["ticker"]] = {"mismatch_raw": -1.0, "mismatch_dir_48h": 6.0, "mismatch_dir_news": 5.5}
        # Produktion hätte gefiltert (kein Ergebnis)


def test_results_are_census_prefixed_and_never_touch_direction():
    picks, design = lc.select(_cands(10), "2026-11-10", 2)
    noted = []
    res = lc.run(picks, design, None, analyzer=_Analyzer(skip=[picks[1]["ticker"]]), scorer_cls=_Scorer,
                 note=lambda t, **kw: noted.append((t, kw)))
    assert res["analyzed"] == 1 and res["not_analyzed"] == 1
    (_, kw0), (_, kw1) = noted
    assert "direction" not in kw0 and kw0["census_direction"] == "BULLISH" and kw0["census_impact"] == 7
    assert kw0["census_mismatch"] == -1.0 and kw0["census_mismatch_dir_news"] == 5.5
    assert kw0["llm_census"]["status"] == "analyzed" and kw0["llm_census"]["pi"] == pytest.approx(0.2)
    assert kw1 == {"llm_census": {**design, "status": "skipped_time"}}
    assert lc.run([], design, None) == {"status": "empty", **design}


def test_report_uses_only_analyzed_rows_and_direction_adjusted_returns():
    rows = []
    for i in range(12):
        bear = i % 2 == 1
        rows.append({"date": f"2026-11-{10 + i:02d}", "features": {
            "llm_census": {"status": "analyzed"}, "census_direction": "BEARISH" if bear else "BULLISH",
            "census_impact": i, "census_mismatch": i / 2},
            "outcomes": {"ret_20d": (-0.01 * i) if bear else (0.01 * i)}})     # Thesenrichtung trifft, stärker mit i
    rows.append({"date": "2026-11-30", "features": {"llm_census": {"status": "skipped_time"}},
                 "outcomes": {"ret_20d": 9.9}})
    rep = lc.report(rows)
    h = rep["horizons"]["20d"]
    assert rep["analyzed_rows"] == 12 and h["n"] == 12
    assert h["ic_census_impact"] == pytest.approx(1.0) and h["mean_dir_return"] == pytest.approx(0.055)
    assert rep["horizons"]["45d"]["n"] == 0 and rep["production_influence"] == "NONE"


def test_settings_and_deadline_keep_a_reserve(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("k: 3\nunknown: 1\n")
    s = lc.settings(p)
    assert s["k"] == 3 and "unknown" not in s and s["enabled"] is True
    import time
    now = time.monotonic()
    d = lc.deadline_for(now, 600, {"budget_s": 240, "min_reserve_s": 120})
    assert d <= now + 240 + 1 and d <= now + 600 - 120 + 1

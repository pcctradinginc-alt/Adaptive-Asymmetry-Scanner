"""Audit 2026-11-26: Provider-Ausfall in Stufe 1, Guard-Regel, Ledger-Schreibstatus, Offline-LLM-Zensus."""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

from modules import data_ingestion as di
from modules import llm_census as lc
from modules import llm_census_offline as lco
from modules.workflow_health import artifact_ok

# echte Zahlen der Hard-Filter-Läufe (Scanner-Logs)
NOV25 = {"total": 518, "no_data": 56, "market_cap": 462, "avg_volume": 0, "rel_volume": 0, "passed": 0}
NOV23 = {"total": 518, "no_data": 0, "market_cap": 0, "avg_volume": 70, "rel_volume": 190, "passed": 258}


def test_outage_detection_on_real_counts():
    out = di.hard_filter_outage(NOV25, {"ok": False, "with_market_cap": 0})
    assert out["kind"] == "provider_outage" and out["share_suspect"] == 1.0
    assert di.hard_filter_outage(NOV23, None) is None
    # Kontrollabfrage ok -> viele 'zu kleine' Werte sind ein Marktbefund, kein Ausfall
    assert di.hard_filter_outage(NOV25, {"ok": True, "with_market_cap": 3}) is None
    # >= 50 % ganz ohne Daten ist immer ein Ausfall
    assert di.hard_filter_outage({"total": 10, "no_data": 5, "passed": 0}, None)["reason"] == "no_data"
    assert di.hard_filter_outage({"total": 0}, None) is None


def test_run_probes_only_when_suspicious(monkeypatch):
    calls = []
    monkeypatch.setattr(di, "get_universe", lambda: [f"T{i}" for i in range(10)])
    monkeypatch.setattr(di.DataIngestion, "_get_current_vix", lambda self: 20.0)
    monkeypatch.setattr(di.DataIngestion, "_provider_probe", staticmethod(lambda: calls.append(1) or {"ok": False}))

    class _Yf:
        class Ticker:
            def __init__(self, t):
                self.fast_info = type("F", (), {"last_price": 1.0})()
    monkeypatch.setattr(di, "yf", _Yf)

    def small(self, ticker, _s, vix):          # info ohne marketCap -> V1 zählt 'Market Cap < 2 Mrd.'
        return None, {"no_data": 0, "market_cap": 1, "avg_volume": 0, "dollar_volume": 0,
                      "rel_volume": 0, "no_news": 0, "passed": 0}
    monkeypatch.setattr(di.DataIngestion, "_evaluate_ticker", small)
    ing = di.DataIngestion()
    assert ing.run() == [] and calls == [1]
    assert ing.provider_outage["market_cap_reject"] == 10 and ing.filter_stats["market_cap"] == 10

    def normal(self, ticker, _s, vix):
        return None, {"no_data": 0, "market_cap": 0, "avg_volume": 0, "dollar_volume": 0,
                      "rel_volume": 1, "no_news": 0, "passed": 0}
    monkeypatch.setattr(di.DataIngestion, "_evaluate_ticker", normal)
    calls.clear()
    ing = di.DataIngestion()
    ing.run()
    assert calls == [] and ing.provider_outage is None


def test_artifact_unless_field(tmp_path):
    spec = {"kind": "file", "path": "r/{date}.json", "unless_field": "stats.data_failure"}
    day = date(2026, 11, 25)
    assert artifact_ok(spec, day, tmp_path) is False                       # fehlt
    (tmp_path / "r").mkdir()
    f = tmp_path / "r" / "2026-11-25.json"
    f.write_text(json.dumps({"stats": {"data_failure": {"kind": "provider_outage"}}}))
    assert artifact_ok(spec, day, tmp_path) is False                       # Datenausfall = nicht versorgt
    f.write_text(json.dumps({"stats": {"stop_reason": "Keine Kandidaten nach Hard-Filter."}}))
    assert artifact_ok(spec, day, tmp_path) is True
    f.write_text("{kaputt")
    assert artifact_ok(spec, day, tmp_path) is False
    assert artifact_ok({"kind": "file", "path": "r/{date}.json"}, day, tmp_path) is True   # ohne Regel wie bisher


def test_ledger_flush_reports_status(tmp_path, monkeypatch):
    from modules import candidate_ledger as cl
    monkeypatch.setattr(cl, "_state", {"date": None, "entries": {}, "flushed": False})
    assert cl.flush(tmp_path) == {"status": "empty", "rows": 0}
    assert cl.flush(tmp_path) == {"status": "already_flushed"}
    monkeypatch.setattr(cl, "_state", {"date": "2026-11-26", "entries": {"AAA": [{}]}, "flushed": False})
    monkeypatch.setattr(cl, "_compute_event_id", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    res = cl.flush(tmp_path)
    assert res["status"] == "failed" and "disk full" in res["error"]


def _frozen(root, day, frozen_at):
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{day}.json").write_text(json.dumps({"schema": "offline_llm_census_v1", "frozen_at": frozen_at,
                                                  "design": {"day": day, "n": 10, "k": 1}, "candidates": []}))


def test_offline_census_picks_fresh_cohorts_and_marks_stale(tmp_path):
    inp, out = tmp_path / "in", tmp_path / "out"
    now = datetime(2026, 11, 26, 4, 0, tzinfo=timezone.utc)
    _frozen(inp, "2026-11-25", (now - timedelta(hours=10)).isoformat())     # Cron am Folgetag (real 4–7 h später)
    _frozen(inp, "2026-11-19", (now - timedelta(days=7)).isoformat())
    _frozen(inp, "2026-11-24", (now - timedelta(hours=30)).isoformat())
    (out).mkdir()
    (out / "2026-11-24.json").write_text("{}")                                 # schon ausgewertet
    fresh, stale = lco.pending_cohorts(inp, out, now)
    assert fresh == ["2026-11-25"] and stale == ["2026-11-19"]


def test_offline_census_main_writes_stale_once_without_llm(tmp_path, monkeypatch):
    inp, out = tmp_path / "in", tmp_path / "out"
    _frozen(inp, "2026-11-13", "2026-11-13T18:00:00+00:00")
    monkeypatch.setattr(lco, "_run_day", lambda *a: (_ for _ in ()).throw(AssertionError("kein LLM für alte Kohorte")))
    assert lco.main(["--input", str(inp), "--out", str(out)]) == 0
    rep = json.loads((out / "2026-11-13.json").read_text())
    assert rep["status"] == "SKIPPED" and rep["reason"] == "stale_cohort" and rep["rows"] == []
    assert lco.main(["--input", str(inp), "--out", str(out)]) == 0            # append-only, kein zweites Mal


class _Scorer:
    def __init__(self):
        self.shadow = {}

    def _score(self, a):
        return {"features": {"mismatch": 1.0}}


class _VetoAnalyzer:
    """Wie DeepAnalysis.run: ohne keep_vetoed würde das VETO verworfen."""
    def run(self, shortlist, deadline=None, keep_vetoed=False):
        out = []
        for c in shortlist:
            da = {"impact": 5, "surprise": 4, "direction": "BULLISH", "red_team": {"red_team_verdict": "VETO"}}
            if keep_vetoed:
                out.append({**c, "deep_analysis": da})
        return out


def test_census_keeps_vetoed_analyses():
    notes = []
    res = lc.run([{"ticker": "KMB"}], {"day": "2026-11-10", "k": 1, "n": 2}, None, analyzer=_VetoAnalyzer(),
                 scorer_cls=_Scorer, note=lambda t, **f: notes.append((t, f)))
    assert res["analyzed"] == 1
    assert notes[0][1]["census_red_team_verdict"] == "VETO" and notes[0][1]["llm_census"]["status"] == "analyzed"


def test_deep_analysis_keep_vetoed_is_opt_in(monkeypatch):
    from modules.deep_analysis import DeepAnalysis
    da = DeepAnalysis.__new__(DeepAnalysis)
    da.client = None
    veto = {"impact": 5, "surprise": 4, "direction": "BULLISH", "red_team": {"red_team_verdict": "VETO"}}
    monkeypatch.setattr(DeepAnalysis, "_prefetch_batch", lambda self, s, d, r=None: {})
    monkeypatch.setattr(DeepAnalysis, "_analyze", lambda self, c, ctx=None: dict(veto))
    assert da.run([{"ticker": "KMB"}]) == []                                   # Produktion unverändert
    kept = da.run([{"ticker": "KMB"}], keep_vetoed=True)
    assert kept and kept[0]["deep_analysis"]["red_team"]["red_team_verdict"] == "VETO"


def test_report_includes_offline_census_rows(tmp_path):
    out = tmp_path / "off"
    out.mkdir()
    rows_off = [{"ticker": f"T{i}", "fields": {"census_impact": i, "census_direction": "BULLISH",
                                               "census_red_team_verdict": "VETO" if i % 2 else "PASS",
                                               "llm_census": {"status": "analyzed", "pi": 0.1}}} for i in range(6)]
    (out / "2026-11-25.json").write_text(json.dumps({"status": "DONE", "day": "2026-11-25", "rows": rows_off}))
    (out / "2026-11-13.json").write_text(json.dumps({"status": "SKIPPED", "day": "2026-11-13", "rows": []}))
    ledger = [{"date": "2026-11-25", "ticker": f"T{i}", "features": {}, "outcomes": {"ret_20d": 0.01 * i}}
              for i in range(6)] + [{"date": "2026-11-25", "ticker": "ZZZ", "features": {}, "outcomes": {}}]
    merged = lc.with_offline_census(ledger, out)
    rep = lc.report(merged, date(2026, 12, 30))
    assert rep["analyzed_rows"] == 6 and rep["days"] == 1
    h = rep["horizons"]["20d"]
    assert h["n"] == 6 and h["ic_census_impact"] == 1.0
    assert h["by_red_team"]["VETO"]["n"] == 3 and h["by_red_team"]["NO_VETO"]["n"] == 3
    assert merged[-1]["features"] == {}                                        # nicht gezogene Zeile unverändert

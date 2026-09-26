"""
Tests für modules/candidate_ledger.py – Universal Candidate Ledger.

Alle Preis-Abrufe werden gemockt (kein Netzwerkzugriff). Fokus:
  - record/reject/pass/flush schreibt korrekte Zeilen
  - Dedup bei zweitem Flush am selben Tag
  - update_outcomes befüllt Horizonte korrekt (inkl. BEARISH-Sign-Flip)
  - Fehler im Preis-Abruf lassen die Funktionen nie crashen
  - summarize liefert plausible Kennzahlen
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from modules import candidate_ledger as cl


@pytest.fixture(autouse=True)
def _reset_state():
    cl._state["date"] = None
    cl._state["config_hash"] = "unknown"
    cl._state["pipeline_version"] = "unknown"
    cl._state["entries"] = {}
    cl._state["flushed"] = False
    yield
    cl._state["date"] = None
    cl._state["entries"] = {}
    cl._state["flushed"] = False


@pytest.fixture
def ledger_root(tmp_path):
    return tmp_path / "candidate_ledger"


def _read_jsonl(path: Path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ── start_run ────────────────────────────────────────────────────────────────

def test_start_run_sets_state(monkeypatch):
    monkeypatch.setattr(cl, "_compute_config_hash", lambda: "abc123")
    monkeypatch.setattr(cl, "_detect_pipeline_version", lambda: "v9.9")
    cl.start_run("2026-09-26")
    assert cl._state["date"] == "2026-09-26"
    assert cl._state["config_hash"] == "abc123"
    assert cl._state["pipeline_version"] == "v9.9"
    assert cl._state["entries"] == {}
    assert cl._state["flushed"] is False


# ── note / mark_rejected / mark_passed ───────────────────────────────────────

def test_note_upserts_fields():
    cl.start_run("2026-09-26")
    cl.note("AAPL", stage="universe")
    cl.note("AAPL", stage="deep_analysis", direction="BULLISH", impact=6)
    e = cl._state["entries"]["AAPL"]
    assert e["stage"] == "deep_analysis"
    assert e["direction"] == "BULLISH"
    assert e["features"]["impact"] == 6


def test_note_ignores_missing_ticker():
    cl.start_run("2026-09-26")
    cl.note(None, stage="universe")
    cl.note("", stage="universe")
    assert cl._state["entries"] == {}


def test_mark_rejected_sets_status_and_stage():
    cl.start_run("2026-09-26")
    cl.note("XOM", stage="quick_mc")
    cl.mark_rejected("XOM", "mc_below_threshold")
    e = cl._state["entries"]["XOM"]
    assert e["status"] == "rejected"
    assert e["reject_reason"] == "mc_below_threshold"
    assert e["reject_stage"] == "quick_mc"


def test_mark_rejected_tolerates_none_ticker():
    cl.start_run("2026-09-26")
    cl.mark_rejected(None, "some_reason")  # must not raise
    assert cl._state["entries"] == {}


def test_mark_passed_sets_status():
    cl.start_run("2026-09-26")
    cl.note("MSFT", stage="trade_proposal")
    cl.mark_passed("MSFT")
    assert cl._state["entries"]["MSFT"]["status"] == "proposed"


# ── flush ────────────────────────────────────────────────────────────────────

def test_flush_writes_expected_fields(monkeypatch, ledger_root):
    monkeypatch.setattr(cl, "_fetch_prices_batch", lambda tickers: {t: 123.45 for t in tickers})
    cl.start_run("2026-09-26")
    cl.note("AAPL", stage="universe")
    cl.note("AAPL", stage="deep_analysis", direction="BULLISH", impact=6)
    cl.mark_passed("AAPL")
    cl.note("XOM", stage="quick_mc")
    cl.mark_rejected("XOM", "mc_below_threshold")

    cl.flush(reports_dir_root=ledger_root)

    month_file = ledger_root / "2026-09.jsonl"
    assert month_file.exists()
    rows = {r["ticker"]: r for r in _read_jsonl(month_file)}

    assert rows["AAPL"]["status"] == "proposed"
    assert rows["AAPL"]["direction"] == "BULLISH"
    assert rows["AAPL"]["features"]["impact"] == 6
    assert rows["AAPL"]["entry_price"] == 123.45
    assert rows["AAPL"]["outcomes"] == {}
    assert rows["AAPL"]["date"] == "2026-09-26"

    assert rows["XOM"]["status"] == "rejected"
    assert rows["XOM"]["reject_reason"] == "mc_below_threshold"
    assert rows["XOM"]["reject_stage"] == "quick_mc"


def test_flush_is_idempotent_within_same_run(monkeypatch, ledger_root):
    calls = []
    monkeypatch.setattr(cl, "_fetch_prices_batch", lambda tickers: (calls.append(tickers), {t: 1.0 for t in tickers})[1])
    cl.start_run("2026-09-26")
    cl.note("AAPL", stage="universe")
    cl.flush(reports_dir_root=ledger_root)
    cl.flush(reports_dir_root=ledger_root)  # second call in same run: no-op
    rows = _read_jsonl(ledger_root / "2026-09.jsonl")
    assert len(rows) == 1
    assert len(calls) == 1


def test_flush_dedups_across_runs_same_day(monkeypatch, ledger_root):
    monkeypatch.setattr(cl, "_fetch_prices_batch", lambda tickers: {t: 50.0 for t in tickers})

    cl.start_run("2026-09-26")
    cl.note("AAPL", stage="universe")
    cl.flush(reports_dir_root=ledger_root)

    # Zweiter Lauf am selben Tag (Pipeline kann 2x/Tag laufen)
    cl.start_run("2026-09-26")
    cl.note("AAPL", stage="universe")
    cl.note("MSFT", stage="universe")
    cl.flush(reports_dir_root=ledger_root)

    rows = _read_jsonl(ledger_root / "2026-09.jsonl")
    tickers = sorted(r["ticker"] for r in rows)
    assert tickers == ["AAPL", "MSFT"]


def test_flush_never_raises_when_price_fetch_throws(monkeypatch, ledger_root):
    def boom(tickers):
        raise RuntimeError("network down")
    monkeypatch.setattr(cl, "_fetch_prices_batch", boom)
    cl.start_run("2026-09-26")
    cl.note("AAPL", stage="universe")
    cl.flush(reports_dir_root=ledger_root)  # must not raise


def test_flush_handles_empty_run(ledger_root):
    cl.start_run("2026-09-26")
    cl.flush(reports_dir_root=ledger_root)  # nothing noted → no-op, no crash
    assert not (ledger_root / "2026-09.jsonl").exists()


# ── update_outcomes ──────────────────────────────────────────────────────────

def _write_ledger_line(root: Path, month: str, row: dict):
    root.mkdir(parents=True, exist_ok=True)
    with open(root / f"{month}.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


def test_update_outcomes_fills_bullish_horizon(monkeypatch, ledger_root):
    entry_date = "2026-08-01"
    _write_ledger_line(ledger_root, "2026-08", {
        "date": entry_date, "ticker": "AAPL", "pipeline_version": "v8.3",
        "config_hash": "abc", "status": "proposed", "reject_stage": None,
        "reject_reason": None, "direction": "BULLISH", "features": {},
        "entry_price": 100.0, "outcomes": {},
    })

    entry_dt = datetime.strptime(entry_date, "%Y-%m-%d")
    hist = {"AAPL": [(entry_dt + timedelta(days=d), 100.0 + d) for d in range(0, 40)]}
    monkeypatch.setattr(cl, "_fetch_history_batch", lambda tickers, period_days: hist)

    today = (entry_dt + timedelta(days=25)).strftime("%Y-%m-%d")
    cl.update_outcomes(today, root=ledger_root)

    rows = _read_jsonl(ledger_root / "2026-08.jsonl")
    outcomes = rows[0]["outcomes"]
    assert "ret_5d" in outcomes
    assert "ret_20d" in outcomes
    assert "ret_45d" not in outcomes  # horizon not elapsed yet
    # price at +5d = 105 -> ret = 0.05
    assert outcomes["ret_5d"] == pytest.approx(0.05, abs=1e-4)
    assert outcomes["ret_20d"] == pytest.approx(0.20, abs=1e-4)
    assert "mfe" in outcomes and "mae" in outcomes
    assert outcomes["mfe"] >= outcomes["ret_20d"]


def test_update_outcomes_flips_sign_for_bearish(monkeypatch, ledger_root):
    entry_date = "2026-08-01"
    _write_ledger_line(ledger_root, "2026-08", {
        "date": entry_date, "ticker": "XOM", "pipeline_version": "v8.3",
        "config_hash": "abc", "status": "rejected", "reject_stage": "quick_mc",
        "reject_reason": "mc_below_threshold", "direction": "BEARISH", "features": {},
        "entry_price": 100.0, "outcomes": {},
    })

    entry_dt = datetime.strptime(entry_date, "%Y-%m-%d")
    # underlying rises 10% by day 5 -> bearish counterfactual loss (negative)
    hist = {"XOM": [(entry_dt + timedelta(days=d), 100.0 + d * 2) for d in range(0, 10)]}
    monkeypatch.setattr(cl, "_fetch_history_batch", lambda tickers, period_days: hist)

    today = (entry_dt + timedelta(days=6)).strftime("%Y-%m-%d")
    cl.update_outcomes(today, root=ledger_root)

    rows = _read_jsonl(ledger_root / "2026-08.jsonl")
    outcomes = rows[0]["outcomes"]
    # price at +5d = 110 -> raw ret = 0.10 -> bearish flipped = -0.10
    assert outcomes["ret_5d"] == pytest.approx(-0.10, abs=1e-4)


def test_update_outcomes_backfills_missing_entry_price(monkeypatch, ledger_root):
    """Entry-Preis fehlte beim flush → wird aus der Historie nachgetragen."""
    entry_date = "2026-08-01"
    _write_ledger_line(ledger_root, "2026-08", {
        "date": entry_date, "ticker": "AAPL", "pipeline_version": "v8.3",
        "config_hash": "abc", "status": "proposed", "reject_stage": None,
        "reject_reason": None, "direction": "BULLISH", "features": {},
        "entry_price": None, "outcomes": {},
    })
    d0 = datetime.strptime(entry_date, "%Y-%m-%d")
    hist = [(d0 + timedelta(days=k), 100.0 + k) for k in range(0, 31)]
    monkeypatch.setattr(cl, "_fetch_history_batch", lambda t, p: {"AAPL": hist})

    cl.update_outcomes((d0 + timedelta(days=30)).strftime("%Y-%m-%d"), root=ledger_root)
    row = _read_jsonl(ledger_root / "2026-08.jsonl")[0]
    assert row["entry_price"] == 100.0
    assert abs(row["outcomes"]["ret_5d"] - 0.05) < 1e-9
    assert abs(row["outcomes"]["ret_20d"] - 0.20) < 1e-9
    assert "ret_45d" not in row["outcomes"]


def test_update_outcomes_no_history_keeps_row_unchanged(monkeypatch, ledger_root):
    entry_date = "2026-08-01"
    _write_ledger_line(ledger_root, "2026-08", {
        "date": entry_date, "ticker": "AAPL", "pipeline_version": "v8.3",
        "config_hash": "abc", "status": "rejected", "reject_stage": "universe",
        "reject_reason": "ingress_invalid_data", "direction": None, "features": {},
        "entry_price": None, "outcomes": {},
    })
    monkeypatch.setattr(cl, "_fetch_history_batch", lambda t, p: {})
    today = (datetime.strptime(entry_date, "%Y-%m-%d") + timedelta(days=30)).strftime("%Y-%m-%d")
    cl.update_outcomes(today, root=ledger_root)
    rows = _read_jsonl(ledger_root / "2026-08.jsonl")
    assert rows[0]["outcomes"] == {} and rows[0]["entry_price"] is None


def test_update_outcomes_never_raises_when_fetch_throws(monkeypatch, ledger_root):
    entry_date = "2026-08-01"
    _write_ledger_line(ledger_root, "2026-08", {
        "date": entry_date, "ticker": "AAPL", "pipeline_version": "v8.3",
        "config_hash": "abc", "status": "proposed", "reject_stage": None,
        "reject_reason": None, "direction": "BULLISH", "features": {},
        "entry_price": 100.0, "outcomes": {},
    })

    def boom(tickers, period_days):
        raise RuntimeError("network down")
    monkeypatch.setattr(cl, "_fetch_history_batch", boom)

    today = (datetime.strptime(entry_date, "%Y-%m-%d") + timedelta(days=30)).strftime("%Y-%m-%d")
    cl.update_outcomes(today, root=ledger_root)  # must not raise


def test_update_outcomes_skips_old_files(monkeypatch, ledger_root):
    """Dateien älter als 5 Monate werden nicht mehr angefasst."""
    old_month = "2025-01"
    _write_ledger_line(ledger_root, old_month, {
        "date": "2025-01-05", "ticker": "AAPL", "pipeline_version": "v8.3",
        "config_hash": "abc", "status": "proposed", "reject_stage": None,
        "reject_reason": None, "direction": "BULLISH", "features": {},
        "entry_price": 100.0, "outcomes": {},
    })
    called = {"n": 0}
    def fake_fetch(tickers, period_days):
        called["n"] += 1
        return {}
    monkeypatch.setattr(cl, "_fetch_history_batch", fake_fetch)

    cl.update_outcomes("2026-09-26", root=ledger_root)
    assert called["n"] == 0


# ── summarize ────────────────────────────────────────────────────────────────

def test_summarize_aggregates_by_reason(ledger_root):
    _write_ledger_line(ledger_root, "2026-08", {
        "date": "2026-08-01", "ticker": "A", "status": "proposed",
        "reject_reason": None, "direction": "BULLISH", "features": {},
        "entry_price": 100.0, "outcomes": {"ret_20d": 0.10, "ret_45d": 0.15},
    })
    _write_ledger_line(ledger_root, "2026-08", {
        "date": "2026-08-01", "ticker": "B", "status": "rejected",
        "reject_reason": "mc_below_threshold", "direction": "BULLISH", "features": {},
        "entry_price": 50.0, "outcomes": {"ret_20d": -0.05, "ret_45d": -0.02},
    })
    _write_ledger_line(ledger_root, "2026-08", {
        "date": "2026-08-02", "ticker": "C", "status": "rejected",
        "reject_reason": "mc_below_threshold", "direction": "BULLISH", "features": {},
        "entry_price": 30.0, "outcomes": {"ret_20d": 0.02, "ret_45d": 0.03},
    })

    summary = cl.summarize(ledger_root)

    assert summary["proposed"]["n"] == 1
    assert summary["proposed"]["mean_ret_20d"] == pytest.approx(0.10)
    assert summary["proposed"]["share_positive"] == pytest.approx(1.0)

    reason = summary["mc_below_threshold"]
    assert reason["n"] == 2
    assert reason["mean_ret_20d"] == pytest.approx((-0.05 + 0.02) / 2)
    assert reason["share_positive"] == pytest.approx(0.5)


def test_summarize_handles_missing_root(tmp_path):
    result = cl.summarize(tmp_path / "does_not_exist")
    assert result == {}


def test_summarize_never_raises_on_corrupt_line(ledger_root):
    ledger_root.mkdir(parents=True, exist_ok=True)
    with open(ledger_root / "2026-08.jsonl", "w") as f:
        f.write("not valid json\n")
        f.write(json.dumps({
            "date": "2026-08-01", "ticker": "A", "status": "proposed",
            "reject_reason": None, "outcomes": {"ret_20d": 0.1},
        }) + "\n")
    result = cl.summarize(ledger_root)
    assert result["proposed"]["n"] == 1


def test_flush_marks_unlabeled_drop_with_last_stage(monkeypatch, ledger_root):
    """Kandidat ohne reject()/mark_passed → status 'dropped' mit letzter Stufe."""
    monkeypatch.setattr(cl, "_fetch_prices_batch", lambda tickers: {})
    cl.start_run("2026-09-25")
    cl.note("XOM", stage="universe")
    cl.flush(ledger_root)
    row = _read_jsonl(ledger_root / "2026-09.jsonl")[0]
    assert row["status"] == "dropped"
    assert row["reject_stage"] == "universe"
    assert row["reject_reason"] == "unlabeled_after_universe"

"""
Tests für modules/candidate_ledger.py – Universal Candidate Ledger.

Alle Preis-Abrufe werden gemockt (kein Netzwerkzugriff). Fokus:
  - record/reject/pass/flush schreibt korrekte Zeilen
  - Dedup bei zweitem Flush am selben Tag
  - update_outcomes befüllt Horizonte korrekt (inkl. BEARISH-Sign-Flip)
  - Fehler im Preis-Abruf lassen die Funktionen nie crashen
  - summarize liefert plausible Kennzahlen
  - signal_timestamp wird beim ersten note() je Kandidat/Lauf gesetzt und
    im geflushten Row persistiert
"""

import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from modules import candidate_ledger as cl


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    cl._state["date"] = None
    cl._state["config_hash"] = "unknown"
    cl._state["pipeline_version"] = "unknown"
    cl._state["entries"] = {}
    cl._state["flushed"] = False
    # Default für alle bestehenden (Pre-P0-A) Tests: keine Tradier-Quotes
    # (kein API-Key im Testlauf — degradiert eh auf {}), Session="regular",
    # damit der bisherige "letzter yfinance-Preis"-Pfad (jetzt yf_last)
    # unverändert greift, sofern ein Test nicht explizit etwas anderes will.
    monkeypatch.setattr(cl.market_snapshot, "fetch_underlying_quotes", lambda tickers: {})
    monkeypatch.setattr(cl.market_snapshot, "us_market_session", lambda ts: "regular")
    # Review-Fix (replicated iv_rank): Default für alle Tests, die real_option
    # via select_contract mocken, aber nicht am replizierten iv_rank
    # interessiert sind — vermeidet echte Netzwerk-Calls (yf.download/
    # Tradier-Chain) und macht strategy_source deterministisch
    # "iv_history_fallback"/"default_long" statt zufällig vom Testlauf-
    # Netzwerkzugriff abhängig. Tests, die real_strategy explizit testen,
    # überschreiben dies gezielt.
    monkeypatch.setattr(cl, "_fetch_history_batch", lambda tickers, period_days: {})
    monkeypatch.setattr(cl.market_snapshot, "fetch_term_iv_point", lambda ticker, spot, min_dte=7: None)
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
    e = cl._state["entries"]["AAPL"][0]
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
    e = cl._state["entries"]["XOM"][0]
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
    assert cl._state["entries"]["MSFT"][0]["status"] == "proposed"


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


def test_summarize_includes_opt_ret_stats(ledger_root):
    _write_ledger_line(ledger_root, "2026-08", {
        "date": "2026-08-01", "ticker": "A", "status": "proposed",
        "reject_reason": None, "direction": "BULLISH", "features": {},
        "entry_price": 100.0,
        "outcomes": {"ret_20d": 0.10, "ret_45d": 0.15, "opt_ret_20d": 0.40, "opt_ret_45d": 0.55},
    })
    _write_ledger_line(ledger_root, "2026-08", {
        "date": "2026-08-01", "ticker": "B", "status": "rejected",
        "reject_reason": "mc_below_threshold", "direction": "BULLISH", "features": {},
        "entry_price": 50.0,
        "outcomes": {"ret_20d": -0.05, "ret_45d": -0.02, "opt_ret_20d": -0.30, "opt_ret_45d": -0.20},
    })

    summary = cl.summarize(ledger_root)
    assert summary["proposed"]["mean_opt_ret_20d"] == pytest.approx(0.40)
    assert summary["proposed"]["opt_share_positive"] == pytest.approx(1.0)
    assert summary["mc_below_threshold"]["mean_opt_ret_20d"] == pytest.approx(-0.30)
    assert summary["mc_below_threshold"]["opt_share_positive"] == pytest.approx(0.0)


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


# ── hypo_option (Black-Scholes-Counterfactual) ───────────────────────────────

def test_flush_builds_hypo_call_for_bullish(monkeypatch, ledger_root):
    monkeypatch.setattr(cl, "_fetch_prices_batch", lambda tickers: {t: 100.0 for t in tickers})
    cl.start_run("2026-09-26")
    cl.note("AAPL", stage="deep_analysis", direction="BULLISH", ttm="4-8 Wochen")
    cl.mark_passed("AAPL")
    cl.flush(reports_dir_root=ledger_root)

    row = _read_jsonl(ledger_root / "2026-09.jsonl")[0]
    hypo = row["hypo_option"]
    assert hypo["kind"] == "call"
    assert hypo["strike"] == 100
    assert hypo["dte"] == 120  # ttm_to_dte_floor("4-8 Wochen")
    assert hypo["iv_source"] == "default"
    assert hypo["iv"] == pytest.approx(0.35)
    assert hypo["entry_premium"] > 0
    assert hypo["spread_cost"] == pytest.approx(0.05)


def test_flush_builds_hypo_put_for_bearish(monkeypatch, ledger_root):
    monkeypatch.setattr(cl, "_fetch_prices_batch", lambda tickers: {t: 50.0 for t in tickers})
    cl.start_run("2026-09-26")
    cl.note("XOM", stage="deep_analysis", direction="BEARISH", ttm="6 Monate")
    cl.mark_passed("XOM")
    cl.flush(reports_dir_root=ledger_root)

    row = _read_jsonl(ledger_root / "2026-09.jsonl")[0]
    hypo = row["hypo_option"]
    assert hypo["kind"] == "put"
    assert hypo["strike"] == 50
    assert hypo["dte"] == 140


def test_flush_hypo_uses_implied_vol_when_present(monkeypatch, ledger_root):
    monkeypatch.setattr(cl, "_fetch_prices_batch", lambda tickers: {t: 100.0 for t in tickers})
    cl.start_run("2026-09-26")
    cl.note("AAPL", stage="deep_analysis", direction="BULLISH", implied_vol=0.42)
    cl.flush(reports_dir_root=ledger_root)
    row = _read_jsonl(ledger_root / "2026-09.jsonl")[0]
    assert row["hypo_option"]["iv_source"] == "implied"
    assert row["hypo_option"]["iv"] == pytest.approx(0.42)


def test_flush_hypo_uses_realized_sigma_when_no_iv(monkeypatch, ledger_root):
    monkeypatch.setattr(cl, "_fetch_prices_batch", lambda tickers: {t: 100.0 for t in tickers})
    cl.start_run("2026-09-26")
    cl.note("AAPL", stage="mismatch", sigma_30d=0.02)
    cl.note("AAPL", stage="deep_analysis", direction="BULLISH")
    cl.flush(reports_dir_root=ledger_root)
    row = _read_jsonl(ledger_root / "2026-09.jsonl")[0]
    hypo = row["hypo_option"]
    assert hypo["iv_source"] == "realized"
    assert hypo["iv"] == pytest.approx(0.02 * (252 ** 0.5), abs=1e-4)


def test_flush_skips_hypo_when_direction_unknown(monkeypatch, ledger_root):
    monkeypatch.setattr(cl, "_fetch_prices_batch", lambda tickers: {t: 100.0 for t in tickers})
    cl.start_run("2026-09-26")
    cl.note("XOM", stage="universe")
    cl.flush(reports_dir_root=ledger_root)
    row = _read_jsonl(ledger_root / "2026-09.jsonl")[0]
    assert "hypo_option" not in row


def test_flush_hypo_never_raises_with_missing_price(monkeypatch, ledger_root):
    monkeypatch.setattr(cl, "_fetch_prices_batch", lambda tickers: {t: None for t in tickers})
    cl.start_run("2026-09-26")
    cl.note("AAPL", stage="deep_analysis", direction="BULLISH")
    cl.flush(reports_dir_root=ledger_root)  # must not raise
    row = _read_jsonl(ledger_root / "2026-09.jsonl")[0]
    assert "hypo_option" not in row


# ── opt_ret_{h}d (Options-Counterfactual über update_outcomes) ───────────────

def test_update_outcomes_computes_opt_ret_for_known_path(monkeypatch, ledger_root):
    entry_date = "2026-08-01"
    _write_ledger_line(ledger_root, "2026-08", {
        "date": entry_date, "ticker": "AAPL", "pipeline_version": "v8.3",
        "config_hash": "abc", "status": "proposed", "reject_stage": None,
        "reject_reason": None, "direction": "BULLISH", "features": {},
        "entry_price": 100.0, "outcomes": {},
        "hypo_option": {
            "kind": "call", "strike": 100, "dte": 120, "iv": 0.35,
            "iv_source": "default", "entry_premium": 6.7723, "spread_cost": 0.05,
        },
    })
    entry_dt = datetime.strptime(entry_date, "%Y-%m-%d")
    # Underlying steigt: +5% nach 20 Tagen
    hist = {"AAPL": [(entry_dt + timedelta(days=d), 100.0 + d * 0.25) for d in range(0, 40)]}
    monkeypatch.setattr(cl, "_fetch_history_batch", lambda tickers, period_days: hist)

    today = (entry_dt + timedelta(days=25)).strftime("%Y-%m-%d")
    cl.update_outcomes(today, root=ledger_root)

    row = _read_jsonl(ledger_root / "2026-08.jsonl")[0]
    outcomes = row["outcomes"]
    assert "opt_ret_5d" in outcomes
    assert "opt_ret_20d" in outcomes
    # Underlying up -> call gains value -> opt_ret should be positive and
    # (due to leverage) larger in magnitude than the underlying return.
    assert outcomes["opt_ret_20d"] > outcomes["ret_20d"]


def test_update_outcomes_opt_ret_put_gains_when_underlying_falls(monkeypatch, ledger_root):
    entry_date = "2026-08-01"
    _write_ledger_line(ledger_root, "2026-08", {
        "date": entry_date, "ticker": "XOM", "pipeline_version": "v8.3",
        "config_hash": "abc", "status": "rejected", "reject_stage": "quick_mc",
        "reject_reason": "mc_below_threshold", "direction": "BEARISH", "features": {},
        "entry_price": 100.0, "outcomes": {},
        "hypo_option": {
            "kind": "put", "strike": 100, "dte": 120, "iv": 0.35,
            "iv_source": "default", "entry_premium": 6.7723, "spread_cost": 0.05,
        },
    })
    entry_dt = datetime.strptime(entry_date, "%Y-%m-%d")
    hist = {"XOM": [(entry_dt + timedelta(days=d), 100.0 - d * 0.5) for d in range(0, 10)]}
    monkeypatch.setattr(cl, "_fetch_history_batch", lambda tickers, period_days: hist)

    today = (entry_dt + timedelta(days=6)).strftime("%Y-%m-%d")
    cl.update_outcomes(today, root=ledger_root)

    row = _read_jsonl(ledger_root / "2026-08.jsonl")[0]
    outcomes = row["outcomes"]
    assert outcomes["opt_ret_5d"] > 0  # put gains as underlying falls


def test_update_outcomes_opt_ret_capped_at_minus_one(monkeypatch, ledger_root):
    entry_date = "2026-08-01"
    _write_ledger_line(ledger_root, "2026-08", {
        "date": entry_date, "ticker": "AAPL", "pipeline_version": "v8.3",
        "config_hash": "abc", "status": "proposed", "reject_stage": None,
        "reject_reason": None, "direction": "BULLISH", "features": {},
        "entry_price": 100.0, "outcomes": {},
        "hypo_option": {
            "kind": "call", "strike": 100, "dte": 20, "iv": 0.35,
            "iv_source": "default", "entry_premium": 3.0, "spread_cost": 0.05,
        },
    })
    entry_dt = datetime.strptime(entry_date, "%Y-%m-%d")
    # Underlying crashes hard -> deep OTM call at expiry -> near-total loss
    hist = {"AAPL": [(entry_dt + timedelta(days=d), 60.0) for d in range(0, 25)]}
    monkeypatch.setattr(cl, "_fetch_history_batch", lambda tickers, period_days: hist)

    today = (entry_dt + timedelta(days=25)).strftime("%Y-%m-%d")
    cl.update_outcomes(today, root=ledger_root)

    row = _read_jsonl(ledger_root / "2026-08.jsonl")[0]
    assert row["outcomes"]["opt_ret_20d"] >= -1.0


def test_update_outcomes_no_crash_when_hypo_missing(monkeypatch, ledger_root):
    entry_date = "2026-08-01"
    _write_ledger_line(ledger_root, "2026-08", {
        "date": entry_date, "ticker": "AAPL", "pipeline_version": "v8.3",
        "config_hash": "abc", "status": "proposed", "reject_stage": None,
        "reject_reason": None, "direction": "BULLISH", "features": {},
        "entry_price": 100.0, "outcomes": {},
    })
    entry_dt = datetime.strptime(entry_date, "%Y-%m-%d")
    hist = {"AAPL": [(entry_dt + timedelta(days=d), 100.0 + d) for d in range(0, 10)]}
    monkeypatch.setattr(cl, "_fetch_history_batch", lambda tickers, period_days: hist)

    today = (entry_dt + timedelta(days=6)).strftime("%Y-%m-%d")
    cl.update_outcomes(today, root=ledger_root)  # must not raise

    row = _read_jsonl(ledger_root / "2026-08.jsonl")[0]
    assert "ret_5d" in row["outcomes"]
    assert "opt_ret_5d" not in row["outcomes"]


def test_update_outcomes_backfills_hypo_option_when_entry_price_missing(monkeypatch, ledger_root):
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
    assert "hypo_option" in row
    assert row["hypo_option"]["kind"] == "call"
    assert "opt_ret_5d" in row["outcomes"]


# ── Stage-Notizen landen in features (flush) ─────────────────────────────────

def test_note_fields_across_stages_end_up_in_flushed_features(monkeypatch, ledger_root):
    monkeypatch.setattr(cl, "_fetch_prices_batch", lambda tickers: {t: 100.0 for t in tickers})
    cl.start_run("2026-09-26")
    cl.note("AAPL", stage="mismatch", mismatch=6.5)
    cl.note("AAPL", stage="deep_analysis", direction="BULLISH", impact=6, surprise=4)
    cl.note("AAPL", stage="quick_mc", quick_mc_hit_rate=0.6)
    cl.note("AAPL", stage="final_mc", final_mc_hit_rate=0.55)
    cl.note("AAPL", stage="trade_proposal", trade_score=88)
    cl.mark_passed("AAPL")
    cl.flush(reports_dir_root=ledger_root)

    row = _read_jsonl(ledger_root / "2026-09.jsonl")[0]
    f = row["features"]
    assert f["impact"] == 6
    assert f["surprise"] == 4
    assert f["mismatch"] == 6.5
    assert f["quick_mc_hit_rate"] == 0.6
    assert f["final_mc_hit_rate"] == 0.55
    assert f["trade_score"] == 88


# ── signal_timestamp (Pre-Registrierungs-Zeitgate für den Challenger) ───────

def test_note_sets_signal_timestamp_on_first_note(monkeypatch):
    fixed = datetime(2026, 9, 26, 14, 3, 7, tzinfo=timezone.utc)

    class FakeDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed

    monkeypatch.setattr(cl, "datetime", FakeDateTime)
    cl.start_run("2026-09-26")
    cl.note("AAPL", stage="universe")
    assert cl._state["entries"]["AAPL"][0]["signal_timestamp"] == "2026-09-26T14:03:07+00:00"


def test_note_signal_timestamp_is_set_once_across_multiple_notes(monkeypatch):
    times = iter([
        datetime(2026, 9, 26, 10, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 9, 26, 11, 0, 0, tzinfo=timezone.utc),
    ])

    class FakeDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return next(times)

    monkeypatch.setattr(cl, "datetime", FakeDateTime)
    cl.start_run("2026-09-26")
    cl.note("AAPL", stage="universe")
    cl.note("AAPL", stage="deep_analysis", direction="BULLISH")
    # Second note() must NOT advance the timestamp — it stays at the first
    # value noted for this ticker in this run.
    assert cl._state["entries"]["AAPL"][0]["signal_timestamp"] == "2026-09-26T10:00:00+00:00"


def test_flush_persists_signal_timestamp(monkeypatch, ledger_root):
    monkeypatch.setattr(cl, "_fetch_prices_batch", lambda tickers: {t: 100.0 for t in tickers})
    fixed = datetime(2026, 9, 26, 14, 3, 7, tzinfo=timezone.utc)

    class FakeDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed

    monkeypatch.setattr(cl, "datetime", FakeDateTime)
    cl.start_run("2026-09-26")
    cl.note("AAPL", stage="universe")
    cl.flush(reports_dir_root=ledger_root)

    row = _read_jsonl(ledger_root / "2026-09.jsonl")[0]
    assert row["signal_timestamp"] == "2026-09-26T14:03:07+00:00"


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


# ── P0-A: Entry-Policy nach Session (nie letzter Preis bei pre/post/closed) ──

def test_flush_entry_regular_session_uses_quote_mid(monkeypatch, ledger_root):
    monkeypatch.setattr(cl.market_snapshot, "us_market_session", lambda ts: "regular")
    monkeypatch.setattr(
        cl.market_snapshot, "fetch_underlying_quotes",
        lambda tickers: {t: {"bid": 100.0, "ask": 100.4, "mid": 100.2, "last": 100.1,
                              "prev_close": 99.0, "open": 100.0, "quote_ts": "t", "source": "tradier"}
                         for t in tickers},
    )
    monkeypatch.setattr(cl, "_fetch_prices_batch", lambda tickers: {t: 999.0 for t in tickers})  # darf nicht genutzt werden

    cl.start_run("2026-09-26")
    cl.note("AAPL", stage="universe")
    cl.flush(reports_dir_root=ledger_root)

    row = _read_jsonl(ledger_root / "2026-09.jsonl")[0]
    assert row["entry_basis"] == "quote_mid"
    assert row["entry_price"] == pytest.approx(100.2)
    assert row["session"] == "regular"
    assert row["underlying"]["prev_close"] == pytest.approx(99.0)
    assert row["gap_at_entry"] == pytest.approx(100.2 / 99.0 - 1.0, abs=1e-4)


@pytest.mark.parametrize("session", ["pre", "post", "closed"])
def test_flush_entry_non_regular_session_defers_to_next_open(monkeypatch, ledger_root, session):
    monkeypatch.setattr(cl.market_snapshot, "us_market_session", lambda ts: session)
    monkeypatch.setattr(cl.market_snapshot, "fetch_underlying_quotes",
                         lambda tickers: {t: {"bid": 100.0, "ask": 100.4, "mid": 100.2,
                                               "last": 100.1, "prev_close": 99.0, "open": None,
                                               "quote_ts": "t", "source": "tradier"} for t in tickers})
    monkeypatch.setattr(cl, "_fetch_prices_batch", lambda tickers: {t: 999.0 for t in tickers})

    cl.start_run("2026-09-26")
    cl.note("AAPL", stage="universe")
    cl.flush(reports_dir_root=ledger_root)

    row = _read_jsonl(ledger_root / "2026-09.jsonl")[0]
    assert row["entry_basis"] == "next_open"
    assert row["entry_price"] is None
    assert row["session"] == session
    # NIEMALS der letzte yfinance-Preis oder der vorherige Schlusskurs:
    assert row["entry_price"] != 999.0
    assert row["entry_price"] != 99.0


def test_flush_entry_regular_session_no_quote_falls_back_to_yfinance(monkeypatch, ledger_root):
    monkeypatch.setattr(cl.market_snapshot, "us_market_session", lambda ts: "regular")
    monkeypatch.setattr(cl.market_snapshot, "fetch_underlying_quotes", lambda tickers: {})  # kein Key/Fehler
    monkeypatch.setattr(cl, "_fetch_prices_batch", lambda tickers: {t: 123.45 for t in tickers})

    cl.start_run("2026-09-26")
    cl.note("AAPL", stage="universe")
    cl.flush(reports_dir_root=ledger_root)

    row = _read_jsonl(ledger_root / "2026-09.jsonl")[0]
    assert row["entry_basis"] == "yf_last"
    assert row["entry_price"] == pytest.approx(123.45)


# ── P0-A: next_open Entry-Preis-Fill in update_outcomes (fake Open-Historie) ─

def test_update_outcomes_fills_next_open_entry_from_pre_market_signal(monkeypatch, ledger_root):
    """Pre-market-Signal an einem Handelstag → Open DESSELBEN Tages."""
    from datetime import date as _date
    _write_ledger_line(ledger_root, "2026-08", {
        "date": "2026-08-04", "ticker": "AAPL", "pipeline_version": "v8.3",
        "config_hash": "abc", "status": "proposed", "reject_stage": None,
        "reject_reason": None, "direction": "BULLISH", "features": {},
        "entry_price": None, "entry_basis": "next_open", "session": "pre",
        "signal_timestamp": "2026-08-04T13:00:00+00:00",  # 09:00 EDT, vor Open
        "underlying": {"prev_close": 98.0}, "outcomes": {},
    })
    hist = {"AAPL": [
        (_date(2026, 8, 4), 101.0, 102.0),
        (_date(2026, 8, 5), 103.0, 104.0),
    ]}
    monkeypatch.setattr(cl, "_fetch_history_open_batch", lambda tickers, period_days: hist)
    monkeypatch.setattr(cl, "_fetch_history_batch", lambda tickers, period_days: {})

    cl.update_outcomes("2026-08-05", root=ledger_root)

    row = _read_jsonl(ledger_root / "2026-08.jsonl")[0]
    assert row["entry_price"] == pytest.approx(101.0)
    assert row["entry_effective_date"] == "2026-08-04"
    assert row["gap_at_entry"] == pytest.approx(101.0 / 98.0 - 1.0, abs=1e-4)
    assert "hypo_option" in row  # wird nachgetragen, sobald der Preis bekannt ist


def test_update_outcomes_fills_next_open_entry_from_post_market_signal(monkeypatch, ledger_root):
    """Post-market/closed-Signal → Open des NÄCHSTEN Handelstags."""
    from datetime import date as _date
    _write_ledger_line(ledger_root, "2026-08", {
        "date": "2026-08-04", "ticker": "AAPL", "pipeline_version": "v8.3",
        "config_hash": "abc", "status": "proposed", "reject_stage": None,
        "reject_reason": None, "direction": "BULLISH", "features": {},
        "entry_price": None, "entry_basis": "next_open", "session": "post",
        "signal_timestamp": "2026-08-04T21:00:00+00:00",  # 17:00 EDT, nach Close
        "underlying": {"prev_close": 98.0}, "outcomes": {},
    })
    hist = {"AAPL": [
        (_date(2026, 8, 4), 101.0, 102.0),   # darf NICHT genommen werden (Signal war NACH Close)
        (_date(2026, 8, 5), 103.0, 104.0),
    ]}
    monkeypatch.setattr(cl, "_fetch_history_open_batch", lambda tickers, period_days: hist)
    monkeypatch.setattr(cl, "_fetch_history_batch", lambda tickers, period_days: {})

    cl.update_outcomes("2026-08-06", root=ledger_root)

    row = _read_jsonl(ledger_root / "2026-08.jsonl")[0]
    assert row["entry_price"] == pytest.approx(103.0)
    assert row["entry_effective_date"] == "2026-08-05"


def test_update_outcomes_next_open_not_yet_available_keeps_row_pending(monkeypatch, ledger_root):
    _write_ledger_line(ledger_root, "2026-08", {
        "date": "2026-08-04", "ticker": "AAPL", "pipeline_version": "v8.3",
        "config_hash": "abc", "status": "proposed", "reject_stage": None,
        "reject_reason": None, "direction": "BULLISH", "features": {},
        "entry_price": None, "entry_basis": "next_open", "session": "post",
        "signal_timestamp": "2026-08-04T21:00:00+00:00",
        "underlying": {"prev_close": 98.0}, "outcomes": {},
    })
    monkeypatch.setattr(cl, "_fetch_history_open_batch", lambda tickers, period_days: {})
    monkeypatch.setattr(cl, "_fetch_history_batch", lambda tickers, period_days: {})

    cl.update_outcomes("2026-08-05", root=ledger_root)  # must not raise

    row = _read_jsonl(ledger_root / "2026-08.jsonl")[0]
    assert row["entry_price"] is None
    assert row["entry_basis"] == "next_open"


def test_update_outcomes_horizons_measured_from_entry_effective_date(monkeypatch, ledger_root):
    """Nach dem next_open-Fill laufen ret_{h}d ab entry_effective_date, nicht
    ab dem ursprünglichen Signal-Datum."""
    from datetime import date as _date
    _write_ledger_line(ledger_root, "2026-08", {
        "date": "2026-08-04", "ticker": "AAPL", "pipeline_version": "v8.3",
        "config_hash": "abc", "status": "proposed", "reject_stage": None,
        "reject_reason": None, "direction": "BULLISH", "features": {},
        "entry_price": None, "entry_basis": "next_open", "session": "post",
        "signal_timestamp": "2026-08-04T21:00:00+00:00",
        "underlying": {"prev_close": 98.0}, "outcomes": {},
    })
    open_hist = {"AAPL": [(_date(2026, 8, 5), 100.0, 100.0)]}
    close_hist = {"AAPL": [(datetime(2026, 8, 5) + timedelta(days=d), 100.0 + d) for d in range(0, 10)]}
    monkeypatch.setattr(cl, "_fetch_history_open_batch", lambda tickers, period_days: open_hist)
    monkeypatch.setattr(cl, "_fetch_history_batch", lambda tickers, period_days: close_hist)

    # 5 Kalendertage nach dem SIGNAL (08-04), aber erst 4 Tage nach dem
    # effektiven Entry (08-05) — ret_5d darf also NOCH NICHT gefüllt sein.
    cl.update_outcomes("2026-08-09", root=ledger_root)
    row = _read_jsonl(ledger_root / "2026-08.jsonl")[0]
    assert "ret_5d" not in row["outcomes"]

    # Ein Tag später (5 Tage nach 08-05) ist der Horizont erreicht.
    cl.update_outcomes("2026-08-10", root=ledger_root)
    row = _read_jsonl(ledger_root / "2026-08.jsonl")[0]
    assert "ret_5d" in row["outcomes"]
    assert row["outcomes"]["ret_5d"] == pytest.approx(0.05, abs=1e-4)


# ── P0-B: echter Options-Kontrakt bei flush() ────────────────────────────────

def test_flush_builds_real_option_when_direction_and_spot_known(monkeypatch, ledger_root):
    monkeypatch.setattr(cl.market_snapshot, "us_market_session", lambda ts: "regular")
    monkeypatch.setattr(cl.market_snapshot, "fetch_underlying_quotes",
                         lambda tickers: {t: {"bid": 100.0, "ask": 100.4, "mid": 100.2,
                                               "last": 100.1, "prev_close": 99.0, "open": 100.0,
                                               "quote_ts": "t", "source": "tradier"} for t in tickers})
    fake_contract = {"symbol": "AAPL261120C00100000", "strike": 100.0, "expiry": "2026-11-20",
                      "dte": 55, "bid": 5.0, "ask": 5.4, "mid": 5.2, "iv": 0.3, "delta": 0.5,
                      "open_interest": 500, "quote_ts": "t"}
    calls = []
    def fake_select(ticker, direction, dte_floor, spot):
        calls.append((ticker, direction, dte_floor, spot))
        return fake_contract
    monkeypatch.setattr(cl.market_snapshot, "select_contract", fake_select)

    cl.start_run("2026-09-26")
    cl.note("AAPL", stage="deep_analysis", direction="BULLISH", ttm="4-8 Wochen")
    cl.flush(reports_dir_root=ledger_root)

    row = _read_jsonl(ledger_root / "2026-09.jsonl")[0]
    ro = row["real_option"]
    assert ro["symbol"] == fake_contract["symbol"]
    assert ro["bid"] == pytest.approx(5.0) and ro["ask"] == pytest.approx(5.4)
    # Regular Session -> sofortiger Entry (kein Look-ahead-Problem, Optionen
    # handeln ja gerade JETZT):
    assert ro["entry_pending"] is False
    assert ro["entry_quote_ts"] == "t"
    assert ro["entry_date"] == "2026-09-26"
    assert "snapshot_quote" not in ro
    assert len(calls) == 1
    assert calls[0][0] == "AAPL" and calls[0][1] == "BULLISH"
    assert calls[0][3] == pytest.approx(100.2)  # spot = entry_price (quote_mid)


def test_flush_real_strategy_uses_batched_closes_and_term_point(monkeypatch, ledger_root):
    """End-to-End: flush() holt 1y-Closes für ALLE snapshot-ausgewählten
    Ticker in EINEM gebündelten Call (_fetch_history_batch) und fragt pro
    Kandidat höchstens EINEN zusätzlichen Term-Structure-Punkt ab
    (market_snapshot.fetch_term_iv_point) — beides fließt in
    real_strategy.strategy_source='replicated' ein."""
    monkeypatch.setattr(cl.market_snapshot, "us_market_session", lambda ts: "regular")
    monkeypatch.setattr(cl.market_snapshot, "fetch_underlying_quotes",
                         lambda tickers: {t: {"bid": 100.0, "ask": 100.4, "mid": 100.2,
                                               "last": 100.1, "prev_close": 99.0, "open": 100.0,
                                               "quote_ts": "t", "source": "tradier"} for t in tickers})
    fake_contract = {"symbol": "AAPL261120C00100000", "strike": 100.0, "expiry": "2026-11-20",
                      "dte": 55, "bid": 5.0, "ask": 5.4, "mid": 5.2, "iv": 0.3, "delta": 0.5,
                      "open_interest": 500, "quote_ts": "t"}
    monkeypatch.setattr(cl.market_snapshot, "select_contract", lambda t, d, f, s: fake_contract)
    monkeypatch.setattr(
        cl.market_snapshot, "select_spread_short_leg",
        lambda ticker, expiry, opt_type, strike: {
            "symbol": "AAPL_SHORT", "strike": strike * 1.10, "bid": 2.0, "ask": 2.2, "mid": 2.1,
        },
    )

    closes = _high_recent_vol_closes()
    history_calls = []

    def fake_history_batch(tickers, period_days):
        history_calls.append((tuple(sorted(tickers)), period_days))
        return {t: [(datetime(2026, 1, 1) + timedelta(days=i), c) for i, c in enumerate(closes)] for t in tickers}

    monkeypatch.setattr(cl, "_fetch_history_batch", fake_history_batch)

    term_calls = []
    monkeypatch.setattr(
        cl.market_snapshot, "fetch_term_iv_point",
        lambda ticker, spot, min_dte=7: term_calls.append(ticker) or None,
    )

    cl.start_run("2026-09-26")
    cl.note("AAPL", stage="deep_analysis", direction="BULLISH", ttm="4-8 Wochen")
    cl.flush(reports_dir_root=ledger_root)

    # EIN gebündelter _fetch_history_batch-Call für alle betroffenen Ticker:
    assert len(history_calls) == 1
    assert history_calls[0][0] == ("AAPL",)
    assert history_calls[0][1] == 365
    # Höchstens EIN zusätzlicher Term-Structure-Call für diesen einen Kandidaten:
    assert term_calls == ["AAPL"]

    row = _read_jsonl(ledger_root / "2026-09.jsonl")[0]
    rs = row["real_strategy"]
    assert rs["strategy_source"] == "replicated"
    assert rs["strategy"] == "BULL_CALL_SPREAD"
    assert len(rs["legs"]) == 2


def test_flush_real_option_pre_market_signal_is_entry_pending(monkeypatch, ledger_root):
    """Optionen handeln nur in der regulären Session: ein pre/post/closed
    ausgewählter Kontrakt darf NICHT sofort mit seiner (stale) Quote als
    Entry gebucht werden — dieselbe Look-ahead-Falle wie beim Underlying."""
    monkeypatch.setattr(cl.market_snapshot, "us_market_session", lambda ts: "pre")
    monkeypatch.setattr(cl.market_snapshot, "fetch_underlying_quotes",
                         lambda tickers: {t: {"bid": 100.0, "ask": 100.4, "mid": 100.2,
                                               "last": 100.1, "prev_close": 99.0, "open": None,
                                               "quote_ts": "t", "source": "tradier"} for t in tickers})
    fake_contract = {"symbol": "AAPL261120C00100000", "strike": 100.0, "expiry": "2026-11-20",
                      "dte": 55, "bid": 5.0, "ask": 5.4, "mid": 5.2, "iv": 0.3, "delta": 0.5,
                      "open_interest": 500, "quote_ts": "stale-t"}
    monkeypatch.setattr(cl.market_snapshot, "select_contract", lambda t, d, f, s: fake_contract)

    cl.start_run("2026-09-26")
    cl.note("AAPL", stage="deep_analysis", direction="BULLISH", ttm="4-8 Wochen")
    cl.flush(reports_dir_root=ledger_root)

    row = _read_jsonl(ledger_root / "2026-09.jsonl")[0]
    ro = row["real_option"]
    assert ro["entry_pending"] is True
    assert ro["bid"] is None and ro["ask"] is None and ro["mid"] is None
    assert ro["snapshot_quote"] == {"bid": 5.0, "ask": 5.4, "mid": 5.2, "quote_ts": "stale-t"}
    assert "entry_date" not in ro
    # Kontrakt-Metadaten (Symbol/Strike/Expiry/DTE/IV/Delta/OI) bleiben trotzdem erhalten:
    assert ro["symbol"] == fake_contract["symbol"]
    assert ro["strike"] == fake_contract["strike"]
    assert ro["expiry"] == fake_contract["expiry"]


def test_flush_real_option_skip_reason_without_direction(monkeypatch, ledger_root):
    monkeypatch.setattr(cl.market_snapshot, "us_market_session", lambda ts: "regular")
    monkeypatch.setattr(cl.market_snapshot, "fetch_underlying_quotes", lambda tickers: {})
    monkeypatch.setattr(cl, "_fetch_prices_batch", lambda tickers: {t: 100.0 for t in tickers})

    cl.start_run("2026-09-26")
    cl.note("XOM", stage="universe")
    cl.flush(reports_dir_root=ledger_root)

    row = _read_jsonl(ledger_root / "2026-09.jsonl")[0]
    assert "real_option" not in row
    assert "real_option_skip_reason" not in row  # gar nicht erst versucht (keine Richtung)


def test_flush_real_option_respects_max_snapshot_cap(monkeypatch, ledger_root):
    monkeypatch.setattr(cl.market_snapshot, "us_market_session", lambda ts: "regular")
    monkeypatch.setattr(cl.market_snapshot, "fetch_underlying_quotes",
                         lambda tickers: {t: {"bid": 100.0, "ask": 100.4, "mid": 100.2,
                                               "last": 100.1, "prev_close": 99.0, "open": 100.0,
                                               "quote_ts": "t", "source": "tradier"} for t in tickers})
    monkeypatch.setattr(cl, "_max_option_snapshots", lambda: 2)
    calls = []
    def fake_select(ticker, direction, dte_floor, spot):
        calls.append(ticker)
        return {"symbol": f"{ticker}_OPT", "strike": 100.0, "expiry": "2026-11-20", "dte": 55,
                "bid": 5.0, "ask": 5.4, "mid": 5.2, "iv": 0.3, "delta": 0.5,
                "open_interest": 500, "quote_ts": "t"}
    monkeypatch.setattr(cl.market_snapshot, "select_contract", fake_select)

    cl.start_run("2026-09-26")
    for t in ["AAA", "BBB", "CCC"]:
        cl.note(t, stage="deep_analysis", direction="BULLISH", ttm="4-8 Wochen")
    cl.flush(reports_dir_root=ledger_root)

    rows = {r["ticker"]: r for r in _read_jsonl(ledger_root / "2026-09.jsonl")}
    with_option = [t for t, r in rows.items() if "real_option" in r]
    without_option = [t for t, r in rows.items() if r.get("real_option_skip_reason") == "budget_random_exclusion"]
    assert len(with_option) == 2
    assert len(without_option) == 1
    assert len(calls) == 2  # Budget begrenzt auch die API-Calls selbst
    for r in rows.values():
        assert r["snapshot_eligible_n"] == 3
        assert r["snapshot_budget"] == 2
    assert sum(1 for r in rows.values() if r["snapshot_selected"]) == 2


def test_flush_real_option_no_api_key_degrades(monkeypatch, ledger_root):
    """Ohne TRADIER_API_KEY liefert market_snapshot.select_contract None
    (ungemockt) — die Ledger-Zeile bekommt trotzdem einen Reason, nie einen
    Crash."""
    monkeypatch.delenv("TRADIER_API_KEY", raising=False)
    monkeypatch.setattr(cl.market_snapshot, "us_market_session", lambda ts: "regular")
    monkeypatch.setattr(cl.market_snapshot, "fetch_underlying_quotes", lambda tickers: {})
    monkeypatch.setattr(cl, "_fetch_prices_batch", lambda tickers: {t: 100.0 for t in tickers})

    cl.start_run("2026-09-26")
    cl.note("AAPL", stage="deep_analysis", direction="BULLISH", ttm="4-8 Wochen")
    cl.flush(reports_dir_root=ledger_root)

    row = _read_jsonl(ledger_root / "2026-09.jsonl")[0]
    assert "real_option" not in row
    assert row["real_option_skip_reason"] == "no_api_key"


# ── P0-B: real_opt_ret_{h}d in update_outcomes (bid/ask, Intrinsic, late) ────
#
# Optionen handeln nur regulär → Fill/Marks passieren nur, wenn der Lauf
# selbst in einer regulären Session steht (_now_utc()/market_snapshot.
# us_market_session gesteuert). Für diese Tests: 2026-08-20 (Do) 15:00 UTC
# = 10:00 EDT (regular); 2026-08-20 22:00 UTC = 17:00 EDT (post).
REGULAR_NOW     = datetime(2026, 8, 20, 15, 0, tzinfo=timezone.utc)
NON_REGULAR_NOW = datetime(2026, 8, 20, 22, 0, tzinfo=timezone.utc)


def test_update_outcomes_real_opt_ret_bid_ask_math(monkeypatch, ledger_root):
    entry_date = "2026-08-01"
    _write_ledger_line(ledger_root, "2026-08", {
        "date": entry_date, "ticker": "AAPL", "pipeline_version": "v8.3",
        "config_hash": "abc", "status": "proposed", "reject_stage": None,
        "reject_reason": None, "direction": "BULLISH", "features": {},
        "entry_price": 100.0, "outcomes": {},
        "real_option": {"symbol": "AAPL261231C00100000", "strike": 100.0,
                         "expiry": "2026-12-31", "dte": 150, "entry_pending": False,
                         "entry_date": entry_date,
                         "bid": 5.0, "ask": 5.4, "mid": 5.2, "iv": 0.3, "delta": 0.5,
                         "open_interest": 500, "quote_ts": "t"},
    })
    entry_dt = datetime.strptime(entry_date, "%Y-%m-%d")
    monkeypatch.setattr(cl, "_fetch_history_batch", lambda t, p: {})
    monkeypatch.setattr(cl.market_snapshot, "fetch_option_quotes",
                         lambda symbols: {"AAPL261231C00100000": {"bid": 7.0, "ask": 7.4, "mid": 7.2, "ts": "t"}})
    monkeypatch.setattr(cl, "_now_utc", lambda: REGULAR_NOW)

    today = (entry_dt + timedelta(days=5)).strftime("%Y-%m-%d")
    cl.update_outcomes(today, root=ledger_root)

    row = _read_jsonl(ledger_root / "2026-08.jsonl")[0]
    outcomes = row["outcomes"]
    assert outcomes["real_opt_ret_5d"] == pytest.approx(7.0 / 5.4 - 1.0, abs=1e-4)      # exit_bid/entry_ask
    assert outcomes["real_opt_ret_mid_5d"] == pytest.approx(7.2 / 5.2 - 1.0, abs=1e-4)  # exit_mid/entry_mid
    assert "real_opt_mark_date_5d" in outcomes
    assert "late_mark" not in row


def test_update_outcomes_real_opt_ret_skips_outside_regular_session(monkeypatch, ledger_root):
    """Läuft feedback.py außerhalb der regulären Session, wird weder gefüllt
    noch markiert — das übernimmt erst der nächste Regular-Session-Lauf."""
    entry_date = "2026-08-01"
    _write_ledger_line(ledger_root, "2026-08", {
        "date": entry_date, "ticker": "AAPL", "pipeline_version": "v8.3",
        "config_hash": "abc", "status": "proposed", "reject_stage": None,
        "reject_reason": None, "direction": "BULLISH", "features": {},
        "entry_price": 100.0, "outcomes": {},
        "real_option": {"symbol": "AAPL261231C00100000", "strike": 100.0,
                         "expiry": "2026-12-31", "dte": 150, "entry_pending": False,
                         "entry_date": entry_date,
                         "bid": 5.0, "ask": 5.4, "mid": 5.2, "iv": 0.3, "delta": 0.5,
                         "open_interest": 500, "quote_ts": "t"},
    })
    entry_dt = datetime.strptime(entry_date, "%Y-%m-%d")
    monkeypatch.setattr(cl, "_fetch_history_batch", lambda t, p: {})
    called = {"n": 0}
    def spy(symbols):
        called["n"] += 1
        return {"AAPL261231C00100000": {"bid": 7.0, "ask": 7.4, "mid": 7.2, "ts": "t"}}
    monkeypatch.setattr(cl.market_snapshot, "fetch_option_quotes", spy)
    monkeypatch.setattr(cl, "_now_utc", lambda: NON_REGULAR_NOW)
    # Autouse-Fixture patcht us_market_session default-mäßig auf "regular" —
    # für diesen Test die tatsächliche Session-Logik (bzw. "post") nutzen.
    monkeypatch.setattr(cl.market_snapshot, "us_market_session", lambda ts: "post")

    today = (entry_dt + timedelta(days=5)).strftime("%Y-%m-%d")
    cl.update_outcomes(today, root=ledger_root)

    row = _read_jsonl(ledger_root / "2026-08.jsonl")[0]
    assert "real_opt_ret_5d" not in row["outcomes"]
    assert called["n"] == 0


def test_update_outcomes_real_opt_ret_uses_intrinsic_when_expired(monkeypatch, ledger_root):
    entry_date = "2026-08-01"
    _write_ledger_line(ledger_root, "2026-08", {
        "date": entry_date, "ticker": "AAPL", "pipeline_version": "v8.3",
        "config_hash": "abc", "status": "proposed", "reject_stage": None,
        "reject_reason": None, "direction": "BULLISH", "features": {},
        "entry_price": 100.0, "outcomes": {},
        "real_option": {"symbol": "AAPL260806C00100000", "strike": 100.0,
                         "expiry": "2026-08-06", "dte": 5, "entry_pending": False,
                         "entry_date": entry_date,
                         "bid": 3.0, "ask": 3.4, "mid": 3.2, "iv": 0.3, "delta": 0.5,
                         "open_interest": 500, "quote_ts": "t"},
    })
    entry_dt = datetime.strptime(entry_date, "%Y-%m-%d")
    # Underlying schließt am Verfallstag (08-06, d=5) bei 110 -> Intrinsic Call = 10
    hist = {"AAPL": [(entry_dt + timedelta(days=d), 100.0 + d * 2) for d in range(0, 10)]}
    monkeypatch.setattr(cl, "_fetch_history_batch", lambda t, p: hist)
    called = {"n": 0}
    def fake_quotes(symbols):
        called["n"] += 1
        return {}
    monkeypatch.setattr(cl.market_snapshot, "fetch_option_quotes", fake_quotes)
    monkeypatch.setattr(cl, "_now_utc", lambda: REGULAR_NOW)

    today = (entry_dt + timedelta(days=10)).strftime("%Y-%m-%d")  # nach Verfall
    cl.update_outcomes(today, root=ledger_root)

    row = _read_jsonl(ledger_root / "2026-08.jsonl")[0]
    outcomes = row["outcomes"]
    # Intrinsic = max(110-100, 0) = 10 -> real_opt_ret_5d = 10/3.4 - 1
    assert outcomes["real_opt_ret_5d"] == pytest.approx(10.0 / 3.4 - 1.0, abs=1e-4)
    assert outcomes["real_opt_ret_mid_5d"] == pytest.approx(10.0 / 3.2 - 1.0, abs=1e-4)


def test_update_outcomes_real_opt_ret_flags_late_mark(monkeypatch, ledger_root):
    entry_date = "2026-08-01"
    _write_ledger_line(ledger_root, "2026-08", {
        "date": entry_date, "ticker": "AAPL", "pipeline_version": "v8.3",
        "config_hash": "abc", "status": "proposed", "reject_stage": None,
        "reject_reason": None, "direction": "BULLISH", "features": {},
        "entry_price": 100.0, "outcomes": {},
        "real_option": {"symbol": "AAPL261231C00100000", "strike": 100.0,
                         "expiry": "2026-12-31", "dte": 150, "entry_pending": False,
                         "entry_date": entry_date,
                         "bid": 5.0, "ask": 5.4, "mid": 5.2, "iv": 0.3, "delta": 0.5,
                         "open_interest": 500, "quote_ts": "t"},
    })
    entry_dt = datetime.strptime(entry_date, "%Y-%m-%d")
    monkeypatch.setattr(cl, "_fetch_history_batch", lambda t, p: {})
    monkeypatch.setattr(cl.market_snapshot, "fetch_option_quotes",
                         lambda symbols: {"AAPL261231C00100000": {"bid": 7.0, "ask": 7.4, "mid": 7.2, "ts": "t"}})
    monkeypatch.setattr(cl, "_now_utc", lambda: REGULAR_NOW)

    # 20 Tage nach Entry: ret_5d wäre seit 15 Tagen fällig gewesen (>7 Tage spät).
    today = (entry_dt + timedelta(days=20)).strftime("%Y-%m-%d")
    cl.update_outcomes(today, root=ledger_root)

    row = _read_jsonl(ledger_root / "2026-08.jsonl")[0]
    assert row["late_mark"] is True


def test_update_outcomes_real_opt_ret_never_raises_on_bad_data(monkeypatch, ledger_root):
    entry_date = "2026-08-01"
    _write_ledger_line(ledger_root, "2026-08", {
        "date": entry_date, "ticker": "AAPL", "pipeline_version": "v8.3",
        "config_hash": "abc", "status": "proposed", "reject_stage": None,
        "reject_reason": None, "direction": "BULLISH", "features": {},
        "entry_price": 100.0, "outcomes": {},
        "real_option": {"symbol": "AAPL261231C00100000", "strike": 100.0,
                         "expiry": "2026-12-31", "dte": 150, "entry_pending": False,
                         "entry_date": entry_date,
                         "bid": None, "ask": None, "mid": None, "iv": 0.3, "delta": 0.5,
                         "open_interest": 500, "quote_ts": "t"},
    })
    entry_dt = datetime.strptime(entry_date, "%Y-%m-%d")
    monkeypatch.setattr(cl, "_fetch_history_batch", lambda t, p: {})
    monkeypatch.setattr(cl, "_now_utc", lambda: REGULAR_NOW)

    def boom(symbols):
        raise RuntimeError("network down")
    monkeypatch.setattr(cl.market_snapshot, "fetch_option_quotes", boom)

    today = (entry_dt + timedelta(days=5)).strftime("%Y-%m-%d")
    cl.update_outcomes(today, root=ledger_root)  # must not raise
    row = _read_jsonl(ledger_root / "2026-08.jsonl")[0]
    assert "real_opt_ret_5d" not in row["outcomes"]


# ── P0-B: real_option Entry-Fill (nur in regulärer Session) ─────────────────

def test_update_outcomes_fills_pending_real_option_entry_in_regular_session(monkeypatch, ledger_root):
    _write_ledger_line(ledger_root, "2026-08", {
        "date": "2026-08-01", "ticker": "AAPL", "pipeline_version": "v8.3",
        "config_hash": "abc", "status": "proposed", "reject_stage": None,
        "reject_reason": None, "direction": "BULLISH", "features": {},
        "entry_price": None, "entry_basis": "next_open", "session": "pre",
        "signal_timestamp": "2026-08-01T13:00:00+00:00", "outcomes": {},
        "real_option": {"symbol": "AAPL261231C00100000", "strike": 100.0,
                         "expiry": "2026-12-31", "dte": 150, "entry_pending": True,
                         "bid": None, "ask": None, "mid": None, "iv": 0.3, "delta": 0.5,
                         "open_interest": 500, "quote_ts": "stale",
                         "snapshot_quote": {"bid": 5.0, "ask": 5.4, "mid": 5.2, "quote_ts": "stale"}},
    })
    monkeypatch.setattr(cl, "_fetch_history_open_batch", lambda tickers, period_days: {})
    monkeypatch.setattr(cl, "_fetch_history_batch", lambda tickers, period_days: {})
    calls = []
    def fake_quotes(symbols):
        calls.append(list(symbols))
        return {"AAPL261231C00100000": {"bid": 6.0, "ask": 6.4, "mid": 6.2, "ts": "live"}}
    monkeypatch.setattr(cl.market_snapshot, "fetch_option_quotes", fake_quotes)
    monkeypatch.setattr(cl, "_now_utc", lambda: REGULAR_NOW)

    cl.update_outcomes("2026-08-20", root=ledger_root)

    row = _read_jsonl(ledger_root / "2026-08.jsonl")[0]
    ro = row["real_option"]
    assert ro["entry_pending"] is False
    assert ro["bid"] == pytest.approx(6.0) and ro["ask"] == pytest.approx(6.4) and ro["mid"] == pytest.approx(6.2)
    assert ro["entry_date"] == "2026-08-20"
    assert "entry_filled_at" in ro
    assert len(calls) == 1 and calls[0] == ["AAPL261231C00100000"]


def test_update_outcomes_pending_real_option_not_filled_outside_regular_session(monkeypatch, ledger_root):
    _write_ledger_line(ledger_root, "2026-08", {
        "date": "2026-08-01", "ticker": "AAPL", "pipeline_version": "v8.3",
        "config_hash": "abc", "status": "proposed", "reject_stage": None,
        "reject_reason": None, "direction": "BULLISH", "features": {},
        "entry_price": None, "entry_basis": "next_open", "session": "pre",
        "signal_timestamp": "2026-08-01T13:00:00+00:00", "outcomes": {},
        "real_option": {"symbol": "AAPL261231C00100000", "strike": 100.0,
                         "expiry": "2026-12-31", "dte": 150, "entry_pending": True,
                         "bid": None, "ask": None, "mid": None, "iv": 0.3, "delta": 0.5,
                         "open_interest": 500, "quote_ts": "stale",
                         "snapshot_quote": {"bid": 5.0, "ask": 5.4, "mid": 5.2, "quote_ts": "stale"}},
    })
    monkeypatch.setattr(cl, "_fetch_history_open_batch", lambda tickers, period_days: {})
    monkeypatch.setattr(cl, "_fetch_history_batch", lambda tickers, period_days: {})
    called = {"n": 0}
    def spy(symbols):
        called["n"] += 1
        return {"AAPL261231C00100000": {"bid": 6.0, "ask": 6.4, "mid": 6.2, "ts": "live"}}
    monkeypatch.setattr(cl.market_snapshot, "fetch_option_quotes", spy)
    monkeypatch.setattr(cl, "_now_utc", lambda: NON_REGULAR_NOW)
    monkeypatch.setattr(cl.market_snapshot, "us_market_session", lambda ts: "post")

    cl.update_outcomes("2026-08-20", root=ledger_root)

    row = _read_jsonl(ledger_root / "2026-08.jsonl")[0]
    ro = row["real_option"]
    assert ro["entry_pending"] is True
    assert ro["bid"] is None
    assert called["n"] == 0


def test_update_outcomes_real_option_horizon_counted_from_entry_date_not_signal_date(monkeypatch, ledger_root):
    """Horizonte für real_opt_ret laufen ab dem TATSÄCHLICH gefüllten
    Entry-Datum, nicht ab dem ursprünglichen Signal-Datum (das könnte Tage
    früher gewesen sein, wenn der Kontrakt pre-market gewählt wurde)."""
    _write_ledger_line(ledger_root, "2026-08", {
        "date": "2026-08-01", "ticker": "AAPL", "pipeline_version": "v8.3",
        "config_hash": "abc", "status": "proposed", "reject_stage": None,
        "reject_reason": None, "direction": "BULLISH", "features": {},
        "entry_price": None, "entry_basis": "next_open", "session": "pre",
        "signal_timestamp": "2026-08-01T13:00:00+00:00", "outcomes": {},
        "real_option": {"symbol": "AAPL261231C00100000", "strike": 100.0,
                         "expiry": "2026-12-31", "dte": 150, "entry_pending": True,
                         "bid": None, "ask": None, "mid": None, "iv": 0.3, "delta": 0.5,
                         "open_interest": 500, "quote_ts": "stale",
                         "snapshot_quote": {"bid": 5.0, "ask": 5.4, "mid": 5.2, "quote_ts": "stale"}},
    })
    monkeypatch.setattr(cl, "_fetch_history_open_batch", lambda tickers, period_days: {})
    monkeypatch.setattr(cl, "_fetch_history_batch", lambda tickers, period_days: {})
    monkeypatch.setattr(cl.market_snapshot, "fetch_option_quotes",
                         lambda symbols: {"AAPL261231C00100000": {"bid": 6.0, "ask": 6.4, "mid": 6.2, "ts": "live"}})
    monkeypatch.setattr(cl, "_now_utc", lambda: REGULAR_NOW)  # füllt am 2026-08-20

    # Fill-Lauf: Entry wird erst jetzt (08-20) tatsächlich gebucht.
    cl.update_outcomes("2026-08-20", root=ledger_root)
    row = _read_jsonl(ledger_root / "2026-08.jsonl")[0]
    assert row["real_option"]["entry_date"] == "2026-08-20"
    assert "real_opt_ret_5d" not in row["outcomes"]  # noch kein Horizont seit dem Fill verstrichen

    # 5 Tage nach dem FILL (nicht nach dem ursprünglichen Signal 08-01) markieren:
    monkeypatch.setattr(cl.market_snapshot, "fetch_option_quotes",
                         lambda symbols: {"AAPL261231C00100000": {"bid": 7.0, "ask": 7.4, "mid": 7.2, "ts": "live2"}})
    cl.update_outcomes("2026-08-25", root=ledger_root)
    row = _read_jsonl(ledger_root / "2026-08.jsonl")[0]
    assert "real_opt_ret_5d" in row["outcomes"]
    assert row["outcomes"]["real_opt_ret_5d"] == pytest.approx(7.0 / 6.4 - 1.0, abs=1e-4)


# ── P2: signal_id / event_id, Dedup je Event ─────────────────────────────────

def test_flush_assigns_signal_id_and_event_id(monkeypatch, ledger_root):
    monkeypatch.setattr(cl, "_fetch_prices_batch", lambda tickers: {t: 100.0 for t in tickers})
    cl.start_run("2026-09-26")
    cl.note("AAPL", stage="universe")
    cl.flush(reports_dir_root=ledger_root)
    row = _read_jsonl(ledger_root / "2026-09.jsonl")[0]
    assert isinstance(row["signal_id"], str) and len(row["signal_id"]) == 32
    assert isinstance(row["event_id"], str) and len(row["event_id"]) == 12


def test_flush_same_event_twice_dedups_to_one_row(monkeypatch, ledger_root):
    monkeypatch.setattr(cl, "_fetch_prices_batch", lambda tickers: {t: 100.0 for t in tickers})

    cl.start_run("2026-09-26")
    cl.note("AAPL", stage="deep_analysis", direction="BULLISH", event_key="FDA approval PDUFA")
    cl.flush(reports_dir_root=ledger_root)

    # Zweiter Lauf am selben Tag, dasselbe Event (z.B. Rerun der Pipeline)
    cl.start_run("2026-09-26")
    cl.note("AAPL", stage="deep_analysis", direction="BULLISH", event_key="FDA approval PDUFA")
    cl.flush(reports_dir_root=ledger_root)

    rows = _read_jsonl(ledger_root / "2026-09.jsonl")
    assert len(rows) == 1


def test_flush_two_distinct_events_same_ticker_same_day_both_kept(monkeypatch, ledger_root):
    monkeypatch.setattr(cl, "_fetch_prices_batch", lambda tickers: {t: 100.0 for t in tickers})

    cl.start_run("2026-09-26")
    cl.note("AAPL", stage="deep_analysis", direction="BULLISH", event_key="FDA approval PDUFA")
    cl.flush(reports_dir_root=ledger_root)

    cl.start_run("2026-09-26")
    cl.note("AAPL", stage="deep_analysis", direction="BEARISH", event_key="Guidance cut Q3")
    cl.flush(reports_dir_root=ledger_root)

    rows = _read_jsonl(ledger_root / "2026-09.jsonl")
    assert len(rows) == 2
    event_ids = {r["event_id"] for r in rows}
    assert len(event_ids) == 2


def test_flush_event_id_fallback_without_event_key_is_ticker_and_date(monkeypatch, ledger_root):
    """Ohne event_key (bisheriges Verhalten): Fallback sha1(ticker+date) —
    ein zweiter Lauf ohne event_key am selben Tag ist weiterhin ein Dup."""
    monkeypatch.setattr(cl, "_fetch_prices_batch", lambda tickers: {t: 100.0 for t in tickers})

    cl.start_run("2026-09-26")
    cl.note("AAPL", stage="universe")
    cl.flush(reports_dir_root=ledger_root)

    cl.start_run("2026-09-26")
    cl.note("AAPL", stage="universe")
    cl.flush(reports_dir_root=ledger_root)

    rows = _read_jsonl(ledger_root / "2026-09.jsonl")
    assert len(rows) == 1


def test_flush_records_code_sha_and_model_ids(monkeypatch, ledger_root):
    monkeypatch.setenv("GITHUB_SHA", "0123456789abcdef")
    monkeypatch.setattr(cl, "_fetch_prices_batch", lambda tickers: {})
    cl.start_run("2026-09-25")
    cl.note("XOM", stage="universe")
    cl.flush(ledger_root)
    row = _read_jsonl(ledger_root / "2026-09.jsonl")[0]
    assert row["code_sha"] == "0123456789ab"
    assert row["model_ids"].get("models.deep_analysis")


# ── P0-1: Signal-Identität — mehrere Events desselben Tickers im selben Lauf ─

def test_note_two_events_same_ticker_creates_two_signals_in_memory():
    """Zwei verschiedene event_keys für denselben Ticker im selben Lauf
    dürfen NICHT auf ein Signal kollabieren (das war der P0-1-Bug)."""
    cl.start_run("2026-09-26")
    cl.note("AAPL", stage="universe", sigma_30d=0.3)
    cl.note("AAPL", stage="deep_analysis", direction="BULLISH", event_key="FDA approval PDUFA")
    cl.note("AAPL", stage="deep_analysis", direction="BEARISH", event_key="Guidance cut Q3")

    signals = cl._state["entries"]["AAPL"]
    assert len(signals) == 2
    event_keys = {s["event_key"] for s in signals}
    assert event_keys == {"FDA approval PDUFA", "Guidance cut Q3"}
    signal_ids = {s["signal_id"] for s in signals}
    assert len(signal_ids) == 2  # distinct IDs
    # Das ticker-weite Feld (vor dem ersten event_key notiert) wurde in
    # beide Signale übernommen (copy-on-branch):
    for s in signals:
        assert s["features"]["sigma_30d"] == pytest.approx(0.3)
    directions = {s["direction"] for s in signals}
    assert directions == {"BULLISH", "BEARISH"}


def test_note_same_event_key_twice_updates_one_signal():
    cl.start_run("2026-09-26")
    cl.note("AAPL", stage="deep_analysis", direction="BULLISH", event_key="FDA approval PDUFA")
    cl.note("AAPL", stage="rl_scoring", event_key="FDA approval PDUFA", final_mc_hit_rate=0.6)

    signals = cl._state["entries"]["AAPL"]
    assert len(signals) == 1
    assert signals[0]["stage"] == "rl_scoring"
    assert signals[0]["features"]["final_mc_hit_rate"] == pytest.approx(0.6)


def test_note_without_event_key_updates_all_signals_of_ticker():
    cl.start_run("2026-09-26")
    cl.note("AAPL", stage="deep_analysis", direction="BULLISH", event_key="Event A")
    cl.note("AAPL", stage="deep_analysis", direction="BEARISH", event_key="Event B")
    # Ein späterer note()-Aufruf OHNE event_key (z.B. ein Feature, das für
    # den Ticker als Ganzes gilt) muss BEIDE Signale aktualisieren.
    cl.note("AAPL", trade_score=77)

    signals = cl._state["entries"]["AAPL"]
    assert len(signals) == 2
    for s in signals:
        assert s["features"]["trade_score"] == 77


def test_mark_rejected_without_event_key_applies_to_all_signals():
    cl.start_run("2026-09-26")
    cl.note("AAPL", stage="deep_analysis", direction="BULLISH", event_key="Event A")
    cl.note("AAPL", stage="deep_analysis", direction="BEARISH", event_key="Event B")
    cl.mark_rejected("AAPL", "mismatch_overreaction")

    signals = cl._state["entries"]["AAPL"]
    assert all(s["status"] == "rejected" for s in signals)
    assert all(s["reject_reason"] == "mismatch_overreaction" for s in signals)


def test_mark_passed_with_event_key_applies_only_to_that_signal():
    cl.start_run("2026-09-26")
    cl.note("AAPL", stage="deep_analysis", direction="BULLISH", event_key="Event A")
    cl.note("AAPL", stage="deep_analysis", direction="BEARISH", event_key="Event B")
    cl.mark_passed("AAPL", event_key="Event A")

    signals = {s["event_key"]: s for s in cl._state["entries"]["AAPL"]}
    assert signals["Event A"]["status"] == "proposed"
    assert signals["Event B"]["status"] == "seen"


def test_flush_two_events_same_ticker_single_run_two_rows_distinct_ids(monkeypatch, ledger_root):
    monkeypatch.setattr(cl, "_fetch_prices_batch", lambda tickers: {t: 100.0 for t in tickers})

    cl.start_run("2026-09-26")
    cl.note("AAPL", stage="deep_analysis", direction="BULLISH", event_key="FDA approval PDUFA")
    cl.note("AAPL", stage="deep_analysis", direction="BEARISH", event_key="Guidance cut Q3")
    cl.flush(reports_dir_root=ledger_root)

    rows = _read_jsonl(ledger_root / "2026-09.jsonl")
    assert len(rows) == 2
    assert all(r["ticker"] == "AAPL" for r in rows)
    assert len({r["signal_id"] for r in rows}) == 2
    assert len({r["event_id"] for r in rows}) == 2
    directions = {r["direction"] for r in rows}
    assert directions == {"BULLISH", "BEARISH"}


# ── P0-2: real_strategy (Produktions-Strategie-Counterfactual) ──────────────

_LONG_CALL_RAW = {
    "symbol": "AAPL_LONG", "strike": 100.0, "expiry": "2026-11-20", "dte": 55,
    "bid": 5.0, "ask": 5.4, "mid": 5.2, "iv": 0.4, "delta": 0.5,
    "open_interest": 500, "quote_ts": "t",
}


def test_build_real_strategy_default_long_without_iv_history(monkeypatch):
    monkeypatch.setattr(cl, "_resolve_candidate_iv_rank", lambda ticker, iv: None)
    e = {"direction": "BULLISH", "features": {}}
    rs = cl._build_real_strategy(e, "AAPL", 100.0, _LONG_CALL_RAW)
    assert rs["strategy_source"] == "default_long"
    assert rs["strategy"] == "LONG_CALL"
    assert len(rs["legs"]) == 1
    assert rs["legs"][0]["side"] == "long"


def test_build_real_strategy_bull_call_spread_when_iv_rank_high(monkeypatch):
    monkeypatch.setattr(cl, "_resolve_candidate_iv_rank", lambda ticker, iv: 80.0)
    monkeypatch.setattr(
        cl.market_snapshot, "select_spread_short_leg",
        lambda ticker, expiry, opt_type, strike: {
            "symbol": "AAPL_SHORT", "strike": strike * 1.10, "bid": 2.0, "ask": 2.2, "mid": 2.1,
        },
    )
    e = {"direction": "BULLISH", "features": {}}
    rs = cl._build_real_strategy(e, "AAPL", 100.0, _LONG_CALL_RAW)
    assert rs["strategy"] == "BULL_CALL_SPREAD"
    assert rs["strategy_source"] == "iv_history_fallback"
    assert len(rs["legs"]) == 2
    assert {l["side"] for l in rs["legs"]} == {"long", "short"}
    assert rs["net_debit_entry"] == pytest.approx(5.4 - 2.0)
    assert rs["net_mid_entry"] == pytest.approx(5.2 - 2.1)


def test_build_real_strategy_bear_put_spread_uses_put_leg(monkeypatch):
    monkeypatch.setattr(cl, "_resolve_candidate_iv_rank", lambda ticker, iv: 80.0)
    captured = {}

    def fake_short_leg(ticker, expiry, opt_type, strike):
        captured["opt_type"] = opt_type
        return {"symbol": "AAPL_SHORT_PUT", "strike": strike * 1.10, "bid": 1.5, "ask": 1.7, "mid": 1.6}

    monkeypatch.setattr(cl.market_snapshot, "select_spread_short_leg", fake_short_leg)
    e = {"direction": "BEARISH", "features": {}}
    long_put_raw = dict(_LONG_CALL_RAW, symbol="AAPL_LONG_PUT")
    rs = cl._build_real_strategy(e, "AAPL", 100.0, long_put_raw)
    assert rs["strategy"] == "BEAR_PUT_SPREAD"
    assert captured["opt_type"] == "put"


def test_build_real_strategy_falls_back_to_long_when_short_leg_illiquid(monkeypatch):
    monkeypatch.setattr(cl, "_resolve_candidate_iv_rank", lambda ticker, iv: 80.0)
    monkeypatch.setattr(
        cl.market_snapshot, "select_spread_short_leg",
        lambda *a, **k: {"symbol": "X", "strike": 110.0, "bid": 0, "ask": 0.1, "mid": 0.05},
    )
    e = {"direction": "BULLISH", "features": {}}
    rs = cl._build_real_strategy(e, "AAPL", 100.0, _LONG_CALL_RAW)
    assert rs["strategy_source"] == "spread_no_liquidity"
    assert rs["strategy"] == "LONG_CALL"
    assert len(rs["legs"]) == 1


def test_build_real_strategy_uses_dealer_gamma_and_vix_from_features(monkeypatch):
    """dealer_gamma_state/vix_structure kommen aus den vom pipeline.py
    genoteten Features (siehe pipeline.py Stufe 10 note()-Aufruf)."""
    captured = {}

    def fake_choose_strategy(iv_rank, is_bullish, dealer_gamma_state, vix_structure):
        captured["dealer_gamma_state"] = dealer_gamma_state
        captured["vix_structure"] = vix_structure
        return "LONG_CALL", 52.0, "test"

    monkeypatch.setattr(cl, "_resolve_candidate_iv_rank", lambda ticker, iv: 30.0)
    import modules.options_designer as od
    monkeypatch.setattr(od, "choose_strategy", fake_choose_strategy)

    e = {
        "direction": "BULLISH",
        "features": {
            "dealer_gamma_state": {"data_available": True, "net_gamma_sign": "negative"},
            "vix_structure": "backwardation",
        },
    }
    cl._build_real_strategy(e, "AAPL", 100.0, _LONG_CALL_RAW)
    assert captured["dealer_gamma_state"] == {"data_available": True, "net_gamma_sign": "negative"}
    assert captured["vix_structure"] == "backwardation"


def test_build_real_strategy_none_without_direction():
    e = {"direction": None, "features": {}}
    assert cl._build_real_strategy(e, "AAPL", 100.0, _LONG_CALL_RAW) is None


# ── Review-Fix: real_strategy iv_rank — replicated (echte 1y-Closes) ────────
#
# Ground-truth-Reihenfolge: production > replicated > iv_history_fallback >
# default_long. Diese Tests decken "replicated" (echte Closes/Term-Structure,
# via compute_iv_rank_components) und "production" (Ground Truth aus
# pipeline.py) sowie strategy_match ab.

def _high_recent_vol_closes(n=300):
    """Deterministische Kursreihe mit stark erhöhter Vola in den letzten 40
    Tagen (niedrige Vola davor) → rv_score nahe 100 (siehe
    compute_iv_rank_components: Percentile-Rank der aktuellen rollierenden
    21d-RV innerhalb der 1y-Verteilung)."""
    closes = []
    price = 100.0
    for i in range(n):
        amp   = 0.001 if i < n - 40 else 0.05
        drift = amp * math.sin(i)
        price *= (1 + drift)
        closes.append(round(price, 4))
    return closes


IV_SPREAD_GATE_FOR_TEST = 52.0  # entspricht options_designer.IV_SPREAD_GATE


def test_build_real_strategy_replicated_spread_when_rv_high(monkeypatch):
    """Mit echten (hier: synthetischen, aber realistisch geformten) 1y-Closes
    UND hoher aktueller Realized-Vol wird — wie in der Produktion — ein
    Spread statt eines Long-Legs gewählt (strategy_source='replicated')."""
    closes = _high_recent_vol_closes()
    monkeypatch.setattr(
        cl.market_snapshot, "select_spread_short_leg",
        lambda ticker, expiry, opt_type, strike: {
            "symbol": "AAPL_SHORT", "strike": strike * 1.10, "bid": 2.0, "ask": 2.2, "mid": 2.1,
        },
    )
    e = {"direction": "BULLISH", "features": {}}
    rs = cl._build_real_strategy(e, "AAPL", 100.0, _LONG_CALL_RAW, closes=closes, term_point2=None)

    assert rs["strategy_source"] == "replicated"
    assert rs["strategy"] == "BULL_CALL_SPREAD"
    assert rs["iv_rank"] > IV_SPREAD_GATE_FOR_TEST
    assert rs["replicated_strategy"] == "BULL_CALL_SPREAD"
    assert rs["iv_rank_components"]["n_term_points"] == 1  # nur Long-Leg-Punkt
    assert len(rs["legs"]) == 2


def test_build_real_strategy_replicated_uses_term_point2_when_available(monkeypatch):
    closes = _high_recent_vol_closes()
    e = {"direction": "BULLISH", "features": {}}
    rs = cl._build_real_strategy(
        e, "AAPL", 100.0, _LONG_CALL_RAW, closes=closes, term_point2=(10, 0.3),
    )
    assert rs["iv_rank_components"]["n_term_points"] == 2


def test_build_real_strategy_replicated_falls_back_when_too_few_closes(monkeypatch):
    """<60 Closes → replizierter iv_rank nicht bestimmbar → nächster
    Fallback (iv_history_fallback, hier ohne Historie → default_long)."""
    e = {"direction": "BULLISH", "features": {}}
    rs = cl._build_real_strategy(e, "AAPL", 100.0, _LONG_CALL_RAW, closes=[100.0] * 10)
    assert rs["strategy_source"] == "default_long"
    assert "replicated_strategy" not in rs


def test_build_real_strategy_production_ground_truth_overrides_replicated(monkeypatch):
    """Wenn pipeline.py im selben Lauf einen Trade-Proposal erzeugt hat
    (production_strategy/production_iv_rank genotet), hat das Vorrang vor
    dem replizierten Wert — aber beide werden gespeichert (strategy_match)."""
    closes = _high_recent_vol_closes()  # würde allein ein SPREAD ergeben
    e = {
        "direction": "BULLISH",
        "features": {
            "production_strategy": "LONG_CALL",     # Produktion hat NICHT gespreadet
            "production_iv_rank":  35.0,
        },
    }
    rs = cl._build_real_strategy(e, "AAPL", 100.0, _LONG_CALL_RAW, closes=closes, term_point2=None)

    assert rs["strategy_source"] == "production"
    assert rs["strategy"] == "LONG_CALL"
    assert rs["iv_rank"] == pytest.approx(35.0)
    assert rs["production_strategy"] == "LONG_CALL"
    assert rs["production_iv_rank"] == pytest.approx(35.0)
    # Der replizierte Wert wird TROTZDEM gespeichert (für spätere Auswertung):
    assert rs["replicated_strategy"] == "BULL_CALL_SPREAD"
    assert rs["strategy_match"] is False  # production LONG_CALL != replicated SPREAD


def test_build_real_strategy_strategy_match_true_when_agreeing(monkeypatch):
    closes = _high_recent_vol_closes()
    e = {
        "direction": "BULLISH",
        "features": {
            "production_strategy": "BULL_CALL_SPREAD",  # stimmt mit replicated überein
            "production_iv_rank":  85.0,
        },
    }
    monkeypatch.setattr(
        cl.market_snapshot, "select_spread_short_leg",
        lambda ticker, expiry, opt_type, strike: {
            "symbol": "AAPL_SHORT", "strike": strike * 1.10, "bid": 2.0, "ask": 2.2, "mid": 2.1,
        },
    )
    rs = cl._build_real_strategy(e, "AAPL", 100.0, _LONG_CALL_RAW, closes=closes, term_point2=None)
    assert rs["strategy_source"] == "production"
    assert rs["strategy_match"] is True


def test_build_real_strategy_production_without_replicated_data_no_match_field(monkeypatch):
    """Keine Closes (z.B. History-Download fehlgeschlagen) → kein replizierter
    Wert → strategy_match wird gar nicht erst gesetzt (None/fehlt)."""
    e = {
        "direction": "BULLISH",
        "features": {"production_strategy": "LONG_CALL", "production_iv_rank": 40.0},
    }
    rs = cl._build_real_strategy(e, "AAPL", 100.0, _LONG_CALL_RAW, closes=None)
    assert rs["strategy_source"] == "production"
    assert "strategy_match" not in rs
    assert "replicated_strategy" not in rs


# ── P0-2: real_strategy Entry-Fill (Multi-Leg Session-Gating) ───────────────

def test_fill_real_strategy_entries_requires_regular_session(monkeypatch):
    monkeypatch.setattr(cl.market_snapshot, "us_market_session", lambda ts: "pre")
    row = {"real_strategy": {"entry_pending": True, "legs": [
        {"symbol": "LONG", "side": "long", "bid": None, "ask": None, "mid": None},
    ]}}
    changed = cl._fill_real_strategy_entries([row])
    assert changed is False
    assert row["real_strategy"]["entry_pending"] is True


def test_fill_real_strategy_entries_partial_leg_quote_keeps_pending(monkeypatch):
    monkeypatch.setattr(cl.market_snapshot, "us_market_session", lambda ts: "regular")
    monkeypatch.setattr(
        cl.market_snapshot, "fetch_option_quotes",
        lambda symbols: {"LONG": {"bid": 5.0, "ask": 5.4, "mid": 5.2}},  # SHORT-Quote fehlt
    )
    row = {"real_strategy": {"entry_pending": True, "legs": [
        {"symbol": "LONG", "side": "long", "bid": None, "ask": None, "mid": None},
        {"symbol": "SHORT", "side": "short", "bid": None, "ask": None, "mid": None},
    ]}}
    changed = cl._fill_real_strategy_entries([row])
    assert changed is False
    assert row["real_strategy"]["entry_pending"] is True


def test_fill_real_strategy_entries_full_fill_sets_net_debit(monkeypatch):
    monkeypatch.setattr(cl.market_snapshot, "us_market_session", lambda ts: "regular")
    now = datetime(2026, 9, 26, 15, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(cl, "_now_utc", lambda: now)
    monkeypatch.setattr(
        cl.market_snapshot, "fetch_option_quotes",
        lambda symbols: {
            "LONG":  {"bid": 5.0, "ask": 5.4, "mid": 5.2},
            "SHORT": {"bid": 2.0, "ask": 2.2, "mid": 2.1},
        },
    )
    row = {"real_strategy": {"entry_pending": True, "legs": [
        {"symbol": "LONG", "side": "long", "bid": None, "ask": None, "mid": None},
        {"symbol": "SHORT", "side": "short", "bid": None, "ask": None, "mid": None},
    ]}}
    changed = cl._fill_real_strategy_entries([row])
    assert changed is True
    rs = row["real_strategy"]
    assert rs["entry_pending"] is False
    assert rs["net_debit_entry"] == pytest.approx(5.4 - 2.0)
    assert rs["net_mid_entry"] == pytest.approx(5.2 - 2.1)
    assert rs["entry_date"] == "2026-09-26"


def test_fill_real_strategy_entries_single_leg_uses_ask_and_mid(monkeypatch):
    """LONG_CALL/LONG_PUT (kein Spread, strategy_source=default_long/computed
    ohne Short-Leg) hat nur einen Leg — net_debit_entry/net_mid_entry
    entsprechen dann einfach Ask/Mid des Long-Legs."""
    monkeypatch.setattr(cl.market_snapshot, "us_market_session", lambda ts: "regular")
    monkeypatch.setattr(
        cl.market_snapshot, "fetch_option_quotes",
        lambda symbols: {"LONG": {"bid": 5.0, "ask": 5.4, "mid": 5.2}},
    )
    row = {"real_strategy": {"entry_pending": True, "legs": [
        {"symbol": "LONG", "side": "long", "bid": None, "ask": None, "mid": None},
    ]}}
    changed = cl._fill_real_strategy_entries([row])
    assert changed is True
    rs = row["real_strategy"]
    assert rs["net_debit_entry"] == pytest.approx(5.4)
    assert rs["net_mid_entry"] == pytest.approx(5.2)


# ── P0-2: real_strategy Marks (Exit-Mathematik, Floor bei 0, Expiry) ────────

def test_fill_real_strategy_marks_spread_floors_conservative_at_zero(monkeypatch):
    monkeypatch.setattr(cl.market_snapshot, "us_market_session", lambda ts: "regular")
    monkeypatch.setattr(
        cl.market_snapshot, "fetch_option_quotes",
        lambda symbols: {
            "LONG":  {"bid": 1.0, "ask": 1.2, "mid": 1.1},
            "SHORT": {"bid": 1.5, "ask": 1.6, "mid": 1.55},  # short ask > long bid → negativ vor Floor
        },
    )
    row = {
        "ticker": "AAPL", "direction": "BULLISH",
        "real_strategy": {
            "entry_pending": False, "entry_date": "2026-08-01", "expiry": "2027-01-01",
            "net_debit_entry": 3.0, "net_mid_entry": 2.9,
            "legs": [
                {"symbol": "LONG",  "strike": 100.0, "side": "long",  "bid": None, "ask": None, "mid": None},
                {"symbol": "SHORT", "strike": 110.0, "side": "short", "bid": None, "ask": None, "mid": None},
            ],
        },
        "outcomes": {},
    }
    changed = cl._fill_real_strategy_marks([row], datetime(2026, 8, 6))
    assert changed is True
    assert row["outcomes"]["real_strat_ret_5d"] == pytest.approx(0.0 / 3.0 - 1.0, abs=1e-4)  # gefloort bei 0
    exit_mid = 1.1 - 1.55
    assert row["outcomes"]["real_strat_ret_mid_5d"] == pytest.approx(exit_mid / 2.9 - 1.0, abs=1e-4)
    assert row["outcomes"]["real_strat_mark_date_5d"] == "2026-08-06"


def test_fill_real_strategy_marks_bull_call_spread_positive_exit(monkeypatch):
    monkeypatch.setattr(cl.market_snapshot, "us_market_session", lambda ts: "regular")
    monkeypatch.setattr(
        cl.market_snapshot, "fetch_option_quotes",
        lambda symbols: {
            "LONG":  {"bid": 6.0, "ask": 6.2, "mid": 6.1},
            "SHORT": {"bid": 1.0, "ask": 1.1, "mid": 1.05},
        },
    )
    row = {
        "ticker": "AAPL", "direction": "BULLISH",
        "real_strategy": {
            "entry_pending": False, "entry_date": "2026-08-01", "expiry": "2027-01-01",
            "net_debit_entry": 3.0, "net_mid_entry": 2.9,
            "legs": [
                {"symbol": "LONG",  "strike": 100.0, "side": "long",  "bid": None, "ask": None, "mid": None},
                {"symbol": "SHORT", "strike": 110.0, "side": "short", "bid": None, "ask": None, "mid": None},
            ],
        },
        "outcomes": {},
    }
    cl._fill_real_strategy_marks([row], datetime(2026, 8, 6))
    exit_conservative = 6.0 - 1.1
    exit_mid = 6.1 - 1.05
    assert row["outcomes"]["real_strat_ret_5d"] == pytest.approx(exit_conservative / 3.0 - 1.0, abs=1e-4)
    assert row["outcomes"]["real_strat_ret_mid_5d"] == pytest.approx(exit_mid / 2.9 - 1.0, abs=1e-4)


def test_fill_real_strategy_marks_uses_intrinsic_after_expiry(monkeypatch):
    monkeypatch.setattr(cl.market_snapshot, "us_market_session", lambda ts: "regular")
    monkeypatch.setattr(cl.market_snapshot, "fetch_option_quotes", lambda symbols: {})
    monkeypatch.setattr(
        cl, "_fetch_history_batch",
        lambda tickers, period_days: {"AAPL": [(datetime(2026, 8, 1), 115.0)]},
    )
    row = {
        "ticker": "AAPL", "direction": "BULLISH",
        "real_strategy": {
            "entry_pending": False, "entry_date": "2026-07-01", "expiry": "2026-08-01",
            "net_debit_entry": 3.0, "net_mid_entry": 2.9,
            "legs": [
                {"symbol": "LONG",  "strike": 100.0, "side": "long",  "bid": None, "ask": None, "mid": None},
                {"symbol": "SHORT", "strike": 110.0, "side": "short", "bid": None, "ask": None, "mid": None},
            ],
        },
        "outcomes": {},
    }
    changed = cl._fill_real_strategy_marks([row], datetime(2026, 8, 20))
    assert changed is True
    # Long-Intrinsic=115-100=15, Short-Intrinsic=115-110=5 → net=10
    assert row["outcomes"]["real_strat_ret_45d"] == pytest.approx(10.0 / 3.0 - 1.0, abs=1e-4)


def test_fill_real_strategy_marks_bearish_put_spread_intrinsic_at_expiry(monkeypatch):
    monkeypatch.setattr(cl.market_snapshot, "us_market_session", lambda ts: "regular")
    monkeypatch.setattr(cl.market_snapshot, "fetch_option_quotes", lambda symbols: {})
    monkeypatch.setattr(
        cl, "_fetch_history_batch",
        lambda tickers, period_days: {"AAPL": [(datetime(2026, 8, 1), 85.0)]},
    )
    row = {
        "ticker": "AAPL", "direction": "BEARISH",
        "real_strategy": {
            "entry_pending": False, "entry_date": "2026-07-01", "expiry": "2026-08-01",
            "net_debit_entry": 3.0, "net_mid_entry": 2.9,
            "legs": [
                {"symbol": "LONG",  "strike": 100.0, "side": "long",  "bid": None, "ask": None, "mid": None},
                {"symbol": "SHORT", "strike": 90.0,  "side": "short", "bid": None, "ask": None, "mid": None},
            ],
        },
        "outcomes": {},
    }
    changed = cl._fill_real_strategy_marks([row], datetime(2026, 8, 20))
    assert changed is True
    # Long-Put-Intrinsic=100-85=15, Short-Put-Intrinsic=90-85=5 → net=10
    assert row["outcomes"]["real_strat_ret_45d"] == pytest.approx(10.0 / 3.0 - 1.0, abs=1e-4)


def test_fill_real_strategy_marks_single_leg_no_floor_needed(monkeypatch):
    """Single-Leg (kein Spread): exit_conservative ist einfach long_bid."""
    monkeypatch.setattr(cl.market_snapshot, "us_market_session", lambda ts: "regular")
    monkeypatch.setattr(
        cl.market_snapshot, "fetch_option_quotes",
        lambda symbols: {"LONG": {"bid": 7.0, "ask": 7.2, "mid": 7.1}},
    )
    row = {
        "ticker": "AAPL", "direction": "BULLISH",
        "real_strategy": {
            "entry_pending": False, "entry_date": "2026-08-01", "expiry": "2027-01-01",
            "net_debit_entry": 6.4, "net_mid_entry": 6.2,
            "legs": [
                {"symbol": "LONG", "strike": 100.0, "side": "long", "bid": None, "ask": None, "mid": None},
            ],
        },
        "outcomes": {},
    }
    cl._fill_real_strategy_marks([row], datetime(2026, 8, 6))
    assert row["outcomes"]["real_strat_ret_5d"] == pytest.approx(7.0 / 6.4 - 1.0, abs=1e-4)
    assert row["outcomes"]["real_strat_ret_mid_5d"] == pytest.approx(7.1 / 6.2 - 1.0, abs=1e-4)


def test_fill_real_strategy_marks_outside_regular_session_noop(monkeypatch):
    monkeypatch.setattr(cl.market_snapshot, "us_market_session", lambda ts: "closed")
    row = {
        "ticker": "AAPL", "direction": "BULLISH",
        "real_strategy": {
            "entry_pending": False, "entry_date": "2026-08-01", "expiry": "2027-01-01",
            "net_debit_entry": 3.0, "net_mid_entry": 2.9,
            "legs": [{"symbol": "LONG", "strike": 100.0, "side": "long", "bid": None, "ask": None, "mid": None}],
        },
        "outcomes": {},
    }
    changed = cl._fill_real_strategy_marks([row], datetime(2026, 8, 6))
    assert changed is False
    assert "real_strat_ret_5d" not in row["outcomes"]


# ── P1: neutraler Options-Snapshot-Budget (keine "erste N") ─────────────────

class _FakeUUID4:
    def __init__(self, hexval):
        self.hex = hexval


def test_flush_budget_selection_is_not_first_n_and_deterministic(monkeypatch, ledger_root):
    """Die Budget-Auswahl darf NICHT einfach die ersten N Kandidaten in
    Verarbeitungsreihenfolge nehmen — sie muss deterministisch (seeded by
    date) aber unabhängig von der Reihenfolge sein."""
    fixed_ids = [
        "00000000000000000000000000000001",
        "00000000000000000000000000000026",
        "0000000000000000000000000000004b",
        "00000000000000000000000000000070",
        "00000000000000000000000000000095",
    ]
    id_iter = iter(fixed_ids)
    monkeypatch.setattr(cl.uuid, "uuid4", lambda: _FakeUUID4(next(id_iter)))

    monkeypatch.setattr(cl.market_snapshot, "us_market_session", lambda ts: "regular")
    monkeypatch.setattr(
        cl.market_snapshot, "fetch_underlying_quotes",
        lambda tickers: {t: {"bid": 100.0, "ask": 100.4, "mid": 100.2,
                              "last": 100.1, "prev_close": 99.0, "open": 100.0,
                              "quote_ts": "t", "source": "tradier"} for t in tickers},
    )
    monkeypatch.setattr(cl, "_fetch_prices_batch", lambda tickers: {t: 100.0 for t in tickers})
    monkeypatch.setattr(cl, "_max_option_snapshots", lambda: 2)
    monkeypatch.setattr(
        cl.market_snapshot, "select_contract",
        lambda ticker, direction, dte_floor, spot: {
            "symbol": f"{ticker}_OPT", "strike": 100.0, "expiry": "2026-11-20", "dte": 55,
            "bid": 5.0, "ask": 5.4, "mid": 5.2, "iv": 0.3, "delta": 0.5,
            "open_interest": 500, "quote_ts": "t",
        },
    )
    monkeypatch.setattr(cl, "_build_real_strategy", lambda *a, **k: None)

    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]  # note()-Reihenfolge

    cl.start_run("2026-09-26")
    for t in tickers:
        cl.note(t, stage="deep_analysis", direction="BULLISH", ttm="4-8 Wochen")
    signal_ids = {t: cl._state["entries"][t][0]["signal_id"] for t in tickers}
    cl.flush(reports_dir_root=ledger_root)

    rows = {r["ticker"]: r for r in _read_jsonl(ledger_root / "2026-09.jsonl")}
    selected = {t for t, r in rows.items() if r["snapshot_selected"]}
    assert len(selected) == 2

    # NICHT die ersten beiden in note()-Aufrufreihenfolge:
    assert selected != set(tickers[:2])

    # Deterministisch: entspricht exakt sha1(date+signal_id)-Ranking.
    ranked = sorted(
        tickers,
        key=lambda t: hashlib.sha1(f"2026-09-26:{signal_ids[t]}".encode("utf-8")).hexdigest(),
    )
    assert selected == set(ranked[:2])

    # Erneuter Flush-Lauf (gleicher Tag, gleiche Signal-IDs) liefert
    # dieselbe Auswahl (deterministisch, nicht zufällig pro Aufruf):
    id_iter2 = iter(fixed_ids)
    monkeypatch.setattr(cl.uuid, "uuid4", lambda: _FakeUUID4(next(id_iter2)))
    cl.start_run("2026-09-26")
    for t in tickers:
        cl.note(t, stage="deep_analysis", direction="BULLISH", ttm="4-8 Wochen")
    cl.flush(reports_dir_root=ledger_root)
    rows2 = _read_jsonl(ledger_root / "2026-09.jsonl")
    assert len(rows2) == 5  # dedup (gleiche event_id je Ticker/Tag) — kein zweiter Satz Zeilen


# ── P1: rl.model_sha256 in model_ids ────────────────────────────────────────

def test_compute_model_ids_includes_rl_model_sha256(monkeypatch, tmp_path):
    model_file = tmp_path / "ppo.zip"
    model_file.write_bytes(b"fake-ppo-weights")
    expected = hashlib.sha256(b"fake-ppo-weights").hexdigest()[:12]

    fake_config_yaml = f"""
models:
  deep_analysis: "claude-sonnet-test"
rl:
  model_path: "{model_file.as_posix()}"
finbert:
  model_name: "finbert-test"
"""
    monkeypatch.setattr(cl.Path, "read_text", lambda self, *a, **k: fake_config_yaml if self.name == "config.yaml" else "")

    ids = cl._compute_model_ids()
    assert ids.get("rl.model_sha256") == expected
    assert ids.get("rl.model_path") == model_file.as_posix()


def test_compute_model_ids_no_sha256_when_model_file_missing(monkeypatch, tmp_path):
    missing_path = tmp_path / "does_not_exist.zip"
    fake_config_yaml = f"""
rl:
  model_path: "{missing_path.as_posix()}"
"""
    monkeypatch.setattr(cl.Path, "read_text", lambda self, *a, **k: fake_config_yaml if self.name == "config.yaml" else "")

    ids = cl._compute_model_ids()
    assert "rl.model_sha256" not in ids
    assert ids.get("rl.model_path") == missing_path.as_posix()

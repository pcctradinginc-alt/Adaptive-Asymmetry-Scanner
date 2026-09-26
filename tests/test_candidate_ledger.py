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

import json
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
    assert cl._state["entries"]["AAPL"]["signal_timestamp"] == "2026-09-26T14:03:07+00:00"


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
    assert cl._state["entries"]["AAPL"]["signal_timestamp"] == "2026-09-26T10:00:00+00:00"


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
    assert row["real_option"] == fake_contract
    assert len(calls) == 1
    assert calls[0][0] == "AAPL" and calls[0][1] == "BULLISH"
    assert calls[0][3] == pytest.approx(100.2)  # spot = entry_price (quote_mid)


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
    without_option = [t for t, r in rows.items() if r.get("real_option_skip_reason") == "max_snapshots_reached"]
    assert len(with_option) == 2
    assert len(without_option) == 1
    assert len(calls) == 2  # Budget begrenzt auch die API-Calls selbst


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

def test_update_outcomes_real_opt_ret_bid_ask_math(monkeypatch, ledger_root):
    entry_date = "2026-08-01"
    _write_ledger_line(ledger_root, "2026-08", {
        "date": entry_date, "ticker": "AAPL", "pipeline_version": "v8.3",
        "config_hash": "abc", "status": "proposed", "reject_stage": None,
        "reject_reason": None, "direction": "BULLISH", "features": {},
        "entry_price": 100.0, "outcomes": {},
        "real_option": {"symbol": "AAPL261231C00100000", "strike": 100.0,
                         "expiry": "2026-12-31", "dte": 150,
                         "bid": 5.0, "ask": 5.4, "mid": 5.2, "iv": 0.3, "delta": 0.5,
                         "open_interest": 500, "quote_ts": "t"},
    })
    entry_dt = datetime.strptime(entry_date, "%Y-%m-%d")
    monkeypatch.setattr(cl, "_fetch_history_batch", lambda t, p: {})
    monkeypatch.setattr(cl.market_snapshot, "fetch_option_quotes",
                         lambda symbols: {"AAPL261231C00100000": {"bid": 7.0, "ask": 7.4, "mid": 7.2, "ts": "t"}})

    today = (entry_dt + timedelta(days=5)).strftime("%Y-%m-%d")
    cl.update_outcomes(today, root=ledger_root)

    row = _read_jsonl(ledger_root / "2026-08.jsonl")[0]
    outcomes = row["outcomes"]
    assert outcomes["real_opt_ret_5d"] == pytest.approx(7.0 / 5.4 - 1.0, abs=1e-4)      # exit_bid/entry_ask
    assert outcomes["real_opt_ret_mid_5d"] == pytest.approx(7.2 / 5.2 - 1.0, abs=1e-4)  # exit_mid/entry_mid
    assert "real_opt_mark_date_5d" in outcomes
    assert "late_mark" not in row


def test_update_outcomes_real_opt_ret_uses_intrinsic_when_expired(monkeypatch, ledger_root):
    entry_date = "2026-08-01"
    _write_ledger_line(ledger_root, "2026-08", {
        "date": entry_date, "ticker": "AAPL", "pipeline_version": "v8.3",
        "config_hash": "abc", "status": "proposed", "reject_stage": None,
        "reject_reason": None, "direction": "BULLISH", "features": {},
        "entry_price": 100.0, "outcomes": {},
        "real_option": {"symbol": "AAPL260806C00100000", "strike": 100.0,
                         "expiry": "2026-08-06", "dte": 5,
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
                         "expiry": "2026-12-31", "dte": 150,
                         "bid": 5.0, "ask": 5.4, "mid": 5.2, "iv": 0.3, "delta": 0.5,
                         "open_interest": 500, "quote_ts": "t"},
    })
    entry_dt = datetime.strptime(entry_date, "%Y-%m-%d")
    monkeypatch.setattr(cl, "_fetch_history_batch", lambda t, p: {})
    monkeypatch.setattr(cl.market_snapshot, "fetch_option_quotes",
                         lambda symbols: {"AAPL261231C00100000": {"bid": 7.0, "ask": 7.4, "mid": 7.2, "ts": "t"}})

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
                         "expiry": "2026-12-31", "dte": 150,
                         "bid": None, "ask": None, "mid": None, "iv": 0.3, "delta": 0.5,
                         "open_interest": 500, "quote_ts": "t"},
    })
    entry_dt = datetime.strptime(entry_date, "%Y-%m-%d")
    monkeypatch.setattr(cl, "_fetch_history_batch", lambda t, p: {})

    def boom(symbols):
        raise RuntimeError("network down")
    monkeypatch.setattr(cl.market_snapshot, "fetch_option_quotes", boom)

    today = (entry_dt + timedelta(days=5)).strftime("%Y-%m-%d")
    cl.update_outcomes(today, root=ledger_root)  # must not raise
    row = _read_jsonl(ledger_root / "2026-08.jsonl")[0]
    assert "real_opt_ret_5d" not in row["outcomes"]


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

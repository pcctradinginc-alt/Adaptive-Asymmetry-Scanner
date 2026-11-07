"""1-Tages-Label nie aus dem Schluss des Einstiegstags (Bewertung 2026-11-06).

Live: ret_1d von Freitags-Signalen = Schluss desselben Freitags (Freitag + 1 Kalendertag = Samstag, "letzter Schluss
<= Zieltag"). 1.776 von 8.381 ret_1d-Labels maßen so nur den Rest der Einstiegssitzung (Streuung 0,7 % statt 3,2 %).
Neue Labels nehmen den ersten Schluss nach dem Einstiegstag und tragen dessen Datum; alte Zeilen bleiben unverändert
und werden von Lesern (factor_monitor) über one_day_label_is_same_session ausgefiltert.
"""
import json
from datetime import datetime, timedelta

import pytest

from modules import candidate_ledger as cl
from modules import factor_monitor as fm


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    cl._state.update({"date": None, "entries": {}, "flushed": False})
    monkeypatch.setattr(cl.market_snapshot, "fetch_underlying_quotes", lambda tickers: {})
    monkeypatch.setattr(cl, "_fetch_history_batch", lambda tickers, period_days: {})
    yield
    cl._state.update({"date": None, "entries": {}, "flushed": False})


def _weekday_bars(start: datetime, days: int, base: float = 100.0):
    """Tagesbalken nur an Werktagen (wie yfinance), Schluss = base + Kalendertag-Index."""
    return [(start + timedelta(days=d), base + d) for d in range(days) if (start + timedelta(days=d)).weekday() < 5]


def _row(date: str, ticker: str = "AAPL", **outcomes) -> dict:
    return {"date": date, "ticker": ticker, "pipeline_version": "v8.4", "config_hash": "abc", "status": "rejected",
            "reject_stage": "universe", "reject_reason": "prescreen_no", "direction": None, "features": {},
            "entry_price": 100.0, "outcomes": dict(outcomes)}


def _write(root, rows):
    root.mkdir(parents=True, exist_ok=True)
    with open(root / f"{rows[0]['date'][:7]}.jsonl", "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)


def _read(root, month):
    return [json.loads(line) for line in (root / f"{month}.jsonl").read_text().splitlines() if line.strip()]


def test_friday_one_day_label_uses_monday_close_once_monday_is_complete(monkeypatch, tmp_path):
    root = tmp_path / "ledger"
    friday = datetime(2026, 10, 30)
    _write(root, [_row("2026-10-30")])
    hist = {"AAPL": _weekday_bars(friday, 10)}          # Fr 100, Mo 103, Di 104, ...
    monkeypatch.setattr(cl, "_fetch_history_batch", lambda tickers, period_days: hist)

    # Montag (Sitzung läuft noch): kein Label, auch nicht aus dem Freitagsschluss
    monday_bar_only = {"AAPL": _weekday_bars(friday, 4)}
    monkeypatch.setattr(cl, "_fetch_history_batch", lambda tickers, period_days: monday_bar_only)
    cl.update_outcomes("2026-11-02", root=root)
    assert "ret_1d" not in _read(root, "2026-10")[0]["outcomes"]

    monkeypatch.setattr(cl, "_fetch_history_batch", lambda tickers, period_days: hist)
    cl.update_outcomes("2026-11-03", root=root)
    oc = _read(root, "2026-10")[0]["outcomes"]
    assert oc["ret_1d"] == pytest.approx(0.03)           # Montagsschluss 103 / Einstieg 100
    assert oc[cl.ONE_DAY_CLOSE_KEY] == "2026-11-02"
    assert oc["mfe"] == pytest.approx(0.03)              # derselbe Zeitraum wie ret_1d


def test_midweek_one_day_label_is_unchanged(monkeypatch, tmp_path):
    root = tmp_path / "ledger"
    wednesday = datetime(2026, 10, 28)
    _write(root, [_row("2026-10-28")])
    monkeypatch.setattr(cl, "_fetch_history_batch", lambda tickers, period_days: {"AAPL": _weekday_bars(wednesday, 12)})
    cl.update_outcomes("2026-10-30", root=root)
    oc = _read(root, "2026-10")[0]["outcomes"]
    assert oc["ret_1d"] == pytest.approx(0.01) and oc[cl.ONE_DAY_CLOSE_KEY] == "2026-10-29"


def test_longer_horizons_keep_the_on_or_before_rule(monkeypatch, tmp_path):
    root = tmp_path / "ledger"
    monday = datetime(2026, 10, 26)
    _write(root, [_row("2026-10-26")])
    monkeypatch.setattr(cl, "_fetch_history_batch", lambda tickers, period_days: {"AAPL": _weekday_bars(monday, 30)})
    cl.update_outcomes("2026-11-20", root=root)
    oc = _read(root, "2026-10")[0]["outcomes"]
    assert oc["ret_5d"] == pytest.approx(0.04)           # Mo + 5 = Sa -> Freitagsschluss (Index 4), wie bisher
    assert oc["ret_20d"] == pytest.approx(0.18)          # Mo + 20 = So -> Freitag (Index 18)
    assert "ret_5d_close_date" not in oc


@pytest.mark.parametrize("date, outcomes, expected", [
    ("2026-10-30", {"ret_1d": 0.001}, True),                                       # Freitag, alte Regel
    ("2026-11-25", {"ret_1d": 0.001}, True),                                       # Vortag Thanksgiving
    ("2026-10-29", {"ret_1d": 0.02}, False),                                       # Donnerstag
    ("2026-10-30", {"ret_1d": 0.03, "ret_1d_close_date": "2026-11-02"}, False),    # Freitag, neue Regel
    ("2026-10-30", {}, False),                                                     # kein Label
])
def test_same_session_detection(date, outcomes, expected):
    assert cl.one_day_label_is_same_session(_row(date, **outcomes)) is expected


def test_factor_monitor_drops_same_session_one_day_labels(tmp_path):
    ledger = tmp_path / "ledger"
    _write(ledger, [_row("2026-10-29", "A", ret_1d=0.02, ret_5d=0.03),
                    _row("2026-10-30", "B", ret_1d=0.001, ret_5d=0.04),
                    _row("2026-10-30", "C", ret_1d=0.03, ret_1d_close_date="2026-11-02", ret_5d=0.05)])
    events = {e["ticker"]: e["outcomes"] for e in fm.load_events(history_path=tmp_path / "none.json",
                                                                 ledger_dir=ledger)}
    assert events["A"]["gross_1d"] == 0.02
    assert "gross_1d" not in events["B"] and events["B"]["gross_5d"] == 0.04
    assert events["C"]["gross_1d"] == 0.03


def test_short_history_never_substitutes_a_later_close(monkeypatch, tmp_path):
    """Beginnt die geladene Historie erst nach dem Zieltag, bleibt das Label leer (wie bisher) – kein späterer Ersatz."""
    root = tmp_path / "ledger"
    friday = datetime(2026, 10, 30)
    _write(root, [_row("2026-10-30")])
    late_only = {"AAPL": [b for b in _weekday_bars(friday, 40) if b[0] >= friday + timedelta(days=25)]}
    monkeypatch.setattr(cl, "_fetch_history_batch", lambda tickers, period_days: late_only)
    cl.update_outcomes("2026-12-10", root=root)
    oc = _read(root, "2026-10")[0]["outcomes"]
    assert "ret_1d" not in oc and "ret_5d" not in oc and "ret_20d" not in oc

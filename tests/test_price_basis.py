"""Preisbasis-Prüfung (Live 2026-10, CTVA-Abspaltung am 01.10.) – nur SYNTHETISCHE Fixtures.

Yahoo bereinigt die Kurshistorie nach Split/Abspaltung rückwirkend (auch Close ohne auto_adjust). Ledger, die einen
Einstiegspreis zum Signalzeitpunkt speichern und später gegen die heutige Historie rechnen, erhielten dadurch
Schein-Renditen (Kandidaten-Ledger CTVA ret_1d +0,84 / opt_ret_1d +9,45; Shadow-Ledger −84 % auf 20/45/60 T).
Geprüft: Erkennung über gespeicherten Preis vs. Schluss am Einstiegstag, keine weiteren Horizonte, nur widersprüchliche
gefüllte Werte werden ungültig (nachvollziehbar), echte Kursstürze bleiben, Shadow-Korrektur append-only + idempotent.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta

import pytest

from modules import candidate_ledger as cl
from modules import shadow_ledger as sl

K = 0.162                      # rückwirkender Bereinigungsfaktor der Historie (CTVA: ~12,6 / 77,6)
EX = datetime(2026, 10, 1)     # Stichtag der Kapitalmaßnahme


def adjusted_history(start: datetime, days: int, raw_before: float = 77.6, after: float = 12.57) -> list:
    """Heutige (rückwirkend bereinigte) Historie: vor dem Stichtag raw*K, danach unverändert – keine Lücke."""
    return [(start + timedelta(days=d), round((raw_before * K) if start + timedelta(days=d) < EX else after, 4))
            for d in range(days)]


def write(root, rows):
    root.mkdir(parents=True, exist_ok=True)
    (root / "2026-09.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def read(root):
    return [json.loads(ln) for ln in (root / "2026-09.jsonl").read_text().splitlines()]


def row(day, ticker="CTVA", direction="BEARISH", entry=77.625, outcomes=None):
    return {"date": day, "ticker": ticker, "direction": direction, "status": "rejected", "entry_price": entry,
            "entry_basis": "quote_mid", "features": {}, "outcomes": outcomes or {}}


def test_basis_change_invalidates_contradicting_values_and_stops_horizons(tmp_path, monkeypatch):
    root = tmp_path / "ledger"
    # 30.09. BEARISH: ret_1d wurde gegen den gespeicherten Rohpreis gerechnet (Live-Wert)
    write(root, [row("2026-09-30", outcomes={"ret_1d": 0.8381, "opt_ret_1d": 9.4534, "mfe": 0.8464, "mae": -0.0003})])
    calls = []

    def fetch(tickers, period_days):
        calls.append(tuple(tickers))
        return {"CTVA": adjusted_history(datetime(2026, 9, 20), 30)}
    monkeypatch.setattr(cl, "_fetch_history_batch", fetch)
    cl.update_outcomes("2026-10-17", root=root)
    oc = read(root)[0]["outcomes"]
    assert "ret_1d" not in oc and "opt_ret_1d" not in oc and "mfe" not in oc       # nie mehr als Outcome lesbar
    assert not any(k.startswith("ret_") for k in oc)                               # keine weiteren Horizonte
    pb = oc["price_basis"]
    assert pb["status"] == "CHANGED" and pb["factor"] == pytest.approx(K, abs=0.01) and pb["invalidated"] == ["ret_1d"]
    inv = read(root)[0]["outcomes_invalidated"]
    assert inv["reason"] == "PRICE_BASIS_CHANGED" and inv["ret_1d"] == 0.8381 and inv["opt_ret_1d"] == 9.4534
    # idempotent: zweiter Lauf ändert nichts und lädt für die Zeile keine Historie mehr
    calls.clear()
    before = (root / "2026-09.jsonl").read_text()
    cl.update_outcomes("2026-10-18", root=root)
    assert (root / "2026-09.jsonl").read_text() == before and calls == []


def test_values_consistent_with_history_are_kept(tmp_path, monkeypatch):
    """29.09.: ret_1d (Ziel 30.09., vor dem Stichtag, damals korrekt) bleibt; ret_5d über den Stichtag wird ungültig."""
    root = tmp_path / "ledger"
    write(root, [row("2026-09-29", direction=None, entry=78.0, outcomes={"ret_1d": -0.0051, "ret_5d": -0.8472})])
    hist = adjusted_history(datetime(2026, 9, 20), 30)
    hist = [(d, round(78.0 * K, 4) if d == datetime(2026, 9, 29) else c) for d, c in hist]
    monkeypatch.setattr(cl, "_fetch_history_batch", lambda t, p: {"CTVA": hist})
    cl.update_outcomes("2026-10-17", root=root)
    r = read(root)[0]
    assert r["outcomes"]["ret_1d"] == -0.0051                                      # konsistent -> bleibt
    assert "ret_5d" not in r["outcomes"] and r["outcomes_invalidated"]["ret_5d"] == -0.8472
    assert r["outcomes"]["price_basis"]["invalidated"] == ["ret_5d"]


def test_genuine_crash_is_not_a_basis_change(tmp_path, monkeypatch):
    """Echter Kurssturz NACH dem Einstieg: Schluss am Einstiegstag passt zum gespeicherten Preis -> normale Rendite."""
    root = tmp_path / "ledger"
    write(root, [row("2026-09-01", ticker="XYZ", direction="BULLISH", entry=100.0)])
    d0 = datetime(2026, 9, 1)
    hist = [(d0 + timedelta(days=d), 101.0 if d == 0 else 55.0) for d in range(30)]
    monkeypatch.setattr(cl, "_fetch_history_batch", lambda t, p: {"XYZ": hist})
    cl.update_outcomes("2026-09-25", root=root)
    oc = read(root)[0]["outcomes"]
    assert "price_basis" not in oc and oc["ret_5d"] == pytest.approx(-0.45) and oc["ret_20d"] == pytest.approx(-0.45)


def test_other_rows_unaffected(tmp_path, monkeypatch):
    root = tmp_path / "ledger"
    write(root, [row("2026-09-30"), row("2026-09-30", ticker="ABC", direction="BULLISH", entry=50.0)])
    d0 = datetime(2026, 9, 20)
    monkeypatch.setattr(cl, "_fetch_history_batch", lambda t, p: {
        "CTVA": adjusted_history(d0, 30), "ABC": [(d0 + timedelta(days=d), 40.0 + d) for d in range(30)]})
    cl.update_outcomes("2026-10-17", root=root)
    ctva, abc = read(root)
    assert ctva["outcomes"]["price_basis"]["status"] == "CHANGED" and "ret_1d" not in ctva["outcomes"]
    assert "price_basis" not in abc["outcomes"] and abc["outcomes"]["ret_1d"] == pytest.approx(0.02, abs=1e-4)


# ── Shadow-Ledger ────────────────────────────────────────────────────────────
SNAP = {"ticker": "CTVA", "entry_date": "2026-08-05", "reject_reason": "final_mc_survivor", "strategy": "LONG_CALL",
        "simulation": {"current_price": 79.8}, "deep_analysis": {"direction": "BULLISH"}, "features": {},
        "option": {"strike": 80, "expiry": "2027-01-15", "ask": 5.0}, "outcome": None}


def shadow_history(level: float):
    def fn(ticker, start, end):
        s, e = date.fromisoformat(start), date.fromisoformat(end)
        return [((s + timedelta(days=k)).isoformat(), level * (1 + 0.001 * k)) for k in range((e - s).days + 1)]
    return fn


def test_shadow_outcome_uses_entry_close_when_snapshot_on_other_basis():
    rec = {"entry_date": "2026-08-05", "snapshot": SNAP, "direction": "BULLISH", "strategy": "LONG_CALL"}
    closes = [("2026-08-05", 12.93), ("2026-08-25", 12.46)]
    out = sl._underlying_outcome(rec, closes, "2026-08-25")
    assert out["ref_price_source"] == "entry_day_close" and out["price_basis_factor"] == pytest.approx(0.162, abs=1e-3)
    assert out["underlying_return"] == pytest.approx(12.46 / 12.93 - 1, abs=1e-6)
    same = sl._underlying_outcome(rec, [("2026-08-05", 79.0), ("2026-08-25", 70.0)], "2026-08-25")
    assert same["ref_price_source"] == "snapshot.simulation.current_price" and "price_basis_factor" not in same
    gap = sl._underlying_outcome(rec, [("2026-08-20", 40.0), ("2026-08-25", 41.0)], "2026-08-25")   # Datenlücke
    assert gap["ref_price_source"] == "snapshot.simulation.current_price" and "price_basis_factor" not in gap


def test_shadow_correction_appends_and_replaces_contaminated_evaluation(tmp_path):
    led = tmp_path / "ledger"
    sl.register([SNAP], "test", date(2026, 8, 5), led)
    sid = sl.shadow_id(SNAP)
    bad = {"shadow_id": sid, "horizon": 20, "maturity_date": "2026-08-25", "kind": "EVALUATED",
           "evaluated_at": "2026-10-12T22:32:44", "underlying_ref_price": 79.8,
           "ref_price_source": "snapshot.simulation.current_price", "underlying_close_at_maturity": 12.4606,
           "last_close_date": "2026-08-25", "underlying_return": -0.843852, "mfe": -0.843852, "mae": -0.857991,
           "option_return": None, "option_method": None, "option_note": None, "option_reliable": False,
           "outcome_quality": "UNDERLYING_ONLY"}
    sl._append_events(led, [bad], date(2026, 10, 12))
    n = sl.correct_price_basis(date(2026, 10, 17), shadow_history(12.93), led)
    assert n == 1
    final = sl._index(sl._events(led))[sid]["h"]["20"]["final"]
    assert final["corrects"] == "2026-10-12T22:32:44" and final["correction_reason"] == "PRICE_BASIS_CHANGED"
    assert final["kind"] == "EVALUATED" and abs(final["underlying_return"]) < 0.05
    assert final["corrected_values"]["underlying_return"] == -0.843852
    assert sum(1 for e in sl._events(led) if e.get("shadow_id") == sid) == 2      # append-only, Original bleibt
    assert sl.correct_price_basis(date(2026, 10, 18), shadow_history(12.93), led) == 0   # idempotent


def test_shadow_correction_keeps_genuine_moves_and_old_evaluations(tmp_path):
    led = tmp_path / "ledger"
    sl.register([SNAP], "test", date(2026, 8, 5), led)
    sid = sl.shadow_id(SNAP)
    ev = {"shadow_id": sid, "horizon": 20, "maturity_date": "2026-08-25", "kind": "EVALUATED",
          "evaluated_at": "2026-10-12T22:32:44", "underlying_ref_price": 79.8,
          "ref_price_source": "snapshot.simulation.current_price", "underlying_return": -0.4, "mfe": -0.3, "mae": -0.45}
    sl._append_events(led, [ev], date(2026, 10, 12))
    assert sl.correct_price_basis(date(2026, 10, 17), shadow_history(79.5), led) == 0    # gleiche Basis: echter Sturz
    assert sl._index(sl._events(led))[sid]["h"]["20"]["final"]["underlying_return"] == -0.4
    calls = []

    def spy(*a):
        calls.append(a)
        return shadow_history(12.93)(*a)
    assert sl.correct_price_basis(date(2027, 1, 30), spy, led) == 0 and calls == []     # außerhalb des Rückblicks


def test_extreme_filled_value_is_checked_once_without_pending_horizon(tmp_path, monkeypatch):
    """Zeile ohne offenen Horizont, aber mit extremer gefüllter Rendite: sofort geprüft (nicht erst beim nächsten
    fälligen Horizont), bei gleicher Basis einmal als OK markiert -> kein täglicher Neuabruf."""
    root = tmp_path / "ledger"
    write(root, [row("2026-09-29", ticker="XYZ", direction="BULLISH", entry=100.0,
                     outcomes={"ret_1d": -0.41, "ret_5d": -0.43})])
    d0 = datetime(2026, 9, 25)
    calls = []

    def fetch(tickers, period_days):
        calls.append(tuple(tickers))
        return {"XYZ": [(d0 + timedelta(days=d), 100.5 if d <= 4 else 58.0) for d in range(25)]}
    monkeypatch.setattr(cl, "_fetch_history_batch", fetch)
    cl.update_outcomes("2026-10-12", root=root)                   # 13 T: ret_20d noch nicht fällig
    oc = read(root)[0]["outcomes"]
    assert calls == [("XYZ",)] and oc["price_basis"]["status"] == "OK" and oc["ret_5d"] == -0.43
    calls.clear()
    cl.update_outcomes("2026-10-13", root=root)
    assert calls == []

"""V5 (2026-10-18): Preis-Exits nur auf zulässiger Quote, Spread-Stop auf dem Fair Value.

Auslöser KDP 15.10.: Stop um 22:33 UTC auf der Combo-Quote von 15:59:50 ET (Long 2,25/4,00, Short 1,30/2,05 ->
ausführbar −87,9 %, Fair Value −12,1 %); der nächste RTH-Fill lag bei −18,2 %."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import feedback

UTC = timezone.utc
KDP = {"ticker": "KDP", "strategy": "BULL_CALL_SPREAD", "entry_date": "2026-10-15", "entry_debit": 1.65,
       "simulation": {"current_price": 33.81},
       "option": {"strike": 33.0, "expiry": "2027-03-19", "spread_leg": {"strike": 36.0}}}
LC = {"ticker": "MO", "strategy": "LONG_CALL", "entry_date": "2026-09-24", "entry_debit": 7.35,
      "simulation": {"current_price": 69.17}, "option": {"strike": 65.0, "expiry": "2027-03-19", "ask": 7.35}}
IN_WINDOW = datetime(2026, 10, 15, 17, 54, tzinfo=UTC)          # 13:54 ET, Donnerstag
AFTER_CLOSE = datetime(2026, 10, 15, 22, 33, tzinfo=UTC)        # 18:33 ET


def _trade(base):
    return {**base, "option": {**base["option"]}}


def _spread_meta(lb, la, sb, sa, qt):
    return {"method": "spread_quote", "spread_quotes": {"long": {"bid": lb, "ask": la, "quote_time": qt},
                                                        "short": {"bid": sb, "ask": sa, "quote_time": qt}}}


def _dec(trade, outcome, meta, now):
    return feedback.decide_exit(trade, outcome, meta, now.replace(tzinfo=None), now)   # today naiv wie in main()


def _ago(now, minutes):
    return (now - timedelta(minutes=minutes)).isoformat()


def test_kdp_after_hours_quote_defers_stop():
    t = _trade(KDP)
    meta = _spread_meta(2.25, 4.0, 1.3, 2.05, "2026-10-15T19:59:50+00:00")
    executable = (2.25 - 2.05) / 1.65 - 1                                      # −87,9 % wie im Produktionslauf
    assert feedback.check_exit_rules(t, executable, AFTER_CLOSE.replace(tzinfo=None)) == "stop_loss"  # alte Regel hätte gestoppt
    assert _dec(t, executable, meta, AFTER_CLOSE) is None
    d = t["exit_deferral"]
    assert d["reason"] == "DECISION_OUTSIDE_WINDOW" and d["signal"] == "stop_loss" and d["days"] == ["2026-10-15"]
    assert "exit_decision" not in t


def test_kdp_in_window_fair_value_no_stop():
    t = _trade(KDP)
    meta = _spread_meta(2.9, 3.1, 1.45, 1.65, _ago(IN_WINDOW, 2))           # Monitor 15.10. 17:54 UTC
    chk = feedback.exit_quote_check(t, meta, IN_WINDOW)
    assert chk["admissible"] and chk["fair_value_outcome"] == pytest.approx(1.45 / 1.65 - 1, abs=1e-4)
    assert _dec(t, 1.25 / 1.65 - 1, meta, IN_WINDOW) is None


def test_spread_stop_on_fair_value_with_admissible_quote():
    t = _trade(KDP)
    meta = _spread_meta(1.0, 1.1, 0.35, 0.45, _ago(IN_WINDOW, 3))           # Mid 0,65 -> −60,6 %
    executable = (1.0 - 0.45) / 1.65 - 1
    assert _dec(t, executable, meta, IN_WINDOW) == "stop_loss"
    ed = t["exit_decision"]
    assert ed["basis"] == "ADMISSIBLE_QUOTE" and ed["version"] == "exit-decision-v2"
    assert ed["stop_outcome"] == pytest.approx(0.65 / 1.65 - 1, abs=1e-4) and ed["quote_check"] == "OK"


def test_wide_spread_executable_below_stop_but_fair_value_above_no_stop():
    t = _trade(KDP)
    meta = _spread_meta(1.4, 2.6, 0.9, 1.3, _ago(IN_WINDOW, 1))             # ausführbar 0,10 -> −94 %
    executable = (1.4 - 1.3) / 1.65 - 1
    assert executable < -0.5
    assert _dec(t, executable, meta, IN_WINDOW) is None   # Fair Value 0,90 -> −45 %


def test_take_profit_stays_on_executable_value():
    t = _trade(KDP)                                                          # TP = 70 % des Max-Gewinns: +57,3 %
    meta = _spread_meta(4.0, 4.6, 1.3, 1.5, _ago(IN_WINDOW, 1))
    executable = (4.0 - 1.5) / 1.65 - 1                                      # +51,5 % < TP, Mid wäre +72,7 %
    assert _dec(t, executable, meta, IN_WINDOW) is None


def test_deferral_cap_forces_decision_after_three_trading_days():
    t = _trade(KDP)
    for day in ("2026-10-15", "2026-10-16"):
        now = datetime.fromisoformat(day + "T22:33:00+00:00")
        meta = _spread_meta(0.5, 0.6, 0.3, 0.4, day + "T19:59:50+00:00")
        assert _dec(t, -0.8, meta, now) is None
    now = datetime(2026, 10, 19, 22, 33, tzinfo=UTC)
    meta = _spread_meta(0.5, 0.6, 0.3, 0.4, "2026-10-19T19:59:50+00:00")
    assert _dec(t, -0.8, meta, now) == "stop_loss"
    assert t["exit_decision"]["basis"] == "FORCED_AFTER_DEFERRAL" and t["exit_decision"]["deferral_days"] == 3


def test_same_day_reruns_count_once_and_admissible_quote_clears_deferral():
    t = _trade(KDP)
    for minute in (0, 30):
        now = AFTER_CLOSE + timedelta(minutes=minute)
        _dec(t, -0.8, _spread_meta(0.5, 0.6, 0.3, 0.4, "2026-10-15T19:59:50+00:00"), now)
    assert t["exit_deferral"]["days"] == ["2026-10-15"]
    nxt = datetime(2026, 10, 16, 15, 0, tzinfo=UTC)
    assert _dec(t, -0.2, _spread_meta(3.3, 3.7, 1.7, 1.95, _ago(nxt, 2)), nxt) is None
    assert "exit_deferral" not in t


def test_long_call_stop_on_bid_with_fresh_rth_quote():
    t = _trade(LC)
    meta = {"method": "option_quote", "option_quote": {"bid": 3.9, "ask": 4.1, "quote_time": _ago(IN_WINDOW, 4)}}
    assert _dec(t, 3.9 / 7.35 - 1, meta, IN_WINDOW) == "stop_loss"
    assert t["exit_decision"]["stop_outcome"] == pytest.approx(3.9 / 7.35 - 1, abs=1e-4)   # Long Call: Bid-Basis


@pytest.mark.parametrize("quote_time, reason", [
    (lambda now: _ago(now, 45), "QUOTE_STALE"),                               # älter als 30 min
    (lambda now: "2026-10-15T13:00:00+00:00", "QUOTE_OUTSIDE_RTH"),           # 09:00 ET, vorbörslich
    (lambda now: None, "QUOTE_TIME_MISSING"),                                 # yfinance ohne Zeitstempel
])
def test_inadmissible_long_call_quotes_defer(quote_time, reason):
    t = _trade(LC)
    meta = {"method": "option_quote", "option_quote": {"bid": 3.0, "ask": 3.2, "quote_time": quote_time(IN_WINDOW)}}
    assert _dec(t, 3.0 / 7.35 - 1, meta, IN_WINDOW) is None
    assert t["exit_deferral"]["reason"] == reason


def test_modeled_outcome_never_triggers_price_exit_directly():
    t = _trade(LC)
    assert _dec(t, -0.6, {"method": "delta_approx"}, IN_WINDOW) is None
    assert t["exit_deferral"]["reason"] == "METHOD_DELTA_APPROX"


def test_settlement_and_time_exit_not_deferred():
    t = _trade(KDP)
    assert _dec(t, -1.0, {"method": "expiry_intrinsic"}, AFTER_CLOSE) == "stop_loss"
    assert t["exit_decision"]["basis"] == "SETTLEMENT"
    late = _trade(LC)                                                         # 50 % der Laufzeit verstrichen
    day = datetime(2027, 1, 10, 22, 0, tzinfo=UTC)
    assert _dec(late, 0.05, {"method": "option_quote"}, day) == "time_exit"
    assert late["exit_decision"]["basis"] == "NO_PRICE_SIGNAL"


def test_invalid_combo_quote_defers():
    t = _trade(KDP)
    meta = _spread_meta(2.0, 1.8, 1.0, 1.2, _ago(IN_WINDOW, 1))              # Long gekreuzt
    assert _dec(t, -0.7, meta, IN_WINDOW) is None
    assert t["exit_deferral"]["reason"] == "QUOTE_INVALID"


def test_single_leg_quote_reaches_decision_through_compute_outcome(monkeypatch):
    q = {"bid": 3.0, "ask": 3.2, "source": "tradier", "quote_time": _ago(IN_WINDOW, 1)}
    monkeypatch.setattr(feedback, "get_option_quote", lambda *a, **k: q)
    meta = {}
    out = feedback.compute_outcome(_trade(LC), 70.0, meta)
    assert out == pytest.approx(3.0 / 7.35 - 1) and meta["option_quote"] is q
    assert feedback.exit_quote_check(_trade(LC), meta, IN_WINDOW)["admissible"]

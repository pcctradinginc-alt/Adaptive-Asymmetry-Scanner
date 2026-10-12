"""Timing-Policy timing-v2 (2026-10-11): Entry/Exit nur zu zulässigen Optionsquotes, nie rückwirkend, Altbestand v1
unverändert. Synthetische Quotes prüfen nur die Zeit-/Preis-Mechanik – kein wirtschaftlicher Nachweis."""
from __future__ import annotations

import json
import random
from datetime import date, datetime, timedelta, timezone

import pytest

import feedback
from modules import candidate_ledger as cl
from modules import evidence_ledger as el
from modules import option_quotes as oq
from modules import timing_policy as tp

UTC = timezone.utc
L, S, M = "AAPL270319C00200000", "AAPL270319C00210000", "MSFT270319C00400000"


def utc(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


def quote(sym: str, at: str, bid: float, ask: float, *, quote_time: str | None = None) -> None:
    t = utc(at)
    q = int(utc(quote_time).timestamp() * 1000) if quote_time else int(t.timestamp() * 1000)
    oq.record([{"symbol": sym, "bid": bid, "ask": ask, "bid_date": q, "ask_date": q}], underlying=sym[:4].rstrip("0123456789"),
              source="test", context="test", retrieved_at=t)


@pytest.fixture(autouse=True)
def archive(tmp_path, monkeypatch):
    root = tmp_path / "quotes"
    monkeypatch.setattr(oq, "ROOT", root)
    monkeypatch.setattr("modules.market_snapshot.fetch_option_quotes",
                        lambda *a, **k: pytest.fail("kein Live-Abruf im Test"))
    return root


def trade(sym=L, *, policy="timing-v2", decision="2026-10-13T16:00:00+00:00", strategy="LONG_CALL",
          short=None, expiry="2027-03-19", ticker="AAPL"):
    opt = {"option_symbol": sym, "strike": 200.0, "expiry": expiry, "ask": 10.0, "bid": 9.6}
    if short:
        opt["spread_leg"] = {"symbol": short, "strike": 210.0}
    t = {"ticker": ticker, "entry_date": decision[:10], "strategy": strategy, "option": opt, "entry_debit": 10.0,
         "features": {}, "outcome": None}
    if policy:
        t.update(timing_policy=policy, decision_time=decision)
    return t


# ── Policy und Konfiguration ─────────────────────────────────────────────────
def test_policy_versions_and_defaults(tmp_path):
    assert tp.current() == "timing-v2"
    assert tp.policy_of({}) == "timing-v1" and tp.policy_of({"timing_policy": "timing-v2"}) == "timing-v2"
    tp._load.cache_clear()
    try:
        assert tp._load(str(tmp_path / "fehlt.yaml")) == {}
    finally:
        tp._load.cache_clear()
    assert tp.params()["decision_quote_max_age_minutes"] == 15 and tp.params()["fill_session_grace_hours"] == 12


# ── Kalender: US- vs. EU-Zeitumstellung, Feiertage, verkürzte Sitzungen, Instrumente ──
@pytest.mark.parametrize("ts, inside", [
    ("2026-03-06T14:00", False),   # beide Winterzeit: 09:00 ET
    ("2026-03-16T14:00", True),    # USA schon Sommerzeit (08.03.), EU noch nicht (29.03.): 10:00 ET
    ("2026-03-30T14:00", True),    # beide Sommerzeit
    ("2026-10-28T14:00", True),    # EU schon Winterzeit (25.10.), USA noch Sommerzeit (01.11.): 10:00 ET
    ("2026-11-04T14:00", False),   # beide Winterzeit: 09:00 ET
    ("2026-11-04T15:00", True)])
def test_window_follows_new_york_not_europe(ts, inside):
    assert tp.in_window(utc(ts), L) is inside


@pytest.mark.parametrize("decision, day", [
    ("2026-11-25T21:30", "2026-11-27"),   # Mittwoch nach Schluss -> Thanksgiving zu -> Freitag (verkürzt)
    ("2026-11-27T17:55", "2026-11-30"),   # verkürzter Tag nach 12:50 ET -> Montag
    ("2026-04-02T21:00", "2026-04-06"),   # Gründonnerstag nach Schluss -> Good Friday zu -> Montag
    ("2026-10-16T23:59", "2026-10-19"),   # Freitagabend -> Montag
    ("2026-10-17T00:30", "2026-10-19"),   # Samstag 00:30 UTC (Freitag 20:30 ET) -> Montag
    ("2025-01-08T22:00", "2025-01-10"),   # Sonderschließung 2025-01-09 übersprungen
    ("2026-10-14T13:00", "2026-10-14")])  # vor Eröffnung -> selber Tag
def test_fill_day_skips_closed_days(decision, day):
    assert tp.fill_day(utc(decision), L).isoformat() == day


def test_instrument_specific_close():
    spy = "SPY261120C00500000"
    assert oq.session_of("2026-10-14T20:05:00+00:00", spy) == "RTH"        # 16:05 ET: SPY-Optionen handeln bis 16:15
    assert oq.session_of("2026-10-14T20:05:00+00:00", L) == "POST"         # Aktienoption schon geschlossen
    assert tp.in_window(utc("2026-10-14T20:05"), spy) and not tp.in_window(utc("2026-10-14T20:05"), L)
    assert oq.session_of("2026-11-27T18:10:00+00:00", "SPXW261127C05000000") == "RTH"   # verkürzt bis 13:15 ET


# ── Entry ────────────────────────────────────────────────────────────────────
def test_entry_after_close_goes_to_next_session_never_earlier():
    quote(L, "2026-10-14T19:00", 9.0, 9.5)                    # vor der Entscheidung -> nie
    quote(L, "2026-10-15T14:05", 9.9, 10.3)                   # 10:05 ET nächster Tag -> Fill
    t = trade(decision="2026-10-14T21:30:00+00:00")
    e, recs = tp.evaluate_entry(t, utc("2026-10-15T22:00"))
    assert e["status"] == "FILLED" and e["day"] == "2026-10-15" and e["price"] == 10.3
    assert tp._ts(e["time"]) >= tp._ts(t["decision_time"])


def test_entry_in_window_uses_fresh_decision_quote_but_not_a_stale_one():
    quote(L, "2026-10-14T15:50", 9.0, 9.4)                    # 10 min vor Entscheidung -> zulässig
    e, _ = tp.evaluate_entry(trade(decision="2026-10-14T16:00:00+00:00"), utc("2026-10-14T16:01"))
    assert e["status"] == "FILLED" and e["price"] == 9.4 and e["time"] == "2026-10-14T16:00:00+00:00"   # Zeit = Entscheidung
    e2, _ = tp.evaluate_entry(trade(decision="2026-10-14T16:30:00+00:00"), utc("2026-10-14T16:31"))
    assert e2["status"] == "PENDING"                          # 40 min alte Quote -> nicht zulässig


def test_entry_quote_missing_pending_then_unusable_after_session():
    t = trade(decision="2026-10-14T21:30:00+00:00")
    assert tp.evaluate_entry(t, utc("2026-10-15T12:00"))[0]["reason"] == "AWAITING_SESSION"
    assert tp.evaluate_entry(t, utc("2026-10-15T18:00"))[0]["reason"] == "AWAITING_ADMISSIBLE_QUOTE"
    late = tp.evaluate_entry(t, utc("2026-10-16T07:49"))[0]   # Fensterende 19:50 UTC + 12 h = 07:50 UTC
    assert late["status"] == "PENDING"
    gone = tp.evaluate_entry(t, utc("2026-10-16T07:51"))[0]
    assert gone["status"] == "UNUSABLE" and gone["reason"] == "NO_ADMISSIBLE_QUOTE_IN_SESSION"


def test_late_committed_quote_is_point_in_time():
    quote(L, "2026-10-15T15:00", 9.9, 10.2)
    t = trade(decision="2026-10-14T21:30:00+00:00")
    assert tp.evaluate_entry(t, utc("2026-10-15T14:59"))[0]["status"] == "PENDING"   # noch nicht abgerufen
    assert tp.evaluate_entry(t, utc("2026-10-15T15:01"))[0]["status"] == "FILLED"


def test_inadmissible_quotes_never_fill():
    quote(L, "2026-10-15T13:45", 9.9, 10.2)                   # 09:45 ET: Eröffnungsphase
    quote(L, "2026-10-15T19:55", 9.9, 10.2)                   # 15:55 ET: Schlussminuten
    quote(L, "2026-10-15T15:00", 0.0, 10.2)                   # Bid 0
    quote(L, "2026-10-15T15:10", 9.9, 10.2, quote_time="2026-10-13T15:00")   # Börsenzeit > 26 h alt (STALE)
    quote(L, "2026-10-15T15:20", 10.5, 10.2)                  # gekreuzt (UNUSABLE)
    e, _ = tp.evaluate_entry(trade(decision="2026-10-14T21:30:00+00:00"), utc("2026-10-15T22:00"))
    assert e["status"] == "PENDING"


@pytest.mark.parametrize("seed", range(5))
def test_property_no_retroactive_fill(seed):
    """Zufällige Entscheidungen und Quotes über mehrere Wochen: jeder Fill liegt nie vor der Entscheidung und
    stammt nie aus einer Quote vor der zulässigen Grenze."""
    rng = random.Random(seed)
    base = utc("2026-10-12T00:00")
    for _ in range(60):
        quote(L, (base + timedelta(minutes=rng.randrange(0, 14 * 1440))).isoformat()[:16], 9.5, 10.0)
    for _ in range(40):
        d = base + timedelta(minutes=rng.randrange(0, 12 * 1440))
        e, recs = tp.evaluate_entry(trade(decision=d.isoformat()), base + timedelta(days=15))
        if e["status"] != "FILLED":
            continue
        limit = d - timedelta(minutes=15) if tp.in_window(d, L) else d
        assert tp._ts(e["time"]) >= d
        assert all(tp._ts(r["retrieved_at"]) >= limit for r in recs)
        assert e["day"] == tp.fill_day(limit, L).isoformat()      # nur die früheste mögliche Sitzung


def test_spread_entry_needs_legs_from_the_same_capture_round():
    quote(L, "2026-10-15T15:00", 9.9, 10.2)
    quote(S, "2026-10-15T16:00", 4.0, 4.3)                    # 60 min später -> keine gemeinsame Runde
    t = trade(decision="2026-10-14T21:30:00+00:00", strategy="BULL_CALL_SPREAD", short=S)
    assert tp.evaluate_entry(t, utc("2026-10-15T17:00"))[0]["status"] == "PENDING"
    quote(S, "2026-10-15T15:05", 4.1, 4.4)
    e, _ = tp.evaluate_entry(t, utc("2026-10-15T17:00"))
    assert e["status"] == "FILLED" and e["price"] == pytest.approx(10.2 - 4.1)


# ── Exit und Verfall ─────────────────────────────────────────────────────────
def closed(t, close_time):
    return {**t, "close_time": close_time}


def test_exit_after_close_uses_next_session_bid():
    quote(L, "2026-10-13T16:00", 9.6, 10.0)                   # Entry 10.0
    quote(L, "2026-10-14T18:00", 20.0, 20.5)                  # vor dem Exit-Entscheid -> nie
    quote(L, "2026-10-15T15:00", 13.0, 13.4)                  # nächste Sitzung -> Exit-Bid 13.0
    t = closed(trade(), "2026-10-14T21:30:00+00:00")
    blk, _, xr = tp.evaluate(t, utc("2026-10-15T21:00"))
    assert blk["status"] == "OK" and blk["outcome"] == pytest.approx(0.3) and blk["method"] == "option_quote"
    assert tp._ts(blk["exit"]["time"]) >= tp._ts(t["close_time"]) and xr[0]["bid"] == 13.0


def test_exit_decision_before_entry_fill_is_unusable():
    quote(L, "2026-10-15T15:00", 9.9, 10.2)                   # Entry erst am 15.
    t = closed(trade(decision="2026-10-14T21:30:00+00:00"), "2026-10-15T01:00:00+00:00")   # Exit-Entscheid davor
    blk, *_ = tp.evaluate(t, utc("2026-10-15T21:00"))
    assert blk["status"] == "UNUSABLE" and "EXIT_BEFORE_ENTRY" in blk["reasons"]


@pytest.mark.parametrize("strategy, short, close, expected", [
    ("LONG_CALL", None, 215.0, 15.0), ("LONG_PUT", None, 185.0, 15.0),
    ("BULL_CALL_SPREAD", S, 230.0, 10.0), ("LONG_CALL", None, 190.0, 0.0)])
def test_expired_option_settles_officially(strategy, short, close, expected):
    sym = L if "PUT" not in strategy else "AAPL261016P00200000"
    t = trade(sym=sym, strategy=strategy, short=short, expiry="2026-10-16", decision="2026-10-13T16:00:00+00:00")
    quote(sym, "2026-10-13T16:00", 9.6, 10.0)
    if short:
        quote(short, "2026-10-13T16:01", 0.0 + 4.0, 4.2)
    t = closed(t, "2026-10-16T21:30:00+00:00")                # Verfallstag nach Schluss -> Settlement
    blk, *_ = tp.evaluate(t, utc("2026-10-17T12:00"), close_fn=lambda tk, d: close)
    assert blk["status"] == "OK" and blk["method"] == "expiry_intrinsic" and blk["price_type"] == "SETTLEMENT_INTRINSIC"
    assert blk["exit"]["price"] == pytest.approx(expected) and blk["exit"]["day"] == "2026-10-16"


def test_settlement_close_missing_pending_then_unusable():
    quote(L, "2026-10-13T16:00", 9.6, 10.0)
    t = closed(trade(expiry="2026-10-16"), "2026-10-19T15:00:00+00:00")
    assert tp.evaluate(t, utc("2026-10-19T16:00"), close_fn=lambda *a: None)[0]["status"] == "PENDING"
    blk = tp.evaluate(t, utc("2026-10-22T16:00"), close_fn=lambda *a: None)[0]
    assert blk["status"] == "UNUSABLE" and "EXIT_SETTLEMENT_CLOSE_MISSING" in blk["reasons"]


def test_never_ok_without_option_quote():
    """Ohne archivierte Optionsquote nie ein Outcome – kein Basiswert-, Modell- oder Entscheidungspreis."""
    t = closed({**trade(), "close_price": 250.0, "outcome_at_decision": 0.9, "simulation": {"current_price": 200}},
               "2026-10-14T21:30:00+00:00")
    for now in ("2026-10-14T22:00", "2026-10-15T18:00", "2026-11-30T12:00"):
        blk = tp.evaluate(t, utc(now))[0]
        assert blk["status"] != "OK" and blk["outcome"] is None


# ── Feedback: Champion-Entscheidung unverändert, Ausführung nach Policy ────────
_ORIG = {"learn": feedback.learn_on_close, "record": feedback.record_trade_outcome}   # nie doppelt umhüllen


class _FakeDT(datetime):
    now_value: datetime = datetime(2026, 1, 1, tzinfo=UTC)

    @classmethod
    def now(cls, tz=None):
        return cls.now_value if tz else cls.now_value.replace(tzinfo=None)

    @classmethod
    def utcnow(cls):
        return cls.now_value.replace(tzinfo=None)


def run_main(monkeypatch, tmp_path, history, now, *, market_open=None, outcome=0.6, exit_reason="take_profit"):
    from modules import email_reporter, final_mc_ledger, rl_robust_shadow, universe_v2_ledger
    from modules.expectation_alpha import config as ea_cfg
    _FakeDT.now_value = now
    calls = {"compute": 0, "price": 0, "learn": [], "record": []}
    monkeypatch.setattr(feedback, "datetime", _FakeDT)
    monkeypatch.setattr(feedback, "load_history", lambda: history)
    monkeypatch.setattr(feedback, "save_history", lambda h: None)
    if market_open is not None:
        monkeypatch.setattr(feedback, "_market_open_day", lambda t: market_open)

    def price(tk):
        calls["price"] += 1
        return 100.0

    def comp(t, cur, meta=None):
        calls["compute"] += 1
        if meta is not None:
            meta.update(method="option_quote", price_type="OBSERVED_QUOTE")
        return outcome
    real_learn, real_record = _ORIG["learn"], _ORIG["record"]

    def learn(h, t, o, r):
        calls["learn"].append((t["ticker"], round(o, 4), r))
        return real_learn(h, t, o, r)

    def record(t, o, meta, *a, **k):
        calls["record"].append((t["ticker"], round(o, 4), bool(k.get("exit_quotes"))))
        return real_record(t, o, meta, *a, path=tmp_path / "trade_outcomes.jsonl", **k)
    for name, fn in (("get_current_price", price), ("compute_outcome", comp), ("learn_on_close", learn),
                     ("record_trade_outcome", record), ("check_exit_rules", lambda t, o, d: exit_reason),
                     ("register_exit_sim", lambda *a, **k: None), ("_spread_execution_monitor", lambda *a, **k: None),
                     ("_spread_execution_exit", lambda *a, **k: None), ("compute_pearson_weights", lambda h: {}),
                     ("evaluate_shadow_trades", lambda *a, **k: None), ("maybe_notify_rl_arming", lambda h: None)):
        monkeypatch.setattr(feedback, name, fn)
    monkeypatch.setattr(feedback.candidate_ledger, "update_outcomes", lambda *a, **k: None)
    monkeypatch.setattr(universe_v2_ledger, "resolve_outcomes", lambda **k: 0)
    monkeypatch.setattr(final_mc_ledger, "resolve_outcomes", lambda **k: 0)
    monkeypatch.setattr(ea_cfg, "enabled", lambda: False)
    monkeypatch.setattr(rl_robust_shadow, "train_robust_shadow", lambda h: None)
    monkeypatch.setattr(email_reporter, "send_exit_alert_email", lambda *a, **k: None)
    feedback.main()
    return calls


def test_feedback_v1_closes_immediately_v2_defers_to_admissible_quote(monkeypatch, tmp_path):
    """Regression (Abweichung sichtbar): identischer Exit-Entscheid nach Schluss. timing-v1 (Altbestand) schließt
    sofort zum Entscheidungs-Return, timing-v2 wartet auf die nächste zulässige Quote."""
    quote(L, "2026-10-13T16:00", 9.6, 10.0)
    v1, v2 = trade(M, policy=None, ticker="MSFT"), trade()
    hist = {"active_trades": [v1, v2], "closed_trades": []}
    c = run_main(monkeypatch, tmp_path, hist, utc("2026-10-14T21:30"), market_open=True)
    assert [t["ticker"] for t in hist["closed_trades"]] == ["MSFT"] and hist["closed_trades"][0]["outcome"] == 0.6
    pend = hist["exit_pending_trades"]
    assert [t["ticker"] for t in pend] == ["AAPL"] and pend[0]["outcome"] is None
    assert pend[0]["outcome_at_decision"] == 0.6 and pend[0]["close_reason"] == "take_profit"   # Entscheidung gleich
    assert [x[0] for x in c["learn"]] == ["MSFT"]              # v2 noch nicht gelernt
    # nächste Sitzung: zulässige Quote -> Ausführung, Outcome, Lernen genau einmal
    quote(L, "2026-10-15T15:00", 13.0, 13.4)
    c2 = run_main(monkeypatch, tmp_path, hist, utc("2026-10-15T21:30"), market_open=True)
    v2c = [t for t in hist["closed_trades"] if t["ticker"] == "AAPL"]
    assert len(v2c) == 1 and v2c[0]["outcome"] == 0.3 and v2c[0]["outcome_method"] == "option_quote"
    assert v2c[0]["outcome_evaluation"]["status"] == "OK" and v2c[0]["outcome_reliable"] is True
    assert c2["learn"] == [("AAPL", 0.3, True)] and c2["record"] == [("AAPL", 0.3, True)]
    assert hist["exit_pending_trades"] == []
    # wiederholter Lauf: nichts doppelt
    c3 = run_main(monkeypatch, tmp_path, hist, utc("2026-10-15T23:30"), market_open=True)
    assert c3["learn"] == [] and len([t for t in hist["closed_trades"] if t["ticker"] == "AAPL"]) == 1
    recs = [json.loads(x) for x in (tmp_path / "trade_outcomes.jsonl").read_text().splitlines()]
    assert sum(r["ticker"] == "AAPL" for r in recs) == 1


def test_feedback_exit_in_window_executes_immediately(monkeypatch, tmp_path):
    quote(L, "2026-10-13T16:00", 9.6, 10.0)
    quote(L, "2026-10-14T16:58", 12.0, 12.4)                  # Exit-Quote desselben Laufs (2 min vor Entscheid)
    hist = {"active_trades": [trade()], "closed_trades": []}
    run_main(monkeypatch, tmp_path, hist, utc("2026-10-14T17:00"), market_open=True)
    assert hist["closed_trades"][0]["outcome"] == 0.2 and hist.get("exit_pending_trades") == []


def test_feedback_on_market_holiday_makes_no_market_outcome(monkeypatch, tmp_path):
    quote(L, "2026-10-13T16:00", 9.6, 10.0)
    hist = {"active_trades": [trade(), trade(M, policy=None, ticker="MSFT")], "closed_trades": []}
    c = run_main(monkeypatch, tmp_path, hist, utc("2026-11-26T19:00"))      # Thanksgiving, echter Kalender
    assert c["compute"] == 0 and c["price"] == 0 and hist["closed_trades"] == []
    assert len(hist["active_trades"]) == 2


def test_feedback_without_admissible_quote_ends_unusable_not_learned(monkeypatch, tmp_path):
    quote(L, "2026-10-13T16:00", 9.6, 10.0)
    hist = {"active_trades": [trade()], "closed_trades": []}
    run_main(monkeypatch, tmp_path, hist, utc("2026-10-14T21:30"), market_open=True)
    c = run_main(monkeypatch, tmp_path, hist, utc("2026-10-16T12:00"), market_open=True)   # Sitzung 15.10. + Karenz vorbei
    assert hist["closed_trades"] == [] and hist["exit_pending_trades"] == [] and c["learn"] == []
    bad = hist["closed_trades_unusable"]
    assert len(bad) == 1 and bad[0]["outcome"] is None
    assert bad[0]["timing"]["reasons"] == ["EXIT_NO_ADMISSIBLE_QUOTE_IN_SESSION"]


def test_counterfactual_trades_follow_the_same_policy(monkeypatch, tmp_path):
    quote(L, "2026-10-13T16:00", 9.6, 10.0)
    hist = {"active_trades": [], "closed_trades": [], "counterfactual_trades": [trade()]}
    run_main(monkeypatch, tmp_path, hist, utc("2026-10-14T21:30"), market_open=True)
    assert not hist.get("counterfactual_closed")                        # nicht zum Entscheidungs-Return geschlossen
    assert [t["ticker"] for t in hist["counterfactual_exit_pending"]] == ["AAPL"]
    quote(L, "2026-10-15T15:00", 11.0, 11.4)
    run_main(monkeypatch, tmp_path, hist, utc("2026-10-15T21:30"), market_open=True)
    assert hist["counterfactual_closed"][0]["outcome"] == 0.1 and hist["counterfactual_exit_pending"] == []


def test_pipeline_stamps_decision_time_and_policy():
    import pipeline
    rec = pipeline.build_trade_record({"ticker": "AAPL", "strategy": "LONG_CALL",
                                       "option": {"ask": 10.0, "bid": 9.6, "strike": 200, "expiry": "2027-03-19"}},
                                      "2026-10-14")
    assert rec["timing_policy"] == "timing-v2" and tp._ts(rec["decision_time"]) is not None
    assert rec["entry_debit"] == 10.0                                    # Champion-Konvention unverändert


# ── Kandidaten, Capture, Evidenz ─────────────────────────────────────────────
def cand_row(policy="timing-v2", signal="2026-10-14T21:30:00+00:00"):
    r = {"signal_id": "s1", "ticker": "AAPL", "direction": "BULLISH", "date": signal[:10], "signal_timestamp": signal,
         "outcomes": {}, "real_option": {"symbol": L, "expiry": "2027-03-19", "strike": 200.0, "bid": None,
                                         "ask": None, "mid": None, "entry_pending": True}}
    if policy:
        r["timing_policy"] = policy
    return r


def test_candidate_v2_entry_from_next_session_only(monkeypatch):
    monkeypatch.setattr(cl, "_now_utc", lambda: utc("2026-10-15T22:00"))
    quote(L, "2026-10-14T19:00", 9.0, 9.5)                                # vor dem Signal -> nie
    quote(L, "2026-10-15T15:00", 9.9, 10.3)
    rows = [cand_row(), cand_row(policy=None)]
    assert cl._fill_v2_entries(rows) is True
    ro = rows[0]["real_option"]
    assert ro["entry_status"] == "FILLED" and ro["ask"] == 10.3 and ro["entry_date"] == "2026-10-15"
    assert rows[1]["real_option"]["entry_pending"] is True               # Altbestand unberührt


def test_capture_targets_respect_decision_times():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import rth_marks
    t_open = trade(decision="2026-10-15T16:30:00+00:00")
    t_pend = {**trade(M, ticker="MSFT"), "close_time": "2026-10-15T17:00:00+00:00"}
    tg = rth_marks.trade_targets({"active_trades": [t_open], "exit_pending_trades": [t_pend]}, utc("2026-10-15T18:00"))
    assert tg[L] == utc("2026-10-15T16:30") and tg[M] == utc("2026-10-15T17:00")


def test_scorecard_separates_timing_policies():
    def rec(pol, i):
        return el._rec(prediction_id=f"p{pol}{i}", source_store="t", population="CHAMPION_TRADE", entity="A",
                       decision_time=f"2026-09-{i + 1:02d}", direction=1, features={}, horizon=None,
                       horizon_unit="holding", outcome=0.1, outcome_net=0.09, cost=0.01, outcome_method="option_quote",
                       price_type="OBSERVED_QUOTE", quality="RELIABLE", label_available_at=f"2026-09-{i + 2:02d}",
                       meta={"timing_policy": pol})
    recs = [rec("timing-v1", i) for i in range(3)] + [rec("timing-v2", i) for i in range(2)]
    cells = el.scorecard(recs, as_of=date(2026, 10, 11))["cells"]
    assert cells["CHAMPION_TRADE|timing=timing-v1"]["n"] == 3 and cells["CHAMPION_TRADE|timing=timing-v2"]["n"] == 2
    assert cells["CHAMPION_TRADE|ALL"]["n"] == 5


# ── Nachbesserungen nach der unabhängigen Prüfung ────────────────────────────
def test_filled_entry_is_frozen_against_later_committed_earlier_quotes():
    quote(L, "2026-10-15T15:00", 9.9, 10.3)
    t = trade(decision="2026-10-14T21:30:00+00:00")
    t["timing"] = {"entry": tp.evaluate_entry(t, utc("2026-10-15T16:00"))[0]}
    assert t["timing"]["entry"]["price"] == 10.3
    quote(L, "2026-10-15T14:10", 8.0, 8.4)                   # später committet, aber früher abgerufen
    e, recs = tp.evaluate_entry(t, utc("2026-10-15T20:00"))
    assert e["price"] == 10.3 and recs[0]["ask"] == 10.3     # festgestellter Fill bleibt reproduzierbar


def test_contract_symbol_is_normalized_like_the_archive():
    quote(L, "2026-10-15T15:00", 9.9, 10.3)
    t = trade(sym=" aapl270319c00200000 ", decision="2026-10-14T21:30:00+00:00")
    assert tp.evaluate_entry(t, utc("2026-10-15T16:00"))[0]["status"] == "FILLED"


def test_settlement_waits_for_the_official_close():
    quote(L, "2026-10-13T16:00", 9.6, 10.0)
    t = closed(trade(expiry="2026-10-16"), "2026-10-16T19:55:00+00:00")   # 15:55 ET am Verfallstag
    calls = []

    def cf(tk, d):
        calls.append(d)
        return 230.0
    assert tp.evaluate(t, utc("2026-10-16T20:30"), close_fn=cf)[0]["status"] == "PENDING" and calls == []
    assert tp.evaluate(t, utc("2026-10-16T21:05"), close_fn=cf)[0]["status"] == "OK"         # Schluss + 1 h

"""Quant-Audit Options-Erwartungswert (2026-10-17), Bericht: docs/OPTION_EV_AUDIT_2026-10-17.md.

Teil 1 – behobene technische Fehler ohne Einfluss auf Handelsentscheidungen:
  a) Shadow-Exit-Monitoring bekam nie einen Quote-Zeitstempel -> STALE feuerte nie (KDP 15.10.:
     Stop-Entscheid 22:33 UTC auf einer Quote von 19:59:50 UTC, Combo 0,20/2,70).
  b) Post-mortem maß die Einstiegskosten eines Spreads nur am Long-Bein (KDP: 6,5 % statt 24 %).
Teil 2 – P0-1/P0-2 im Entscheidungsweg: seit V1/V2 (2026-10-18) behoben, die Invarianten
  "Gate-ROI <= Maximalertrag" gelten jetzt als normale Regressionstests."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

import feedback
from modules import mirofish_simulation as mf
from modules import option_quotes as oq
from modules import options_designer as od
from modules import spread_execution as se
from modules import trade_memory as tm

ROOT = Path(__file__).resolve().parent.parent
HISTORY = ROOT / "outputs" / "history.json"
CFG = se.load_cfg()

# KDP-Spread 15.10. (archivierte Quotes, outputs/options_archive/quotes/2026-10-15.jsonl)
KDP_TRADE = {"ticker": "KDP", "strategy": "BULL_CALL_SPREAD", "entry_date": "2026-10-15", "entry_debit": 1.65,
             "simulation": {"current_price": 33.81},
             "option": {"strike": 33.0, "expiry": "2027-03-19", "spread_leg": {"strike": 36.0}}}
KDP_LONG_22H = {"bid": 2.25, "ask": 4.0, "source": "tradier", "quote_time": "2026-10-15T19:59:50+00:00"}
KDP_SHORT_22H = {"bid": 1.3, "ask": 2.05, "source": "tradier", "quote_time": "2026-10-15T19:59:50+00:00"}


def _real_closed(ticker):
    if not HISTORY.exists():
        pytest.skip("outputs/history.json fehlt")
    trades = json.loads(HISTORY.read_text())["closed_trades"]
    hit = [t for t in trades if t.get("ticker") == ticker and t.get("outcome_method")]
    if not hit:
        pytest.skip(f"{ticker}-Trade nicht in history.json")
    return hit[0]


# ── a) Shadow-Exit: STALE ────────────────────────────────────────────────────

def test_quote_time_of_tradier_payload():
    assert oq.quote_time_of({"bid_date": 0, "ask_date": 1760558390000}) == datetime(
        2025, 10, 15, 19, 59, 50, tzinfo=timezone.utc)                       # 0 = "kein Datum" übersprungen
    assert oq.quote_time_of({"bid_date": "0", "trade_date": 0}) is None
    assert oq.quote_time_of({}) is None


def test_tradier_quote_carries_quote_time_prices_unchanged(monkeypatch):
    payload = {"options": {"option": [{"option_type": "call", "strike": 33.0, "bid": 2.25, "ask": 4.0,
                                       "bid_date": 1760558390000, "ask_date": 1760558380000}]}}

    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return payload

    monkeypatch.setattr(feedback.requests, "get", lambda *a, **k: R())
    monkeypatch.setattr(feedback, "_archive_exit_quote", lambda *a, **k: None)
    q = feedback._tradier_option_quote("KDP", 33.0, "2027-03-19", "call")
    assert q["bid"] == 2.25 and q["ask"] == 4.0 and q["source"] == "tradier"
    assert q["quote_time"] == "2025-10-15T19:59:50+00:00"
    # Illiquides Bein: Bid seit Langem unverändert, Ask frisch -> jüngere Seite zählt (kein falsches STALE).
    assert feedback._quote_time_iso({"bid_date": 1760540000000, "ask_date": 1760558390000}) == \
        "2025-10-15T19:59:50+00:00"
    assert feedback._quote_time_iso({"bid_date": 0, "ask_date": 0}) is None


def test_spread_price_unchanged_by_quote_time(monkeypatch):
    legs = iter([KDP_LONG_22H, KDP_SHORT_22H])
    monkeypatch.setattr(feedback, "get_option_quote", lambda *a, **k: next(legs))
    detail = {}
    net = feedback.get_current_spread_price("KDP", KDP_TRADE["option"], "BULL_CALL_SPREAD", detail)
    assert net == pytest.approx(0.20)                                        # Produktion: Long-Bid − Short-Ask
    assert detail["spread_quotes"]["long"]["quote_time"] == KDP_LONG_22H["quote_time"]


def test_kdp_after_hours_quote_is_stale_in_shadow(monkeypatch):
    rec = []
    monkeypatch.setattr(se, "record", lambda *a, **k: rec.append(a) or True)
    trade = json.loads(json.dumps(KDP_TRADE))
    meta = {"spread_quotes": {"long": KDP_LONG_22H, "short": KDP_SHORT_22H}}
    obs = feedback._spread_execution_monitor(trade, meta, 34.18, datetime(2026, 10, 15))
    assert obs["quote_quality"] == "STALE" and obs["executable_pnl_pct"] is None
    assert trade["spread_stop_shadow"]["shadow_stop"] == "NO_RELIABLE_QUOTE"
    assert trade["spread_stop_shadow"]["production_stop_hit"] is False
    assert rec and rec[0][0] == "monitor"
    # Ohne Zeitstempel (Zustand vor dem Fix) wäre dieselbe Quote VERY_WIDE mit −87,9 % "ausführbar".
    old = se.monitor(trade, KDP_LONG_22H, KDP_SHORT_22H, 34.18, CFG)
    assert old["quote_quality"] == "VERY_WIDE" and old["executable_pnl_pct"] == pytest.approx(-0.8788, abs=1e-4)


def test_fresh_quote_not_stale_and_oldest_leg_decides(monkeypatch):
    monkeypatch.setattr(se, "record", lambda *a, **k: True)
    now = datetime.now(timezone.utc)
    fresh = (now - timedelta(minutes=2)).isoformat()
    old = (now - timedelta(minutes=90)).isoformat()
    long_q = {"bid": 3.3, "ask": 3.7, "quote_time": fresh}
    obs = feedback._spread_execution_monitor(json.loads(json.dumps(KDP_TRADE)),
                                             {"spread_quotes": {"long": long_q,
                                                                "short": {"bid": 1.7, "ask": 1.95, "quote_time": fresh}}},
                                             34.18, now)
    assert obs["quote_quality"] != "STALE" and obs["executable_pnl_pct"] == pytest.approx(1.35 / 1.65 - 1, abs=1e-4)
    obs = feedback._spread_execution_monitor(json.loads(json.dumps(KDP_TRADE)),
                                             {"spread_quotes": {"long": long_q,
                                                                "short": {"bid": 1.7, "ask": 1.95, "quote_time": old}}},
                                             34.18, now)
    assert obs["quote_quality"] == "STALE"


def test_missing_quote_time_keeps_previous_behaviour(monkeypatch):
    monkeypatch.setattr(se, "record", lambda *a, **k: True)
    yf_long = {"bid": 3.3, "ask": 3.7, "source": "yfinance"}                 # yfinance: kein Zeitstempel
    obs = feedback._spread_execution_monitor(json.loads(json.dumps(KDP_TRADE)),
                                             {"spread_quotes": {"long": yf_long, "short": KDP_SHORT_22H}},
                                             34.18, datetime(2026, 10, 15))
    assert obs["quote_quality"] != "STALE"


def test_production_stop_rule_unchanged():
    # Die Schwellen in check_exit_rules sind unverändert; die Zulässigkeit der Quote prüft seit V5 decide_exit.
    t = {"strategy": "BULL_CALL_SPREAD", "entry_date": "2026-10-15",
         "option": {"strike": 33.0, "expiry": "2027-03-19", "spread_leg": {"strike": 36.0}}, "entry_debit": 1.65}
    assert feedback.check_exit_rules(t, -0.8788, datetime(2026, 10, 15)) == "stop_loss"


# ── b) Post-mortem: Einstiegskosten eines Spreads ───────────────────────────

def _pm_trade(**kw):
    t = {"ticker": "AAA", "entry_date": "2026-04-01", "close_date": "2026-04-02", "strategy": "BULL_CALL_SPREAD",
         "features": {"impact": 5, "surprise": 4, "mismatch": 3.0, "z_score": 0.3, "sigma_30d": 0.02},
         "simulation": {"current_price": 100.0, "hit_rate": 0.7, "sigma": 0.02},
         "deep_analysis": {"direction": "BULLISH", "data_confidence": "high"},
         "close_price": 101.0, "outcome": -0.18, "outcome_method": "spread_quote",
         "entry_quote": {"spread_pct": 0.0645},
         "option": {"bid": 2.9, "ask": 3.1, "spread_leg": {"bid": 1.45, "ask": 1.65}}}
    t.update(kw)
    return t


def test_spread_entry_cost_uses_combo():
    c = tm.build_cases({"closed_trades": [_pm_trade()]})[0]
    assert c["position"]["spread_pct"] == pytest.approx(0.2424, abs=1e-4)   # (1,65 − 1,25) / 1,65
    assert "entry_cost_high" in c["failure"]["labels"]


def test_long_call_and_spread_without_leg_unchanged():
    lc = tm.build_cases({"closed_trades": [_pm_trade(strategy="LONG_CALL")]})[0]
    assert lc["position"]["spread_pct"] == 0.0645 and "entry_cost_high" not in lc["failure"]["labels"]
    no_leg = _pm_trade(option={"bid": 2.9, "ask": 3.1})                      # Short-Bein verworfen
    c = tm.build_cases({"closed_trades": [no_leg]})[0]
    assert c["position"]["spread_pct"] == 0.0645


def test_real_kdp_and_mo_post_mortem_costs():
    kdp = tm.build_cases({"closed_trades": [_real_closed("KDP")]})[0]
    assert kdp["position"]["spread_pct"] == pytest.approx(0.2424, abs=1e-4)
    assert "entry_cost_high" in kdp["failure"]["labels"]
    mo = tm.build_cases({"closed_trades": [_real_closed("MO")]})[0]
    assert mo["position"]["spread_pct"] == pytest.approx(0.0612, abs=1e-4)  # Long Call: unverändert


# ── Teil 2: P0 im Entscheidungsweg (V1/V2 behoben: Invarianten als Regressionstests) ──

ADM = {"S": 84.39, "K1": 77.5, "K2": 85.0, "long_ask": 12.4, "long_bid": 11.5, "short_bid": 7.5,
       "short_ask": 8.0, "iv": 0.3377, "dte": 178}


def _adm_option():
    debit = ADM["long_ask"] - ADM["short_bid"]
    return {"strike": ADM["K1"], "bid": ADM["long_bid"], "ask": ADM["long_ask"], "implied_vol": ADM["iv"],
            "dte": ADM["dte"], "net_debit": debit,
            "spread_leg": {"strike": ADM["K2"], "bid": ADM["short_bid"], "ask": ADM["short_ask"]}}, debit


def test_mc_pnl_gate_spread_mean_bounded_by_max_payoff(monkeypatch):
    monkeypatch.setattr(mf, "_get_hist_params", lambda t: (0.0177, 0.003))   # Drift wie im Live-Lauf rückgerechnet
    s = mf.MirofishSimulation()
    s.rng = np.random.default_rng(5)
    opt, debit = _adm_option()
    cand = {"ticker": "ADM", "simulation": {"current_price": ADM["S"]}, "features": {"sigma_30d": 0.0177},
            "deep_analysis": {"impact": 4, "surprise": 4}}
    r = s.simulate_option_pnl(cand, opt, ADM["dte"], history={}, n_paths=4000)
    max_roi = (ADM["K2"] - ADM["K1"]) / debit - 1                          # +53,1 %
    assert r["mean_pnl_pct"] <= max_roi


def test_compute_roi_spread_bounded_by_max_payoff():
    d = od.OptionsDesigner.__new__(od.OptionsDesigner)
    d._vix_ts = {}
    opt, debit = _adm_option()
    roi = d._compute_roi(opt, {"current_price": ADM["S"], "target_price": ADM["S"] * 1.20}, 40,
                         {"min_roi": 0.15}, "BULL_CALL_SPREAD", 0.75, "OTHER")
    assert roi["roi_net"] <= (ADM["K2"] - ADM["K1"]) / debit - 1

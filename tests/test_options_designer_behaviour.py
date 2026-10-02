"""Audit P1-8: Verhaltenstests für options_designer (Trade-Konstruktion der
täglichen Mail) – Fake-Optionsketten (yfinance + Tradier), keine Netzaufrufe."""
from __future__ import annotations

import types
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from modules import options_designer as od


def _exp(days: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days + 1)).strftime("%Y-%m-%d")


def _chain(current=100.0, iv=0.35, oi_peak=100.0, bid_zero_above=None):
    strikes = np.arange(80.0, 125.0, 5.0)
    rows = []
    for k in strikes:
        intrinsic = max(current - k, 0.0)
        ask = round(intrinsic + 8.0 * np.exp(-abs(k - current) / 30), 2)
        bid = round(ask * 0.95, 2)
        if bid_zero_above is not None and k > bid_zero_above:
            bid = 0.0
        rows.append({"strike": k, "bid": bid, "ask": ask, "openInterest": 2000 if k == oi_peak else 300,
                     "impliedVolatility": iv})
    calls = pd.DataFrame(rows)
    puts = calls.copy()
    return types.SimpleNamespace(calls=calls, puts=puts)


class FakeTicker:
    def __init__(self, closes=None, info=None, options=(), chain=None):
        self._closes = closes if closes is not None else list(100 * np.exp(np.cumsum(np.full(260, 0.001))))
        self.info = info if info is not None else {"currentPrice": 100.0}
        self.options = tuple(options)
        self._chain = chain or _chain()

    def history(self, period="1y", **k):
        n = 35 if period == "35d" else len(self._closes)
        return pd.DataFrame({"Close": self._closes[-n:]})

    def option_chain(self, d):
        return self._chain


class Gates:
    def __init__(self, earnings=False, vix=20.0):
        self._e, self.last_vix = earnings, vix

    def has_upcoming_earnings(self, t):
        return self._e


@pytest.fixture
def designer(monkeypatch):
    monkeypatch.delenv("TRADIER_API_KEY", raising=False)
    monkeypatch.setattr(od, "get_macro_context", lambda: {"vix_term_structure": {"structure": "contango"}})
    monkeypatch.setattr(od.MirofishSimulation, "simulate_option_pnl",
                        lambda self, **k: {"expected_pnl_pct": 0.25, "mean_pnl_pct": 0.6, "pnl_std": 0.8,
                                           "hold_days": 100, "ou_method": "heuristic", "ou_n_days": 0})
    return od.OptionsDesigner(Gates(), history={})


def _signal(**over):
    s = {"ticker": "ABC", "info": {"sector": "Technology"},
         "deep_analysis": {"direction": "BULLISH", "time_to_materialization": "4-8 Wochen",
                           "bear_case_severity": 3, "catalyst": "new product"},
         "simulation": {"current_price": 100.0, "target_price": 140.0}, "quick_mc": {"hit_rate": 0.65},
         "features": {"sigma_30d": 0.02}}
    s.update(over)
    return s


def _yf(monkeypatch, tickers: dict):
    monkeypatch.setattr(od.yf, "Ticker", lambda t: tickers.get(t) or tickers["default"])


# ── reine Hilfsfunktionen ───────────────────────────────────────────────────

def test_helpers():
    assert od.ttm_to_dte_floor("6 Monate") == 140 and od.ttm_to_dte_floor("ca. 3 Monate") == 120
    assert od.ttm_to_dte_floor("unbekannt") == 120
    assert od._safe_float(complex(2, 3)) == 2.0 and od._safe_float("x") == 0.0
    assert od._classify_catalyst_type({"alpha_signals": {"fda_catalyst": True}}) == "FDA"
    assert od._classify_catalyst_type({"deep_analysis": {"catalyst": "Q3 earnings beat"}}) == "EARNINGS"
    assert od._classify_catalyst_type({"deep_analysis": {"catalyst": "Übernahme durch X"}}) == "MA"
    assert od._classify_catalyst_type({"alpha_signals": {"insider_cluster": True}}) == "INSIDER"
    assert od._classify_catalyst_type({}) == "OTHER"


def test_strike_window_adaptive_for_low_prices():
    assert od._strike_window(900, 14, 60) == (1.03, 0.97)
    otm, itm = od._strike_window(100, 14, 60)
    assert otm == pytest.approx(1.15) and itm == pytest.approx(0.85)


def test_dynamic_min_roi_by_vix(designer):
    assert designer._get_dynamic_min_roi(0.10, 15) == 0.07
    assert designer._get_dynamic_min_roi(0.10, 25) == 0.10
    assert designer._get_dynamic_min_roi(0.10, 40) == 0.12


def test_bear_case_and_days_to(designer):
    assert designer._bear_case_ok({"ticker": "A", "deep_analysis": {"bear_case_severity": 9}}) is False
    assert designer._bear_case_ok({"ticker": "A", "deep_analysis": {"bear_case_severity": 2}}) is True
    assert designer._days_to(_exp(30)) == 30 and designer._days_to("kaputt") == 0


# ── ROI ─────────────────────────────────────────────────────────────────────

def test_compute_roi_invalid_cost_fails_closed(designer):
    r = designer._compute_roi({"ask": 0}, {"current_price": 100}, 40, {"min_roi": 0.1}, "LONG_CALL")
    assert r["passes_roi_gate"] is False and r["roi_net"] == 0.0


def test_compute_roi_long_call_vs_spread(designer):
    opt = {"bid": 7.6, "ask": 8.0, "strike": 100, "implied_vol": 0.35, "dte": 200, "net_debit": 5.0}
    sim = {"current_price": 100, "target_price": 140}
    lc = designer._compute_roi(opt, sim, 40, {"min_roi": 0.1}, "LONG_CALL", 0.65, "OTHER")
    sp = designer._compute_roi(opt, sim, 40, {"min_roi": 0.1}, "BULL_CALL_SPREAD", 0.65, "OTHER")
    assert lc["passes_roi_gate"] and 0.4 < lc["delta"] < 0.8 and lc["breakeven"] == 108.0
    assert sp["is_spread"] and sp["cost_basis"] == 5.0 and sp["commission_pct"] > lc["commission_pct"]
    flat = designer._compute_roi(opt, {"current_price": 100, "target_price": 95}, 40, {"min_roi": 0.1}, "LONG_CALL")
    assert flat["roi_gross"] == 0.0 and not flat["passes_roi_gate"]          # kein Kursziel über Kurs -> kein Gate


def test_compute_roi_event_crush_and_bad_iv(designer):
    opt = {"bid": 7.6, "ask": 8.0, "strike": 100, "implied_vol": 9.0, "dte": 30}
    sim = {"current_price": 100, "target_price": 120}
    fda = designer._compute_roi(opt, sim, 90, {"min_roi": 0.1}, "LONG_CALL", 0.6, "FDA")
    ins = designer._compute_roi(opt, sim, 90, {"min_roi": 0.1}, "LONG_CALL", 0.6, "INSIDER")
    assert fda["iv_drop_assumed"] > ins["iv_drop_assumed"] and fda["vega_loss"] >= ins["vega_loss"]
    tr = designer._compute_roi({**opt, "delta": 0.42}, sim, 50, {"min_roi": 0.1}, "LONG_CALL")
    assert tr["delta"] == 0.42                                               # Tradier-Delta hat Vorrang


# ── Optionskette ────────────────────────────────────────────────────────────

def test_find_option_yfinance_long_and_spread(designer):
    t = FakeTicker(options=[_exp(5), _exp(200)])
    o = designer._find_option_yfinance("ABC", "LONG_CALL", 100.0, 150, 365, t)
    assert o["strike"] == 100.0 and o["data_source"] == "yfinance" and o["dte"] == 200
    s = designer._find_option_yfinance("ABC", "BULL_CALL_SPREAD", 100.0, 150, 365, t)
    assert s["spread_leg"]["strike"] > s["strike"] and s["net_debit"] > 0
    assert designer._find_option_yfinance("ABC", "LONG_CALL", 100.0, 400, 500, t) is None
    illiquid = FakeTicker(options=[_exp(200)], chain=types.SimpleNamespace(
        calls=_chain().calls.assign(openInterest=1), puts=_chain().puts))
    assert designer._find_option_yfinance("ABC", "LONG_CALL", 100.0, 150, 365, illiquid) is None


def test_spread_without_short_leg_bid_falls_back(designer):
    t = FakeTicker(options=[_exp(200)], chain=_chain(bid_zero_above=100))
    s = designer._find_option_yfinance("ABC", "BULL_CALL_SPREAD", 100.0, 150, 365, t)
    assert "spread_leg" not in s and "net_debit" not in s


def _tradier_raw(current=100.0):
    raw = []
    for k in np.arange(85.0, 120.0, 5.0):
        for typ in ("call", "put"):
            raw.append({"option_type": typ, "strike": k, "bid": 7.5, "ask": 8.0, "open_interest": 900 if k == 100 else 200,
                        "volume": 10, "symbol": f"ABC{k}{typ}",
                        "greeks": {"mid_iv": 0.33, "delta": 0.6 if typ == "call" else -0.4}})
    return raw


def test_tradier_path(designer, monkeypatch):
    designer._use_tradier = True
    monkeypatch.setattr(od, "_tradier_expirations", lambda s: [_exp(3), _exp(200)])
    monkeypatch.setattr(od, "_tradier_chain", lambda s, e: _tradier_raw())
    o = designer._find_option_for_dte("ABC", "LONG_CALL", 100.0, 150, 365)
    assert o["data_source"] == "tradier" and o["strike"] == 100.0 and o["delta"] == 0.6
    sp = designer._find_option_tradier("ABC", "BULL_CALL_SPREAD", 100.0, 150, 365)
    assert sp["spread_leg"] and sp["net_debit"] == round(8.0 - sp["spread_leg"]["bid"], 2)
    assert designer._get_atm_straddle("ABC", 100.0, _exp(200)) == pytest.approx(0.16)
    pts = designer._term_structure_tradier("ABC", 100.0)
    assert pts and pts[0][1] == pytest.approx(0.33)
    monkeypatch.setattr(od, "_tradier_expirations", lambda s: [])
    t = FakeTicker(options=[_exp(200)])
    assert designer._find_option_for_dte("ABC", "LONG_CALL", 100.0, 150, 365, t)["data_source"] == "yfinance"


def test_tradier_http_helpers(monkeypatch):
    class R:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self.payload
    monkeypatch.setattr(od.requests, "get", lambda url, **k: R(
        {"expirations": {"date": "2026-12-18"}} if "expirations" in url else {"options": {"option": {"strike": 1}}}))
    assert od._tradier_expirations("X") == ["2026-12-18"] and od._tradier_chain("X", "2026-12-18") == [{"strike": 1}]

    def boom(*a, **k):
        raise OSError("down")
    monkeypatch.setattr(od.requests, "get", boom)
    assert od._tradier_expirations("X") == [] and od._tradier_chain("X", "d") == []
    df = od._tradier_chain_to_df([{"option_type": "call", "strike": 10, "greeks": {"mid_iv": 0.0}},
                                  {"option_type": "put", "strike": 10}], "call")
    assert len(df) == 1 and df["impliedVolatility"].iloc[0] == 0.30
    assert od._tradier_chain_to_df([], "call").empty


# ── IV-Rank, Sektor ─────────────────────────────────────────────────────────

def test_iv_rank_paths(designer):
    assert designer._get_iv_rank("ABC", FakeTicker(closes=[100.0] * 10)) is None      # zu wenig Historie
    rv_only = designer._get_iv_rank("ABC", FakeTicker(info={}))
    assert rv_only is not None and 0 <= rv_only <= 100
    full = designer._get_iv_rank("ABC", FakeTicker(options=[_exp(30), _exp(60), _exp(120)]))
    assert 0 <= full <= 100


def test_sector_momentum(designer, monkeypatch):
    up = FakeTicker(closes=list(np.linspace(100, 130, 260)))
    down = FakeTicker(closes=[130.0] * 225 + list(np.linspace(130, 80, 35)))       # −38 % in 35 Tagen
    _yf(monkeypatch, {"XLK": up, "default": up})
    s = _signal()
    assert designer._sector_momentum_ok(s, t=down) is False and s["sector_momentum"]["rel_strength"] < -0.15
    assert designer._sector_momentum_ok(_signal(), t=up) is True
    _yf(monkeypatch, {"default": FakeTicker(closes=[])})
    assert designer._sector_momentum_ok(_signal(), t=FakeTicker(closes=[])) is True     # nicht messbar -> durch


# ── End-to-End über run() ───────────────────────────────────────────────────

def test_run_produces_proposal_with_long_dated_option(designer, monkeypatch):
    tk = FakeTicker(options=[_exp(30), _exp(90), _exp(200)])
    _yf(monkeypatch, {"default": tk})
    out = designer.run([_signal()])
    assert len(out) == 1
    p = out[0]
    assert p["dte_tier"] == "Long-Term" and p["option"]["dte"] >= 150          # DTE-Floor 120 respektiert
    assert p["roi_analysis"]["passes_roi_gate"] and p["edge_vs_implied"] > 0 and p["mc_pnl_mean"] == 0.6


def test_run_rejections_are_labelled(designer, monkeypatch):
    tk = FakeTicker(options=[_exp(200)])
    _yf(monkeypatch, {"default": tk})
    assert designer.run([_signal(deep_analysis={"bear_case_severity": 9, "direction": "BULLISH"})]) == []
    assert designer.skip_reasons["ABC"] == "options_design_bear_case"
    designer.gates = Gates(earnings=True)
    assert designer.run([_signal()]) == [] and designer.skip_reasons["ABC"] == "earnings_gate"
    designer.gates = Gates()
    _yf(monkeypatch, {"default": FakeTicker(closes=[100.0] * 10, options=[_exp(200)])})
    assert designer.run([_signal()]) == [] and designer.skip_reasons["ABC"] == "iv_rank_unavailable"


def test_mc_pnl_gate_rejects_and_logs_roi_reject(designer, monkeypatch):
    monkeypatch.setattr(od.MirofishSimulation, "simulate_option_pnl",
                        lambda self, **k: {"expected_pnl_pct": -0.4, "mean_pnl_pct": -0.1, "pnl_std": 0.5,
                                           "hold_days": 100, "ou_method": "heuristic", "ou_n_days": 0})
    _yf(monkeypatch, {"default": FakeTicker(options=[_exp(200)])})
    assert designer.run([_signal()]) == []
    assert designer.roi_reject_log and designer.roi_reject_log[0]["ticker"] == "ABC"


def test_no_edge_over_breakeven_rejects(designer, monkeypatch):
    _yf(monkeypatch, {"default": FakeTicker(options=[_exp(200)])})
    s = _signal(simulation={"current_price": 100.0, "target_price": 101.0})
    assert designer.run([s]) == []

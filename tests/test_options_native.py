"""Options-native Ziele (option_learner.decompose / native_* / options_native_report, bs_pricing.implied_vol).

Geprüft werden:
  * implizite Volatilität und Delta (Black-Scholes) sind konsistent;
  * die Zerlegung der Mid-Rendite in Delta, Gamma, Theta und Vega ist exakt (Summe = Mid-Rendite) und trifft eine
    bekannte IV-Verschiebung; Underlying nur am selben Mark-Tag, Richtungsbereinigung für BEARISH;
  * Spreads werden je Leg zerlegt;
  * die Strukturwahl nach ausführbarem Erwartungswert handelt nur bei positivem Erwartungswert;
  * Schatten-Entscheidungen landen prospektiv im eigenen Journal (append-only, einmal je Signal), nie im Basis-Journal;
  * der Bericht ist beschreibend, ohne Produktionswirkung.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest

from modules import option_learner as ol
from modules.bs_pricing import bs_delta, bs_price, implied_vol

R = ol.NATIVE_RATE


def _sym(k: float, right: str = "C", exp: str = "261218") -> str:
    return f"ABC{exp}{right}{round(k * 1000):08d}"


def _row(day: str, sid: str, *, s0=100.0, move=0.03, iv0=0.30, iv1=0.33, k=100.0, right="C", direction="BULLISH",
         half=0.05, dte=60, hold=7, at_mark_shift=0, rel_volume=0.0, spread_k2=None):
    """BS-konsistente Ledger-Zeile: Entry-Mid zu iv0, Mark-Mid zu iv1 bei S1 = S0·(1+move) nach `hold` Tagen."""
    d0 = date.fromisoformat(day)
    expiry, mark = d0 + timedelta(days=dte), d0 + timedelta(days=hold)
    t0, t1 = dte / 365.0, (dte - hold) / 365.0
    kind = "call" if right == "C" else "put"
    s1 = s0 * (1 + move)
    adj = -move if direction == "BEARISH" else move
    sym = _sym(k, right, expiry.strftime("%y%m%d"))
    m0, m1 = bs_price(s0, k, t0, iv0, R, kind), bs_price(s1, k, t1, iv1, R, kind)
    row = {"date": day, "signal_id": sid, "ticker": "ABC", "entry_price": s0, "direction": direction,
           "features": {"rel_volume": rel_volume, "scan_day_ret": 0.0},
           "real_option": {"symbol": sym, "bid": m0 - half, "ask": m0 + half, "mid": m0, "expiry": expiry.isoformat()},
           "outcomes": {"real_opt_ret_mid_5d": m1 / m0 - 1, "real_opt_ret_5d": (m1 - half) / (m0 + half) - 1,
                        "real_opt_mark_date_5d": mark.isoformat(), "ret_5d_at_mark": adj,
                        "ret_5d_at_mark_date": (mark + timedelta(days=at_mark_shift)).isoformat()}}
    if spread_k2 is not None:
        sym2 = _sym(spread_k2, right, expiry.strftime("%y%m%d"))
        n0 = m0 - bs_price(s0, spread_k2, t0, iv0, R, kind)
        n1 = m1 - bs_price(s1, spread_k2, t1, iv1, R, kind)
        q2 = bs_price(s0, spread_k2, t0, iv0, R, kind)
        row["real_strategy"] = {
            "strategy": "BULL_CALL_SPREAD", "expiry": expiry.isoformat(), "net_mid_entry": n0,
            "net_debit_entry": n0 + 2 * half,
            "legs": [{"symbol": sym, "side": "long", "bid": m0 - half, "ask": m0 + half, "mid": m0},
                     {"symbol": sym2, "side": "short", "bid": q2 - half, "ask": q2 + half, "mid": q2}]}
        row["outcomes"].update({"real_strat_ret_mid_5d": n1 / n0 - 1,
                                "real_strat_ret_5d": (n1 - 2 * half) / (n0 + 2 * half) - 1,
                                "real_strat_mark_date_5d": mark.isoformat()})
    return row


def _pseudo(row, name="single"):
    return next(e["row"] for e in ol.expressions(row) if e["expression"] == name)


def test_implied_vol_and_delta_are_black_scholes_consistent():
    for kind in ("call", "put"):
        p = bs_price(100, 95, 0.4, 0.27, R, kind)
        assert implied_vol(p, 100, 95, 0.4, R, kind) == pytest.approx(0.27, abs=1e-6)
    h = 1e-4
    num = (bs_price(100 + h, 100, 0.5, 0.2, R, "call") - bs_price(100 - h, 100, 0.5, 0.2, R, "call")) / (2 * h)
    assert bs_delta(100, 100, 0.5, 0.2, R, "call") == pytest.approx(num, abs=1e-6)
    assert bs_delta(100, 100, 0.5, 0.2, R, "put") == pytest.approx(num - 1, abs=1e-6)
    assert implied_vol(0.01, 100, 50, 0.5, R, "call") is None                  # unter innerem Wert


def test_decomposition_is_exact_and_recovers_the_iv_shift():
    row = _row("2026-09-01", "s1")
    comp, why = ol.decompose(row, _pseudo(row), 5)
    assert why == "OK" and abs(comp["check"]) < 1e-8
    assert comp["delta"] + comp["gamma"] + comp["theta"] + comp["vega"] == pytest.approx(comp["mid_ret"], abs=1e-5)
    assert comp["sigma0"] == pytest.approx(0.30, abs=1e-5) and comp["dvol"] == pytest.approx(0.03, abs=1e-4)
    assert comp["delta"] > 0 and comp["gamma"] > 0 and comp["theta"] < 0 and comp["vega"] > 0
    assert comp["move"] == pytest.approx(0.03) and comp["cost"] < 0              # ausführbar schlechter als Mid
    flat = _row("2026-09-01", "s2", iv1=0.30)
    assert ol.decompose(flat, _pseudo(flat), 5)[0]["vega"] == pytest.approx(0.0, abs=1e-5)


def test_bearish_rows_use_the_raw_underlying_move():
    row = _row("2026-09-01", "b1", move=-0.04, right="P", direction="BEARISH")
    assert row["outcomes"]["ret_5d_at_mark"] == pytest.approx(0.04)             # Ledger: richtungsbereinigt
    comp, why = ol.decompose(row, _pseudo(row), 5)
    assert why == "OK" and comp["move"] == pytest.approx(-0.04) and comp["delta"] > 0
    assert comp["dvol"] == pytest.approx(0.03, abs=1e-4)


def test_underlying_only_from_the_option_mark_day_otherwise_no_label():
    row = _row("2026-09-01", "m1", at_mark_shift=1)
    assert ol.decompose(row, _pseudo(row), 5) == (None, "NO_UNDERLYING_AT_MARK")
    del row["outcomes"]["ret_5d_at_mark"]
    assert ol.decompose(row, _pseudo(row), 5) == (None, "NO_UNDERLYING_AT_MARK")
    row = _row("2026-09-01", "m2")
    del row["outcomes"]["real_opt_ret_mid_5d"]
    assert ol.decompose(row, _pseudo(row), 5) == (None, "NO_LABEL")
    row = _row("2026-09-01", "m3")
    row["real_option"]["mid"] = 0.0
    assert ol.decompose(row, _pseudo(row), 5)[1] == "INCOMPLETE_CONTRACT"


def test_spread_is_decomposed_per_leg():
    row = _row("2026-09-01", "sp", spread_k2=110.0)
    comp, why = ol.decompose(row, _pseudo(row, "spread"), 5)
    assert why == "OK" and comp["legs"] == 2 and abs(comp["check"]) < 1e-8
    assert comp["dvol"] == pytest.approx(0.03, abs=1e-3)
    assert comp["delta"] > 0 and comp["move"] == pytest.approx(0.03)
    single = ol.decompose(row, _pseudo(row), 5)[0]
    assert comp["mid_ret"] != pytest.approx(single["mid_ret"])                    # eigene Struktur, eigenes Label


class _Const:
    def __init__(self, v):
        self.v = v

    def predict(self, xs):
        return np.full(len(xs), self.v)


def test_ev_choice_trades_only_with_positive_executable_ev():
    row = _row("2026-09-01", "c1", spread_k2=110.0)
    up = ol.native_choice(row, {"move": _Const(0.08), "dvol": _Const(0.0)}, 5)
    assert up["action"] == "TRADE" and set(up["evs"]) == {"single", "spread"}
    assert up["ev"] == max(up["evs"].values()) and up["expression"] in up["evs"]
    flat = ol.native_choice(row, {"move": _Const(0.0), "dvol": _Const(0.0)}, 5)
    assert flat["action"] == "ABSTAIN" and all(v < 0 for v in flat["evs"].values())   # Theta + Spread + Kommission
    assert ol.native_choice(row, {"move": None, "dvol": None}, 5) == {"action": "ABSTAIN",
                                                                      "reason": "keine bewertbare Struktur",
                                                                      "evs": {}}


def _world(n_days=24, per_day=8, seed=5):
    rng = np.random.default_rng(seed)
    rows, cands = [], []
    d0 = date(2026, 6, 1)
    for k in range(n_days):
        day = (d0 + timedelta(days=7 * k)).isoformat()
        for j in range(per_day):
            rv = float(rng.normal())
            row = _row(day, f"{day}-{j}", move=float(0.03 * rv + rng.normal(0, 0.01)),
                       iv1=float(0.30 + rng.normal(0, 0.02)), rel_volume=rv)
            rows.append(row)
            pr = _pseudo(row)
            ld = date.fromisoformat(row["outcomes"]["real_opt_mark_date_5d"])
            cands.append({"date": day, "signal_id": row["signal_id"], "ticker": "ABC",
                          "exprs": [{"expression": "single", "contract": [row["real_option"]["symbol"]],
                                     "status": "VERIFIED", "x": ol.features(pr), "y": ol.net_label(pr, 5),
                                     "label_day": ld}]})
    return rows, cands


def test_shadow_decisions_are_prospective_append_only_and_separate(tmp_path):
    rows, cands = _world()
    day = date.fromisoformat(rows[-1]["date"])
    now = datetime(2026, 11, 8, tzinfo=timezone.utc)
    data, reasons = ol.native_dataset(rows, cands, day, 5)
    assert reasons["OK"] == len(data) and all(d["label_day"] <= day for d in data)
    n = ol.native_decisions(tmp_path, rows, cands, day, 5, now)
    assert n == 8 and ol.native_decisions(tmp_path, rows, cands, day, 5, now) == 0          # einmal je Signal
    js = [json.loads(x) for x in (tmp_path / ol.NATIVE_JOURNAL).read_text().splitlines()]
    assert len(js) == 8 and {j["policy"] for j in js} == {ol.NATIVE_POLICY}
    assert all(j["train_cutoff"] == day.isoformat() and j["n_train"] == len(data) for j in js)
    assert not (tmp_path / ol.JOURNAL).exists()                                              # nie im Basis-Journal
    models = ol._native_models(data, day)
    assert models["move"] is not None and set(models["move"].names) <= set(ol.LEDGER_FEATURES)
    assert models["move"].describe()["top_coefficients"]["rel_volume"] > 0.01                # lernt die Bewegung
    early = date.fromisoformat(rows[0]["date"])
    assert ol.native_decisions(tmp_path / "e", rows, cands, early, 5, now) == 0             # ohne Labels kein Modell


def test_report_is_descriptive_and_scores_due_shadow_decisions(tmp_path):
    rows, cands = _world()
    last = date.fromisoformat(rows[-1]["date"])
    prior = last - timedelta(days=21)
    ol.native_decisions(tmp_path, rows, cands, prior, 5, datetime(2026, 11, 1, tzinfo=timezone.utc))
    rep = ol.options_native_report(rows, cands, last + timedelta(days=30), 5, out=tmp_path)
    assert rep["production_influence"] == "NONE" and rep["decomposed"] == len(rows)
    assert rep["reasons"] == {"OK": len(rows)}
    assert set(rep["components"]) == {*ol.NATIVE_COMPONENTS, "mid_ret", "exec_ret"}
    assert rep["components"]["delta"]["mean_diff"] is not None
    assert set(rep["variance_share"]) == {"delta", "gamma", "theta", "vega"}
    assert rep["predictability_ic"]["move"]["mean_ic"] > 0.3                                  # Bewegung hängt an rel_volume
    assert rep["ev_choice_walk_forward"]["value_vs_never"]["n"] > 0
    pro = rep["ev_choice_prospective"]
    assert pro["decisions"] == 8 and pro["pending"] == 0 and pro["missing_outcomes"] == 0
    assert pro["value_vs_never"]["n"] == 8
    assert ol._safe_native(rows, None, last, 5, tmp_path)["error"]                            # Fehler -> nur Hinweis

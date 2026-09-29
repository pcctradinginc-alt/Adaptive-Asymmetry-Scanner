"""Faktor-Monitor: echte Signale werden erkannt, Rauschen bekommt Gewicht 0,
kein Look-ahead im Walk-Forward, Decay/Leakage/Redundanz/Kosten-Flags."""
from __future__ import annotations

import json
import random
from datetime import date, timedelta

from modules import factor_monitor as fm


def _events(signal=0.0, n_days=120, per_day=3, seed=1, decay_after=None):
    rnd = random.Random(seed)
    ev = []
    start = date(2026, 1, 1)
    for d in range(n_days):
        day = (start + timedelta(days=d)).isoformat()
        for _ in range(per_day):
            x, noise = rnd.gauss(0, 1), rnd.gauss(0, 1)
            s = signal if decay_after is None or d < decay_after else 0.0
            y = s * x + rnd.gauss(0, 1)
            ev.append({"date": day, "ticker": "T", "source": "closed_trades", "outcome_date": day,
                       "features": {"x_sig": x, "noise": noise, "x_dup": x * 2 + 0.001 * noise},
                       "outcomes": {"net": y}})
    return ev


def test_real_signal_is_high_value_and_weighted():
    ev = _events(signal=0.6)
    r = fm.evaluate_feature(ev, "x_sig", "net")
    assert "high_value" in fm.classify(r, "x_sig")
    w = fm.weights_from(ev, ["x_sig", "noise"], "net")
    assert w["x_sig"] > 0.9 and w["noise"] == 0.0


def test_noise_gets_zero_weight_and_small_samples_are_shrunk():
    ev = _events(signal=0.0, seed=4)
    assert fm.weights_from(ev, ["noise", "x_sig"], "net") == {"noise": 0.0, "x_sig": 0.0}
    few = _events(signal=0.6, n_days=10)             # < MIN_EFF_N unabhängige Tage
    assert fm.weights_from(few, ["x_sig"], "net")["x_sig"] == 0.0
    assert fm.shrink(0.3, 30) < 0.3 * 0.5 + 1e-9


def test_decay_is_detected():
    ev = _events(signal=0.8, n_days=180, decay_after=60, seed=2)
    r = fm.evaluate_feature(ev, "x_sig", "net")
    assert r["ewma_rank_ic"] < r["rank_ic"]
    assert "decaying" in fm.classify(r, "x_sig")


def test_redundancy_and_leakage_flags():
    ev = _events(signal=0.3)
    pairs = fm.redundancy(ev, ["x_sig", "x_dup", "noise"])
    assert any({p["a"], p["b"]} == {"x_sig", "x_dup"} for p in pairs)
    assert "potential_leakage" in fm.classify({"coverage": 1, "rank_ic": 0.1, "n": 50,
                                               "rank_ic_ci90": [0.0, 0.2]}, "realized_pnl")


def test_walk_forward_never_trains_on_future_or_unrealized_outcomes():
    ev = _events(signal=0.6, n_days=150)
    # Outcome erst weit in der Zukunft realisiert -> darf nicht ins Training
    for e in ev:
        e["outcome_date"] = "2099-01-01"
    wf = fm.walk_forward(ev, ["x_sig"], "net")
    assert wf["folds"] == []
    ev2 = _events(signal=0.6, n_days=150)
    wf2 = fm.walk_forward(ev2, ["x_sig", "noise"], "net")
    assert wf2["folds"] and all(f["n_train"] > 0 for f in wf2["folds"])
    for f in wf2["folds"]:
        assert all(e["date"] < f"{f['test_month']}-01" for e in ev2[:f["n_train"]])
    assert wf2["adaptive"]["oos_rank_ic"] > 0.2


def test_psi_detects_scale_shift():
    assert fm.psi([float(i) for i in range(100)], [float(i) * 50 for i in range(100)]) > fm.PSI_DRIFT
    assert fm.psi([float(i) for i in range(100)], [float(i) + 0.1 for i in range(100)]) < 0.05


def test_run_writes_database_and_never_claims_production(tmp_path):
    hist = {"closed_trades": [{"ticker": "A", "entry_date": e["date"], "close_date": e["date"],
                               "outcome": e["outcomes"]["net"], "features": e["features"]}
                              for e in _events(signal=0.5, n_days=90)],
            "shadow_trades": [{"ticker": "B", "entry_date": "2026-03-01", "outcome": 0.5,
                               "features": {"x_sig": 9.9}}]}          # ohne close_date -> ignoriert
    hp = tmp_path / "history.json"
    hp.write_text(json.dumps(hist))
    rep = fm.run(history_path=hp, ledger_dir=tmp_path / "none", reports_dir=tmp_path / "none",
                 out_dir=tmp_path / "out", today=date(2026, 6, 1))
    assert rep["sources"] == {"closed_trades": 270}
    assert "x_sig" in rep["outcomes"]["net"]["summary"]["high_value"]
    w = json.loads((tmp_path / "out" / "factor_weights_shadow.json").read_text())
    assert w["production_use"] is False
    rows = (tmp_path / "out" / "factor_performance.jsonl").read_text().splitlines()
    assert rows and json.loads(rows[0])["run_date"] == "2026-06-01"


def test_cost_check_flags_gross_only_alpha():
    rep = {"outcomes": {"gross_45d": {"features": {"f": {"tercile_spread": 0.05}}},
                        "net_45d": {"features": {"f": {"tercile_spread": -0.02}}}}}
    assert fm._cost_check(rep)[0]["flag"] == "only_profitable_before_costs"


def test_decayed_factor_gets_lower_weight_than_stable_one():
    stable = fm.weights_from(_events(signal=0.8, n_days=180, seed=2), ["x_sig", "noise"], "net")
    ev = _events(signal=0.8, n_days=180, decay_after=60, seed=2)
    decayed_ic = fm.evaluate_feature(ev, "x_sig", "net")["recent_3m_rank_ic"]
    raw = fm.shrink(decayed_ic, 180)
    assert abs(raw) < 0.1          # nach Decay kaum noch Gewicht (vor Normierung)
    assert stable["x_sig"] > 0.9


def test_market_regimes_use_only_prior_day_closes(monkeypatch):
    import pandas as pd

    idx = pd.date_range("2025-01-01", periods=420, freq="D")
    class _T:
        def __init__(self, sym):
            base = {"SPY": [100 + i for i in range(420)], "^TNX": [4.5] * 420, "^IRX": [5.0] * 420}[sym]
            base[-1] = 1.0 if sym == "SPY" else base[-1]      # Crash AM Signaltag
            self.s = pd.Series(base, index=idx)
        def history(self, start=None):
            return {"Close": self.s}
    import yfinance
    monkeypatch.setattr(yfinance, "Ticker", _T)
    d = idx[-1].date().isoformat()
    r = fm.market_regimes([d])[d]
    assert r == {"trend": "bull", "rates": "high_rates", "curve": "inverted"}   # Crash erst nach d sichtbar


def test_regime_dependent_detection():
    reg = {"trend=bull": {"rank_ic": 0.4, "n_eff_dates": 60}, "trend=bear": {"rank_ic": -0.3, "n_eff_dates": 60}}
    assert fm._regime_dependent(reg)
    assert not fm._regime_dependent({"a": {"rank_ic": 0.1, "n_eff_dates": 60},
                                     "b": {"rank_ic": 0.12, "n_eff_dates": 60}})


def test_llm_value_add_paired_against_price_baseline(tmp_path):
    import random as _r
    rnd = _r.Random(2)
    rows = []
    for d in range(40):
        day = f"2026-{1 + d // 28:02d}-{1 + d % 28:02d}"
        for _ in range(4):
            dr = rnd.choice([-0.02, 0.02])
            true_up = rnd.random() < 0.5
            raw = (0.03 if true_up else -0.03) + rnd.gauss(0, 0.01)
            direction = "BULLISH" if true_up else "BEARISH"          # LLM "weiß" die Richtung
            llm = raw if direction == "BULLISH" else -raw
            rows.append({"date": day, "ticker": "T", "direction": direction,
                         "features": {"scan_day_ret": dr}, "outcomes": {"ret_20d": llm}})
    rows.append({"date": "2026-03-01", "ticker": "X", "direction": None, "features": {}, "outcomes": {}})
    led = tmp_path / "led"
    led.mkdir()
    (led / "2026-01.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    res = fm.llm_value_add(led, horizons=(20,))["h20"]
    assert res["n"] == 160 and res["verdict"] == "llm_beats_baseline"
    assert res["llm_mean"] > 0.02 and abs(res["baseline_mean"]) < 0.01

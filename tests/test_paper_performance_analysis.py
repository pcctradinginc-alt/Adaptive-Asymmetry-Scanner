"""Audit P1-9: Ursachenanalyse nutzt nur zuverlässige Paper-Outcomes und
vergleicht die MC-Trefferwahrscheinlichkeit mit der realisierten Quote."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "ppa", Path(__file__).resolve().parent.parent / "scripts" / "paper_performance_analysis.py")
ppa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ppa)


def _t(outcome, hr, strat="LONG_CALL", month="2026-04-10", approx=False):
    t = {"outcome": outcome, "simulation": {"hit_rate": hr}, "strategy": strat, "entry_date": month,
         "features": {"impact": 5, "surprise": 4}}
    if approx:
        t["outcome_method_reconstructed"] = "delta_approx"
    return t


def test_reliable_only_and_mc_calibration():
    hist = {"closed_trades": [_t(0.5, 0.9), _t(-0.5, 0.9), _t(-0.4, 0.9), _t(2.0, 0.9, approx=True),
                              _t(-0.6, 0.6, "BULL_CALL_SPREAD"), {"outcome": None}]}
    r = ppa.analyse(hist)
    assert r["n_closed"] == 5 and r["n_reliable"] == 4
    hi = r["mc_hit_rate_calibration"][">=0.75"]
    assert hi["n"] == 3 and hi["predicted_hit_rate"] == 0.9 and abs(hi["win_rate"] - 0.333) < 1e-3
    assert r["by_strategy"]["BULL_CALL_SPREAD"]["profit_factor"] == 0.0
    assert ppa.render(r).startswith("# Paper-Performance")

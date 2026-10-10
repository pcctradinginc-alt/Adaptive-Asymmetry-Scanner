"""Target-Hit Score (2026-10-09): simulation.hit_rate ist ein unkalibriertes Ranking-Signal (Underlying
erreicht Kursziel), keine Gewinnwahrscheinlichkeit. Nur Bezeichnung geändert – Zahl und Logik identisch."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from modules import email_reporter as er
from modules import score_labels as sl
from modules.outcomes import RELIABILITY_DEFINITION

ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN = [r"\bHit[- ]Rate\b", r"Predicted Hit Rate", r"Win Probability", r"Probability of Profit"]


def _proposal(hit=0.702):
    from modules.reporter import compute_exit_rules
    p = {"ticker": "ABC", "strategy": "LONG_CALL",
         "trade_score": {"total": 90, "grade": "A", "best_argument_for": "pro", "best_argument_against": "contra"},
         "deep_analysis": {"direction": "BULLISH", "impact": 6, "surprise": 5, "catalyst_confidence": 7,
                           "time_to_materialization": "4-8 Wochen"},
         "simulation": {"hit_rate": hit, "n_paths": 10000, "current_price": 100.0, "target_price": 110.0,
                        "sigma": 0.02, "alpha": 0.001},
         "mc_hit_rate": hit, "features": {"impact": 6, "surprise": 5, "mismatch": 2.0},
         "option": {"strike": 105, "expiry": "2026-12-18", "dte": 77, "bid": 4.0, "ask": 4.2},
         "roi_analysis": {"delta": 0.55, "theta_daily_pct": 0.02, "vega_loss": 0.05, "breakeven": 109.2,
                          "breakeven_pct": 0.04},
         "implied_move_pct": 8.0, "model_move_pct": 14.0, "edge_vs_implied": 6.0}
    p["exit_rules"] = compute_exit_rules(p)
    return p


def _no_probability_claim(text: str):
    for pat in FORBIDDEN:
        assert not re.search(pat, text), pat
    # "Gewinnwahrscheinlichkeit" nur verneint ("keine (kalibrierte) Gewinnwahrscheinlichkeit")
    for m in re.finditer(r"Gewinnwahrscheinlichkeit", text):
        assert re.search(r"keine( kalibrierte)? $", text[max(0, m.start() - 20):m.start()]), text[m.start() - 40:m.end()]


@pytest.fixture()
def outbox(monkeypatch):
    sent = []
    monkeypatch.setattr(er, "_send_smtp", lambda subject, html: sent.append((subject, html)))
    monkeypatch.setattr(er, "_external_context_html", lambda: "<!--ext-->")
    return sent


def test_daily_report_labels_score_not_probability(tmp_path):
    from modules.reporter import Reporter
    Reporter(tmp_path).save("2026-10-08", [_proposal()], {"model_weights": {}})
    md = (tmp_path / "2026-10-08.md").read_text()
    assert "Target-Hit Score (uncalibrated):** 70.2%" in md and "Ranking signal only" in md
    assert sl.TARGET_HIT_EXPLANATION in md
    _no_probability_claim(md)
    js = json.loads((tmp_path / "2026-10-08.json").read_text())
    assert js["proposals"][0]["simulation"]["hit_rate"] == 0.702              # Zahl unverändert gespeichert


def test_trade_mail_uses_target_hit_label(outbox):
    er.send_email([_proposal()], "2026-10-08", {})
    html = outbox[-1][1]
    assert "Target-Hit Score" in html and "(uncalibrated)" in html and "70%" in html
    _no_probability_claim(re.sub(r"<[^>]+>", " ", html))


def test_score_text_and_status_uncalibrated_without_forward_data():
    assert sl.calibration_status(None) == "uncalibrated"
    pp = {"reliability_definition": RELIABILITY_DEFINITION, "calibration_oos": {"n_evaluated": 0}}
    assert sl.calibration_status(pp) == "uncalibrated"
    txt = sl.score_text(0.72, pp)
    assert txt == "Target-Hit Score: 72 % (uncalibrated) – Ranking signal only"
    # historische Approximation ohne gültiges Artefakt zählt nie als Kalibrierung
    assert sl.calibration_status({"calibration_oos": {"n_evaluated": 500, "calibrated_better": True}}) == "uncalibrated"


def test_explanation_wording():
    e = sl.TARGET_HIT_EXPLANATION
    assert "keine kalibrierte Gewinnwahrscheinlichkeit des Options-Trades" in e
    assert "nicht automatisch höhere reale Gewinnchance" in e


def test_reporting_sources_have_no_probability_labels():
    for f in ("modules/email_reporter.py", "modules/reporter.py", "reports/weekly.py"):
        src = (ROOT / f).read_text(encoding="utf-8")
        assert "MC Hit-Rate" not in src and "**Hit-Rate:**" not in src, f
        assert "Kalibrierte Wahrscheinlichkeit" not in src and "MC-Band-Wahrscheinlichkeit" not in src, f


def test_numeric_score_and_gates_unchanged():
    """Nur Labels: Quick-/Final-MC-Gates und der LLM-Prompt nutzen weiter simulation.hit_rate unverändert."""
    pipe = (ROOT / "pipeline.py").read_text(encoding="utf-8")
    assert "if hit_rate < mc_threshold:" in pipe and "if hit_rate < final_threshold:" in pipe
    da = (ROOT / "modules/deep_analysis.py").read_text(encoding="utf-8")
    assert "Hit-Rate: {mc_hit_rate:.1%}" in da                          # Prompt unverändert (Produktionsinput)
    assert sl.score_text(0.702, None, digits=1).startswith("Target-Hit Score: 70.2 %")

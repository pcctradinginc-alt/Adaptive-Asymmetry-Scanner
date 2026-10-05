"""Reporting-/Integrity-Regression (2026-10-05): eine Reliability-Definition überall, zwei getrennte
Kalibrierungs-Ebenen. Bestand nach Owner-Entscheidung: RELIABLE 2 / UNKNOWN 77 / RECONSTRUCTED 40 / APPROXIMATED 0."""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import pytest

from modules import inquiry as iq
from modules import learning_health as lh
from modules.outcomes import RELIABILITY_DEFINITION, artifact_is_current, class_counts
from reports import weekly

TODAY = date(2026, 10, 5)
COUNTS = "RELIABLE 2 · UNKNOWN 77 · RECONSTRUCTED 40 · APPROXIMATED 0"


def _w(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))


def _trade(i, **kw):
    t = {"ticker": f"T{i}", "entry_date": "2026-06-01", "close_date": "2026-09-20", "outcome": -0.3 if i % 2 else 0.4,
         "strategy": "LONG_CALL", "simulation": {"hit_rate": 0.7}, "features": {"impact": 5, "surprise": 4}}
    t.update(kw)
    return t


def _history():
    closed = ([_trade(i, outcome_method="spread_quote") for i in range(2)]
              + [_trade(100 + i) for i in range(77)]                                          # Altbestand ohne Methode
              + [_trade(200 + i, outcome_method_reconstructed="delta_approx") for i in range(40)])
    return {"closed_trades": closed, "active_trades": [], "shadow_trades": []}


@pytest.fixture()
def report(tmp_path):
    """Echter Bestand + VERALTETE Artefakte (vor der Reklassifizierung erzeugt, ohne Definitions-Stempel)."""
    root = tmp_path
    _w(root / "outputs/history.json", _history())
    _w(root / "outputs/research/failure_analysis.json",                     # alte Definition: 93 "reliable"
       {"reliable": {"primary": {"signal_wrong": {"share": 0.6, "n": 56}, "timing_too_early": {"share": 0.4, "n": 37}}},
        "unreliable": {"primary": {}}})
    _w(root / "outputs/intelligence/contract_proposals.json",               # alte Definition: 79 "verlässliche"
       {"n_trades": 79, "candidates_tested": 6, "proposals": [], "rejected": [{}] * 6})
    _w(root / "outputs/research/ml_research.json",                          # historisch "kalibriert=True"
       {"generated": "2026-10-03", "calibration": {"coverage": 0.8, "target": 0.8, "interval_calibrated": True,
                                                   "brier": 0.2, "brier_base": 0.21}})
    _w(root / "outputs/research/paper_performance_analysis.json",           # 0 prospektive Kalibrier-Outcomes
       {"reliability_definition": RELIABILITY_DEFINITION, "n_closed": 119, "n_reliable": 2,
        "mc_hit_rate_calibration": {"0.65-0.75": {"n": 1, "predicted_hit_rate": 0.7, "win_rate": 0.0}},
        "calibration_oos": {"n_evaluated": 0, "calibrated_better": None}})
    _w(root / "outputs/research/hypothesis_db.json",                        # damit §9 (Failure-Analyse) rendert
       {"hypotheses": {"H1": {"id": "H1", "title": "t", "status": "rejected", "created_at": "2026-10-01"}}})
    _w(root / "outputs/intelligence/promotion_state.json",                  # damit §18 (Vorschläge) rendert
       {"hypotheses": {"PROM-X@v1": {"state": "PROSPECTIVE_CHALLENGER", "evidence": {}}}})
    data = weekly.collect(root, TODAY, state_path=root / "st.json")
    return data, weekly.render_text(data)


def test_report_shows_exact_outcome_classes(report):
    data, text = report
    assert data["forward"]["outcome_classes"] == {"RELIABLE": 2, "UNKNOWN": 77, "RECONSTRUCTED": 40, "APPROXIMATED": 0}
    assert COUNTS in text
    assert class_counts(_history()["closed_trades"]) == {"RELIABLE": 2, "UNKNOWN": 77, "RECONSTRUCTED": 40}


def test_no_section_counts_unknown_or_reconstructed_as_reliable(report):
    _, text = report
    # Jede Zahl, die als reliable/zuverlässig/verlässlich bezeichnet wird, darf den RELIABLE-Bestand (2) nicht übersteigen
    pat = re.compile(r"(\d+)\s+(?:RELIABLE[- ]?(?:Trades|Outcomes)|verlässlich\w*|zuverlässig\w*|reliable)", re.I)
    for line in text.splitlines():
        if "NICHT RELIABLE" in line or "EXPLORATIV" in line:
            continue
        for m in pat.finditer(line):
            assert int(m.group(1)) <= 2, line
    assert "79 verlässliche" not in text and "79 RELIABLE" not in text
    # Forward-Kennzahl "nur RELIABLE" nutzt höchstens die 2 RELIABLE-Trades
    assert all(w["reliable"]["closed"] <= 2 for w in report[0]["forward"]["windows"])


def test_failure_analysis_reliable_only_or_explicitly_exploratory(report):
    _, text = report
    lines = [l for l in text.splitlines() if l.startswith("Failure-Analyse")]
    assert lines
    for l in lines:
        assert ("nur RELIABLE" in l) or ("EXPLORATIV" in l and "UNKNOWN/RECONSTRUCTED" in l and "NICHT RELIABLE" in l), l
    assert any("EXPLORATIV" in l and "n=93" in l for l in lines)            # veraltetes Artefakt sichtbar als explorativ


def test_failure_analysis_current_artifact_is_reliable_only(tmp_path, monkeypatch):
    from modules import trade_memory as tm
    h = _history()
    hp = tmp_path / "history.json"
    hp.write_text(json.dumps(h))
    monkeypatch.setattr(tm, "_load_regimes_for", lambda dates: {})          # keine Regime-Abhängigkeit im Test
    agg = tm.run(history_path=hp, out_dir=tmp_path / "out")["aggregate"]
    n_rel = sum(v["n"] for v in agg["reliable"]["primary"].values())
    assert agg["reliability_definition"] == RELIABILITY_DEFINITION and n_rel <= 1   # 1 RELIABLE-Verlierer
    assert agg["outcome_classes"] == {"RELIABLE": 2, "UNKNOWN": 77, "RECONSTRUCTED": 40}
    md = (tmp_path / "out" / "failure_analysis.md").read_text()
    assert "Zuverlässige Verlusttrades" not in md and "RELIABLE Verlusttrades" in md


def test_hypothesis_reporting_never_calls_79_reliable(report):
    _, text = report
    line = next(l for l in text.splitlines() if l.startswith("NEUE HYPOTHESEN-VORSCHLÄGE"))
    assert "79" in line and "NICHT RELIABLE" in line and "verlässlich" not in line
    # aktuelle Generatoren: nur RELIABLE-Population, mit Definitions-Stempel
    from modules import abstention_proposals as ap
    res = ap.propose(_history())
    assert res["n_trades"] == 2 and res["reliability_definition"] == RELIABILITY_DEFINITION
    # Research-Inquiry: Failure-Befunde nur aus Artefakten mit gültiger Definition
    stale = {"reliable": {"primary": {"signal_wrong": {"share": 0.6, "n": 56}}}}
    assert not any(a["id"].startswith("failure:") for a in iq.anomalies({}, None, stale, {}))


def test_paper_performance_and_calibration_population_reliable_only():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import paper_performance_analysis as ppa
    r = ppa.analyse(_history())
    assert r["n_reliable"] == 2 and r["reliability_definition"] == RELIABILITY_DEFINITION
    assert sum(v["n"] for v in r["mc_hit_rate_calibration"].values()) <= 2


def test_historical_and_forward_calibration_have_distinct_labels(report):
    _, text = report
    assert lh.HISTORICAL_WF_LABEL == "HISTORICAL_WALK_FORWARD_CALIBRATION"
    assert lh.LIVE_FORWARD_LABEL == "LIVE_FORWARD_CALIBRATION"
    hist = [l for l in text.splitlines() if "HISTORICAL_WALK_FORWARD_CALIBRATION:" in l and "kalibriert=True" in l]
    assert hist and all("KEINE Forward-Evidenz" in l and "LIVE_FORWARD" not in l for l in hist)
    # "calibrated=True" darf nirgends ohne historisches Label stehen
    for l in text.splitlines():
        if re.search(r"kalibriert=True|calibrated=True", l):
            assert "HISTORICAL_WALK_FORWARD_CALIBRATION" in l and "historisch" in l, l
    assert "Calibration (Forward)" not in text and "Calibration status" not in text


def test_zero_forward_outcomes_means_need_more_data_despite_historical(report):
    _, text = report
    lf = lh.live_forward_calibration({"reliability_definition": RELIABILITY_DEFINITION,
                                      "calibration_oos": {"n_evaluated": 0}})
    assert lf["status"] == "NEED_MORE_DATA / UNCALIBRATED" and lf["n"] == 0
    assert lh.live_forward_calibration(None)["status"] == "NEED_MORE_DATA / UNCALIBRATED"
    # veraltetes Artefakt ohne Stempel zählt nie als Forward-Evidenz
    assert lh.live_forward_calibration({"calibration_oos": {"n_evaluated": 500, "calibrated_better": True}})["n"] == 0
    live = [l for l in text.splitlines() if "LIVE_FORWARD_CALIBRATION:" in l]
    assert live and all("NEED_MORE_DATA / UNCALIBRATED" in l for l in live)


def test_artifact_check():
    assert not artifact_is_current({"n_trades": 79})
    assert not artifact_is_current({"reliability_definition": RELIABILITY_DEFINITION}, 79, 2)
    assert artifact_is_current({"reliability_definition": RELIABILITY_DEFINITION}, 2, 2)


def test_production_logic_unchanged():
    """Nur Reporting/Artefakt-Kennzeichnung geändert: Reliability-Definition und Produktionspfade identisch."""
    from modules.outcomes import is_reliable_outcome, outcome_class
    cases = [({"outcome": 0.1, "outcome_method": "spread_quote"}, True, "RELIABLE"),
             ({"outcome": 0.1}, False, "UNKNOWN"),
             ({"outcome": 0.1, "outcome_method_reconstructed": "delta_approx"}, False, "RECONSTRUCTED"),
             ({"outcome": 0.1, "outcome_method": "delta_approx"}, False, "APPROXIMATED")]
    for t, rel, cls in cases:
        assert is_reliable_outcome(t) is rel and outcome_class(t) == cls
    root = Path(__file__).resolve().parent.parent
    prod = ["pipeline.py", "modules/production_intelligence_adapter.py", "modules/promotion_controller.py",
            "modules/hypothesis_contract.py", "config/promotion_policy.yaml", "config.yaml"]
    for f in prod:
        assert "reliability_definition" not in (root / f).read_text(encoding="utf-8") if (root / f).exists() else True

"""
tests/test_config_thresholds.py

Stellt sicher, dass alle Gate-Schwellen zentral in config.yaml (Sektion
`gates`) liegen und dass die alten hart-codierten Literale aus pipeline.py
und modules/email_reporter.py verschwunden sind.
"""

from pathlib import Path

from modules.config import cfg

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_gates_section_has_all_keys():
    gates = cfg.gates
    assert gates.impact_min == 4
    assert gates.surprise_min == 3
    assert gates.mismatch_max == 7.0
    assert gates.final_mc_min_short == 0.45
    assert gates.final_mc_min_long == 0.50
    assert gates.trade_score_min == 55
    assert gates.shadow_score_min == 40


def test_pipeline_has_no_hardcoded_gate_literals():
    source = (REPO_ROOT / "pipeline.py").read_text()
    forbidden = [
        "impact >= 4 and surprise >= 3",
        "score >= 55",
        "0.45 if final_dte",
    ]
    for pattern in forbidden:
        assert pattern not in source, f"hard-coded gate literal still present: {pattern!r}"


def test_email_reporter_has_no_hardcoded_score_gate():
    source = (REPO_ROOT / "modules" / "email_reporter.py").read_text()
    assert ">= 65" not in source

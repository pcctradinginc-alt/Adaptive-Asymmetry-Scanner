"""Forschungsumfang (Aktionsplan 2026-11-09, Punkt 7): eingefrorene Generatoren und eingefrorener Code.

Geprüft: jeder eingefrorene Generator ist im ML-Research-Workflow tatsächlich hinter der Scope-Bedingung (der
Scope-Schritt läuft vorher); Auswertungsschritte registrierter Hypothesen bleiben ungebremst; die Gesundheitsanzeige
meldet FROZEN statt "veraltet"; eingefrorener Code ist per SHA-256 gepinnt (kein stiller Ausbau).
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

from modules import learning_health, research_scope

WF = Path(".github/workflows/ml_research.yml")


def _step_list():
    jobs = yaml.safe_load(WF.read_text(encoding="utf-8"))["jobs"]
    return next(iter(jobs.values()))["steps"]


def test_frozen_generators_are_skipped_after_the_scope_step_and_evaluations_keep_running():
    steps = _step_list()
    ids = [s.get("id") for s in steps]
    assert "research_scope" in ids
    scope_at = ids.index("research_scope")
    frozen = research_scope.frozen_steps()
    assert set(frozen) == {"factory_plan", "alpha_explore", "director", "feature_discovery_v3",
                           "feature_director_v1", "causal", "next_intel"}
    for sid in frozen:
        assert sid in ids, f"eingefrorener Schritt {sid} fehlt im Workflow"
        st = steps[ids.index(sid)]
        assert ids.index(sid) > scope_at
        assert f"steps.research_scope.outputs.frozen_{sid} != 'true'" in str(st.get("if"))
    for keep in ("error_research", "factory_eval", "factory_forward", "alpha_factor_forward", "alpha_option_forward",
                 "option_learner", "ledger_registration", "champion_protocol", "feature_validation_v1"):
        st = steps[ids.index(keep)]
        assert "research_scope" not in str(st.get("if") or ""), f"Auswertungsschritt {keep} darf nicht gebremst sein"


def test_cli_writes_github_outputs(tmp_path, monkeypatch):
    out = tmp_path / "gh_out"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    assert research_scope.main(["--github-output"]) == 0
    lines = out.read_text().split()
    assert "frozen_factory_plan=true" in lines and len(lines) == len(research_scope.frozen_steps())


def test_learning_health_reports_frozen_generation_and_removed_rl(tmp_path):
    rep = learning_health.assess(root=tmp_path, now=datetime(2026, 11, 10, tzinfo=timezone.utc))
    assert rep["paths"]["research_generation"]["status"] == "FROZEN"
    assert "rl" not in rep["paths"]                                     # RL/PPO entfernt: kein Pfad mehr
    assert "research_generation" not in rep["stalled_or_broken"]


def test_frozen_code_is_pinned_and_unchanged():
    scope = research_scope.load()
    code = scope.get("frozen_code") or {}
    assert len(code) >= 10, "eingefrorener Code muss gepinnt sein"
    assert research_scope.code_drift() == {}, (
        "Eingefrorene Engine geändert – Ausbau ist ausgesetzt (config/research_scope.yaml). Bewusste Änderung nur "
        "mit aktualisiertem Hash und Begründung im PR.")
    assert all(re.fullmatch(r"[0-9a-f]{64}", v) for v in code.values())

"""Audit 2026-11-07 (P6): veraltete Forschungsstatus.

#165 ordnete nur per Textähnlichkeit gesperrte Hypothesen NOT_TESTED zu – aber nur in research_lab.canonical_status.
Die Leser lasen das GESPEICHERTE Feld: outputs/research/hypothesis_db.json (Stand 03.11.) führt 15 solche Einträge
noch als canonical_status REJECTED, und research.yml schreibt die Datei erst am 02.12. neu. Live vorher:
research_memory 62 REJECTED; nachher 47 REJECTED + 15 NOT_TESTED. Echte Testergebnisse bleiben unverändert.
"""
from __future__ import annotations

import json

from modules import hypothesis_contract as hc
from modules import inquiry
from modules import meta_cognition as mc
from modules import research_lab as rl
from modules import research_memory as rm

BLOCKED = {"id": "H_B", "status": "blocked_similar_to_rejected", "canonical_status": "REJECTED", "source": "director",
           "signal": "rev_1m * step(vix - 20)", "similar_to": "H_R"}
REJECTED = {"id": "H_R", "status": "rejected", "canonical_status": "REJECTED", "source": "director",
            "signal": "rev_1m * step(-spy_trend_200)"}
ACCEPTED = {"id": "H_A", "status": "accepted", "canonical_status": "ACCEPTED", "source": "config",
            "walk_forward": {"base": {"mean": 0.01}}}
DB = {"hypotheses": {r["id"]: r for r in (BLOCKED, REJECTED, ACCEPTED)}}


def test_one_status_rule_for_lab_and_readers():
    assert rm.hypothesis_status(BLOCKED) == rl.canonical_status(BLOCKED) == "NOT_TESTED"
    assert rm.hypothesis_status(REJECTED) == "REJECTED" and rm.hypothesis_status(ACCEPTED) == "EXPLORATORY"
    assert rm.hypothesis_status({"status": "prior_result"}) == "REJECTED"
    assert rm.hypothesis_status({"status": "passed_pending_locked"}) == "RETEST_LATER"
    assert rm.hypothesis_status(None) == "INCONCLUSIVE" and rl.CANONICAL is rm.HYPOTHESIS_CANONICAL


def test_research_memory_does_not_count_untested_as_rejected(tmp_path):
    p = tmp_path / "hypothesis_db.json"
    p.write_text(json.dumps(DB))
    st = sorted(e["status"] for e in rm.collect({"hypothesis_db": p}))
    assert st == ["EXPLORATORY", "NOT_TESTED", "REJECTED"]
    assert "NOT_TESTED" not in rm.TESTED                              # zählt nicht als geprüft/widerlegt


def test_meta_cognition_research_value_skips_untested():
    tr = mc.research_value(DB, {}, {})["director"]
    assert tr["experiments"] == 1 and tr["rejected"] == 1             # vorher 2 Experimente, 2 verworfen


def test_inquiry_and_contract_inventory_use_the_canonical_status(tmp_path, monkeypatch):
    director = {"candidates": [{"research_id": "H_B", "source_finding": "x", "question": "q"}]}
    out = inquiry.explanations({"id": "x", "kind": "anomaly", "keys": ["x"]}, director, DB, None, None, None, [])
    assert [e["status"] for e in out if e["id"] == "H_B"] == ["NOT_TESTED"]
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / "research").mkdir(parents=True)
    (tmp_path / "outputs" / "research" / "hypothesis_db.json").write_text(json.dumps(DB))
    inv = {r["hypothesis_id"]: r["status"] for r in hc.research_inventory()}
    assert inv == {"H_B": "NOT_TESTED", "H_R": "REJECTED", "H_A": "EXPLORATORY"}

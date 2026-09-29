"""Das Bewertungsprotokoll liegt außerhalb der Kontrolle jedes Research-Agenten:
jede Änderung an config/research_protocol.yaml muss hier sichtbar den Hash
mitändern (Review durch CODEOWNERS). Zusätzlich Mindeststrenge, die auch bei
geändertem Hash nie unterschritten werden darf."""
from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PROTOCOL = ROOT / "config" / "research_protocol.yaml"
PINNED_SHA256 = "cf0427cef935c41085f6ae4e685633222cd0b1279298bd050a0e2002c93113a3"


def test_protocol_hash_is_pinned():
    assert hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() == PINNED_SHA256, \
        "Bewertungsprotokoll geändert – nur zulässig, um es strenger zu machen (Review durch CODEOWNERS)"


def test_protocol_minimum_strictness():
    p = yaml.safe_load(PROTOCOL.read_text())
    assert p["costs"]["base_per_side"] >= 0.0010 and p["costs"]["stress_per_side"] >= 0.0025
    assert p["multiple_testing"]["fdr_q"] <= 0.10
    a = p["hypothesis_acceptance"]
    assert a["min_t_months"] >= 2.0 and a["min_years_positive"] >= 0.6 and a["locked_must_be_positive"]
    c = p["promotion_criteria"]
    assert c["wf_min_t_months"] >= 2.0 and c["fwd_min_cohorts"] >= 26 and c["locked_positive"]
    assert p["periods"]["discovery_end"] <= f"{p['periods']['first_test_year']}-01-01"


def test_protected_paths_in_codeowners():
    owners = (ROOT / ".github" / "CODEOWNERS").read_text()
    for path in ("config/research_protocol.yaml", "tests/test_research_protocol.py", "modules/research_lab.py",
                 "modules/ml_research.py", "config/model_registry.yaml"):
        assert path in owners, path

"""Document real, research-only PortWatch/NCEI downstream users."""
from pathlib import Path

import yaml

from modules import source_health as sh

ROOT = Path(__file__).resolve().parents[1]


def test_shadow_consumers_are_visible_without_inventing_trading_dependencies():
    cfg = yaml.safe_load((ROOT / "config/source_health.yaml").read_text())
    deps = sh.downstream(cfg)
    for sid in ("imf_portwatch_ports", "imf_portwatch_chokepoints",
                "imf_portwatch_disruptions", "ncei_normals"):
        assert deps[sid]["other"]
        assert any("external_context (SHADOW)" in x for x in deps[sid]["other"])
        assert deps[sid]["decisions"] == []
        assert deps[sid]["models"] == []


def test_weekly_ncei_report_is_research_only_not_a_core_precondition():
    wf = (ROOT / ".github/workflows/ml_research.yml").read_text()
    assert "python -m modules.ncei_missingness_audit --as-of" in wf
    assert "ncei_missingness=${{ steps.ncei_missingness.outcome }}" in wf
    assert "id: ncei_missingness" in wf

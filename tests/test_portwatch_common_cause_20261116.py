"""Regression: common-cause PortWatch provider incident must not disable the risk gate."""
from modules import source_health as sh


def test_portwatch_provider_correlation_does_not_count_as_independent_upstreams():
    def source():
        return dict(status="BROKEN", criticality="IMPORTANT", kind="registry",
                    downstream_dependencies={"features": [], "decisions": [], "hypotheses": []})
    snap = {"sources": {
        "imf_portwatch_ports": source(),
        "imf_portwatch_chokepoints": source(),
    }, "features": {}}
    sm = sh.data_safe_mode(snap, sh.load_config())
    assert sm["active"] is True  # No silently relaxing data safety
    assert "2 wichtige Quellen ausgefallen aus 1 unabhängigen Ausfall-Domänen" in " ".join(sm["global_reasons"])
    assert "imf_portwatch_arcgis" in " ".join(sm["global_reasons"])

"""Knowledge Graph: Kanten nur mit Quelle/Evidenz/Konfidenz, keine erfundenen
Kanten, Versionierung (Hash + append-only), Propagation, Widersprüche,
ticker_evidence, event_impact. Kein Netz."""
from __future__ import annotations

import json

import pytest

from modules import knowledge_graph as kg

SECTOR_MAP = {"CAT": "Industrials", "XOM": "Energy", "JPM": "Financial Services"}
IND_EXP = {"industries": {"Industrials": {"road_freight_relevance": "MEDIUM", "maritime_relevance": "LOW",
                                          "weather_relevance": "NONE"}},
           "sector_fallback": {"Industrials": "Industrials", "Energy": "Unbekannt"}}


def _causal(sign_a=1, sign_b=1):
    ev = {"predictive_lead": "HIGH", "conditional_robustness": "MEDIUM", "stability": "HIGH", "replication": "MEDIUM",
          "economic_plausibility": "HIGH", "causal_confidence": "MODERATE"}
    return {"generated": "2026-09-29T00:00:00+00:00", "relations": [
        {"id": "CAUSAL_HYPOTHESIS_001", "driver": "indpro_yoy", "target": "XLI", "horizon_weeks": 13,
         "level": "causal_hypothesis", "evidence": ev, "observed_sign": sign_a, "stats": {"q_bh": 0.01}},
        {"id": "REL_002", "driver": "freight_yoy", "target": "XLI", "horizon_weeks": 13,
         "level": "temporally_leading", "evidence": {**ev, "causal_confidence": "LOW"}, "observed_sign": sign_b,
         "stats": {"q_bh": 0.04}},
        {"id": "REL_003", "driver": "wti_63d", "target": "XLE", "horizon_weeks": 4, "level": "correlation",
         "evidence": {**ev, "causal_confidence": "LOW"}, "observed_sign": 1, "stats": {}}]}


def _graph(**kw):
    return kg.build_graph(sector_map=SECTOR_MAP, industry_exposure=IND_EXP, causal=kw.get("causal", _causal()),
                          now="2026-09-29T00:00:00+00:00")


def test_every_edge_has_source_evidence_confidence_and_no_invented_edges():
    g = _graph()
    assert kg.validate_graph(g) == []
    types = {e["type"] for e in g["edges"]}
    assert "exposed_to" in types and "belongs_to_sector" in types and "historically_leads" in types
    # NONE-Relevanz -> keine Kante; Korrelation -> keine Kausalkante; unbekannte Fallback-Industrie -> keine Kante
    assert not any(e["to"] == "Theme:weather" for e in g["edges"])
    assert not any("wti_63d" in e["from"] for e in g["edges"])
    assert not any(e["type"] == "maps_to_industry" and e["from"] == "Sector:Energy" for e in g["edges"])
    low = [e for e in g["edges"] if e["confidence"] == "LOW"]
    assert low and all(e["uncertain"] for e in low if e["evidence_type"] == "curated_config")


def test_edge_without_source_is_rejected():
    b = kg._Builder("2026-01-01")
    b.node("Sector:A", "Sector")
    b.node("Sector:B", "Sector")
    with pytest.raises(ValueError):
        b.edge("x", "Sector:A", "Sector:B", source="", confidence="HIGH", evidence_type="curated_config", uncertain=False)
    with pytest.raises(ValueError):
        b.edge("x", "Sector:A", "Sector:B", source="s", confidence="HIGH", evidence_type="llm_guess", uncertain=False)


def test_versioning_hash_and_append_only(tmp_path):
    g1 = _graph()
    kg.save(g1, tmp_path)
    g2 = _graph(causal=_causal(sign_b=-1))
    assert g2["version"] != g1["version"]
    kg.save(g2, tmp_path)
    lines = (tmp_path / "knowledge_graph_versions.jsonl").read_text().splitlines()
    assert len(lines) >= 2 and json.loads(lines[0])["version"] == g1["version"]
    assert kg.load_version(g1["version"], tmp_path) is not None
    assert _graph()["version"] == g1["version"]                                  # deterministisch


def test_propagation_and_ticker_evidence():
    g = _graph()
    hits = kg.propagate(g, "Indicator:indpro_yoy", max_depth=3)
    nodes = {h["node"] for h in hits}
    assert "Index:XLI" in nodes and "Sector:Industrials" in nodes
    ev = kg.ticker_evidence(g, "CAT")
    assert ev["leading_indicators"] and ev["contradictions"] == []
    assert kg.ticker_evidence(g, "UNKNOWN_TICKER")["leading_indicators"] == []


def test_contradictions_detected():
    g = _graph(causal=_causal(sign_a=1, sign_b=-1))
    ev = kg.ticker_evidence(g, "CAT")
    assert ev["contradictions"], ev


def test_event_impact_direction():
    g = _graph()
    imp = kg.event_impact(g, "indpro_yoy", 1)
    assert imp and any("XLI" in str(i) or "Industrials" in str(i) for i in imp)

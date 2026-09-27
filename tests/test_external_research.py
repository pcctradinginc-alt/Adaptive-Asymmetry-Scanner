"""
tests/test_external_research.py – modules/external/research.py

Prüft:
  - effective_n / independent_date_count-Berechnung (Design-Effekt, rho-Clipping).
  - Reifegrad-Gating der Hypothesen-Vorschläge (n, unabhängige Tage, Spanne, KI).
  - Budget-Gate (max_external_hypotheses_per_month) + max. 2/Feature.
  - Konkurrierende Hypothesen (beide Richtungen) können unabhängig auftreten.
  - NIEMALS ein Write auf challengers.yaml/config.yaml.
"""

import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.external import research


def _row(d: date, outcome: float, freight_state="EXPANSION", relation="SUPPORT",
         agreement="agreement", ticker="T"):
    return {
        "date": d.isoformat(),
        "ticker": ticker,
        "direction": "BULLISH",
        "catalyst_type": "EARNINGS",
        "outcomes": {"real_strat_ret_45d": outcome},
        "external": {
            "states": {"global_freight_state": freight_state},
            "primitives": {"road_shipping_agreement": agreement},
            "relation": {"relation": relation, "materiality": 0.6},
            "ticker_exposure": {"industry": "Industrials", "road_freight_relevance": "HIGH",
                                 "maritime_relevance": "MEDIUM"},
        },
    }


def _many_rows(n_days: int, outcome: float, **kwargs) -> list[dict]:
    base = date(2026, 1, 1)
    rows = []
    for i in range(n_days):
        rows.append(_row(base + timedelta(days=i), outcome, ticker=f"T{i % 5}", **kwargs))
    return rows


def test_effective_n_no_intra_date_correlation_equals_n():
    # One row per date -> single-row clusters -> rho should be 0 (no within-cluster
    # variance to compare against between-cluster), effective_n == n.
    rows = _many_rows(40, 0.1)
    eff = research.effective_n(rows, "outcomes.real_strat_ret_45d")
    assert eff["n"] == 40
    assert eff["independent_date_count"] == 40
    assert eff["effective_n"] == 40.0
    assert eff["rho"] == 0.0


def test_effective_n_shrinks_with_multiple_rows_per_date_and_between_date_variance():
    # 10 distinct dates, 4 rows/day: rows on the SAME day share the same
    # outcome, but outcomes differ strongly ACROSS days -> high intra-day
    # correlation -> effective_n well below n.
    base = date(2026, 1, 1)
    rows = []
    for day_i in range(10):
        day_outcome = 0.5 if day_i % 2 == 0 else -0.5
        for j in range(4):
            rows.append(_row(base + timedelta(days=day_i), day_outcome, ticker=f"T{j}"))
    eff = research.effective_n(rows, "outcomes.real_strat_ret_45d")
    assert eff["n"] == 40
    assert eff["independent_date_count"] == 10
    assert eff["rho"] > 0.5   # near-perfect within-day agreement
    assert eff["effective_n"] < 40


def test_rho_is_clipped_to_0_1():
    rows = _many_rows(5, 0.1)
    rho = research.estimate_intra_date_rho(rows, "outcomes.real_strat_ret_45d")
    assert 0.0 <= rho <= 1.0


def test_analyze_group_empty():
    stats = research.analyze_group([], "outcomes.real_strat_ret_45d")
    assert stats["n"] == 0
    assert stats["mean"] is None


def test_maturity_gate_rejects_thin_data():
    catalog_path = Path(__file__).parent.parent / "config" / "external_hypotheses.yaml"
    rows = _many_rows(20, 0.1)  # n too small, span too small
    proposals = research.generate_hypothesis_proposals(rows, today=date(2026, 9, 27),
                                                         catalog_path=catalog_path)
    assert proposals == []


def test_maturity_gate_accepts_mature_data_and_never_writes_files(tmp_path, monkeypatch):
    catalog_path = Path(__file__).parent.parent / "config" / "external_hypotheses.yaml"
    rows = _many_rows(65, 0.10)

    challengers_path = Path(__file__).parent.parent / "challengers.yaml"
    config_path = Path(__file__).parent.parent / "config.yaml"
    before_challengers = challengers_path.read_bytes()
    before_config = config_path.read_bytes()

    proposals = research.generate_hypothesis_proposals(rows, today=date(2026, 9, 27),
                                                         catalog_path=catalog_path)
    assert len(proposals) >= 1
    for p in proposals:
        assert "retrospektiv/in-sample" in p["snippet"]
        assert "NICHT promotion-fähig" in p["snippet"]
        assert "start_date" in p["snippet"]

    # Never touched the real registry files.
    assert challengers_path.read_bytes() == before_challengers
    assert config_path.read_bytes() == before_config


def test_budget_gate_limits_number_of_proposals(tmp_path):
    catalog_path = Path(__file__).parent.parent / "config" / "external_hypotheses.yaml"
    rows = _many_rows(65, 0.10)

    cfg_yaml = tmp_path / "cfg.yaml"
    import modules.external.research as research_mod

    class _FakeCfg:
        pass

    def fake_research_config():
        return {"enable_hypothesis_generation": True, "max_external_hypotheses_per_month": 1}

    orig = research_mod._research_config
    research_mod._research_config = fake_research_config
    try:
        proposals = research.generate_hypothesis_proposals(rows, today=date(2026, 9, 27),
                                                             catalog_path=catalog_path)
        assert len(proposals) <= 1
    finally:
        research_mod._research_config = orig


def test_disabled_research_yields_no_proposals():
    catalog_path = Path(__file__).parent.parent / "config" / "external_hypotheses.yaml"
    rows = _many_rows(65, 0.10)
    import modules.external.research as research_mod

    def fake_research_config():
        return {"enable_hypothesis_generation": False}

    orig = research_mod._research_config
    research_mod._research_config = fake_research_config
    try:
        proposals = research.generate_hypothesis_proposals(rows, today=date(2026, 9, 27),
                                                             catalog_path=catalog_path)
        assert proposals == []
    finally:
        research_mod._research_config = orig


def test_competing_hypotheses_both_directions_can_fire_independently():
    catalog_path = Path(__file__).parent.parent / "config" / "external_hypotheses.yaml"
    expansion_rows = _many_rows(65, 0.10, freight_state="EXPANSION")
    contraction_rows = [
        {**r, "external": {**r["external"],
                            "states": {"global_freight_state": "CONTRACTION"}}}
        for r in _many_rows(65, -0.10, freight_state="CONTRACTION")
    ]
    rows = expansion_rows + contraction_rows

    proposals = research.generate_hypothesis_proposals(rows, today=date(2026, 9, 27),
                                                         catalog_path=catalog_path)
    ids = {p["hypothesis_id"] for p in proposals}
    # Both H1A (expansion -> outperformance) and H1B (contraction -> underperformance)
    # are independently eligible from the same catalogue/run.
    assert "H1A" in ids
    assert "H1B" in ids


def test_max_thresholds_per_feature_enforced():
    catalog_path = Path(__file__).parent.parent / "config" / "external_hypotheses.yaml"
    catalog = research.load_hypothesis_catalog(catalog_path)
    features = [h.get("feature") for h in catalog]
    # freight_state groups H1A + H1B — exactly MAX_THRESHOLDS_PER_FEATURE (2)
    assert features.count("freight_state") == research.MAX_THRESHOLDS_PER_FEATURE


def test_never_assumes_direction_catalog_has_competing_pairs():
    catalog_path = Path(__file__).parent.parent / "config" / "external_hypotheses.yaml"
    catalog = research.load_hypothesis_catalog(catalog_path)
    ids = {h["id"] for h in catalog}
    pairs = [("H1A", "H1B"), ("H2A", "H2B"), ("H3A", "H3B"), ("H4A", "H4B"),
             ("H7A", "H7B"), ("H9A", "H9B")]
    for a, b in pairs:
        assert a in ids and b in ids, f"{a}/{b} missing from catalogue"


def test_load_hypothesis_catalog_missing_file_returns_empty():
    assert research.load_hypothesis_catalog(Path("does/not/exist.yaml")) == []


def test_is_mature_row():
    assert research.is_mature_row({"outcomes": {"real_strat_ret_45d": 0.1}}, 45)
    assert research.is_mature_row({"outcomes": {"ret_45d": 0.1}}, 45)
    assert not research.is_mature_row({"outcomes": {}}, 45)
    assert not research.is_mature_row({}, 45)


def test_analyze_external_buckets_tolerates_missing_external_key():
    rows = [{"date": "2026-01-01", "outcomes": {"ret_45d": 0.1}}]  # no "external"
    buckets = research.analyze_external_buckets(rows)
    assert all(v == {} for v in buckets.values())

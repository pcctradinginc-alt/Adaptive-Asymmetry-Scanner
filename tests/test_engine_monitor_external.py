"""
tests/test_engine_monitor_external.py – modules/engine_monitor.py external
Lern-Health-Checks (Warnungen, kein Gate).
"""

import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules import engine_monitor


def _empty_history(**overrides) -> dict:
    hist = {
        "feature_stats": {}, "active_trades": [], "closed_trades": [],
        "model_weights": {"impact": 0.35, "mismatch": 0.45, "eps_drift": 0.20},
        "shadow_trades": [], "trailing_sim": [],
    }
    hist.update(overrides)
    return hist


def test_external_health_metrics_present_with_no_data():
    warnings: list[str] = []
    metrics = engine_monitor._check_external_health(_empty_history(), date(2026, 9, 27), warnings)
    assert "missingness" in metrics
    assert "source_health" in metrics
    assert "feature_stats_external" in metrics
    assert "promoted_dominance" in metrics


def test_external_missingness_warns_when_primitive_constant(monkeypatch):
    today = date(2026, 9, 27)
    rows = [
        {"date": (today - timedelta(days=i)).isoformat(),
         "external": {"primitives": {"suez_z": 0.5}}}
        for i in range(25)
    ]
    monkeypatch.setattr(engine_monitor, "_external_ledger_rows_last_30d", lambda t: rows)
    warnings: list[str] = []
    metrics = engine_monitor._check_external_missingness(rows, warnings)
    assert metrics["suez_z"]["constant"] is True
    assert metrics["suez_z"]["stale"] is True
    assert any("konstant" in w and "suez_z" in w for w in warnings)


def test_external_constant_within_source_cadence_no_warning():
    """Regression 2026-10-01: 4 Ledger-Tage mit unveränderten Wochen-/Monats-
    daten (PortWatch Stand 27.09., BTS-TSI Stand Juni) lösten 8 Fehlalarme aus."""
    today = date(2026, 10, 1)
    prims = {"suez_z": -0.35, "us_freight_tsi_z": -2.6, "freight_global_z": -0.86,
             "weather_disruption_index": 0.0}
    rows = [{"date": (today - timedelta(days=i)).isoformat(),
             "external": {"primitives": dict(prims)}} for i in range(4)]
    # Monatsquelle 40 Tage konstant ist ebenfalls normal; Wetter 0.0 nie "eingefroren".
    rows += [{"date": (today - timedelta(days=40)).isoformat(),
              "external": {"primitives": {"us_freight_tsi_z": -2.6,
                                          "weather_disruption_index": 0.0}}}]
    warnings: list[str] = []
    m = engine_monitor._check_external_missingness(rows, warnings)
    assert not any("konstant" in w for w in warnings)
    assert m["us_freight_tsi_z"]["constant"] and not m["us_freight_tsi_z"]["stale"]


def test_external_missingness_warns_when_mostly_missing(monkeypatch):
    today = date(2026, 9, 27)
    rows = [
        {"date": (today - timedelta(days=i)).isoformat(), "external": {"primitives": {}}}
        for i in range(10)
    ]
    warnings: list[str] = []
    metrics = engine_monitor._check_external_missingness(rows, warnings)
    assert metrics["suez_z"]["missing_ratio"] == 1.0
    assert any("missing" in w for w in warnings)


def test_external_missingness_no_warning_when_healthy():
    today = date(2026, 9, 27)
    rows = [
        {"date": (today - timedelta(days=i)).isoformat(),
         "external": {"primitives": {"suez_z": 0.1 * i}}}
        for i in range(10)
    ]
    warnings: list[str] = []
    engine_monitor._check_external_missingness(rows, warnings)
    assert not any("suez_z" in w for w in warnings)


def test_stale_high_criticality_source_warns(monkeypatch):
    def fake_load_health(archive_root):
        return {"ncei_cdo": {"status": "PASS", "staleness": "STALE", "criticality": "high"}}

    def fake_load_source_configs(sources_dir="config/external_sources"):
        return {"ncei_cdo": {"criticality": "high"}}

    monkeypatch.setattr("modules.external.registry.load_health", fake_load_health)
    monkeypatch.setattr("modules.external.registry.load_source_configs", fake_load_source_configs)

    warnings: list[str] = []
    metrics = engine_monitor._check_stale_high_criticality_sources(warnings)
    assert "ncei_cdo" in metrics["stale_high_criticality"]
    assert any("STALE" in w for w in warnings)


def test_stale_low_criticality_source_no_warning(monkeypatch):
    def fake_load_health(archive_root):
        return {"some_low_source": {"status": "PASS", "staleness": "STALE", "criticality": "low"}}

    def fake_load_source_configs(sources_dir="config/external_sources"):
        return {"some_low_source": {"criticality": "low"}}

    monkeypatch.setattr("modules.external.registry.load_health", fake_load_health)
    monkeypatch.setattr("modules.external.registry.load_source_configs", fake_load_source_configs)

    warnings: list[str] = []
    metrics = engine_monitor._check_stale_high_criticality_sources(warnings)
    assert metrics["stale_high_criticality"] == []
    assert warnings == []


def test_feature_stats_external_tiny_effective_n_warns():
    history = _empty_history(feature_stats_external={
        "freight_state": {"EXPANSION": {"count": 3, "mean": 0.1}},
    })
    warnings: list[str] = []
    metrics = engine_monitor._check_feature_stats_external_reliability(history, [], warnings)
    # Without ledger rows there's no effective_n available, so nothing should
    # be flagged as "tiny" (the check must not fabricate an effective_n).
    assert metrics["tiny_effective_n_buckets"] == []


def test_promoted_external_dominance_placeholder_warns(monkeypatch):
    class FakeLearning:
        promoted_external_features = ["freight_global_z"]

    class FakeExternalContext:
        learning = FakeLearning()

    class FakeCfg:
        external_context = FakeExternalContext()

    monkeypatch.setattr("modules.config.cfg", FakeCfg())
    warnings: list[str] = []
    metrics = engine_monitor._check_promoted_external_dominance(warnings)
    assert metrics["promoted_external_features"] == ["freight_global_z"]
    assert any("promotet" in w for w in warnings)


def test_promoted_external_dominance_empty_by_default():
    warnings: list[str] = []
    metrics = engine_monitor._check_promoted_external_dominance(warnings)
    assert metrics["promoted_external_features"] == []
    assert warnings == []


def test_build_health_report_includes_external_metrics(tmp_path):
    reports_dir = tmp_path / "daily_reports"
    reports_dir.mkdir()
    health = engine_monitor.build_health_report(_empty_history(), reports_dir, date(2026, 9, 27))
    assert "external" in health["metrics"]


def test_external_health_checks_never_raise_on_broken_config(monkeypatch):
    """Jede externe Sub-Prüfung ist einzeln try/except-geschützt — ein Fehler
    in einer Quelle darf die anderen nicht verhindern und den Report nie
    zum Absturz bringen."""
    def boom(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(engine_monitor, "_external_ledger_rows_last_30d", boom)
    warnings: list[str] = []
    metrics = engine_monitor._check_external_health(_empty_history(), date(2026, 9, 27), warnings)
    assert isinstance(metrics, dict)

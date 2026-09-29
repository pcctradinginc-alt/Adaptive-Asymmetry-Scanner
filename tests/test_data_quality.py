"""Data-Quality-Gates (Audit 2026-09-29): kein PASS ohne brauchbare Daten."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from modules.external import data_quality as dq
from modules.external import orchestrator as orch
from modules.external import registry as reg
from modules.external.pit import AvailabilityPrecision, Observation
from modules.external.sources.base import Connector, ConnectorResult, SourceStatus

UTC = timezone.utc
NOW = datetime(2026, 9, 29, 12, tzinfo=UTC)


def _o(value, t=None, metric="m", entity="E", avail=None):
    t = t or NOW - timedelta(days=40)
    return Observation(source_id="s", dataset="d", series_id="x", entity_id=entity, metric=metric,
                       value=value, unit="u", observation_time=t, available_at=avail or NOW,
                       retrieved_at=NOW, availability_precision=AvailabilityPrecision.CONSERVATIVE_DATE,
                       parser_version="1")


def test_empty_result_is_severe_unless_allowed():
    assert dq.assess([], {}, NOW)["issues"] == ["EMPTY_RESULT"]
    assert dq.assess([], {"may_be_empty": True}, NOW)["issues"] == []


def test_null_nan_duplicates_future_range():
    obs = [_o(None, entity="A"), _o(None, entity="B"), _o(float("nan"), entity="C"), _o(1.0, entity="D"),
           _o(2.0, entity="D"), _o(5.0, t=NOW + timedelta(days=10), entity="F"), _o(1e9, entity="G")]
    r = dq.assess(obs, {"plausible_ranges": {"m": [0, 1000]}}, NOW)
    assert {"NON_FINITE", "DUPLICATE_CONFLICT", "FUTURE_OBSERVATION", "OUT_OF_RANGE"} <= set(r["issues"])
    assert r["severe"]


def test_available_before_period_is_flagged_for_period_data():
    """Eine Monatsstatistik kann nicht vor Beginn ihrer Periode vorliegen."""
    t = datetime(2026, 9, 1, tzinfo=UTC)
    r = dq.assess([_o(1.0, t=t, avail=t - timedelta(days=3))], {"frequency": "monthly"}, NOW)
    assert "AVAILABLE_BEFORE_PERIOD" in r["issues"]


def test_scale_shift_detects_unit_change():
    hist = [_o(1000.0 + i, entity=str(i)) for i in range(20)]
    new = [_o(1.0 + i / 1000, entity=str(i)) for i in range(20)]          # Tonnen -> Tsd. Tonnen
    assert "SCALE_SHIFT" in dq.assess(new, {}, NOW, history=hist)["issues"]
    assert dq.assess(hist, {}, NOW, history=hist)["issues"] == []


class _EmptyPass(Connector):
    source_id = "empty"

    def fetch(self, now):
        return ConnectorResult(source_id="empty", status=SourceStatus.PASS, observations=[], raw=[])


def test_orchestrator_downgrades_empty_pass_to_warn(tmp_path):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("_tec", Path(__file__).with_name("test_external_core.py"))
    tec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tec)
    VALID_SOURCE, _fake_registry = tec.VALID_SOURCE, tec._fake_registry
    sources = {"empty": dict(VALID_SOURCE, source_id="empty", family="maritime")}
    registry = _fake_registry(tmp_path, sources, {"empty": _EmptyPass})
    summary = orch.run_ingestion(NOW, registry=registry)
    assert summary["sources"]["empty"]["status"] == "WARN"
    assert summary["sources"]["empty"]["dq_issues"] == ["EMPTY_RESULT"]
    assert registry.load_health()["empty"]["dq"]["issues"] == ["EMPTY_RESULT"]


def test_event_sources_are_marked_may_be_empty():
    r = reg.SourceRegistry()
    assert r.sources["nhc_storms"].get("may_be_empty") is True
    assert r.sources["imf_portwatch_disruptions"].get("may_be_empty") is True

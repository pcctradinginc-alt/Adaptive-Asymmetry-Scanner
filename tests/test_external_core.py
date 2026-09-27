"""Tests für die geteilte Infrastruktur der External-Data Factory
(modules/external/archive.py, registry.py, features.py, orchestrator.py,
alerts.py). Keine Netzwerkzugriffe — nur Fixtures/Mocks."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
import yaml

from modules.external.pit import (
    AvailabilityPrecision, Observation, available_as_of, ensure_utc,
)
from modules.external.sources.base import ConnectorResult, RawRecord, SourceStatus, Connector
from modules.external.archive import ExternalArchive
from modules.external import registry as reg
from modules.external import features as feat
from modules.external import alerts as al
from modules.external import orchestrator as orch


UTC = timezone.utc


def mk_obs(source_id="src_a", dataset="daily", series_id="S1", entity_id="E1",
           metric="m", value=1.0, obs_time="2026-01-01T00:00:00+00:00",
           available_at="2026-01-01T00:00:00+00:00",
           retrieved_at="2026-01-01T00:00:00+00:00",
           precision=AvailabilityPrecision.EXACT_TIMESTAMP,
           vintage_time=None, forecast_issue_time=None, forecast_valid_time=None,
           **kw) -> Observation:
    return Observation(
        source_id=source_id, dataset=dataset, series_id=series_id, entity_id=entity_id,
        metric=metric, value=value, unit="idx",
        observation_time=obs_time, available_at=available_at, retrieved_at=retrieved_at,
        availability_precision=precision, parser_version="1",
        vintage_time=vintage_time, forecast_issue_time=forecast_issue_time,
        forecast_valid_time=forecast_valid_time, **kw,
    )


# ── pit.py ────────────────────────────────────────────────────────────────

def test_pit_filter_excludes_future_available_at():
    o1 = mk_obs(available_at="2026-01-01T00:00:00+00:00", value=1.0)
    o2 = mk_obs(available_at="2026-02-01T00:00:00+00:00", value=2.0,
                obs_time="2026-01-02T00:00:00+00:00")
    out = available_as_of([o1, o2], datetime(2026, 1, 15, tzinfo=UTC))
    assert o1 in out and o2 not in out


def test_pit_filter_picks_latest_vintage_up_to_t():
    base = dict(obs_time="2026-01-01T00:00:00+00:00")
    v1 = mk_obs(value=1.0, available_at="2026-01-02T00:00:00+00:00", **base)
    v2 = mk_obs(value=1.5, available_at="2026-01-05T00:00:00+00:00", **base)
    out = available_as_of([v1, v2], datetime(2026, 1, 10, tzinfo=UTC))
    assert len(out) == 1 and out[0].value == 1.5
    out_early = available_as_of([v1, v2], datetime(2026, 1, 3, tzinfo=UTC))
    assert len(out_early) == 1 and out_early[0].value == 1.0


def test_naive_datetime_becomes_utc():
    o = mk_obs(obs_time=datetime(2026, 1, 1), available_at=datetime(2026, 1, 1),
               retrieved_at=datetime(2026, 1, 1))
    assert o.observation_time.tzinfo is not None
    assert o.observation_time.utcoffset() == timedelta(0)


def test_forecast_requires_issue_time():
    with pytest.raises(ValueError):
        mk_obs(forecast_valid_time="2026-02-01T00:00:00+00:00")


def test_forecast_issue_time_is_part_of_identity():
    common = dict(obs_time="2026-06-01T00:00:00+00:00",
                  forecast_valid_time="2026-06-01T00:00:00+00:00")
    f1 = mk_obs(forecast_issue_time="2026-05-01T00:00:00+00:00", value=1.0, **common)
    f2 = mk_obs(forecast_issue_time="2026-05-15T00:00:00+00:00", value=1.2, **common)
    assert f1.identity_key() != f2.identity_key()


# ── archive.py ────────────────────────────────────────────────────────────

def test_store_observations_dedupe_and_revision(tmp_path):
    a = ExternalArchive(root=tmp_path)
    o_new = mk_obs(value=1.0)
    counts1 = a.store_observations([o_new])
    assert counts1 == {"new": 1, "duplicate": 0, "revision": 0}

    o_dup = mk_obs(value=1.0)  # gleiche Identität, gleicher Wert
    counts2 = a.store_observations([o_dup])
    assert counts2 == {"new": 0, "duplicate": 1, "revision": 0}

    o_rev = mk_obs(value=1.2, available_at="2026-01-10T00:00:00+00:00")  # neuer Wert
    counts3 = a.store_observations([o_rev])
    assert counts3 == {"new": 0, "duplicate": 0, "revision": 1}

    history = a.vintage_history("src_a", "S1", "E1", "m",
                                 observation_time="2026-01-01T00:00:00+00:00", dataset="daily")
    assert [h.value for h in history] == [1.0, 1.2]


def test_store_observations_never_overwrites_rows(tmp_path):
    a = ExternalArchive(root=tmp_path)
    a.store_observations([mk_obs(value=1.0)])
    a.store_observations([mk_obs(value=2.0, available_at="2026-01-05T00:00:00+00:00")])
    all_rows = a.load("src_a")
    assert len(all_rows) == 2
    assert sorted(o.value for o in all_rows) == [1.0, 2.0]


def test_archive_idempotence_across_two_runs(tmp_path):
    a1 = ExternalArchive(root=tmp_path)
    obs = [mk_obs(value=1.0), mk_obs(value=1.0, entity_id="E2")]
    c1 = a1.store_observations(obs)
    assert c1["new"] == 2

    a2 = ExternalArchive(root=tmp_path)  # neue Instanz = zweiter Run
    c2 = a2.store_observations(obs)
    assert c2 == {"new": 0, "duplicate": 2, "revision": 0}
    assert len(a2.load("src_a")) == 2


def test_as_of_value_known_on_date_x(tmp_path):
    a = ExternalArchive(root=tmp_path)
    obs_time = "2026-03-01T00:00:00+00:00"
    a.store_observations([mk_obs(value=10.0, obs_time=obs_time,
                                  available_at="2026-03-02T00:00:00+00:00")])
    a.store_observations([mk_obs(value=11.0, obs_time=obs_time,
                                  available_at="2026-03-20T00:00:00+00:00")])
    known_early = a.as_of("src_a", datetime(2026, 3, 10, tzinfo=UTC))
    known_late = a.as_of("src_a", datetime(2026, 4, 1, tzinfo=UTC))
    assert known_early[0].value == 10.0
    assert known_late[0].value == 11.0


def test_store_raw_hash_only_policy_no_payload(tmp_path):
    a = ExternalArchive(root=tmp_path)
    rec = RawRecord(source_id="src_a", dataset="d", url="https://x", fingerprint="fp",
                     retrieved_at=datetime(2026, 1, 1, tzinfo=UTC), status_code=200,
                     content_type="application/json", content_hash="abc123", bytes=10,
                     content=b"hello world")
    res = a.store_raw(rec, policy="hash_only")
    assert res["stored"] and res["payload_stored"] == "hash_only"
    files = list((tmp_path / "raw" / "src_a").rglob("*"))
    assert not any(f.suffix == ".gz" for f in files)


def test_store_raw_gzip_policy_stores_payload(tmp_path):
    a = ExternalArchive(root=tmp_path)
    content = b"x" * 100
    rec = RawRecord(source_id="src_a", dataset="d", url="https://x", fingerprint="fp",
                     retrieved_at=datetime(2026, 1, 1, tzinfo=UTC), status_code=200,
                     content_type="application/json", content_hash="hash1", bytes=len(content),
                     content=content)
    res = a.store_raw(rec, policy="gzip", raw_max_bytes=1000)
    assert res["payload_stored"] == "gzip"
    files = list((tmp_path / "raw" / "src_a").rglob("*.raw.gz"))
    assert len(files) == 1


def test_store_raw_dedupe_by_content_hash(tmp_path):
    a = ExternalArchive(root=tmp_path)
    rec = RawRecord(source_id="src_a", dataset="d", url="https://x", fingerprint="fp",
                     retrieved_at=datetime(2026, 1, 1, tzinfo=UTC), status_code=200,
                     content_type="application/json", content_hash="dup1", bytes=5,
                     content=b"hello")
    r1 = a.store_raw(rec)
    r2 = a.store_raw(rec)
    assert r1["stored"] and not r2["stored"] and r2["duplicate"]


def test_store_raw_never_persists_credentials():
    # RawRecord besitzt keine Header/Credential-Felder -> nichts zu leaken.
    assert not hasattr(RawRecord, "headers")
    assert set(RawRecord.__dataclass_fields__) & {"headers", "api_key", "token"} == set()


def test_write_manifest_contains_expected_fields(tmp_path):
    a = ExternalArchive(root=tmp_path)
    now = datetime(2026, 5, 1, tzinfo=UTC)
    path = a.write_manifest("run123", [{"source_id": "src_a", "bytes": 10}], now=now)
    data = json.loads(path.read_text())
    assert data["run_id"] == "run123"
    assert "git_sha" in data and "config_hash" in data
    assert data["entries"][0]["source_id"] == "src_a"
    assert path == tmp_path / "manifests" / "2026-05-01" / "run123.json"


def test_storage_telemetry_projections(tmp_path):
    a = ExternalArchive(root=tmp_path)
    a.store_observations([mk_obs(value=float(i)) for i in range(5)] +
                          [mk_obs(value=float(i), entity_id="E2") for i in range(5)])
    rec = RawRecord(source_id="src_a", dataset="d", url="u", fingerprint="fp",
                     retrieved_at=datetime(2026, 1, 1, tzinfo=UTC), status_code=200,
                     content_type="application/json", content_hash="h1", bytes=1000,
                     content=b"x" * 1000)
    a.store_raw(rec, policy="gzip", raw_max_bytes=10000)
    telemetry = a.storage_telemetry(warn_mb_1y=0.0000001)  # winzige Schwelle -> flagged
    assert telemetry["projections_bytes"]["1y"] == pytest.approx(
        telemetry["total_bytes_per_day"] * 365)
    assert telemetry["flagged"] is True
    assert (tmp_path / "health" / "storage_telemetry.json").exists()


# ── registry.py ────────────────────────────────────────────────────────────

VALID_SOURCE = dict(
    source_id="road_x", display_name="Road X", family="road_freight",
    authority="Amt X", official_homepage="https://x.example", machine_endpoint="https://x.example/api",
    access_method="rest", requires_auth=False, auth_env_variable=None,
    license_reference="https://x.example/license", frequency="daily",
    expected_update_cadence="1d", supports_history=True, supports_vintages=False,
    supports_release_time=True, pit_quality="EXACT_DATE", enabled=True,
    criticality="medium", license_status="OK",
)


def test_registry_loads_valid_yaml(tmp_path):
    d = tmp_path / "external_sources"
    d.mkdir()
    (d / "road.yaml").write_text(yaml.safe_dump([VALID_SOURCE]))
    sources = reg.load_source_configs(d)
    assert "road_x" in sources


def test_registry_missing_field_raises(tmp_path):
    d = tmp_path / "external_sources"
    d.mkdir()
    bad = dict(VALID_SOURCE)
    del bad["license_status"]
    (d / "road.yaml").write_text(yaml.safe_dump([bad]))
    with pytest.raises(reg.RegistryError):
        reg.load_source_configs(d)


def test_registry_missing_dir_returns_empty(tmp_path):
    assert reg.load_source_configs(tmp_path / "does_not_exist") == {}


def test_gate_review_required_blocks():
    cfg = dict(VALID_SOURCE, license_status="REVIEW_REQUIRED")
    fetchable, reason = reg.gate_source(cfg)
    assert not fetchable and "REVIEW_REQUIRED" in reason


def test_gate_status_override_blocks():
    cfg = dict(VALID_SOURCE, status_override="DEFERRED")
    fetchable, reason = reg.gate_source(cfg)
    assert not fetchable and "status_override" in reason


def test_gate_auth_missing(monkeypatch):
    monkeypatch.delenv("SOME_MISSING_ENV_VAR", raising=False)
    cfg = dict(VALID_SOURCE, requires_auth=True, auth_env_variable="SOME_MISSING_ENV_VAR")
    fetchable, reason = reg.gate_source(cfg)
    assert not fetchable and reason == "AUTH_MISSING"


def test_gate_auth_present(monkeypatch):
    monkeypatch.setenv("SOME_ENV_VAR", "x")
    cfg = dict(VALID_SOURCE, requires_auth=True, auth_env_variable="SOME_ENV_VAR")
    fetchable, reason = reg.gate_source(cfg)
    assert fetchable


def test_evaluate_staleness():
    now = datetime(2026, 1, 10, tzinfo=UTC)
    fresh = reg.evaluate_staleness(datetime(2026, 1, 9, tzinfo=UTC), "1d", now)
    stale = reg.evaluate_staleness(datetime(2026, 1, 1, tzinfo=UTC), "1d", now)
    assert fresh == "FRESH" and stale == "STALE"


def test_readiness_never_uses_returns_and_no_auto_production():
    cfg = dict(VALID_SOURCE, frequency="daily")
    health = {"status": SourceStatus.PASS.value, "parse_failures": 0,
              "pit_integrity_failures": 0, "staleness": "FRESH", "consecutive_failures": 0}
    stats = {"observation_count": 300, "confirmatory_observation_count": 300,
             "calendar_span_days": 400, "schema_ok": True, "pit_ok": True}
    readiness = reg.compute_readiness(cfg, health, stats)
    # Erfüllt CHALLENGER-Schwellen, ist aber NICHT in promoted_external_features
    assert readiness == reg.DataReadiness.CHALLENGER_READY


def test_readiness_collecting_when_no_observations():
    cfg = dict(VALID_SOURCE)
    health = {"status": SourceStatus.PASS.value}
    assert reg.compute_readiness(cfg, health, {"observation_count": 0}) == reg.DataReadiness.COLLECTING


def test_readiness_blocked_on_review_required():
    cfg = dict(VALID_SOURCE, license_status="REVIEW_REQUIRED")
    assert reg.compute_readiness(cfg, {}, {}) == reg.DataReadiness.BLOCKED


# ── features.py ──────────────────────────────────────────────────────────

def _series(values, start=datetime(2026, 1, 1, tzinfo=UTC), step_days=1):
    return [(start + timedelta(days=i * step_days), v) for i, v in enumerate(values)]


def test_rolling_zscore_basic():
    s = _series([1, 1, 1, 1, 1, 10])
    z = feat.rolling_zscore(s, window_n=5, min_periods=3)
    assert z[-1] is not None and z[-1] > 1.0


def test_yoy_only_with_year_ago_value():
    s = _series([1.0] * 10)  # keine Vorjahresdaten
    y = feat.yoy(s)
    assert all(v is None for v in y)

    s2 = _series([10.0, 12.0], start=datetime(2025, 1, 1, tzinfo=UTC), step_days=365)
    y2 = feat.yoy(s2)
    assert y2[1] == pytest.approx(0.2)


def test_acceleration_is_z_short_minus_z_medium():
    acc = feat.acceleration([1.0, None, 2.0], [0.5, 0.5, 0.5])
    assert acc[0] == pytest.approx(0.5)
    assert acc[1] is None


def test_breadth_min_valid():
    values = {"a": 1.0, "b": -1.0, "c": None}
    b = feat.breadth(values, threshold=0.5, min_valid=3)
    assert b["valid_count"] == 2 and b["breadth_valid"] is False
    b2 = feat.breadth(values, threshold=0.5, min_valid=2)
    assert b2["breadth_valid"] is True
    assert b2["positive_breadth"] == 0.5 and b2["negative_breadth"] == 0.5


def test_classify_state_thresholds():
    assert feat.classify_state(2.0) == "STRONG_EXPANSION"
    assert feat.classify_state(0.7) == "EXPANSION"
    assert feat.classify_state(0.0) == "NEUTRAL"
    assert feat.classify_state(-0.7) == "CONTRACTION"
    assert feat.classify_state(-2.0) == "STRONG_CONTRACTION"


def test_combine_states_low_confidence_single_fresh_source():
    entries = [{"source_id": "a", "z": 2.0, "is_fresh": True, "age_days": 1}]
    out = feat.combine_states(entries)
    assert out["fresh_source_count"] == 1
    assert out["confidence"] <= 0.3


def test_combine_states_all_stale_low_confidence():
    entries = [{"source_id": "a", "z": 2.0, "is_fresh": False, "age_days": 30},
               {"source_id": "b", "z": 1.8, "is_fresh": False, "age_days": 40}]
    out = feat.combine_states(entries)
    assert out["confidence"] <= 0.3


def test_divergence_agreement_labels():
    assert feat.divergence(1.0, 1.2)["agreement"] == "BOTH_EXPANDING"
    assert feat.divergence(-1.0, -1.2)["agreement"] == "BOTH_CONTRACTING"
    assert feat.divergence(None, 1.0)["agreement"] == "UNKNOWN"


def test_drop_incomplete_last():
    s = _series([1.0, 2.0, 3.0])
    far_future_end = lambda t: datetime(2999, 1, 1, tzinfo=UTC)
    out = feat.drop_incomplete_last(s, far_future_end)
    assert len(out) == 2
    past_end = lambda t: datetime(2000, 1, 1, tzinfo=UTC)
    out2 = feat.drop_incomplete_last(s, past_end)
    assert len(out2) == 3


# ── orchestrator.py / alerts.py ────────────────────────────────────────────

class _FailingConnector(Connector):
    source_id = "fails"

    def fetch(self, now):
        raise RuntimeError("boom")


class _OkConnector(Connector):
    source_id = "ok_src"

    def fetch(self, now):
        obs = [mk_obs(source_id="ok_src", value=1.0, obs_time=now.isoformat(),
                       available_at=now.isoformat(), retrieved_at=now.isoformat())]
        raw = [RawRecord(source_id="ok_src", dataset="d", url="u", fingerprint="fp",
                          retrieved_at=now, status_code=200, content_type="application/json",
                          content_hash="h", bytes=3, content=b"abc")]
        return ConnectorResult(source_id="ok_src", status=SourceStatus.PASS,
                                observations=obs, raw=raw,
                                latest_observation_time=now)


def _fake_registry(tmp_path, sources, connectors):
    r = reg.SourceRegistry.__new__(reg.SourceRegistry)
    r.sources_dir = "unused"
    r.archive_root = tmp_path
    r.sources = sources
    r.connectors = connectors
    return r


def test_orchestrator_non_fatal_connector_failure(tmp_path):
    sources = {"fails": dict(VALID_SOURCE, source_id="fails", family="road_freight")}
    registry = _fake_registry(tmp_path, sources, {"fails": _FailingConnector})
    summary = orch.run_ingestion(datetime(2026, 1, 1, tzinfo=UTC), registry=registry)
    assert summary["sources"]["fails"]["status"] == "FAIL"


def test_orchestrator_archives_ok_connector(tmp_path):
    sources = {"ok_src": dict(VALID_SOURCE, source_id="ok_src", family="maritime")}
    registry = _fake_registry(tmp_path, sources, {"ok_src": _OkConnector})
    now = datetime(2026, 2, 1, tzinfo=UTC)
    summary = orch.run_ingestion(now, registry=registry)
    assert summary["sources"]["ok_src"]["status"] == "PASS"
    archive = ExternalArchive(tmp_path)
    assert len(archive.load("ok_src")) == 1


def test_orchestrator_gates_review_required(tmp_path):
    sources = {"blocked": dict(VALID_SOURCE, source_id="blocked", license_status="REVIEW_REQUIRED")}
    registry = _fake_registry(tmp_path, sources, {})
    summary = orch.run_ingestion(datetime(2026, 1, 1, tzinfo=UTC), registry=registry)
    assert "REVIEW_REQUIRED" in summary["sources"]["blocked"]["reason"]


def test_preflight_reports_all_sources_without_archiving(tmp_path):
    sources = {
        "ok_src": dict(VALID_SOURCE, source_id="ok_src"),
        "blocked": dict(VALID_SOURCE, source_id="blocked", enabled=False),
    }
    registry = _fake_registry(tmp_path, sources, {"ok_src": _OkConnector})
    report = orch.preflight(datetime(2026, 1, 1, tzinfo=UTC), registry=registry)
    ids = {r["source_id"] for r in report}
    assert ids == {"ok_src", "blocked"}
    assert not (tmp_path / "normalized").exists()


def test_decide_alerts_only_for_listed_conditions():
    before = {"s1": {"status": "PASS", "consecutive_failures": 0}}
    after = {
        "s1": {"status": "SCHEMA_CHANGED", "consecutive_failures": 0, "criticality": "low"},
        "s2": {"status": "AUTH_MISSING", "consecutive_failures": 0, "criticality": "low"},
        "s3": {"status": "PASS", "consecutive_failures": 3, "criticality": "low"},
        "s4": {"status": "PASS", "consecutive_failures": 0, "criticality": "high", "staleness": "STALE"},
        "s5": {"status": "PASS", "consecutive_failures": 1, "criticality": "low", "staleness": "STALE"},
    }
    alerts = al.decide_alerts(before, after)
    types = {a["type"] for a in alerts}
    assert types == {"SCHEMA_CHANGED", "AUTH_MISSING", "CONSECUTIVE_FAILURES", "STALE_HIGH_CRITICALITY"}
    # s5 ist low-criticality + stale -> KEIN Alarm für Staleness
    assert not any(a["source_id"] == "s5" for a in alerts)


def test_check_pit_integrity_flags_available_after_retrieved():
    bad = mk_obs(available_at="2026-01-05T00:00:00+00:00", retrieved_at="2026-01-01T00:00:00+00:00")
    errors = al.check_pit_integrity([bad])
    assert errors and "available_at > retrieved_at" in errors[0]


def test_check_pit_integrity_allows_available_before_observation():
    ok = mk_obs(obs_time="2026-02-01T00:00:00+00:00", available_at="2026-01-01T00:00:00+00:00",
                retrieved_at="2026-01-01T00:00:00+00:00")
    assert al.check_pit_integrity([ok]) == []


def test_revision_back_to_earlier_value_is_new_vintage(tmp_path):
    """A → B → A: die Rückkehr auf A ist eine echte Revision, kein Duplikat."""
    from datetime import datetime, timezone, timedelta
    from modules.external.archive import ExternalArchive
    from modules.external.pit import Observation, AvailabilityPrecision
    arch = ExternalArchive(root=str(tmp_path))
    t0 = datetime(2026, 9, 1, tzinfo=timezone.utc)

    def ob(v, days):
        ts = t0 + timedelta(days=days)
        return Observation("src", "ds", "ser", "", "m", v, "idx", t0, ts, ts,
                           AvailabilityPrecision.EXACT_DATE, "1")

    assert arch.store_observations([ob(1.0, 1)])["new"] == 1
    assert arch.store_observations([ob(2.0, 2)])["revision"] == 1
    assert arch.store_observations([ob(1.0, 3)])["revision"] == 1
    assert arch.store_observations([ob(1.0, 4)])["duplicate"] == 1
    known = arch.as_of("src", t0 + timedelta(days=2, hours=1))
    assert [o.value for o in known] == [2.0]


def test_preflight_reads_license_review_sources_but_skips_overrides(monkeypatch):
    """REVIEW_REQUIRED-Lizenz: lesender Preflight ja (keine Archivierung),
    status_override/disabled: kein Abruf."""
    from modules.external import orchestrator

    class FakeConn:
        def preflight(self, now):
            return {"source_id": "x", "status": "PASS"}

    class FakeReg:
        def iter_sources(self, family=None):
            return [
                {"source_id": "lic", "family": "f", "enabled": True, "license_status": "REVIEW_REQUIRED"},
                {"source_id": "def", "family": "f", "enabled": True, "status_override": "DEFERRED"},
            ]
        def build_connector(self, sid):
            return FakeConn()

    out = {r["source_id"] if r["source_id"] != "x" else "lic": r
           for r in orchestrator.preflight(registry=FakeReg())}
    assert out["lic"]["archiving"] == "blocked_until_license_review"
    assert out["def"]["status"].startswith("status_override")

"""PIT and shadow-learning regression tests for feature discovery v3."""
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modules.external.pit import AvailabilityPrecision, Observation
from modules.research_feature_store import PITResearchStore, signal_time
from modules import research_feature_discovery as fd


def dt(day: str):
    return datetime.fromisoformat(day).replace(tzinfo=timezone.utc)


def observation(src, metric, value, obs_day, available, *, unit="count", series=None,
                attrs=None, vintage=None, precision=AvailabilityPrecision.EXACT_DATE):
    return Observation(
        source_id=src, dataset="test", series_id=series or metric,
        entity_id="US", metric=metric, value=value, unit=unit,
        observation_time=dt(obs_day), available_at=dt(available),
        retrieved_at=dt("2026-11-12T00:00:00"),
        vintage_time=dt(vintage) if vintage else None,
        availability_precision=precision, parser_version="test",
        attrs=attrs or {}, quality_status="OK",
    )


class Archive:
    def __init__(self, rows=()):
        self.rows = {}
        for row in rows:
            self.rows.setdefault(row.source_id, []).append(row)

    def load(self, source_id):
        return self.rows.get(source_id, [])


def macro_rows():
    return [
        observation("fred_regime_macro", "fed_total_assets", 8000000,
                    "2026-10-20T00:00:00", "2026-10-22T00:00:00"),
        observation("fred_liquidity", "us_treasury_general_account", 400000,
                    "2026-10-20T00:00:00", "2026-10-22T00:00:00"),
        observation("fred_liquidity", "us_on_rrp", 500,
                    "2026-10-20T00:00:00", "2026-10-22T00:00:00"),
        observation("fred_regime_macro", "fed_total_assets", 8100000,
                    "2026-11-08T00:00:00", "2026-11-09T00:00:00"),
        observation("fred_liquidity", "us_treasury_general_account", 450000,
                    "2026-11-08T00:00:00", "2026-11-09T00:00:00"),
        observation("fred_liquidity", "us_on_rrp", 450,
                    "2026-11-08T00:00:00", "2026-11-09T00:00:00"),
    ]


def test_macro_units_and_asof_availability():
    store = PITResearchStore(Archive(macro_rows()))
    # As-of Nov 9 does not have a full day after the ALFRED date-only release.
    assert store.net_liquidity(dt("2026-11-09T12:00:00")) is None
    # After Nov 10 UTC: 8100 - 450 - 450 = 7200 USD billions.
    assert store.net_liquidity(dt("2026-11-10T01:00:00")) == pytest.approx(7200)


def test_macro_revision_does_not_leak_backwards():
    rows = macro_rows()
    rows.append(observation("fred_liquidity", "us_on_rrp", 800,
                            "2026-11-08T00:00:00", "2026-11-11T00:00:00",
                            vintage="2026-11-11T00:00:00"))
    store = PITResearchStore(Archive(rows))
    assert store.net_liquidity(dt("2026-11-10T01:00:00")) == pytest.approx(7200)
    assert store.net_liquidity(dt("2026-11-12T01:00:00")) == pytest.approx(6850)


def test_missing_liquidity_component_remains_unknown():
    store = PITResearchStore(Archive(macro_rows()[:2]))
    assert store.net_liquidity(dt("2026-10-23T00:00:00")) is None


def test_schedule_unknown_before_first_snapshot_and_revisions():
    from modules.external.sources.calendars import SNAPSHOT_SERIES
    snap = lambda available, events: observation(
        "fomc_calendar", "scheduled_events", len(events),
        available, available, series=SNAPSHOT_SERIES, attrs={"events": events},
        precision=AvailabilityPrecision.EXACT_TIMESTAMP)
    old = snap("2026-11-05T15:00:00", ["fomc|2026-11-12T19:00:00+00:00|1"])
    new = snap("2026-11-10T12:00:00", ["fomc|2026-11-19T19:00:00+00:00|1"])
    store = PITResearchStore(Archive([old, new]))
    assert store.event_distance("fomc_calendar", dt("2026-11-04T20:00:00")) is None
    assert store.event_distance("fomc_calendar", dt("2026-11-09T20:00:00")) == pytest.approx(2.9583333)
    assert store.event_distance("fomc_calendar", dt("2026-11-11T20:00:00")) == pytest.approx(7.9583333)


def test_panel_copy_and_missing_calendar_not_treated_as_no_event():
    store = PITResearchStore(Archive())
    original = pd.DataFrame({
        "date": [pd.Timestamp("2026-11-08")],
        "ticker": ["AAA"], "beta_126": [0.4],
    })
    out = store.enrich_panel(original)
    assert len(original.columns) == 3
    assert pd.isna(out.loc[0, "fd3_fomc_near_x_beta"])
    assert pd.isna(out.loc[0, "fd3_liquidity_x_beta"])


def test_model_paired_eligible_rows_and_train_only_normalization():
    p, _ = fd.load_protocol()
    base = p["baseline_features"]
    rows = []
    for day in pd.date_range("2025-01-03", periods=34, freq="W-FRI"):
        for i in range(32):
            rows.append({
                "date": day, "ticker": f"T{i:02d}",
                "label_end_20": day + pd.offsets.BDay(20),
                "fwd_xs_20": (i / 500) - 0.03,
                "fd3_liquidity_x_beta": i / 100,
                **{x: i / 100 + j / 1000 for j, x in enumerate(base)},
            })
    frame = pd.DataFrame(rows)
    d = frame["date"].max()
    eligible = frame[frame["date"] == d]
    past = frame[(frame["date"] < d) & (frame["label_end_20"] < d)]
    a, b, same = fd._pair(past, eligible, p, "fd3_liquidity_x_beta")
    assert len(same) == 32 and len(a) == len(b) == 32
    assert np.isfinite(a).all() and np.isfinite(b).all()


def test_forward_predictions_freeze_once_and_never_use_future_target(tmp_path):
    p, digest = fd.load_protocol()
    b = p["baseline_features"]
    rows = []
    end = pd.Timestamp("2026-11-06")
    for d in pd.date_range(end - pd.Timedelta(weeks=40), end, freq="W-FRI"):
        for i in range(35):
            rows.append({
                "date": d, "ticker": f"T{i:02d}",
                "label_end_20": d + pd.offsets.BDay(20),
                "fwd_xs_20": np.nan if d == end else i / 100 - 0.15,
                **{x: i / 100 + j / 500 for j, x in enumerate(b)},
                "fd3_liquidity_x_beta": i / 200,
                **{h["feature"]: i / 300 for h in p["hypotheses"][1:]},
            })
    panel = pd.DataFrame(rows)
    journal = tmp_path / "predictions.jsonl"
    now = dt("2026-11-08T12:00:00")
    count = fd.freeze_latest(panel, p, digest, journal, now)
    assert count == 35 * len(p["hypotheses"])
    assert fd.freeze_latest(panel, p, digest, journal, now) == 0
    for item in fd._read_jsonl(journal):
        assert item["label"] == "FUTURE_UNOBSERVED"
        assert "fwd_xs_20" not in item
        assert item["research_only"] is True
    result = fd.prospective(panel, p["hypotheses"][0], p, digest, journal, now.date())
    assert result["independent_dates"] == 0


def test_protocol_is_immutable_budgeted_and_shadow_only(tmp_path):
    p, digest = fd.load_protocol()
    assert p["research_only"] is True and len(digest) == 64
    p["research_only"] = False
    path = tmp_path / "bad.yaml"
    import yaml
    path.write_text(yaml.safe_dump(p))
    with pytest.raises(ValueError):
        fd.load_protocol(path)
    p["research_only"] = True
    p["evaluation"]["max_hypotheses"] = 2
    path.write_text(yaml.safe_dump(p))
    with pytest.raises(ValueError):
        fd.load_protocol(path)


def test_workflow_shadow_not_production():
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github/workflows/ml_research.yml").read_text()
    assert "modules.research_feature_discovery" in workflow
    assert workflow.index("modules.ml_research --mode") < workflow.index("modules.research_feature_discovery")
    source = (root / "pipeline.py").read_text()
    assert "research_feature_discovery" not in source


def test_fd3_in_canonical_research_memory_never_claims_success(tmp_path):
    import json
    from modules import research_memory as rm
    report = tmp_path / "fd3.json"
    report.write_text(json.dumps({
        "protocol_sha256": "a" * 64,
        "features": {"fd3_liquidity_x_beta": {"dates_available": 31}},
        "hypotheses": {
            "liq_beta_v1": {
                "feature": "fd3_liquidity_x_beta", "source": "fred_liquidity",
                "premise": "net liquidity research",
                "prospective": {"independent_dates": 9,
                                "mean_paired_top_xs_delta": 0.03,
                                "mean_paired_rank_ic_delta": 0.07},
            }
        }
    }))
    entries = rm.collect({"feature_discovery_v3": report})
    matches = [e for e in entries if e["kind"] == "feature_discovery_v3"]
    assert len(matches) == 1
    assert matches[0]["status"] == "PENDING_FORWARD"
    assert matches[0]["prospective"] is False
    assert matches[0]["status"] not in rm.SUCCESS

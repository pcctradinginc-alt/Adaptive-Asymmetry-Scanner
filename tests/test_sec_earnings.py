"""B1: Earnings-Release-Zeitstempel aus 8-K Item 2.02 (synthetische Filings, streng PIT)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from modules.external.pit import AvailabilityPrecision, Observation
from modules.external.sources import sec_earnings as sea


def filing(acc: str, accepted: str, items=("2.02", "9.01"), form: str = "8-K", cik: str = "cik:0000000001"):
    return Observation(source_id="sec_submissions", dataset="filings", series_id=acc, entity_id=cik, metric="filing",
                       value=1.0, unit="count", observation_time=accepted, available_at=accepted,
                       retrieved_at="2026-10-15T00:00:00+00:00",
                       availability_precision=AvailabilityPrecision.EXACT_TIMESTAMP, parser_version="x",
                       attrs={"form": form, "items": list(items)})


OBS = [filing("A1", "2025-10-28T20:05:00+00:00"),                  # 16:05 EDT -> AMC
       filing("A2", "2026-01-27T12:00:00+00:00"),                  # 07:00 EST -> BMO
       filing("A3", "2026-04-28T15:00:00+00:00"),                  # 11:00 EDT -> RTH
       filing("A4", "2026-07-28T20:30:00+00:00"),
       filing("X1", "2026-07-01T20:30:00+00:00", items=("5.02",)),  # kein Earnings-8-K
       filing("X2", "2026-07-02T20:30:00+00:00", form="10-Q", items=()),
       filing("A4", "2026-07-28T20:30:00+00:00")]                  # Duplikat


def test_session_new_york_time_dst_safe():
    assert sea.session_of(pd.Timestamp("2026-01-27T14:29:00Z")) == 0.0      # 09:29 EST
    assert sea.session_of(pd.Timestamp("2026-01-27T14:30:00Z")) == 0.5      # 09:30 EST
    assert sea.session_of(pd.Timestamp("2026-07-28T20:00:00Z")) == 1.0      # 16:00 EDT
    assert sea.session_of(pd.Timestamp("2026-07-28T19:59:00Z")) == 0.5


def test_release_events_only_8k_item_202_deduplicated():
    ev = sea.release_events(OBS)
    assert list(ev["accession"]) == ["A1", "A2", "A3", "A4"]
    assert list(ev["session"]) == [1.0, 0.0, 0.5, 1.0]


def test_features_point_in_time():
    since = {"cik:0000000001": pd.Timestamp("2020-01-01", tz="UTC")}
    t = sea.build_feature_table(OBS, ["2026-07-28", "2026-07-29"], {"AAA": "cik:0000000001"}, since)
    d0, d1 = t.iloc[0], t.iloc[1]
    # Stichtag 28.07. 21:00 UTC: A4 (20:30 UTC) ist schon öffentlich
    assert d0["sec_days_since_earnings_release"] == 0.5 / 24 and d0["sec_earnings_release_session"] == 1.0
    assert d0["sec_earnings_releases_365d"] == 4.0 and d0["alt_sec_earnings_available"] == 1.0
    assert d1["sec_days_since_earnings_release"] == 1 + 0.5 / 24
    t2 = sea.build_feature_table(OBS, ["2026-07-28"], {"AAA": "cik:0000000001"}, since)
    assert t2.iloc[0]["feature_version"] == sea.FEATURE_VERSION
    early = sea.build_feature_table(OBS, ["2026-07-27"], {"AAA": "cik:0000000001"}, since).iloc[0]
    assert early["sec_earnings_release_session"] == 0.5 and early["sec_earnings_releases_365d"] == 3.0


def test_without_covered_history_features_are_nan_not_zero():
    t = sea.build_feature_table(OBS, ["2026-07-28"], {"AAA": "cik:0000000001", "BBB": "cik:0000000002"},
                                {"cik:0000000001": pd.Timestamp("2026-01-01", tz="UTC")})
    assert t[list(sea.FEATURES)].isna().all().all()                         # Abdeckung < 365 T bzw. fehlend
    assert (t["alt_sec_earnings_available"] == 0).all()
    covered_no_release = sea.build_feature_table(
        OBS, ["2026-07-28"], {"BBB": "cik:0000000002"}, {"cik:0000000002": pd.Timestamp("2020-01-01", tz="UTC")})
    row = covered_no_release.iloc[0]
    assert row["sec_earnings_releases_365d"] == 0.0 and np.isnan(row["sec_days_since_earnings_release"])


def test_empty_inputs():
    assert sea.release_events([]).empty
    assert list(sea.build_feature_table([], ["2026-07-28"], {}, {}).columns)[:3] == ["date", "ticker", "cik"]

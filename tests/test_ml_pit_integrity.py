"""ML-Panel: Sektor point-in-time (nie heutiger Sektor rückwirkend) und quantifizierter Survivorship-Bias."""
from __future__ import annotations

import numpy as np
import pandas as pd

from modules import ml_research as mr


def test_sector_only_from_first_observation_onwards():
    p = pd.DataFrame({"date": pd.to_datetime(["2018-06-01", "2018-10-05", "2026-05-01", "2026-05-01"]),
                      "ticker": ["GOOGL", "GOOGL", "GOOGL", "XOM"]})
    hist = {"GOOGL": [("2018-09-28", "Communication Services")]}
    s = mr.pit_sector(p, hist)
    assert list(s) == ["unknown", "Communication Services", "Communication Services", "unknown"]


def test_sector_history_keeps_first_observation_and_is_append_only(tmp_path):
    path = tmp_path / "h.jsonl"
    mr.record_sector_observations([{"ticker": "A", "sector": "Tech", "observed_at": "2026-05-01", "source": "x"},
                                   {"ticker": "A", "sector": "Tech", "observed_at": "2026-04-01", "source": "y"}], path)
    mr.record_sector_observations([{"ticker": "A", "sector": "Tech", "observed_at": "2026-10-03", "source": "z"},
                                   {"ticker": "A", "sector": "Health", "observed_at": "2026-10-03", "source": "z"}],
                                  path)
    h = mr.read_sector_history(path)
    assert h["A"] == [("2026-04-01", "Tech"), ("2026-10-03", "Health")]


def test_build_panel_never_uses_current_sector_retroactively():
    cal = pd.bdate_range("2019-01-01", "2019-12-31")
    spy = pd.DataFrame({"Open": 100.0, "Close": 100.0}, index=cal)
    df = pd.DataFrame({"Open": np.linspace(10, 12, len(cal)), "Close": np.linspace(10, 12, len(cal)),
                       "Volume": 1e6, "High": 12.5, "Low": 9.5}, index=cal)
    p = mr.build_panel({"AAA": df}, spy, sectors={"AAA": [("2026-10-03", "Tech")]}, start="2019-01-01")
    assert not p.empty and set(p["sector"]) == {"unknown"}


def test_survivorship_report_quantifies_missing_members():
    d = pd.Timestamp("2020-01-03")
    panel = pd.DataFrame({"date": [d] * 30, "ticker": [f"T{i}" for i in range(30)],
                          "fwd_xs_20": np.linspace(-0.1, 0.1, 30)})
    membership = {f"T{i}": [["2000-01-01", None]] for i in range(40)}       # 10 Mitglieder ohne Kurse
    r = mr.survivorship_report(panel, membership)
    assert r["available"] and r["mean_missing_member_share"] == 0.25
    assert r["worst_case_xs_mean_shift"] < 0 and r["missing_share_by_year"] == {2020: 0.25}

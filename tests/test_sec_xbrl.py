"""SEC XBRL companyfacts: Parser/Schema-Drift, PIT inkl. Revisionen (jüngstes Filing <= t,
nie spätere Korrekturen rückwirkend), Q4-Ableitung, Staleness -> NaN, inkrementeller
budgetierter Ingest, Registry-Einbindung."""
from __future__ import annotations

import types
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from modules.external.http import SchemaError
from modules.external.sources import sec_xbrl as sx
from modules.external.sources.sec_ingest import obs_to_rows

RT = datetime(2026, 10, 3, tzinfo=timezone.utc)
QE = {1: ("01-01", "03-31"), 2: ("04-01", "06-30"), 3: ("07-01", "09-30")}


def _facts(years=range(2012, 2020), restate=None):
    rev, eps, assets, shares = [], [], [], []
    for y in years:
        for q, (s, e) in QE.items():
            filed = (pd.Timestamp(f"{y}-{e}") + pd.Timedelta(days=40)).date().isoformat()
            rev.append({"start": f"{y}-{s}", "end": f"{y}-{e}", "val": 100 * 1.1 ** (y - 2012), "form": "10-Q",
                        "filed": filed, "accn": f"q{y}{q}"})
            eps.append({"start": f"{y}-{s}", "end": f"{y}-{e}", "val": 1 + 0.1 * (y - 2012) + 0.03 * ((y + q) % 3),
                        "form": "10-Q", "filed": filed, "accn": f"q{y}{q}"})
            assets.append({"end": f"{y}-{e}", "val": 1000 * 1.05 ** (y - 2012), "form": "10-Q", "filed": filed,
                           "accn": f"q{y}{q}"})
            shares.append({"end": f"{y}-{e}", "val": 500 - 10 * (y - 2012), "form": "10-Q", "filed": filed,
                           "accn": f"q{y}{q}"})
        rev.append({"start": f"{y}-01-01", "end": f"{y}-12-31", "val": 4 * 100 * 1.1 ** (y - 2012) + 40,
                    "form": "10-K", "filed": f"{y + 1}-02-20", "accn": f"k{y}"})
    rev += restate or []
    rev.append({"start": "2015-01-01", "end": "2015-03-31", "val": 1, "form": "8-K", "filed": "2015-04-02", "accn": "x"})
    return {"cik": 1, "facts": {"us-gaap": {"Revenues": {"units": {"USD": rev}},
                                            "EarningsPerShareDiluted": {"units": {"USD/shares": eps}},
                                            "Assets": {"units": {"USD": assets}}},
                                "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": shares}}}}}


def _table(payload, dates):
    rows = obs_to_rows(sx.parse_companyfacts(payload, "1", RT, "h"))
    return sx.build_feature_table(rows, pd.to_datetime(dates), {"X": "1", "NOFACTS": "2"}).set_index(["ticker", "date"])


def test_parser_pit_fields_and_schema_drift():
    obs = sx.parse_companyfacts(_facts(), "1", RT, "abc")
    o = next(x for x in obs if x.attrs["accn"] == "q20151")
    assert o.available_at == datetime(2015, 5, 11, tzinfo=timezone.utc) and o.vintage_time.date().isoformat() == "2015-05-10"
    assert o.availability_precision.value == "CONSERVATIVE_DATE" and o.payload_hash == "abc"
    assert not any(x.attrs["form"] == "8-K" for x in obs)                          # nur periodische Formulare
    assert {x.metric for x in obs} == {"revenue", "eps_diluted", "assets", "shares"}
    with pytest.raises(SchemaError):
        sx.parse_companyfacts({"cik": 1}, "1", RT)
    with pytest.raises(SchemaError):
        sx.parse_companyfacts({"facts": {"us-gaap": {"Revenues": {"units": {"USD": [{"end": "2015-03-31"}]}}}}}, "1", RT)


def test_features_point_in_time_and_q4_derivation():
    t = _table(_facts(), ["2015-05-08", "2015-05-15", "2016-02-26", "2021-06-04"])
    x = t.loc["X"]
    assert not np.isnan(x.loc["2015-05-08", "xbrl_rev_yoy"])                       # Q4 2014 bekannt (Feb 2015)
    assert x.loc["2015-05-15", "xbrl_rev_yoy"] == pytest.approx(np.log(1.1))         # Q1 2015 ab 11.05. bekannt
    # Q4 2015 = GJ − (Q1+Q2+Q3) = 40 + 100*1.1^3 -> gegen Q4 2014 = 40 + 100*1.1^2
    assert x.loc["2016-02-26", "xbrl_rev_yoy"] == pytest.approx(np.log((40 + 133.1) / (40 + 121)))
    assert x.loc["2016-02-26", "xbrl_asset_growth"] == pytest.approx(np.log(1.05))
    assert x.loc["2016-02-26", "xbrl_share_change"] == pytest.approx(np.log(470 / 480))         # letzter Stand Q3 2015
    assert x.loc["2016-02-26", "alt_xbrl_available"] == 1.0
    assert np.isnan(x.loc["2021-06-04", "xbrl_rev_yoy"])                             # veraltet -> NaN, nie 0
    n = t.loc["NOFACTS"]
    assert n[list(sx.FEATURES)].isna().all().all() and (n["alt_xbrl_available"] == 0).all()


def test_restatement_never_used_before_it_was_filed():
    restated = [{"start": "2015-01-01", "end": "2015-03-31", "val": 200.0, "form": "10-K/A", "filed": "2016-08-01",
                 "accn": "restate"}]
    t = _table(_facts(restate=restated), ["2015-06-05", "2016-08-05"]).loc["X"]
    base = _table(_facts(), ["2015-06-05"]).loc["X"]
    assert t.loc["2015-06-05", "xbrl_rev_yoy"] == base.loc["2015-06-05", "xbrl_rev_yoy"]    # damaliger Wissensstand
    obs = sx.parse_companyfacts(_facts(restate=restated), "1", RT)
    df = pd.DataFrame([{"metric": o.metric, "start": pd.Timestamp(o.attrs["start"], tz="UTC") if o.attrs["start"] else pd.NaT,
                        "end": pd.Timestamp(o.attrs["end"], tz="UTC"), "avail": pd.Timestamp(o.available_at),
                        "value": o.value} for o in obs])
    before = sx._known(df, pd.Timestamp("2016-07-01", tz="UTC"))
    after = sx._known(df, pd.Timestamp("2016-08-03", tz="UTC"))
    pick = lambda k: k[(k["metric"] == "revenue") & (k["end"] == pd.Timestamp("2015-03-31", tz="UTC"))
                       & ((k["end"] - k["start"]).dt.days < 100)]["value"].iloc[0]
    assert pick(before) == pytest.approx(133.1) and pick(after) == 200.0             # Korrektur erst ab Einreichung


def test_sue_needs_history_and_is_standardized():
    t = _table(_facts(), ["2013-05-17", "2016-11-11", "2019-11-15"]).loc["X"]
    assert np.isnan(t.loc["2013-05-17", "xbrl_sue"])                                 # zu wenig Historie
    assert np.isfinite(t.loc["2016-11-11", "xbrl_sue"])


def test_incremental_budgeted_ingest(tmp_path, monkeypatch):
    monkeypatch.setattr(sx, "STORE", tmp_path / "xbrl.csv.gz")
    calls = []

    def get(url, headers=None, timeout=None):
        calls.append(url)
        if url.endswith("0000000003.json"):
            raise RuntimeError("404 für url (nicht retrybar)")
        if url.endswith("0000000004.json"):
            return types.SimpleNamespace(json=lambda: {"bad": 1}, retrieved_at=RT, content_hash="z")
        return types.SimpleNamespace(json=lambda: _facts(years=[2015, 2016]), retrieved_at=RT, content_hash="h1")
    state = {}
    r = sx.ingest(state, ["1", "2", "3", "4"], {}, get, sleep=lambda s: None, budget=3)
    assert r["budget_exhausted"] and r["not_found"] == 1 and len(calls) == 3
    assert state["ciks"]["1"]["payload_hash"] == "h1" and state["ciks"]["3"]["not_found"]
    r2 = sx.ingest(state, ["1", "2", "3", "4"], {}, get, sleep=lambda s: None, budget=10)
    assert r2["schema_errors"] == 1 and calls[-1].endswith("0000000004.json") and r2["new_rows"] == 0
    assert sx.due_ciks(state, ["1", "2"], {"1": "2099-01-01"}) == ["1"]              # neues Filing -> erneut
    assert sx.due_ciks(state, ["1", "2"], {}) == []


def test_registered_as_separate_source_for_ablation():
    from modules.alt_data.registry import ALT_FEATURES, SOURCES
    s = SOURCES["sec_xbrl_fundamentals"]
    assert s["availability_col"] == "alt_xbrl_available" and set(s["features"]) == set(sx.FEATURES)
    assert ALT_FEATURES["xbrl_sue"]["source"] == "sec_xbrl_fundamentals"
    assert s["contracts"] == ["sec_companyfacts"]


def test_unchanged_comparatives_dropped_revisions_kept():
    f = _facts(years=[2015, 2016])
    rev = f["facts"]["us-gaap"]["Revenues"]["units"]["USD"]
    rev.append({**rev[0], "form": "10-Q", "filed": "2016-05-10", "accn": "q20161-comp"})          # gleiche Zahl
    rev.append({**rev[1], "val": 999.0, "form": "10-Q", "filed": "2016-08-10", "accn": "q20162-rev"})  # Revision
    obs = [o for o in sx.parse_companyfacts(f, "1", RT) if o.metric == "revenue"]
    accns = {o.attrs["accn"] for o in obs}
    assert "q20161-comp" not in accns and "q20162-rev" in accns and "q20151" in accns

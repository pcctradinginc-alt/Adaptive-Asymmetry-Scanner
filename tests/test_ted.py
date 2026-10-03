"""TED (Phase 3): Parser, strenger Namensabgleich, PIT, gemessene Feld-Abdeckung,
inkrementeller budgetierter Ingest, Features nur innerhalb belegter Abdeckung."""
from __future__ import annotations

import math
import types
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from modules.external.http import SchemaError
from modules.external.sources import ted_events as te
from modules.external.sources import ted_features as tf
from modules.external.sources import ted_ingest as ti

NOW = datetime(2026, 10, 3, 6, tzinfo=timezone.utc)
ALIASES = ["ACCENTURE PLC", "Accenture Ltd"]


def notice(pub, date, winners, value=None, cur=None):
    n = {"publication-number": pub, "publication-date": date, "notice-type": "can-standard",
         "winner-name": {"eng": winners}, "classification-cpv": ["72000000"], "buyer-country": ["DEU"]}
    if value is not None:
        n["total-value"] = value
    if cur is not None:
        n["total-value-cur"] = cur
    return n


def test_parse_search_flattens_and_validates():
    ns, total = te.parse_search({"notices": [notice("1-2024", "2024-03-05+01:00", ["Accenture GmbH"])],
                                 "totalNoticeCount": 7})
    assert total == 7 and ns[0]["winner-name"] == ["Accenture GmbH"] and ns[0]["buyer-country"] == ["DEU"]
    with pytest.raises(SchemaError):
        te.parse_search({"results": []})


def test_winner_matching_is_strict():
    assert te.match_winner("Accenture plc", ALIASES) == "HIGH"
    assert te.match_winner("Accenture Technology Solutions GmbH", ALIASES) == "MEDIUM"
    assert te.match_winner("Accenturex Services", ALIASES) is None
    assert te.match_winner("IBM Belgium BVBA", ["IBM"]) is None                     # zu kurz/mehrdeutig
    assert te.match_winner("IBM", ["International Business Machines Corp", "IBM"]) == "HIGH"
    assert te.match_winner("", ALIASES) is None


def test_observations_pit_and_value_only_eur():
    rt = NOW
    obs = te.notices_to_observations("0001467373", [
        notice("100-2024", "2024-03-05", ["Accenture plc"], value=1_000_000, cur="EUR"),
        notice("101-2024", "2024-03-06", ["Accenture Technology Solutions"], value=5_000, cur="PLN"),
        notice("102-2024", "2024-03-07", ["Someone Else SA"], value=9, cur="EUR")], ALIASES, rt)
    assert [o.series_id for o in obs] == ["ted:100-2024:0001467373", "ted:101-2024:0001467373"]
    a, b = obs
    assert a.available_at == datetime(2024, 3, 6, tzinfo=timezone.utc) and a.value == 1_000_000
    assert b.value is None and b.attrs["confidence"] == "MEDIUM"                    # PLN nie umgerechnet
    assert a.availability_precision.value == "CONSERVATIVE_DATE" and a.attrs["cpv2"] == "72"


def test_field_start_requires_contiguous_coverage():
    assert ti.field_start({"2021": 0.1, "2022": 0.7, "2023": 0.9, "2024": 0.95}, 2024) == 2022
    assert ti.field_start({"2022": 0.9, "2023": None, "2024": 0.9}, 2024) == 2024
    assert ti.field_start({"2024": 0.2}, 2024) is None


class FakeTED:
    def __init__(self, per_year: dict, fail_years=(), probe_share=None):
        self.per_year, self.fail_years, self.calls = per_year, set(fail_years), []
        self.probe_share = probe_share or {}

    def __call__(self, body):
        q = body["query"]
        self.calls.append(q)
        year = int(q.split("publication-date >= ")[1][:4])
        if q.startswith("form-type"):
            share = self.probe_share.get(year, 1.0)
            ns = [notice(f"p{i}", f"{year}-01-02", ["X"] if i < share * 10 else []) for i in range(10)]
            return types.SimpleNamespace(json=lambda ns=ns: {"notices": ns, "totalNoticeCount": 10})
        if year in self.fail_years:
            raise RuntimeError("TED 503")
        allns = self.per_year.get(year, [])
        p = body["page"]
        chunk = allns[(p - 1) * te.PAGE_LIMIT: p * te.PAGE_LIMIT]
        return types.SimpleNamespace(json=lambda c=chunk, t=len(allns): {"notices": c, "totalNoticeCount": t})


def test_probe_and_incremental_budgeted_ingest(tmp_path, monkeypatch):
    monkeypatch.setattr(ti, "STORE", tmp_path / "awards.csv.gz")
    fake = FakeTED({2025: [notice(f"{i}-2025", "2025-05-01", ["Accenture plc"]) for i in range(150)],
                    2026: [notice("1-2026", "2026-02-01", ["Accenture plc"], 10.0, "EUR")]},
                   fail_years={2024}, probe_share={2023: 0.1})
    state = {}
    pr = ti.probe(state, 2023, NOW, post=fake, sleep=lambda s: None)
    assert pr["field_start_year"] == 2024                                           # 2023 zu dünn
    ents = [("0001467373", "accenture", ALIASES)]
    r = ti.ingest(state, ents, NOW, post=fake, sleep=lambda s: None)
    years = state["entities"]["0001467373"]["years"]
    assert years["2024"]["complete"] is False and any("2024" in e for e in r["errors"])
    assert years["2025"]["complete"] and years["2025"]["n_matched"] == 150          # 2 Seiten
    assert r["new_rows"] == 151
    n = len(fake.calls)
    r2 = ti.ingest(state, ents, NOW, post=fake, sleep=lambda s: None)
    assert r2["new_rows"] == 0                                                       # append-only, keine Dubletten
    queried = [c for c in fake.calls[n:] if "2025" in c.split(">= ")[1][:4]]
    assert queried == []                                                             # vollständiges Vorjahr nie erneut
    big = FakeTED({2026: [notice(f"{i}-26", "2026-01-05", ["Accenture plc"]) for i in range(te.PAGE_LIMIT * te.MAX_PAGES + 5)]})
    st2 = {"field_start_year": 2026}
    r3 = ti.ingest(st2, ents, NOW, post=big, sleep=lambda s: None)
    assert r3["truncated"] == 1 and st2["entities"]["0001467373"]["years"]["2026"]["complete"] is False
    st3 = {"field_start_year": 2025}
    r4 = ti.ingest(st3, ents, NOW, post=fake, sleep=lambda s: None, budget=1)
    assert r4["budget_exhausted"] and "2025" not in st3["entities"]["0001467373"]["years"]


def _years(*ys, fetched="2026-10-03T06:00:00+00:00"):
    return {str(y): {"complete": True, "fetched_at": fetched} for y in ys}


def test_features_coverage_pit_and_zero_semantics():
    awards = pd.DataFrame({"avail": pd.to_datetime(["2025-06-02", "2025-06-10", "2026-01-15"], utc=True),
                           "value": [1e6, np.nan, np.nan]})
    dates = [pd.Timestamp("2024-03-01"), pd.Timestamp("2025-06-01"), pd.Timestamp("2025-06-02"),
             pd.Timestamp("2025-12-31"), pd.Timestamp("2026-09-30"), pd.Timestamp("2026-10-30")]
    f = tf.features_cik(dates, awards, _years(2024, 2025, 2026), field_start_year=2024)
    assert np.isnan(f["ted_awards_90d"][0])                                          # Fenster reicht vor 2024
    assert f["ted_awards_90d"][1] == 0 and f["ted_any_award_365d"][1] == 0          # echte Null in Abdeckung
    assert f["ted_award_value_365d"][1] == 0.0
    assert f["ted_awards_90d"][2] == 1                                               # verfügbar am 02.06. (<= 21:00)
    assert f["ted_award_value_365d"][3] == pytest.approx(math.log1p(1e6))            # NaN-Wert ignoriert
    assert np.isnan(f["ted_awards_90d"][5])                                          # nach Abrufzeitpunkt -> NaN
    only_nan = tf.features_cik([pd.Timestamp("2026-02-01")], awards.iloc[2:], _years(2025, 2026), 2025)
    assert np.isnan(only_nan["ted_award_value_365d"][0]) and only_nan["ted_awards_90d"][0] == 1
    none = tf.features_cik(dates, awards, _years(2024, 2025, 2026), field_start_year=None)
    assert none.isna().all().all()                                                   # ohne Probe nie 0
    gap = tf.features_cik(dates, awards, {**_years(2024, 2026), "2025": {"complete": False, "fetched_at": "x"}}, 2024)
    assert np.isnan(gap["ted_awards_90d"][3])                                        # unvollständiges Jahr -> NaN


def test_feature_table_and_registry():
    obs = pd.DataFrame({"entity_id": ["1"], "available_at": ["2025-06-02T00:00:00+00:00"], "value": ["5.0"]})
    t = tf.build_feature_table(obs, [pd.Timestamp("2025-06-06")], {"ACN": "1", "ZZZ": "2"},
                               {"1": _years(2024, 2025, 2026)}, 2024)
    acn, zzz = t.set_index("ticker").loc["ACN"], t.set_index("ticker").loc["ZZZ"]
    assert acn["ted_awards_90d"] == 1 and acn["alt_ted_available"] == 1.0
    assert np.isnan(zzz["ted_awards_90d"]) and zzz["alt_ted_available"] == 0.0       # nie abgefragt -> NaN
    from modules.alt_data.registry import ALT_FEATURES, SOURCES
    assert SOURCES["ted_procurement"]["availability_col"] == "alt_ted_available"
    assert ALT_FEATURES["ted_awards_z"]["source"] == "ted_procurement"


def test_ted_contracts_valid():
    from modules.alt_data import contracts as ac
    cs = {c["hypothesis_id"]: c for c in ac.load()}
    for hid in ("ALT-TED-001", "ALT-TED-002"):
        assert ac.validate(cs[hid]) == [], hid

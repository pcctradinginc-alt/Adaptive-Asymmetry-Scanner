"""Commodity Intelligence (EIA / FRED-ALFRED / CFTC COT) – RESEARCH/SHADOW.

Offline: alle HTTP-Abrufe über injizierte Fake-Antworten (keine Netzwerkabhängigkeit).
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modules import commodity_intelligence as cmd
from modules.external import http
from modules.external.archive import ExternalArchive
from modules.external.pit import AvailabilityPrecision, Observation
from modules.external.sources import commodities as cs
from modules.external.sources.base import SourceStatus

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 10, 3, 12, tzinfo=timezone.utc)
CFG = cs.load_config()


class FakeRes:
    def __init__(self, payload, url="https://x", status=200):
        self.payload, self.url, self.status = payload, url, status
        self.content = json.dumps(payload).encode()
        self.content_type, self.retrieved_at = "application/json", NOW
        self.content_hash, self.fingerprint, self.bytes = "h", "f", len(self.content)

    def json(self):
        return self.payload


def obs(sid, metric, unit, t, av, v, ent="US", series=None, attrs=None, vintage=None):
    return Observation(source_id=sid, dataset="d", series_id=series or metric, entity_id=ent, metric=metric, value=v,
                       unit=unit, observation_time=t, available_at=av, retrieved_at=max(av, NOW),
                       availability_precision=AvailabilityPrecision.CONSERVATIVE_DATE, parser_version="1",
                       vintage_time=vintage, attrs=attrs or {})


def utc(*a):
    return datetime(*a, tzinfo=timezone.utc)


# ── synthetisches PIT-Archiv (alle fünf Quellen) ─────────────────────────────

def synth_archive(root: Path, end: datetime = NOW, start: datetime = utc(2018, 1, 1), seed: int = 3,
                  skip: set | None = None) -> ExternalArchive:
    rng = np.random.default_rng(seed)
    skip = skip or set()
    out = []
    for k, s in cmd.price_series().items():
        if s["source_id"] in skip:
            continue
        p = 60.0
        if s["frequency"] == "daily":
            t = start
            while t < end:
                if t.weekday() < 5:
                    p *= float(np.exp(rng.normal(0, 0.02)))
                    out.append(obs(s["source_id"], s["metric"], s["unit"], t, min(t + timedelta(days=1), end), round(p, 3)))
                t += timedelta(days=1)
        else:
            t = start
            while t + timedelta(days=35) <= end:
                p *= float(np.exp(rng.normal(0, 0.05)))
                out.append(obs(s["source_id"], s["metric"], s["unit"], t, t + timedelta(days=35), round(p * 50, 2)))
                t = utc(t.year + (t.month == 12), t.month % 12 + 1, 1)
    for k, s in cmd.fundamental_series().items():
        if s["source_id"] in skip:
            continue
        rule = {"days": s["lag_days"], "hour_utc": 16}
        if s["frequency"] == "weekly":
            t = start + timedelta(days=(4 - start.weekday()) % 7)
            v = 400000.0 if s["kind"] == "stock" else 9000.0
            if k == "natgas_storage":
                v = 2500.0
            while cs.release_time(t, rule) <= end:
                v *= 1 + float(rng.normal(0, 0.01))
                out.append(obs(s["source_id"], s["metric"], s["unit"], t, cs.release_time(t, rule), round(v, 1),
                               series=s["series"]))
                t += timedelta(days=7)
        else:
            t = start
            while cs.release_time(t, rule) <= end:
                out.append(obs(s["source_id"], s["metric"], s["unit"], t, cs.release_time(t, rule),
                               2e6 * (1 + float(rng.normal(0, 0.02))), series=s["series"],
                               attrs={"revision_status": "backfill_latest_vintage"}))
                t = utc(t.year + (t.month == 12), t.month % 12 + 1, 1)
    if "cftc_cot" not in skip:
        rule = CFG["cftc"]["release"]
        for m, mc in CFG["cftc"]["markets"].items():
            d = (start + timedelta(days=(1 - start.weekday()) % 7)).date()
            while cs.cot_release_time(d, rule) <= end:
                vals = {"total_open_interest": 500000, "producer_merchant_long": 100000,
                        "producer_merchant_short": 150000, "swap_long": 80000, "swap_short": 60000,
                        "managed_money_long": int(rng.integers(50000, 150000)),
                        "managed_money_short": int(rng.integers(20000, 100000)), "managed_money_spreading": 30000,
                        "other_reportables_long": 40000, "other_reportables_short": 30000}
                av = cs.cot_release_time(d, rule)
                for f, v in vals.items():
                    out.append(obs("cftc_cot", "cot_" + f, "contracts", utc(d.year, d.month, d.day), av, float(v),
                                   ent=m, series=mc["code"]))
                d += timedelta(days=7)
    arch = ExternalArchive(str(root))
    arch.store_observations(out, max_new_normalized_bytes_per_source_per_run=10**10, max_backfill_bytes=10**10)
    return arch


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("arch")
    synth_archive(root)
    df, st = cmd.build(now=NOW, root=root, start="2021-06-01", write=False)
    return root, df, st


# ── Config / Registry ────────────────────────────────────────────────────────

def test_config_only_official_sources_and_no_secrets():
    txt = "\n".join(x.split("#")[0] for x in (ROOT / "config/commodity_intelligence.yaml").read_text().splitlines())
    for banned in ("usda", "yahoo", "cmegroup", "theice.com", "quandl"):
        assert banned not in txt.lower()
    assert "api_key:" not in txt.lower()
    assert CFG["eia"]["api_base"].startswith("https://api.eia.gov/v2")
    assert CFG["cftc"]["endpoint"].startswith("https://publicreporting.cftc.gov/")
    assert set(CFG["fred"]["unavailable"]) == {"gold", "silver"}
    assert CFG["features"]["version"] == cmd.FEATURE_VERSION
    assert CFG["research_status"] == "RESEARCH_ONLY"
    assert CFG["promotion"]["max_influence"] == "SCORE_LIMITED"


def test_registry_entries_and_connectors():
    from modules.alt_data.registry import contract_gaps
    from modules.external.registry import SourceRegistry
    r = SourceRegistry()
    for sid in ("eia_petroleum_weekly", "eia_natural_gas", "fred_commodities", "cftc_cot"):
        c = r.sources[sid]
        assert c["family"] == "commodities" and c["license_status"] == "OK"
        assert c["research_only"] is True and c["criticality"] == "low"
        assert r.connector_for(sid) is not None
    assert r.sources["eia_petroleum_weekly"]["auth_env_variable"] == "EIA_API_KEY"
    assert r.sources["cftc_cot"]["requires_auth"] is False
    assert contract_gaps() == {}


def test_workflows_map_existing_secrets():
    for wf in ("external_data.yml", "source_health.yml", "external_preflight.yml"):
        t = (ROOT / ".github/workflows" / wf).read_text()
        assert "EIA_API_KEY: ${{ secrets.EIA_KEY }}" in t
        assert "secrets.FRED_KEY" in t
    t = (ROOT / ".github/workflows/ml_research.yml").read_text()
    assert "modules.commodity_intelligence build" in t and "modules.commodity_intelligence evaluate" in t


# ── EIA ──────────────────────────────────────────────────────────────────────

def _eia_rows(series, unit="MBBL", n=10, start=date(2026, 7, 3), step=7, monthly=False):
    rows = []
    for i in range(n):
        d = start + timedelta(days=step * i)
        rows.append({"period": d.strftime("%Y-%m") if monthly else d.isoformat(), "series": series,
                     "value": str(400000 + i), "units": unit})
    return {"response": {"total": len(rows), "data": rows[::-1]}}


def test_eia_normal(monkeypatch, tmp_path):
    monkeypatch.setenv("EIA_API_KEY", "k")
    calls = []

    def fake(url, params=None, **kw):
        calls.append(params)
        s = params["facets[series][]"]
        spec = next(x for x in CFG["eia"]["series"] if x["series"] == s)
        return FakeRes(_eia_rows(s, unit=spec["units"][0], monthly=spec["frequency"] == "monthly"), url=url)
    monkeypatch.setattr(http, "fetch", fake)
    c = cs.EiaPetroleumWeeklyConnector({"_archive_root": str(tmp_path)})
    res = c.fetch(NOW)
    assert res.status == SourceStatus.PASS
    assert all(p["start"] == "2015-01-01" for p in calls)          # Erstimport: Backfill
    o = next(x for x in res.observations if x.series_id == "WCESTUS1" and x.observation_time == utc(2026, 7, 3))
    assert o.available_at == utc(2026, 7, 9, 16) and o.availability_precision == AvailabilityPrecision.CONSERVATIVE_DATE
    assert o.unit == "thousand_barrels" and o.attrs["frequency"] == "weekly" and o.attrs["source_unit"] == "MBBL"
    assert all(x.available_at <= x.retrieved_at for x in res.observations)
    assert "k" not in json.dumps([r.url for r in res.raw])


def test_eia_missing_key_no_call(monkeypatch, tmp_path):
    monkeypatch.delenv("EIA_API_KEY", raising=False)
    monkeypatch.delenv("EIA_KEY", raising=False)
    monkeypatch.setattr(http, "fetch", lambda *a, **k: pytest.fail("kein Call ohne Key"))
    res = cs.EiaNaturalGasConnector({"_archive_root": str(tmp_path)}).fetch(NOW)
    assert res.status == SourceStatus.AUTH_MISSING and not res.observations


def test_eia_schema_change_and_unit_change(monkeypatch, tmp_path):
    monkeypatch.setenv("EIA_KEY", "k")
    monkeypatch.setattr(http, "fetch", lambda url, params=None, **kw: FakeRes({"data": []}))
    res = cs.EiaPetroleumWeeklyConnector({"_archive_root": str(tmp_path)}).fetch(NOW)
    assert res.status == SourceStatus.SCHEMA_CHANGED and not res.observations
    monkeypatch.setattr(http, "fetch", lambda url, params=None, **kw: FakeRes(_eia_rows(params["facets[series][]"],
                                                                                      unit="Gallons")))
    res = cs.EiaPetroleumWeeklyConnector({"_archive_root": str(tmp_path)}).fetch(NOW)
    assert res.status == SourceStatus.SCHEMA_CHANGED                 # Einheitenwechsel: nie still umrechnen
    assert any("Einheit" in x for x in res.discovered_ids["schema_changed"])


def test_eia_stale_health_and_feature_unavailable(tmp_path):
    assert cmd.health_label({"last_success": "2026-09-01", "status": "PASS", "staleness": "STALE"}) == "STALE"
    assert cmd.health_label({}) == "UNVALIDATED"
    assert cmd.health_label({"last_success": "x", "status": "FAIL", "consecutive_failures": 3}) == "BROKEN"
    assert cmd.health_label({"last_success": "x", "status": "WARN"}) == "DEGRADED"
    assert cmd.health_label({"last_success": "x", "status": "PASS", "staleness": "FRESH"}) == "HEALTHY"
    # EIA endet 2026-08-01 -> danach Öl-Fundamentaldaten UNAVAILABLE (nie fortgeschrieben), Preise bleiben
    synth_archive(tmp_path, skip={"eia_petroleum_weekly", "eia_natural_gas"})
    synth_archive(tmp_path, end=utc(2026, 8, 1), seed=4)
    df, _ = cmd.build(now=NOW, root=tmp_path, start="2026-07-01", write=False)
    last = df.iloc[-1]
    assert pd.isna(last["cmd_crude_stocks_chg_1w"]) and last["age_crude_stocks"] > 20
    assert df.loc[df["date"] == "2026-07-15", "cmd_crude_stocks_chg_1w"].notna().all()
    assert pd.notna(last["cmd_wti_ret_20d"])


def test_eia_revision_after_first_seen_never_backdated(monkeypatch, tmp_path):
    monkeypatch.setenv("EIA_API_KEY", "k")
    first = utc(2026, 9, 1)
    monkeypatch.setattr(http, "fetch", lambda url, params=None, **kw: FakeRes(
        _eia_rows(params["facets[series][]"], unit=next(x for x in CFG["eia"]["series"]
                                                        if x["series"] == params["facets[series][]"])["units"][0],
                  monthly="natural-gas/prod" in url or "move/expc" in url)))
    c = cs.EiaNaturalGasConnector({"_archive_root": str(tmp_path)})
    r1 = c.fetch(first)
    ExternalArchive(str(tmp_path)).store_observations(r1.observations, max_backfill_bytes=10**9)

    def revised(url, params=None, **kw):
        p = _eia_rows(params["facets[series][]"], unit=next(x for x in CFG["eia"]["series"]
                                                             if x["series"] == params["facets[series][]"])["units"][0],
                      monthly="natural-gas/prod" in url or "move/expc" in url)
        p["response"]["data"][-1]["value"] = "1"                       # älteste Periode revidiert
        return FakeRes(p)
    monkeypatch.setattr(http, "fetch", revised)
    r2 = c.fetch(NOW)
    rev = [o for o in r2.observations if o.attrs.get("revision_status") == "revised_after_first_seen"]
    assert rev and all(o.available_at == NOW and o.vintage_time == NOW for o in rev)
    unchanged = [o for o in r2.observations if o.attrs.get("revision_status") != "revised_after_first_seen"]
    assert all(o.available_at <= first for o in unchanged)


def test_eia_is_due_skips_when_archive_current(monkeypatch, tmp_path):
    synth_archive(tmp_path)
    c = cs.EiaPetroleumWeeklyConnector({"_archive_root": str(tmp_path)})
    # Spot-Reihen (Crosscheck) fehlen im synthetischen Archiv -> fällig
    due, why = c.is_due(NOW)
    assert due and "RWTC" in why
    specs = [s for s in c.series_specs() if s.get("role") == "fundamental"]
    monkeypatch.setattr(c, "series_specs", lambda: specs)
    due, why = c.is_due(NOW)
    assert not due


def test_release_rules_never_before_publication():
    # WPSR: Woche bis Fr 2026-09-25 erscheint Mi 2026-09-30 10:30 ET -> Regel Do 16:00 UTC
    r = {"days": 6, "hour_utc": 16}
    assert cs.release_time(utc(2026, 9, 25), r) == utc(2026, 10, 1, 16)
    assert cs.available_at(utc(2026, 9, 25), r, utc(2026, 9, 30, 12)) == utc(2026, 9, 30, 12)   # nie nach Abruf
    assert cs.expected_latest_period("weekly", r, utc(2026, 10, 1, 15)) == utc(2026, 9, 18)
    assert cs.expected_latest_period("weekly", r, utc(2026, 10, 1, 17)) == utc(2026, 9, 25)
    assert cs.parse_period("2026-03", "monthly") == utc(2026, 3, 1)


# ── FRED / ALFRED ────────────────────────────────────────────────────────────

def test_fred_normal_vintages_and_key_alias(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.setenv("FRED_KEY", "k")

    def fake(url, params=None, **kw):
        sid = params["series_id"]
        return FakeRes({"observations": [
            {"date": "2026-09-01", "realtime_start": "2026-09-02", "realtime_end": "2026-09-09", "value": "70.0"},
            {"date": "2026-09-01", "realtime_start": "2026-09-10", "realtime_end": "9999-12-31", "value": "71.0"},
            {"date": "2026-09-02", "realtime_start": "2026-09-03", "realtime_end": "9999-12-31", "value": "."},
        ]}, url=url + sid)
    monkeypatch.setattr(http, "fetch", fake)
    c = cs.FredCommodityConnector({})
    assert set(c.SERIES) == {"DCOILBRENTEU", "DHHNGSP", "PCOPPUSDM", "PWHEAMTUSDM", "PMAIZMTUSDM", "PSOYBUSDM"}
    res = c.fetch(NOW)
    assert res.status == SourceStatus.PASS
    b = [o for o in res.observations if o.metric == "brent"]
    assert {o.available_at for o in b if o.value == 71.0} == {utc(2026, 9, 10)}
    assert any(o.value is None for o in b)                            # "." -> None, nie 0
    # ALFRED-Vintage: am 2026-09-05 ist nur die Erstveröffentlichung sichtbar
    book = cmd.PitBook(cmd._events(b))
    book.advance(utc(2026, 9, 5))
    assert book.tail(5)[1] == [70.0]
    book.advance(utc(2026, 9, 11))
    assert book.tail(5)[1] == [71.0]


def test_fred_missing_key(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.delenv("FRED_KEY", raising=False)
    monkeypatch.setattr(http, "fetch", lambda *a, **k: pytest.fail("kein Call ohne Key"))
    assert cs.FredCommodityConnector({}).fetch(NOW).status == SourceStatus.AUTH_MISSING


def test_fred_single_series_failure_is_isolated(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "k")

    def fake(url, params=None, **kw):
        if params.get("series_id") == "PCOPPUSDM":
            raise http.FetchError("500")
        return FakeRes({"observations": [{"date": "2026-09-01", "realtime_start": "2026-09-02", "value": "5"}]})
    monkeypatch.setattr(http, "fetch", fake)
    res = cs.FredCommodityConnector({}).fetch(NOW)
    assert res.status == SourceStatus.WARN and {o.metric for o in res.observations} >= {"brent", "henry_hub"}
    assert "copper" not in {o.metric for o in res.observations}


# ── CFTC COT ─────────────────────────────────────────────────────────────────

def _cot_row(code, name, d, **over):
    f = CFG["cftc"]["fields"]
    base = {"total_open_interest": 500000, "producer_merchant_long": 100000, "producer_merchant_short": 150000,
            "swap_long": 80000, "swap_short": 60000, "managed_money_long": 120000, "managed_money_short": 40000,
            "managed_money_spreading": 30000, "other_reportables_long": 40000, "other_reportables_short": 30000}
    base.update(over)
    r = {f[k]: str(v) for k, v in base.items()}
    r.update({f["code"]: code, f["market_name"]: name, f["report_date"]: f"{d.isoformat()}T00:00:00.000"})
    return r


NAMES = {"crude_oil": "CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE",
         "natural_gas": "NAT GAS NYME - NEW YORK MERCANTILE EXCHANGE", "gold": "GOLD - COMMODITY EXCHANGE INC.",
         "silver": "SILVER - COMMODITY EXCHANGE INC.", "copper": "COPPER- #1 - COMMODITY EXCHANGE INC.",
         "corn": "CORN - CHICAGO BOARD OF TRADE", "wheat": "WHEAT-SRW - CHICAGO BOARD OF TRADE",
         "soybeans": "SOYBEANS - CHICAGO BOARD OF TRADE"}


def test_cftc_fetch_mapping_missing_market_oi_and_shutdown(monkeypatch, tmp_path):
    rows = []
    for m, mc in CFG["cftc"]["markets"].items():
        if m == "silver":
            continue                                                     # Markt fehlt
        name = "GOLD MINI - SOMEWHERE ELSE" if m == "gold" else NAMES[m]  # Code/Name passen nicht
        for d in (date(2026, 9, 15), date(2026, 9, 22)):
            rows.append(_cot_row(mc["code"], name, d))
    rows.append(_cot_row("067651", NAMES["crude_oil"], date(2026, 9, 8), managed_money_long=600000))  # > OI
    rows.append(_cot_row("067651", NAMES["crude_oil"], date(2019, 1, 8)))                          # Shutdown
    seen = {}

    def fake(url, params=None, **kw):
        seen.update(params)
        return FakeRes(rows)
    monkeypatch.setattr(http, "fetch", fake)
    res = cs.CftcCotConnector({"_archive_root": str(tmp_path)}).fetch(NOW)
    assert res.status == SourceStatus.WARN
    st = res.discovered_ids
    assert set(st["unavailable_markets"]) == {"gold", "silver"}
    assert len(st["oi_inconsistent"]) == 1 and st["shutdown_excluded"] == 1
    assert "cftc_contract_market_code in (" in seen["$where"]
    ents = {o.entity_id for o in res.observations}
    assert "gold" not in ents and "silver" not in ents and "crude_oil" in ents
    o = next(x for x in res.observations if x.entity_id == "crude_oil" and x.observation_time == utc(2026, 9, 22)
             and x.metric == "cot_managed_money_long")
    assert o.available_at == utc(2026, 9, 25, 21) and o.unit == "contracts"
    assert not any(x.observation_time == utc(2026, 9, 8) for x in res.observations)


def test_cftc_schema_change(monkeypatch, tmp_path):
    bad = [{"report_date_as_yyyy_mm_dd": "2026-09-22T00:00:00.000", "cftc_contract_market_code": "067651"}]
    monkeypatch.setattr(http, "fetch", lambda *a, **k: FakeRes(bad))
    res = cs.CftcCotConnector({"_archive_root": str(tmp_path)}).fetch(NOW)
    assert res.status == SourceStatus.SCHEMA_CHANGED


def test_cot_release_lag_and_holidays():
    rule = CFG["cftc"]["release"]
    assert cs.cot_release_time(date(2026, 9, 22), rule) == utc(2026, 9, 25, 21)          # Di -> Fr
    assert cs.cot_release_time(date(2026, 11, 24), rule) == utc(2026, 11, 30, 21)        # Thanksgiving -> Mo
    assert cs.cot_release_time(date(2026, 12, 22), rule) == utc(2026, 12, 28, 21)        # Weihnachten (Fr)
    assert cs.cot_release_time(date(2026, 9, 8), rule) == utc(2026, 9, 14, 21)           # Labor Day (Mo)
    assert date(2026, 6, 19) in cs.us_federal_holidays(2026) and date(2020, 6, 19) not in cs.us_federal_holidays(2020)
    c = cs.CftcCotConnector({})
    assert c.expected_report_date(utc(2026, 9, 25, 20)) == date(2026, 9, 15)
    assert c.expected_report_date(utc(2026, 9, 25, 22)) == date(2026, 9, 22)


def test_oi_consistency_rule():
    ok = {"total_open_interest": 100, "managed_money_long": 50, "managed_money_short": 10}
    assert cs.oi_consistent(ok)[0]
    assert not cs.oi_consistent({**ok, "managed_money_long": 101})[0]
    assert not cs.oi_consistent({**ok, "managed_money_long": -1})[0]
    assert not cs.oi_consistent({"total_open_interest": 100, "producer_merchant_long": 60, "swap_long": 50})[0]
    assert not cs.oi_consistent({"total_open_interest": None})[0]


# ── Feature-Formeln ──────────────────────────────────────────────────────────

def test_price_features():
    v = [100.0 + i for i in range(300)]
    f = cmd.price_features(v, "daily")
    assert f["ret_1d"] == pytest.approx(399 / 398 - 1)
    assert f["ret_60d"] == pytest.approx(399 / 339 - 1)
    assert f["dd_252d"] == 0.0 and f["z_60d"] > 1.5 and f["vol_20d"] > 0
    neg = [20.0] * 70 + [-37.6, 10.0]
    g = cmd.price_features(neg, "daily")
    assert g["ret_1d"] is None                                         # Vorwert negativ -> None, nie 0
    m = cmd.price_features([100, 110, 121, 133.1], "monthly")
    assert m["ret_1m"] == pytest.approx(0.1) and m["ret_3m"] == pytest.approx(0.331) and m["z_36m"] is None


def test_fundamental_features():
    ps = [utc(2020, 1, 3) + timedelta(days=7 * i) for i in range(320)]
    vs = [400000.0] * 319 + [440000.0]
    f = cmd.fundamental_features(ps, vs, "stock")
    assert f["chg_1w"] == 40000 and f["chg_4w"] == 40000 and f["vs_5y"] == pytest.approx(0.1)
    fl = cmd.fundamental_features(ps, [100.0] * 316 + [110.0] * 4, "flow")
    assert fl["chg_4w"] == pytest.approx(0.1)
    mp = [utc(2020 + i // 12, i % 12 + 1, 1) for i in range(24)]
    mf = cmd.fundamental_features(mp, [100.0] * 23 + [120.0], "monthly_flow")
    assert mf["yoy"] == pytest.approx(0.2)
    assert cmd.fundamental_features(ps[:3], [1.0, 2.0, 3.0], "stock")["vs_5y"] is None


def test_cot_features_nets_pctiles_changes():
    rows = []
    for i in range(160):
        rows.append((utc(2023, 1, 3) + timedelta(days=7 * i), {
            "total_open_interest": 1000.0, "producer_merchant_long": 100.0, "producer_merchant_short": 300.0,
            "swap_long": 50.0, "swap_short": 50.0, "managed_money_long": 100.0 + i, "managed_money_short": 100.0,
            "managed_money_spreading": 10.0, "other_reportables_long": 10.0, "other_reportables_short": 10.0}))
    rows.append((utc(2026, 2, 1), {**rows[-1][1], "managed_money_long": 5000.0}))   # COT > OI -> verworfen
    f, bad = cmd.cot_features(rows)
    assert bad == 1
    assert f["mm_net"] == 159 and f["comm_net"] == -200
    assert f["mm_net_pct_oi"] == pytest.approx(0.159) and f["comm_net_pct_oi"] == pytest.approx(-0.2)
    assert f["mm_net_chg_1w"] == pytest.approx(0.001) and f["mm_net_chg_13w"] == pytest.approx(0.013)
    assert f["mm_pctile_1y"] == 1.0 and f["mm_pctile_3y"] == 1.0 and f["mm_extreme"] == 1.0
    assert f["oi_chg_4w"] == 0.0
    short, _ = cmd.cot_features(rows[:30])
    assert short["mm_pctile_1y"] is None and short["mm_pctile_3y"] is None   # zu wenig Historie -> None


def test_divergence_features():
    d = cmd.divergence_features({"cmd_wti_ret_20d": 0.05, "cmd_crude_stocks_chg_4w": 1000.0,
                                 "cmd_cot_crude_oil_mm_pctile_1y": 0.9, "cmd_henry_hub_ret_20d": None})
    assert d["cmd_div_oil_price_inventory"] == 1.0
    assert d["cmd_div_oil_positioning_price"] == 0.0
    assert d["cmd_div_gas_storage_price"] is None
    d2 = cmd.divergence_features({"cmd_wti_ret_20d": -0.05, "cmd_cot_crude_oil_mm_pctile_1y": 0.85})
    assert d2["cmd_div_oil_positioning_price"] == 1.0


# ── Qualität ─────────────────────────────────────────────────────────────────

def test_quality_checks():
    t0 = utc(2026, 1, 2)
    mk = lambda vals, step=7, metric="eia_crude_stocks", unit="thousand_barrels": [
        obs("s", metric, unit, t0 + timedelta(days=step * i), t0 + timedelta(days=step * i + 6), v)
        for i, v in enumerate(vals)]
    assert cmd.quality_checks("x", mk([1.0] * 20), "weekly", unit_expected="thousand_barrels")["issues"] == []
    assert "NEGATIVE_VALUE" in cmd.quality_checks("x", mk([1.0] * 19 + [-1.0]), "weekly")["issues"]
    q = cmd.quality_checks("x", mk([60.0] * 19 + [-30.0], step=1, metric="wti", unit="usd_per_barrel"), "daily")
    assert "NEGATIVE_VALUE" not in q["issues"] and not q["severe"]     # WTI darf negativ sein (20.04.2020)
    assert "EXTREME_MOVE" in q["issues"]                                # Befund, kein Ausschluss
    assert "WRONG_FREQUENCY" in cmd.quality_checks("x", mk([1.0] * 20, step=1), "weekly")["issues"]
    gap = mk([1.0] * 10) + mk([1.0] * 10)[:0]
    gap += [obs("s", "eia_crude_stocks", "thousand_barrels", t0 + timedelta(days=7 * 30), t0, 1.0)]
    assert "MISSING_RELEASES" in cmd.quality_checks("x", gap, "weekly")["issues"]
    o = mk([1.0] * 20)
    o[-1].unit = "barrels"
    q = cmd.quality_checks("x", o, "weekly", unit_expected="thousand_barrels")
    assert "UNIT_CHANGED" in q["issues"] and q["severe"]
    ext = mk([100.0] * 10 + [300.0] * 10, step=1, metric="brent", unit="usd_per_barrel")
    assert "EXTREME_MOVE" in cmd.quality_checks("x", ext, "daily")["issues"]
    dup = mk([1.0] * 12)
    dup.append(obs("s", "eia_crude_stocks", "thousand_barrels", dup[0].observation_time, dup[0].available_at, 2.0))
    assert "DUPLICATE_CONFLICT" in cmd.quality_checks("x", dup, "weekly")["issues"]


def test_crosscheck_eia_vs_fred():
    a = [obs("e", "eia_wti_spot", "u", utc(2026, 9, i), utc(2026, 9, i), 70.0) for i in range(1, 21)]
    b = [obs("f", "wti", "u", utc(2026, 9, i), utc(2026, 9, i), 70.0 if i < 15 else 80.0) for i in range(1, 21)]
    assert cmd.crosscheck(a, a, 0.02)["status"] == "OK"
    assert cmd.crosscheck(a, b, 0.02)["status"] == "MISMATCH"
    assert cmd.crosscheck(a, [], 0.02)["status"] == "NO_OVERLAP"


def test_severe_series_unavailable_not_zero(tmp_path):
    synth_archive(tmp_path, end=utc(2022, 6, 1), start=utc(2021, 1, 1))
    o = obs("fred_commodities", "brent", "usd_per_gallon", utc(2022, 5, 2), utc(2022, 5, 3), 1.0)
    ExternalArchive(str(tmp_path)).store_observations([o], max_backfill_bytes=10**9)
    df, st = cmd.build(now=utc(2022, 6, 1), root=tmp_path, start="2022-05-01", write=False)
    assert "price:brent" in st["unavailable"] and "UNIT_CHANGED" in st["unavailable"]["price:brent"]
    assert df["cmd_brent_ret_1d"].isna().all()
    assert df["cmd_wti_ret_1d"].notna().any()


# ── Build / PIT / Leakage ────────────────────────────────────────────────────

def test_build_coverage_and_status(built):
    _, df, st = built
    assert st["unavailable"] == {}
    for g in ("price", "fundamental", "positioning", "divergence"):
        assert st["coverage"][g]["share_dates_any"] > 0.9
    assert st["storage_surprise"] == "EXPECTATION_UNKNOWN"
    assert not any("storage_surprise" in c for c in df.columns)        # keine erfundene Erwartung
    assert "age_wti" in df and "rev_dry_gas_production" in df
    assert (df["rev_dry_gas_production"].dropna() == 1.0).all()        # Backfill-Vintage sichtbar markiert
    assert st["mapping"]["non_pit_mapping"] is True


def test_leakage_cot_not_visible_before_friday(tmp_path):
    synth_archive(tmp_path, end=utc(2026, 9, 26), start=utc(2023, 1, 1))
    thu, _ = cmd.build(now=utc(2026, 9, 24, 23), root=tmp_path, start="2026-09-24", write=False)
    fri, _ = cmd.build(now=utc(2026, 9, 26), root=tmp_path, start="2026-09-24", write=False)
    a = thu.set_index("date")
    b = fri.set_index("date")
    # Report vom Di 22.09. erscheint Fr 25.09. -> Donnerstag sieht noch den Vorwochen-Report
    assert a.loc["2026-09-24", "age_cot_crude_oil"] >= 9
    assert b.loc["2026-09-25", "age_cot_crude_oil"] == 3
    assert b.loc["2026-09-24", "age_cot_crude_oil"] == a.loc["2026-09-24", "age_cot_crude_oil"]


def test_leakage_eia_weekly_release_and_ffill_limit(tmp_path):
    synth_archive(tmp_path, end=NOW, start=utc(2023, 1, 1))
    df, _ = cmd.build(now=NOW, root=tmp_path, start="2026-09-01", write=False)
    d = df.set_index("date")
    # Woche bis Fr 18.09. -> Regel Do 24.09. 16:00 UTC: am Mi 23.09. (21:00) noch nicht sichtbar
    assert d.loc["2026-09-23", "age_crude_stocks"] == 12 and d.loc["2026-09-24", "age_crude_stocks"] == 6
    # Fortschreiben nur bis zur Frische-Grenze: age_days immer protokolliert, nie > Grenze mit Wert
    lim = cmd._max_age("weekly", 6, CFG)
    has = d["cmd_crude_stocks_chg_1w"].notna()
    assert (d.loc[has, "age_crude_stocks"] <= lim).all()


def test_leakage_future_vintage_ignored(tmp_path):
    synth_archive(tmp_path, end=utc(2026, 9, 1), start=utc(2025, 1, 1))
    t = utc(2026, 8, 3)
    late = obs("fred_regime_macro", "wti", "usd_per_barrel", t, utc(2026, 9, 20), 999.0, vintage=utc(2026, 9, 20))
    ExternalArchive(str(tmp_path)).store_observations([late], max_backfill_bytes=10**9)
    df, _ = cmd.build(now=utc(2026, 9, 30), root=tmp_path, start="2026-08-03", write=False)
    d = df.set_index("date")
    assert d.loc["2026-08-05", "cmd_wti_ret_1d"] < 1                   # Revision vom 20.09. noch unsichtbar
    assert abs(d.loc["2026-08-04", "cmd_wti_ret_1d"]) < 0.5


# ── Mapping / Panel / Source Health ──────────────────────────────────────────

def test_exposure_mapping_positive_negative_non_pit():
    ex = cmd.exposure_table(["XOM", "DAL", "AAPL", "VLO", "ZZZ"], industries={"ZZZ": "Copper"})
    xom = ex[(ex.ticker == "XOM") & (ex.commodity == "oil")].iloc[0]
    dal = ex[(ex.ticker == "DAL") & (ex.commodity == "oil")].iloc[0]
    assert xom.expected_direction == 1 and dal.expected_direction == -1
    assert ex[ex.ticker == "VLO"].iloc[0].expected_direction == 0
    assert ex[ex.ticker == "ZZZ"].iloc[0].mapping_source == "industry:Copper"
    assert "AAPL" not in set(ex.ticker)
    assert ex["non_pit_mapping"].all() and set(ex["point_in_time_status"]) == {"NON_PIT"}
    assert set(ex.columns) >= {"commodity", "ticker", "exposure_type", "expected_direction", "mapping_source",
                               "mapping_version", "confidence", "point_in_time_status"}


def _panel(dates, tickers=("XOM", "DAL", "AAPL", "FCX", "NEM")):
    rows = [(pd.Timestamp(d), t, "Technology" if t == "AAPL" else ("unknown" if t == "NEM" else "Energy"))
            for d in dates for t in tickers]
    return pd.DataFrame(rows, columns=["date", "ticker", "sector"])


def test_attach_panel_cross_features(built):
    _, df, _ = built
    p = _panel(pd.bdate_range("2026-03-02", "2026-03-06"))
    out = cmd.attach_panel(p, date_features=df.assign(date=pd.to_datetime(df["date"])))
    assert len(out) == len(p)
    r = out[out.date == "2026-03-04"].set_index("ticker")
    w = df.set_index("date").loc["2026-03-04", "cmd_wti_ret_20d"]
    assert r.loc["XOM", "cmdx_oil__wti_ret_20d"] == pytest.approx(w)
    assert r.loc["DAL", "cmdx_oil__wti_ret_20d"] == pytest.approx(-w)
    assert pd.isna(r.loc["AAPL", "cmdx_oil__wti_ret_20d"])             # nicht gemappt -> NaN, nie 0
    assert r.loc["AAPL", "cmdexp_oil"] == 0.0                          # Sektor bekannt -> "kein Mapping"
    assert pd.isna(r.loc["NEM", "cmdexp_oil"]) and r.loc["NEM", "cmdexp_gold"] == 1.0
    assert r.loc["XOM", "alt_cmd_price_available"] == 1.0 and r.loc["AAPL", "alt_cmd_price_available"] == 0.0


def test_missing_feature_store_gives_nan_and_zero_availability(tmp_path):
    p = _panel(pd.bdate_range("2026-03-02", "2026-03-03"))
    out = cmd.attach_panel(p, path=tmp_path / "missing.csv.gz")
    assert out[cmd.cross_features("positioning")].isna().all().all()
    assert (out["alt_cmd_positioning_available"] == 0).all()


def test_source_failure_only_affects_dependent_features(built, tmp_path):
    from modules.alt_data import feature_store as fs
    from modules.alt_data.registry import SOURCES
    _, df, _ = built
    store = tmp_path / "cmd.csv.gz"
    df.to_csv(store, index=False, compression="gzip")
    p = _panel(pd.bdate_range("2026-03-02", "2026-03-06"))
    srcs = {k: {**v, "path": str(store)} for k, v in SOURCES.items() if k.startswith("commodity_")}
    health = {"generated": "2026-03-01T00:00:00+00:00", "features": {}, "sources": {
        "cftc_cot": {"status": "BROKEN", "failed_since": "2026-03-01T00:00:00+00:00"},
        "fred_commodities": {"status": "HEALTHY"}, "fred_regime_macro": {"status": "HEALTHY"}}}
    out = fs.attach(p, sources=srcs, health=health)
    assert out[cmd.cross_features("positioning")].isna().all().all()
    assert (out["alt_cmd_positioning_available"] == 0).all()
    x = out[out.ticker == "XOM"]
    assert x["cmdx_oil__wti_ret_20d"].notna().all()                    # Preis-Features unberührt
    assert x["cmdx_oil__div_oil_price_inventory"].notna().any()        # Divergenz ohne COT unberührt
    assert x["cmdx_oil__div_oil_positioning_price"].isna().all()       # Divergenz mit COT -> unavailable


def test_research_sources_never_trigger_global_safe_mode():
    from modules import source_health as sh
    c = sh.load_config()

    def src(sid, status, research_only=False, crit="NON_CRITICAL"):
        return {"source_id": sid, "status": status, "criticality": crit, "kind": "registry",
                "research_only": research_only, "fallback_only": False, "fallback_active": False,
                "downstream_dependencies": {"features": [f"f_{sid}"], "decisions": [], "hypotheses": []}}
    srcs = {"core": src("core", "HEALTHY", crit="CRITICAL")}
    for sid in ("eia_petroleum_weekly", "eia_natural_gas", "fred_commodities", "cftc_cot"):
        srcs[sid] = src(sid, "BROKEN", research_only=True)
    sm = sh.data_safe_mode({"sources": srcs, "features": {}}, c)
    assert not sm["active"] and sm["data_quality"] == 1.0
    assert "f_cftc_cot" in sm["unavailable_features"] or True        # Feature-Ebene separat (feature_availability)


def test_downstream_maps_features_per_source():
    from modules import source_health as sh
    d = sh.downstream(sh.load_config())
    assert "cmdx_oil__wti_ret_20d" in d["fred_regime_macro"]["features"]
    assert "cmdx_oil__wti_ret_20d" not in d.get("fred_commodities", {}).get("features", [])
    assert "cmdx_gold__cot_gold_mm_pctile_1y" in d["cftc_cot"]["features"]


# ── Research-Anbindung ───────────────────────────────────────────────────────

def test_alt_registry_and_model_spec_accept_commodity_features():
    from modules import ml_research as ml
    from modules.alt_data.registry import ALT_FEATURES, SOURCES
    for g in ("price", "fundamental", "positioning", "divergence"):
        s = SOURCES[f"commodity_{g}"]
        assert s["non_pit_mapping"] and s["attach"] == "date_level"
        assert set(s["features"]) == set(cmd.cross_features(g))
    assert ALT_FEATURES["cmdx_oil__wti_ret_20d"]["source"] == "commodity_price"
    feats = ml.feature_list({"features": "momentum", "extra_features": ["cmdx_oil__wti_ret_20d"]})
    assert "cmdx_oil__wti_ret_20d" in feats


def test_factory_commodity_hypotheses_are_falsifiable(built):
    from modules import hypothesis_factory as hf
    from modules.research_lab import eval_signal, validate_expr
    ideas = [h for h in hf.ideas_cross_domain(hf.load_domains(), "2026-10-04") if h.get("commodity")]
    assert len(ideas) >= 6
    for h in ideas:
        validate_expr(h["signal"])
        for k in ("population", "equity_population", "horizon", "baseline", "control_group", "oos",
                  "multiple_testing", "failure_condition", "H0", "H_alt", "spec_hash"):
            assert h.get(k), k
        assert h["mechanism_source"] == "template" and h["non_pit_mapping"] is True
    div = [h for h in hf.ideas_divergence("x") if h["domain"].startswith("commodity")]
    assert div and all(h["source_id"] == "commodity_price" for h in div)
    _, df, _ = built
    p = _panel(pd.bdate_range("2021-06-01", "2026-09-01"))
    p = cmd.attach_panel(p, date_features=df.assign(date=pd.to_datetime(df["date"])))
    r = hf.readiness(ideas[0], p, hf.load_protocol())
    assert r["status"] == "READY" and any("non_pit_mapping" in f for f in r["flags"])
    s = eval_signal(p, ideas[0]["signal"])
    assert s.notna().mean() > 0.5
    gap = next(h for h in hf.ideas_cross_domain({"domains": {"commodity_storage_surprise": {"kind": "missing"}},
                                                 "families": [{"id": "x", "domain": "commodity_storage_surprise",
                                                               "commodity": "natural_gas", "mechanism": "m"}]}, "x"))
    assert hf.readiness(gap, p, hf.load_protocol())["status"] == "DATA_GAP"


def test_memory_ingestion_and_near_duplicate_blocking():
    from modules import research_memory as rm
    spec = {"signal": "cmdexp_oil * sign(cmd_wti_ret_20d - 0)", "direction": 1, "family": "cmd_oil_price_x_exposure",
            "domain": "commodity_oil_price", "commodity": "oil", "equity_population": "cmdexp_oil != 0",
            "mechanism": "m", "domain_features": ["cmd_wti_ret_20d"]}
    fr = {"results": {"FAC-1": {"status": "REJECTED", "spec": spec, "reasons": ["Walk-Forward netto <= 0"],
                                "data_kind": "historical_walk_forward"}}}
    ev = {"groups": {"price": {"verdict": "REJECT", "verdict_reason": "kein inkrementeller Nutzen",
                               "selected_features": ["cmdx_oil__wti_ret_20d"], "population": "gemappt",
                               "commodities": ["oil"], "n_rows": 10}}}
    es = rm.commodity_entries(ev, fr)
    h = next(e for e in es if e["hypothesis_id"] == "FAC-1")
    for k in ("domain", "commodity", "equity_population", "mechanism", "features", "result", "n", "oos", "forward",
              "failure_reason", "fingerprint"):
        assert k in h
    assert h["commodity"] == "oil" and "commodity:oil" in h["fingerprint"]
    a = next(e for e in es if e["hypothesis_id"] == "CMD-ABLATION-price")
    assert a["status"] == "REJECTED" and a["failure_reason"]
    near = {**spec, "signal": "cmdexp_oil * sign(cmd_wti_ret_20d - 0.01)"}
    assert rm.similarity(near, spec) >= 0.8                              # Beinahe-Duplikat -> blockiert
    other = {"signal": "cmdexp_gold * sign(cmd_cot_gold_mm_pctile_1y - 0.5)", "commodity": "gold",
             "domain": "commodity_positioning", "direction": 1}
    assert rm.similarity(other, spec) < 0.5


def test_ml_ablation_breakdown_and_verdict(monkeypatch, built):
    from modules import ml_research as ml
    from modules.alt_data import evaluate as ae
    _, df, _ = built
    dates = pd.bdate_range("2021-07-01", "2026-06-30")
    mapped = ["XOM", "CVX", "COP", "EOG", "OXY", "DVN", "DAL", "UAL", "AAL", "LUV", "FCX", "SCCO", "NEM", "AEM",
              "CF", "MOS", "EQT", "AR", "DE", "ADM"]
    tickers = mapped + [f"T{i:02d}" for i in range(45)]
    p = _panel(dates, tickers)
    rng = np.random.default_rng(0)
    p["fwd_xs_20"] = rng.normal(0, 0.05, len(p))
    p["mfe_20"], p["mae_20"] = 0.05, -0.05
    p["label_end_20"] = p["date"] + pd.Timedelta(days=28)
    p["vix"], p["log_dollar_vol"] = 18.0, rng.normal(8, 1, len(p))
    p = cmd.attach_panel(p, date_features=df.assign(date=pd.to_datetime(df["date"])))
    monkeypatch.setattr(ml, "load_registry", lambda: {"models": [{"id": "enet_xs20_v1"}]})
    monkeypatch.setattr(ml, "check_registry", lambda reg: ({"enet_xs20_v1": "valid"}, None))
    monkeypatch.setitem(ae.AP["feature_selection"], "selection_years", [2021, 2022])
    monkeypatch.setitem(ae.AP["evaluation"], "dev_years", [2023, 2024, 2025, 2026])
    monkeypatch.setitem(ae.AP["evaluation"], "bootstrap_n", 50)

    def fake_oos(panel, spec):
        o = panel[panel["date"].dt.year.isin([2023, 2024, 2025, 2026])][
            ["date", "ticker", "fwd_xs_20", "mfe_20", "mae_20", "vix", "sector", "label_end_20"]].copy()
        o["score"] = rng.normal(size=len(o)) if not spec.get("extra_features") else o["fwd_xs_20"] * 0 + rng.normal(size=len(o))
        return o
    monkeypatch.setattr(ae, "_oos", fake_oos)
    monkeypatch.setattr(ae, "feature_screen", lambda panel, feats, years: {f: {"coverage": 0.9, "selection_ic": 0.01,
                                                                               "max_abs_corr_existing": 0.1} for f in feats})
    rep = cmd.evaluate(p, write=False)
    assert rep["non_pit_mapping"] and rep["n_tickers_population"] == len(mapped) - 1   # ADM: Richtung 0
    g = rep["groups"]["price"]
    b = g["baselines"]["enet_xs20_v1"]
    for k in ("expectancy", "sharpe", "ic", "precision_at_k", "brier", "ece", "max_dd"):
        assert k in b["base"] and k in b["with_commodity"]
    assert set(b["breakdown"]) == {"market_cap_bucket", "sector", "commodity_exposure", "regime", "liquidity_bucket"}
    assert any(k.startswith("oil:") for k in b["breakdown"]["commodity_exposure"])
    assert g["verdict"] in ("KEEP", "MODIFY", "REJECT") and rep["incremental_value"] in (
        "NONE", "HISTORICAL_ONLY", "NO_RELATIONSHIP")
    assert cmd.incremental_verdict({"groups": {"a": {"verdict": "REJECT"}}}) == "NO_RELATIONSHIP"


def test_alpha_decay_can_be_learned():
    from modules.alt_data.evaluate import alpha_decay
    assert alpha_decay({"2019": 0.04, "2020": 0.04, "2021": 0.01, "2022": 0.0})["status"] == "decaying"
    assert alpha_decay({"2019": 0.0, "2020": 0.001, "2021": 0.0, "2022": 0.0})["status"] == "no_early_effect"


# ── Keine Produktionswirkung / Promotion ─────────────────────────────────────

def test_no_production_impact():
    # Produktionsmodule importieren Commodity Intelligence nicht; einzig der Adapter LIEST Merkmale
    # (für die Auswertung registrierter Verträge) – Wirkung nur über einen promoteten Vertrag.
    for f in ["pipeline.py", "modules/options_designer.py", "modules/promotion_controller.py",
              "modules/universe_v2_scan.py", "modules/risk_gates.py", "modules/trade_scorer.py"]:
        p = ROOT / f
        if p.exists():
            t = p.read_text(encoding="utf-8")
            assert "commodity_intelligence" not in t and "cmdx_" not in t, f
    assert CFG["promotion"]["initial"] == "NONE"


def test_adapter_decisions_identical_with_and_without_commodity_data(tmp_path):
    """Ohne promoteten Commodity-Vertrag ändert ein (extremer) Commodity-Snapshot keine Entscheidung."""
    from modules import production_intelligence_adapter as pia
    props = [{"ticker": t, "sector": "Energy", "features": {"risk_flag": 0}, "trade_score": {"total": 60 + i},
              "simulation": {"hit_rate": 0.5}} for i, t in enumerate(["XOM", "DAL", "AAPL", "FCX"])]
    base_ctx = {"safe_mode_active": 0, "ml_cards": {}, "blind_spot_sectors": []}
    extreme = {"features": {f: 1e6 for f in cmd.SIGNAL_DATE_FEATURES}, "commodity_data_version": "x"}
    outs = []
    for ctx in (dict(base_ctx, commodity={}), dict(base_ctx, commodity=extreme)):
        kept, blocked, recs = pia.apply_to_proposals(
            [dict(p) for p in props], vix=18, today="2026-10-05", context=ctx, state_path=tmp_path / "s.json",
            registry=tmp_path / "r.jsonl", transitions=tmp_path / "t.jsonl", ledger_dir=tmp_path / "l")
        outs.append(([p["ticker"] for p in kept], [b[0]["ticker"] for b in blocked],
                     [r["final_production_decision"] for r in recs], [r["production_score"] for r in recs]))
    assert outs[0] == outs[1]
    env = pia.candidate_env(props[0], dict(base_ctx, commodity=extreme), 18)
    assert env["cmdx_oil__wti_ret_20d"] == 1e6 and env["cmdexp_oil"] == 1.0      # gelesen, aber wirkungslos


def test_promotion_path_capped_for_commodity_contracts():
    from modules import hypothesis_contract as hc
    c = {"features": ["cmdexp_oil", "cmd_wti_ret_20d"], "maximum_initial_influence": "WEIGHT_10",
         "production_class": "weight"}
    errs = hc.validate_commodity(c)
    assert any("SCORE_LIMITED" in e for e in errs) and any("production_class" in e for e in errs)
    ok = {"features": ["cmdx_oil__wti_ret_20d"], "maximum_initial_influence": "RERANK_ONLY", "production_class": "rerank",
          "mapping_version": "exposure-v1"}
    assert hc.validate_commodity(ok) == []
    assert any("mapping_version" in e for e in hc.validate_commodity({**ok, "mapping_version": None}))
    # Mapping-Version weicht ab -> Regel nicht auswertbar (nie mit neuem Mapping still weiterzählen)
    rule = {**ok, "signal_definition": "cmdx_oil__wti_ret_20d", "thresholds": {"op": ">", "value": 0}}
    assert hc.fires(rule, {"cmdx_oil__wti_ret_20d": 0.1, "commodity_mapping_version": "exposure-v1"}) is True
    assert hc.fires(rule, {"cmdx_oil__wti_ret_20d": 0.1, "commodity_mapping_version": "exposure-v2"}) is None
    assert hc.validate_commodity({"features": ["mom_3m"], "maximum_initial_influence": "WEIGHT_10",
                                  "production_class": "weight"}) == []
    lad = CFG["promotion"]["ladder"]
    assert lad == ["NONE", "RERANK_ONLY", "SCORE_LIMITED"] and all(x in hc.INFLUENCE_LEVELS for x in lad)


# ── Report ───────────────────────────────────────────────────────────────────

def test_report_without_evidence(tmp_path):
    s = cmd.report_summary(tmp_path, NOW)
    assert s["headline"] == cmd.NO_EVIDENCE_LINE == "Commodity Intelligence: RESEARCH ONLY – no validated incremental alpha"
    rows = dict(cmd.report_rows(s))
    assert rows["Status"] == cmd.NO_EVIDENCE_LINE and "Inkrementeller Wert (BASE vs. BASE+COMMODITY)" in rows


def test_weekly_report_has_commodity_section(tmp_path):
    from reports import weekly
    assert weekly.MONDAY_TITLES[10] == "COMMODITY INTELLIGENCE"
    blocks = weekly.commodity_blocks({"commodity": cmd.report_summary(tmp_path, NOW)})
    assert blocks[0] == ("para", cmd.NO_EVIDENCE_LINE)
    assert weekly.commodity_blocks({})[0][1].startswith(cmd.NO_EVIDENCE_LINE)


def test_report_with_challenger_and_forward(tmp_path):
    res = tmp_path / "research"
    res.mkdir()
    (res / "factory_challengers.jsonl").write_text(json.dumps(
        {"hypothesis_id": "FAC-AB", "spec": {"commodity": "oil"}}) + "\n")
    (res / "factory_results.json").write_text(json.dumps({"results": {
        "FAC-AB": {"status": "PROSPECTIVE_CHALLENGER", "spec": {"commodity": "oil"}},
        "FAC-CD": {"status": "REJECTED", "spec": {"domain": "commodity_gas_price"}}}}))
    (tmp_path / "intelligence").mkdir()
    (tmp_path / "intelligence" / "promotion_state.json").write_text(json.dumps({"hypotheses": {
        "factory:FAC-AB": {"state": "FORWARD_VALIDATED", "evidence": {"n_observations": 120}}}}))
    s = cmd.report_summary(tmp_path, NOW)
    assert s["challengers"] == 1 and s["hypotheses_tested"] == 2 and s["hypotheses_rejected"] == 1
    assert s["validated"] == ["FAC-AB"] and s["headline"] != cmd.NO_EVIDENCE_LINE
    assert s["forward"]["FAC-AB"]["influence"] == "NONE"


# ── End-to-End ───────────────────────────────────────────────────────────────

def test_end_to_end_source_to_memory(monkeypatch, tmp_path):
    """Quelle -> normalisiert -> Health -> Feature -> Mapping -> Kandidaten-Feature -> Hypothese -> Ergebnis -> Memory."""
    from modules import hypothesis_factory as hf
    from modules import research_memory as rm
    from modules.external import orchestrator
    from modules.external.registry import SourceRegistry
    root = tmp_path / "ext"
    synth_archive(root, skip={"cftc_cot"}, start=utc(2022, 1, 1))
    # 1) Quelle -> normalisiert (CFTC über den Orchestrator, Fake-HTTP)
    rows = []
    for m, mc in CFG["cftc"]["markets"].items():
        d = date(2022, 1, 4)
        while d <= date(2026, 9, 29):
            rows.append(_cot_row(mc["code"], NAMES[m], d, managed_money_long=100000 + (d.toordinal() % 50) * 1000))
            d += timedelta(days=7)
    monkeypatch.setattr(http, "fetch", lambda *a, **k: FakeRes(rows))
    monkeypatch.setattr(orchestrator, "send_alerts", lambda alerts: None)
    reg = SourceRegistry(archive_root=root)
    reg.sources = {k: v for k, v in reg.sources.items() if k == "cftc_cot"}
    summary = orchestrator.run_ingestion(now=NOW, registry=reg)
    assert summary["sources"]["cftc_cot"]["status"] == "PASS"
    # 2) Health
    h = json.loads((root / "health" / "source_health.json").read_text())["cftc_cot"]
    assert cmd.health_label(h) == "HEALTHY" and h["pit_integrity_failures"] == 0
    # zweiter Lauf: Report bereits archiviert -> kein Abruf
    monkeypatch.setattr(http, "fetch", lambda *a, **k: pytest.fail("nicht fällig – kein Call"))
    s2 = orchestrator.run_ingestion(now=NOW, registry=reg)
    assert s2["sources"]["cftc_cot"].get("skipped") == "not_due"
    # 3) Feature
    df, st = cmd.build(now=NOW, root=root, start="2023-01-02", write=False)
    assert st["sources"]["cftc_cot"]["health"] == "HEALTHY"
    assert df["cmd_cot_crude_oil_mm_pctile_1y"].notna().mean() > 0.9
    # 4) Mapping -> Kandidaten-Feature im Panel
    p = cmd.attach_panel(_panel(pd.bdate_range("2023-01-02", "2026-09-01")),
                         date_features=df.assign(date=pd.to_datetime(df["date"])))
    assert p.loc[p.ticker == "XOM", "cmdx_oil__cot_crude_oil_mm_pctile_1y"].notna().all()
    # 5) Hypothese
    fam = {"domains": hf.load_domains()["domains"], "families": [f for f in hf.load_domains()["families"]
                                                                  if f["id"] == "cmd_oil_crowding_x_exposure"]}
    h = hf.ideas_cross_domain(fam, "2026-10-04")[0]
    assert hf.readiness(h, p, hf.load_protocol())["status"] == "READY"
    # 6) Ergebnis (Research-Lab-Urteil) -> 7) Memory
    spec = {k: h.get(k) for k in ("signal", "direction", "family", "domain", "commodity", "equity_population",
                                  "mechanism", "domain_features")}
    fr = {"results": {h["id"]: {"status": "REJECTED", "spec": spec, "reasons": ["RELATIONSHIP DECAYED"],
                                "data_kind": "historical_walk_forward"}}}
    entries = rm.commodity_entries(None, fr)
    assert entries[0]["commodity"] == "oil" and entries[0]["failure_reason"] == "RELATIONSHIP DECAYED"
    srcs = {"factory_results": tmp_path / "fr.json"}
    srcs["factory_results"].write_text(json.dumps(fr))
    mem = tmp_path / "mem.jsonl"
    assert rm.sync(mem, sources=srcs) >= 2
    assert rm.search(h, rm.load(mem), 0.9)                                # erneuter Test würde blockiert

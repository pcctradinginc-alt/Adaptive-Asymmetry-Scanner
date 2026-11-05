"""IMF SDMX 3.0 research feeds: decoding, fail-closed validation, PIT vintages, archive path."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from modules.external import http
from modules.external.pit import available_as_of
from modules.external.sources import imf_sdmx3 as m
from modules.external.sources.base import SourceStatus

NOW = datetime(2026, 11, 5, 6, tzinfo=timezone.utc)


def res(payload, url="https://api.imf.org/external/sdmx/3.0/data/dataflow/x", ctype=m.DATA_ACCEPT):
    if isinstance(payload, (dict, list)):
        payload = json.dumps(payload).encode()
    return http.FetchResult(url=url, status=200, content=payload, content_type=ctype,
                            retrieved_at=NOW, content_hash="sha", fingerprint="fp",
                            bytes=len(payload))


def message(spec, series, times):
    """series: list of (codes-dict, {time_index: value})."""
    dims = []
    for d in spec["dimensions"]:
        vals = []
        for codes, _ in series:
            if {"id": codes[d]} not in vals:
                vals.append({"id": codes[d]})
        dims.append({"id": d, "values": vals})
    rows = {}
    for codes, obs in series:
        key = ":".join(str(next(i for i, v in enumerate(dims[j]["values"]) if v["id"] == codes[d]))
                       for j, d in enumerate(spec["dimensions"]))
        rows[key] = {"observations": {str(k): [v, None, 0] for k, v in obs.items()}}
    return {"data": {"structures": [{"dimensions": {
        "series": dims, "observation": [{"id": "TIME_PERIOD", "values": [{"value": t} for t in times]}]}}],
        "dataSets": [{"structure": 0, "series": rows}]}}


TIMES = {"A": ("2024", "2025", "2026"), "Q": ("2025-Q4", "2026-Q1", "2026-Q2"),
         "M": ("2026-M06", "2026-M07", "2026-M08")}


def full_message(sid, times=None, value="1.5"):
    spec = m.SPECS[sid]
    times = times or TIMES[spec["key"]["FREQUENCY"][0]]
    series = []
    for country in spec["key"]["COUNTRY"][:2]:
        for ind in spec["key"]["INDICATOR"]:
            codes = {"COUNTRY": country, "INDICATOR": ind}
            for d in spec["dimensions"]:
                if d not in codes:
                    allowed = spec["key"].get(d)
                    codes[d] = allowed[0] if allowed else "_T"
            series.append((codes, {i: value for i in range(len(times))}))
    return message(spec, series, times)


CATALOG = (b'<?xml version="1.0"?><message:Structure xmlns:message="m" xmlns:s="s"><message:Structures>'
           b'<s:Dataflows>'
           b'<s:Dataflow id="WEO" agencyID="IMF.RES" version="9.0.0"/>'
           b'<s:Dataflow id="WEO_2025_OCT_VINTAGE" agencyID="IMF.RES" version="1.0.0"/>'
           b'<s:Dataflow id="WEO_2026_APR_VINTAGE" agencyID="IMF.RES" version="1.0.0"/>'
           b'<s:Dataflow id="WEO_2026_DEC_VINTAGE" agencyID="IMF.RES" version="1.0.0"/>'
           b'<s:Dataflow id="WEO_2019_APR_VINTAGE" agencyID="IMF.RES" version="1.0.0"/>'
           b'<s:Dataflow id="WEO_2025_OCT_VINTAGE" agencyID="IMF.OTHER" version="1.0.0"/>'
           b'</s:Dataflows></message:Structures></message:Structure>')


class Fake:
    def __init__(self, sid, *, vintage_value="1.2", fail_catalog=False, current=None):
        self.sid, self.calls = sid, []
        self.vintage_value, self.fail_catalog, self.current = vintage_value, fail_catalog, current

    def __call__(self, url, params=None, headers=None, **kw):
        self.calls.append((url, params, headers))
        if url.endswith("/structure/dataflow"):
            if self.fail_catalog:
                raise http.FetchError("500 für catalog")
            return res(CATALOG, url, ctype=m.STRUCTURE_ACCEPT)
        assert headers["Accept"] == m.DATA_ACCEPT
        assert "c[TIME_PERIOD]" not in (params or {})
        if "_VINTAGE/" in url:
            return res(full_message(self.sid, value=self.vintage_value), url)
        return res(full_message(self.sid) if self.current is None else self.current, url)


ACTIVE = ("imf_fsi", "imf_weo", "imf_mfs", "imf_gdd", "imf_fiscal_monitor", "imf_climate", "imf_fas")


@pytest.mark.parametrize("sid", ACTIVE)
def test_every_active_feed_decodes_and_requests_verified_key(sid):
    fake = Fake(sid)
    r = m.CONNECTORS[sid]({}, fetch_fn=fake).fetch(NOW)
    assert r.status == SourceStatus.PASS, r.message
    spec = m.SPECS[sid]
    url, params, _ = fake.calls[0]
    assert url == f'{m.BASE}/data/dataflow/{spec["agency"]}/{spec["flow"]}/+/{m.build_key(spec)}'
    assert params["startPeriod"] == str(NOW.year - spec["years_back"])
    assert {o.entity_id for o in r.observations} >= set(spec["key"]["COUNTRY"][:2])
    current = [o for o in r.observations if o.attrs["edition"] == "current"]
    assert current and all(o.available_at == NOW for o in current)
    assert all(o.attrs["research_only"] for o in r.observations)


def test_key_uses_plus_for_multi_values_and_star_for_wildcards():
    assert m.build_key(m.SPECS["imf_weo"]) == "USA+DEU+JPN+GBR+FRA+CHN.NGDP_RPCH+PCPIPCH+LUR+GGXWDG_NGDP+BCA_NGDPD.A"
    assert m.build_key(m.SPECS["imf_fas"]).split(".")[2:5] == ["*", "*", "*"]


@pytest.mark.parametrize("raw,period", [("2025", (2025, 1)), ("2025-Q3", (2025, 7)),
                                        ("2026-M05", (2026, 5)), ("2026-05", (2026, 5))])
def test_period_formats(raw, period):
    d = m.parse_period(raw)
    assert (d.year, d.month) == period and d.tzinfo is not None


def test_unknown_period_format_fails_closed():
    with pytest.raises(m.SDMX3Error):
        m.parse_period("2026-W12")


def test_official_vintages_become_conservative_historical_revisions():
    r = m.CONNECTORS["imf_weo"]({}, fetch_fn=Fake("imf_weo")).fetch(NOW)
    assert r.status == SourceStatus.PASS
    assert r.discovered_ids["vintages"] == ["WEO_2025_OCT_VINTAGE", "WEO_2026_APR_VINTAGE"]
    by_edition = {o.attrs["edition"]: o.available_at for o in r.observations}
    assert by_edition["WEO_2025_OCT_VINTAGE"] == datetime(2025, 11, 1, tzinfo=timezone.utc)
    assert by_edition["WEO_2026_APR_VINTAGE"] == datetime(2026, 5, 1, tzinfo=timezone.utc)
    # Same target (series/country/period) across editions -> revision history.
    one = [o for o in r.observations if o.entity_id == "USA" and o.attrs["indicator"] == "LUR"
           and o.observation_time.year == 2024]
    assert len({o.identity_key() for o in one}) == 1 and len(one) == 3
    asof = available_as_of(r.observations, datetime(2026, 1, 15, tzinfo=timezone.utc))
    assert asof and all(o.attrs["edition"] == "WEO_2025_OCT_VINTAGE" for o in asof)
    assert not available_as_of(r.observations, datetime(2025, 10, 31, tzinfo=timezone.utc))
    assert all(o.available_at <= o.retrieved_at for o in r.observations)


def test_vintage_month_mapping():
    assert m.vintage_available_at("FM_2025_DEC_VINTAGE", "FM") == datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert m.vintage_available_at("MFS_MA_2026_JUNE_VINTAGE", "MFS_MA") == datetime(2026, 7, 1, tzinfo=timezone.utc)
    assert m.vintage_available_at("WEO_FOO_VINTAGE", "WEO") is None
    assert m.vintage_available_at("WEOX_2025_OCT_VINTAGE", "WEO") is None


def test_vintage_failure_keeps_current_data_but_warns():
    r = m.CONNECTORS["imf_weo"]({}, fetch_fn=Fake("imf_weo", fail_catalog=True)).fetch(NOW)
    assert r.status == SourceStatus.WARN
    assert r.observations and "vintages incomplete" in r.message


def test_projection_flag_for_current_and_future_years():
    r = m.CONNECTORS["imf_weo"]({}, fetch_fn=Fake("imf_weo")).fetch(NOW)
    cur = [o for o in r.observations if o.attrs["edition"] == "current"]
    assert {o.observation_time.year for o in cur if o.attrs["is_projection"]} == {2026}
    for o in r.observations:
        if o.attrs["is_projection"]:
            assert o.forecast_issue_time == o.available_at and o.forecast_valid_time == o.observation_time
        else:
            assert o.forecast_issue_time is None and o.forecast_valid_time is None


def _spec_msg(sid, mutate):
    msg = full_message(sid)
    mutate(msg)
    return msg


@pytest.mark.parametrize("mutate,reason", [
    (lambda x: x["data"]["structures"][0]["dimensions"]["series"].reverse(), "DIMENSION_ORDER"),
    (lambda x: x["data"]["structures"][0]["dimensions"]["series"][0]["values"].__setitem__(0, {"id": "XXX"}),
     "UNREQUESTED_CODE"),
    (lambda x: next(iter(x["data"]["dataSets"][0]["series"].values()))["observations"].__setitem__("0", ["abc"]),
     "NON_NUMERIC"),
    (lambda x: next(iter(x["data"]["dataSets"][0]["series"].values()))["observations"].__setitem__("0", ["nan"]),
     "NON_FINITE"),
    (lambda x: next(iter(x["data"]["dataSets"][0]["series"].values()))["observations"].__setitem__("9", ["1"]),
     "TIME_INDEX"),
    (lambda x: x["data"]["dataSets"].append({}), "MULTIPLE"),
])
def test_malformed_payloads_fail_closed_without_partial_observations(mutate, reason):
    fake = Fake("imf_gdd", current=_spec_msg("imf_gdd", mutate))
    r = m.CONNECTORS["imf_gdd"]({}, fetch_fn=fake).fetch(NOW)
    assert r.status == SourceStatus.SCHEMA_CHANGED and reason in r.message
    assert r.observations == []


@pytest.mark.parametrize("raw", [b"", b"<html>blocked</html>", b"{not json", b'{"data": {}}'])
def test_empty_html_or_non_sdmx_bodies_fail(raw):
    fake = Fake("imf_gdd", current=raw)
    r = m.CONNECTORS["imf_gdd"]({}, fetch_fn=fake).fetch(NOW)
    assert r.status == SourceStatus.SCHEMA_CHANGED and not r.observations


def test_missing_required_indicator_is_schema_changed():
    spec = m.SPECS["imf_gdd"]
    series = [({"COUNTRY": c, "INDICATOR": "FL_PS_POGDP_PT", "FREQUENCY": "A"}, {0: "100"})
              for c in ("USA", "DEU")]
    fake = Fake("imf_gdd", current=message(spec, series, ("2024",)))
    r = m.CONNECTORS["imf_gdd"]({}, fetch_fn=fake).fetch(NOW)
    assert r.status == SourceStatus.SCHEMA_CHANGED and "REQUIRED_INDICATOR_ABSENT" in r.message


def test_official_missing_value_is_skipped_never_zero():
    spec = m.SPECS["imf_fiscal_monitor"]
    msg = full_message("imf_fiscal_monitor")
    first = next(iter(msg["data"]["dataSets"][0]["series"].values()))
    first["observations"]["0"] = [None]
    fake = Fake("imf_fiscal_monitor", current=msg, fail_catalog=True)
    r = m.CONNECTORS["imf_fiscal_monitor"]({}, fetch_fn=fake).fetch(NOW)
    assert r.observations and all(o.value != 0 for o in r.observations)
    assert len([o for o in r.observations]) == len(spec["key"]["INDICATOR"]) * 2 * 3 - 1


def test_upstream_http_error_is_fail_not_schema():
    def boom(url, **kw):
        raise http.FetchError("500 für x")
    r = m.CONNECTORS["imf_fsi"]({}, fetch_fn=boom).fetch(NOW)
    assert r.status == SourceStatus.FAIL and not r.observations


def test_naive_now_rejected():
    r = m.CONNECTORS["imf_fsi"]({}, fetch_fn=Fake("imf_fsi")).fetch(datetime(2026, 11, 5))
    assert r.status == SourceStatus.SCHEMA_CHANGED


def test_registry_seven_active_research_only_and_aipi_disabled():
    from modules.external.registry import SourceRegistry, gate_source
    reg = SourceRegistry()
    imf = {sid: cfg for sid, cfg in reg.sources.items() if cfg.get("family") == "imf_macro"}
    assert set(imf) == set(ACTIVE) | {"imf_aipi"}
    for sid in ACTIVE:
        cfg = imf[sid]
        assert gate_source(cfg) == (True, "ok")
        assert reg.connector_for(sid) is m.CONNECTORS[sid]
        assert cfg["research_only"] is True and cfg["production_promotion_allowed"] is False
        assert cfg["run_only_by_family"] is True and cfg["min_refetch_days"] >= 14
        spec = m.SPECS[sid]
        assert cfg["agency"] == spec["agency"] and cfg["dataflow"] == spec["flow"]
    assert imf["imf_aipi"]["enabled"] is False
    assert gate_source(imf["imf_aipi"]) == (False, "disabled")
    assert reg.connector_for("imf_aipi") is None


def test_ingest_writes_archive_and_research_audit_reads_it(tmp_path, monkeypatch):
    """End-to-end offline: connector -> orchestrator ingest -> archive -> research summary."""
    from modules import imf_macro_research as research
    from modules.external import orchestrator
    from modules.external.archive import ExternalArchive
    from modules.external.registry import SourceRegistry

    reg = SourceRegistry()
    reg.archive_root = tmp_path / "archive"
    fakes = {sid: Fake(sid) for sid in ACTIVE}
    for sid in ACTIVE:
        cls = type(f"F_{sid}", (m.IMFSDMX3Connector,),
                   {"source_id": sid, "__init__": (lambda self, cfg=None, _f=fakes[sid]:
                                                   m.IMFSDMX3Connector.__init__(self, cfg, fetch_fn=_f))})
        reg.connectors[sid] = cls
    summary = orchestrator.run_ingestion(now=NOW, families=["imf_macro"], registry=reg)
    for sid in ACTIVE:
        assert summary["sources"][sid]["status"] in ("PASS", "WARN"), summary["sources"][sid]
    archive = ExternalArchive(reg.archive_root)
    report = research.summarize(NOW, registry=reg, archive=archive)
    assert report["expected_sources"] == sorted(ACTIVE)
    assert report["all_sources_ready"] is True, {k: (v.get("ready"), v.get("status"), v.get("n_valid_observations"), v.get("n_countries"), v.get("staleness")) for k, v in report["sources"].items()}
    assert report["sources"]["imf_aipi"]["status"] == "DISABLED"
    assert report["candidate_features"]
    weo = [f for f in report["candidate_features"] if f["source_id"] == "imf_weo"]
    assert any("as_observed_forecast_revision" in f for f in weo)
    assert report["model_input_enabled"] is False

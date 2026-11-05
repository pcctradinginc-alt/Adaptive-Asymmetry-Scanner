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


def _flow(fid, agency="IMF.RES", vintage=None, updated=None):
    ann = b""
    if vintage:
        ann += (b'<c:Annotation id="VINTAGE"><c:AnnotationTitle>Vintage for ' + vintage.encode()
                + b'</c:AnnotationTitle></c:Annotation>')
    if updated:
        ann += b'<c:Annotation id="lastUpdatedAt"><c:AnnotationValue>' + updated.encode() + b'</c:AnnotationValue></c:Annotation>'
    return (b'<s:Dataflow id="' + fid.encode() + b'" agencyID="' + agency.encode() + b'" version="1.0.0">'
            + (b"<c:Annotations>" + ann + b"</c:Annotations>" if ann else b"") + b"</s:Dataflow>")


CATALOG = (b'<?xml version="1.0"?><message:Structure xmlns:message="m" xmlns:s="s" xmlns:c="c">'
           b'<message:Structures><s:Dataflows>'
           + _flow("WEO", updated="2025-10-08T18:00:33.590181Z")
           # Edition 2025-10-14, but modified 2026-03-30 -> only provable from 2026-03-31.
           + _flow("WEO_2025_OCT_VINTAGE", vintage="2025-10-14", updated="2026-03-30T18:36:09.484648Z")
           # No lastUpdatedAt -> no verifiable release -> first seen at retrieval.
           + _flow("WEO_2026_APR_VINTAGE", vintage="2026-04-14")
           # Modified before its edition date -> edition date governs (next midnight).
           + _flow("WEO_2026_JUL_VINTAGE", vintage="2026-07-20", updated="2026-07-01T10:00:00Z")
           + _flow("WEO_2026_DEC_VINTAGE", vintage="2026-12-14", updated="2026-10-01T00:00:00Z")
           + _flow("WEO_2019_APR_VINTAGE", vintage="2019-04-09", updated="2019-04-01T00:00:00Z")
           + _flow("WEO_2025_OCT_VINTAGE", agency="IMF.OTHER", vintage="2025-10-14", updated="2025-10-01T00:00:00Z")
           + b'</s:Dataflows></message:Structures></message:Structure>')


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


def test_vintages_released_only_with_official_metadata_backed_date():
    r = m.CONNECTORS["imf_weo"]({}, fetch_fn=Fake("imf_weo")).fetch(NOW)
    assert r.status == SourceStatus.PASS
    assert r.discovered_ids["vintages"] == ["WEO_2025_OCT_VINTAGE", "WEO_2026_JUL_VINTAGE"]
    assert r.discovered_ids["vintages_without_verified_release"] == ["WEO_2026_APR_VINTAGE"]
    by = {o.attrs["edition"]: o for o in r.observations}
    oct_ = by["WEO_2025_OCT_VINTAGE"]
    assert oct_.available_at == datetime(2026, 3, 31, tzinfo=timezone.utc)  # NOT 2025-11-01 / 2025-10-14
    assert oct_.attrs["availability_basis"] == "IMF_LAST_UPDATED_AT"
    assert oct_.attrs["official_edition_date"] == "2025-10-14"
    assert by["WEO_2026_JUL_VINTAGE"].available_at == datetime(2026, 7, 21, tzinfo=timezone.utc)
    assert "WEO_2026_APR_VINTAGE" not in by  # unverifiable release -> not ingested
    assert by["current"].attrs["availability_basis"] == "RETRIEVAL_TIME"
    # Nothing from a vintage is visible before its metadata-backed date.
    assert not [o for o in available_as_of(r.observations, datetime(2026, 3, 30, 23, tzinfo=timezone.utc))]
    asof = available_as_of(r.observations, datetime(2026, 5, 1, tzinfo=timezone.utc))
    assert asof and {o.attrs["edition"] for o in asof} == {"WEO_2025_OCT_VINTAGE"}
    assert all(o.available_at <= o.retrieved_at for o in r.observations)
    assert all(o.attrs["strict_backtest_eligible"] for o in r.observations)


def test_verified_availability_rule():
    d = lambda *a: datetime(*a, tzinfo=timezone.utc)
    assert m.verified_vintage_availability(d(2025, 10, 14), d(2026, 3, 30, 18)) == d(2026, 3, 31)
    assert m.verified_vintage_availability(d(2026, 7, 20), d(2026, 7, 1)) == d(2026, 7, 21)
    assert m.verified_vintage_availability(None, d(2026, 2, 9, 21)) == d(2026, 2, 10)
    assert m.verified_vintage_availability(d(2025, 10, 14), None) is None
    assert m.vintage_month("MFS_MA_2026_JUNE_VINTAGE", "MFS_MA") == (2026, 6)
    assert m.vintage_month("WEO_FOO_VINTAGE", "WEO") is None
    assert m.vintage_month("WEOX_2025_OCT_VINTAGE", "WEO") is None


def test_future_dated_last_updated_is_not_trusted():
    cat = CATALOG.replace(b"2026-03-30T18:36:09.484648Z", b"2027-01-01T00:00:00Z")
    vs = m.discover_vintages(res(cat, ctype=m.STRUCTURE_ACCEPT), "IMF.RES", "WEO", NOW)
    oct_ = next(v for v in vs if v["flow_id"] == "WEO_2025_OCT_VINTAGE")
    assert oct_["last_updated"] is None


def test_research_gate_excludes_legacy_heuristic_vintage_rows(tmp_path):
    from modules import imf_macro_research as research
    r = m.CONNECTORS["imf_weo"]({}, fetch_fn=Fake("imf_weo")).fetch(NOW)
    legacy = next(o for o in r.observations if o.attrs["edition"] == "WEO_2025_OCT_VINTAGE")
    import dataclasses
    old_attrs = {k: v for k, v in legacy.attrs.items()
                 if k not in ("availability_basis", "strict_backtest_eligible")}
    old = dataclasses.replace(legacy, attrs=old_attrs, available_at=datetime(2025, 11, 1, tzinfo=timezone.utc))
    assert not research.strict_backtest_eligible(old)
    assert research.strict_backtest_eligible(legacy)
    cur = next(o for o in r.observations if o.attrs["edition"] == "current")
    assert research.strict_backtest_eligible(dataclasses.replace(cur, attrs={"edition": "current"}))


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

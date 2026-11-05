"""IMF statistical research feeds via the official IMF SDMX 3.0 API (api.imf.org).

One reader for all IMF macro research feeds (WEO, FM, GDD, FSIC, MFS_MA, CCI,
FAS). Every agency, dataflow, dimension order and code below was verified
against the live IMF registry on 2026-11-05 (dataflow list, DSD with
references=all, and bounded data queries from a GitHub-hosted runner):

* Data path: /data/dataflow/{agency}/{flow}/+/{key}; "+" = latest stable
  version. "*" as version is ambiguous (multiple DSDs) and returns HTTP 500.
* Time window: startPeriod/endPeriod work. c[TIME_PERIOD] is NOT honoured
  (ignored or empty result) and must not be used.
* Accept "application/vnd.sdmx.data+json;version=2.0.0" (with patch level).
* Country codes are ISO-3 (USA, DEU, ...); multi-value key parts use "+".
* The IMF DataMapper (www.imf.org/external/datamapper) answers 403 (Akamai)
  to CI runners; it is NOT used and never bypassed.

PIT contract: the "current" dataflow is stamped available_at = retrieval time
(conservative, prospective only). Official IMF *vintage* dataflows
(e.g. WEO_2025_OCT_VINTAGE) are frozen editions; they are stamped with the
first day of the month after the edition month (always later than the real
publication date), which gives genuine, conservative historical vintages for
revision research without reconstructing anything.

Research only: no model input, no scoring, no production promotion.
"""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

from modules.external import http
from modules.external.pit import AvailabilityPrecision, Observation
from modules.external.sources.base import Connector, ConnectorResult, RawRecord, SourceStatus

BASE = "https://api.imf.org/external/sdmx/3.0"
DATA_ACCEPT = "application/vnd.sdmx.data+json;version=2.0.0"
STRUCTURE_ACCEPT = "application/vnd.sdmx.structure+xml;version=3.0.0"
MAX_DATA_BYTES = 3_000_000
MAX_CATALOG_BYTES = 4_000_000
MAX_SERIES = 250
MAX_OBSERVATIONS = 6_000
MAX_VINTAGES = 4
PARSER_VERSION = "imf_sdmx3_json2_v2"

MAJOR = ("USA", "DEU", "JPN", "GBR", "FRA", "CHN")
_MONTHS = {m: i for i, m in enumerate(
    ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"), 1)}

# Official IMF codes (verified 2026-11-05). None = wildcard, any returned code
# is accepted but recorded in the series id.
SPECS: dict[str, dict] = {
    "imf_weo": {
        "agency": "IMF.RES", "flow": "WEO",
        "dimensions": ("COUNTRY", "INDICATOR", "FREQUENCY"),
        "key": {"COUNTRY": MAJOR, "INDICATOR": ("NGDP_RPCH", "PCPIPCH", "LUR",
                                                 "GGXWDG_NGDP", "BCA_NGDPD"),
                "FREQUENCY": ("A",)},
        "units": {"NGDP_RPCH": "percent_change", "PCPIPCH": "percent_change",
                  "LUR": "percent", "GGXWDG_NGDP": "percent_of_gdp",
                  "BCA_NGDPD": "percent_of_gdp"},
        "years_back": 5, "years_forward": 5, "projections": True,
        "vintage_prefix": "WEO",
    },
    "imf_fiscal_monitor": {
        "agency": "IMF.FAD", "flow": "FM",
        "dimensions": ("COUNTRY", "INDICATOR", "FREQUENCY"),
        "key": {"COUNTRY": MAJOR, "INDICATOR": ("GNLB_S13_POGDP_PT", "GPB_S13_POGDP_PT",
                                                 "G63G_S13_POGDP_PT"),
                "FREQUENCY": ("A",)},
        "units": {"GNLB_S13_POGDP_PT": "percent_of_gdp", "GPB_S13_POGDP_PT": "percent_of_gdp",
                  "G63G_S13_POGDP_PT": "percent_of_gdp"},
        "years_back": 5, "years_forward": 5, "projections": True,
        "vintage_prefix": "FM",
    },
    "imf_gdd": {
        "agency": "IMF.FAD", "flow": "GDD",
        "dimensions": ("COUNTRY", "INDICATOR", "FREQUENCY"),
        "key": {"COUNTRY": MAJOR, "INDICATOR": ("FL_PS_POGDP_PT", "FL_S14_POGDP_PT",
                                                 "FL_S11_POGDP_PT", "FL_S13_POGDP_PT"),
                "FREQUENCY": ("A",)},
        "units": {"FL_PS_POGDP_PT": "percent_of_gdp", "FL_S14_POGDP_PT": "percent_of_gdp",
                  "FL_S11_POGDP_PT": "percent_of_gdp", "FL_S13_POGDP_PT": "percent_of_gdp"},
        "years_back": 10, "years_forward": 0,
    },
    "imf_fsi": {
        "agency": "IMF.STA", "flow": "FSIC",
        "dimensions": ("COUNTRY", "SECTOR", "INDICATOR", "FREQUENCY"),
        "key": {"COUNTRY": MAJOR, "SECTOR": ("S12CFSI",),
                "INDICATOR": ("AQ12_CFSI_PT", "FSI688_CFSI_PT", "AQ14_CFSI_PT", "ROE_CFSI_PT"),
                "FREQUENCY": ("Q",)},
        "units": {"AQ12_CFSI_PT": "percent", "FSI688_CFSI_PT": "percent",
                  "AQ14_CFSI_PT": "percent", "ROE_CFSI_PT": "percent"},
        "years_back": 4, "years_forward": 0,
    },
    "imf_mfs": {
        "agency": "IMF.STA", "flow": "MFS_MA",
        "dimensions": ("COUNTRY", "INDICATOR", "UNIT", "FREQUENCY"),
        # Euro-area members report no national broad money -> not requested.
        "key": {"COUNTRY": ("USA", "JPN", "GBR", "CHN", "CAN", "AUS", "CHE", "IND"),
                "INDICATOR": ("BM_MAI",), "UNIT": ("XDC", "USD"), "FREQUENCY": ("M",)},
        "units": {"BM_MAI": "currency_level"},
        "unit_dimension": "UNIT",
        "years_back": 3, "years_forward": 0,
        "vintage_prefix": "MFS_MA",
    },
    "imf_climate": {
        "agency": "IMF.STA", "flow": "CCI",
        "dimensions": ("COUNTRY", "INDICATOR",
                       "GEOGRAPHIC_MARINE_ZONES_AND_CLIMATIC_CLASSIFICATIONS", "FREQUENCY"),
        "key": {"COUNTRY": MAJOR, "INDICATOR": ("AMGST_dC", "NDT_NUM"),
                "GEOGRAPHIC_MARINE_ZONES_AND_CLIMATIC_CLASSIFICATIONS": None,
                "FREQUENCY": ("A",)},
        "units": {"AMGST_dC": "degrees_celsius_vs_1951_1980", "NDT_NUM": "count"},
        "years_back": 10, "years_forward": 0,
    },
    "imf_fas": {
        "agency": "IMF.STA", "flow": "FAS",
        "dimensions": ("COUNTRY", "INDICATOR", "SECTOR", "COUNTERPARTY_SECTOR", "SEX",
                       "TYPE_OF_TRANSFORMATION", "FREQUENCY"),
        "key": {"COUNTRY": MAJOR, "INDICATOR": ("FA26", "FA18", "OUTL_A", "OUTD_L"),
                "SECTOR": None, "COUNTERPARTY_SECTOR": None, "SEX": None,
                "TYPE_OF_TRANSFORMATION": ("PHTADLT_NUM", "POGDP"),
                "FREQUENCY": ("A",)},
        "units": {"FA26": "per_100k_adults_or_pct_gdp", "FA18": "per_100k_adults_or_pct_gdp",
                  "OUTL_A": "per_100k_adults_or_pct_gdp", "OUTD_L": "per_100k_adults_or_pct_gdp"},
        "unit_dimension": "TYPE_OF_TRANSFORMATION",
        "years_back": 8, "years_forward": 0,
    },
}


class SDMX3Error(ValueError):
    pass


def _record(source_id, dataset, response):
    return RawRecord(source_id=source_id, dataset=dataset, url=response.url,
                     fingerprint=response.fingerprint,
                     retrieved_at=response.retrieved_at, status_code=response.status,
                     content_type=response.content_type,
                     content_hash=response.content_hash, bytes=response.bytes)


def build_key(spec: dict) -> str:
    parts = []
    for dim in spec["dimensions"]:
        codes = spec["key"].get(dim)
        parts.append("*" if codes is None else "+".join(codes))
    return ".".join(parts)


def _json(response):
    if not 0 < response.bytes <= MAX_DATA_BYTES:
        raise SDMX3Error("EMPTY_OR_OVERSIZED_RESPONSE")
    if response.content.lstrip()[:1] != b"{" or "html" in (response.content_type or "").lower():
        raise SDMX3Error("NOT_SDMX_JSON")
    try:
        parsed = response.json()
    except (ValueError, UnicodeDecodeError) as exc:
        raise SDMX3Error("INVALID_JSON") from exc
    if (not isinstance(parsed, dict) or
            not isinstance(parsed.get("data"), dict) or
            not isinstance(parsed["data"].get("dataSets"), list) or
            not isinstance(parsed["data"].get("structures"), list)):
        raise SDMX3Error("NOT_SDMX_DATA_MESSAGE")
    return parsed["data"]


def parse_period(value) -> datetime:
    raw = value
    if isinstance(value, dict):
        raw = value.get("id") or value.get("value") or value.get("start")
    if not isinstance(raw, str):
        raise SDMX3Error("UNRESOLVED_TIME_PERIOD")
    m = re.fullmatch(r"(\d{4})", raw)
    if m:
        return datetime(int(raw), 1, 1, tzinfo=timezone.utc)
    m = re.fullmatch(r"(\d{4})-Q([1-4])", raw)
    if m:
        return datetime(int(m[1]), 1 + (int(m[2]) - 1) * 3, 1, tzinfo=timezone.utc)
    m = re.fullmatch(r"(\d{4})-M?(0[1-9]|1[0-2])", raw)
    if m:
        return datetime(int(m[1]), int(m[2]), 1, tzinfo=timezone.utc)
    raise SDMX3Error(f"UNSUPPORTED_TIME_PERIOD:{raw[:20]}")


def vintage_available_at(flow_id: str, prefix: str) -> datetime | None:
    """WEO_2025_OCT_VINTAGE -> 2025-11-01T00:00Z (conservative: after the edition month)."""
    m = re.fullmatch(rf"{re.escape(prefix)}_(\d{{4}})_([A-Z]+)_VINTAGE", flow_id)
    if not m or m[2][:3] not in _MONTHS:
        return None
    year, month = int(m[1]), _MONTHS[m[2][:3]]
    return datetime(year + (month == 12), month % 12 + 1, 1, tzinfo=timezone.utc)


def parse_data(response, *, source_id: str, spec: dict, dataset: str,
               available_at: datetime, vintage: str | None = None) -> list[Observation]:
    data = _json(response)
    if len(data["structures"]) != 1 or len(data["dataSets"]) != 1:
        raise SDMX3Error("UNSUPPORTED_MULTIPLE_STRUCTURES_OR_DATASETS")
    dims = data["structures"][0].get("dimensions") or {}
    series_dims, time_dims = dims.get("series"), dims.get("observation")
    expected = list(spec["dimensions"])
    if (not isinstance(series_dims, list)
            or [d.get("id") for d in series_dims] != expected
            or not isinstance(time_dims, list) or len(time_dims) != 1
            or time_dims[0].get("id") != "TIME_PERIOD"):
        raise SDMX3Error("SDMX3_DIMENSION_ORDER_OR_SCHEMA_CHANGED")
    times = time_dims[0].get("values")
    if not isinstance(times, list) or len(times) > 2_000:
        raise SDMX3Error("TIME_VALUES_SCHEMA_CHANGED")
    ds = data["dataSets"][0]
    if not isinstance(ds, dict):
        raise SDMX3Error("DATASET_SCHEMA_CHANGED")
    rows = ds.get("series")
    if rows is None:
        rows = {}
    if not isinstance(rows, dict) or len(rows) > MAX_SERIES:
        raise SDMX3Error("SDMX3_INVALID_OR_OVERSIZED_SERIES")
    retrieved = response.retrieved_at
    lo = retrieved.year - spec["years_back"]
    hi = retrieved.year + spec["years_forward"]
    unit_dim = spec.get("unit_dimension")
    obs, used = [], set()
    for skey, item in rows.items():
        idx = skey.split(":")
        if len(idx) != len(expected) or not all(i.isdecimal() for i in idx):
            raise SDMX3Error("SDMX3_INVALID_SERIES_KEY")
        codes = {}
        for d, i in zip(series_dims, idx):
            values = d.get("values")
            if not isinstance(values, list) or int(i) >= len(values):
                raise SDMX3Error("SERIES_CODE_INDEX_OUT_OF_RANGE")
            code = values[int(i)]
            if not isinstance(code, dict) or not isinstance(code.get("id"), str):
                raise SDMX3Error("SERIES_CODE_NOT_VERIFIABLE")
            allowed = spec["key"].get(d["id"])
            if allowed is not None and code["id"] not in allowed:
                raise SDMX3Error(f"UNREQUESTED_CODE:{d['id']}={code['id'][:30]}")
            codes[d["id"]] = code["id"]
        if not isinstance(item, dict) or not isinstance(item.get("observations", {}), dict):
            raise SDMX3Error("MISSING_SDMX3_OBSERVATIONS")
        country, indicator = codes["COUNTRY"], codes["INDICATOR"]
        extra = [codes[d] for d in expected if d not in ("COUNTRY", "INDICATOR", "FREQUENCY")]
        series_id = ":".join([spec["agency"], spec["flow"], indicator, *extra, codes["FREQUENCY"]])
        metric = "_".join(["imf", spec["flow"].lower(), indicator.lower(),
                           *(e.lower() for e in extra if e not in ("_T", "_Z", "NA"))])
        unit = spec["units"][indicator]
        if unit_dim:
            unit = f"{unit}:{codes[unit_dim]}"
        for t_idx, cell in (item.get("observations") or {}).items():
            if not isinstance(t_idx, str) or not t_idx.isdecimal() or int(t_idx) >= len(times):
                raise SDMX3Error("INVALID_OBSERVATION_TIME_INDEX")
            if not isinstance(cell, list) or not cell:
                raise SDMX3Error("INVALID_OBSERVATION_VALUE")
            value = cell[0]
            if value is None:
                continue  # official missing value never becomes zero
            if isinstance(value, bool):
                raise SDMX3Error("BOOLEAN_OBSERVATION")
            try:
                number = float(value)
            except (ValueError, TypeError) as exc:
                raise SDMX3Error("NON_NUMERIC_OBSERVATION") from exc
            if not math.isfinite(number):
                raise SDMX3Error("NON_FINITE_OBSERVATION")
            when = parse_period(times[int(t_idx)])
            if when.year < lo or when.year > hi:
                continue
            identity = (series_id, country, when)
            if identity in used:
                raise SDMX3Error("DUPLICATE_OBSERVATION")
            used.add(identity)
            projection = bool(spec.get("projections")) and when.year >= available_at.year
            obs.append(Observation(
                source_id=source_id, dataset=spec["flow"], series_id=series_id,
                entity_id=country, metric=metric, value=number, unit=unit,
                observation_time=when, available_at=available_at, retrieved_at=retrieved,
                vintage_time=available_at,
                # Projections are forecasts issued with the edition: valid for
                # their period, never "observed" before it (PIT + DQ contract).
                forecast_issue_time=available_at if projection else None,
                forecast_valid_time=when if projection else None,
                availability_precision=AvailabilityPrecision.CONSERVATIVE_DATE,
                parser_version=PARSER_VERSION, payload_hash=response.content_hash,
                attrs={"publisher": "IMF", "agency": spec["agency"], "dataflow": dataset,
                       "indicator": indicator, "dimensions": codes,
                       "transport": "IMF_SDMX3_JSON2", "edition": vintage or "current",
                       "is_projection": projection,
                       "historical_pit_verified": vintage is not None,
                       "research_only": True},
            ))
            if len(obs) > MAX_OBSERVATIONS:
                raise SDMX3Error("TOO_MANY_OBSERVATIONS")
    return obs


def discover_vintages(catalog_response, agency: str, prefix: str, now: datetime) -> list[tuple[str, datetime]]:
    """Official frozen editions of a dataflow from the IMF dataflow catalogue."""
    if not 0 < catalog_response.bytes <= MAX_CATALOG_BYTES:
        raise SDMX3Error("CATALOG_EMPTY_OR_OVERSIZED")
    head = catalog_response.content[:4096].upper()
    if b"<!DOCTYPE" in head or b"<!ENTITY" in head:
        raise SDMX3Error("XML_DTD_DISALLOWED")
    try:
        root = ET.fromstring(catalog_response.content)
    except ET.ParseError as exc:
        raise SDMX3Error("CATALOG_XML_PARSE_ERROR") from exc
    found = {}
    for e in root.iter():
        if e.tag.rsplit("}", 1)[-1] != "Dataflow" or e.get("agencyID") != agency:
            continue
        when = vintage_available_at(e.get("id") or "", prefix)
        if when is not None and when <= now and when.year >= now.year - 3:
            found[e.get("id")] = when
    return sorted(found.items(), key=lambda kv: kv[1])[-MAX_VINTAGES:]


class IMFSDMX3Connector(Connector):
    parser_version = PARSER_VERSION

    def __init__(self, source_cfg=None, *, fetch_fn=None):
        super().__init__(source_cfg)
        self._fetch = fetch_fn or http.fetch

    def _data(self, spec, flow, now):
        return self._fetch(
            f'{BASE}/data/dataflow/{spec["agency"]}/{flow}/+/{build_key(spec)}',
            params={"startPeriod": str(now.year - spec["years_back"]),
                    "endPeriod": str(now.year + spec["years_forward"])},
            headers={"Accept": DATA_ACCEPT}, retries=3, timeout=60)

    def fetch(self, now: datetime) -> ConnectorResult:
        sid = self.source_id
        spec = SPECS[sid]
        raw: list[RawRecord] = []
        try:
            if now.tzinfo is None:
                raise SDMX3Error("NAIVE_NOW")
            res = self._data(spec, spec["flow"], now)
            raw.append(_record(sid, spec["flow"], res))
            obs = parse_data(res, source_id=sid, spec=spec, dataset=spec["flow"],
                             available_at=res.retrieved_at)
            missing = [c for c in spec["key"]["INDICATOR"]
                       if not any(o.attrs["indicator"] == c for o in obs)]
            if missing:
                raise SDMX3Error(f"REQUIRED_INDICATOR_ABSENT:{','.join(missing)}")
        except (http.FetchError, SDMX3Error, ValueError, TypeError, KeyError) as exc:
            status = SourceStatus.FAIL if isinstance(exc, http.FetchError) else SourceStatus.SCHEMA_CHANGED
            return ConnectorResult(source_id=sid, status=status, raw=raw,
                                   message=f"IMF SDMX3 {spec['flow']}: {type(exc).__name__}: {exc}")

        notes, vintages = [], []
        if spec.get("vintage_prefix"):
            # Additive only: an unavailable vintage never discards current data.
            try:
                cat = self._fetch(f"{BASE}/structure/dataflow",
                                  headers={"Accept": STRUCTURE_ACCEPT}, retries=2, timeout=90)
                raw.append(_record(sid, "dataflow_catalog", cat))
                for flow_id, when in discover_vintages(cat, spec["agency"], spec["vintage_prefix"], now):
                    vres = self._data(spec, flow_id, now)
                    raw.append(_record(sid, flow_id, vres))
                    vobs = parse_data(vres, source_id=sid, spec=spec, dataset=flow_id,
                                      available_at=min(when, vres.retrieved_at), vintage=flow_id)
                    obs.extend(vobs)
                    vintages.append(flow_id)
            except (http.FetchError, SDMX3Error, ValueError, TypeError, KeyError) as exc:
                notes.append(f"vintages incomplete ({type(exc).__name__}: {str(exc)[:120]})")

        countries = sorted({o.entity_id for o in obs})
        status = SourceStatus.PASS if len(countries) >= 2 and not notes else SourceStatus.WARN
        return ConnectorResult(
            source_id=sid, status=status, raw=raw, observations=obs,
            latest_observation_time=max(o.observation_time for o in obs),
            message=(f"{spec['flow']}: {len(obs)} IMF SDMX3 research observations, "
                     f"{len(countries)} countries, {len(vintages)} official vintages"
                     + (f"; {'; '.join(notes)}" if notes else "")),
            discovered_ids={"agency": spec["agency"], "flow": spec["flow"],
                            "key": build_key(spec), "countries": countries,
                            "vintages": vintages, "research_only": True},
        )


def _connector(source_id: str):
    return type(f"IMF_{source_id}_Connector", (IMFSDMX3Connector,), {"source_id": source_id})


CONNECTORS = {sid: _connector(sid) for sid in SPECS}

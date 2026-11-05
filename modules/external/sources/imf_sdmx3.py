"""Official IMF SDMX 3.0 research reader with embedded JSON 2.0 schema validation.

Used in place of DataMapper's Akamai-blocked JSON endpoint for independently
verified IMF.RES WEO and IMF.FAD FM. Strictly prospective PIT; no live trading.
The data response itself carries ordered dimensions, so no positional guesses.
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timezone

from modules.external import http
from modules.external.pit import AvailabilityPrecision, Observation
from modules.external.sources.base import ConnectorResult, RawRecord, SourceStatus

BASE = "https://api.imf.org/external/sdmx/3.0"
MAX_BYTES = 2_500_000
PARSER_VERSION = "imf_sdmx3_json2_v1"
# These are official code+unit mappings, never inferred from a different dataset.
FLOW_SPECS = {
    "imf_weo": {
        "agency": "IMF.RES", "flow": "WEO",
        "indicators": {"NGDP_RPCH": "percent", "PCPIPCH": "percent", "LUR": "percent"},
        "countries": ("DEU", "USA"), "frequency": "A",
    },
    "imf_fiscal_monitor": {
        "agency": "IMF.FAD", "flow": "FM",
        "indicators": {"GGXWDG_NGDP": "percent_of_gdp",
                       "GGXONLB_NGDP": "percent_of_gdp"},
        "countries": ("DEU", "USA"), "frequency": "A",
    },
    "imf_fsi": {
        "agency": "IMF.STA", "flow": "FSIC",
        "indicators": {"NPLR_PT": "percent"},
        "countries": ("DE", "US"), "frequency": "Q",
        "dimensions": ("COUNTRY", "SECTOR", "INDICATOR", "FREQUENCY"),
        "sector": "_T",
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


def _json(response):
    if not 0 < response.bytes <= MAX_BYTES:
        raise SDMX3Error("EMPTY_OR_OVERSIZED_RESPONSE")
    if response.content.lstrip()[:1] != b"{" or "html" in response.content_type.lower():
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


def _period(value):
    if isinstance(value, str):
        raw = value
    elif isinstance(value, dict):
        raw = value.get("id") or value.get("name") or value.get("start") or value.get("value")
        if isinstance(raw, dict):
            raw = raw.get("en") or next(iter(raw.values()), None)
    else:
        raw = None
    if not isinstance(raw, str):
        raise SDMX3Error("UNRESOLVED_TIME_PERIOD")
    if re.fullmatch(r"\d{4}", raw):
        return datetime(int(raw), 1, 1, tzinfo=timezone.utc)
    if re.fullmatch(r"\d{4}-Q[1-4]", raw):
        return datetime(int(raw[:4]), 1 + (int(raw[-1]) - 1) * 3, 1, tzinfo=timezone.utc)
    if re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", raw):
        return datetime(int(raw[:4]), int(raw[-2:]), 1, tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError as exc:
        raise SDMX3Error("UNSUPPORTED_TIME_PERIOD") from exc


def parse_data(response, *, source_id: str, country: str,
               indicator: str, spec: dict):
    data = _json(response)
    if len(data["structures"]) != 1 or len(data["dataSets"]) != 1:
        raise SDMX3Error("UNSUPPORTED_MULTIPLE_STRUCTURES_OR_DATASETS")
    structure = data["structures"][0]
    dim = structure.get("dimensions") or {}
    series_dims = dim.get("series")
    time_dims = dim.get("observation")
    expected_dimensions = list(spec.get("dimensions") or ("COUNTRY", "INDICATOR", "FREQUENCY"))
    if (not isinstance(series_dims, list)
            or [d.get("id") for d in series_dims] != expected_dimensions
            or not isinstance(time_dims, list) or len(time_dims) != 1
            or time_dims[0].get("id") != "TIME_PERIOD"):
        raise SDMX3Error("SDMX3_DIMENSION_ORDER_OR_SCHEMA_CHANGED")
    times = time_dims[0].get("values")
    if not isinstance(times, list) or len(times) > 250:
        raise SDMX3Error("TIME_VALUES_SCHEMA_CHANGED")
    ds = data["dataSets"][0]
    if not isinstance(ds, dict) or ds.get("structure", 0) != 0:
        raise SDMX3Error("DATASET_STRUCTURE_REFERENCE_MISMATCH")
    rows = ds.get("series")
    if not isinstance(rows, dict) or len(rows) > 12:
        raise SDMX3Error("SDMX3_MISSING_OR_OVERSIZED_SERIES")
    obs = []
    used = set()
    for skey, item in rows.items():
        indexes = skey.split(":")
        if len(indexes) != len(expected_dimensions) or not all(i.isdecimal() for i in indexes):
            raise SDMX3Error("SDMX3_INVALID_SERIES_KEY")
        keys = []
        for d, idx in zip(series_dims, indexes):
            values = d.get("values")
            if not isinstance(values, list) or int(idx) >= len(values):
                raise SDMX3Error("SERIES_CODE_INDEX_OUT_OF_RANGE")
            code = values[int(idx)]
            if not isinstance(code, dict) or not isinstance(code.get("id"), str):
                raise SDMX3Error("SERIES_CODE_NOT_VERIFIABLE")
            keys.append(code["id"])
        desired = {"COUNTRY": country, "INDICATOR": indicator, "FREQUENCY": spec["frequency"]}
        if "SECTOR" in expected_dimensions:
            desired["SECTOR"] = spec["sector"]
        if keys != [desired[k] for k in expected_dimensions]:
            raise SDMX3Error("UNREQUESTED_COUNTRY_INDICATOR_OR_FREQUENCY")
        if not isinstance(item, dict) or not isinstance(item.get("observations"), dict):
            raise SDMX3Error("MISSING_SDMX3_OBSERVATIONS")
        for time_index, cell in item["observations"].items():
            if not isinstance(time_index, str) or not time_index.isdecimal() or int(time_index) >= len(times):
                raise SDMX3Error("INVALID_OBSERVATION_TIME_INDEX")
            if not isinstance(cell, list) or len(cell) < 1:
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
            when = _period(times[int(time_index)])
            if when.year < 2015 or when.year > response.retrieved_at.year + 5:
                continue  # explicit bounded research years
            identity = (country, indicator, when)
            if identity in used:
                raise SDMX3Error("DUPLICATE_OBSERVATION")
            used.add(identity)
            is_future = source_id == "imf_weo" and when.year > response.retrieved_at.year
            obs.append(Observation(
                source_id=source_id, dataset=spec["flow"],
                series_id=f'{spec["agency"]}:{spec["flow"]}:{indicator}:{spec["frequency"]}',
                entity_id=country, metric=f'imf_{spec["flow"].lower()}_{indicator.lower()}',
                value=number, unit=spec["indicators"][indicator],
                observation_time=when, available_at=response.retrieved_at,
                retrieved_at=response.retrieved_at,
                availability_precision=AvailabilityPrecision.CONSERVATIVE_DATE,
                parser_version=PARSER_VERSION, payload_hash=response.content_hash,
                forecast_issue_time=response.retrieved_at if is_future else None,
                forecast_valid_time=when if is_future else None,
                attrs={"publisher": "IMF", "agency": spec["agency"],
                       "dataflow": spec["flow"], "indicator": indicator,
                       "transport": "IMF_SDMX3_JSON2", "historical_pit_verified": False,
                       "projection_cutoff_verified": False, "research_only": True},
            ))
            if len(obs) > 50:
                raise SDMX3Error("TOO_MANY_OBSERVATIONS")
    if not obs:
        raise SDMX3Error("NO_OBSERVATIONS_FOR_REQUESTED_SERIES")
    return obs


def fetch_research(source_id, now, *, fetch_fn=None):
    if source_id not in FLOW_SPECS:
        raise SDMX3Error("UNAPPROVED_DATAFLOW")
    spec = FLOW_SPECS[source_id]
    fetch = fetch_fn or http.fetch
    raw, obs = [], []
    try:
        if now.tzinfo is None:
            raise SDMX3Error("NAIVE_NOW")
        for code in spec["indicators"]:
            for country in spec["countries"]:
                dimension_values = {"COUNTRY": country, "INDICATOR": code,
                                    "FREQUENCY": spec["frequency"], "SECTOR": spec.get("sector")}
                dims = spec.get("dimensions") or ("COUNTRY", "INDICATOR", "FREQUENCY")
                key = ".".join(dimension_values[d] for d in dims)
                res = fetch(
                    f'{BASE}/data/dataflow/{spec["agency"]}/{spec["flow"]}/+/{key}',
                    params={"startPeriod": str(max(2015, now.year - 5)),
                            "endPeriod": str(now.year + (5 if source_id == "imf_weo" else 0)),
                            "format": "jsondata"},
                    headers={"Accept": "application/vnd.sdmx.data+json;version=2.0"},
                    retries=2, timeout=35,
                )
                raw.append(_record(source_id, spec["flow"], res))
                obs.extend(parse_data(res, source_id=source_id,
                                      country=country, indicator=code, spec=spec))
        return ConnectorResult(
            source_id=source_id, status=SourceStatus.PASS, raw=raw,
            observations=obs, latest_observation_time=max(o.observation_time for o in obs),
            message=f'{spec["flow"]}: {len(obs)} verified SDMX-3.0 prospective research observations',
            discovered_ids={"agency": spec["agency"], "flow": spec["flow"],
                            "indicators": list(spec["indicators"]),
                            "countries": list(spec["countries"]),
                            "historical_pit_verified": False, "research_only": True},
        )
    except (http.FetchError, SDMX3Error, ValueError, TypeError, KeyError) as exc:
        status = SourceStatus.FAIL if isinstance(exc, http.FetchError) else SourceStatus.SCHEMA_CHANGED
        return ConnectorResult(
            source_id=source_id, status=status, raw=raw,
            message=f"IMF SDMX3: {type(exc).__name__}: {exc}",
        )

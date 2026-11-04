"""IMF DataMapper: bounded statistical research snapshots, never historic PIT backfill.

Four registered research feeds: WEO, GDD, Fiscal Monitor, AIPI. Official
indicator catalog is checked on each successful fetch; dataset/indicator IDs,
country IDs, units and values are NOT inferred from arbitrary CSV columns.
Only a small, preselected indicator/country universe is downloaded. IMF data
copyright: non-commercial private research with attribution, no LLM training.
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timezone

from modules.external import http
from modules.external.pit import AvailabilityPrecision, Observation
from modules.external.sources.base import Connector, ConnectorResult, RawRecord, SourceStatus

BASE = "https://www.imf.org/external/datamapper/api/v1"
COUNTRIES = ("USA", "DEU", "CHN", "JPN", "GBR", "FRA")
MAX_RESPONSE = 3_000_000
MIN_YEAR = 2015
PARSER_VERSION = "1"


def _raw(sid: str, name: str, result) -> RawRecord:
    return RawRecord(source_id=sid, dataset=name, url=result.url,
                     fingerprint=result.fingerprint, retrieved_at=result.retrieved_at,
                     status_code=result.status, content_type=result.content_type,
                     content_hash=result.content_hash, bytes=result.bytes)


class IMFDataMapperError(ValueError):
    pass


def _json(result) -> dict:
    if result.bytes <= 0 or result.bytes > MAX_RESPONSE:
        raise IMFDataMapperError("EMPTY_OR_OVERSIZED_RESPONSE")
    if (result.content.lstrip()[:1] != b"{"
            or "html" in result.content_type.lower()):
        raise IMFDataMapperError("NOT_JSON")
    try:
        obj = result.json()
    except (ValueError, UnicodeDecodeError) as e:
        raise IMFDataMapperError("INVALID_JSON") from e
    if not isinstance(obj, dict):
        raise IMFDataMapperError("JSON_NOT_OBJECT")
    return obj


def parse_indicator(result, code: str, dataset: str, cfg: dict,
                    retrieved: datetime) -> list[Observation]:
    obj = _json(result)
    values = obj.get("values")
    if not isinstance(values, dict) or not isinstance(values.get(code), dict):
        raise IMFDataMapperError(f"INDICATOR_RESPONSE_MISMATCH:{code}")
    series = values[code]
    countries = set(cfg.get("countries") or COUNTRIES)
    if not countries or not countries.issubset(set(COUNTRIES)):
        raise IMFDataMapperError("UNAPPROVED_COUNTRY_SELECTION")
    wanted_years = set(range(int(cfg.get("start_year", MIN_YEAR)),
                              min(retrieved.year + 6, 2040)))
    if len(wanted_years) > 30:
        raise IMFDataMapperError("TOO_MANY_YEARS")
    unit = cfg["_validated_units"][code]
    obs = []
    for country in sorted(countries):
        points = series.get(country)
        if points is None:
            continue
        if not isinstance(points, dict):
            raise IMFDataMapperError("NON_DICT_COUNTRY_SERIES")
        for raw_year, value in points.items():
            if not re.fullmatch(r"\d{4}", str(raw_year)):
                raise IMFDataMapperError("INVALID_YEAR_KEY")
            year = int(raw_year)
            if year not in wanted_years or value is None:
                continue
            if isinstance(value, bool):
                raise IMFDataMapperError("BOOLEAN_VALUE")
            try:
                n = float(value)
            except (TypeError, ValueError) as e:
                raise IMFDataMapperError("NON_NUMERIC_VALUE") from e
            if not math.isfinite(n):
                raise IMFDataMapperError("NON_FINITE_VALUE")
            valid = datetime(year, 1, 1, tzinfo=timezone.utc)
            is_future_projection = (dataset == "WEO" and year > retrieved.year)
            obs.append(Observation(
                source_id=cfg["source_id"], dataset=dataset, series_id=f"{dataset}:{code}",
                entity_id=country, metric=f"imf_{dataset.lower()}_{code.lower()}",
                value=n, unit=unit, observation_time=valid,
                available_at=retrieved, retrieved_at=retrieved,
                availability_precision=AvailabilityPrecision.CONSERVATIVE_DATE,
                parser_version=PARSER_VERSION, payload_hash=result.content_hash,
                forecast_issue_time=retrieved if is_future_projection else None,
                forecast_valid_time=valid if is_future_projection else None,
                attrs={"indicator_code": code, "country": country,
                       "transport": "IMF_DataMapper_v1", "dataset": dataset,
                       "projection": is_future_projection, "historical_pit_verified": False}),
            ))
    if len(obs) > 500:
        raise IMFDataMapperError("OBSERVATION_LIMIT")
    return obs


class _IMFDataMapperConnector(Connector):
    parser_version = PARSER_VERSION
    DATASET = ""
    INDICATORS: tuple[str, ...] = ()

    def __init__(self, source_cfg=None, *, fetch_fn=None):
        super().__init__(source_cfg)
        self._fetch = fetch_fn or http.fetch

    def fetch(self, now: datetime) -> ConnectorResult:
        sid, raw, observations = self.source_id, [], []
        try:
            if now.tzinfo is None:
                raise IMFDataMapperError("NAIVE_NOW")
            # Verify catalog identity and source dataset *before* importing data.
            catalog = self._fetch(f"{BASE}/indicators", timeout=40, retries=2)
            raw.append(_raw(sid, "indicator_catalog", catalog))
            metadata = _json(catalog).get("indicators")
            if not isinstance(metadata, dict):
                raise IMFDataMapperError("CATALOG_SCHEMA_CHANGED")
            units = {}
            for code in self.INDICATORS:
                m = metadata.get(code)
                if not isinstance(m, dict) or m.get("dataset") != self.DATASET:
                    raise IMFDataMapperError(f"CATALOG_DATASET_MISMATCH:{code}")
                if not m.get("label") or not m.get("unit"):
                    raise IMFDataMapperError(f"CATALOG_UNIT_OR_LABEL_MISSING:{code}")
                units[code] = str(m["unit"])
            cfg = {**self.cfg, "source_id": sid, "_validated_units": units}
            for code in self.INDICATORS:
                result = self._fetch(f"{BASE}/{code}", timeout=40, retries=2)
                raw.append(_raw(sid, self.DATASET, result))
                selected = parse_indicator(result, code, self.DATASET,
                                           cfg, result.retrieved_at)
                if not selected:
                    raise IMFDataMapperError(f"NO_VALID_OBSERVATIONS_FOR_INDICATOR:{code}")
                observations.extend(selected)
            if not observations:
                raise IMFDataMapperError("NO_VERIFIED_OBSERVATIONS")
            latest = max(x.observation_time for x in observations)
            countries = sorted({x.entity_id for x in observations})
            status = SourceStatus.PASS if len(countries) >= 2 else SourceStatus.WARN
            return ConnectorResult(
                source_id=sid, status=status, raw=raw, observations=observations,
                latest_observation_time=latest, message=(
                    f"{self.DATASET}: {len(observations)} research-only PIT snapshots; "
                    f"{len(countries)} validated countries; original IMF releases NOT reconstructed"),
                discovered_ids={"flow": self.DATASET, "indicators": list(self.INDICATORS),
                                "countries": countries, "transport": "IMF_DataMapper_v1",
                                "historical_pit_verified": False},
            )
        except (http.FetchError, IMFDataMapperError, ValueError, TypeError, KeyError) as e:
            kind = (SourceStatus.SCHEMA_CHANGED if isinstance(e, (IMFDataMapperError, TypeError, KeyError))
                    else SourceStatus.FAIL)
            return ConnectorResult(source_id=sid, status=kind, raw=raw,
                                   message=f"IMF DataMapper: {type(e).__name__}: {e}")


class IMFWEOConnector(_IMFDataMapperConnector):
    source_id = "imf_weo"
    DATASET = "WEO"
    INDICATORS = ("NGDP_RPCH", "PCPIPCH", "LUR")


class IMFGDDConnector(_IMFDataMapperConnector):
    source_id = "imf_gdd"
    DATASET = "GDD"
    INDICATORS = ("Privatedebt_all", "HH_ALL", "NFC_ALL")


class IMFFiscalMonitorConnector(_IMFDataMapperConnector):
    source_id = "imf_fiscal_monitor"
    DATASET = "FM"
    INDICATORS = ("GGXCNL_G01_GDP_PT", "G_XWDG_G01_GDP_PT")


class IMFAIPIConnector(_IMFDataMapperConnector):
    source_id = "imf_aipi"
    DATASET = "AIPI"
    INDICATORS = ("AI_PI", "DI", "HCLMP")


CONNECTORS = {c.source_id: c for c in (
    IMFWEOConnector, IMFGDDConnector, IMFFiscalMonitorConnector, IMFAIPIConnector
)}

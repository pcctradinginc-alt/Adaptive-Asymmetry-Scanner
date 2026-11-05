"""Four bounded official IMF SDMX-2.1 research feeds: FSI, MFS, FAS, Climate.

No guessed series ID: each selected country/indicator comes from an IMF
Dataflow's DSD/codelists resolved at runtime. Unknown dimensions/schema stop
ingestion. Historical values are ONLY available from first observed retrieval
(PIT, no retroactive vintage fabrication). No production learner integration.
"""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

from modules.external import http
from modules.external.pit import AvailabilityPrecision, Observation
from modules.external.sources.base import Connector, ConnectorResult, RawRecord, SourceStatus

BASE = "https://api.imf.org/external/sdmx/2.1"
MAX_METADATA_BYTES = 12_000_000
MAX_DATA_BYTES = 4_000_000
PARSER_VERSION = "1"
COUNTRY_CODES = (("USA", "US"), ("DEU", "DE"), ("CHN", "CN"), ("JPN", "JP"))
COUNTRY_LABELS = (("united states",), ("germany",), ("china",), ("japan",))


class IMFSDMXError(ValueError):
    pass


def _local(elem) -> str:
    return str(elem.tag).rsplit("}", 1)[-1]


def _raw(sid: str, dataset: str, r) -> RawRecord:
    return RawRecord(source_id=sid, dataset=dataset, url=r.url, fingerprint=r.fingerprint,
                     retrieved_at=r.retrieved_at, status_code=r.status,
                     content_type=r.content_type, content_hash=r.content_hash, bytes=r.bytes)


def _xml(res, *, limit: int) -> ET.Element:
    if not (0 < res.bytes <= limit) or res.content.lstrip()[:1] != b"<":
        raise IMFSDMXError("EMPTY_OR_OVERSIZED_OR_NON_XML")
    if b"<!DOCTYPE" in res.content[:8192].upper() or b"<!ENTITY" in res.content[:8192].upper():
        raise IMFSDMXError("XML_DTD_DISALLOWED")
    try:
        root = ET.fromstring(res.content)
    except ET.ParseError as e:
        raise IMFSDMXError("XML_PARSE_ERROR") from e
    if _local(root) not in ("Structure", "StructureSpecificData", "GenericData"):
        raise IMFSDMXError("UNEXPECTED_SDMX_ROOT")
    return root


def _children(e, name: str):
    return [x for x in e if _local(x) == name]


def _first(e, name: str):
    return next((x for x in e.iter() if _local(x) == name), None)


def _label(e) -> str:
    return next(((x.text or "").strip() for x in e if _local(x) == "Name"
                 and x.get("{http://www.w3.org/XML/1998/namespace}lang", "en").startswith("en")),
                next(((x.text or "").strip() for x in e if _local(x) == "Name"), ""))


def _catalog(root: ET.Element, flow: str):
    dflow = next((x for x in root.iter() if _local(x) == "Dataflow" and x.get("id") == flow), None)
    if dflow is None or not str(dflow.get("agencyID", "")).startswith("IMF"):
        raise IMFSDMXError("DATAFLOW_ID_OR_PUBLISHER_MISMATCH")
    structure_ref = _first(dflow, "Ref")
    expected_dsd_id = structure_ref.get("id") if structure_ref is not None else None
    if not expected_dsd_id:
        raise IMFSDMXError("DATAFLOW_DSD_LINK_MISSING")
    ds = next((x for x in root.iter() if _local(x) == "DataStructure"
               and x.get("id") == expected_dsd_id), None)
    if ds is None or not str(ds.get("agencyID", "")).startswith("IMF"):
        raise IMFSDMXError("DATASTRUCTURE_NOT_VERIFIED")
    dl = _first(ds, "DimensionList")
    if dl is None:
        raise IMFSDMXError("MISSING_DIMENSIONLIST")
    dimensions = []
    for e in dl:
        kind = _local(e)
        if kind == "TimeDimension":
            continue
        if kind not in ("Dimension", "MeasureDimension"):
            raise IMFSDMXError(f"UNSUPPORTED_DIMENSION_TYPE:{kind}")
        enum = _first(e, "Enumeration")
        ref = _first(enum, "Ref") if enum is not None else None
        if ref is None:
            # IMF STA DSDs (MFS_MA, CCI, FAS) can define COUNTRY via
            # ConceptIdentity alone, with the codelist in the official
            # concept's CoreRepresentation. Resolve that exact chain:
            # dimension -> concept scheme/version -> concept -> enumeration.
            identity = _first(e, "ConceptIdentity")
            identity_ref = _first(identity, "Ref") if identity is not None else None
            # A DSD dimension's concept can use a different official ID
            # (e.g. COUNTRY / REF_AREA). Verify the *referenced* concept
            # and its publisher/version; never require conceptID == dimensionID.
            if identity_ref is None or not identity_ref.get("id"):
                raise IMFSDMXError(f"DIMENSION_CONCEPT_IDENTITY_MISSING:{e.get('id')}")
            parent_id = identity_ref.get("maintainableParentID")
            parent_version = identity_ref.get("maintainableParentVersion")
            owner = identity_ref.get("agencyID")
            if not (parent_id and parent_version and owner and owner.startswith("IMF")):
                raise IMFSDMXError(f"DIMENSION_CONCEPT_PARENT_INVALID:{e.get('id')}")
            schemes = [x for x in root.iter() if _local(x) == "ConceptScheme"
                       and x.get("id") == parent_id and x.get("agencyID") == owner
                       and x.get("version") == parent_version]
            if len(schemes) != 1:
                raise IMFSDMXError(f"CONCEPT_SCHEME_NOT_RESOLVED:{owner}:{parent_id}:{parent_version}")
            concepts = [x for x in schemes[0] if _local(x) == "Concept"
                        and x.get("id") == identity_ref.get("id")]
            if len(concepts) != 1:
                raise IMFSDMXError(f"DIMENSION_CONCEPT_NOT_RESOLVED:{e.get('id')}")
            core = _first(concepts[0], "CoreRepresentation")
            enum = _first(core, "Enumeration") if core is not None else None
            ref = _first(enum, "Ref") if enum is not None else None
            if ref is None:
                raise IMFSDMXError(f"CONCEPT_CORE_ENUMERATION_MISSING:{e.get('id')}")
        if not ref.get("id"):
            raise IMFSDMXError(f"DIMENSION_CODELIST_REF_MISSING:{e.get('id')}")
        dimensions.append((str(e.get("id", "")).upper(), ref.get("id")))
    if not dimensions or len(dimensions) > 12 or len({x[0] for x in dimensions}) != len(dimensions):
        raise IMFSDMXError("INVALID_DIMENSIONS")
    lists = {}
    for e in root.iter():
        if _local(e) != "Codelist" or not e.get("id"):
            continue
        codes = {}
        for code in _children(e, "Code"):
            cid = code.get("id")
            if cid:
                codes[cid] = _label(code)
        lists[e.get("id")] = codes
    return dimensions, lists, {"flow": flow, "agency": dflow.get("agencyID"),
                               "dsd": ds.get("id"), "dimensions": [d for d, _ in dimensions]}


def _select(dimensions, lists, flow: str, phrases: tuple[str, ...]):
    """Resolve a bounded query exclusively from the publisher's codelists.

    Some IMF datasets call the indicator dimension something other than
    INDICATOR.  An unfamiliar dimension is accepted as the indicator axis
    only if exactly one dimension's *official* codelist contains an
    unambiguous approved label.  No positional or code-name guessing.
    """
    indicator_names = {"INDICATOR", "INDICATOR_ID", "SERIES",
                       "INDICATOR_CODE", "SERIES_ID", "ITEM"}
    country_names = {"COUNTRY", "REF_AREA", "GEO", "GEOGRAPHY"}
    freq_names = {"FREQUENCY", "FREQ"}

    def approved(codes):
        found = []
        for phrase in phrases:
            matches = [code for code, label in codes.items()
                       if phrase.casefold() in label.casefold()]
            if len(matches) == 1:
                found.append(matches[0])
        return list(dict.fromkeys(found))[:3]

    # Identify a nonstandard indicator dimension only from positive,
    # uniquely attributable metadata evidence.  Multiple candidates
    # are an ambiguous schema and must fail closed.
    candidates = [(dim, approved(lists.get(list_id) or {}))
                  for dim, list_id in dimensions
                  if dim not in country_names | freq_names]
    identified = [(dim, matches) for dim, matches in candidates if matches]
    if len(identified) != 1:
        raise IMFSDMXError(
            f"INDICATOR_AXIS_NOT_UNAMBIGUOUS:flow={flow}:"
            f"candidates={[(d, len(m)) for d, m in identified]!r}"
        )
    indicator_dim, indicator_codes = identified[0]
    vals, selected = {}, {}
    found_country = False
    unrecognized = 0
    for dim, list_id in dimensions:
        codes = lists.get(list_id) or {}
        if not codes:
            raise IMFSDMXError(
                f"MISSING_OFFICIAL_CODELIST:{dim}:ref={list_id!r}"
            )
        if dim in country_names:
            country = []
            for aliases, labels in zip(COUNTRY_CODES, COUNTRY_LABELS):
                matches = [c for c in aliases if c in codes and any(
                    label in codes[c].casefold() for label in labels)]
                if matches:
                    country.append(matches[0])
            if len(country) < 2:
                raise IMFSDMXError("INSUFFICIENT_VERIFIED_COUNTRIES")
            vals[dim] = "+".join(country)
            selected["countries"] = country
            found_country = True
        elif dim == indicator_dim:
            vals[dim] = "+".join(indicator_codes)
            selected["indicator_codes"] = indicator_codes
            selected["indicator_labels"] = {c: codes[c] for c in indicator_codes}
            selected["indicator_dimension"] = dim
        elif dim in freq_names:
            choices = (("Q", "A") if flow == "FSIC" else ("A", "Q", "M"))
            freq = next((f for f in choices if f in codes), None)
            if not freq:
                raise IMFSDMXError("UNSUPPORTED_FREQUENCY")
            vals[dim] = freq
            selected["frequency"] = freq
        else:
            unrecognized += 1
            vals[dim] = ""
    if not found_country or unrecognized > 2:
        raise IMFSDMXError("UNSUPPORTED_DATAFLOW_DIMENSIONS")
    key = ".".join(vals[d] for d, _ in dimensions)
    return key, selected


def _parse_period(raw: str) -> datetime:
    s = str(raw or "")
    if re.fullmatch(r"\d{4}", s):
        return datetime(int(s), 1, 1, tzinfo=timezone.utc)
    m = re.fullmatch(r"(\d{4})-?Q([1-4])", s)
    if m:
        return datetime(int(m[1]), 1 + (int(m[2]) - 1) * 3, 1, tzinfo=timezone.utc)
    m = re.fullmatch(r"(\d{4})-?M(0[1-9]|1[0-2])", s) or re.fullmatch(r"(\d{4})-(0[1-9]|1[0-2])", s)
    if m:
        return datetime(int(m[1]), int(m[2]), 1, tzinfo=timezone.utc)
    raise IMFSDMXError(f"UNSUPPORTED_TIME_PERIOD:{s[:16]}")


def _parse_data(root: ET.Element, source_id: str, flow: str, selected: dict, res):
    obs = []
    seen = set()
    countries = set(selected["countries"])
    indicators = set(selected["indicator_codes"])
    indicator_dim = selected["indicator_dimension"]
    for series in root.iter():
        if _local(series) != "Series":
            continue
        attrs = dict(series.attrib)
        if not attrs:
            # GenericData with nested SeriesKey requires a separate parser.
            raise IMFSDMXError("GENERIC_SERIES_SCHEMA_NOT_SUPPORTED")
        country = next((attrs[k] for k in ("COUNTRY", "REF_AREA", "GEO", "GEOGRAPHY") if k in attrs), "")
        indicator = attrs.get(indicator_dim, "")
        if country not in countries or indicator not in indicators:
            raise IMFSDMXError("DATA_QUERY_SCOPE_MISMATCH")
        for x in series:
            if _local(x) != "Obs":
                continue
            period = x.get("TIME_PERIOD") or x.get("TIME")
            value = x.get("OBS_VALUE")
            if period is None or value is None:
                raise IMFSDMXError("OBSERVATION_SCHEMA_CHANGED")
            when = _parse_period(period)
            if when > res.retrieved_at:
                raise IMFSDMXError("SDMX_FUTURE_PERIOD_NOT_APPROVED")
            try:
                num = float(value)
            except ValueError as e:
                raise IMFSDMXError("SDMX_NON_NUMERIC_VALUE") from e
            if not math.isfinite(num):
                raise IMFSDMXError("SDMX_NON_FINITE_VALUE")
            unique = (tuple(sorted(attrs.items())), period)
            if unique in seen:
                raise IMFSDMXError("DUPLICATE_SERIES_OBSERVATION")
            seen.add(unique)
            series_id = f"{flow}:" + "|".join(f"{k}={v}" for k, v in sorted(attrs.items()))
            obs.append(Observation(
                source_id=source_id, dataset=flow, series_id=series_id,
                entity_id=country, metric=f"imf_{flow.lower()}_{indicator.lower()}",
                value=num, unit=attrs.get("UNIT_MEASURE") or attrs.get("UNIT") or "imf_reported",
                observation_time=when, available_at=res.retrieved_at,
                retrieved_at=res.retrieved_at,
                availability_precision=AvailabilityPrecision.CONSERVATIVE_DATE,
                parser_version=PARSER_VERSION, payload_hash=res.content_hash,
                attrs={"indicator_code": indicator, "indicator_label":
                       selected["indicator_labels"].get(indicator),
                       "dimensions": attrs, "transport": "IMF_SDMX_2.1",
                       "historical_pit_verified": False},
            ))
            if len(obs) > 1500:
                raise IMFSDMXError("DATA_SERIES_LIMIT")
    if not obs:
        raise IMFSDMXError("NO_VALID_OBSERVATIONS")
    return obs


class _IMFSDMXConnector(Connector):
    parser_version = PARSER_VERSION
    FLOW = ""
    PHRASES: tuple[str, ...] = ()

    def __init__(self, source_cfg=None, *, fetch_fn=None):
        super().__init__(source_cfg)
        self._fetch = fetch_fn or http.fetch

    def fetch(self, now: datetime) -> ConnectorResult:
        # SDMX 2.1 FSIC metadata is 204 on the publisher's retired gateway.
        # On the 3.0 gateway, IMF.STA/FSIC and its verified 4-dimension DSD
        # exist. Mocks keep exercising the independent legacy parser.
        if self.source_id == "imf_fsi" and self._fetch is http.fetch:
            from modules.external.sources.imf_sdmx3 import fetch_research
            return fetch_research(self.source_id, now)
        sid, raw = self.source_id, []
        try:
            if now.tzinfo is None:
                raise IMFSDMXError("NAIVE_NOW")
            meta = self._fetch(f"{BASE}/dataflow/all/{self.FLOW}/latest",
                               params={"references": "all"}, timeout=45, retries=2)
            raw.append(_raw(sid, "official_schema", meta))
            dims, codelists, discovered = _catalog(_xml(meta, limit=MAX_METADATA_BYTES), self.FLOW)
            key, selected = _select(dims, codelists, self.FLOW, self.PHRASES)
            data = self._fetch(f"{BASE}/data/{self.FLOW}/{key}",
                               params={"lastNObservations": 4},
                               headers={"Accept": "application/xml"},
                               timeout=45, retries=2)
            raw.append(_raw(sid, self.FLOW, data))
            observations = _parse_data(_xml(data, limit=MAX_DATA_BYTES), sid,
                                       self.FLOW, selected, data)
            latest = max(x.observation_time for x in observations)
            countries = {x.entity_id for x in observations}
            return ConnectorResult(source_id=sid,
                status=SourceStatus.PASS if len(countries) >= 2 else SourceStatus.WARN,
                raw=raw, observations=observations, latest_observation_time=latest,
                message=f"{self.FLOW}: {len(observations)} verified, prospectively PIT-only research observations",
                discovered_ids={**discovered, **selected,
                                "historical_pit_verified": False, "research_only": True})
        except (http.FetchError, IMFSDMXError, ET.ParseError, ValueError, TypeError, KeyError) as e:
            status = SourceStatus.FAIL if isinstance(e, http.FetchError) else SourceStatus.SCHEMA_CHANGED
            return ConnectorResult(source_id=sid, status=status, raw=raw,
                                   message=f"IMF SDMX {self.FLOW}: {type(e).__name__}: {e}")


class IMFFSIConnector(_IMFSDMXConnector):
    source_id = "imf_fsi"
    FLOW = "FSIC"
    PHRASES = ("nonperforming loans to total gross loans",
               "regulatory capital to risk-weighted assets")


class IMFMFSConnector(_IMFSDMXConnector):
    source_id = "imf_mfs"
    FLOW = "MFS_MA"
    PHRASES = ("broad money", "claims on private sector")


class IMFFASConnector(_IMFSDMXConnector):
    source_id = "imf_fas"
    FLOW = "FAS"
    PHRASES = ("automated teller machines per 100,000 adults",
               "commercial bank branches per 100,000 adults")


class IMFClimateConnector(_IMFSDMXConnector):
    source_id = "imf_climate"
    FLOW = "CCI"
    PHRASES = ("carbon dioxide emissions", "fossil fuel subsidies", "carbon intensity")


CONNECTORS = {c.source_id: c for c in (
    IMFFSIConnector, IMFMFSConnector, IMFFASConnector, IMFClimateConnector
)}

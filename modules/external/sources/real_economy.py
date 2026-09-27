"""
modules/external/sources/real_economy.py – Konnektoren für die Datenfamilie
"real_economy" (Umfrage- vs. Hard-Data-Signale: Wirtschaftsstimmung/
Industrievertrauen vs. Industrieproduktion), damit
hard_data_vs_survey_divergence (siehe modules/external/context.py) aus
echten Beobachtungen statt aus einem Platzhalter berechnet wird.

Grundregeln (siehe modules/external/sources/base.py):
  - Nur offizielle, maschinenlesbare Endpunkte (modules.external.http.fetch).
  - Serien-/Dataset-Codes werden zur LAUFZEIT über offizielle Katalog-/Such-
    endpunkte entdeckt (Eurostat TOC) und in ConnectorResult.discovered_ids
    abgelegt. Wo Laufzeit-Discovery nicht möglich ist, steht die erwartete
    ID mit dem Kommentar "verify in preflight" in der Config; passt die
    Antwort nicht zum erwarteten Schema, liefert der Konnektor
    SCHEMA_CHANGED/FAIL statt eines stillschweigenden Fallbacks.
  - Die konkreten Eurostat-Dimensions-CODES für "welcher indic/nace_r2-Wert
    ist die gesuchte Umfrage-/Produktionsserie" werden NIE fest codiert oder
    geraten -- sie werden aus den vom Datensatz selbst mitgelieferten
    Dimension-LABELS (dimension.<dim>.category.label, offizieller Eurostat-
    Text) per Substring-Suche ermittelt (siehe _select_metric_by_label).
    Nur der harmonisierte s_adj-Code (SA/SCA — Eurostat-weit einheitlich
    dokumentiert, nicht datensatzspezifisch geraten) wird als Query-Filter
    vorkonfiguriert.
  - Fehler landen NIE als Exception in der Pipeline, sondern im
    ConnectorResult (status FAIL/AUTH_MISSING/SCHEMA_CHANGED/...).

Zeit-Parsing: Eurostat liefert monatliche Perioden in mindestens zwei
gesehenen Schreibweisen ("2026-08" und "2026M08" bzw. "2026-M08") --
_parse_monthly_period() akzeptiert alle drei, NIE eine erfundene vierte
Variante stillschweigend.

JSON-Stat-2.0-Parsing: Die generische Dimensions-Indizierung (Stride-
Berechnung, Duplicate-Identity-Erkennung) folgt demselben Muster wie
EurostatRoadFreightConnector._parse_jsonstat (modules/external/sources/
road_freight.py) -- die Zeit-unabhängige Duplicate-Identity-Fehlerklasse
wird von dort IMPORTIERT statt dupliziert (siehe _DuplicateIdentityError-
Import unten); die Zeit-Parsing-Funktion ist hier bewusst eine eigene
(_parse_monthly_period), weil die dortige _parse_eurostat_time() nur
Jahres-/Quartals-/"YYYY-Mmm"-Labels kennt, nicht aber die für diese
monatlichen Datensätze beobachteten "YYYYMmm"/"YYYY-mm"-Varianten.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
from datetime import datetime, timezone

from modules.external import http
from modules.external.pit import AvailabilityPrecision, Observation, utc_now
from modules.external.sources.base import Connector, ConnectorResult, RawRecord, SourceStatus
from modules.external.sources.road_freight import _DuplicateIdentityError

log = logging.getLogger(__name__)

PARSER_VERSION = "1"


def _raw(source_id: str, dataset: str, res: http.FetchResult) -> RawRecord:
    return RawRecord(
        source_id=source_id, dataset=dataset, url=res.url, fingerprint=res.fingerprint,
        retrieved_at=res.retrieved_at, status_code=res.status, content_type=res.content_type,
        content_hash=res.content_hash, bytes=res.bytes,
    )


def _parse_monthly_period(label: str) -> datetime:
    """Akzeptiert "2026-08", "2026M08" und "2026-M08" (alle live bei
    Eurostat-Monatsdatensätzen beobachteten Schreibweisen) -- jede andere
    Schreibweise ist ein ValueError (SCHEMA_CHANGED beim Aufrufer), NIE
    stillschweigend geraten."""
    label = label.strip()
    m = re.match(r"^(\d{4})-?M(\d{2})$", label)
    if m:
        year, month = int(m.group(1)), int(m.group(2))
    else:
        m = re.match(r"^(\d{4})-(\d{2})$", label)
        if not m:
            raise ValueError(f"Unbekanntes monatliches Eurostat-Zeitformat: {label!r}")
        year, month = int(m.group(1)), int(m.group(2))
    if not (1 <= month <= 12):
        raise ValueError(f"Ungültiger Monat in Eurostat-Zeitformat: {label!r}")
    return datetime(year, month, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Gemeinsame Eurostat-TOC-Discovery + generisches JSON-stat-2.0-Parsing
# (lokal hier gehalten statt in road_freight.py verändert, siehe Docstring
# oben zur Reuse-Begründung: die Discovery-/Parsing-ORCHESTRIERUNG ist pro
# Datensatzfamilie an self.cfg gebunden, nur die Zeit-/Fehler-Bausteine
# werden importiert).
# ---------------------------------------------------------------------------

def _discover_dataset_code(toc_text: str, search_terms: list[str], code_re: "re.Pattern",
                            expected_code: str | None) -> tuple[str | None, dict]:
    rows = list(csv.reader(io.StringIO(toc_text), delimiter="\t"))
    if not rows or len(rows) < 2:
        return None, {}
    header = [h.strip().lower() for h in rows[0]]
    try:
        title_idx = header.index("title")
        code_idx = header.index("code")
    except ValueError:
        return None, {}
    type_idx = header.index("type") if "type" in header else None
    matches = []
    for row in rows[1:]:
        needed = max(title_idx, code_idx, type_idx or 0)
        if len(row) <= needed:
            continue
        title, code = row[title_idx].strip(), row[code_idx].strip()
        entry_type = row[type_idx].strip().lower() if type_idx is not None else "dataset"
        if entry_type != "dataset":
            continue
        if not code_re.match(code):
            continue
        if any(term.lower() in title.lower() for term in search_terms):
            matches.append((code, title))
    if not matches:
        return None, {}
    matches.sort(key=lambda m: m[0])
    chosen = matches[0][0]
    if expected_code and any(c == expected_code for c, _ in matches):
        chosen = expected_code
    return chosen, {"candidates": dict(matches), "chosen": chosen}


def _parse_eurostat_jsonstat_generic(data: dict, code: str, source_id: str, dataset: str,
                                      retrieved_at: datetime, metric_fn, unit_fn=None,
                                      time_parser=_parse_monthly_period):
    """Generisches JSON-stat-2.0-Parsing über ALLE Dimensionen (nicht nur
    geo/time) -- siehe EurostatRoadFreightConnector._parse_jsonstat für das
    identische Grundmuster (Stride-Berechnung, Duplicate-Identity-Schutz).

    `metric_fn(dim_parts, dim_label_maps) -> str | None`: liefert den
    Metric-Namen für eine Dimensions-Kombination, ODER None, wenn diese
    Kombination NICHT übernommen werden soll (z.B. falscher indic-/nace_r2-
    Code laut offiziellem Label -- nie ein Code-Raten, siehe Modul-Docstring).

    Rückgabe: (observations, latest_obs_time, release_time, parse_failures,
    diagnostics). observations ist None bei einem nicht mal minimal
    auswertbaren Schema (SCHEMA_CHANGED beim Aufrufer)."""
    dims = (data.get("dimension") or {})
    ids = data.get("id") or []
    sizes = data.get("size") or []
    values = data.get("value")
    diagnostics: dict = {}
    if not ids or not sizes or values is None or "geo" not in dims or "time" not in dims:
        return None, None, None, 0, diagnostics

    release_time = None
    updated_raw = data.get("updated") or (data.get("extension") or {}).get("updated")
    if updated_raw:
        try:
            release_time = datetime.fromisoformat(str(updated_raw).replace("Z", "+00:00"))
            if release_time.tzinfo is None:
                release_time = release_time.replace(tzinfo=timezone.utc)
        except ValueError:
            release_time = None

    def index_map(dim_name: str) -> dict[int, str]:
        cat = ((dims.get(dim_name) or {}).get("category") or {})
        index = cat.get("index") or {}
        if isinstance(index, dict):
            return {v: k for k, v in index.items()}
        return {i: str(v) for i, v in enumerate(index)}

    def label_map(dim_name: str) -> dict[str, str]:
        """code -> offizieller Label-Text. JSON-stat 2.0 erlaubt für
        category.label sowohl ein {code: label}-Dict als auch eine Liste
        (parallel zu category.index positioniert) -- beide Formen werden
        unterstützt, NIE nur die Dict-Form angenommen (siehe
        test_eurostat_industrial_production_duplicate_identity_raises_schema_changed,
        das die Listenform für category.index/label verwendet)."""
        cat = ((dims.get(dim_name) or {}).get("category") or {})
        idx = cat.get("index")
        lbl = cat.get("label")
        if isinstance(lbl, dict):
            return dict(lbl)
        if isinstance(lbl, list):
            if isinstance(idx, dict):
                inv = {pos: code for code, pos in idx.items()}
                codes = [inv.get(i, str(i)) for i in range(len(lbl))]
            elif isinstance(idx, list):
                codes = idx
            else:
                codes = list(range(len(lbl)))
            return {str(c): str(l) for c, l in zip(codes, lbl)}
        return {}

    pos = {name: i for i, name in enumerate(ids)}
    if "geo" not in pos or "time" not in pos:
        return None, None, None, 0, diagnostics

    geo_of = index_map("geo")
    time_of = index_map("time")
    other_dim_ids = sorted(d for d in ids if d not in ("geo", "time"))
    dim_maps = {d: index_map(d) for d in other_dim_ids}
    dim_label_maps = {d: label_map(d) for d in other_dim_ids}

    strides = [1] * len(ids)
    for i in range(len(ids) - 2, -1, -1):
        strides[i] = strides[i + 1] * sizes[i + 1]

    observations: list[Observation] = []
    latest: datetime | None = None
    parse_failures = 0
    seen_identities: dict[tuple, str] = {}
    raw_values = values if isinstance(values, dict) else {str(i): v for i, v in enumerate(values)}

    for key_str, val in raw_values.items():
        try:
            flat_idx = int(key_str)
        except (TypeError, ValueError):
            parse_failures += 1
            continue
        idxs = []
        remainder = flat_idx
        for stride in strides:
            idxs.append(remainder // stride)
            remainder = remainder % stride
        try:
            geo_label = geo_of[idxs[pos["geo"]]]
            time_label = time_of[idxs[pos["time"]]]
        except (IndexError, KeyError):
            parse_failures += 1
            continue
        try:
            dim_parts = {d: dim_maps[d][idxs[pos[d]]] for d in other_dim_ids}
        except (IndexError, KeyError):
            parse_failures += 1
            continue
        try:
            obs_time = time_parser(time_label)
        except ValueError:
            parse_failures += 1
            continue

        metric = metric_fn(dim_parts, dim_label_maps)
        if metric is None:
            continue  # bewusst nicht übernommene Dimensions-Kombination

        dimkey = "|".join(f"{d}={dim_parts[d]}" for d in sorted(dim_parts))
        series_id = f"{code}:{dimkey}" if dimkey else code
        unit_code = unit_fn(dim_parts) if unit_fn else dim_parts.get("unit")

        identity = (series_id, geo_label, metric, obs_time.isoformat())
        if identity in seen_identities:
            raise _DuplicateIdentityError(
                f"identity={identity!r} kollidiert bei value-Keys "
                f"{seen_identities[identity]!r} und {key_str!r} (Datensatz {code})")
        seen_identities[identity] = key_str

        avail_precision = (AvailabilityPrecision.EXACT_TIMESTAMP if release_time
                            else AvailabilityPrecision.CONSERVATIVE_DATE)
        available_at = release_time or retrieved_at
        observations.append(Observation(
            source_id=source_id, dataset=dataset, series_id=series_id,
            entity_id=geo_label, metric=metric,
            value=float(val) if val is not None else None, unit=unit_code or "unknown",
            observation_time=obs_time, available_at=available_at, retrieved_at=retrieved_at,
            availability_precision=avail_precision, parser_version=PARSER_VERSION,
            attrs=dict(dim_parts),
        ))
        if latest is None or obs_time > latest:
            latest = obs_time
    return observations, latest, release_time, parse_failures, diagnostics


def _select_metric_by_label(dim_parts: dict, dim_label_maps: dict, dim_name: str,
                             keyword_map: dict[str, tuple[str, ...]]) -> str | None:
    """Ermittelt den Metric-Namen für dim_parts[dim_name] über dessen
    OFFIZIELLES Label (dim_label_maps[dim_name][code]) statt über den Code
    selbst -- der Code wird NIE geraten/fest codiert (siehe Modul-Docstring).
    keyword_map: {metric_name: (erforderliche lowercase Substrings,...)}."""
    code = dim_parts.get(dim_name)
    if code is None:
        return None
    label = (dim_label_maps.get(dim_name, {}).get(code) or "").lower()
    if not label:
        return None
    for metric_name, keywords in keyword_map.items():
        if all(kw in label for kw in keywords):
            return metric_name
    return None


class _EurostatGenericMonthlyConnector(Connector):
    """Gemeinsame Fetch-/Discovery-/Parse-Orchestrierung für die beiden
    Eurostat-Monatsdatensätze dieser Familie (Sentiment, Industrieproduktion).
    Subklassen liefern nur die datensatzspezifischen Konstanten + _metric_fn."""

    TOC_URL = "https://ec.europa.eu/eurostat/api/dissemination/catalogue/toc/txt?lang=en"
    DATA_BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"

    DATASET_CODE_RE: "re.Pattern"
    SEARCH_TERMS: list[str]
    EXPECTED_DATASET_CODE: str
    COUNTRIES: list[str]
    DIMENSION_FILTERS: dict[str, list[str]]
    DATASET_NAME: str  # Observation.dataset

    def _metric_fn(self, dim_parts: dict, dim_label_maps: dict) -> str | None:
        raise NotImplementedError

    def _unit_fn(self, dim_parts: dict) -> str | None:
        return dim_parts.get("unit")

    def _discover(self, raw: list[RawRecord]) -> tuple[str | None, dict]:
        search_terms = self.cfg.get("search_terms", self.SEARCH_TERMS)
        expected_code = self.cfg.get("expected_dataset_code", self.EXPECTED_DATASET_CODE)
        try:
            res = http.fetch(self.TOC_URL)
        except http.FetchError as e:
            log.warning("Eurostat TOC nicht erreichbar (%s): %s", self.source_id, e)
            return None, {}
        raw.append(_raw(self.source_id, "toc", res))
        try:
            text = res.content.decode("utf-8")
        except UnicodeDecodeError:
            return None, {}
        code, disc = _discover_dataset_code(text, search_terms, self.DATASET_CODE_RE, expected_code)
        if code is None:
            return None, {}
        return code, {f"{self.source_id}_candidates": disc.get("candidates"),
                       f"{self.source_id}_chosen": disc.get("chosen")}

    def _fetch_dataset(self, code: str, raw: list[RawRecord]) -> tuple["http.FetchResult | None", str | None]:
        url = f"{self.DATA_BASE}/{code}"
        countries = self.cfg.get("countries") or self.COUNTRIES
        dimension_filters = self.cfg.get("dimension_filters") or self.DIMENSION_FILTERS
        params: dict = {"format": "JSON", "lang": "en", "geo": countries}
        for dim_id, codes in dimension_filters.items():
            params[dim_id] = codes
        try:
            res = http.fetch(url, params=params)
        except http.FetchError as e:
            if "400" in str(e) and len(params) > 3:
                try:
                    res = http.fetch(url, params={"format": "JSON", "lang": "en", "geo": countries})
                except http.FetchError as e2:
                    return None, f"{e} | Retry ohne Dimensionsfilter: {e2}"
            else:
                return None, str(e)
        raw.append(_raw(self.source_id, "jsonstat", res))
        return res, None

    def fetch(self, now: datetime) -> ConnectorResult:
        raw: list[RawRecord] = []
        discovered: dict = {}
        code, disc = self._discover(raw)
        discovered.update(disc)
        expected_code = self.cfg.get("expected_dataset_code", self.EXPECTED_DATASET_CODE)
        if code is None:
            code = expected_code
            if not code:
                return ConnectorResult(
                    source_id=self.source_id, status=SourceStatus.FAIL, raw=raw,
                    message="Eurostat-TOC-Discovery fehlgeschlagen und keine "
                            "expected_dataset_code in der Config.",
                    discovered_ids=discovered,
                )
            discovered["fallback_dataset_code_source"] = "config.expected_dataset_code (verify in preflight)"

        res, err = self._fetch_dataset(code, raw)
        if err is not None and "404" in err and expected_code and expected_code != code:
            discovered[f"{self.source_id}_discovered_code_404"] = code
            code = expected_code
            discovered[f"{self.source_id}_chosen"] = code
            res, err = self._fetch_dataset(code, raw)
        if err is not None:
            return ConnectorResult(
                source_id=self.source_id, status=SourceStatus.FAIL, raw=raw,
                message=f"Eurostat-Datenabruf fehlgeschlagen ({code}): {err}",
                discovered_ids=discovered,
            )
        try:
            data = res.json()
        except json.JSONDecodeError:
            return ConnectorResult(
                source_id=self.source_id, status=SourceStatus.SCHEMA_CHANGED, raw=raw,
                message="Eurostat-Antwort kein valides JSON (JSON-stat 2.0 erwartet).",
                discovered_ids=discovered,
            )

        dims_present = set((data.get("dimension") or {}).keys())
        configured_filters = self.cfg.get("dimension_filters") or self.DIMENSION_FILTERS
        missing_filter_dims = sorted(set(configured_filters) - dims_present)
        if missing_filter_dims:
            discovered[f"{self.source_id}_dimension_filters_not_in_dataset"] = missing_filter_dims

        try:
            observations, latest, release_time, parse_failures, parse_diag = _parse_eurostat_jsonstat_generic(
                data, code, self.source_id, self.DATASET_NAME, res.retrieved_at,
                metric_fn=self._metric_fn, unit_fn=self._unit_fn)
        except _DuplicateIdentityError as e:
            return ConnectorResult(
                source_id=self.source_id, status=SourceStatus.SCHEMA_CHANGED, raw=raw,
                message=f"Eurostat-JSON liefert mehrere Werte für dieselbe Beobachtungs-"
                        f"Identität (Dimension fehlt/wird nicht erfasst): {e}",
                discovered_ids=discovered,
            )
        discovered.update(parse_diag)
        if observations is None:
            return ConnectorResult(
                source_id=self.source_id, status=SourceStatus.SCHEMA_CHANGED, raw=raw,
                message="Eurostat-JSON entspricht nicht dem erwarteten JSON-stat-2.0-Schema "
                        "(dimension/value fehlen).",
                discovered_ids=discovered,
            )
        if not observations and parse_failures == 0:
            return ConnectorResult(
                source_id=self.source_id, status=SourceStatus.SCHEMA_CHANGED, raw=raw,
                message="Keine der Dimensions-Kombinationen entsprach der gesuchten Metrik "
                        "laut offiziellem Label (indic/nace_r2/s_adj) -- Labels in preflight prüfen.",
                discovered_ids=discovered,
            )
        status = SourceStatus.WARN if parse_failures else SourceStatus.PASS
        return ConnectorResult(
            source_id=self.source_id, status=status, observations=observations, raw=raw,
            message=f"{len(observations)} Beobachtungen aus Datensatz {code}",
            latest_observation_time=latest, latest_release_time=release_time,
            discovered_ids=discovered, parse_failures=parse_failures,
        )


# ---------------------------------------------------------------------------
# 1) eurostat_sentiment – Economic Sentiment Indicator + Industrial confidence
#    (Eurostat Business & Consumer Survey results, i.d.R. Datensatzfamilie
#    ei_bssi_m_r2 -- verify in preflight)
# ---------------------------------------------------------------------------

class EurostatSentimentConnector(_EurostatGenericMonthlyConnector):
    """Discovery über die offizielle Eurostat-TOC (Titelsuche "sentiment
    indicators"). expected_dataset_code "ei_bssi_m_r2" ist nur Notfall-
    Fallback, falls die TOC-Discovery selbst nicht auswertbar ist -- verify
    in preflight, da zur Erstellungszeit keine Live-Katalogabfrage möglich
    war.

    Die indic-Codes für "Economic sentiment indicator" (ESI) und "Industrial
    confidence indicator" werden NICHT fest codiert, sondern aus dem
    offiziellen Label-Text der indic-Dimension des jeweils abgerufenen
    Datensatzes ausgewählt (_select_metric_by_label) -- nur der s_adj-Filter
    (SA, saisonbereinigt) wird als Eurostat-weit harmonisierter Code
    vorkonfiguriert."""

    source_id = "eurostat_sentiment"
    parser_version = PARSER_VERSION
    DATASET_NAME = "sentiment"

    DATASET_CODE_RE = re.compile(r"^ei_bssi")
    SEARCH_TERMS = ["sentiment indicators"]
    EXPECTED_DATASET_CODE = "ei_bssi_m_r2"  # verify in preflight
    COUNTRIES = ["EU27_2020", "DE", "FR", "IT", "ES", "NL", "PL"]
    # s_adj=SA ist ein Eurostat-weit harmonisierter Code (nicht
    # datensatzspezifisch geraten) -- reduziert das Antwortvolumen auf
    # ausschließlich saisonbereinigte Reihen (Aufgabenstellung).
    DIMENSION_FILTERS = {"s_adj": ["SA"]}

    METRIC_LABEL_KEYWORDS = {
        "esi": ("economic sentiment indicator",),
        "industrial_confidence": ("industrial confidence indicator",),
    }

    def _metric_fn(self, dim_parts: dict, dim_label_maps: dict) -> str | None:
        s_adj = dim_parts.get("s_adj")
        if s_adj is not None and s_adj != "SA":
            return None
        return _select_metric_by_label(dim_parts, dim_label_maps, "indic", self.METRIC_LABEL_KEYWORDS)


# ---------------------------------------------------------------------------
# 2) eurostat_industrial_production – Volume index of production in industry,
#    total industry excluding construction, calendar & seasonally adjusted
#    (i.d.R. Datensatzfamilie sts_inpr_m -- verify in preflight)
# ---------------------------------------------------------------------------

class EurostatIndustrialProductionConnector(_EurostatGenericMonthlyConnector):
    """Discovery über die offizielle Eurostat-TOC (Titelsuche "Production in
    industry"). expected_dataset_code "sts_inpr_m" ist nur Notfall-Fallback
    -- verify in preflight.

    nace_r2 ("total industry excluding construction") und indic_bt
    ("production volume index") werden über den offiziellen Label-Text
    ausgewählt, NIE über geratene Codes. s_adj=SCA (kalender- UND
    saisonbereinigt) ist ein Eurostat-weit harmonisierter Code."""

    source_id = "eurostat_industrial_production"
    parser_version = PARSER_VERSION
    DATASET_NAME = "industrial_production"

    DATASET_CODE_RE = re.compile(r"^sts_inpr")
    SEARCH_TERMS = ["Production in industry"]
    EXPECTED_DATASET_CODE = "sts_inpr_m"  # verify in preflight
    COUNTRIES = ["EU27_2020", "DE", "FR", "IT", "ES", "NL", "PL"]
    # s_adj=SCA (calendar and seasonally adjusted data) ist ebenfalls ein
    # harmonisierter Eurostat-Code.
    # Live-Preflight 2026-09-27: nur mit s_adj antwortet Eurostat 413
    # (Antwort zu groß, alle NACE-Abteilungen x Einheiten). B-D = "Industry
    # (except construction)", I21 = Index 2021=100 -- beides offizielle
    # Eurostat-Codes; die Label-Prüfung unten bleibt die inhaltliche Kontrolle.
    DIMENSION_FILTERS = {"s_adj": ["SCA"], "nace_r2": ["B-D"], "unit": ["I21"]}

    def _metric_fn(self, dim_parts: dict, dim_label_maps: dict) -> str | None:
        s_adj = dim_parts.get("s_adj")
        if s_adj is not None and s_adj != "SCA":
            return None
        # nur Indexniveaus, nie Veränderungsraten (PCH_*) in dieselbe Metrik
        unit = dim_parts.get("unit")
        if unit is not None:
            unit_label = (dim_label_maps.get("unit", {}).get(unit) or unit).lower()
            if "index" not in unit_label and not unit.upper().startswith("I"):
                return None
        nace = dim_parts.get("nace_r2")
        indic_bt = dim_parts.get("indic_bt")
        if nace is None or indic_bt is None:
            return None
        nace_label = (dim_label_maps.get("nace_r2", {}).get(nace) or "").lower()
        indic_label = (dim_label_maps.get("indic_bt", {}).get(indic_bt) or "").lower()
        if "except construction" not in nace_label:
            return None
        if "production" not in indic_label:
            return None
        return "production_volume_index"


# ---------------------------------------------------------------------------
# 3) fred_us_macro – US-Pendant: UMCSENT (Umfrage) + INDPRO (Hard Data), NUR
#    mit FRED_API_KEY (ALFRED-Vintages); ohne Key KEIN HTTP-Call (AUTH_MISSING).
# ---------------------------------------------------------------------------

class FredUsMacroConnector(Connector):
    """ALFRED (FRED-Vintages-API) für zwei Serien:
      - UMCSENT: University of Michigan: Consumer Sentiment (Umfrage)
      - INDPRO:  Industrial Production: Total Index (Hard Data)
    Beide Series-IDs sind offizielle, stabile FRED-Kennungen (nicht
    datensatz-familienspezifisch geraten) -- dennoch "verify in preflight"
    dokumentiert, da hier keine Live-Verifikation möglich war.

    OHNE FRED_API_KEY wird laut Aufgabenstellung KEIN HTTP-Call ausgeführt
    (kein fredgraph.csv-Fallback wie bei bts_freight_tsi) -- sofort
    AUTH_MISSING."""

    source_id = "fred_us_macro"
    parser_version = PARSER_VERSION

    FRED_API_BASE = "https://api.stlouisfed.org/fred"

    SERIES = {
        "UMCSENT": {"metric": "us_umcsent", "dataset": "survey",
                    "search_text": "University of Michigan: Consumer Sentiment"},
        "INDPRO": {"metric": "us_indpro", "dataset": "hard_data",
                   "search_text": "Industrial Production: Total Index"},
    }  # verify in preflight

    def _search_series_id(self, api_key: str, expected_id: str, search_text: str,
                           raw: list[RawRecord]) -> str | None:
        url = f"{self.FRED_API_BASE}/series/search"
        params = {"search_text": search_text, "api_key": api_key, "file_type": "json"}
        try:
            res = http.fetch(url, params=params)
        except http.FetchError as e:
            log.warning("FRED series/search fehlgeschlagen (%s): %s", expected_id, e)
            return None
        raw.append(_raw(self.source_id, "series_search", res))
        try:
            data = res.json()
        except json.JSONDecodeError:
            return None
        wanted = search_text.lower()
        for s in data.get("seriess", []):
            if (s.get("title") or "").lower() == wanted:
                return s.get("id")
        return None

    def _fetch_series(self, series_key: str, api_key: str, raw: list[RawRecord],
                       discovered: dict) -> tuple[list[Observation], datetime | None, int, ConnectorResult | None]:
        spec = self.SERIES[series_key]
        expected_series_id = self.cfg.get("expected_series_ids", {}).get(series_key, series_key)
        url = f"{self.FRED_API_BASE}/series/observations"
        params = {
            "series_id": expected_series_id, "api_key": api_key, "file_type": "json",
            "realtime_start": "1776-07-04", "realtime_end": "9999-12-31",
        }
        try:
            res = http.fetch(url, params=params)
        except http.AuthError:
            return [], None, 0, ConnectorResult(
                source_id=self.source_id, status=SourceStatus.AUTH_MISSING, raw=raw,
                message="FRED_API_KEY abgelehnt.", discovered_ids=discovered,
            )
        except http.FetchError as e:
            return [], None, 0, ConnectorResult(
                source_id=self.source_id, status=SourceStatus.FAIL, raw=raw,
                message=f"FRED ALFRED-Abruf fehlgeschlagen ({series_key}): {e}",
                discovered_ids=discovered,
            )
        raw.append(_raw(self.source_id, f"alfred_{series_key.lower()}", res))
        try:
            data = res.json()
        except json.JSONDecodeError:
            return [], None, 0, ConnectorResult(
                source_id=self.source_id, status=SourceStatus.SCHEMA_CHANGED, raw=raw,
                message=f"FRED-Antwort ({series_key}) kein valides JSON.",
                discovered_ids=discovered,
            )

        series_id = expected_series_id
        if "error_message" in data:
            discovered[f"{series_key}_expected_series_id_rejected"] = series_id
            alt = self._search_series_id(api_key, series_id, spec["search_text"], raw)
            if not alt:
                return [], None, 0, ConnectorResult(
                    source_id=self.source_id, status=SourceStatus.SCHEMA_CHANGED, raw=raw,
                    message=f"FRED lehnt Series-ID '{series_id}' ab ({data.get('error_message')}) "
                            f"und series/search fand keinen Ersatz für {series_key}.",
                    discovered_ids=discovered,
                )
            discovered[f"{series_key}_series_id"] = alt
            series_id = alt
            params["series_id"] = alt
            try:
                res = http.fetch(url, params=params)
            except http.FetchError as e:
                return [], None, 0, ConnectorResult(
                    source_id=self.source_id, status=SourceStatus.FAIL, raw=raw,
                    message=f"FRED ALFRED-Abruf (entdeckte ID {alt}) fehlgeschlagen: {e}",
                    discovered_ids=discovered,
                )
            raw.append(_raw(self.source_id, f"alfred_{series_key.lower()}_retry", res))
            try:
                data = res.json()
            except json.JSONDecodeError:
                return [], None, 0, ConnectorResult(
                    source_id=self.source_id, status=SourceStatus.SCHEMA_CHANGED, raw=raw,
                    message=f"FRED-Antwort (Discovery-Serie {series_key}) kein valides JSON.",
                    discovered_ids=discovered,
                )
        else:
            discovered[f"{series_key}_series_id"] = series_id

        obs_raw = data.get("observations")
        if obs_raw is None:
            return [], None, 0, ConnectorResult(
                source_id=self.source_id, status=SourceStatus.SCHEMA_CHANGED, raw=raw,
                message=f"FRED-Antwort ({series_key}) ohne 'observations' -> Schema geändert.",
                discovered_ids=discovered,
            )

        observations: list[Observation] = []
        latest: datetime | None = None
        parse_failures = 0
        for o in obs_raw:
            try:
                obs_time = datetime.strptime(o["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                vintage = datetime.strptime(o["realtime_start"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except (KeyError, ValueError):
                parse_failures += 1
                continue
            raw_val = o.get("value", ".")
            value = None if raw_val in (".", "", None) else self._safe_float(raw_val)
            observations.append(Observation(
                source_id=self.source_id, dataset=spec["dataset"], series_id=series_id,
                entity_id="US", metric=spec["metric"], value=value, unit="index_points",
                observation_time=obs_time, available_at=vintage, retrieved_at=res.retrieved_at,
                availability_precision=AvailabilityPrecision.EXACT_DATE,
                vintage_time=vintage, parser_version=self.parser_version,
                attrs={"realtime_end": o.get("realtime_end")},
            ))
            if latest is None or obs_time > latest:
                latest = obs_time
        return observations, latest, parse_failures, None

    def fetch(self, now: datetime) -> ConnectorResult:
        api_key = os.environ.get("FRED_API_KEY")
        if not api_key:
            return ConnectorResult(
                source_id=self.source_id, status=SourceStatus.AUTH_MISSING,
                message="FRED_API_KEY nicht gesetzt -> kein API-Call ausgeführt "
                        "(kein fredgraph.csv-Fallback für real_economy).",
            )
        raw: list[RawRecord] = []
        discovered: dict = {}
        all_observations: list[Observation] = []
        latest: datetime | None = None
        total_parse_failures = 0

        for series_key in self.SERIES:
            obs, series_latest, parse_failures, err = self._fetch_series(series_key, api_key, raw, discovered)
            if err is not None:
                err.discovered_ids = discovered
                err.raw = raw
                return err
            all_observations.extend(obs)
            total_parse_failures += parse_failures
            if series_latest is not None and (latest is None or series_latest > latest):
                latest = series_latest

        if not all_observations and total_parse_failures == 0:
            return ConnectorResult(
                source_id=self.source_id, status=SourceStatus.SCHEMA_CHANGED, raw=raw,
                message="Keine auswertbaren Beobachtungen in ALFRED-Antworten (UMCSENT/INDPRO).",
                discovered_ids=discovered,
            )
        status = SourceStatus.WARN if total_parse_failures else SourceStatus.PASS
        return ConnectorResult(
            source_id=self.source_id, status=status, observations=all_observations, raw=raw,
            message=f"{len(all_observations)} ALFRED-Vintage-Beobachtungen (UMCSENT+INDPRO)",
            latest_observation_time=latest, discovered_ids=discovered,
            parse_failures=total_parse_failures,
        )

    @staticmethod
    def _safe_float(s) -> float | None:
        try:
            return float(s)
        except (TypeError, ValueError):
            return None


CONNECTORS: dict[str, type[Connector]] = {
    "eurostat_sentiment": EurostatSentimentConnector,
    "eurostat_industrial_production": EurostatIndustrialProductionConnector,
    "fred_us_macro": FredUsMacroConnector,
}

"""
modules/external/sources/road_freight.py – Road-Freight-Konnektoren.

Familie "road_freight": Lkw-/Straßengüterverkehr als Frühindikator für
Industrie-/Konsum-Aktivität (Truck-Toll-Fahrleistungsindex DE, Freight TSI US,
Straßengüterverkehr EU, LKW-Statistik JP).

Grundregeln (siehe modules/external/sources/base.py):
  - Nur offizielle, maschinenlesbare Endpunkte (modules.external.http.fetch).
  - Serien-/Tabellen-IDs werden zur LAUFZEIT über offizielle Katalog-/Such-
    endpunkte entdeckt und in ConnectorResult.discovered_ids abgelegt.
    Wo Laufzeit-Discovery nicht möglich ist, steht die erwartete ID mit dem
    Kommentar "verify in preflight" in der Config; passt die Antwort nicht
    zum erwarteten Schema, liefert der Konnektor SCHEMA_CHANGED/FAIL mit
    einer klaren Nachricht statt eines stillschweigenden Fallbacks.
  - Fehler landen NIE als Exception in der Pipeline, sondern im
    ConnectorResult (status FAIL/AUTH_MISSING/SCHEMA_CHANGED/...).
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone

from modules.external import http
from modules.external.pit import AvailabilityPrecision, Observation, utc_now
from modules.external.sources.base import Connector, ConnectorResult, RawRecord, SourceStatus

log = logging.getLogger(__name__)

PARSER_VERSION = "1"


def _raw(source_id: str, dataset: str, res: http.FetchResult) -> RawRecord:
    return RawRecord(
        source_id=source_id, dataset=dataset, url=res.url, fingerprint=res.fingerprint,
        retrieved_at=res.retrieved_at, status_code=res.status, content_type=res.content_type,
        content_hash=res.content_hash, bytes=res.bytes,
    )


# ---------------------------------------------------------------------------
# 1) destatis_truck_toll – Destatis/BALM Lkw-Maut-Fahrleistungsindex
# ---------------------------------------------------------------------------

class DestatisTruckTollConnector(Connector):
    """GENESIS-Online REST API 2020 (Destatis).

    Discovery: catalogue/find (Volltextsuche) nach "Lkw-Maut-Fahrleistungsindex"
    liefert die zuständige(n) Tabelle(n) der Familie 42191-*. Wird die
    Katalogsuche selbst blockiert (z.B. Netz/Auth), fällt der Konnektor auf
    die in der Config hinterlegte erwartete Tabellen-Kennung zurück
    ("expected_table_code", Kommentar "verify in preflight") – eine
    abweichende Antwortstruktur führt dann zu SCHEMA_CHANGED, nie zu einer
    stillschweigenden Umdeutung.
    """

    source_id = "destatis_truck_toll"
    parser_version = PARSER_VERSION

    BASE = "https://www-genesis.destatis.de/genesisWS/rest/2020"
    # Regionale Kürzel, die wir konsistent unterstützen wollen, falls die
    # Tabelle Bundesländer separat ausweist (siehe Aufgabenstellung).
    STATE_LABELS = {
        "Baden-Württemberg": "BW", "Bayern": "BY", "Nordrhein-Westfalen": "NW",
        "Niedersachsen": "NI", "Hamburg": "HH",
    }

    def _credentials(self) -> tuple[str, str]:
        user = os.environ.get("DESTATIS_USER") or self.cfg.get("gast_username", "GAST")
        pw = os.environ.get("DESTATIS_PASSWORD") or self.cfg.get("gast_password", "GAST")
        return user, pw

    def _find_table_code(self, now: datetime, raw: list[RawRecord]) -> tuple[str | None, dict]:
        """Katalogsuche (data/find bzw. catalogue/tables) nach dem offiziellen
        Titel des Fahrleistungsindex. Gibt (table_code, discovered_ids) zurück;
        table_code ist None, wenn die Suche selbst nicht auswertbar war."""
        user, pw = self._credentials()
        search_term = self.cfg.get("search_term", "Lkw-Maut-Fahrleistungsindex")
        url = f"{self.BASE}/catalogue/tables"
        params = {
            "username": user, "password": pw, "selection": "42191*",
            "searchcriterion": "Code", "language": "de", "pagelength": 50,
        }
        try:
            res = http.fetch(url, params=params)
        except http.AuthError as e:
            # Katalogsuche selbst ohne Zugang -> nicht hart scheitern; die
            # Verbindlichkeit liegt beim data/tablefile-Abruf danach (der bei
            # echtem Auth-Problem regulär AUTH_MISSING liefert).
            log.warning("destatis catalogue/tables Auth-Fehler: %s", e)
            return None, {}
        except http.FetchError as e:
            log.warning("destatis catalogue/tables nicht erreichbar: %s", e)
            return None, {}
        raw.append(_raw(self.source_id, "catalogue", res))
        try:
            data = res.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None, {}
        entries = data.get("List") or data.get("Tables") or []
        if not entries:
            return None, {}
        candidates = []
        for e in entries:
            code = e.get("Code") or e.get("Kennung") or ""
            title = (e.get("Content") or e.get("Title") or "").lower()
            if code.startswith("42191") and (
                "fahrleistung" in title or "maut" in title or search_term.lower() in title
                or not title
            ):
                candidates.append((code, e.get("Content") or e.get("Title") or ""))
        if not candidates:
            return None, {}
        candidates.sort(key=lambda c: c[0])
        chosen = candidates[0][0]
        discovered = {c[0]: c[1] for c in candidates}
        return chosen, {"destatis_table_codes": discovered, "destatis_chosen_table": chosen}

    def _fetch_tablefile(self, table_code: str, fmt: str, raw: list[RawRecord],
                          ) -> tuple["http.FetchResult | None", ConnectorResult | None]:
        """POST (application/x-www-form-urlencoded, Credentials als HTTP-
        Header -- aktuelles GENESIS-REST-2020-Verhalten) zuerst; nur bei
        einem Nicht-Auth-Fehler (z.B. Endpoint akzeptiert an dieser Stelle
        nur GET) Fallback auf GET mit Credentials als Query-Parameter
        (älteres/alternatives Verhalten). Ein echter Auth-Fehler wird NIE
        stillschweigend per GET erneut versucht."""
        user, pw = self._credentials()
        url = f"{self.BASE}/data/tablefile"
        body = {"name": table_code, "area": "all", "format": fmt, "language": "de"}
        try:
            res = http.fetch(url, method="POST", data=body, headers={"username": user, "password": pw})
            raw.append(_raw(self.source_id, "daily_index", res))
            return res, None
        except http.AuthError:
            return None, ConnectorResult(
                source_id=self.source_id, status=SourceStatus.AUTH_MISSING, raw=raw,
                message="GENESIS-Online: Zugangsdaten abgelehnt (DESTATIS_USER/DESTATIS_PASSWORD "
                        "oder GAST-Fallback prüfen).",
            )
        except http.FetchError as e:
            log.warning("destatis data/tablefile POST fehlgeschlagen (format=%s): %s -> GET-Fallback", fmt, e)

        params = {"username": user, "password": pw, **body}
        try:
            res = http.fetch(url, params=params)
        except http.AuthError:
            return None, ConnectorResult(
                source_id=self.source_id, status=SourceStatus.AUTH_MISSING, raw=raw,
                message="GENESIS-Online: Zugangsdaten abgelehnt (DESTATIS_USER/DESTATIS_PASSWORD "
                        "oder GAST-Fallback prüfen).",
            )
        except http.FetchError as e:
            return None, ConnectorResult(
                source_id=self.source_id, status=SourceStatus.FAIL, raw=raw,
                message=f"GENESIS data/tablefile Abruf fehlgeschlagen (POST und GET, format={fmt}): {e}",
            )
        raw.append(_raw(self.source_id, "daily_index", res))
        return res, None

    def fetch(self, now: datetime) -> ConnectorResult:
        raw: list[RawRecord] = []
        discovered: dict = {}
        table_code, disc = self._find_table_code(now, raw)
        discovered.update(disc)

        if table_code is None:
            table_code = self.cfg.get("expected_table_code")
            if not table_code:
                return ConnectorResult(
                    source_id=self.source_id, status=SourceStatus.FAIL, raw=raw,
                    message="Katalogsuche fehlgeschlagen und keine expected_table_code "
                            "in der Config hinterlegt.",
                    discovered_ids=discovered,
                )
            discovered["fallback_table_code_source"] = "config.expected_table_code (verify in preflight)"

        observations: list = []
        latest_obs = None
        parse_failures = 0
        last_diag: dict = {}
        # GENESIS liefert je Tabelle/Instanz teils nur "ffcsv", teils nur
        # "csv" aus -- beide Formate werden versucht, bevor laut mit
        # SCHEMA_CHANGED gescheitert wird.
        for fmt in ("ffcsv", "csv"):
            res, err = self._fetch_tablefile(table_code, fmt, raw)
            if err is not None:
                err.discovered_ids = discovered
                return err

            try:
                text = res.content.decode("utf-8-sig")
            except UnicodeDecodeError:
                last_diag = {"format": fmt, "content_type": res.content_type,
                             "body_snippet": "<nicht UTF-8-dekodierbar>"}
                continue

            # Manche GENESIS-Antworten sind JSON-Umschläge mit Fehlerstatus
            # statt direktem ffcsv/csv (HTTP 200!) – das prüfen wir zuerst,
            # um klar mit Code/Content zu scheitern statt SCHEMA_CHANGED zu
            # erraten.
            stripped = text.lstrip()
            if stripped.startswith("{"):
                try:
                    envelope = json.loads(stripped)
                    status_obj = envelope.get("Status") or {}
                    status_code = status_obj.get("Code")
                    if status_code not in (0, None):
                        return ConnectorResult(
                            source_id=self.source_id, status=SourceStatus.FAIL, raw=raw,
                            message=f"GENESIS meldet Status {status_code}: "
                                    f"{status_obj.get('Content')} (format={fmt})",
                            discovered_ids=discovered,
                        )
                except json.JSONDecodeError:
                    pass

            observations, latest_obs, parse_failures = self._parse_ffcsv(text, table_code, res.retrieved_at)
            if observations or parse_failures:
                break
            last_diag = {"format": fmt, "content_type": res.content_type, "body_snippet": text[:600]}
        else:
            discovered["diagnostics"] = last_diag
            return ConnectorResult(
                source_id=self.source_id, status=SourceStatus.SCHEMA_CHANGED, raw=raw,
                message="ffcsv/csv-Antwort enthielt keine erkennbaren Datenzeilen/Spalten "
                        "(erwartetes GENESIS-Schema für beide Formate geprüft) – vermutlich Schemaänderung.",
                discovered_ids=discovered,
            )

        status = SourceStatus.WARN if parse_failures else SourceStatus.PASS
        return ConnectorResult(
            source_id=self.source_id, status=status, observations=observations, raw=raw,
            message=f"{len(observations)} Beobachtungen aus Tabelle {table_code}"
                    + (f", {parse_failures} Zeilen nicht parsebar" if parse_failures else ""),
            latest_observation_time=latest_obs, discovered_ids=discovered,
            parse_failures=parse_failures,
        )

    def _parse_ffcsv(self, text: str, table_code: str, retrieved_at: datetime
                      ) -> tuple[list[Observation], datetime | None, int]:
        """GENESIS ffcsv: Semikolon-getrennt, Header + Datenzeilen. Erwartete
        Spalten (nicht garantiert stabil zwischen Tabellenversionen):
        Zeit / Zeit_Label (Datum), 1_Auspraegung_Label (Region/Bundesland),
        2_Auspraegung_Label (Bereinigungsart), Wert."""
        reader = csv.reader(io.StringIO(text), delimiter=";")
        rows = list(reader)
        if not rows or len(rows) < 2:
            return [], None, 0
        header = [h.strip().strip('"') for h in rows[0]]

        def col(*names: str) -> int | None:
            for n in names:
                if n in header:
                    return header.index(n)
            return None

        idx_time = col("Zeit", "Zeit_Label", "Zeitmerkmal_Code")
        idx_value = col("Wert", "wert")
        idx_region = col("1_Auspraegung_Label", "1_Merkmal_Label")
        idx_adjust = col("2_Auspraegung_Label", "2_Merkmal_Label")

        if idx_time is None or idx_value is None:
            return [], None, 0

        observations: list[Observation] = []
        latest: datetime | None = None
        parse_failures = 0
        for row in rows[1:]:
            if len(row) <= max(idx_time, idx_value):
                parse_failures += 1
                continue
            raw_date = row[idx_time].strip().strip('"')
            raw_val = row[idx_value].strip().strip('"').replace(",", ".")
            try:
                obs_date = self._parse_genesis_date(raw_date)
            except ValueError:
                parse_failures += 1
                continue
            value = None if raw_val in ("", ".", "-", "x") else self._safe_float(raw_val)
            region_label = row[idx_region].strip().strip('"') if idx_region is not None and idx_region < len(row) else ""
            adjust_label = row[idx_adjust].strip().strip('"') if idx_adjust is not None and idx_adjust < len(row) else ""

            entity_id = self.STATE_LABELS.get(region_label, "") if region_label and region_label.lower() != "deutschland" else ""
            if region_label and region_label not in self.STATE_LABELS and region_label.lower() != "deutschland":
                # Unbekannte Region -> nicht raten, aber nicht verwerfen: als
                # eigene entity_id durchreichen statt Länderzuordnung zu erfinden.
                entity_id = region_label

            metric = "index_unadjusted"
            adj_lower = adjust_label.lower()
            if "kalender" in adj_lower and "saison" in adj_lower:
                metric = "index_sa"
            elif "unbereinigt" in adj_lower or adjust_label == "":
                metric = "index_unadjusted"
            else:
                metric = f"index_{adjust_label}" if adjust_label else "index_unadjusted"

            obs = Observation(
                source_id=self.source_id, dataset="daily_index", series_id=table_code,
                entity_id=entity_id, metric=metric, value=value, unit="index_points",
                observation_time=obs_date, available_at=retrieved_at, retrieved_at=retrieved_at,
                availability_precision=AvailabilityPrecision.CONSERVATIVE_DATE,
                parser_version=self.parser_version,
                attrs={"table_code": table_code, "region_label": region_label, "adjustment_label": adjust_label},
            )
            observations.append(obs)
            if latest is None or obs_date > latest:
                latest = obs_date
        return observations, latest, parse_failures

    @staticmethod
    def _safe_float(s: str) -> float | None:
        try:
            return float(s)
        except ValueError:
            return None

    @staticmethod
    def _parse_genesis_date(raw: str) -> datetime:
        raw = raw.strip()
        for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%Y%m%d"):
            try:
                return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        raise ValueError(f"Unbekanntes Datumsformat: {raw}")


# ---------------------------------------------------------------------------
# 2) bts_freight_tsi – US BTS Freight Transportation Services Index
# ---------------------------------------------------------------------------

class BtsFreightTsiConnector(Connector):
    """Primär FRED/ALFRED (Vintages) bei vorhandenem FRED_API_KEY, sonst
    fredgraph.csv (nur letzter Stand, keine Vintages).

    "TSIFRGHT" gilt laut Aufgabenstellung als FRED-Series-ID des Freight TSI –
    unsicher, daher: verify in preflight. Bei API-Key wird zusätzlich
    fred/series/search als Discovery-Fallback genutzt, falls die erwartete
    ID vom Server abgelehnt wird.
    """

    source_id = "bts_freight_tsi"
    parser_version = PARSER_VERSION

    FRED_API_BASE = "https://api.stlouisfed.org/fred"
    FREDGRAPH_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"

    def _search_series_id(self, api_key: str, raw: list[RawRecord]) -> str | None:
        url = f"{self.FRED_API_BASE}/series/search"
        params = {"search_text": "Freight Transportation Services Index",
                   "api_key": api_key, "file_type": "json"}
        try:
            res = http.fetch(url, params=params)
        except http.FetchError as e:
            log.warning("FRED series/search fehlgeschlagen: %s", e)
            return None
        raw.append(_raw(self.source_id, "series_search", res))
        try:
            data = res.json()
        except json.JSONDecodeError:
            return None
        for s in data.get("seriess", []):
            title = (s.get("title") or "").lower()
            if "freight" in title and "transportation services index" in title:
                return s.get("id")
        return None

    def fetch(self, now: datetime) -> ConnectorResult:
        raw: list[RawRecord] = []
        discovered: dict = {}
        expected_series_id = self.cfg.get("expected_series_id", "TSIFRGHT")
        api_key = os.environ.get("FRED_API_KEY")

        if api_key:
            return self._fetch_alfred(now, api_key, expected_series_id, raw, discovered)
        return self._fetch_fredgraph_fallback(now, expected_series_id, raw, discovered)

    def _fetch_alfred(self, now, api_key, expected_series_id, raw, discovered) -> ConnectorResult:
        series_id = expected_series_id
        url = f"{self.FRED_API_BASE}/series/observations"
        params = {
            "series_id": series_id, "api_key": api_key, "file_type": "json",
            "realtime_start": "1776-07-04", "realtime_end": "9999-12-31",
        }
        try:
            res = http.fetch(url, params=params)
        except http.AuthError:
            return ConnectorResult(
                source_id=self.source_id, status=SourceStatus.AUTH_MISSING, raw=raw,
                message="FRED_API_KEY abgelehnt.", discovered_ids=discovered,
            )
        except http.FetchError as e:
            return ConnectorResult(
                source_id=self.source_id, status=SourceStatus.FAIL, raw=raw,
                message=f"FRED ALFRED-Abruf fehlgeschlagen: {e}", discovered_ids=discovered,
            )
        raw.append(_raw(self.source_id, "alfred_vintages", res))
        try:
            data = res.json()
        except json.JSONDecodeError:
            return ConnectorResult(
                source_id=self.source_id, status=SourceStatus.SCHEMA_CHANGED, raw=raw,
                message="FRED-Antwort kein valides JSON.", discovered_ids=discovered,
            )

        if "error_message" in data:
            # erwartete Series-ID vom Server abgelehnt -> Discovery-Fallback
            discovered["expected_series_id_rejected"] = series_id
            alt = self._search_series_id(api_key, raw)
            if not alt:
                return ConnectorResult(
                    source_id=self.source_id, status=SourceStatus.SCHEMA_CHANGED, raw=raw,
                    message=f"FRED lehnt Series-ID '{series_id}' ab ({data.get('error_message')}) "
                            "und series/search fand keinen Ersatz -> series_id in preflight verifizieren.",
                    discovered_ids=discovered,
                )
            discovered["bts_freight_tsi_series_id"] = alt
            params["series_id"] = alt
            try:
                res = http.fetch(url, params=params)
            except http.FetchError as e:
                return ConnectorResult(
                    source_id=self.source_id, status=SourceStatus.FAIL, raw=raw,
                    message=f"FRED ALFRED-Abruf (entdeckte ID {alt}) fehlgeschlagen: {e}",
                    discovered_ids=discovered,
                )
            raw.append(_raw(self.source_id, "alfred_vintages_retry", res))
            series_id = alt
            try:
                data = res.json()
            except json.JSONDecodeError:
                return ConnectorResult(
                    source_id=self.source_id, status=SourceStatus.SCHEMA_CHANGED, raw=raw,
                    message="FRED-Antwort (Discovery-Serie) kein valides JSON.",
                    discovered_ids=discovered,
                )
        else:
            discovered["bts_freight_tsi_series_id"] = series_id

        obs_raw = data.get("observations")
        if obs_raw is None:
            return ConnectorResult(
                source_id=self.source_id, status=SourceStatus.SCHEMA_CHANGED, raw=raw,
                message="FRED-Antwort ohne 'observations' -> Schema geändert.",
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
                source_id=self.source_id, dataset="freight_tsi", series_id=series_id,
                entity_id="US", metric="us_freight_tsi", value=value, unit="index_points",
                observation_time=obs_time, available_at=vintage, retrieved_at=res.retrieved_at,
                availability_precision=AvailabilityPrecision.EXACT_DATE,
                vintage_time=vintage, parser_version=self.parser_version,
                attrs={"realtime_end": o.get("realtime_end")},
            ))
            if latest is None or obs_time > latest:
                latest = obs_time

        if not observations and parse_failures == 0:
            return ConnectorResult(
                source_id=self.source_id, status=SourceStatus.SCHEMA_CHANGED, raw=raw,
                message="Keine auswertbaren Beobachtungen in ALFRED-Antwort.",
                discovered_ids=discovered,
            )
        status = SourceStatus.WARN if parse_failures else SourceStatus.PASS
        return ConnectorResult(
            source_id=self.source_id, status=status, observations=observations, raw=raw,
            message=f"{len(observations)} ALFRED-Vintage-Beobachtungen für {series_id}",
            latest_observation_time=latest, discovered_ids=discovered,
            parse_failures=parse_failures,
        )

    def _fetch_fredgraph_fallback(self, now, expected_series_id, raw, discovered) -> ConnectorResult:
        url = self.FREDGRAPH_CSV
        params = {"id": expected_series_id}
        try:
            # fredgraph.csv ist gelegentlich langsam/rate-limited; ohne
            # FRED_API_KEY ist das der einzige Zugang -> großzügigeres
            # Timeout + Retries statt sofort FAIL bei einem einzelnen
            # Read-Timeout.
            res = http.fetch(url, params=params, timeout=60, retries=2)
        except http.FetchError as e:
            return ConnectorResult(
                source_id=self.source_id, status=SourceStatus.FAIL, raw=raw,
                message=f"fredgraph.csv-Fallback fehlgeschlagen: {e}. "
                        "Empfehlung: FRED_API_KEY setzen, um stattdessen die stabilere "
                        "ALFRED-Vintages-API zu nutzen.",
                discovered_ids=discovered,
            )
        raw.append(_raw(self.source_id, "fredgraph_csv_fallback", res))
        text = res.content.decode("utf-8", errors="replace")
        lines = [l for l in text.strip().split("\n") if l and not l.upper().startswith("DATE")]
        if not lines:
            return ConnectorResult(
                source_id=self.source_id, status=SourceStatus.SCHEMA_CHANGED, raw=raw,
                message="fredgraph.csv lieferte keine Datenzeilen (Series-ID in preflight verifizieren: "
                        f"'{expected_series_id}'). Empfehlung: FRED_API_KEY setzen (ALFRED-API statt "
                        "fredgraph.csv-Fallback).",
                discovered_ids={**discovered, "diagnostics": {"body_snippet": text[:600],
                                                                "content_type": res.content_type}},
            )
        last_date, last_value = None, None
        for line in lines:
            parts = line.split(",")
            if len(parts) < 2:
                continue
            try:
                d = datetime.strptime(parts[0].strip(), "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            v = None if parts[1].strip() in (".", "") else self._safe_float(parts[1].strip())
            last_date, last_value = d, v
        if last_date is None:
            return ConnectorResult(
                source_id=self.source_id, status=SourceStatus.SCHEMA_CHANGED, raw=raw,
                message="fredgraph.csv: keine parsebare Zeile gefunden.",
                discovered_ids=discovered,
            )
        discovered["bts_freight_tsi_series_id"] = expected_series_id
        obs = Observation(
            source_id=self.source_id, dataset="freight_tsi", series_id=expected_series_id,
            entity_id="US", metric="us_freight_tsi", value=last_value, unit="index_points",
            observation_time=last_date, available_at=res.retrieved_at, retrieved_at=res.retrieved_at,
            availability_precision=AvailabilityPrecision.CONSERVATIVE_DATE,
            parser_version=self.parser_version,
        )
        return ConnectorResult(
            source_id=self.source_id, status=SourceStatus.WARN, observations=[obs], raw=raw,
            message="Kein FRED_API_KEY gesetzt -> fredgraph.csv-Fallback (nur letzter Stand, "
                    "supports_vintages=false in diesem Modus).",
            latest_observation_time=last_date, discovered_ids=discovered,
        )

    @staticmethod
    def _safe_float(s) -> float | None:
        try:
            return float(s)
        except (TypeError, ValueError):
            return None


# ---------------------------------------------------------------------------
# 3) eurostat_road_freight – Eurostat road_go_ta_* (JSON-stat 2.0)
# ---------------------------------------------------------------------------

class _DuplicateIdentityError(ValueError):
    """Zwei JSON-stat-Werte mappen auf dieselbe Beobachtungs-Identität
    (series_id, entity_id, metric, observation_time) -- wird NIE
    stillschweigend zusammengefasst, siehe _parse_jsonstat()."""


class EurostatRoadFreightConnector(Connector):
    """Discovery über die offizielle Eurostat-TOC (table of contents,
    https://ec.europa.eu/eurostat/api/dissemination/catalogue/toc/txt) –
    Volltextsuche nach "Road freight transport" liefert den offiziellen
    Datensatz-Code (keine erfundenen Codes)."""

    source_id = "eurostat_road_freight"
    parser_version = PARSER_VERSION

    TOC_URL = "https://ec.europa.eu/eurostat/api/dissemination/catalogue/toc/txt?lang=en"
    DATA_BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
    COUNTRIES = ["DE", "FR", "IT", "ES", "PL", "NL", "BE", "AT", "CZ", "SE", "EU27_2020"]

    # Kuratierte Slice-Defaults (siehe config/external_sources/road_freight.yaml
    # -> dimension_filters). Reduziert das Antwortvolumen UND stellt sicher,
    # dass genau EINE gut-definierte Serie je Land/Einheit ankommt (nie alle
    # ~55 nst07/tra_type/carriage-Kombinationen ungefiltert).
    DIMENSION_FILTERS = {"unit": ["THS_T", "MIO_TKM"], "tra_type": ["TOTAL"], "carriage": ["TOT"]}

    # Nur echte Datensatz-Codes (type == "dataset" in der TOC) UND das
    # dokumentierte Namensmuster road_go_ta_* -- die TOC enthält auch
    # Ordner/Tabellen-Einträge (z.B. "road_go" als Kategorie-Knoten), die
    # NIE als Datensatz-Endpoint verwendet werden dürfen (führte live zu
    # 404 auf .../data/road_go).
    DATASET_CODE_RE = re.compile(r"^road_go_ta_")

    def _discover_dataset_code(self, raw: list[RawRecord]) -> tuple[str | None, dict]:
        search_terms = self.cfg.get("search_terms", ["Road freight transport"])
        expected_code = self.cfg.get("expected_dataset_code")
        try:
            res = http.fetch(self.TOC_URL)
        except http.FetchError as e:
            log.warning("Eurostat TOC nicht erreichbar: %s", e)
            return None, {}
        raw.append(_raw(self.source_id, "toc", res))
        try:
            text = res.content.decode("utf-8")
        except UnicodeDecodeError:
            return None, {}
        rows = list(csv.reader(io.StringIO(text), delimiter="\t"))
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
            if not self.DATASET_CODE_RE.match(code):
                continue
            if any(term.lower() in title.lower() for term in search_terms):
                matches.append((code, title))
        if not matches:
            return None, {}
        matches.sort(key=lambda m: m[0])
        chosen = matches[0][0]
        # Bevorzuge die konfigurierte erwartete Dataset-Code, WENN sie
        # tatsächlich unter den (echten Datensatz-)Treffern der TOC auftaucht
        # -- rät nie eine ID, die nicht in den TOC-Kandidaten steht.
        if expected_code and any(c == expected_code for c, _ in matches):
            chosen = expected_code
        return chosen, {"eurostat_road_freight_candidates": dict(matches), "eurostat_chosen": chosen}

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
            return None, str(e)
        raw.append(_raw(self.source_id, "jsonstat", res))
        return res, None

    def fetch(self, now: datetime) -> ConnectorResult:
        raw: list[RawRecord] = []
        discovered: dict = {}
        code, disc = self._discover_dataset_code(raw)
        discovered.update(disc)
        expected_code = self.cfg.get("expected_dataset_code")
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
            # der TOC-discovered Code ist nicht (mehr) abrufbar -> ein Retry
            # mit der konfigurierten erwarteten Dataset-Code, statt sofort
            # laut zu scheitern.
            discovered["eurostat_discovered_code_404"] = code
            code = expected_code
            discovered["eurostat_chosen"] = code
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

        try:
            observations, latest, release_time, parse_failures, parse_diag = self._parse_jsonstat(
                data, code, res.retrieved_at)
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
        status = SourceStatus.WARN if parse_failures else SourceStatus.PASS
        return ConnectorResult(
            source_id=self.source_id, status=status, observations=observations, raw=raw,
            message=f"{len(observations)} Beobachtungen aus Datensatz {code}",
            latest_observation_time=latest, latest_release_time=release_time,
            discovered_ids=discovered, parse_failures=parse_failures,
        )

    def _parse_jsonstat(self, data: dict, code: str, retrieved_at: datetime):
        """Parst ALLE JSON-stat-2.0-Dimensionen generisch (nicht nur
        geo/time/unit) -- jede Kombination der übrigen Dimensionen
        (nst07/tra_type/carriage/...) ist eine EIGENE Serie/Identität, nie
        stillschweigend mit anderen Kombinationen zusammengefasst.

        Rückgabe: (observations, latest_obs_time, release_time,
                   parse_failures, diagnostics_dict). observations is None
        bei einem Schema, das nicht mal minimal auswertbar ist (dann
        SCHEMA_CHANGED beim Aufrufer). Bei einer echten Identitäts-Kollision
        wird _DuplicateIdentityError geworfen (ebenfalls SCHEMA_CHANGED,
        aber mit einer klaren, spezifischen Nachricht statt Stille)."""
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

        pos = {name: i for i, name in enumerate(ids)}
        if "geo" not in pos or "time" not in pos:
            return None, None, None, 0, diagnostics

        geo_of = index_map("geo")
        time_of = index_map("time")
        # ALLE übrigen Dimensionen (nicht nur "unit") werden generisch
        # indiziert -- das ist der Kernfix: vorher wurden nst07/tra_type/
        # carriage/... komplett ignoriert und bis zu 55 Werte teilten sich
        # dieselbe (series_id, entity_id, metric, observation_time)-Identität.
        other_dim_ids = sorted(d for d in ids if d not in ("geo", "time"))
        dim_maps = {d: index_map(d) for d in other_dim_ids}

        # Diagnostik: konfigurierte Slice-Filter, die dieser Datensatz gar
        # nicht kennt (nie stillschweigend danach filtern -- siehe fetch()).
        configured_filters = self.cfg.get("dimension_filters") or self.DIMENSION_FILTERS
        missing_filter_dims = sorted(set(configured_filters) - set(ids))
        if missing_filter_dims:
            diagnostics["eurostat_dimension_filters_not_in_dataset"] = missing_filter_dims

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
                obs_time = self._parse_eurostat_time(time_label)
            except ValueError:
                parse_failures += 1
                continue

            dimkey = "|".join(f"{d}={dim_parts[d]}" for d in sorted(dim_parts))
            series_id = f"{code}:{dimkey}" if dimkey else code
            unit_code = dim_parts.get("unit")
            metric = f"road_freight_{unit_code.lower()}" if unit_code else "road_freight_unknown_unit"

            identity = (series_id, geo_label, metric, obs_time.isoformat())
            if identity in seen_identities:
                raise _DuplicateIdentityError(
                    f"identity={identity!r} kollidiert bei value-Keys "
                    f"{seen_identities[identity]!r} und {key_str!r} (Datensatz {code})")
            seen_identities[identity] = key_str

            avail_precision = AvailabilityPrecision.EXACT_TIMESTAMP if release_time else AvailabilityPrecision.CONSERVATIVE_DATE
            available_at = release_time or retrieved_at
            observations.append(Observation(
                source_id=self.source_id, dataset="road_freight", series_id=series_id,
                entity_id=geo_label, metric=metric,
                value=float(val) if val is not None else None, unit=unit_code or "unknown",
                observation_time=obs_time, available_at=available_at, retrieved_at=retrieved_at,
                availability_precision=avail_precision, parser_version=self.parser_version,
                attrs=dict(dim_parts),
            ))
            if latest is None or obs_time > latest:
                latest = obs_time
        return observations, latest, release_time, parse_failures, diagnostics

    @staticmethod
    def _parse_eurostat_time(label: str) -> datetime:
        label = label.strip()
        if len(label) == 4 and label.isdigit():
            return datetime(int(label), 1, 1, tzinfo=timezone.utc)
        if "Q" in label:
            year, q = label.split("-Q")
            month = (int(q) - 1) * 3 + 1
            return datetime(int(year), month, 1, tzinfo=timezone.utc)
        if "M" in label:
            year, m = label.split("-M")
            return datetime(int(year), int(m), 1, tzinfo=timezone.utc)
        raise ValueError(f"Unbekanntes Eurostat-Zeitformat: {label}")


# ---------------------------------------------------------------------------
# 4) estat_jp_truck – Japan e-Stat (自動車輸送統計 / MLIT Motor Vehicle Transport)
# ---------------------------------------------------------------------------

class EstatJpTruckConnector(Connector):
    """e-Stat API v3. Ohne ESTAT_APP_ID wird KEIN HTTP-Call ausgeführt
    (AUTH_MISSING sofort)."""

    source_id = "estat_jp_truck"
    parser_version = PARSER_VERSION

    BASE = "https://api.e-stat.go.jp/rest/3.0/app/json"
    SEARCH_WORD = "自動車輸送統計"

    def fetch(self, now: datetime) -> ConnectorResult:
        app_id = os.environ.get("ESTAT_APP_ID")
        if not app_id:
            return ConnectorResult(
                source_id=self.source_id, status=SourceStatus.AUTH_MISSING,
                message="ESTAT_APP_ID nicht gesetzt -> kein API-Call ausgeführt.",
            )
        raw: list[RawRecord] = []
        discovered: dict = {}

        list_url = f"{self.BASE}/getStatsList"
        try:
            res = http.fetch(list_url, params={"appId": app_id, "searchWord": self.SEARCH_WORD, "limit": 20})
        except http.AuthError:
            return ConnectorResult(
                source_id=self.source_id, status=SourceStatus.AUTH_MISSING, raw=raw,
                message="e-Stat getStatsList: ESTAT_APP_ID abgelehnt.",
            )
        except http.FetchError as e:
            return ConnectorResult(
                source_id=self.source_id, status=SourceStatus.FAIL, raw=raw,
                message=f"e-Stat getStatsList fehlgeschlagen: {e}",
            )
        raw.append(_raw(self.source_id, "stats_list", res))
        try:
            data = res.json()
        except json.JSONDecodeError:
            return ConnectorResult(
                source_id=self.source_id, status=SourceStatus.SCHEMA_CHANGED, raw=raw,
                message="getStatsList: kein valides JSON.",
            )
        result_root = (data.get("GET_STATS_LIST") or {}).get("DATALIST_INF") or {}
        tables = result_root.get("TABLE_INF")
        if tables is None:
            return ConnectorResult(
                source_id=self.source_id, status=SourceStatus.SCHEMA_CHANGED, raw=raw,
                message="getStatsList-Antwort ohne TABLE_INF -> Schema geändert.",
            )
        if isinstance(tables, dict):
            tables = [tables]
        if not tables:
            return ConnectorResult(
                source_id=self.source_id, status=SourceStatus.FAIL, raw=raw,
                message=f"Keine Tabellen für Suchwort '{self.SEARCH_WORD}' gefunden.",
            )
        chosen = tables[0]
        stats_data_id = chosen.get("@id")
        if not stats_data_id:
            return ConnectorResult(
                source_id=self.source_id, status=SourceStatus.SCHEMA_CHANGED, raw=raw,
                message="Gefundene Tabelle ohne @id -> Schema geändert.",
            )
        discovered["estat_jp_truck_stats_data_id"] = stats_data_id
        discovered["estat_jp_truck_candidates"] = [t.get("@id") for t in tables]

        data_url = f"{self.BASE}/getStatsData"
        try:
            res2 = http.fetch(data_url, params={"appId": app_id, "statsDataId": stats_data_id})
        except http.FetchError as e:
            return ConnectorResult(
                source_id=self.source_id, status=SourceStatus.FAIL, raw=raw,
                message=f"e-Stat getStatsData fehlgeschlagen: {e}", discovered_ids=discovered,
            )
        raw.append(_raw(self.source_id, "stats_data", res2))
        try:
            data2 = res2.json()
        except json.JSONDecodeError:
            return ConnectorResult(
                source_id=self.source_id, status=SourceStatus.SCHEMA_CHANGED, raw=raw,
                message="getStatsData: kein valides JSON.", discovered_ids=discovered,
            )

        observations, latest, parse_failures = self._parse_estat(data2, stats_data_id, res2.retrieved_at)
        if observations is None:
            return ConnectorResult(
                source_id=self.source_id, status=SourceStatus.SCHEMA_CHANGED, raw=raw,
                message="getStatsData-Antwort entspricht nicht dem erwarteten "
                        "STATISTICAL_DATA/DATA_INF/VALUE-Schema.",
                discovered_ids=discovered,
            )
        status = SourceStatus.WARN if parse_failures else SourceStatus.PASS
        return ConnectorResult(
            source_id=self.source_id, status=status, observations=observations, raw=raw,
            message=f"{len(observations)} Beobachtungen aus statsDataId {stats_data_id}",
            latest_observation_time=latest, discovered_ids=discovered,
            parse_failures=parse_failures,
        )

    def _parse_estat(self, data: dict, stats_data_id: str, retrieved_at: datetime):
        root = (data.get("GET_STATS_DATA") or {}).get("STATISTICAL_DATA")
        if root is None:
            return None, None, 0
        data_inf = root.get("DATA_INF")
        if data_inf is None:
            return None, None, 0
        values = data_inf.get("VALUE")
        if values is None:
            return None, None, 0
        if isinstance(values, dict):
            values = [values]

        # Metrik-Namen aus CLASS_INF (@id -> @name) auflösen, falls vorhanden.
        class_names: dict[str, str] = {}
        class_obj = (root.get("CLASS_INF") or {}).get("CLASS_OBJ")
        if isinstance(class_obj, dict):
            class_obj = [class_obj]
        for c in class_obj or []:
            classes = c.get("CLASS")
            if isinstance(classes, dict):
                classes = [classes]
            for cls in classes or []:
                if cls.get("@code"):
                    class_names[cls["@code"]] = cls.get("@name", cls["@code"])

        observations = []
        latest = None
        parse_failures = 0
        for v in values:
            time_code = v.get("@time")
            val_raw = v.get("$")
            if time_code is None:
                parse_failures += 1
                continue
            try:
                obs_time = self._parse_estat_time(time_code)
            except ValueError:
                parse_failures += 1
                continue
            cat_code = v.get("@cat01", "")
            metric_name = class_names.get(cat_code, cat_code or "value")
            area_code = v.get("@area", "")
            value = None if val_raw in (None, "", "-", "***") else self._safe_float(val_raw)
            observations.append(Observation(
                source_id="estat_jp_truck", dataset="motor_vehicle_transport",
                series_id=stats_data_id, entity_id=area_code or "JP",
                metric=f"jp_truck_{metric_name}", value=value, unit="see_metric",
                observation_time=obs_time, available_at=retrieved_at, retrieved_at=retrieved_at,
                availability_precision=AvailabilityPrecision.CONSERVATIVE_DATE,
                parser_version=PARSER_VERSION, attrs={"cat01": cat_code, "area": area_code},
            ))
            if latest is None or obs_time > latest:
                latest = obs_time
        return observations, latest, parse_failures

    @staticmethod
    def _safe_float(s) -> float | None:
        try:
            return float(s)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_estat_time(code: str) -> datetime:
        code = code.strip()
        if len(code) == 6:  # YYYYMM
            return datetime(int(code[:4]), int(code[4:6]), 1, tzinfo=timezone.utc)
        if len(code) == 4:  # YYYY
            return datetime(int(code), 1, 1, tzinfo=timezone.utc)
        if len(code) == 8:  # YYYYMMDD
            return datetime(int(code[:4]), int(code[4:6]), int(code[6:8]), tzinfo=timezone.utc)
        raise ValueError(f"Unbekanntes e-Stat-Zeitformat: {code}")


CONNECTORS: dict[str, type[Connector]] = {
    "destatis_truck_toll": DestatisTruckTollConnector,
    "bts_freight_tsi": BtsFreightTsiConnector,
    "eurostat_road_freight": EurostatRoadFreightConnector,
    "estat_jp_truck": EstatJpTruckConnector,
    # fhwa_faf: kein Live-Konnektor (siehe scripts/build_faf_exposure.py) –
    # bewusst nicht in CONNECTORS, Registry-Eintrag hat status_override DEFERRED.
    # viapass_be, asfinag_at, nbs_cn, mot_cn, kosis_kr: Preflight-only-Einträge
    # ohne Konnektor-Klasse (siehe config/external_sources/road_freight.yaml).
}

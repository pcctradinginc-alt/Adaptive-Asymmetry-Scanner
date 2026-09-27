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
from pathlib import Path

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

def _decode_genesis_tablefile(content: bytes) -> str | None:
    """GENESIS liefert data/tablefile je nach Konto/Format als ZIP mit genau
    einer CSV (Content-Type application/zip) oder direkt als Text. Text:
    UTF-8 (mit BOM), sonst CP1252 (klassisches GENESIS-CSV). None, wenn
    weder Text noch ein ZIP mit CSV/TXT."""
    import io
    import zipfile
    if content[:2] == b"PK":
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                names = [n for n in zf.namelist() if n.lower().endswith((".csv", ".txt"))]
                if not names:
                    return None
                content = zf.read(names[0])
        except zipfile.BadZipFile:
            return None
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return content.decode(enc)
        except UnicodeDecodeError:
            continue
    return None


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
        body = {"name": table_code, "area": "all", "format": fmt, "language": "de",
                # ohne startyear liefert GENESIS nur die letzten ~2 Jahre
                # (Preflight 2026-09-27: 24 Monate) -> zu kurz für 3J-Z-Scores
                "startyear": str(self.cfg.get("start_year", 2008))}
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

            text = _decode_genesis_tablefile(res.content)
            if text is None:
                last_diag = {"format": fmt, "content_type": res.content_type,
                             "body_snippet": "<nicht dekodierbar (weder UTF-8/CP1252 noch ZIP mit CSV)>"}
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
            if not observations:
                # klassisches GENESIS-CSV (Kopfblock, Jahr;Monat;Werte je
                # Bereinigungsart) -- so liefert GENESIS 42191-0001 mit Konto
                observations, latest_obs, parse_failures = self._parse_classic_csv(
                    text, table_code, res.retrieved_at)
            if observations:
                break
            # 0 Beobachtungen (auch bei unparsebaren Zeilen, z.B. ffcsv einer
            # Monatstabelle mit Zeit=Jahr) -> nächstes Format versuchen
            last_diag = {"format": fmt, "content_type": res.content_type, "body_snippet": text[:600],
                         "parse_failures": parse_failures}
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

    GERMAN_MONTHS = {"januar": 1, "februar": 2, "märz": 3, "maerz": 3, "april": 4, "mai": 5,
                     "juni": 6, "juli": 7, "august": 8, "september": 9, "oktober": 10,
                     "november": 11, "dezember": 12}

    @staticmethod
    def _classic_metric(label: str) -> str | None:
        """Spaltenlabel -> Metrik. X13 kalender- und saisonbereinigt ist die
        Standard-SA-Reihe (index_sa); Alternativverfahren eigene Metriken."""
        l = label.lower()
        if not l:
            return None
        if "originalwert" in l:
            return "index_unadjusted"
        if "trend" in l:
            return "index_trend"
        if "saison" in l:
            return "index_sa" if ("x13" in l or "bv4" not in l) else "index_sa_bv41"
        if "kalender" in l:
            return "index_calendar_adjusted"
        return None

    def _parse_classic_csv(self, text: str, table_code: str, retrieved_at: datetime
                           ) -> tuple[list[Observation], datetime | None, int]:
        """Klassisches GENESIS-CSV: Kopfzeilen, dann eine Spaltenkopfzeile mit
        den Bereinigungsarten (z.B. ';;Originalwerte;X13 ... saisonbereinigt;
        ...'), dann Datenzeilen 'Jahr;Monat;Wert;...' (Jahr nur in der ersten
        Zeile eines Jahres) oder 'TT.MM.JJJJ;;Wert;...'. Fußzeilen (__, ©)
        werden übersprungen. Nie raten: unbekannte Spaltenlabels entfallen."""
        rows = list(csv.reader(io.StringIO(text), delimiter=";"))
        header_i = next((i for i, r in enumerate(rows)
                         if any(self._classic_metric(c.strip()) for c in r[1:])), None)
        if header_i is None:
            return [], None, 0
        header = [c.strip() for c in rows[header_i]]
        metric_cols = {i: self._classic_metric(c) for i, c in enumerate(header)
                       if i >= 1 and self._classic_metric(c)}
        observations: list[Observation] = []
        latest: datetime | None = None
        parse_failures = 0
        year: int | None = None
        for row in rows[header_i + 1:]:
            if not row or row[0].strip().startswith(("__", "©", "Stand")):
                continue
            c0 = row[0].strip()
            c1 = row[1].strip().lower() if len(row) > 1 else ""
            obs_date = None
            dataset = "monthly_index"
            if re.fullmatch(r"\d{4}", c0):
                year = int(c0)
            if re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", c0):
                obs_date = datetime.strptime(c0, "%d.%m.%Y").replace(tzinfo=timezone.utc)
                dataset = "daily_index"
            elif year is not None and c1 in self.GERMAN_MONTHS:
                obs_date = datetime(year, self.GERMAN_MONTHS[c1], 1, tzinfo=timezone.utc)
            if obs_date is None:
                if any(ch.strip() for ch in row[2:]) and (c0 or c1):
                    parse_failures += 1
                continue
            values = {}
            for i, metric in metric_cols.items():
                if i >= len(row):
                    continue
                raw_val = row[i].strip().replace(".", "").replace(",", ".")
                values[i] = None if raw_val in ("", "-", "x", "...") else self._safe_float(raw_val)
            if all(v is None for v in values.values()):
                # Platzhalterzeilen für noch nicht veröffentlichte Monate
                # (GENESIS listet das ganze laufende Jahr) -> keine Beobachtung,
                # sonst läge latest_observation_time in der Zukunft.
                continue
            for i, value in values.items():
                metric = metric_cols[i]
                observations.append(Observation(
                    source_id=self.source_id, dataset=dataset, series_id=table_code,
                    entity_id="", metric=metric, value=value, unit="index_points",
                    observation_time=obs_date, available_at=retrieved_at, retrieved_at=retrieved_at,
                    availability_precision=AvailabilityPrecision.CONSERVATIVE_DATE,
                    parser_version=self.parser_version,
                    attrs={"table_code": table_code, "adjustment_label": header[i]},
                ))
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
# 1b) destatis_truck_toll_download – Destatis EXDAT-Downloadseite, OHNE
#     GENESIS-Login (öffentlicher .xlsx/.csv-Download auf destatis.de)
# ---------------------------------------------------------------------------

class DestatisTruckTollDownloadConnector(Connector):
    """Lkw-Maut-Fahrleistungsindex ohne GENESIS-Zugangsdaten: Destatis
    veröffentlicht den experimentellen Datensatz zusätzlich als direkten
    Download (.xlsx/.csv) auf einer offiziellen destatis.de-Seite unter
    Service/EXDAT (z.B. .../lkw-maut-fahrleistungsindex.html).

    Discovery: die offizielle Seiten-URL wird geladen; ein Link auf eine
    .xlsx- oder .csv-Datei wird NUR akzeptiert, wenn er (nach Auflösung
    relativer Pfade) auf der Domain destatis.de liegt -- nie ein beliebiger
    externer Link. Mehrere Kandidaten -> .xlsx bevorzugt (Excel enthält
    i.d.R. beide Bereinigungsarten in getrennten Spalten); sonst der erste
    gefundene Kandidat, alle werden in discovered_ids protokolliert.

    Metrik-/Entity-Konventionen identisch zu DestatisTruckTollConnector
    (index_sa / index_unadjusted, entity_id="" für Deutschland gesamt),
    damit modules/external/sources/road_freight_features.de_truck_* beide
    Konnektoren gleichermaßen bedienen kann.
    """

    source_id = "destatis_truck_toll_download"
    parser_version = PARSER_VERSION

    PAGE_URL = "https://www.destatis.de/DE/Service/EXDAT/Datensaetze/lkw-maut-fahrleistungsindex.html"
    ALLOWED_DOMAIN = "destatis.de"
    _LINK_RE = re.compile(r'href="([^"]+?\.(?:xlsx|csv))(\?[^"]*)?"', re.IGNORECASE)

    def _discover_download_url(self, raw: list[RawRecord]) -> tuple[str | None, dict]:
        page_url = self.cfg.get("download_page_url", self.PAGE_URL)
        try:
            res = http.fetch(page_url)
        except http.FetchError as e:
            log.warning("Destatis-EXDAT-Seite nicht erreichbar: %s", e)
            return None, {}
        raw.append(_raw(self.source_id, "download_page", res))
        try:
            html_text = res.content.decode("utf-8", errors="replace")
        except UnicodeDecodeError:
            return None, {}

        from urllib.parse import urljoin, urlparse

        candidates: list[str] = []
        for m in self._LINK_RE.finditer(html_text):
            href = m.group(1)
            resolved = urljoin(page_url, href)
            host = urlparse(resolved).netloc.lower()
            if host == self.ALLOWED_DOMAIN or host.endswith("." + self.ALLOWED_DOMAIN):
                candidates.append(resolved)
        if not candidates:
            return None, {}
        discovered = {"destatis_truck_toll_download_candidates": candidates}
        xlsx = [c for c in candidates if c.lower().split("?")[0].endswith(".xlsx")]
        chosen = xlsx[0] if xlsx else candidates[0]
        discovered["destatis_truck_toll_download_chosen"] = chosen
        return chosen, discovered

    def fetch(self, now: datetime) -> ConnectorResult:
        raw: list[RawRecord] = []
        discovered: dict = {}
        download_url, disc = self._discover_download_url(raw)
        discovered.update(disc)
        if download_url is None:
            download_url = self.cfg.get("expected_download_url")
            if not download_url:
                return ConnectorResult(
                    source_id=self.source_id, status=SourceStatus.FAIL, raw=raw,
                    message="Konnte auf der offiziellen Destatis-EXDAT-Seite keinen "
                            ".xlsx/.csv-Downloadlink auf destatis.de finden, und keine "
                            "expected_download_url in der Config.",
                    discovered_ids=discovered,
                )
            discovered["fallback_download_url_source"] = "config.expected_download_url (verify in preflight)"

        try:
            res = http.fetch(download_url)
        except http.FetchError as e:
            return ConnectorResult(
                source_id=self.source_id, status=SourceStatus.FAIL, raw=raw,
                message=f"Download fehlgeschlagen ({download_url}): {e}", discovered_ids=discovered,
            )
        raw.append(_raw(self.source_id, "daily_index_file", res))

        is_xlsx = download_url.lower().split("?")[0].endswith(".xlsx")
        if is_xlsx:
            observations, latest, parse_failures, diag = self._parse_xlsx(res.content, res.retrieved_at)
        else:
            observations, latest, parse_failures, diag = self._parse_csv(res.content, res.retrieved_at)

        if observations is None:
            discovered.update(diag)
            if diag.get("missing_dependency"):
                return ConnectorResult(
                    source_id=self.source_id, status=SourceStatus.FAIL, raw=raw,
                    message=diag["missing_dependency"], discovered_ids=discovered,
                )
            return ConnectorResult(
                source_id=self.source_id, status=SourceStatus.SCHEMA_CHANGED, raw=raw,
                message="Downloaddatei entspricht nicht dem erwarteten Schema "
                        "(Datums- und/oder Indexspalten nicht gefunden).",
                discovered_ids=discovered,
            )
        if not observations:
            return ConnectorResult(
                source_id=self.source_id, status=SourceStatus.SCHEMA_CHANGED, raw=raw,
                message="Downloaddatei enthielt keine auswertbaren Datenzeilen.",
                discovered_ids=discovered,
            )
        status = SourceStatus.WARN if parse_failures else SourceStatus.PASS
        return ConnectorResult(
            source_id=self.source_id, status=status, observations=observations, raw=raw,
            message=f"{len(observations)} Beobachtungen aus {download_url}"
                    + (f", {parse_failures} Zeilen nicht parsebar" if parse_failures else ""),
            latest_observation_time=latest, discovered_ids=discovered,
            parse_failures=parse_failures,
        )

    @staticmethod
    def _classify_columns(headers: list[str]) -> dict[int, str]:
        """Spaltenindex -> Metrikname, aus Header-Text abgeleitet (gleiche
        Sprache/Konvention wie DestatisTruckTollConnector._parse_ffcsv)."""
        col_metric: dict[int, str] = {}
        for i, h in enumerate(headers):
            hl = (h or "").strip().lower()
            if not hl:
                continue
            has_kalender = "kalender" in hl
            has_saison = "saison" in hl
            if (has_kalender and has_saison) or "bereinigt" in hl and "unbereinigt" not in hl:
                col_metric[i] = "index_sa"
            elif "unbereinigt" in hl or "original" in hl or "rohwert" in hl:
                col_metric[i] = "index_unadjusted"
        return col_metric

    def _parse_xlsx(self, content: bytes, retrieved_at: datetime
                     ) -> tuple[list[Observation] | None, datetime | None, int, dict]:
        try:
            import openpyxl
        except ImportError:
            return None, None, 0, {"missing_dependency":
                "openpyxl nicht installiert -> Voraussetzung für die .xlsx-Verarbeitung "
                "des Destatis-EXDAT-Downloads fehlt (siehe requirements.txt)."}
        try:
            wb = openpyxl.load_workbook(io.BytesIO(content), data_only=True, read_only=True)
        except Exception as e:  # noqa: BLE001 - defekte/unerwartete xlsx -> laut scheitern
            return None, None, 0, {"diagnostics": {"xlsx_load_error": str(e)}}

        best: tuple[list[Observation], datetime | None, int] | None = None
        for ws in wb.worksheets:
            rows_iter = ws.iter_rows(values_only=True)
            try:
                header_row = next(rows_iter)
            except StopIteration:
                continue
            headers = [str(c) if c is not None else "" for c in header_row]
            date_idx = None
            for i, h in enumerate(headers):
                hl = h.strip().lower()
                if hl in ("datum", "date", "tag") or "datum" in hl:
                    date_idx = i
                    break
            col_metric = self._classify_columns(headers)
            if date_idx is None or not col_metric:
                continue

            observations: list[Observation] = []
            latest: datetime | None = None
            parse_failures = 0
            for row in rows_iter:
                if row is None or date_idx >= len(row):
                    continue
                raw_date = row[date_idx]
                obs_date = self._coerce_date(raw_date)
                if obs_date is None:
                    if raw_date not in (None, ""):
                        parse_failures += 1
                    continue
                for col_i, metric in col_metric.items():
                    if col_i >= len(row):
                        continue
                    value = self._coerce_float(row[col_i])
                    observations.append(Observation(
                        source_id=self.source_id, dataset="daily_index", series_id="lkw_maut_fahrleistungsindex",
                        entity_id="", metric=metric, value=value, unit="index_points",
                        observation_time=obs_date, available_at=retrieved_at, retrieved_at=retrieved_at,
                        availability_precision=AvailabilityPrecision.CONSERVATIVE_DATE,
                        parser_version=self.parser_version, attrs={"sheet": ws.title},
                    ))
                if latest is None or obs_date > latest:
                    latest = obs_date
            if observations and (best is None or len(observations) > len(best[0])):
                best = (observations, latest, parse_failures)

        wb.close()
        if best is None:
            return None, None, 0, {"diagnostics": {"reason": "keine Tabelle mit Datums- und Indexspalten gefunden"}}
        return best[0], best[1], best[2], {}

    def _parse_csv(self, content: bytes, retrieved_at: datetime
                    ) -> tuple[list[Observation] | None, datetime | None, int, dict]:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            return None, None, 0, {"diagnostics": {"reason": "CSV nicht UTF-8-dekodierbar"}}
        delimiter = ";" if text.count(";") >= text.count(",") else ","
        rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
        if len(rows) < 2:
            return None, None, 0, {"diagnostics": {"body_snippet": text[:600]}}
        headers = [h.strip() for h in rows[0]]
        date_idx = None
        for i, h in enumerate(headers):
            hl = h.strip().lower()
            if hl in ("datum", "date", "tag") or "datum" in hl:
                date_idx = i
                break
        col_metric = self._classify_columns(headers)
        if date_idx is None or not col_metric:
            return None, None, 0, {"diagnostics": {"body_snippet": text[:600], "headers": headers}}

        observations: list[Observation] = []
        latest: datetime | None = None
        parse_failures = 0
        for row in rows[1:]:
            if date_idx >= len(row):
                parse_failures += 1
                continue
            obs_date = self._coerce_date(row[date_idx])
            if obs_date is None:
                if row[date_idx].strip():
                    parse_failures += 1
                continue
            for col_i, metric in col_metric.items():
                if col_i >= len(row):
                    continue
                value = self._coerce_float(row[col_i])
                observations.append(Observation(
                    source_id=self.source_id, dataset="daily_index", series_id="lkw_maut_fahrleistungsindex",
                    entity_id="", metric=metric, value=value, unit="index_points",
                    observation_time=obs_date, available_at=retrieved_at, retrieved_at=retrieved_at,
                    availability_precision=AvailabilityPrecision.CONSERVATIVE_DATE,
                    parser_version=self.parser_version,
                ))
            if latest is None or obs_date > latest:
                latest = obs_date
        return observations, latest, parse_failures, {}

    @staticmethod
    def _coerce_date(raw) -> datetime | None:
        if raw is None:
            return None
        if isinstance(raw, datetime):
            return raw.replace(tzinfo=timezone.utc) if raw.tzinfo is None else raw.astimezone(timezone.utc)
        s = str(raw).strip().strip('"')
        if not s:
            return None
        for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%Y%m%d"):
            try:
                return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    @staticmethod
    def _coerce_float(raw) -> float | None:
        if raw is None:
            return None
        if isinstance(raw, (int, float)):
            return float(raw)
        s = str(raw).strip().strip('"').replace(",", ".")
        if s in ("", ".", "-", "x"):
            return None
        try:
            return float(s)
        except ValueError:
            return None


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
# 2b) bts_open_data_tsi – US BTS Transportation Services Index, OHNE API-Key
#     (offizielles Socrata-Open-Data-Portal data.bts.gov)
# ---------------------------------------------------------------------------

class BtsOpenDataTsiConnector(Connector):
    """Freight-TSI ohne FRED_API_KEY: das offizielle Socrata-Open-Data-Portal
    von BTS (data.bts.gov) verlangt keinerlei Zugangsdaten.

    Discovery (NIE eine Datensatz-ID erfinden):
      1. Socrata Discovery API (api.us.socrata.com/api/catalog/v1) mit
         domains=data.bts.gov + Volltextsuche nach "Transportation Services
         Index" -> Kandidaten werden in discovered_ids protokolliert; der
         Datensatz, dessen Name "Transportation Services Index" enthält,
         wird gewählt. Mehrere Treffer -> der zuletzt aktualisierte wird
         gewählt, aber als "bts_open_data_tsi_ambiguous" geflaggt.
      2. /api/views/<id>.json (Socrata-Metadaten) liefert die echten
         Spaltennamen -> eine Spalte, die sowohl "freight" als auch
         "tsi"/"index" enthält, wird als us_freight_tsi gemappt (bei
         mehreren Kandidaten wird die saisonbereinigte bevorzugt); eine
         Spalte, die "truck" enthält, wird zusätzlich als us_trucking_index
         gemappt, sofern eindeutig vorhanden. Kein Ratespiel: fehlt die
         Freight-Spalte, ist das Ergebnis SCHEMA_CHANGED.
      3. SODA-Zeilen via /resource/<id>.json mit $limit/$order/$offset
         (Pagination) von data.bts.gov.

    PIT: kein offizieller Zeilenwert-Release-Zeitstempel pro Zeile bekannt
    -> available_at = retrieved_at (CONSERVATIVE_DATE) für neu gesehene
    Zeilen; die Datensatz-Metadaten liefern zusätzlich rowsUpdatedAt, das
    als source_release_time (informativ) mitgeführt wird.
    """

    source_id = "bts_open_data_tsi"
    parser_version = PARSER_VERSION

    CATALOG_URL = "https://api.us.socrata.com/api/catalog/v1"
    DOMAIN = "data.bts.gov"
    SEARCH_QUERY = "Transportation Services Index"
    PAGE_LIMIT = 5000

    def _discover_dataset(self, raw: list[RawRecord]) -> tuple[str | None, dict]:
        params = {"domains": self.DOMAIN, "search_context": self.DOMAIN, "q": self.SEARCH_QUERY}
        try:
            res = http.fetch(self.CATALOG_URL, params=params)
        except http.FetchError as e:
            log.warning("Socrata-Katalogsuche nicht erreichbar: %s", e)
            return None, {}
        raw.append(_raw(self.source_id, "catalog", res))
        try:
            data = res.json()
        except json.JSONDecodeError:
            return None, {}
        results = data.get("results") or []
        candidates = []
        for r in results:
            resource = r.get("resource") or {}
            name = resource.get("name") or ""
            rid = resource.get("id")
            updated = resource.get("updatedAt") or resource.get("data_updated_at")
            if not rid or not name:
                continue
            candidates.append({"id": rid, "name": name, "updatedAt": updated})
        matches = [c for c in candidates if "transportation services index" in c["name"].lower()]
        discovered = {"bts_open_data_tsi_candidates": candidates}
        if not matches:
            return None, discovered
        if len(matches) > 1:
            # Uneindeutig -> zuletzt aktualisierten wählen, aber laut flaggen
            # statt stillschweigend zu raten.
            matches_sorted = sorted(matches, key=lambda c: c["updatedAt"] or "", reverse=True)
            discovered["bts_open_data_tsi_ambiguous"] = [c["id"] for c in matches_sorted]
            chosen = matches_sorted[0]
        else:
            chosen = matches[0]
        discovered["bts_open_data_tsi_dataset_id"] = chosen["id"]
        return chosen["id"], discovered

    def _fetch_metadata(self, dataset_id: str, raw: list[RawRecord]) -> tuple[dict | None, str | None]:
        url = f"https://{self.DOMAIN}/api/views/{dataset_id}.json"
        try:
            res = http.fetch(url)
        except http.FetchError as e:
            return None, str(e)
        raw.append(_raw(self.source_id, "views_metadata", res))
        try:
            data = res.json()
        except json.JSONDecodeError:
            return None, "views-Metadaten kein valides JSON"
        return data, None

    @staticmethod
    def _pick_columns(columns: list[dict]) -> tuple[str | None, str | None, str | None]:
        """Gibt (date_field, freight_field, trucking_field) zurück -- nie
        erfunden, nur aus tatsächlich vorhandenen fieldNames abgeleitet."""
        date_field = None
        freight_candidates = []
        truck_candidates = []
        for c in columns:
            field = (c.get("fieldName") or "").lower()
            name = (c.get("name") or "").lower()
            if not field:
                continue
            if date_field is None and (field in ("date", "period", "month")
                                        or "date" in field or field == "period"):
                date_field = c.get("fieldName")
            if "freight" in field or "freight" in name:
                if "tsi" in field or "tsi" in name or "index" in field or "index" in name:
                    freight_candidates.append(c.get("fieldName"))
            if "truck" in field or "truck" in name:
                truck_candidates.append(c.get("fieldName"))

        def _prefer_seasonally_adjusted(cands: list[str]) -> str | None:
            if not cands:
                return None
            adjusted = [c for c in cands if "unadjust" in c.lower() or "_nsa" in c.lower()
                        or "not_seasonally" in c.lower()]
            preferred = [c for c in cands if c not in adjusted]
            pool = preferred or cands
            return sorted(pool)[0]

        freight_field = _prefer_seasonally_adjusted(freight_candidates)
        truck_field = _prefer_seasonally_adjusted(truck_candidates)
        # trucking-Spalte nur übernehmen, wenn sie eindeutig NICHT die
        # bereits gewählte Freight-Spalte ist (nie doppelt mappen).
        if truck_field == freight_field:
            truck_field = None
        return date_field, freight_field, truck_field

    def _fetch_rows(self, dataset_id: str, date_field: str, raw: list[RawRecord]
                     ) -> tuple[list[dict] | None, str | None]:
        url = f"https://{self.DOMAIN}/resource/{dataset_id}.json"
        page_limit = int(self.cfg.get("page_limit") or self.PAGE_LIMIT)
        rows: list[dict] = []
        offset = 0
        while True:
            params = {"$limit": page_limit, "$order": date_field, "$offset": offset}
            try:
                res = http.fetch(url, params=params)
            except http.FetchError as e:
                return None, str(e)
            raw.append(_raw(self.source_id, "soda_rows", res))
            try:
                page = res.json()
            except json.JSONDecodeError:
                return None, "SODA-Antwort kein valides JSON"
            if not isinstance(page, list):
                return None, "SODA-Antwort kein JSON-Array"
            rows.extend(page)
            if len(page) < page_limit:
                break
            offset += page_limit
            if offset > 200_000:  # Sicherheitsgrenze gegen Endlos-Pagination
                break
        return rows, None

    def fetch(self, now: datetime) -> ConnectorResult:
        raw: list[RawRecord] = []
        discovered: dict = {}
        dataset_id, disc = self._discover_dataset(raw)
        discovered.update(disc)
        expected_id = self.cfg.get("expected_dataset_id")
        if dataset_id is None:
            dataset_id = expected_id
            if not dataset_id:
                return ConnectorResult(
                    source_id=self.source_id, status=SourceStatus.FAIL, raw=raw,
                    message="Socrata-Katalogsuche fand keinen Datensatz mit 'Transportation "
                            "Services Index' im Namen und keine expected_dataset_id in der Config.",
                    discovered_ids=discovered,
                )
            discovered["fallback_dataset_id_source"] = "config.expected_dataset_id (verify in preflight)"

        meta, err = self._fetch_metadata(dataset_id, raw)
        if err is not None:
            return ConnectorResult(
                source_id=self.source_id, status=SourceStatus.FAIL, raw=raw,
                message=f"BTS-Open-Data views-Metadaten-Abruf fehlgeschlagen ({dataset_id}): {err}",
                discovered_ids=discovered,
            )
        columns = meta.get("columns") if isinstance(meta, dict) else None
        if not columns:
            return ConnectorResult(
                source_id=self.source_id, status=SourceStatus.SCHEMA_CHANGED, raw=raw,
                message="views-Metadaten ohne 'columns' -> Schema geändert.",
                discovered_ids=discovered,
            )
        date_field, freight_field, truck_field = self._pick_columns(columns)
        discovered["bts_open_data_tsi_columns"] = {
            "date_field": date_field, "freight_field": freight_field, "truck_field": truck_field,
            "all_field_names": [c.get("fieldName") for c in columns],
        }
        if not date_field or not freight_field:
            return ConnectorResult(
                source_id=self.source_id, status=SourceStatus.SCHEMA_CHANGED, raw=raw,
                message="Konnte weder Datums- noch Freight-TSI-Spalte eindeutig aus den "
                        "views-Metadaten ableiten (erwartete Spaltenmuster nicht gefunden).",
                discovered_ids=discovered,
            )
        rows_updated_raw = meta.get("rowsUpdatedAt")
        source_release_time = None
        if rows_updated_raw is not None:
            try:
                source_release_time = datetime.fromtimestamp(int(rows_updated_raw), tz=timezone.utc)
            except (TypeError, ValueError, OSError):
                source_release_time = None

        rows, err = self._fetch_rows(dataset_id, date_field, raw)
        if err is not None:
            return ConnectorResult(
                source_id=self.source_id, status=SourceStatus.FAIL, raw=raw,
                message=f"SODA-Zeilenabruf fehlgeschlagen ({dataset_id}): {err}",
                discovered_ids=discovered,
            )
        if not rows:
            return ConnectorResult(
                source_id=self.source_id, status=SourceStatus.SCHEMA_CHANGED, raw=raw,
                message="SODA-Endpunkt lieferte keine Zeilen.", discovered_ids=discovered,
            )

        retrieved_at = raw[-1].retrieved_at
        observations: list[Observation] = []
        latest: datetime | None = None
        parse_failures = 0
        for row in rows:
            raw_date = row.get(date_field)
            if not raw_date:
                parse_failures += 1
                continue
            try:
                obs_time = datetime.fromisoformat(str(raw_date).replace("Z", "+00:00"))
                if obs_time.tzinfo is None:
                    obs_time = obs_time.replace(tzinfo=timezone.utc)
                obs_time = obs_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            except ValueError:
                parse_failures += 1
                continue

            for field, metric in ((freight_field, "us_freight_tsi"), (truck_field, "us_trucking_index")):
                if field is None:
                    continue
                raw_val = row.get(field)
                value = self._safe_float(raw_val)
                observations.append(Observation(
                    source_id=self.source_id, dataset="freight_tsi", series_id=dataset_id,
                    entity_id="US", metric=metric, value=value, unit="index_points",
                    observation_time=obs_time, available_at=retrieved_at, retrieved_at=retrieved_at,
                    availability_precision=AvailabilityPrecision.CONSERVATIVE_DATE,
                    source_release_time=source_release_time, parser_version=self.parser_version,
                    attrs={"field_name": field},
                ))
                if latest is None or obs_time > latest:
                    latest = obs_time

        if not observations:
            return ConnectorResult(
                source_id=self.source_id, status=SourceStatus.SCHEMA_CHANGED, raw=raw,
                message="Keine auswertbaren Beobachtungen aus SODA-Zeilen extrahiert.",
                discovered_ids=discovered,
            )
        status = SourceStatus.WARN if parse_failures else SourceStatus.PASS
        return ConnectorResult(
            source_id=self.source_id, status=status, observations=observations, raw=raw,
            message=f"{len(observations)} Beobachtungen aus Datensatz {dataset_id} "
                    f"(freight_field={freight_field}, truck_field={truck_field})",
            latest_observation_time=latest, discovered_ids=discovered,
            parse_failures=parse_failures,
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
    DIMENSION_FILTERS = {"unit": ["THS_T", "MIO_TKM"]}  # weitere Filter nur, wenn live verifiziert

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
            # 400 = ungültiger Dimensionsfilter (z.B. Code existiert im Datensatz
            # nicht). Einmal nur mit geo-Filter wiederholen; Auswahl der Reihen
            # erfolgt dann deterministisch in road_freight_features.
            if "400" in str(e) and len(params) > 3:
                try:
                    res = http.fetch(url, params={"format": "JSON", "lang": "en", "geo": countries})
                except http.FetchError as e2:
                    return None, f"{e} | Retry ohne Dimensionsfilter: {e2}"
                self.filter_fallback = True
            else:
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
        """Rückwärtskompatible Signatur (bestehende Tests/Verhalten
        unverändert): delegiert an _parse_eurostat_jsonstat() OHNE
        previously_seen_periods -> altes First-Party-Verhalten (EXACT_TIMESTAMP
        sobald eine offizielle Release-Zeit bekannt ist, siehe Docstring
        dort). Die First-Seen-PIT-Präzision (siehe Aufgabenstellung Punkt 4)
        ist bewusst NUR im neuen EurostatRoadFreightQuarterlyConnector aktiv,
        um das etablierte, bereits getestete Verhalten dieses Konnektors
        nicht rückwirkend zu ändern."""
        configured_filters = self.cfg.get("dimension_filters") or self.DIMENSION_FILTERS
        obs, latest, release_time, parse_failures, diagnostics, _periods = _parse_eurostat_jsonstat(
            data, code, retrieved_at, configured_filters, source_id=self.source_id,
            parser_version=self.parser_version, previously_seen_periods=None)
        return obs, latest, release_time, parse_failures, diagnostics


def _parse_eurostat_jsonstat(data: dict, code: str, retrieved_at: datetime,
                              configured_filters: dict, *, source_id: str = "eurostat_road_freight",
                              parser_version: str = PARSER_VERSION,
                              previously_seen_periods: set[str] | None = None):
        """Parst ALLE JSON-stat-2.0-Dimensionen generisch (nicht nur
        geo/time/unit) -- jede Kombination der übrigen Dimensionen
        (nst07/tra_type/carriage/...) ist eine EIGENE Serie/Identität, nie
        stillschweigend mit anderen Kombinationen zusammengefasst.

        `previously_seen_periods`: None -> altes Verhalten (EXACT_TIMESTAMP
        sobald release_time bekannt, sonst CONSERVATIVE_DATE). Ein Set (auch
        leeres) -> First-Seen-PIT-Präzision (Aufgabenstellung Punkt 4):
        Beobachtungsperioden (time-Dimension-Labels), die schon im Set
        enthalten sind, gelten als historisch (CONSERVATIVE_DATE,
        available_at = release_time oder retrieved_at "wie bisher"); neue
        Perioden (erstmals bei diesem Abruf gesehen) erhalten
        EXACT_TIMESTAMP mit available_at = release_time (<= retrieved_at,
        Datensatz-Update-Zeitstempel) und attrs["first_seen_at"] =
        retrieved_at.isoformat().

        Rückgabe: (observations, latest_obs_time, release_time,
                   parse_failures, diagnostics_dict, all_periods_seen_set).
        observations ist None bei einem Schema, das nicht mal minimal
        auswertbar ist (dann SCHEMA_CHANGED beim Aufrufer). Bei einer echten
        Identitäts-Kollision wird _DuplicateIdentityError geworfen (ebenfalls
        SCHEMA_CHANGED, aber mit einer klaren, spezifischen Nachricht statt
        Stille)."""
        dims = (data.get("dimension") or {})
        ids = data.get("id") or []
        sizes = data.get("size") or []
        values = data.get("value")
        diagnostics: dict = {}
        if not ids or not sizes or values is None or "geo" not in dims or "time" not in dims:
            return None, None, None, 0, diagnostics, set()

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
            return None, None, None, 0, diagnostics, set()

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
        all_periods_seen: set[str] = set()
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
                obs_time = _parse_eurostat_time(time_label)
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

            attrs = dict(dim_parts)
            if previously_seen_periods is None:
                # Altes/Standard-Verhalten (eurostat_road_freight, monatlich/
                # jährlich): EXACT_TIMESTAMP sobald eine offizielle
                # Release-Zeit bekannt ist, sonst CONSERVATIVE_DATE.
                avail_precision = (AvailabilityPrecision.EXACT_TIMESTAMP if release_time
                                    else AvailabilityPrecision.CONSERVATIVE_DATE)
                available_at = release_time or retrieved_at
            elif not previously_seen_periods or time_label in previously_seen_periods:
                # Erster Abruf überhaupt (Backfill: die ganze Historie ist
                # "neu", ihre echte Erstveröffentlichung aber unbekannt) oder
                # Periode war schon bei einem früheren Abruf dieses
                # Datensatzes vorhanden -> historisch/Revision, konservativ.
                avail_precision = AvailabilityPrecision.CONSERVATIVE_DATE
                available_at = release_time or retrieved_at
            else:
                # Periode erscheint zum ERSTEN MAL bei diesem Abruf -> die
                # offizielle Release-Zeit des Datensatzes ist ein belastbarer
                # (<= retrieved_at) Verfügbarkeits-Zeitpunkt.
                avail_precision = AvailabilityPrecision.EXACT_TIMESTAMP
                available_at = release_time or retrieved_at
                attrs["first_seen_at"] = retrieved_at.isoformat()
            all_periods_seen.add(time_label)

            observations.append(Observation(
                source_id=source_id, dataset="road_freight", series_id=series_id,
                entity_id=geo_label, metric=metric,
                value=float(val) if val is not None else None, unit=unit_code or "unknown",
                observation_time=obs_time, available_at=available_at, retrieved_at=retrieved_at,
                availability_precision=avail_precision, parser_version=parser_version,
                attrs=attrs,
            ))
            if latest is None or obs_time > latest:
                latest = obs_time
        return observations, latest, release_time, parse_failures, diagnostics, all_periods_seen


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
# 3b) eurostat_road_freight_quarterly – Eurostat QUARTERLY road_go_* Datensatz
#     (bessere Timing-Auflösung für z-Score/Beschleunigung als der
#     jährliche/gemischte eurostat_road_freight-Datensatz)
# ---------------------------------------------------------------------------

EUROSTAT_SEEN_PERIODS_STATE_PATH = "outputs/external_data/manifests/eurostat_seen_periods.json"
"""Von EurostatRoadFreightQuarterlyConnector geschriebene/gelesene kleine
Zustandsdatei: {"<dataset_code>": ["2023-Q1", "2023-Q2", ...]}. Hält je
Eurostat-Datensatz-Code die Menge der beim jeweils LETZTEN erfolgreichen
Parse bereits gesehenen time-Dimension-Labels (Perioden) fest -- Grundlage
für die First-Seen-PIT-Präzision (siehe _parse_eurostat_jsonstat()):
Perioden, die schon in dieser Datei stehen, gelten bei einem erneuten Abruf
als historisch (CONSERVATIVE_DATE); Perioden, die NICHT drin stehen, gelten
als neu erschienen (EXACT_TIMESTAMP, available_at = Release-Zeit des
Datensatzes). Die Datei wird nach jedem erfolgreichen Parse überschrieben
(Vereinigungsmenge alt+neu). Kein Netzwerk-/Auth-Bezug, rein lokal."""


class EurostatRoadFreightQuarterlyConnector(Connector):
    """Wie EurostatRoadFreightConnector, aber Discovery beschränkt auf einen
    QUARTERLY-Datensatz der road_go_-Familie (Code beginnt mit "road_go_",
    Titel enthält "quarterly", type=="dataset" in der TOC) -- liefert eine
    deutlich feinere Zeitauflösung für Momentum-/Z-Score-Features als der
    jährliche eurostat_road_freight-Datensatz.

    Nutzt denselben generischen JSON-stat-2.0-Parser (_parse_eurostat_jsonstat)
    wie EurostatRoadFreightConnector, aktiviert aber zusätzlich die
    First-Seen-PIT-Präzision (Aufgabenstellung Punkt 4) über eine kleine,
    lokale Zustandsdatei (siehe EUROSTAT_SEEN_PERIODS_STATE_PATH).
    """

    source_id = "eurostat_road_freight_quarterly"
    parser_version = PARSER_VERSION

    TOC_URL = "https://ec.europa.eu/eurostat/api/dissemination/catalogue/toc/txt?lang=en"
    DATA_BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
    COUNTRIES = EurostatRoadFreightConnector.COUNTRIES
    DIMENSION_FILTERS = {"unit": ["THS_T", "MIO_TKM"]}
    DATASET_CODE_RE = re.compile(r"^road_go_")

    def _discover_dataset_code(self, raw: list[RawRecord]) -> tuple[str | None, dict]:
        search_terms = self.cfg.get("search_terms", ["quarterly"])
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
        if expected_code and any(c == expected_code for c, _ in matches):
            chosen = expected_code
        return chosen, {"eurostat_road_freight_quarterly_candidates": dict(matches),
                         "eurostat_road_freight_quarterly_chosen": chosen}

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

    def _state_path(self) -> Path:
        return Path(self.cfg.get("_seen_periods_state_path", EUROSTAT_SEEN_PERIODS_STATE_PATH))

    def preflight(self, now: datetime) -> dict:
        """Live-Check ohne Seiteneffekt: der First-Seen-Zustand wird NUR vom
        archivierenden Abruf fortgeschrieben, sonst würden Perioden, die der
        Preflight zuerst sieht, später fälschlich als 'schon gesehen' gelten."""
        self._dry_run = True
        try:
            return super().preflight(now)
        finally:
            self._dry_run = False

    def _load_seen_periods(self, code: str) -> set[str]:
        path = self._state_path()
        if not path.exists():
            return set()
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return set()
        return set(data.get(code, []))

    def _save_seen_periods(self, code: str, all_periods: set[str]) -> None:
        path = self._state_path()
        try:
            data = json.loads(path.read_text()) if path.exists() else {}
        except (json.JSONDecodeError, OSError):
            data = {}
        data[code] = sorted(all_periods)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, sort_keys=True))

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
                    message="Eurostat-TOC-Discovery (quarterly) fehlgeschlagen und keine "
                            "expected_dataset_code in der Config.",
                    discovered_ids=discovered,
                )
            discovered["fallback_dataset_code_source"] = "config.expected_dataset_code (verify in preflight)"

        res, err = self._fetch_dataset(code, raw)
        if err is not None and "404" in err and expected_code and expected_code != code:
            discovered["eurostat_discovered_code_404"] = code
            code = expected_code
            discovered["eurostat_road_freight_quarterly_chosen"] = code
            res, err = self._fetch_dataset(code, raw)
        if err is not None:
            return ConnectorResult(
                source_id=self.source_id, status=SourceStatus.FAIL, raw=raw,
                message=f"Eurostat-Datenabruf (quarterly) fehlgeschlagen ({code}): {err}",
                discovered_ids=discovered,
            )
        try:
            data = res.json()
        except json.JSONDecodeError:
            return ConnectorResult(
                source_id=self.source_id, status=SourceStatus.SCHEMA_CHANGED, raw=raw,
                message="Eurostat-Antwort (quarterly) kein valides JSON (JSON-stat 2.0 erwartet).",
                discovered_ids=discovered,
            )

        previously_seen = self._load_seen_periods(code)
        configured_filters = self.cfg.get("dimension_filters") or self.DIMENSION_FILTERS
        try:
            observations, latest, release_time, parse_failures, parse_diag, all_periods = _parse_eurostat_jsonstat(
                data, code, res.retrieved_at, configured_filters, source_id=self.source_id,
                parser_version=self.parser_version, previously_seen_periods=previously_seen)
        except _DuplicateIdentityError as e:
            return ConnectorResult(
                source_id=self.source_id, status=SourceStatus.SCHEMA_CHANGED, raw=raw,
                message=f"Eurostat-JSON (quarterly) liefert mehrere Werte für dieselbe "
                        f"Beobachtungs-Identität: {e}",
                discovered_ids=discovered,
            )
        discovered.update(parse_diag)
        if observations is None:
            return ConnectorResult(
                source_id=self.source_id, status=SourceStatus.SCHEMA_CHANGED, raw=raw,
                message="Eurostat-JSON (quarterly) entspricht nicht dem erwarteten "
                        "JSON-stat-2.0-Schema (dimension/value fehlen).",
                discovered_ids=discovered,
            )
        # Zustandsdatei erst NACH erfolgreichem Parse aktualisieren (nie bei
        # SCHEMA_CHANGED/FAIL einen halbgaren Stand persistieren).
        if not getattr(self, "_dry_run", False):   # Preflight: keine Zustandsänderung
            self._save_seen_periods(code, previously_seen | all_periods)
        discovered["eurostat_road_freight_quarterly_new_periods"] = sorted(all_periods - previously_seen)

        status = SourceStatus.WARN if parse_failures else SourceStatus.PASS
        return ConnectorResult(
            source_id=self.source_id, status=status, observations=observations, raw=raw,
            message=f"{len(observations)} Beobachtungen aus Datensatz {code} (quarterly)",
            latest_observation_time=latest, latest_release_time=release_time,
            discovered_ids=discovered, parse_failures=parse_failures,
        )


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
    STATS_CODE = "00600350"   # 自動車輸送統計調査 (MLIT), amtlicher e-Stat-Code

    def fetch(self, now: datetime) -> ConnectorResult:
        # Leerzeichen/Zeilenumbrüche aus dem Secret-Feld entfernen
        app_id = (os.environ.get("ESTAT_APP_ID") or "").strip()
        if not app_id:
            return ConnectorResult(
                source_id=self.source_id, status=SourceStatus.AUTH_MISSING,
                message="ESTAT_APP_ID nicht gesetzt -> kein API-Call ausgeführt.",
            )
        raw: list[RawRecord] = []
        discovered: dict = {}

        list_url = f"{self.BASE}/getStatsList"
        tables = None
        attempts = []
        # 1) amtlicher Statistik-Code des Kfz-Transportsurveys (自動車輸送統計調査,
        #    statsCode 00600350) -- stabiler als Freitextsuche; 2) Suchwort.
        for params in ({"statsCode": self.cfg.get("stats_code", self.STATS_CODE), "limit": 100},
                       {"searchWord": self.SEARCH_WORD, "limit": 100}):
            try:
                res = http.fetch(list_url, params={"appId": app_id, **params})
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
            root = data.get("GET_STATS_LIST") or {}
            result = root.get("RESULT") or {}
            attempts.append({"query": {k_: v for k_, v in params.items()},
                             "status": result.get("STATUS"), "error_msg": result.get("ERROR_MSG")})
            # STATUS 100 = appId ungültig (e-Stat-API-Spezifikation)
            if str(result.get("STATUS")) == "100":
                return ConnectorResult(
                    source_id=self.source_id, status=SourceStatus.AUTH_MISSING, raw=raw,
                    message=f"e-Stat: appId abgelehnt ({result.get('ERROR_MSG')}).",
                    discovered_ids={"estat_jp_truck_attempts": attempts},
                )
            found = (root.get("DATALIST_INF") or {}).get("TABLE_INF")
            if isinstance(found, dict):
                found = [found]
            if found:
                tables = found
                break
        discovered["estat_jp_truck_attempts"] = attempts
        if not tables:
            return ConnectorResult(
                source_id=self.source_id, status=SourceStatus.FAIL, raw=raw,
                message="e-Stat getStatsList ohne Tabellen: "
                        + "; ".join(f"{a['query']} -> {a['status']} {a['error_msg']}" for a in attempts),
                discovered_ids=discovered,
            )

        def _txt(v):
            return v.get("$", "") if isinstance(v, dict) else (v or "")

        catalog = [{"id": t.get("@id"), "title": _txt(t.get("TITLE")),
                    "stat_name": _txt(t.get("STATISTICS_NAME")), "cycle": t.get("CYCLE"),
                    "survey_date": t.get("SURVEY_DATE"), "updated": t.get("UPDATED_DATE")}
                   for t in tables]
        discovered["estat_jp_truck_candidates"] = catalog[:40]
        # Monatstabellen mit Transport-Tonnage bevorzugen; sonst erste Tabelle
        preferred = [c for c in catalog if c["cycle"] == "月次" and "トン" in c["title"]]
        chosen = next((t for t in tables if preferred and t.get("@id") == preferred[0]["id"]), tables[0])
        stats_data_id = chosen.get("@id")
        if not stats_data_id:
            return ConnectorResult(
                source_id=self.source_id, status=SourceStatus.SCHEMA_CHANGED, raw=raw,
                message="Gefundene Tabelle ohne @id -> Schema geändert.",
            )
        discovered["estat_jp_truck_stats_data_id"] = stats_data_id

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
            # ALLE Klassifikationsdimensionen (@tab, @cat01..@cat15) bilden die
            # Serienidentität -- nur @cat01 ließ verschiedene Reihen (z.B. je
            # @tab/@cat02) auf dieselbe Identität kollabieren.
            dims = sorted(k for k in v if k.startswith("@") and k not in ("@time", "@area", "@unit"))
            dim_key = "|".join(f"{k[1:]}={v[k]}" for k in dims)
            metric_name = "|".join(class_names.get(v[k], v[k]) for k in dims) or "value"
            area_code = v.get("@area", "")
            value = None if val_raw in (None, "", "-", "***") else self._safe_float(val_raw)
            observations.append(Observation(
                source_id="estat_jp_truck", dataset="motor_vehicle_transport",
                series_id=stats_data_id, entity_id=area_code or "JP",
                metric=f"jp_truck_{metric_name}", value=value, unit=v.get("@unit") or "see_metric",
                observation_time=obs_time, available_at=retrieved_at, retrieved_at=retrieved_at,
                availability_precision=AvailabilityPrecision.CONSERVATIVE_DATE,
                parser_version=PARSER_VERSION,
                attrs={"cat01": cat_code, "area": area_code, "dims": dim_key},
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
        if len(code) == 10 and code.isdigit():
            # e-Stat-Standard (Live 2026-09-27): YYYY + Kennung(2) + MM(Start) + MM(Ende)
            # z.B. 2024000101 = Jan 2024, 2024000103 = Q1 2024, 2024000000 =
            # Kalenderjahr, 2024100000 = Fiskaljahr 2024 (ab April).
            year, kind, m_start = int(code[:4]), code[4:6], int(code[6:8])
            if 1 <= m_start <= 12:
                return datetime(year, m_start, 1, tzinfo=timezone.utc)
            if code[6:] == "0000":
                return datetime(year, 4 if kind == "10" else 1, 1, tzinfo=timezone.utc)
        raise ValueError(f"Unbekanntes e-Stat-Zeitformat: {code}")


CONNECTORS: dict[str, type[Connector]] = {
    "destatis_truck_toll": DestatisTruckTollConnector,
    "destatis_truck_toll_download": DestatisTruckTollDownloadConnector,
    "bts_freight_tsi": BtsFreightTsiConnector,
    "bts_open_data_tsi": BtsOpenDataTsiConnector,
    "eurostat_road_freight": EurostatRoadFreightConnector,
    "eurostat_road_freight_quarterly": EurostatRoadFreightQuarterlyConnector,
    "estat_jp_truck": EstatJpTruckConnector,
    # fhwa_faf: kein Live-Konnektor (siehe scripts/build_faf_exposure.py) –
    # bewusst nicht in CONNECTORS, Registry-Eintrag hat status_override DEFERRED.
    # viapass_be, asfinag_at, nbs_cn, mot_cn, kosis_kr: Preflight-only-Einträge
    # ohne Konnektor-Klasse (siehe config/external_sources/road_freight.yaml).
}

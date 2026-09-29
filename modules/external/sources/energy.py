"""
modules/external/sources/energy.py – ENTSO-E Transparency Platform (offizielle
REST-API, Token ENTSOE_API_TOKEN).

  entsoe_power:
    - Day-Ahead-Preise (documentType A44) je Gebotszone, Tagesmittel
      (Baseload) des Lieferungstags in Europe/Berlin-Ortszeit, EUR/MWh
    - Tatsächliche Gesamtlast DE (documentType A65, processType A16),
      Tagesmittel in MW

PIT-Regeln:
  * Day-Ahead-Preise für Lieferungstag D werden in der SDAC-Auktion am Vortag
    (~12:45 CET) veröffentlicht und nicht revidiert -> available_at =
    D-1 23:59:59 Europe/Berlin (konservativ), CONSERVATIVE_DATE.
  * Last wird laufend nachgemeldet/revidiert -> available_at =
    max(D+1 06:00 Europe/Berlin, Abrufzeit): eine spätere Revision kann nie
    rückwirkend als "damals bekannt" erscheinen (Forward-Archiv).
  * Keine Credentials in URL/Raw-Metadaten (securityToken nur als Parameter,
    request_fingerprint/redact_secrets filtern ihn).

XML: IEC-62325-Dokumente (Namespace versionsabhängig -> namespace-agnostisch
geparst). curveType A03: fehlende Positionen = Wert der vorigen Position.
Auflösungen PT15M/PT30M/PT60M; je Tag wird die feinste vorhandene genutzt
(SDAC seit 2025-10 15-Minuten, parallele Stundenreihen würden sonst doppelt
zählen). Acknowledgement-Dokument (keine Daten) -> leer, nie 0.
"""

from __future__ import annotations

import logging
import os
import statistics
import xml.etree.ElementTree as ET
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from modules.external import http
from modules.external.pit import AvailabilityPrecision, Observation, ensure_utc
from modules.external.sources.base import Connector, ConnectorResult, RawRecord, SourceStatus

log = logging.getLogger(__name__)

BERLIN = ZoneInfo("Europe/Berlin")
API_URL = "https://web-api.tp.entsoe.eu/api"
PARSER_VERSION = "1"

PRICE_ZONES = {                       # EIC-Codes der Gebotszonen
    "DE_LU": "10Y1001A1001A82H",
    "FR": "10YFR-RTE------C",
    "NL": "10YNL----------L",
    "IT_NORD": "10Y1001A1001A73I",
    "ES": "10YES-REE------0",
}
LOAD_AREAS = {"DE": "10Y1001A1001A83F"}
RES_MINUTES = {"PT15M": 15, "PT30M": 30, "PT60M": 60}


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _find(el, name):
    for c in el:
        if _local(c.tag) == name:
            return c
    return None


def _findall(el, name):
    return [c for c in el if _local(c.tag) == name]


def _parse_dt(s: str) -> datetime:
    s = s.strip().replace("Z", "+00:00")
    return datetime.fromisoformat(s).astimezone(timezone.utc)


def parse_timeseries_points(xml_text: str, value_tag: str) -> list[tuple[datetime, int, float]]:
    """-> [(Start-UTC des Intervalls, Auflösung in Minuten, Wert)].
    Acknowledgement-Dokumente (keine Daten) -> []."""
    root = ET.fromstring(xml_text)
    if _local(root.tag).startswith("Acknowledgement"):
        return []
    out = []
    for ts in _findall(root, "TimeSeries"):
        curve = (_find(ts, "curveType").text if _find(ts, "curveType") is not None else "A01").strip()
        for period in _findall(ts, "Period"):
            ti = _find(period, "timeInterval")
            res = _find(period, "resolution").text.strip()
            minutes = RES_MINUTES.get(res)
            if ti is None or minutes is None:
                continue
            start, end = _parse_dt(_find(ti, "start").text), _parse_dt(_find(ti, "end").text)
            n = int((end - start).total_seconds() // (minutes * 60))
            given = {}
            for pt in _findall(period, "Point"):
                pos = int(_find(pt, "position").text)
                v = _find(pt, value_tag)
                if v is not None and v.text not in (None, ""):
                    given[pos] = float(v.text)
            last = None
            for pos in range(1, n + 1):
                if pos in given:
                    last = given[pos]
                elif curve != "A03":
                    continue                      # A01: fehlende Position = fehlend (nie 0)
                if last is not None:
                    out.append((start + timedelta(minutes=minutes * (pos - 1)), minutes, last))
    return out


def daily_means(points: list[tuple[datetime, int, float]]) -> dict:
    """Tagesmittel je Lieferungstag (Europe/Berlin), feinste Auflösung je Tag.
    Nur vollständige Tage (>= 90 % der erwarteten Intervalle)."""
    by_day: dict = {}
    for t, minutes, v in points:
        d = t.astimezone(BERLIN).date()
        by_day.setdefault(d, {}).setdefault(minutes, {})[t] = v
    out = {}
    for d, per_res in by_day.items():
        m = min(per_res)
        vals = list(per_res[m].values())
        day_start = datetime.combine(d, time(0), BERLIN)
        day_len_h = ((datetime.combine(d + timedelta(days=1), time(0), BERLIN) - day_start)
                     .total_seconds() / 3600)                     # 23/24/25 h (Zeitumstellung)
        expected = day_len_h * 60 / m
        if len(vals) >= 0.9 * expected:
            out[d] = (statistics.fmean(vals), len(vals), m)
    return out


def _raw(source_id: str, dataset: str, res: http.FetchResult) -> RawRecord:
    return RawRecord(source_id=source_id, dataset=dataset, url=res.url, fingerprint=res.fingerprint,
                     retrieved_at=res.retrieved_at, status_code=res.status, content_type=res.content_type,
                     content_hash=res.content_hash, bytes=res.bytes)


def _windows(start: datetime, end: datetime, max_days: int = 360):
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=max_days), end)
        yield cur, nxt
        cur = nxt


class EntsoePowerConnector(Connector):
    source_id = "entsoe_power"
    parser_version = PARSER_VERSION

    def _fetch_xml(self, params: dict, token: str, raw: list, dataset: str) -> str | None:
        res = http.fetch(API_URL, params={**params, "securityToken": token}, timeout=60)
        raw.append(_raw(self.source_id, dataset, res))
        return res.content.decode("utf-8", errors="replace")

    def fetch(self, now: datetime) -> ConnectorResult:
        token = os.environ.get(self.cfg.get("auth_env_variable") or "ENTSOE_API_TOKEN", "").strip()
        if not token:
            return ConnectorResult(source_id=self.source_id, status=SourceStatus.AUTH_MISSING,
                                   message="ENTSOE_API_TOKEN nicht gesetzt -> kein API-Call.")
        now = ensure_utc(now)
        lookback = int(self.cfg.get("lookback_days", 400))
        end = datetime.combine(now.astimezone(BERLIN).date() + timedelta(days=2), time(0), BERLIN) \
            .astimezone(timezone.utc)
        start = end - timedelta(days=lookback)
        raw: list[RawRecord] = []
        obs: list[Observation] = []
        failures, empty = [], []
        zones = self.cfg.get("price_zones") or list(PRICE_ZONES)
        for zone in zones:
            eic = PRICE_ZONES[zone]
            pts = []
            try:
                for a, b in _windows(start, end):
                    xml = self._fetch_xml({"documentType": "A44", "in_Domain": eic, "out_Domain": eic,
                                           "periodStart": a.strftime("%Y%m%d%H%M"),
                                           "periodEnd": b.strftime("%Y%m%d%H%M")},
                                          token, raw, f"da_price_{zone}")
                    pts += parse_timeseries_points(xml, "price.amount")
            except http.AuthError:
                return ConnectorResult(source_id=self.source_id, status=SourceStatus.AUTH_MISSING, raw=raw,
                                       message="ENTSO-E lehnt das Token ab (401/403).")
            except (http.FetchError, ET.ParseError, ValueError) as e:
                failures.append(f"{zone}: {http.redact_secrets(str(e))[:120]}")
                continue
            days = daily_means(pts)
            if not days:
                empty.append(zone)
            for d, (mean, n, m) in days.items():
                period = datetime.combine(d, time(0), timezone.utc)
                avail = datetime.combine(d - timedelta(days=1), time(23, 59, 59), BERLIN).astimezone(timezone.utc)
                if avail > now:
                    avail = now      # heute abgerufene Auktion für morgen: nie in der Zukunft
                obs.append(Observation(
                    source_id=self.source_id, dataset="day_ahead_price", series_id=f"A44:{eic}",
                    entity_id=zone, metric="da_price_daily_mean", value=round(mean, 4), unit="EUR_per_MWh",
                    observation_time=period, available_at=avail, retrieved_at=now,
                    availability_precision=AvailabilityPrecision.CONSERVATIVE_DATE,
                    parser_version=self.parser_version,
                    attrs={"n_intervals": n, "resolution_min": m, "delivery_day_tz": "Europe/Berlin"}))
        for area, eic in (self.cfg.get("load_areas") or LOAD_AREAS).items():
            pts = []
            try:
                for a, b in _windows(start, end):
                    xml = self._fetch_xml({"documentType": "A65", "processType": "A16",
                                           "outBiddingZone_Domain": eic,
                                           "periodStart": a.strftime("%Y%m%d%H%M"),
                                           "periodEnd": b.strftime("%Y%m%d%H%M")},
                                          token, raw, f"actual_load_{area}")
                    pts += parse_timeseries_points(xml, "quantity")
            except http.AuthError:
                return ConnectorResult(source_id=self.source_id, status=SourceStatus.AUTH_MISSING, raw=raw,
                                       message="ENTSO-E lehnt das Token ab (401/403).")
            except (http.FetchError, ET.ParseError, ValueError) as e:
                failures.append(f"load_{area}: {http.redact_secrets(str(e))[:120]}")
                continue
            days = daily_means(pts)
            if not days:
                empty.append(f"load_{area}")
            for d, (mean, n, m) in days.items():
                period = datetime.combine(d, time(0), timezone.utc)
                avail = max(datetime.combine(d + timedelta(days=1), time(6), BERLIN).astimezone(timezone.utc), now)
                obs.append(Observation(
                    source_id=self.source_id, dataset="actual_load", series_id=f"A65:A16:{eic}",
                    entity_id=area, metric="load_daily_mean", value=round(mean, 1), unit="MW",
                    observation_time=period, available_at=avail, retrieved_at=now,
                    availability_precision=AvailabilityPrecision.CONSERVATIVE_DATE,
                    parser_version=self.parser_version,
                    attrs={"n_intervals": n, "resolution_min": m, "delivery_day_tz": "Europe/Berlin"}))
        latest = max((o.observation_time for o in obs), default=None)
        if failures and not obs:
            status = SourceStatus.FAIL
        elif failures or empty:
            status = SourceStatus.WARN
        else:
            status = SourceStatus.PASS
        msg = f"{len(obs)} Tageswerte ({len(zones)} Preiszonen, Last {list((self.cfg.get('load_areas') or LOAD_AREAS))})"
        if failures:
            msg += f" | Fehler: {'; '.join(failures)}"
        if empty:
            msg += f" | ohne Daten: {empty}"
        return ConnectorResult(source_id=self.source_id, status=status, observations=obs, raw=raw,
                               message=msg, latest_observation_time=latest,
                               parse_failures=len(failures))


CONNECTORS = {"entsoe_power": EntsoePowerConnector}

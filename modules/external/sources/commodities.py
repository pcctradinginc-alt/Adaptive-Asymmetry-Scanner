"""
modules/external/sources/commodities.py – Commodity Intelligence (RESEARCH/SHADOW).

Nur offizielle, kostenlose Quellen (config/commodity_intelligence.yaml):
  eia_petroleum_weekly  EIA Open Data API v2 – Weekly Petroleum Status Report + Ölspots (Crosscheck)
  eia_natural_gas       EIA Open Data API v2 – Speicher (wöchentlich), Henry Hub (Crosscheck),
                        Trockengas-Produktion und LNG-Exporte (monatlich, revidiert)
  fred_commodities      FRED/ALFRED – Brent, Henry Hub, Kupfer, Weizen, Mais, Sojabohnen (Vintages);
                        WTI wird aus fred_regime_macro WIEDERVERWENDET (kein Doppelabruf)
  cftc_cot              CFTC Public Reporting – Disaggregated COT (Futures Only), kein Key

PIT-Regeln:
  * EIA/CFTC: available_at = min(konservative Release-Regel, retrieved_at). Die Regel liegt nie vor
    der echten Veröffentlichung (EIA WPSR Mi, Gasspeicher Do, COT Fr 15:30 ET bzw. Mo nach
    Feiertagswochen); sie liegt nie nach dem Abruf (PIT-Integrität available_at <= retrieved_at).
  * Revisionen, die NACH dem ersten archivierten Wert auftauchen, gelten erst ab dem Abruf
    (available_at = vintage_time = retrieved_at) – nie rückwirkend (kein Restatement-Bias).
  * Historischer Erstimport revidierter Reihen = heutiger Stand, nicht Erstveröffentlichung ->
    attrs.revision_status = "backfill_latest_vintage" (sichtbar, Features markieren es).
  * FRED: ALFRED realtime_start (EXACT_DATE, Revisionen als eigene Vintages).
  * COT-Reports in Government-Shutdown-Fenstern (verspätet veröffentlicht) -> ausgeschlossen.

Fehlerverhalten: fehlender Key -> AUTH_MISSING (kein Call); Einheiten-/Feldänderung ->
SCHEMA_CHANGED für die Serie (nie still umrechnen); unsicheres COT-Mapping -> Markt UNAVAILABLE.
Keine Quelle füllt Lücken mit 0.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, time, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import yaml

from modules.external import http
from modules.external.pit import AvailabilityPrecision, Observation, ensure_utc
from modules.external.sources.base import Connector, ConnectorResult, RawRecord, SourceStatus
from modules.external.sources.real_economy import FredUsMacroConnector

log = logging.getLogger(__name__)

CONFIG_PATH = Path("config/commodity_intelligence.yaml")
PARSER_VERSION = "1"
EIA_SCHEMA_VERSION = "eia-api-v2:response.data[period,series,value,units]"
CFTC_SCHEMA_VERSION = "cftc-pre-72hh-3qpy:disaggregated-futures-only"
DEFAULT_ARCHIVE_ROOT = "outputs/external_data"
BACKFILL_FLAG_DAYS = 30          # Erstimport älter als das -> kein Erstveröffentlichungswert belegt


@lru_cache(maxsize=4)
def _load_config_cached(path: str, mtime: float) -> dict:
    return yaml.safe_load(Path(path).read_text()) or {}


def load_config(path: str | Path | None = None) -> dict:
    p = Path(path or CONFIG_PATH)
    if not p.exists():
        return {}
    return _load_config_cached(str(p), p.stat().st_mtime)


def _raw(source_id: str, dataset: str, res: http.FetchResult) -> RawRecord:
    return RawRecord(source_id=source_id, dataset=dataset, url=res.url, fingerprint=res.fingerprint,
                     retrieved_at=res.retrieved_at, status_code=res.status, content_type=res.content_type,
                     content_hash=res.content_hash, bytes=res.bytes)


def _utc_day(d: date, hour: int = 0) -> datetime:
    return datetime.combine(d, time(hour), timezone.utc)


def _float(v) -> float | None:
    if v is None or v == "" or v == ".":
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x else None          # NaN -> None


# ── Release-Regeln ────────────────────────────────────────────────────────────

def parse_period(period: str, frequency: str) -> datetime:
    """EIA-Periode -> UTC-Datum: täglich/wöchentlich 'YYYY-MM-DD', monatlich 'YYYY-MM'."""
    s = str(period).strip()
    if frequency == "monthly":
        y, m = s[:7].split("-")
        return datetime(int(y), int(m), 1, tzinfo=timezone.utc)
    return datetime.strptime(s[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)


def release_time(period: datetime, rule: dict) -> datetime:
    """Konservativer Veröffentlichungszeitpunkt einer Periode (vor Kappung auf retrieved_at)."""
    days = int(rule.get("days", 0))
    hour = int(rule.get("hour_utc", 16))
    return _utc_day(period.date() + timedelta(days=days), hour)


def available_at(period: datetime, rule: dict, retrieved_at: datetime) -> datetime:
    return min(release_time(period, rule), ensure_utc(retrieved_at))


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    d += timedelta(days=(weekday - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    d = (date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1))
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def _observed(d: date) -> date:
    return d - timedelta(days=1) if d.weekday() == 5 else d + timedelta(days=1) if d.weekday() == 6 else d


def us_federal_holidays(year: int) -> set[date]:
    """US-Bundesfeiertage (OPM-Regeln inkl. Ersatztag); Juneteenth ab 2021."""
    h = {_observed(date(year, 1, 1)), _nth_weekday(year, 1, 0, 3), _nth_weekday(year, 2, 0, 3),
         _last_weekday(year, 5, 0), _observed(date(year, 7, 4)), _nth_weekday(year, 9, 0, 1),
         _nth_weekday(year, 10, 0, 2), _observed(date(year, 11, 11)), _nth_weekday(year, 11, 3, 4),
         _observed(date(year, 12, 25))}
    if year >= 2021:
        h.add(_observed(date(year, 6, 19)))
    # Neujahr des Folgejahres kann auf den 31.12. fallen
    nxt = date(year + 1, 1, 1)
    if nxt.weekday() == 5:
        h.add(date(year, 12, 31))
    return h


def cot_release_time(report_date: date, rule: dict) -> datetime:
    """COT-Stichtag Dienstag -> Veröffentlichung Freitag 15:30 ET. Liegt in der Woche (Mo–Fr) ein
    Bundesfeiertag, verschiebt die CFTC auf den folgenden Montag -> konservativ +6 T.
    hour_utc 21 liegt nach 15:30 ET (19:30/20:30 UTC) – nie zu früh."""
    monday = report_date - timedelta(days=report_date.weekday())
    week = {monday + timedelta(days=i) for i in range(5)}
    hol = us_federal_holidays(report_date.year) | us_federal_holidays(report_date.year - 1)
    offset = int(rule.get("holiday_offset_days", 6)) if week & hol else int(rule.get("weekday_offset_days", 3))
    return _utc_day(report_date + timedelta(days=offset), int(rule.get("hour_utc", 21)))


def expected_latest_period(frequency: str, rule: dict, now: datetime) -> datetime | None:
    """Jüngste Periode, deren konservative Veröffentlichung <= now liegt (Fälligkeitsprüfung)."""
    now = ensure_utc(now)
    if frequency == "weekly":
        d = now.date()
        d -= timedelta(days=(d.weekday() - 4) % 7)          # letzter Freitag (EIA-Wochenende)
        for _ in range(4):
            p = _utc_day(d)
            if release_time(p, rule) <= now:
                return p
            d -= timedelta(days=7)
        return None
    if frequency == "daily":
        d = (now - timedelta(days=int(rule.get("days", 0)) + 1)).date()
        while d.weekday() >= 5:
            d -= timedelta(days=1)
        return _utc_day(d)
    if frequency == "monthly":
        y, m = now.year, now.month
        for _ in range(12):
            p = datetime(y, m, 1, tzinfo=timezone.utc)
            if release_time(p, rule) <= now:
                return p
            y, m = (y - 1, 12) if m == 1 else (y, m - 1)
    return None


# ── Archiv-bewusste Basis (Revisionen nie rückwirkend) ───────────────────────

class _ArchiveAware(Connector):
    parser_version = PARSER_VERSION

    def _archive(self):
        from modules.external.archive import ExternalArchive
        return ExternalArchive(self.cfg.get("_archive_root") or DEFAULT_ARCHIVE_ROOT)

    def _archived(self) -> dict[str, Observation]:
        """identity_key -> jüngste archivierte Version."""
        try:
            rows = self._archive().load(self.source_id)
        except Exception as e:  # noqa: BLE001 - Archiv nicht lesbar -> wie Erstimport, aber geloggt
            log.warning("%s: Archiv nicht lesbar (%r) – Revisionserkennung ohne Historie", self.source_id, e)
            return {}
        out: dict[str, Observation] = {}
        for o in rows:
            k = o.identity_key()
            cur = out.get(k)
            if cur is None or (o.vintage_time or o.available_at) > (cur.vintage_time or cur.available_at):
                out[k] = o
        return out

    @staticmethod
    def latest_by_series(archived: dict[str, Observation]) -> dict[tuple[str, str], datetime]:
        out: dict[tuple[str, str], datetime] = {}
        for o in archived.values():
            k = (o.series_id, o.entity_id)
            if k not in out or o.observation_time > out[k]:
                out[k] = o.observation_time
        return out

    @staticmethod
    def pit_adjust(obs: list[Observation], archived: dict[str, Observation], now: datetime,
                   revised_series: bool) -> int:
        """Revision nach dem ersten archivierten Wert -> available_at = vintage_time = Abruf.
        Erstimport alter Perioden revidierter Reihen -> revision_status backfill_latest_vintage."""
        n_rev = 0
        for o in obs:
            prior = archived.get(o.identity_key())
            if prior is not None:
                if prior.value != o.value:
                    o.available_at = now
                    o.vintage_time = now
                    o.attrs["revision_status"] = "revised_after_first_seen"
                    o.attrs["previous_value"] = prior.value
                    n_rev += 1
                else:
                    o.available_at = prior.available_at
                    o.vintage_time = prior.vintage_time
                continue
            if (now - o.available_at).days > BACKFILL_FLAG_DAYS:
                o.attrs["revision_status"] = ("backfill_latest_vintage" if revised_series
                                              else "backfill_unrevised_assumed")
            else:
                o.attrs.setdefault("revision_status", "first_seen")
        return n_rev


# ── EIA Open Data API v2 ─────────────────────────────────────────────────────

def eia_key() -> str:
    cfg = load_config().get("eia", {})
    for k in cfg.get("env_keys") or ["EIA_API_KEY", "EIA_KEY"]:
        v = os.environ.get(k, "").strip()
        if v:
            return v
    return ""


class EiaConnector(_ArchiveAware):
    """EIA v2 /data-Abfragen je Serie (facets[series][]), Einheiten werden gegen die Config
    geprüft (Abweichung -> SCHEMA_CHANGED, nie Umrechnung)."""

    source_id = ""
    PAGE = 5000

    def series_specs(self) -> list[dict]:
        cfg = self.cfg.get("_commodity_cfg") or load_config()
        return [s for s in (cfg.get("eia", {}).get("series") or []) if s.get("source_id") == self.source_id]

    def is_due(self, now: datetime) -> tuple[bool, str]:
        """Kein Abruf, solange jede Serie die jüngste bereits veröffentlichte Periode im Archiv hat."""
        archived = self.latest_by_series(self._archived())
        missing = []
        for s in self.series_specs():
            exp = expected_latest_period(s["frequency"], s["release"], now)
            have = archived.get((s["series"], "US"))
            if exp is None or have is None or have < exp:
                missing.append(s["series"])
        if missing:
            return True, f"fällig: {missing[:5]}"
        return False, "alle Serien auf dem Stand der letzten Veröffentlichung"

    def _fetch_series(self, spec: dict, key: str, start: str, raw: list) -> tuple[list[dict], str | None]:
        cfg = (self.cfg.get("_commodity_cfg") or load_config()).get("eia", {})
        url = f"{cfg.get('api_base', 'https://api.eia.gov/v2').rstrip('/')}/{spec['route'].strip('/')}/data/"
        rows: list[dict] = []
        offset = 0
        while True:
            params = {"api_key": key, "frequency": spec["frequency"], "data[0]": "value",
                      "facets[series][]": spec["series"], "start": start,
                      "sort[0][column]": "period", "sort[0][direction]": "desc",
                      "offset": offset, "length": self.PAGE}
            res = http.fetch(url, params=params, timeout=60)
            raw.append(_raw(self.source_id, spec["key"], res))
            try:
                body = res.json()
            except ValueError:
                return [], "keine JSON-Antwort"
            resp = body.get("response") if isinstance(body, dict) else None
            if not isinstance(resp, dict) or not isinstance(resp.get("data"), list):
                return [], f"Schema: 'response.data' fehlt ({str(body)[:120]})"
            page = resp["data"]
            rows += page
            total = _float(resp.get("total"))
            offset += len(page)
            if not page or len(page) < self.PAGE or (total is not None and offset >= total):
                break
        return rows, None

    def parse_rows(self, spec: dict, rows: list[dict], now: datetime) -> tuple[list[Observation], list[str]]:
        problems: list[str] = []
        units_ok = {str(u).strip().lower() for u in spec.get("units") or []}
        seen: dict[datetime, float | None] = {}
        out: list[Observation] = []
        for r in rows:
            if str(r.get("series", spec["series"])) != spec["series"]:
                continue
            unit = str(r.get("units") or r.get("unit") or "").strip()
            if units_ok and unit.lower() not in units_ok:
                return [], [f"{spec['series']}: Einheit '{unit}' statt {sorted(units_ok)} (SCHEMA_CHANGED)"]
            try:
                period = parse_period(r["period"], spec["frequency"])
            except (KeyError, ValueError):
                problems.append(f"{spec['series']}: Periode unlesbar")
                continue
            value = _float(r.get("value"))
            if period in seen:
                if seen[period] != value:
                    problems.append(f"{spec['series']}: widersprüchliches Duplikat {period.date()}")
                continue
            seen[period] = value
            out.append(Observation(
                source_id=self.source_id, dataset=spec["key"], series_id=spec["series"], entity_id="US",
                metric=spec["metric"], value=value, unit=spec["unit"], observation_time=period,
                available_at=available_at(period, spec["release"], now), retrieved_at=now,
                availability_precision=AvailabilityPrecision.CONSERVATIVE_DATE,
                parser_version=self.parser_version, source_release_time=None,
                attrs={"frequency": spec["frequency"], "source_unit": unit, "commodity": spec["commodity"],
                       "role": spec.get("role", "fundamental"), "route": spec["route"],
                       "release_rule": dict(spec["release"]), "schema_version": EIA_SCHEMA_VERSION,
                       "release_time_rule": release_time(period, spec["release"]).isoformat()}))
        return out, problems

    def fetch(self, now: datetime) -> ConnectorResult:
        key = eia_key()
        if not key:
            return ConnectorResult(source_id=self.source_id, status=SourceStatus.AUTH_MISSING,
                                   message="EIA_API_KEY/EIA_KEY nicht gesetzt -> kein API-Call.")
        now = ensure_utc(now)
        cfg = (self.cfg.get("_commodity_cfg") or load_config()).get("eia", {})
        archived = self._archived()
        latest = self.latest_by_series(archived)
        raw: list[RawRecord] = []
        obs: list[Observation] = []
        failures, schema, unavailable = [], [], []
        n_rev = 0
        for spec in self.series_specs():
            have = latest.get((spec["series"], "US"))
            start_dt = (have - timedelta(days=int(cfg.get("incremental_days", 120)))) if have else None
            start = (start_dt.strftime("%Y-%m-%d") if start_dt else str(cfg.get("backfill_start", "2015-01-01")))
            if spec["frequency"] == "monthly":
                start = start[:7]
            try:
                rows, err = self._fetch_series(spec, key, start, raw)
            except http.AuthError:
                return ConnectorResult(source_id=self.source_id, status=SourceStatus.AUTH_MISSING, raw=raw,
                                       message="EIA lehnt den API-Key ab (401/403).")
            except http.FetchError as e:
                failures.append(f"{spec['series']}: {http.redact_secrets(str(e))[:120]}")
                continue
            if err:
                schema.append(f"{spec['series']}: {err}")
                continue
            parsed, problems = self.parse_rows(spec, rows, now)
            if problems and not parsed:
                schema += problems
                continue
            failures += problems
            if not parsed:
                unavailable.append(spec["series"])
                continue
            n_rev += self.pit_adjust(parsed, archived, now,
                                     revised_series=spec.get("revision_status") == "revised_monthly")
            obs += parsed
        latest_obs = max((o.observation_time for o in obs), default=None)
        if schema and not obs:
            status = SourceStatus.SCHEMA_CHANGED
        elif failures and not obs:
            status = SourceStatus.FAIL
        elif schema or failures or unavailable:
            status = SourceStatus.WARN
        else:
            status = SourceStatus.PASS
        msg = f"{len(obs)} EIA-Werte, {n_rev} Revisionen (ab Abruf gültig)"
        for label, lst in (("Schema", schema), ("Fehler", failures), ("ohne Daten", unavailable)):
            if lst:
                msg += f" | {label}: {'; '.join(map(str, lst))[:300]}"
        return ConnectorResult(source_id=self.source_id, status=status, observations=obs, raw=raw, message=msg,
                               latest_observation_time=latest_obs, parse_failures=len(failures) + len(schema),
                               discovered_ids={"schema_changed": schema, "unavailable": unavailable,
                                               "revisions": n_rev})


class EiaPetroleumWeeklyConnector(EiaConnector):
    source_id = "eia_petroleum_weekly"


class EiaNaturalGasConnector(EiaConnector):
    source_id = "eia_natural_gas"


# ── FRED/ALFRED (Erweiterung des bestehenden Konnektors) ──────────────────────

class FredCommodityConnector(FredUsMacroConnector):
    """ALFRED-Vintages für Rohstoffpreise aus config/commodity_intelligence.yaml (fred.series).
    Fehler einer Serie stoppen die übrigen nicht (WARN statt Totalausfall)."""

    source_id = "fred_commodities"
    OBSERVATION_START = "2015-01-01"
    LABEL = "Brent/HenryHub/Kupfer/Weizen/Mais/Soja"

    def __init__(self, source_cfg: dict | None = None):
        super().__init__(source_cfg)
        cfg = (self.cfg.get("_commodity_cfg") or load_config()).get("fred", {})
        self.SERIES = {s["series"]: {"metric": s["metric"], "dataset": s.get("frequency", "daily"),
                                     "unit": s["unit"], "search_text": s["search_text"]}
                       for s in cfg.get("series") or []}

    def fetch(self, now: datetime) -> ConnectorResult:
        api_key = (os.environ.get("FRED_API_KEY") or os.environ.get("FRED_KEY") or "").strip()
        if not api_key:
            return ConnectorResult(source_id=self.source_id, status=SourceStatus.AUTH_MISSING,
                                   message="FRED_API_KEY/FRED_KEY nicht gesetzt -> kein API-Call.")
        raw: list[RawRecord] = []
        discovered: dict = {}
        obs: list[Observation] = []
        errors: list[str] = []
        latest = None
        pf = 0
        for key in self.SERIES:
            o, lt, p, err = self._fetch_series(key, api_key, raw, discovered)
            if err is not None:
                if err.status == SourceStatus.AUTH_MISSING:
                    err.raw, err.discovered_ids = raw, discovered
                    return err
                errors.append(f"{key}: {err.status.value} {err.message[:100]}")
                continue
            for x in o:
                x.attrs["frequency"] = self.SERIES[key]["dataset"]
            obs += o
            pf += p
            if lt is not None and (latest is None or lt > latest):
                latest = lt
        if not obs:
            status = SourceStatus.SCHEMA_CHANGED if any("SCHEMA" in e for e in errors) else SourceStatus.FAIL
        else:
            status = SourceStatus.WARN if (errors or pf) else SourceStatus.PASS
        msg = f"{len(obs)} ALFRED-Vintage-Beobachtungen ({self.LABEL})"
        if errors:
            msg += " | " + "; ".join(errors)[:300]
        return ConnectorResult(source_id=self.source_id, status=status, observations=obs, raw=raw, message=msg,
                               latest_observation_time=latest, discovered_ids=discovered, parse_failures=pf)


# ── CFTC Commitments of Traders (Disaggregated, Futures Only) ─────────────────

COT_POSITION_FIELDS = ("producer_merchant_long", "producer_merchant_short", "swap_long", "swap_short",
                       "managed_money_long", "managed_money_short", "managed_money_spreading",
                       "other_reportables_long", "other_reportables_short")
COT_LONG_SIDE = ("producer_merchant_long", "swap_long", "managed_money_long", "managed_money_spreading",
                 "other_reportables_long")
COT_SHORT_SIDE = ("producer_merchant_short", "swap_short", "managed_money_short", "managed_money_spreading",
                  "other_reportables_short")


def oi_consistent(vals: dict) -> tuple[bool, str]:
    """Positionen <= Open Interest; Summe jeder Seite (ohne Non-Reportables) <= OI; nie negativ."""
    oi = vals.get("total_open_interest")
    if oi is None or oi <= 0:
        return False, "OI fehlt/<=0"
    for k in COT_POSITION_FIELDS:
        v = vals.get(k)
        if v is not None and (v < 0 or v > oi):
            return False, f"{k}={v} ausserhalb [0, OI={oi}]"
    for side, keys in (("long", COT_LONG_SIDE), ("short", COT_SHORT_SIDE)):
        s = sum(vals.get(k) or 0 for k in keys)
        if s > oi * 1.0001:
            return False, f"Summe {side}={s} > OI={oi}"
    return True, ""


def in_windows(d: date, windows: list) -> bool:
    for a, b in windows or []:
        if date.fromisoformat(str(a)) <= d <= date.fromisoformat(str(b)):
            return True
    return False


class CftcCotConnector(_ArchiveAware):
    source_id = "cftc_cot"
    PAGE = 50000

    def _ccfg(self) -> dict:
        return (self.cfg.get("_commodity_cfg") or load_config()).get("cftc", {})

    def expected_report_date(self, now: datetime) -> date | None:
        rule = self._ccfg().get("release", {})
        d = ensure_utc(now).date()
        d -= timedelta(days=(d.weekday() - 1) % 7)          # letzter Dienstag
        for _ in range(4):
            if cot_release_time(d, rule) <= ensure_utc(now):
                return d
            d -= timedelta(days=7)
        return None

    def is_due(self, now: datetime) -> tuple[bool, str]:
        exp = self.expected_report_date(now)
        latest = self.latest_by_series(self._archived())
        codes = {m["code"] for m in self._ccfg().get("markets", {}).values()}
        have = [latest.get((c, k)) for k, c in ((k, m["code"]) for k, m in self._ccfg().get("markets", {}).items())]
        if exp is None or not codes or any(h is None or h.date() < exp for h in have):
            return True, f"fällig: Report {exp}"
        return False, f"Report {exp} bereits archiviert"

    def _query(self, codes: list[str], start: str, raw: list) -> list[dict]:
        c = self._ccfg()
        f = c.get("fields", {})
        code_list = ",".join(f"'{x}'" for x in codes)
        rows: list[dict] = []
        offset = 0
        while True:
            params = {"$where": f"{f['code']} in ({code_list}) AND {f['report_date']} >= '{start}T00:00:00'",
                      "$order": f"{f['report_date']} DESC", "$limit": self.PAGE, "$offset": offset}
            res = http.fetch(c["endpoint"], params=params, timeout=90)
            raw.append(_raw(self.source_id, "disaggregated_futures_only", res))
            page = res.json()
            if not isinstance(page, list):
                raise http.SchemaError(f"Antwort keine Liste: {str(page)[:120]}")
            rows += page
            offset += len(page)
            if len(page) < self.PAGE:
                break
        return rows

    def parse_rows(self, rows: list[dict], now: datetime) -> tuple[list[Observation], dict]:
        c = self._ccfg()
        f = c.get("fields", {})
        rule = c.get("release", {})
        windows = c.get("unavailable_report_windows") or []
        stats = {"unavailable_markets": {}, "oi_inconsistent": [], "shutdown_excluded": 0,
                 "non_tuesday": 0, "schema_missing_fields": [], "conflicts": 0}
        if rows:
            missing = sorted({v for v in f.values() if v not in rows[0]})
            if missing:
                stats["schema_missing_fields"] = missing
                return [], stats
        by_code: dict[str, list[dict]] = {}
        for r in rows:
            by_code.setdefault(str(r.get(f["code"], "")).strip(), []).append(r)
        out: list[Observation] = []
        for mkey, m in (c.get("markets") or {}).items():
            mrows = by_code.get(m["code"], [])
            if not mrows:
                stats["unavailable_markets"][mkey] = "keine Zeilen (Markt fehlt)"
                continue
            names = {str(r.get(f["market_name"], "")).upper() for r in mrows}
            bad = [n for n in names if not all(s.upper() in n for s in m.get("name_contains", []))]
            if bad:
                stats["unavailable_markets"][mkey] = f"Name passt nicht zum Code: {bad[0][:80]}"
                continue                                         # Mapping unsicher -> UNAVAILABLE
            seen: dict[date, dict] = {}
            for r in mrows:
                try:
                    rd = datetime.fromisoformat(str(r[f["report_date"]])[:10]).date()
                except (KeyError, ValueError):
                    continue
                if in_windows(rd, windows):
                    stats["shutdown_excluded"] += 1
                    continue
                vals = {k: _float(r.get(f[k])) for k in ("total_open_interest",) + COT_POSITION_FIELDS}
                if rd in seen:
                    if seen[rd] != vals:
                        stats["conflicts"] += 1
                    continue
                seen[rd] = vals
                ok, why = oi_consistent(vals)
                if not ok:
                    stats["oi_inconsistent"].append(f"{mkey} {rd}: {why}")
                    continue
                if rd.weekday() != 1:
                    stats["non_tuesday"] += 1
                rel = cot_release_time(rd, rule)
                avail = min(rel, ensure_utc(now))
                for k, v in vals.items():
                    out.append(Observation(
                        source_id=self.source_id, dataset="disaggregated_futures_only", series_id=m["code"],
                        entity_id=mkey, metric=f"cot_{k}", value=v, unit="contracts",
                        observation_time=_utc_day(rd), available_at=avail, retrieved_at=now,
                        availability_precision=AvailabilityPrecision.CONSERVATIVE_DATE,
                        parser_version=self.parser_version, source_release_time=None,
                        attrs={"frequency": "weekly", "commodity": m.get("commodity", mkey),
                               "report_date": rd.isoformat(), "release_rule_time": rel.isoformat(),
                               "schema_version": CFTC_SCHEMA_VERSION}))
        return out, stats

    def fetch(self, now: datetime) -> ConnectorResult:
        now = ensure_utc(now)
        c = self._ccfg()
        markets = c.get("markets") or {}
        archived = self._archived()
        latest = self.latest_by_series(archived)
        have = [latest.get((m["code"], k)) for k, m in markets.items()]
        if have and all(h is not None for h in have):
            start = (min(have) - timedelta(days=120)).date().isoformat()
        else:
            start = str(c.get("backfill_start", "2018-01-01"))
        raw: list[RawRecord] = []
        try:
            rows = self._query(sorted({m["code"] for m in markets.values()}), start, raw)
        except http.AuthError:
            return ConnectorResult(source_id=self.source_id, status=SourceStatus.FAIL, raw=raw,
                                   message="CFTC antwortet 401/403 (öffentliche API, kein Key vorgesehen).")
        except http.SchemaError as e:
            return ConnectorResult(source_id=self.source_id, status=SourceStatus.SCHEMA_CHANGED, raw=raw,
                                   message=str(e))
        except (http.FetchError, ValueError) as e:
            return ConnectorResult(source_id=self.source_id, status=SourceStatus.FAIL, raw=raw,
                                   message=f"CFTC-Abruf fehlgeschlagen: {http.redact_secrets(str(e))[:200]}")
        obs, stats = self.parse_rows(rows, now)
        if stats["schema_missing_fields"]:
            return ConnectorResult(source_id=self.source_id, status=SourceStatus.SCHEMA_CHANGED, raw=raw,
                                   message=f"COT-Felder fehlen: {stats['schema_missing_fields']}",
                                   discovered_ids=stats)
        n_rev = self.pit_adjust(obs, archived, now, revised_series=False)
        stats["revisions"] = n_rev
        if not obs:
            status = SourceStatus.FAIL
        elif stats["unavailable_markets"] or stats["oi_inconsistent"] or stats["conflicts"]:
            status = SourceStatus.WARN
        else:
            status = SourceStatus.PASS
        msg = (f"{len(obs)} COT-Werte, {len(markets) - len(stats['unavailable_markets'])}/{len(markets)} Märkte, "
               f"{n_rev} Revisionen")
        if stats["unavailable_markets"]:
            msg += f" | UNAVAILABLE: {stats['unavailable_markets']}"
        if stats["oi_inconsistent"]:
            msg += f" | OI-inkonsistent: {len(stats['oi_inconsistent'])}"
        return ConnectorResult(source_id=self.source_id, status=status, observations=obs, raw=raw, message=msg,
                               latest_observation_time=max((o.observation_time for o in obs), default=None),
                               discovered_ids=stats, parse_failures=len(stats["oi_inconsistent"]))


CONNECTORS = {
    "eia_petroleum_weekly": EiaPetroleumWeeklyConnector,
    "eia_natural_gas": EiaNaturalGasConnector,
    "fred_commodities": FredCommodityConnector,
    "cftc_cot": CftcCotConnector,
}

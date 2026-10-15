"""Öffentliche Markt- und Aufmerksamkeitsdaten (RESEARCH/SHADOW, 2026-10-15) – nur Archiv, keine Champion-Wirkung.

  nasdaq100_members      A4  Offizielle Nasdaq-100-Mitgliederliste (nasdaq.com) als tägliche PIT-Momentaufnahme
                             (Wikipedia hat keine Änderungshistorie; ab jetzt entsteht sie). NUR Archiv: get_universe
                             gehört zur eingefrorenen UNIVERSE_V1-Definition (Code-Hash) und bleibt unverändert.
  finra_short_volume     B3  FINRA Reg SHO Daily Short Sale Volume (konsolidiert, CNMS), Pipe-Datei, ohne Key.
  finra_short_interest   B3  FINRA Consolidated Short Interest (zweimal monatlich) über die FINRA Query API.
                             Ohne Zugangsdaten wird EIN unauthentifizierter Versuch gemacht; bei 401/403 ->
                             AUTH_MISSING mit Hinweis auf die Secrets FINRA_API_CLIENT_ID / FINRA_API_CLIENT_SECRET.
  wikimedia_pageviews    B5  Tägliche Wikipedia-Seitenaufrufe (Wikimedia REST, nur menschliche Nutzer) je Ticker;
                             Artikel-Zuordnung aus den Wikipedia-Listen S&P 500 / Nasdaq-100 (Links der Tabellen).

PIT: alle Werte gelten erst ab retrieved_at als bekannt (available_at = Abrufzeit; auch bei Nachholtagen) – nie
früher. Fehlende Tage/Dateien bleiben fehlend (nie 0). Volumen je Lauf begrenzt (Ticker-Universum, Fenster).
"""
from __future__ import annotations

import base64
import logging
import os
import re
import time
from datetime import date, datetime, timedelta, timezone
from urllib.parse import quote, unquote

from modules.external import http
from modules.external.pit import AvailabilityPrecision, Observation
from modules.external.sources.base import Connector, ConnectorResult, RawRecord, SourceStatus

log = logging.getLogger(__name__)
PARSER_VERSION = "market-public-1"
BROWSER_HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AdaptiveAsymmetryScanner research",
                   "Accept": "application/json, text/plain, */*"}
WIKI_HEADERS = {"User-Agent": "AdaptiveAsymmetryScanner/1.0 (research; contact: research@pcctrading.com)"}


def _raw(source_id: str, dataset: str, res: http.FetchResult) -> RawRecord:
    return RawRecord(source_id=source_id, dataset=dataset, url=res.url, fingerprint=res.fingerprint,
                     retrieved_at=res.retrieved_at, status_code=res.status, content_type=res.content_type,
                     content_hash=res.content_hash, bytes=res.bytes)


def _day(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


def _norm_ticker(t: str) -> str:
    return str(t or "").strip().upper().replace(".", "-").replace("/", "-")


def universe_tickers() -> list[str]:
    """Research-Ticker-Universum aus der Entity-Map (ohne Netzwerk); leer, wenn die Map fehlt."""
    try:
        from modules.entity_resolution.store import DEFAULT_PATH, EntityStore
        return sorted({r.ticker for r in EntityStore(DEFAULT_PATH).records
                       if r.usage == "research_ticker_identity" and r.valid_to is None and r.ticker})
    except Exception as e:  # noqa: BLE001
        log.warning("Entity-Map nicht lesbar: %s", e)
        return []


def _business_days(end: date, n: int) -> list[date]:
    out, d = [], end
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return sorted(out)


# ── A4: Nasdaq-100 (offiziell, nasdaq.com) ───────────────────────────────────
NASDAQ100_URL = "https://api.nasdaq.com/api/quote/list-type/nasdaq100"
NDX_MIN, NDX_MAX = 95, 110


def parse_nasdaq100(payload: dict) -> list[str]:
    """{"data": {"data": {"rows": [{"symbol": ...}]}}} (oder data.rows) -> sortierte Ticker; Schemafehler -> ValueError."""
    data = (payload or {}).get("data") or {}
    rows = (data.get("data") or {}).get("rows") if isinstance(data.get("data"), dict) else data.get("rows")
    if not isinstance(rows, list):
        raise ValueError("nasdaq100: data.data.rows fehlt")
    syms = sorted({_norm_ticker(r.get("symbol")) for r in rows if isinstance(r, dict) and r.get("symbol")})
    if not NDX_MIN <= len(syms) <= NDX_MAX:
        raise ValueError(f"nasdaq100: {len(syms)} Mitglieder außerhalb [{NDX_MIN}, {NDX_MAX}]")
    return syms


def fetch_nasdaq100() -> list[str]:
    """Offizielle Liste oder [] (nie werfend) – für Abgleiche/Research, nicht im Produktionsuniversum."""
    try:
        return parse_nasdaq100(http.fetch(NASDAQ100_URL, headers=BROWSER_HEADERS, retries=2).json())
    except Exception as e:  # noqa: BLE001
        log.warning("Nasdaq-100 (nasdaq.com) nicht verfügbar: %s", e)
        return []


class Nasdaq100MembersConnector(Connector):
    source_id = "nasdaq100_members"
    parser_version = PARSER_VERSION

    def fetch(self, now: datetime) -> ConnectorResult:
        try:
            res = http.fetch(NASDAQ100_URL, headers=BROWSER_HEADERS)
        except http.FetchError as e:
            return ConnectorResult(self.source_id, SourceStatus.FAIL, message=f"nasdaq.com: {e}")
        raw = [_raw(self.source_id, "nasdaq100", res)]
        try:
            syms = parse_nasdaq100(res.json())
        except ValueError as e:
            return ConnectorResult(self.source_id, SourceStatus.SCHEMA_CHANGED, raw=raw, message=str(e))
        t = _day(res.retrieved_at.date())
        obs = Observation(source_id=self.source_id, dataset="index_constituents", series_id="NDX", entity_id="NDX",
                          metric="constituent_count", value=float(len(syms)), unit="count", observation_time=t,
                          available_at=res.retrieved_at, retrieved_at=res.retrieved_at,
                          availability_precision=AvailabilityPrecision.EXACT_TIMESTAMP,
                          parser_version=self.parser_version, attrs={"symbols": syms})
        return ConnectorResult(self.source_id, SourceStatus.PASS, observations=[obs], raw=raw,
                               message=f"Nasdaq-100: {len(syms)} Mitglieder", latest_observation_time=t)


# ── B3: FINRA Daily Short Sale Volume ────────────────────────────────────────
FINRA_SHVOL_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{ymd}.txt"
SHVOL_COLS = ("Date", "Symbol", "ShortVolume", "ShortExemptVolume", "TotalVolume", "Market")


def parse_short_volume(text: str, tickers: set[str] | None = None) -> list[dict]:
    """Pipe-Datei -> [{date, symbol, short, exempt, total}] (Kopfzeile Pflicht; Summen-/Leerzeilen übersprungen)."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines or tuple(c.strip() for c in lines[0].split("|"))[:5] != SHVOL_COLS[:5]:
        raise ValueError("FINRA-Short-Volume: Kopfzeile unerwartet")
    out = []
    for ln in lines[1:]:
        p = [c.strip() for c in ln.split("|")]
        if len(p) < 5 or not p[0].isdigit():
            continue
        sym = _norm_ticker(p[1])
        if tickers is not None and sym not in tickers:
            continue
        try:
            out.append({"date": datetime.strptime(p[0], "%Y%m%d").date(), "symbol": sym, "short": float(p[2]),
                        "exempt": float(p[3]), "total": float(p[4])})
        except ValueError:
            continue
    return out


class FinraShortVolumeConnector(Connector):
    source_id = "finra_short_volume"
    parser_version = PARSER_VERSION

    def fetch(self, now: datetime) -> ConnectorResult:
        tickers = set(universe_tickers())
        if not tickers:
            return ConnectorResult(self.source_id, SourceStatus.FAIL, message="Entity-Map/Ticker-Universum fehlt")
        days = _business_days(now.date() - timedelta(days=1), int(self.cfg.get("lookback_business_days", 5)))
        obs, raw, missing = [], [], []
        for d in days:
            try:
                res = http.fetch(FINRA_SHVOL_URL.format(ymd=d.strftime("%Y%m%d")), retries=2)
            except http.FetchError:
                missing.append(d.isoformat())                     # Feiertag / noch nicht veröffentlicht
                continue
            raw.append(_raw(self.source_id, "cnms_daily", res))
            try:
                rows = parse_short_volume(res.content.decode("utf-8", "replace"), tickers)
            except ValueError as e:
                return ConnectorResult(self.source_id, SourceStatus.SCHEMA_CHANGED, raw=raw, message=str(e))
            for r in rows:
                if r["total"] <= 0:
                    continue
                obs.append(Observation(
                    source_id=self.source_id, dataset="reg_sho_daily", series_id=f"CNMS:{r['symbol']}",
                    entity_id=r["symbol"], metric="short_volume_ratio", value=round(r["short"] / r["total"], 6),
                    unit="ratio", observation_time=_day(r["date"]), available_at=res.retrieved_at,
                    retrieved_at=res.retrieved_at, availability_precision=AvailabilityPrecision.EXACT_TIMESTAMP,
                    parser_version=self.parser_version,
                    attrs={"short_volume": r["short"], "short_exempt_volume": r["exempt"], "total_volume": r["total"]}))
        if not raw:
            return ConnectorResult(self.source_id, SourceStatus.FAIL, message=f"keine FINRA-Datei abrufbar ({missing})")
        latest = max((o.observation_time for o in obs), default=None)
        return ConnectorResult(self.source_id, SourceStatus.PASS if obs else SourceStatus.WARN, observations=obs,
                               raw=raw, message=f"{len(obs)} Ticker-Tage; fehlend: {missing or '-'}",
                               latest_observation_time=latest)


# ── B3: FINRA Consolidated Short Interest (Query API) ────────────────────────
FINRA_TOKEN_URL = "https://ews.fip.finra.org/fip/rest/ews/oauth2/access_token?grant_type=client_credentials"
FINRA_SI_URL = "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
SI_FIELDS = ("symbolCode", "settlementDate", "currentShortPositionQuantity")


def parse_short_interest(rows: list, tickers: set[str]) -> list[dict]:
    if not isinstance(rows, list):
        raise ValueError("FINRA-Short-Interest: Antwort ist keine Liste")
    if rows and not all(f in rows[0] for f in SI_FIELDS):
        raise ValueError(f"FINRA-Short-Interest: Felder {SI_FIELDS} fehlen ({sorted(rows[0])[:12]})")
    out = []
    for r in rows:
        sym = _norm_ticker(r.get("symbolCode"))
        if sym not in tickers or r.get("currentShortPositionQuantity") is None:
            continue
        try:
            out.append({"symbol": sym, "settlement": date.fromisoformat(str(r["settlementDate"])[:10]),
                        "short": float(r["currentShortPositionQuantity"]),
                        "adv": None if r.get("averageDailyVolumeQuantity") is None
                        else float(r["averageDailyVolumeQuantity"]),
                        "days_to_cover": None if r.get("daysToCoverQuantity") is None
                        else float(r["daysToCoverQuantity"])})
        except (TypeError, ValueError):
            continue
    return out


class FinraShortInterestConnector(Connector):
    source_id = "finra_short_interest"
    parser_version = PARSER_VERSION
    PAGE, MAX_PAGES = 5000, 12

    def _token(self) -> str | None:
        cid = (os.environ.get("FINRA_API_CLIENT_ID") or "").strip()
        sec = (os.environ.get("FINRA_API_CLIENT_SECRET") or "").strip()
        if not cid or not sec:
            return None
        basic = base64.b64encode(f"{cid}:{sec}".encode()).decode()
        res = http.fetch(FINRA_TOKEN_URL, method="POST", headers={"Authorization": f"Basic {basic}"}, data={})
        return res.json().get("access_token")

    def fetch(self, now: datetime) -> ConnectorResult:
        tickers = set(universe_tickers())
        if not tickers:
            return ConnectorResult(self.source_id, SourceStatus.FAIL, message="Entity-Map/Ticker-Universum fehlt")
        try:
            token = self._token()
        except http.FetchError as e:
            return ConnectorResult(self.source_id, SourceStatus.AUTH_MISSING,
                                   message=f"FINRA-Token abgelehnt (FINRA_API_CLIENT_ID/_SECRET prüfen): {e}")
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        start = (now.date() - timedelta(days=int(self.cfg.get("lookback_days", 40)))).isoformat()
        rows, raw = [], []
        for page in range(self.MAX_PAGES):
            body = {"limit": self.PAGE, "offset": page * self.PAGE,
                    "dateRangeFilters": [{"fieldName": "settlementDate", "startDate": start,
                                          "endDate": now.date().isoformat()}]}
            try:
                res = http.fetch(FINRA_SI_URL, method="POST", headers=headers, json_body=body, retries=2)
            except http.AuthError:
                return ConnectorResult(self.source_id, SourceStatus.AUTH_MISSING, raw=raw,
                                       message="FINRA Query API verlangt Zugangsdaten: kostenloses API-Konto im "
                                               "FINRA API Console anlegen und Secrets FINRA_API_CLIENT_ID + "
                                               "FINRA_API_CLIENT_SECRET setzen.")
            except http.FetchError as e:
                return ConnectorResult(self.source_id, SourceStatus.FAIL, raw=raw, message=f"FINRA API: {e}")
            raw.append(_raw(self.source_id, "consolidated_short_interest", res))
            chunk = res.json() if res.content else []
            rows += chunk if isinstance(chunk, list) else []
            if not isinstance(chunk, list) or len(chunk) < self.PAGE:
                break
        try:
            parsed = parse_short_interest(rows, tickers)
        except ValueError as e:
            return ConnectorResult(self.source_id, SourceStatus.SCHEMA_CHANGED, raw=raw, message=str(e))
        obs = []
        for r in parsed:
            for metric, val, unit in (("short_interest_shares", r["short"], "shares"),
                                      ("days_to_cover", r["days_to_cover"], "days")):
                if val is None:
                    continue
                obs.append(Observation(
                    source_id=self.source_id, dataset="short_interest", series_id=f"SI:{r['symbol']}",
                    entity_id=r["symbol"], metric=metric, value=val, unit=unit,
                    observation_time=_day(r["settlement"]), available_at=raw[-1].retrieved_at,
                    retrieved_at=raw[-1].retrieved_at, availability_precision=AvailabilityPrecision.EXACT_TIMESTAMP,
                    parser_version=self.parser_version, attrs={"avg_daily_volume": r["adv"]}))
        latest = max((o.observation_time for o in obs), default=None)
        return ConnectorResult(self.source_id, SourceStatus.PASS if obs else SourceStatus.WARN, observations=obs,
                               raw=raw, message=f"{len(parsed)} Ticker-Stichtage ({'mit' if token else 'ohne'} Token)",
                               latest_observation_time=latest)


# ── B5: Wikimedia Pageviews ──────────────────────────────────────────────────
WIKI_LISTS = {"sp500": ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "constituents"),
              "nasdaq100": ("https://en.wikipedia.org/wiki/Nasdaq-100", "constituents")}
PAGEVIEWS_URL = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia.org/all-access/user/"
                 "{article}/daily/{start}/{end}")


def parse_wiki_list(html: str, table_id: str = "constituents") -> dict[str, str]:
    """Ticker -> Artikeltitel aus der Konstituenten-Tabelle (erste zwei Spalten: Ticker und verlinkter Name,
    Reihenfolge egal). Nur /wiki/-Links ohne Namensraum."""
    from lxml import html as lh
    doc = lh.fromstring(html)
    tables = doc.xpath(f'//table[@id="{table_id}"]') or doc.xpath('//table[contains(@class,"wikitable")]')
    out: dict[str, str] = {}
    for table in tables:
        for tr in table.xpath(".//tr")[1:]:
            cells = tr.xpath("./td")
            if len(cells) < 2:
                continue
            texts = [c.text_content().strip() for c in cells[:2]]
            links = [c.xpath('.//a[starts-with(@href,"/wiki/")]/@href') for c in cells[:2]]
            tk_i = 0 if re.fullmatch(r"[A-Z][A-Z.\-]{0,6}", texts[0]) else 1
            ticker, article_links = texts[tk_i], links[1 - tk_i]
            art = next((unquote(h.split("/wiki/", 1)[1]) for h in article_links if ":" not in h.split("/wiki/", 1)[1]),
                       None)
            if re.fullmatch(r"[A-Z][A-Z.\-]{0,6}", ticker) and art:
                out[_norm_ticker(ticker)] = art.split("#")[0]
        if out:
            break
    return out


def parse_pageviews(payload: dict) -> list[tuple[date, float]]:
    items = (payload or {}).get("items")
    if not isinstance(items, list):
        raise ValueError("pageviews: items fehlt")
    return [(datetime.strptime(str(i["timestamp"])[:8], "%Y%m%d").date(), float(i["views"])) for i in items
            if i.get("timestamp") and i.get("views") is not None]


class WikimediaPageviewsConnector(Connector):
    source_id = "wikimedia_pageviews"
    parser_version = PARSER_VERSION

    def fetch(self, now: datetime) -> ConnectorResult:
        raw, articles = [], {}
        for name, (url, tid) in WIKI_LISTS.items():
            try:
                res = http.fetch(url, headers=WIKI_HEADERS, retries=2)
            except http.FetchError as e:
                log.warning("Wikipedia-Liste %s: %s", name, e)
                continue
            raw.append(_raw(self.source_id, f"list_{name}", res))
            for tk, art in parse_wiki_list(res.content.decode("utf-8", "replace"), tid).items():
                articles.setdefault(tk, art)
        if len(articles) < int(self.cfg.get("min_articles", 300)):
            return ConnectorResult(self.source_id, SourceStatus.SCHEMA_CHANGED, raw=raw,
                                   message=f"nur {len(articles)} Ticker-Artikel aus den Wikipedia-Listen")
        end = now.date() - timedelta(days=1)
        start = end - timedelta(days=int(self.cfg.get("lookback_days", 3)) - 1)
        obs, failures, pause = [], 0, float(self.cfg.get("request_pause_seconds", 0.05))
        first_raw = None                                   # nur EIN Rohdaten-Eintrag statt ~600 je Lauf
        deadline = time.monotonic() + float(self.cfg.get("max_fetch_seconds", 900)) * 0.9
        for tk, art in sorted(articles.items()):
            if time.monotonic() > deadline:
                failures += 1
                continue
            url = PAGEVIEWS_URL.format(article=quote(art.replace(" ", "_"), safe=""),
                                       start=start.strftime("%Y%m%d00"), end=end.strftime("%Y%m%d00"))
            try:
                res = http.fetch(url, headers=WIKI_HEADERS, retries=2)
                rows = parse_pageviews(res.json())
            except (http.FetchError, ValueError, KeyError):
                failures += 1
                continue
            first_raw = first_raw or _raw(self.source_id, "pageviews", res)
            for d, v in rows:
                obs.append(Observation(
                    source_id=self.source_id, dataset="pageviews_daily", series_id=f"enwiki:{art}", entity_id=tk,
                    metric="wiki_pageviews", value=v, unit="views", observation_time=_day(d),
                    available_at=res.retrieved_at, retrieved_at=res.retrieved_at,
                    availability_precision=AvailabilityPrecision.EXACT_TIMESTAMP, parser_version=self.parser_version,
                    attrs={"article": art}))
            if pause:
                time.sleep(pause)
        if first_raw is not None:
            raw.append(first_raw)
        status = SourceStatus.PASS if obs and failures <= 0.1 * len(articles) else \
            SourceStatus.WARN if obs else SourceStatus.FAIL
        return ConnectorResult(self.source_id, status, observations=obs, raw=raw,
                               message=f"{len(articles)} Artikel, {len(obs)} Tageswerte, {failures} Fehler",
                               latest_observation_time=max((o.observation_time for o in obs), default=None))


CONNECTORS: dict[str, type[Connector]] = {
    "nasdaq100_members": Nasdaq100MembersConnector,
    "finra_short_volume": FinraShortVolumeConnector,
    "finra_short_interest": FinraShortInterestConnector,
    "wikimedia_pageviews": WikimediaPageviewsConnector,
}

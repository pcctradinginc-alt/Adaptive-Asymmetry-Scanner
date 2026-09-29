"""ENTSO-E-Konnektor: XML-Parsing (A03, Auflösungen, Zeitumstellung,
Acknowledgement), Tagesmittel, PIT-Zeiten, Token-Hygiene, Energie-Regime.
Keine Netzwerkzugriffe."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch

from modules.external import http, regime
from modules.external.pit import AvailabilityPrecision, Observation, utc_now
from modules.external.registry import SourceRegistry
from modules.external.sources import energy as en
from modules.external.sources.base import SourceStatus

UTC = timezone.utc
NS = "urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3"


def _doc(periods, curve="A01", tag="price.amount"):
    """periods: [(start_iso, end_iso, resolution, {pos: value})]"""
    ts = []
    for start, end, res, pts in periods:
        points = "".join(f"<Point><position>{p}</position><{tag}>{v}</{tag}></Point>" for p, v in sorted(pts.items()))
        ts.append(f"<TimeSeries><curveType>{curve}</curveType><Period><timeInterval><start>{start}</start>"
                  f"<end>{end}</end></timeInterval><resolution>{res}</resolution>{points}</Period></TimeSeries>")
    return f'<Publication_MarketDocument xmlns="{NS}">{"".join(ts)}</Publication_MarketDocument>'


def test_a03_fills_missing_positions_and_a01_does_not():
    xml = _doc([("2026-09-01T22:00Z", "2026-09-02T02:00Z", "PT60M", {1: 50.0, 3: 70.0})], curve="A03")
    assert [v for _, _, v in en.parse_timeseries_points(xml, "price.amount")] == [50.0, 50.0, 70.0, 70.0]
    xml1 = _doc([("2026-09-01T22:00Z", "2026-09-02T02:00Z", "PT60M", {1: 50.0, 3: 70.0})], curve="A01")
    assert [v for _, _, v in en.parse_timeseries_points(xml1, "price.amount")] == [50.0, 70.0]


def test_daily_mean_prefers_finest_resolution_and_requires_complete_day():
    # Lieferungstag 2026-09-02 (Berlin, UTC+2): 2026-09-01T22:00Z .. 2026-09-02T22:00Z
    q = {i: 100.0 for i in range(1, 97)}                     # 96 Viertelstunden à 100
    h = {i: 40.0 for i in range(1, 25)}                      # parallele Stundenreihe à 40
    xml = _doc([("2026-09-01T22:00Z", "2026-09-02T22:00Z", "PT15M", q),
                ("2026-09-01T22:00Z", "2026-09-02T22:00Z", "PT60M", h)])
    days = en.daily_means(en.parse_timeseries_points(xml, "price.amount"))
    assert days[date(2026, 9, 2)] == (100.0, 96, 15)         # nicht (100+40)/2
    half = _doc([("2026-09-03T22:00Z", "2026-09-04T08:00Z", "PT60M", {i: 1.0 for i in range(1, 11)})])
    assert en.daily_means(en.parse_timeseries_points(half, "price.amount")) == {}   # unvollständig


def test_dst_day_has_25_hours():
    # 2026-10-25: Umstellung auf Winterzeit -> 25 Stunden (UTC 2026-10-24T22:00 .. 2026-10-25T23:00)
    xml = _doc([("2026-10-24T22:00Z", "2026-10-25T23:00Z", "PT60M", {i: float(i) for i in range(1, 26)})])
    days = en.daily_means(en.parse_timeseries_points(xml, "price.amount"))
    assert days[date(2026, 10, 25)][1] == 25


def test_acknowledgement_means_no_data_not_zero():
    ack = '<Acknowledgement_MarketDocument xmlns="x"><Reason><code>999</code></Reason></Acknowledgement_MarketDocument>'
    assert en.parse_timeseries_points(ack, "price.amount") == []


def _res(content: str):
    b = content.encode()
    return http.FetchResult(url=en.API_URL, status=200, content=b, content_type="text/xml",
                            retrieved_at=utc_now(), content_hash="h", fingerprint="fp", bytes=len(b))


def test_connector_pit_times_units_and_token_hygiene(monkeypatch):
    monkeypatch.setenv("ENTSOE_API_TOKEN", "SECRET-123")
    now = datetime(2026, 9, 29, 10, tzinfo=UTC)
    seen = []
    price = _doc([("2026-09-27T22:00Z", "2026-09-28T22:00Z", "PT60M", {i: 80.0 for i in range(1, 25)})])
    load = _doc([("2026-09-27T22:00Z", "2026-09-28T22:00Z", "PT15M", {i: 55000.0 for i in range(1, 97)})],
                tag="quantity")

    def fake(url, params=None, **kw):
        seen.append(params)
        assert url == en.API_URL and "SECRET" not in url
        return _res(price if params["documentType"] == "A44" else load)
    cfg = {**SourceRegistry().sources["entsoe_power"], "price_zones": ["DE_LU"], "lookback_days": 30}
    with patch.object(en.http, "fetch", side_effect=fake):
        res = en.EntsoePowerConnector(cfg).fetch(now)
    assert res.status == SourceStatus.PASS
    assert all(p["securityToken"] == "SECRET-123" for p in seen)
    assert all("SECRET" not in r.url for r in res.raw)
    px = [o for o in res.observations if o.metric == "da_price_daily_mean"][0]
    ld = [o for o in res.observations if o.metric == "load_daily_mean"][0]
    assert px.value == 80.0 and px.unit == "EUR_per_MWh" and px.entity_id == "DE_LU"
    # Lieferungstag 2026-09-28 -> bekannt ab 2026-09-27 23:59:59 Berlin = 21:59:59 UTC
    assert px.available_at == datetime(2026, 9, 27, 21, 59, 59, tzinfo=UTC)
    assert ld.unit == "MW" and ld.available_at >= datetime(2026, 9, 29, 4, tzinfo=UTC)
    assert px.availability_precision == AvailabilityPrecision.CONSERVATIVE_DATE


def test_without_token_no_call_and_errors_are_redacted(monkeypatch):
    monkeypatch.delenv("ENTSOE_API_TOKEN", raising=False)
    with patch.object(en.http, "fetch") as m:
        assert en.EntsoePowerConnector({}).fetch(utc_now()).status == SourceStatus.AUTH_MISSING
    m.assert_not_called()
    msg = http.redact_secrets("Max retries exceeded with url: /api?documentType=A44&securityToken=SECRET-123")
    assert "SECRET" not in msg and "securityToken=***" in msg


def _o(metric, entity, day, value, avail):
    t = datetime(day.year, day.month, day.day, tzinfo=UTC)
    return Observation(source_id="entsoe_power", dataset="d", series_id="s", entity_id=entity, metric=metric,
                       value=value, unit="u", observation_time=t, available_at=avail, retrieved_at=avail,
                       availability_precision=AvailabilityPrecision.CONSERVATIVE_DATE, parser_version="1")


def test_energy_state_pit_and_labels():
    obs = []
    start = date(2025, 8, 1)
    for i in range(420):
        d = start + timedelta(days=i)
        price = 80.0 + (i % 5) if i < 413 else 160.0             # letzte Woche teuer
        obs.append(_o("da_price_daily_mean", "DE_LU", d, price,
                      datetime(d.year, d.month, d.day, tzinfo=UTC) - timedelta(hours=2)))
        load = 50000.0 * (1.05 if i >= 413 else 1.0)
        obs.append(_o("load_daily_mean", "DE", d, load, datetime(d.year, d.month, d.day, tzinfo=UTC) + timedelta(days=1)))
    as_of = datetime.combine(start + timedelta(days=421), datetime.min.time(), UTC)
    st = regime.energy_state(obs, as_of)
    assert st["de_power_price_7d"] == 160.0 and st["labels"]["power"] == "power_expensive"
    assert abs(st["de_load_yoy"] - 0.05) < 1e-9 and st["labels"]["demand"] == "load_up"
    # vor Veröffentlichung der teuren Woche: noch nicht sichtbar
    early = regime.energy_state(obs, datetime.combine(start + timedelta(days=412), datetime.min.time(), UTC))
    assert early["de_power_price_7d"] < 90.0


def test_registry_entry():
    c = SourceRegistry().sources["entsoe_power"]
    assert c["auth_env_variable"] == "ENTSOE_API_TOKEN" and c["requires_auth"] and c["enabled"]
    assert c["family"] == "real_economy" and c["license_status"] == "OK"

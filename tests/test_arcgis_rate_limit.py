"""ArcGIS-Kontingent (HTTP 200 + {"error": {"code": 429}}) respektieren (Audit 2026-11-06).

Live (Snapshot 2026-11-06T15:58Z): imf_portwatch_ports/-chokepoints/-disruptions BROKEN/DEGRADED mit
"[Isolation basis=FAIL(429) where=FAIL(429) ...] ... 'API calls quota exceeded'". Vorher: je Layer 1 Abfrage +
bis zu 5 Isolations-Probes, die alle erneut am Kontingent scheitern, danach dieselbe Kette für die übrigen Layer.
Nachher: keine Probe bei Rate-Limit; das Minutenlimit ("Retry after 60 sec") wird abgewartet und höchstens
RATE_LIMIT_RETRIES-mal erneut gefragt; erst dann Kreisunterbrecher je Host für den Lauf. Fehler bleibt sichtbar.
"""
import pytest

from modules.external.sources import maritime as m

SERVICE = "https://services9.arcgis.com/abc/arcgis/rest/services/Daily_Ports_Data/FeatureServer"
# Wortlaut live 2026-11-06 (outputs/external_data/health/source_health.json)
QUOTA = {"error": {"code": 429, "message": "Unable to perform query. Too many requests.",
                   "details": [("API calls quota exceeded (6380 request units)! maximum allowed request units (6000) "
                                "per Minute. Retry after 60 sec.")]}}
OK = {"features": [{"attributes": {"portid": "p1"}}]}


class _Res:
    def __init__(self, payload):
        self._p = payload

    def json(self):
        return self._p


class _Fetch:
    """Antwortet der Reihe nach mit `payloads`; die letzte Antwort wiederholt sich."""

    def __init__(self, *payloads):
        self.payloads, self.calls = list(payloads), []

    def __call__(self, url, params=None, **kw):
        self.calls.append((url, dict(params or {})))
        return _Res(self.payloads[min(len(self.calls) - 1, len(self.payloads) - 1)])


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    m.reset_rate_limit_breaker()
    waits = []
    monkeypatch.setattr(m, "_sleep", waits.append)
    yield waits
    m.reset_rate_limit_breaker()


@pytest.mark.parametrize("err, expected", [
    (QUOTA["error"], True),
    ({"code": 400, "message": "Too many requests from this IP"}, True),
    ({"code": 400, "message": "x", "details": ["API calls quota exceeded"]}, True),
    ({"code": 400, "message": "Invalid URL"}, False),
    ({"code": 400, "message": "'Invalid field: x' parameter is invalid"}, False),
    (None, False),
])
def test_rate_limit_detection(err, expected):
    assert m.arcgis_rate_limited(err) is expected


@pytest.mark.parametrize("err, wait", [
    (QUOTA["error"], 60.0),
    ({"code": 429, "details": ["Retry after 300 sec."]}, m.RATE_LIMIT_MAX_WAIT_S),   # begrenzt
    ({"code": 429, "message": "Too many requests"}, m.RATE_LIMIT_DEFAULT_WAIT_S),   # ohne Angabe
])
def test_rate_limit_wait_follows_the_service(err, wait):
    assert m.rate_limit_wait_seconds(err) == wait


def test_minute_limit_is_retried_after_the_announced_wait(_reset):
    """Minutenlimit: nach 60 s erneut fragen statt den Lauf sofort aufzugeben."""
    fetch = _Fetch(QUOTA, OK)
    assert m.query_features(SERVICE, 0, "1=1", ["portid"], fetch, isolation_field="ObjectId") == OK["features"]
    assert len(fetch.calls) == 2 and _reset == [60.0]
    assert m.query_features(SERVICE.replace("Ports", "Chokepoints"), 0, "1=1", ["portid"], fetch) == OK["features"]


def test_total_wait_per_host_stays_within_the_run_budget(_reset):
    """Summe der Wartezeiten je Host und Lauf begrenzt (bleibt im Abrufbudget der Quelle, kein SourceTimeout)."""
    fetch = _Fetch(QUOTA, OK, QUOTA, OK, QUOTA)
    assert m.query_features(SERVICE, 0, "1=1", ["portid"], fetch) == OK["features"]
    assert m.query_features(SERVICE, 0, "1=1", ["portid"], fetch) == OK["features"]
    with pytest.raises(m.ArcGISQueryError, match=r"^\[UPSTREAM_RATE_LIMITED\]"):
        m.query_features(SERVICE, 0, "1=1", ["portid"], fetch)
    assert len(fetch.calls) == 5 and sum(_reset) <= m.RATE_LIMIT_WAIT_BUDGET_S and _reset == [60.0, 60.0]
    with pytest.raises(m.PortWatchUpstreamUnavailable):
        m.query_features(SERVICE, 0, "1=1", ["portid"], fetch)
    m.reset_rate_limit_breaker()                                  # neuer Lauf: Budget wieder voll
    fetch2 = _Fetch(QUOTA, OK)
    assert m.query_features(SERVICE, 0, "1=1", ["portid"], fetch2) == OK["features"] and _reset[-1] == 60.0


def test_quota_error_runs_no_isolation_and_blocks_further_calls_to_the_host(_reset):
    fetch = _Fetch(QUOTA)
    with pytest.raises(m.ArcGISQueryError, match=r"^\[UPSTREAM_RATE_LIMITED\]"):
        m.query_features(SERVICE, 0, "1=1", ["portid"], fetch, isolation_field="ObjectId")
    assert len(fetch.calls) == 1 + m.RATE_LIMIT_RETRIES          # vorher: 1 + bis zu 5 Probes, ohne Wartezeit
    assert _reset == [60.0] * m.RATE_LIMIT_RETRIES
    fetch.calls.clear()
    other = SERVICE.replace("Daily_Ports_Data", "Daily_Chokepoints_Data")
    with pytest.raises(m.PortWatchUpstreamUnavailable, match="UPSTREAM_RATE_LIMITED"):
        m.query_features(other, 0, "1=1", ["portid"], fetch)
    with pytest.raises(m.PortWatchUpstreamUnavailable, match="UPSTREAM_RATE_LIMITED"):
        m.fetch_layer_metadata(other, 0, fetch)
    assert fetch.calls == []                                      # kein weiterer Abruf dieses Hosts


def test_metadata_quota_error_is_rate_limited_not_schema_change():
    fetch = _Fetch(QUOTA)
    with pytest.raises(m.PortWatchUpstreamUnavailable, match="UPSTREAM_RATE_LIMITED"):
        m.fetch_layer_metadata(SERVICE, 0, fetch)
    with pytest.raises(m.PortWatchUpstreamUnavailable):
        m.query_features(SERVICE, 0, "1=1", ["portid"], fetch)
    assert len(fetch.calls) == 1 + m.RATE_LIMIT_RETRIES


def test_other_hosts_and_other_errors_keep_the_existing_isolation():
    fetch = _Fetch({"error": {"code": 400, "message": "Unable to perform query", "details": []}})
    with pytest.raises(m.ArcGISQueryError) as exc:
        m.query_features(SERVICE, 0, "1=1", ["portid"], fetch, isolation_field="ObjectId")
    assert "[Isolation" in str(exc.value) and "UPSTREAM_RATE_LIMITED" not in str(exc.value)
    assert len(fetch.calls) > 1                                   # Isolation wie bisher
    ok = _Fetch(OK)
    assert m.query_features("https://other.example.org/FeatureServer", 0, "1=1", ["portid"], ok) == [
        {"attributes": {"portid": "p1"}}]

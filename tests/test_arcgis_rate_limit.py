"""ArcGIS-Kontingent (HTTP 200 + {"error": {"code": 429}}) respektieren (Audit 2026-11-06).

Live (Snapshot 2026-11-06T15:58Z): imf_portwatch_ports/-chokepoints/-disruptions BROKEN/DEGRADED mit
"[Isolation basis=FAIL(429) where=FAIL(429) ...] ... 'API calls quota exceeded'". Vorher: je Layer 1 Abfrage +
bis zu 5 Isolations-Probes, die alle erneut am Kontingent scheitern, danach dieselbe Kette für die übrigen Layer.
Nachher: keine Probe bei Rate-Limit, Kreisunterbrecher je Host für den Lauf, Fehler bleibt sichtbar.
"""
import pytest

from modules.external.sources import maritime as m

SERVICE = "https://services9.arcgis.com/abc/arcgis/rest/services/Daily_Ports_Data/FeatureServer"
QUOTA = {"error": {"code": 429, "message": "Unable to perform query. Too many requests.",
                   "details": ["API calls quota exceeded (6410/6400)"]}}


class _Res:
    def __init__(self, payload):
        self._p = payload

    def json(self):
        return self._p


class _Fetch:
    def __init__(self, payload):
        self.payload, self.calls = payload, []

    def __call__(self, url, params=None, **kw):
        self.calls.append((url, dict(params or {})))
        return _Res(self.payload)


@pytest.fixture(autouse=True)
def _reset():
    m.reset_rate_limit_breaker()
    yield
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


def test_quota_error_runs_no_isolation_and_blocks_further_calls_to_the_host():
    fetch = _Fetch(QUOTA)
    with pytest.raises(m.ArcGISQueryError, match=r"^\[UPSTREAM_RATE_LIMITED\]"):
        m.query_features(SERVICE, 0, "1=1", ["portid"], fetch, isolation_field="ObjectId")
    assert len(fetch.calls) == 1                                  # vorher: 1 + bis zu 5 Probes
    other = SERVICE.replace("Daily_Ports_Data", "Daily_Chokepoints_Data")
    with pytest.raises(m.PortWatchUpstreamUnavailable, match="UPSTREAM_RATE_LIMITED"):
        m.query_features(other, 0, "1=1", ["portid"], fetch)
    with pytest.raises(m.PortWatchUpstreamUnavailable, match="UPSTREAM_RATE_LIMITED"):
        m.fetch_layer_metadata(other, 0, fetch)
    assert len(fetch.calls) == 1                                  # kein weiterer Abruf dieses Hosts


def test_metadata_quota_error_is_rate_limited_not_schema_change():
    fetch = _Fetch(QUOTA)
    with pytest.raises(m.PortWatchUpstreamUnavailable, match="UPSTREAM_RATE_LIMITED"):
        m.fetch_layer_metadata(SERVICE, 0, fetch)
    with pytest.raises(m.PortWatchUpstreamUnavailable):
        m.query_features(SERVICE, 0, "1=1", ["portid"], fetch)
    assert len(fetch.calls) == 1


def test_other_hosts_and_other_errors_keep_the_existing_isolation():
    fetch = _Fetch({"error": {"code": 400, "message": "Unable to perform query", "details": []}})
    with pytest.raises(m.ArcGISQueryError) as exc:
        m.query_features(SERVICE, 0, "1=1", ["portid"], fetch, isolation_field="ObjectId")
    assert "[Isolation" in str(exc.value) and "UPSTREAM_RATE_LIMITED" not in str(exc.value)
    assert len(fetch.calls) > 1                                   # Isolation wie bisher
    ok = _Fetch({"features": [{"attributes": {"portid": "p1"}}]})
    assert m.query_features("https://other.example.org/FeatureServer", 0, "1=1", ["portid"], ok) == [
        {"attributes": {"portid": "p1"}}]

"""HDX/PortWatch 2026-11-04 regressions. Strictly offline; no OCHA network calls."""
from __future__ import annotations

import csv
import io
from datetime import date, datetime, timedelta, timezone

import pytest
import requests

from modules.external.sources import portwatch_hdx_shadow as hdx

NOW = datetime(2026, 11, 4, 21, 0, tzinfo=timezone.utc)
SPEC = hdx.RESOURCES["chokepoints"]
URL = "https://data.humdata.org/dataset/portwatch/resource/test/download/chokepoints.csv"


class Response:
    status_code = 200

    def __init__(self, *, payload=None, body=b"", headers=None, code=200):
        self.payload = payload
        self.body = body
        self.headers = headers or {}
        self.status_code = code
        self.closed = False

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError("HTTP failure")

    def iter_content(self, chunk_size=65536):
        yield self.body

    def close(self):
        self.closed = True


def csv_bytes(spec=SPEC, *, last=date(2026, 10, 23), rows=15, value="12", columns=None):
    buf = io.StringIO()
    fields = columns or ["date", "portname", spec.metric]
    w = csv.DictWriter(buf, fieldnames=fields)
    w.writeheader()
    for i in range(rows):
        d = last - timedelta(days=i)
        w.writerow({k: (d.isoformat() if k == "date"
                        else "Strait of Hormuz" if k == "portname"
                        else value) for k in fields})
    return buf.getvalue().encode("utf-8")


class Session:
    def __init__(self, spec=SPEC, body=None, *, redirect=None, wrong_org=False,
                 bad_id=False, resource_name=None, fail_package_link=False):
        self.spec = spec
        self.body = csv_bytes(spec) if body is None else body
        self.calls = []
        self.closed = False
        self.redirect = redirect
        self.wrong_org = wrong_org
        self.bad_id = bad_id
        self.resource_name = resource_name
        self.fail_package_link = fail_package_link

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if url.endswith("/resource_show"):
            return Response(payload={"success": True, "result": {
                "id": "unrelated" if self.bad_id else self.spec.resource_id,
                "name": self.resource_name or self.spec.filename,
                "format": "CSV",
                "package_id": "verified-dataset", "url": URL}})
        if url.endswith("/package_show"):
            return Response(payload={"success": True, "result": {
                "id": "verified-dataset",
                "organization": {"name": "other" if self.wrong_org else "portwatch"},
                "resources": [] if self.fail_package_link else [{"id": self.spec.resource_id}]}})
        if url == URL:
            if self.redirect:
                return Response(headers={"Location": self.redirect}, code=302)
            return Response(body=self.body)
        raise AssertionError(f"unexpected URL {url!r}")


def test_valid_mirror_is_shadow_only_and_only_known_scope():
    session = Session()
    report = hdx.check_resources(NOW, keys=("chokepoints",), session=session)
    entry = report["resources"]["chokepoints"]
    assert report["status"] == "PASS_SHADOW"
    assert report["mode"] == "SHADOW_ONLY"
    assert report["production_fallback_enabled"] is False
    assert report["safe_mode_override"] is False
    assert report["ports_global_coverage"] is False
    assert entry["rows"] == 15 and entry["latest_observation_date"] == "2026-10-23"
    assert entry["family"] == "chokepoints" and entry["scope"] == "GLOBAL_CHOKEPOINTS"
    assert entry["sha256"] and entry["provenance"]["resource_id"] == SPEC.resource_id
    assert all(call[1].get("allow_redirects") is False for call in session.calls)
    assert all(call[1].get("timeout") for call in session.calls)


def test_country_source_is_never_reported_as_global_coverage():
    spec = hdx.RESOURCES["djibouti"]
    session = Session(spec, body=csv_bytes(spec))
    report = hdx.check_resources(NOW, keys=("djibouti",), session=session)
    assert report["status"] == "PASS_SHADOW"
    assert report["resources"]["djibouti"]["scope"] == "COUNTRY_ONLY"
    assert report["ports_global_coverage"] is False


@pytest.mark.parametrize("bad,expected", [
    (b"", "EMPTY_CSV_DOWNLOAD"),
    (b"<html>sorry</html>", "NOT_CSV_PAYLOAD"),
    (b"date,portname,invalid\n2026-10-23,A,1\n", "CSV_SCHEMA_MISMATCH"),
    (csv_bytes(last=date(2026, 9, 25)), "CSV_STALE"),
    (csv_bytes(last=date(2026, 11, 5)), "CSV_FUTURE_OBSERVATION"),
    (csv_bytes(rows=1), "CSV_INSUFFICIENT_HISTORY"),
    (csv_bytes(value="-1"), "CSV_INVALID_METRIC"),
    (csv_bytes(value="NaN"), "CSV_INVALID_METRIC"),
    (csv_bytes(value="inf"), "CSV_INVALID_METRIC"),
])
def test_bad_downloads_fail_closed_without_any_observations(bad, expected):
    report = hdx.check_resources(NOW, keys=("chokepoints",), session=Session(body=bad))
    assert report["status"] == "FAIL_SHADOW"
    assert report["resources"]["chokepoints"]["error"] == expected
    assert report["production_fallback_enabled"] is False


@pytest.mark.parametrize("options,expected", [
    ({"wrong_org": True}, "HDX_PUBLISHER_MISMATCH"),
    ({"bad_id": True}, "RESOURCE_ID_MISMATCH"),
    ({"resource_name": "other.csv"}, "RESOURCE_NAME_MISMATCH"),
    ({"fail_package_link": True}, "RESOURCE_PACKAGE_LINK_MISMATCH"),
])
def test_unverified_resource_provenance_is_blocked(options, expected):
    report = hdx.check_resources(NOW, keys=("chokepoints",), session=Session(**options))
    assert report["resources"]["chokepoints"]["error"] == expected
    assert report["status"] == "FAIL_SHADOW"


@pytest.mark.parametrize("url", [
    "http://data.humdata.org/evil.csv",
    "https://data.humdata.org.evil.example/hijack",
    "https://private.example.com/file.csv",
    "https://data.humdata.org@internal.example.net/secrets",
    "file:///etc/passwd",
])
def test_untrusted_download_urls_are_never_allowed(url):
    with pytest.raises(hdx.HDXShadowError, match="UNTRUSTED_DOWNLOAD_URL"):
        hdx._validate_url(url)


def test_redirect_to_untrusted_host_never_follows():
    session = Session(redirect="https://evil.example/mirror.csv")
    result = hdx.check_resources(NOW, keys=("chokepoints",), session=session)
    assert result["resources"]["chokepoints"]["error"] == "UNTRUSTED_DOWNLOAD_URL"
    assert len(session.calls) == 3   # resource, package, initial download; no evil request
    # Audit 2026-11-07: der Bericht nennt den abgelehnten Host (nur Hostname) und die Stelle, nie Pfad/Query/Token
    assert result["resources"]["chokepoints"]["rejected_host"] == "evil.example"
    assert result["resources"]["chokepoints"]["rejected_stage"] == "redirect"


def test_rejected_host_never_leaks_credentials_or_paths():
    err = hdx._untrusted("https://user:pw@files.example.org/x.csv?X-Amz-Signature=abc", "resource_url")
    assert (str(err), err.host, err.stage) == ("UNTRUSTED_DOWNLOAD_URL", "files.example.org", "resource_url")
    assert hdx._untrusted(None, "request").host is None and hdx._untrusted("file:///etc/passwd", "x").host is None


def test_download_size_limit_blocks_even_http_200():
    session = Session(body=b"123456789")
    with pytest.raises(hdx.HDXShadowError, match="CSV_SIZE_LIMIT"):
        hdx._read_limited(session, URL, max_bytes=8)


def test_unknown_and_duplicate_resource_selections_block():
    for keys in ((), ("unknown",), ("chokepoints", "chokepoints")):
        with pytest.raises(ValueError):
            hdx.check_resources(NOW, keys=keys, session=Session())


def test_result_cannot_be_used_as_primary_observation():
    report = hdx.check_resources(NOW, keys=("chokepoints",), session=Session())
    assert "observations" not in report
    assert "latest_observation_time" not in report
    assert report["resources"]["chokepoints"]["status"] == "PASS"
    assert report["status"] != "PASS"    # deliberately not a SourceStatus.PASS


def test_download_http_error_fails_only_shadow():
    class Failing(Session):
        def get(self, url, **kwargs):
            if url == URL:
                return Response(code=503)
            return super().get(url, **kwargs)
    result = hdx.check_resources(NOW, keys=("chokepoints",), session=Failing())
    assert result["status"] == "FAIL_SHADOW"
    assert result["resources"]["chokepoints"]["error"] == "HTTP_OR_IO_HTTPError"



def test_invalid_hdx_json_does_not_crash_or_skip_reporting():
    class InvalidJson(Session):
        def get(self, url, **kwargs):
            if url.endswith("/resource_show"):
                class BadResponse(Response):
                    def json(self):
                        raise ValueError("invalid JSON")
                return BadResponse()
            return super().get(url, **kwargs)
    result = hdx.check_resources(NOW, keys=("chokepoints",), session=InvalidJson())
    assert result["status"] == "FAIL_SHADOW"
    assert result["resources"]["chokepoints"]["error"] == "HDX_RESOURCE_SHOW_INVALID_JSON"


def test_missing_cell_and_duplicate_header_are_rejected():
    bad = [
        b"date,portname,n_total\n2026-10-23\n",
        b"date,portname,portname,n_total\n2026-10-23,A,B,1\n",
    ]
    for payload in bad:
        r = hdx.check_resources(NOW, keys=("chokepoints",), session=Session(body=payload))
        assert r["status"] == "FAIL_SHADOW"


def test_all_eight_country_ids_and_global_chokepoints_are_pinned():
    assert len(hdx.RESOURCES) == 9
    assert len({item.resource_id for item in hdx.RESOURCES.values()}) == 9
    assert sorted(k for k, v in hdx.RESOURCES.items() if v.family == "ports") == [
        "djibouti", "kenya", "lebanon", "mozambique", "oman", "somalia", "sudan", "yemen"]
    assert hdx.RESOURCES["chokepoints"].metric == "n_total"
    assert hdx.DEFAULT_KEYS == ("djibouti", "chokepoints")

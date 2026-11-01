"""Regressionstests zum Source-Health-Audit 2026-11-01 (docs/AUDIT_2026-11-01_SOURCE_HEALTH.md).

1. Recovery: Registry-Quellen zählen eine Recovery-Prüfung nur mit einem NEUEN erfolgreichen Ingest
   (2026-10-27: finra_short_interest wurde aus einem einzigen Ingest in zwei Prüfungen als 2/2 bestätigt).
2. PortWatch Disruptions: outFields nur aus den Layer-Metadaten (vorher 6 ungeprüfte Feldnamen).
3. PortWatch-Query-Fehler: minimale read-only Isolation statt einer nicht eingrenzbaren 400-Meldung.
Kein Netz: alle ArcGIS-Antworten sind Fakes.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from modules import source_health as sh
from modules.external.sources import maritime as mw
from tests.test_maritime_source import (
    IDAI,
    NOW,
    SERVICE_URLS,
    FakeResponse,
    _disruptions_fetch,
    _load,
    make_fake_fetch,
)

T0 = datetime(2026, 10, 27, 12, 16, 40, tzinfo=timezone.utc)
ARCGIS_400 = {"error": {"code": 400, "message": "Cannot perform query. Invalid query parameters.",
                        "details": ["Unable to perform query. Please check your parameters."]}}


def _cfg():
    base = sh.load_config()
    return {**base, "core_sources": [], "alt_sources": [], "cache_max_age_days": {}, "downstream_extra": {}}


def _registry_check(last_success, status="PASS", failures=0):
    return {"source_id": "finra_short_interest", "kind": "registry", "criticality": "NON_CRITICAL",
            "registry_status": status, "registry_message": "" if status == "PASS" else "Ingest-Fehler",
            "registry_failures": failures, "last_success": last_success.isoformat() if last_success else None,
            "latest_observation": (T0 - timedelta(days=2)).isoformat(), "max_observation_age_days": 30,
            "staleness": "FRESH", "missing_rate": 0.0}


def _snap(check, prev, now):
    return sh.build_snapshot([check], prev, _cfg(), now, deps={})


# ── 1. Recovery ────────────────────────────────────────────────────────────
def test_registry_recovery_counts_only_new_ingest_successes():
    before = T0 - timedelta(days=4)
    s0 = _snap(_registry_check(before, status="FAIL", failures=3), None, T0 - timedelta(days=1))
    assert s0["sources"]["finra_short_interest"]["status"] == sh.BROKEN
    s1 = _snap(_registry_check(T0), s0, T0 + timedelta(hours=4))                      # neuer Ingest 12:16
    e1 = s1["sources"]["finra_short_interest"]
    assert e1["status"] == sh.DEGRADED and e1["recovery_passes"] == 1
    s2 = _snap(_registry_check(T0), s1, T0 + timedelta(hours=6))                      # derselbe Ingest, 2. Prüfung
    e2 = s2["sources"]["finra_short_interest"]
    assert e2["status"] == sh.DEGRADED, "zweite Prüfung desselben Ingests darf Recovery nicht bestätigen"
    assert e2["recovering"] and e2["recovery_passes"] == 1
    assert any("kein neuer erfolgreicher Abruf" in r for r in e2["reasons"])
    s3 = _snap(_registry_check(T0 + timedelta(days=1)), s2, T0 + timedelta(days=1, hours=4))   # neuer Ingest
    e3 = s3["sources"]["finra_short_interest"]
    assert e3["status"] == sh.HEALTHY and not e3["recovering"] and e3["recovery_passes"] == 0


def test_recovery_without_any_new_success_never_turns_healthy():
    before = T0 - timedelta(days=4)
    snap = _snap(_registry_check(before, status="FAIL", failures=3), None, T0 - timedelta(days=1))
    for h in (4, 6, 8):                                     # Fehler behoben, aber kein neuer erfolgreicher Ingest
        snap = _snap(_registry_check(before), snap, T0 + timedelta(hours=h))
        e = snap["sources"]["finra_short_interest"]
        assert e["status"] == sh.DEGRADED and e["recovering"] and e["recovery_passes"] == 0


def test_recovery_failure_during_recovery_still_breaks_off():
    before = T0 - timedelta(days=4)
    s0 = _snap(_registry_check(before, status="FAIL", failures=3), None, T0 - timedelta(days=1))
    s1 = _snap(_registry_check(T0), s0, T0 + timedelta(hours=4))
    s2 = _snap(_registry_check(T0, status="FAIL", failures=1), s1, T0 + timedelta(days=1))
    assert s2["sources"]["finra_short_interest"]["status"] in (sh.DEGRADED, sh.BROKEN)
    assert not s2["sources"]["finra_short_interest"]["recovering"]


# ── 2. Disruptions: outFields nur aus Metadaten ────────────────────────────
def _strict_layer(fields, sent):
    """ArcGIS Online: ein unbekanntes Feld in outFields -> HTTP 200 mit 400-Fehlerobjekt."""
    base = _disruptions_fetch([IDAI], fields=fields)

    def fetch(url, params=None, **kwargs):
        params = params or {}
        if url.endswith("/query"):
            sent.append(params)
            asked = params["outFields"].split(",")
            if set(asked) - set(fields):
                return FakeResponse(ARCGIS_400)
            return FakeResponse({"features": [{"attributes": {k: IDAI[k] for k in asked}}],
                                 "exceededTransferLimit": False})
        return base(url, params=params, **kwargs)
    return fetch


def test_disruptions_request_only_fields_listed_in_layer_metadata():
    fields = [f for f in mw.DISRUPTION_FIELDS if f not in ("editdate", "n_affectedports")]
    sent: list = []
    res = mw.PortWatchDisruptionsConnector({}, http_fetch=_strict_layer(fields, sent)).fetch(NOW)
    assert res.status.value == "PASS", res.message                 # ungenutzte Felder: kein Datenverlust
    assert len([o for o in res.observations if o.metric == "disruption_alert_level"]) == 2
    assert len(sent) == 1 and "editdate" not in sent[0]["outFields"] and "n_affectedports" not in sent[0]["outFields"]
    # Point-in-Time unverändert: Ereignisbeginn als observation_time, Verfügbarkeit = Abrufzeit
    assert all(o.observation_time.year == 2019 and o.available_at == NOW == o.retrieved_at for o in res.observations)
    assert "nicht im Layer" in res.message and "editdate" in res.message


def test_disruptions_missing_descriptive_field_is_warn_not_pass():
    fields = [f for f in mw.DISRUPTION_FIELDS if f != "severitytext"]
    sent: list = []
    res = mw.PortWatchDisruptionsConnector({}, http_fetch=_strict_layer(fields, sent)).fetch(NOW)
    assert res.status.value == "WARN" and "severitytext" in res.message
    assert all(o.attrs["severitytext"] is None for o in res.observations)    # nie ergänzt


def test_disruptions_full_schema_unchanged_request():
    sent: list = []
    res = mw.PortWatchDisruptionsConnector({}, http_fetch=_strict_layer(list(mw.DISRUPTION_FIELDS), sent)).fetch(NOW)
    assert res.status.value == "PASS" and sent[0]["outFields"] == ",".join(mw.DISRUPTION_FIELDS)
    assert sent[0]["where"] == "1=1" and "orderByFields" not in sent[0]


# ── 3. Isolation von ArcGIS-Query-Fehlern ──────────────────────────────────
def test_portwatch_query_error_is_isolated_with_minimal_read_only_probes(tmp_path):
    base, _ = make_fake_fetch()
    meta_url, q_url = SERVICE_URLS["ports_daily"] + "/0", SERVICE_URLS["ports_daily"] + "/0/query"
    probes: list = []

    def fetch(url, params=None, **kwargs):
        params = params or {}
        if url == meta_url:
            meta = _load("layer_metadata_ports_daily.json")
            fields = [dict(f, type="esriFieldTypeDateOnly") if f["name"] == "date" else f for f in meta["fields"]]
            return FakeResponse({**meta, "fields": fields})
        if url == q_url:
            if params.get("resultRecordCount") == 1:
                probes.append(dict(params))
            if "DATE '" in str(params.get("where", "")):          # simulierter Defekt: nur die Datumsklausel
                return FakeResponse(ARCGIS_400)
            if params.get("resultRecordCount") == 1:
                return FakeResponse({"features": [], "exceededTransferLimit": False})
        return base(url, params=params, **kwargs)

    res = mw.PortWatchPortsConnector(http_fetch=fetch, cache_path=str(tmp_path / "ep.json")).fetch(NOW)
    assert res.status.value == "FAIL" and not res.observations                # fail-closed, keine Daten
    assert res.message.startswith("[Isolation basis=OK where=FAIL(400) outFields=OK orderByFields=OK]")
    assert len(probes) == 4
    assert all(p["resultRecordCount"] == 1 and p["returnGeometry"] == "false" for p in probes)
    assert probes[0] == {"where": "1=1", "outFields": "OBJECTID", "resultRecordCount": 1,
                         "returnGeometry": "false", "f": "json"}           # Objekt-ID-Feld aus den Metadaten
    assert "resultOffset=0" in res.message and "where=date >= DATE '" in res.message
    assert "Unable to perform query" in res.message


def test_isolation_probes_failing_page_offset():
    calls: list = []

    def fetch(url, params=None, **kwargs):
        calls.append(params)
        bad = params.get("resultOffset", 0) >= 4000
        return FakeResponse(ARCGIS_400 if bad else {"features": []})

    iso = mw.isolate_query_error("https://x/FeatureServer", 0, fetch, where="1=1", out_fields="OBJECTID,a",
                                 order_by="date ASC", offset=4000, base_field="OBJECTID")
    assert mw.isolation_summary(iso) == "basis=OK outFields=OK orderByFields=OK resultOffset=FAIL(400)"
    assert len(calls) == 4 and all(c["resultRecordCount"] == 1 for c in calls)


def test_isolation_network_error_never_masks_original_failure():
    def fetch(url, params=None, **kwargs):
        if params.get("resultRecordCount") == 1:
            raise mw.FetchError("timeout")
        return FakeResponse(ARCGIS_400)

    try:
        mw.query_features("https://x/FeatureServer", 0, "1=1", ["a"], fetch, isolation_field="OBJECTID")
    except mw.ArcGISQueryError as e:
        assert "basis=ERROR(FetchError)" in str(e) and "ArcGIS-Query-Fehler" in str(e)
        assert isinstance(e, mw.FetchError)                         # Konnektoren behandeln ihn wie bisher
    else:  # pragma: no cover
        raise AssertionError("ArcGIS-Fehler muss als FetchError durchschlagen")


def test_recovery_survives_restart_via_persisted_snapshot(tmp_path):
    """Neustart = neuer Prozess, der nur die gespeicherte Snapshot-Datei liest (sh.check, kein Mail-Versand)."""
    c = _cfg()

    def run(check, now):
        orig = sh.gather
        sh.gather = lambda *a, **k: [check]
        try:
            return sh.check(now=now, cfg=c, probes=False, mail=False, out_dir=tmp_path)["snapshot"]
        finally:
            sh.gather = orig

    before = T0 - timedelta(days=4)
    run(_registry_check(before, status="FAIL", failures=3), T0 - timedelta(days=1))
    assert run(_registry_check(T0), T0 + timedelta(hours=4))["sources"]["finra_short_interest"]["recovery_passes"] == 1
    for h in (6, 8, 10):                                            # wiederholte Läufe, je frischer Prozess
        e = run(_registry_check(T0), T0 + timedelta(hours=h))["sources"]["finra_short_interest"]
        assert e["status"] == sh.DEGRADED and e["recovery_passes"] == 1
    e = run(_registry_check(T0 + timedelta(days=1)), T0 + timedelta(days=1, hours=4))["sources"]["finra_short_interest"]
    assert e["status"] == sh.HEALTHY


def test_isolation_respects_probe_timeout_and_budget():
    calls: list = []

    def fetch(url, params=None, **kwargs):
        calls.append(kwargs)
        return FakeResponse({"features": []})

    mw.isolate_query_error("https://x/FeatureServer", 0, fetch, where="d >= DATE '2026-10-01'", out_fields="OBJECTID,a",
                           order_by="d ASC", offset=2000, base_field="OBJECTID")
    assert len(calls) == 5                                                        # Obergrenze 5 Probes
    assert all(k == {"retries": 1, "timeout": mw.ISOLATION_PROBE_TIMEOUT_S} for k in calls)
    calls.clear()
    iso = mw.isolate_query_error("https://x/FeatureServer", 0, fetch, where="1=1", out_fields="OBJECTID,a",
                                 order_by=None, offset=0, base_field="OBJECTID", budget_s=-1)
    assert not calls and all(v["result"] == "SKIPPED(Budget)" for v in iso.values())


def test_successful_diagnosis_never_reports_a_repaired_source(tmp_path):
    """Alle Einzelprobes OK, die eigentliche Abfrage scheitert weiter -> FAIL, keine Daten."""
    base, _ = make_fake_fetch()
    q_url = SERVICE_URLS["ports_daily"] + "/0/query"

    def fetch(url, params=None, **kwargs):
        params = params or {}
        if url == q_url:
            if params.get("resultRecordCount") == 1:
                return FakeResponse({"features": [{"attributes": {"OBJECTID": 1}}], "exceededTransferLimit": False})
            return FakeResponse(ARCGIS_400)
        return base(url, params=params, **kwargs)

    res = mw.PortWatchPortsConnector(http_fetch=fetch, cache_path=str(tmp_path / "ep.json")).fetch(NOW)
    assert res.status.value == "FAIL" and not res.observations and not res.raw
    assert res.message.startswith("[Isolation basis=OK where=OK outFields=OK orderByFields=OK]")

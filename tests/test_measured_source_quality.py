"""Audit 2026-11-07 (P2/P3/P6): gemessene statt metadatenbasierte Quellenbewertung, Katalog gegen Registry.

Vorher (echte Daten): research_director.onboarding_checklist meldete "timestamp_integrity ok"/"historical_coverage ok"
allein aus dem Katalogtext – auch für I/B/E/S (kommerziell, kein Zugang: zusätzlich "revision_analysis ok").
wikipedia_pageviews und finra_short_interest standen als "candidate" im Katalog und wurden als NEUE Datenquelle
vorgeschlagen, obwohl wikimedia_pageviews/finra_short_interest seit 15.10. im Archiv laufen. 12 Datenlücken-Vorschläge,
davon 4 bereits laufende Quellen. Nachher: 8 Vorschläge (alle "declared_unverified"), 5 laufende Quellen mit
gemessenem Profil (PortWatch: access failing). Synthetische Daten – kein Alpha-Nachweis.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from modules import research_director as rd
from modules.external import data_quality as dq
from modules.external.pit import AvailabilityPrecision as AP
from modules.external.pit import Observation

NOW = datetime(2026, 11, 7, 12, tzinfo=timezone.utc)


def _obs(day, value=1.0, avail=None, prec=AP.EXACT_TIMESTAMP, vintage=None, series="s", entity="e"):
    t = datetime(2026, 10, 1, tzinfo=timezone.utc) + timedelta(days=day)
    a = avail or t + timedelta(days=1)
    return Observation(source_id="src", dataset="d", series_id=series, entity_id=entity, metric="m", value=value,
                       unit="u", observation_time=t, available_at=a, retrieved_at=a, availability_precision=prec,
                       parser_version="p", vintage_time=vintage)


def test_archive_profile_measures_pit_history_revisions_and_defects():
    v1, v2 = datetime(2026, 10, 3, tzinfo=timezone.utc), datetime(2026, 10, 9, tzinfo=timezone.utc)
    obs = [_obs(0, 1.0, avail=v1, vintage=v1), _obs(0, 1.2, avail=v2, vintage=v2),     # revidiert
           _obs(1, 2.0, avail=v1, vintage=v1), _obs(1, 2.0, avail=v2, vintage=v2),     # gleicher Wert
           _obs(2, None), _obs(3, 4.0, avail=datetime(2026, 9, 1, tzinfo=timezone.utc)),  # vor der Periode
           _obs(4, 5.0, prec=AP.UNKNOWN)]
    p = dq.archive_profile(obs, NOW)
    assert p["n_rows"] == 7 and p["n_keys"] == 5 and p["keys_with_multiple_vintages"] == 2
    assert p["keys_revised"] == 1 and p["revision_rate"] == 0.2 and p["available_before_period"] == 1
    assert p["pit_history_from"] == "2026-09-01" and p["null_share"] == round(1 / 7, 4)
    assert p["confirmatory_share"] == round(6 / 7, 4)
    assert dq.archive_profile([], NOW) == {"n_rows": 0}
    assert dq.archive_profile([_obs(0)], NOW)["revision_rate"] == 0.0              # gemessen: keine Revision


IBES = {"id": "ibes", "provider": "Refinitiv I/B/E/S", "license": "kommerziell, REVIEW_REQUIRED", "pit": "vintages",
        "history_from": 1980, "status": "candidate", "dimension": ["earnings_momentum"]}


def test_catalog_claims_are_never_reported_as_ok_without_measurement():
    chk = rd.onboarding_checklist(IBES)
    assert chk["timestamp_integrity"] == chk["historical_coverage"] == chk["revision_analysis"] == "declared_unverified"
    assert "ok" not in {chk[k] for k in ("timestamp_integrity", "historical_coverage", "revision_analysis")}
    al = rd.active_learning({}, [], [IBES])[0]
    assert al["available"] is False and set(al["unverified"]) == {"historical_coverage", "timestamp_integrity",
                                                                   "revision_analysis"}


def test_measured_profile_decides_each_step():
    good = {"n_rows": 10, "pit_history_from": "2015-01-08", "confirmatory_share": 1.0, "available_before_period": 0,
            "revision_rate": 1.0, "null_share": 0.0, "sources": {"a": {"health_status": "PASS", "staleness": "FRESH"}}}
    chk = rd.onboarding_checklist({**IBES, "measured": good})
    assert (chk["access"], chk["historical_coverage"], chk["timestamp_integrity"], chk["revision_analysis"],
            chk["missing_data_analysis"]) == ("ok", "ok", "ok", "measured", "ok")
    bad = {**good, "pit_history_from": "2026-09-27", "confirmatory_share": 0.5, "revision_rate": None,
           "null_share": 0.9, "sources": {"a": {"health_status": "FAIL", "staleness": "FRESH"}}}
    chk = rd.onboarding_checklist({**IBES, "measured": bad})
    assert (chk["access"], chk["historical_coverage"], chk["timestamp_integrity"], chk["revision_analysis"],
            chk["missing_data_analysis"]) == ("failing", "insufficient", "failing", "not_observable", "high_missing")
    stale = {**good, "sources": {"a": {"health_status": "PASS", "staleness": "STALE"}}}
    assert rd.onboarding_checklist({**IBES, "measured": stale})["access"] == "failing"


class _Archive:
    def __init__(self, data):
        self.data = data

    def load(self, sid):
        return self.data.get(sid, [])


def test_registry_backed_catalog_entry_is_integrated_not_a_new_source(monkeypatch):
    """Realfall: wikipedia_pageviews (Katalog, candidate) = wikimedia_pageviews (Archiv, läuft) – kein Duplikat."""
    monkeypatch.setattr(rd, "_load", lambda p, d: d)
    wiki = {"id": "wikipedia_pageviews", "provider": "Wikimedia", "license": "CC0", "pit": "publication_date",
            "history_from": 2015, "status": "candidate", "dimension": ["attention"],
            "registry_source_ids": ["wikimedia_pageviews"]}
    health = {"sources": {"wikimedia_pageviews": {"status": "PASS"}}}
    cat = rd.annotate_catalog([wiki, IBES], archive=_Archive({"wikimedia_pageviews": [_obs(35, 9.0)]}),
                              health=health, now=NOW)
    w = cat[0]
    assert w["status"] == "integrated" and w["declared_status"] == "candidate"
    assert [a["source"] for a in rd.active_learning({}, [], cat)] == ["ibes"]
    q = rd.integrated_quality(cat)
    assert [x["source"] for x in q] == ["wikipedia_pageviews"]
    assert q[0]["onboarding_checklist"]["historical_coverage"] == "insufficient"     # PIT erst ab Archivbeginn
    assert "historical_coverage" in q[0]["issues"] and q[0]["onboarding_checklist"]["access"] == "ok"
    gap = {"gap_id": "model:no_skill:5d", "kind": "model", "horizon": 5, "severity": 0.8, "finding": "kein Skill"}
    srcs = {(p.get("change") or {}).get("source") for p in rd.option_proposals([gap], None, cat)
            if p["kind"] == "data_source"}
    assert srcs == {"ibes"}                                                         # kein Duplikat-Connector


def test_entry_without_archive_rows_keeps_declared_status(monkeypatch):
    monkeypatch.setattr(rd, "_load", lambda p, d: d)
    e = {**IBES, "registry_source_ids": ["nothing_yet"]}
    cat = rd.annotate_catalog([e], archive=_Archive({}), health={}, now=NOW)
    assert cat[0]["status"] == "candidate" and cat[0]["measured"]["n_rows"] == 0

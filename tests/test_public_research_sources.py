"""Neue öffentliche Research-Quellen (A4, B3, B5, B7, B8, B9, B10) – nur SYNTHETISCHE Fixtures im dokumentierten
Format, http.fetch gemockt. Geprüft: Parser, PIT (available_at nie vor dem Abruf bzw. dem letzten Filing),
Schemafehler laut (SCHEMA_CHANGED), fehlende Werte nie 0, Registry-Einträge RESEARCH_SHADOW.
Die Live-Formate werden zusätzlich im Preflight-Workflow geprüft."""
from __future__ import annotations

import io
import json
import zipfile
from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from modules.external import http
from modules.external.registry import SourceRegistry, discover_connectors
from modules.external.sources import calendars as cal
from modules.external.sources import company_events as ce
from modules.external.sources import market_public as mp
from modules.external.sources.base import SourceStatus

UTC = timezone.utc
NOW = datetime(2026, 10, 15, 12, 0, tzinfo=UTC)
NEW = ("nasdaq100_members", "finra_short_volume", "finra_short_interest", "wikimedia_pageviews", "fomc_calendar",
       "bls_release_calendar", "bea_release_calendar", "clinicaltrials_sponsors", "usaspending_contracts",
       "sec_13f_holdings", "fred_liquidity")


def resp(content, url="https://fixture", status=200, when=NOW):
    body = content if isinstance(content, bytes) else (json.dumps(content) if not isinstance(content, str)
                                                       else content).encode()
    return http.FetchResult(url=url, status=status, content=body, content_type="x", retrieved_at=when,
                            content_hash="h", fingerprint="fp", bytes=len(body))


def test_registry_entries_are_research_shadow_with_connectors():
    reg, conns = SourceRegistry(), discover_connectors()
    for sid in NEW:
        cfg = reg.sources[sid]
        assert sid in conns, sid
        assert cfg["role"] == "RESEARCH_SHADOW" and cfg["research_only"] is True and cfg["criticality"] == "low", sid


# ── A4 Nasdaq-100 ────────────────────────────────────────────────────────────
def _ndx(n=101):
    return {"data": {"data": {"headers": {"symbol": "Symbol"}, "rows": [{"symbol": f"T{i:03d}"} for i in range(n - 1)]
                              + [{"symbol": "BRK.B"}]}}, "status": {"rCode": 200}}


def test_nasdaq100_parser_and_snapshot():
    assert len(mp.parse_nasdaq100(_ndx())) == 101 and "BRK-B" in mp.parse_nasdaq100(_ndx())
    with pytest.raises(ValueError):
        mp.parse_nasdaq100(_ndx(20))                                   # Plausibilitätsgrenze
    with pytest.raises(ValueError):
        mp.parse_nasdaq100({"data": {}})
    with patch.object(mp.http, "fetch", return_value=resp(_ndx())):
        r = mp.Nasdaq100MembersConnector({}).fetch(NOW)
    [o] = r.observations
    assert r.status == SourceStatus.PASS and o.value == 101 and len(o.attrs["symbols"]) == 101
    assert o.available_at == NOW and o.observation_time == datetime(2026, 10, 15, tzinfo=UTC)
    with patch.object(mp.http, "fetch", return_value=resp({"data": None})):
        assert mp.Nasdaq100MembersConnector({}).fetch(NOW).status == SourceStatus.SCHEMA_CHANGED


def test_universe_v1_definition_untouched():
    """A4 archiviert nur: get_universe gehört zur eingefrorenen UNIVERSE_V1-Definition (Code-Hash) und bleibt
    unverändert – eine Umstellung der Produktionsquelle wäre eine eigene, freizugebende V1-Änderung."""
    from modules import universe_v2 as uv2
    assert uv2.v1_unchanged()


# ── B3 FINRA ─────────────────────────────────────────────────────────────────
SHVOL = ("Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
         "20261014|AAPL|4000000|10000|10000000|B,Q,N\n"
         "20261014|BRK.B|100|0|0|N\n"                                 # Volumen 0 -> kein Verhältnis
         "20261014|ZZZZ|5|0|10|Q\n")                                   # nicht im Universum


def test_finra_short_volume(monkeypatch):
    rows = mp.parse_short_volume(SHVOL, {"AAPL", "BRK-B"})
    assert [r["symbol"] for r in rows] == ["AAPL", "BRK-B"]
    with pytest.raises(ValueError):
        mp.parse_short_volume("foo|bar\n1|2")
    monkeypatch.setattr(mp, "universe_tickers", lambda: ["AAPL", "BRK-B"])

    def fake(url, **kw):
        if "20261014" in url:
            return resp(SHVOL, url=url)
        raise http.FetchError("403")                                   # andere Tage: noch nicht/kein Handelstag

    with patch.object(mp.http, "fetch", side_effect=fake):
        r = mp.FinraShortVolumeConnector({"lookback_business_days": 3}).fetch(NOW)
    [o] = r.observations
    assert r.status == SourceStatus.PASS and o.entity_id == "AAPL" and o.value == 0.4
    assert o.observation_time == datetime(2026, 10, 14, tzinfo=UTC) and o.available_at == NOW
    assert o.attrs["total_volume"] == 10_000_000 and "fehlend" in r.message


def test_finra_short_interest_auth_message_and_parse(monkeypatch):
    monkeypatch.setattr(mp, "universe_tickers", lambda: ["AAPL"])
    monkeypatch.delenv("FINRA_API_CLIENT_ID", raising=False)
    with patch.object(mp.http, "fetch", side_effect=http.AuthError("401")):
        r = mp.FinraShortInterestConnector({}).fetch(NOW)
    assert r.status == SourceStatus.AUTH_MISSING and "FINRA_API_CLIENT_SECRET" in r.message
    data = [{"symbolCode": "AAPL", "settlementDate": "2026-09-30", "currentShortPositionQuantity": 1.2e8,
             "averageDailyVolumeQuantity": 5e7, "daysToCoverQuantity": 2.4},
            {"symbolCode": "XYZ", "settlementDate": "2026-09-30", "currentShortPositionQuantity": 5}]
    with patch.object(mp.http, "fetch", return_value=resp(data)):
        r = mp.FinraShortInterestConnector({}).fetch(NOW)
    assert r.status == SourceStatus.PASS
    assert {(o.metric, o.value) for o in r.observations} == {("short_interest_shares", 1.2e8), ("days_to_cover", 2.4)}
    with patch.object(mp.http, "fetch", return_value=resp([{"foo": 1}])):
        assert mp.FinraShortInterestConnector({}).fetch(NOW).status == SourceStatus.SCHEMA_CHANGED


# ── B5 Wikimedia ─────────────────────────────────────────────────────────────
WIKI_HTML = """<html><body><table id="constituents"><tr><th>Symbol</th><th>Security</th></tr>
<tr><td><a href="https://www.nyse.com/quote/XNYS:MMM">MMM</a></td><td><a href="/wiki/3M">3M</a></td></tr>
<tr><td><a href="x">BRK.B</a></td><td><a href="/wiki/Berkshire_Hathaway">Berkshire Hathaway</a></td></tr>
<tr><td>ABC</td><td><a href="/wiki/File:Logo.png">x</a></td></tr></table></body></html>"""
NDX_HTML = """<table id="constituents"><tr><th>Company</th><th>Ticker</th></tr>
<tr><td><a href="/wiki/Apple_Inc.">Apple Inc.</a></td><td>AAPL</td></tr></table>"""


def test_wiki_list_parser_both_column_orders():
    assert mp.parse_wiki_list(WIKI_HTML) == {"MMM": "3M", "BRK-B": "Berkshire_Hathaway"}
    assert mp.parse_wiki_list(NDX_HTML) == {"AAPL": "Apple_Inc."}


def test_pageviews_connector(monkeypatch):
    def fake(url, **kw):
        if "List_of_S" in url:
            return resp(WIKI_HTML, url=url)
        if "Nasdaq-100" in url:
            return resp(NDX_HTML, url=url)
        art = url.split("/user/")[1].split("/")[0]
        if art == "3M":
            raise http.FetchError("404")
        return resp({"items": [{"article": art, "timestamp": "2026101300", "views": 1500},
                               {"article": art, "timestamp": "2026101400", "views": 1700}]}, url=url)

    with patch.object(mp.http, "fetch", side_effect=fake), patch.object(mp.time, "sleep"):
        r = mp.WikimediaPageviewsConnector({"min_articles": 3, "lookback_days": 2}).fetch(NOW)
    assert r.status == SourceStatus.WARN                                # 1 von 3 Artikeln fehlgeschlagen (> 10 %)
    assert {(o.entity_id, o.observation_time.day, o.value) for o in r.observations} == {
        ("BRK-B", 13, 1500.0), ("BRK-B", 14, 1700.0), ("AAPL", 13, 1500.0), ("AAPL", 14, 1700.0)}
    assert all(o.available_at == NOW for o in r.observations)
    with patch.object(mp.http, "fetch", side_effect=fake):
        assert mp.WikimediaPageviewsConnector({}).fetch(NOW).status == SourceStatus.SCHEMA_CHANGED   # < 300 Artikel


# ── B7 Kalender ──────────────────────────────────────────────────────────────
FOMC_HTML = """<h4><a id="1">2026 FOMC Meetings</a></h4>
<div class="row fomc-meeting"><div class="fomc-meeting__month col-xs-5"><strong>October</strong></div>
<div class="fomc-meeting__date col-xs-4">27-28</div></div>
<div class="row fomc-meeting"><div class="fomc-meeting__month col-xs-5"><strong>December</strong></div>
<div class="fomc-meeting__date col-xs-4">8-9*</div></div>
<h4><a id="2">2027 FOMC Meetings</a></h4>
<div class="row fomc-meeting"><div class="fomc-meeting__month"><strong>Jan/Feb</strong></div>
<div class="fomc-meeting__date">31-1</div></div>
<div class="row fomc-meeting"><div class="fomc-meeting__month"><strong>March</strong></div>
<div class="fomc-meeting__date">5 (notation vote)</div></div>"""


def test_fomc_parser_dst_and_month_span():
    ev = cal.parse_fomc(FOMC_HTML)
    assert [e["decision_et"].date() for e in ev] == [date(2026, 10, 28), date(2026, 12, 9), date(2027, 2, 1)]
    assert [e["sep"] for e in ev] == [False, True, False]
    assert ev[0]["decision_et"].astimezone(UTC).hour == 18 and ev[1]["decision_et"].astimezone(UTC).hour == 19
    with pytest.raises(ValueError):
        cal.parse_fomc("<html>nichts</html>")
    with patch.object(cal.http, "fetch", return_value=resp(FOMC_HTML)):
        r = cal.FomcCalendarConnector({}).fetch(NOW)
    assert r.status == SourceStatus.PASS and [o.value for o in r.observations] == [1.0, 2.0, 1.0]
    assert all(o.available_at == NOW for o in r.observations)


ICS = ("BEGIN:VCALENDAR\r\nVERSION:2.0\r\nBEGIN:VEVENT\r\nUID:1\r\nDTSTART;TZID=US-Eastern:20261113T083000\r\n"
       "SUMMARY:Consumer Price Index for October 2026\r\nEND:VEVENT\r\nBEGIN:VEVENT\r\nUID:2\r\n"
       "DTSTART:20261106T133000Z\r\nSUMMARY:Employment Situation for\r\n  October 2026\r\nEND:VEVENT\r\n"
       "BEGIN:VEVENT\r\nUID:3\r\nDTSTART;VALUE=DATE:20200101\r\nSUMMARY:Alt\r\nEND:VEVENT\r\nEND:VCALENDAR\r\n")


def test_ics_parser_and_connector():
    ev = cal.parse_ics(ICS)
    assert [e["summary"] for e in ev] == ["Consumer Price Index for October 2026", "Employment Situation for October 2026",
                                          "Alt"]
    assert ev[0]["start"].astimezone(UTC) == datetime(2026, 11, 13, 13, 30, tzinfo=UTC)    # unbekannte TZID -> NY
    with pytest.raises(ValueError):
        cal.parse_ics("<html>")
    with patch.object(cal.http, "fetch", return_value=resp(ICS)):
        r = cal.BlsReleaseCalendarConnector({}).fetch(NOW)
    assert r.status == SourceStatus.PASS
    assert {o.series_id for o in r.observations} == {"consumer_price_index", "employment_situation"}   # 2020 außerhalb


# ── Entity-Index (B8/B9/B10) ─────────────────────────────────────────────────
def rec(entity_id, ticker=None, usage="pit", name="", parent=None, industry=None, cik=None):
    return SimpleNamespace(entity_id=entity_id, ticker=ticker, cik=cik, usage=usage, canonical_name=name, aliases=[],
                           parent_entity=parent, industry=industry, valid_to=None)


RECORDS = [rec("cik:1", "PFE", "research_ticker_identity", cik="1"), rec("cik:1", name="PFIZER INC",
                                                                         industry="SIC 2834 Pharma"),
           rec("cik:2", "LMT", "research_ticker_identity", cik="2"), rec("cik:2", name="LOCKHEED MARTIN CORP",
                                                                         industry="SIC 3760 Missiles"),
           rec("lei:x", usage="exposure_subsidiary", name="SIKORSKY AIRCRAFT CORPORATION", parent="cik:2"),
           rec("cik:3", "GOOGL", "research_ticker_identity", cik="3"), rec("cik:3", "GOOG", "research_ticker_identity",
                                                                           cik="3"),
           rec("cik:3", name="ALPHABET INC")]


def test_entity_index_and_matching():
    idx = ce.entity_index(RECORDS)
    assert idx["tickers"]["cik:3"] == ["GOOG", "GOOGL"] and idx["sic"]["cik:1"] == "2834"
    assert ce.match_name("Lockheed Martin Corporation", idx) == ("cik:2", "direct")
    assert ce.match_name("SIKORSKY AIRCRAFT CORP", idx) == ("cik:2", "subsidiary")
    assert ce.match_name("ALPHABET INC CL A", idx) == ("cik:3", "direct")
    assert ce.match_name("Unknown LLC", idx) == (None, "")
    assert ce.sponsor_matches("Pfizer", "pfizer") and not ce.sponsor_matches("Pfizerish Labs", "pfizer")


IDX = ce.entity_index(RECORDS)


# ── B8 ClinicalTrials ────────────────────────────────────────────────────────
def study(nct, status, phases, completion, updated, sponsor="Pfizer"):
    return {"protocolSection": {"identificationModule": {"nctId": nct},
                                "statusModule": {"overallStatus": status,
                                                 "primaryCompletionDateStruct": {"date": completion},
                                                 "lastUpdatePostDateStruct": {"date": updated}},
                                "designModule": {"phases": phases},
                                "sponsorCollaboratorsModule": {"leadSponsor": {"name": sponsor}}}}


def test_clinicaltrials_metrics_and_connector(monkeypatch):
    monkeypatch.setattr(ce, "entity_index", lambda: IDX)
    page = {"studies": [study("N1", "RECRUITING", ["PHASE3"], "2026-12", "2026-10-12"),
                        study("N2", "ACTIVE_NOT_RECRUITING", ["PHASE2"], "2028-01-15", "2026-01-01"),
                        study("N3", "COMPLETED", ["PHASE3"], "2025-01", "2026-10-14"),
                        study("N4", "RECRUITING", ["PHASE3"], "2026-11", "2026-10-14", sponsor="Other Pharma")]}
    with patch.object(ce.http, "fetch", return_value=resp(page)), patch.object(ce.time, "sleep"):
        r = ce.ClinicalTrialsConnector({}).fetch(NOW)
    got = {o.metric: o.value for o in r.observations}
    assert got == {"ct_active_trials": 2.0, "ct_active_phase3": 1.0, "ct_primary_completion_90d": 1.0,
                   "ct_updated_7d": 2.0}
    assert {o.entity_id for o in r.observations} == {"PFE"} and r.status == SourceStatus.PASS
    with patch.object(ce.http, "fetch", return_value=resp({"x": 1})), patch.object(ce.time, "sleep"):
        assert ce.ClinicalTrialsConnector({}).fetch(NOW).status == SourceStatus.SCHEMA_CHANGED



# ── B9 USAspending ───────────────────────────────────────────────────────────
def test_usaspending_aggregation(monkeypatch):
    monkeypatch.setattr(ce, "entity_index", lambda: IDX)
    pages = [{"results": [{"Award ID": "A1", "Recipient Name": "LOCKHEED MARTIN CORPORATION", "Award Amount": 1e9},
                          {"Award ID": "A2", "Recipient Name": "SIKORSKY AIRCRAFT CORPORATION", "Award Amount": 5e8},
                          {"Award ID": "A3", "Recipient Name": "SMALL VENDOR LLC", "Award Amount": 1e6}],
              "page_metadata": {"page": 1, "hasNext": False}}]
    calls = []

    def fake(url, **kw):
        calls.append(kw["json_body"])
        return resp(pages[0])

    with patch.object(ce.http, "fetch", side_effect=fake):
        r = ce.UsaSpendingConnector({"window_days": 7}).fetch(NOW)
    assert calls[0]["filters"]["time_period"] == [{"start_date": "2026-10-08", "end_date": "2026-10-14"}]
    got = {o.metric: o for o in r.observations}
    assert got["federal_contracts_active_7d_usd"].value == 1.5e9 and got["federal_contracts_active_7d_count"].value == 2
    assert got["federal_contracts_active_7d_usd"].attrs["via_subsidiary"] == 1
    with patch.object(ce.http, "fetch", return_value=resp({"results": [{"foo": 1}]})):
        assert ce.UsaSpendingConnector({}).fetch(NOW).status == SourceStatus.SCHEMA_CHANGED


# ── B10 SEC 13F ──────────────────────────────────────────────────────────────
def _zip13f() -> bytes:
    sub = ("ACCESSION_NUMBER\tFILING_DATE\tSUBMISSIONTYPE\tCIK\tPERIODOFREPORT\n"
           "A1\t01-AUG-2026\t13F-HR\t900\t30-JUN-2026\n"
           "A2\t14-AUG-2026\t13F-HR\t901\t30-JUN-2026\n"
           "A3\t20-AUG-2026\t13F-HR\t902\t30-JUN-2026\n"          # verspätet (> 45 T)
           "A4\t02-AUG-2026\t13F-HR/A\t903\t30-JUN-2026\n")       # Amendment
    info = ("ACCESSION_NUMBER\tNAMEOFISSUER\tTITLEOFCLASS\tCUSIP\tVALUE\tSSHPRNAMT\tSSHPRNAMTTYPE\tPUTCALL\n"
            "A1\tALPHABET INC\tCAP STK CL A\t02079K305\t1\t100\tSH\t\n"
            "A1\tALPHABET INC\tCAP STK CL C\t02079K107\t1\t50\tSH\t\n"
            "A2\tALPHABET INC\tCAP STK CL A\t02079K305\t1\t200\tSH\t\n"
            "A2\tALPHABET INC\tCALL\t02079K305\t1\t999\tSH\tCall\n"
            "A3\tALPHABET INC\tCAP STK CL A\t02079K305\t1\t7777\tSH\t\n"
            "A4\tALPHABET INC\tCAP STK CL A\t02079K305\t1\t8888\tSH\t\n"
            "A1\tPFIZER INC\tCOM\t717081103\t1\t10\tPRN\t\n")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("SUBMISSION.tsv", sub)
        z.writestr("INFOTABLE.tsv", info)
    return buf.getvalue()


def test_13f_index_ranges_and_aggregation():
    html = ('<a href="/files/structureddata/data/form-13f-data-sets/01jun2026-31aug2026_form13f.zip">x</a>'
            '<a href="https://www.sec.gov/files/2023q4_form13f.zip">y</a>')
    idx = ce.parse_13f_index(html)
    assert set(idx) == {"01jun2026-31aug2026_form13f.zip", "2023q4_form13f.zip"}
    assert ce.set_range("01jun2026-31aug2026_form13f.zip") == (date(2026, 6, 1), date(2026, 8, 31))
    assert ce.set_range("2023q4_form13f.zip") == (date(2023, 10, 1), date(2023, 12, 31))
    rows, st = ce.aggregate_13f(_zip13f(), (date(2026, 6, 1), date(2026, 8, 31)), IDX)
    [r] = rows                                                         # PFE nur PRN -> ausgeschlossen
    assert r["cik"] == "cik:3" and r["holders"] == 2 and r["shares"] == 350        # beide Klassen, ohne Call/spät/A
    assert r["available"] == date(2026, 8, 15) and st["filings"] == 2
    rows2, _ = ce.aggregate_13f(_zip13f(), (date(2026, 7, 2), date(2026, 9, 30)), IDX)
    assert rows2 == []                                                 # Fristfenster nicht vollständig im Zeitraum


def test_13f_connector_processes_one_new_set_and_marks_it(monkeypatch, tmp_path):
    monkeypatch.setattr(ce, "entity_index", lambda: IDX)
    html = '<a href="/files/x/01jun2026-31aug2026_form13f.zip">x</a>'

    def fake(url, **kw):
        return resp(_zip13f() if url.endswith(".zip") else html, url=url)

    with patch.object(ce.http, "fetch", side_effect=fake):
        r = ce.Sec13FConnector({"_archive_root": str(tmp_path)}).fetch(NOW)
    assert r.status == SourceStatus.PASS
    marker = [o for o in r.observations if o.metric == "dataset_processed"]
    assert [o.series_id for o in marker] == ["set:01jun2026-31aug2026_form13f.zip"]
    holders = next(o for o in r.observations if o.metric == "inst_holders")
    assert holders.available_at == datetime(2026, 8, 15, tzinfo=UTC) and holders.attrs["tickers"] == ["GOOG", "GOOGL"]
    from modules.external.archive import ExternalArchive
    ExternalArchive(tmp_path).store_observations(r.observations)
    with patch.object(ce.http, "fetch", side_effect=fake):
        r2 = ce.Sec13FConnector({"_archive_root": str(tmp_path)}).fetch(NOW)
    assert r2.observations == [] and "kein neuer" in r2.message


def test_wiki_parser_accepts_parsoid_links_and_rejects_namespaces():
    html = ('<table id="constituents"><tr><th>Symbol</th><th>Security</th></tr>'
            '<tr><td><a href="https://www.nyse.com/quote/XNYS:MMM">MMM</a></td><td><a href="./3M" title="3M">3M</a></td></tr>'
            '<tr><td>AOS</td><td><a href="./A._O._Smith">A. O. Smith</a></td></tr>'
            '<tr><td>XYZ</td><td><a href="./File:Logo.svg">x</a></td></tr></table>')
    assert mp.parse_wiki_list(html) == {"MMM": "3M", "AOS": "A._O._Smith"}


def test_bea_json_and_html_schedule():
    js = {"National Income and Product Accounts": {"release_dates": ["2026-10-29T08:30:00-04:00",
                                                                     "2027-01-28T08:30:00-05:00"]},
          "International Transactions": {"release_dates": ["2026-12-17T08:30:00-05:00"]}}
    ev = cal.parse_bea_json(js)
    assert {e["summary"] for e in ev} == {"National Income and Product Accounts", "International Transactions"}
    with pytest.raises(ValueError):
        cal.parse_bea_json({"x": 1})
    html = ("<table><tr><th>Date</th><th>Time</th><th>Release</th></tr>"
            "<tr><td>October 29</td><td>8:30 AM</td><td>GDP (Advance Estimate), 3rd Quarter 2026</td></tr>"
            "<tr><td>January 7</td><td>10:00 AM</td><td>Trade in Goods and Services</td></tr></table>")
    ev = cal.parse_bea_schedule_html(html, date(2026, 10, 15))
    assert [(e["start"].date(), e["start"].hour, e["summary"][:3]) for e in ev] == [
        (date(2026, 10, 29), 8, "GDP"), (date(2027, 1, 7), 10, "Tra")]

    def fake(url, **kw):
        if url == cal.BEA_JSON_URL:
            return resp("<html>nicht json</html>", url=url)
        return resp(html, url=url)

    with patch.object(cal.http, "fetch", side_effect=fake):
        r = cal.BeaReleaseCalendarConnector({}).fetch(NOW)
    assert r.status == SourceStatus.PASS and len(r.observations) == 2
    assert all(o.available_at == NOW for o in r.observations)


def test_pageviews_preflight_uses_sample(monkeypatch):
    many = "".join(f"<tr><td>T{chr(65 + i // 26)}{chr(65 + i % 26)}</td><td><a href='./Art{i}'>x</a></td></tr>"
                   for i in range(40))
    html = f"<table id='constituents'><tr><th>S</th><th>N</th></tr>{many}</table>"
    calls = []

    def fake(url, **kw):
        if "wikipedia.org/wiki/" in url:
            return resp(html, url=url)
        calls.append(url)
        return resp({"items": [{"timestamp": "2026101400", "views": 1}]}, url=url)

    conn = mp.WikimediaPageviewsConnector({"min_articles": 10})
    with patch.object(mp.http, "fetch", side_effect=fake), patch.object(mp.time, "sleep"):
        rep = conn.preflight(NOW)
    assert rep["status"] == "PASS" and len(calls) == mp.WikimediaPageviewsConnector.PREFLIGHT_SAMPLE
    assert conn._sample is None                                        # nur für den Preflight

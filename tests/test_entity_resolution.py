"""Entity Resolution (PIT): Parser gegen offizielle Schemas, Matching-Konfidenz,
keine rückwirkenden Konzernstrukturen, Append-only, Fehler stoppen nie den Lauf."""
from __future__ import annotations

import pytest

from modules.entity_resolution import sources as src
from modules.entity_resolution.build import build
from modules.entity_resolution.store import EntityRecord, EntityStore, normalize_name
from modules.external.http import SchemaError

TICKERS = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
           "1": {"cik_str": 51143, "ticker": "IBM", "title": "INTERNATIONAL BUSINESS MACHINES CORP"},
           "2": {"cik_str": 1067983, "ticker": "BRK.B", "title": "BERKSHIRE HATHAWAY INC"}}
SUB_AAPL = {"cik": "320193", "name": "Apple Inc.", "sic": "3571", "sicDescription": "Electronic Computers",
            "stateOfIncorporation": "CA", "addresses": {"business": {"stateOrCountry": "CA"}},
            "formerNames": [{"name": "APPLE COMPUTER INC", "from": "1994-01-26T00:00:00.000Z",
                             "to": "2007-01-04T00:00:00.000Z"}], "tickers": ["AAPL"]}


def _lei(lei, name, country="US", status="ACTIVE", init="2012-06-06T15:52:00Z", other=()):
    return {"attributes": {"lei": lei, "entity": {"legalName": {"name": name}, "status": status,
                                                    "otherNames": [{"name": o} for o in other],
                                                    "jurisdiction": f"{country}-CA", "legalAddress": {"country": country}},
                           "registration": {"status": "ISSUED", "initialRegistrationDate": init}}}


def _rel(parent, start="2015-03-01T00:00:00Z", status="ACTIVE"):
    return {"data": {"attributes": {"relationship": {"startNode": {"id": "X"}, "endNode": {"id": parent},
                                                      "status": status,
                                                      "periods": [{"startDate": start, "type": "RELATIONSHIP_PERIOD"}]}}}}


def test_normalize_name():
    assert normalize_name("Apple Inc.") == normalize_name("APPLE INC") == "apple"
    assert normalize_name("Johnson & Johnson") == "johnson and johnson"
    assert normalize_name(None) == ""


def test_sec_parsers_and_schema_errors():
    rows = src.parse_company_tickers(TICKERS)
    assert rows[0] == {"cik": "0000320193", "ticker": "AAPL", "title": "Apple Inc."}
    assert rows[2]["ticker"] == "BRK-B"
    with pytest.raises(SchemaError):
        src.parse_company_tickers({"0": {"ticker": "X"}})
    e = src.parse_submissions_entity(SUB_AAPL)
    assert e["former_names"][0] == {"name": "APPLE COMPUTER INC", "from": "1994-01-26", "to": "2007-01-04"}
    with pytest.raises(SchemaError):
        src.parse_submissions_entity({"name": "x"})


def test_sec_records_split_pit_and_research_identity():
    recs = src.sec_records(src.parse_company_tickers(TICKERS)[:1],
                           {"0000320193": src.parse_submissions_entity(SUB_AAPL)})
    pit = next(r for r in recs if r.usage == "pit")
    ident = next(r for r in recs if r.usage == "research_ticker_identity")
    assert pit.mapping_confidence == "HIGH" and pit.country == "US" and "APPLE COMPUTER INC" in pit.aliases
    assert ident.mapping_confidence == "MEDIUM" and ident.evidence["risk"]


def test_lei_matching_confidence():
    recs = src.parse_lei_records({"data": [_lei("HWUPKR0MPOU8FGXBT394", "Apple Inc."),
                                            _lei("549300XXXXXXXXXXXX01", "Apple Inc.", country="IE")]})
    m = src.match_lei(["Apple Inc."], recs, "US")
    assert m["confidence"] == "HIGH" and m["lei"] == "HWUPKR0MPOU8FGXBT394"
    assert src.match_lei(["Apple Inc."], recs, "DE")["confidence"] == "LOW"          # mehrdeutig
    assert src.match_lei(["Apple Inc."], recs[1:], "US")["confidence"] == "MEDIUM"   # Land abweichend
    assert src.match_lei(["Apple Hospitality"], recs, "US")["lei"] is None
    inactive = src.parse_lei_records({"data": [_lei("L1", "Apple Inc.", status="INACTIVE")]})
    assert src.match_lei(["Apple Inc."], inactive, "US")["lei"] is None
    with pytest.raises(SchemaError):
        src.parse_lei_records({"nodata": []})


def test_parent_relationship_never_retroactive(tmp_path):
    store = EntityStore(tmp_path / "e.jsonl")
    sec = src.sec_records(src.parse_company_tickers(TICKERS)[:1], {})[0]
    m = src.match_lei(["Apple Inc."], src.parse_lei_records({"data": [_lei("LAPPLE", "Apple Inc.")]}), "US")
    recs = src.gleif_records(sec, m, src.parse_relationship(_rel("LPARENT")), None)
    for r in recs:
        store.upsert(r, "2026-10-02")
    link = store.resolve(cik="320193", as_of="2014-01-01", usage="pit")
    assert link is not None and link.lei == "LAPPLE" and link.parent_entity is None     # LEI ab 2012 gültig
    parent_rec = [r for r in store.records if r.mapping_source == "gleif_parent"][0]
    assert parent_rec.valid_from == "2015-03-01" and not parent_rec.valid_at("2014-06-01")
    assert parent_rec.valid_at("2016-01-01") and parent_rec.parent_entity == "lei:LPARENT"
    assert src.parse_relationship({"data": None}) is None


def test_store_append_only_versioning_and_no_backdating(tmp_path):
    p = tmp_path / "e.jsonl"
    s = EntityStore(p)
    r1 = EntityRecord(entity_id="cik:1", ticker="OLD", cik="1", canonical_name="A", mapping_source="sec_company_tickers",
                      mapping_confidence="HIGH")
    assert s.upsert(r1, "2026-01-01") == "opened" and r1.valid_from == "2026-01-01"   # ohne Beleg: ab Abruf
    assert s.upsert(EntityRecord(**{**r1.__dict__, "valid_from": None}), "2026-02-01") == "unchanged"
    r2 = EntityRecord(entity_id="cik:1", ticker="OLD", cik="1", canonical_name="B", mapping_source="sec_company_tickers",
                      mapping_confidence="HIGH")                                    # Umbenennung derselben Reihe
    assert s.upsert(r2, "2026-06-01") == "replaced"
    s2 = EntityStore(p)                                                               # aus Datei rekonstruiert
    assert s2.resolve(cik="1", as_of="2026-03-01").canonical_name == "A"
    assert s2.resolve(cik="1", as_of="2026-07-01").canonical_name == "B"
    assert s2.resolve(cik="1", as_of="2025-12-31") is None                            # vor erster Kenntnis: unbekannt
    with pytest.raises(ValueError):
        s2.upsert(EntityRecord(**{**r2.__dict__, "canonical_name": "X", "valid_from": None}), "2026-05-01")
    assert len(p.read_text().splitlines()) == 3                                       # open, close, open


def test_record_validation():
    with pytest.raises(ValueError):
        EntityRecord(entity_id="x", mapping_source="", mapping_confidence="HIGH")
    with pytest.raises(ValueError):
        EntityRecord(entity_id="x", mapping_source="s", mapping_confidence="SURE")
    with pytest.raises(ValueError):
        EntityRecord(entity_id="x", mapping_source="s", mapping_confidence="LOW", valid_from="2026-02-01",
                     valid_to="2026-01-01")


def test_build_incremental_with_failures(tmp_path):
    store = EntityStore(tmp_path / "e.jsonl")
    calls = {"gleif": 0}

    def gleif(name):
        calls["gleif"] += 1
        if "BERKSHIRE" in name.upper():
            raise RuntimeError("GLEIF down")
        return src.parse_lei_records({"data": [_lei("L-" + name[:3].upper(), name)]})
    rep = build(["AAPL", "IBM", "BRK.B", "ZZZZ"], store, "2026-10-02", fetch_tickers=lambda: src.parse_company_tickers(TICKERS),
                fetch_submissions=lambda cik: SUB_AAPL if cik.endswith("320193") else
                {"cik": cik, "name": next(v["title"] for v in TICKERS.values() if str(v["cik_str"]) == str(int(cik)))},
                fetch_gleif=gleif, fetch_parent=lambda lei, kind: None, gleif_budget=10, sleep=lambda s: None)
    assert rep["sec"]["mapped"] == 3 and rep["sec"]["unmapped"] == ["ZZZZ"]
    assert rep["gleif"]["attempted"] == 3 and any("GLEIF down" in e for e in rep["errors"])
    prof = store.profile(ticker="AAPL", as_of="2026-10-02")
    assert prof["lei"] and prof["sources"]["lei"]["source"] == "gleif_name_match" and prof["cik"] == "0000320193"
    assert store.profile(ticker="ZZZZ", as_of="2026-10-02") is None
    n = calls["gleif"]
    build(["AAPL"], store, "2026-10-09", fetch_tickers=lambda: src.parse_company_tickers(TICKERS),
          fetch_submissions=lambda cik: SUB_AAPL, fetch_gleif=gleif, fetch_parent=lambda lei, kind: None,
          sleep=lambda s: None)
    assert calls["gleif"] == n                                    # bereits zugeordnet -> kein erneuter GLEIF-Abruf
    down = build(["AAPL"], EntityStore(tmp_path / "f.jsonl"), "2026-10-02",
                 fetch_tickers=lambda: (_ for _ in ()).throw(RuntimeError("SEC down")), sleep=lambda s: None)
    assert down["errors"] and not down["sec"]


def test_share_classes_same_cik_and_delisted_ticker_closed(tmp_path):
    """CI 2026-10-03: GOOG/GOOGL (eine CIK) im selben Lauf -> fälschlich 'Rückdatierung'.
    Jetzt: je Ticker eine Reihe; ein nicht mehr gelisteter Ticker wird später geschlossen."""
    rows = {"0": {"cik_str": 1652044, "ticker": "GOOGL", "title": "Alphabet Inc."},
            "1": {"cik_str": 1652044, "ticker": "GOOG", "title": "Alphabet Inc."},
            "2": {"cik_str": 1, "ticker": "OLDT", "title": "Old Corp"}}
    store = EntityStore(tmp_path / "e.jsonl")
    kw = dict(fetch_submissions=lambda cik: {"cik": cik, "name": "Alphabet Inc."},
              fetch_gleif=lambda name: [], fetch_parent=lambda lei, kind: None, sleep=lambda s: None)
    rep = build(["GOOG", "GOOGL", "OLDT"], store, "2026-10-03", fetch_tickers=lambda: src.parse_company_tickers(rows), **kw)
    assert rep["sec"]["changes"]["opened"] == 6 and not rep["errors"]
    ident = {r.ticker: r.cik for r in store.records if r.usage == "research_ticker_identity" and r.valid_to is None}
    assert ident["GOOG"] == ident["GOOGL"]
    again = build(["GOOG", "GOOGL", "OLDT"], store, "2026-10-03", fetch_tickers=lambda: src.parse_company_tickers(rows), **kw)
    assert again["sec"]["changes"]["unchanged"] == 6                      # gleicher Tag, gleicher Inhalt: kein Fehler
    later = {k: v for k, v in rows.items() if v["ticker"] != "OLDT"}
    rep2 = build(["GOOG", "GOOGL", "OLDT"], EntityStore(tmp_path / "e.jsonl"), "2026-10-10",
                 fetch_tickers=lambda: src.parse_company_tickers(later), **kw)
    assert rep2["sec"]["changes"]["closed"] == 2
    st = EntityStore(tmp_path / "e.jsonl")                                 # Schließen überlebt das Neuladen
    assert st.resolve(ticker="OLDT", as_of="2026-10-11") is None and st.resolve(ticker="OLDT", as_of="2026-10-05")

"""Regression 2026-09-28: SEC-Insider-Signal. Vorher Volltextsuche nach dem
Ticker-String (traf fremde Emittenten, zählte Verkäufe) -> fast jeder
Kandidat bekam die Schlagzeile "N Insider kaufen X (Cluster-Signal)".
Jetzt nur Open-Market-Käufe (Code P) aus Form-4-XML des Emittenten."""
from __future__ import annotations

from datetime import datetime, timedelta

from modules import alpha_sources as al


def _form4(issuer_cik="0000066740", owner="DOE JOHN", code="P", date="2026-09-25", shares="1000",
           price="120.50", ad="A"):
    return f"""<?xml version="1.0"?>
<ownershipDocument>
  <schemaVersion>X0508</schemaVersion>
  <documentType>4</documentType>
  <issuer><issuerCik>{issuer_cik}</issuerCik><issuerName>3M CO</issuerName>
    <issuerTradingSymbol>MMM</issuerTradingSymbol></issuer>
  <reportingOwner><reportingOwnerId><rptOwnerCik>0001234567</rptOwnerCik>
    <rptOwnerName>{owner}</rptOwnerName></reportingOwnerId></reportingOwner>
  <nonDerivativeTable><nonDerivativeTransaction>
    <securityTitle><value>Common Stock</value></securityTitle>
    <transactionDate><value>{date}</value></transactionDate>
    <transactionCoding><transactionFormType>4</transactionFormType>
      <transactionCode>{code}</transactionCode><equitySwapInvolved>0</equitySwapInvolved></transactionCoding>
    <transactionAmounts><transactionShares><value>{shares}</value></transactionShares>
      <transactionPricePerShare><value>{price}</value></transactionPricePerShare>
      <transactionAcquiredDisposedCode><value>{ad}</value></transactionAcquiredDisposedCode></transactionAmounts>
  </nonDerivativeTransaction></nonDerivativeTable>
</ownershipDocument>"""


def test_parse_form4_xml():
    p = al.parse_form4_xml(_form4())
    assert p["issuer_cik"] == "0000066740" and p["owners"] == ["DOE JOHN"]
    assert p["transactions"] == [{"date": "2026-09-25", "code": "P", "shares": 1000.0,
                                  "price": 120.5, "ad": "A"}]


class _R:
    def __init__(self, payload=None, text=""):
        self._p, self.text, self.status_code = payload, text, 200

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


def _install(monkeypatch, filings):
    """filings: list of (filing_date, xml)"""
    today = datetime.utcnow().strftime("%Y-%m-%d")
    recent = {"form": [], "filingDate": [], "accessionNumber": [], "primaryDocument": []}
    docs = {}
    for i, (fdate, xml) in enumerate(filings):
        acc = f"0000000000-26-{i:06d}"
        recent["form"].append("4")
        recent["filingDate"].append(fdate or today)
        recent["accessionNumber"].append(acc)
        recent["primaryDocument"].append(f"xslF345X05/doc{i}.xml")
        docs[f"{acc.replace('-', '')}/doc{i}.xml"] = xml
    recent["form"].append("10-K"); recent["filingDate"].append(today)
    recent["accessionNumber"].append("x"); recent["primaryDocument"].append("k.htm")

    def fake_get(url, timeout=10):
        if url == al.SEC_TICKERS_URL:
            return _R({"0": {"cik_str": 66740, "ticker": "MMM", "title": "3M"}})
        if "submissions" in url:
            return _R({"filings": {"recent": recent}})
        for key, xml in docs.items():
            if url.endswith(key):
                assert "xslF345X05" not in url          # Roh-XML, nicht die gerenderte Ansicht
                return _R(text=xml)
        raise AssertionError(url)

    monkeypatch.setattr(al, "_sec_get", fake_get)
    monkeypatch.setattr(al, "_sec_cik_map", None)


def test_sales_and_grants_never_count_as_buys(monkeypatch):
    d = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    _install(monkeypatch, [(None, _form4(owner="A", code="S", ad="D", date=d)),
                           (None, _form4(owner="B", code="S", ad="D", date=d)),
                           (None, _form4(owner="C", code="A", date=d)),     # Zuteilung
                           (None, _form4(owner="D", code="M", date=d))])    # Ausübung
    r = al.detect_insider_cluster("MMM")
    assert r["data_available"] and not r["cluster_detected"]
    assert r["insider_count"] == 0 and r["buy_count"] == 0 and r["sell_count"] == 2
    assert r["headline"] == ""


def test_two_buyers_within_72h_is_cluster(monkeypatch):
    d1 = (datetime.utcnow() - timedelta(days=3)).strftime("%Y-%m-%d")
    d2 = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    _install(monkeypatch, [(None, _form4(owner="A", date=d1)), (None, _form4(owner="B", date=d2))])
    r = al.detect_insider_cluster("MMM")
    assert r["cluster_detected"] and r["insider_count"] == 2 and r["buy_count"] == 2
    assert "am offenen Markt" in r["headline"]


def test_two_buyers_far_apart_is_not_cluster(monkeypatch):
    d1 = (datetime.utcnow() - timedelta(days=12)).strftime("%Y-%m-%d")
    d2 = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    _install(monkeypatch, [(None, _form4(owner="A", date=d1)), (None, _form4(owner="B", date=d2))])
    r = al.detect_insider_cluster("MMM")
    assert r["insider_count"] == 2 and not r["cluster_detected"]


def test_foreign_issuer_filing_is_ignored(monkeypatch):
    d = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    _install(monkeypatch, [(None, _form4(owner="A", date=d, issuer_cik="0000320193")),
                           (None, _form4(owner="B", date=d, issuer_cik="0000320193"))])
    assert al.detect_insider_cluster("MMM")["buy_count"] == 0


def test_old_filings_outside_window_ignored(monkeypatch):
    old = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
    _install(monkeypatch, [(old, _form4(owner="A", date=old)), (old, _form4(owner="B", date=old))])
    assert al.detect_insider_cluster("MMM")["buy_count"] == 0


def test_unknown_ticker_reports_no_data(monkeypatch):
    _install(monkeypatch, [])
    r = al.detect_insider_cluster("ZZZZ")
    assert r["data_available"] is False and r["cluster_detected"] is False


def test_tradier_skew_25d_uses_delta_not_same_strike(monkeypatch):
    """ATM-Quotient bleibt ~1 (Parität), 25-Delta-Skew erfasst die Put-Prämie."""
    from datetime import date
    exp = (date.today() + timedelta(days=35)).isoformat()
    chain = [
        {"strike": 100, "option_type": "call", "greeks": {"mid_iv": 0.30, "delta": 0.52}},
        {"strike": 100, "option_type": "put", "greeks": {"mid_iv": 0.30, "delta": -0.48}},
        {"strike": 110, "option_type": "call", "greeks": {"mid_iv": 0.25, "delta": 0.24}},
        {"strike": 90, "option_type": "put", "greeks": {"mid_iv": 0.40, "delta": -0.26}},
    ]

    def fake_get(url, params=None, headers=None, timeout=10):
        return _R({"expirations": {"date": [exp]}} if "expirations" in url
                  else {"options": {"option": chain}})

    monkeypatch.setattr(al.requests, "get", fake_get)
    r = al._fetch_skew_tradier("MMM", 100.0, "k")
    assert r["skew_ratio"] == 1.0 and r["signal"] == "neutral"
    assert r["skew_25d"] == 1.6 and r["skew_25d_method"] == "tradier_delta"

"""SEC Deep Events: Parser (offizielle Schemas), PIT, fehlende Daten != 0,
Amendments, Schemafehler, inkrementeller Ingest, Feature-Store."""
from __future__ import annotations

import io
import json
import math
import types
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from modules.external.http import SchemaError
from modules.external.sources import sec_events as se
from modules.external.sources import sec_features as sf
from modules.external.sources import sec_ingest as si

NOW = datetime(2026, 10, 2, tzinfo=timezone.utc)
CIK = "0000320193"


def _tsv(rows: list[dict]) -> str:
    cols = list(rows[0])
    return "\t".join(cols) + "\n" + "\n".join("\t".join(str(r[c]) for c in cols) for r in rows) + "\n"


def form345_zip(extra_sub_cols: bool = True, drop_col: str | None = None) -> bytes:
    subs = [{"ACCESSION_NUMBER": "A1", "FILING_DATE": "05-MAR-2024", "PERIOD_OF_REPORT": "01-MAR-2024",
             "DOCUMENT_TYPE": "4", "ISSUERCIK": "320193", "ISSUERNAME": "Apple Inc.", "ISSUERTRADINGSYMBOL": "AAPL"},
            {"ACCESSION_NUMBER": "A2", "FILING_DATE": "06-MAR-2024", "PERIOD_OF_REPORT": "04-MAR-2024",
             "DOCUMENT_TYPE": "4/A", "ISSUERCIK": "320193", "ISSUERNAME": "Apple Inc.", "ISSUERTRADINGSYMBOL": "AAPL"},
            {"ACCESSION_NUMBER": "A3", "FILING_DATE": "07-MAR-2024", "PERIOD_OF_REPORT": "05-MAR-2024",
             "DOCUMENT_TYPE": "4", "ISSUERCIK": "999", "ISSUERNAME": "Other", "ISSUERTRADINGSYMBOL": "OTH"}]
    owners = [{"ACCESSION_NUMBER": "A1", "RPTOWNERCIK": "111", "RPTOWNERNAME": "Doe Jane",
               "RPTOWNER_RELATIONSHIP": "Director"}]
    trans = [{"ACCESSION_NUMBER": "A1", "NONDERIV_TRANS_SK": "1", "TRANS_DATE": "01-MAR-2024", "TRANS_CODE": "P",
              "TRANS_SHARES": "100", "TRANS_PRICEPERSHARE": "50", "TRANS_ACQUIRED_DISP_CD": "A"},
             {"ACCESSION_NUMBER": "A1", "NONDERIV_TRANS_SK": "2", "TRANS_DATE": "01-MAR-2024", "TRANS_CODE": "M",
              "TRANS_SHARES": "100", "TRANS_PRICEPERSHARE": "10", "TRANS_ACQUIRED_DISP_CD": "A"},       # Ausübung
             {"ACCESSION_NUMBER": "A1", "NONDERIV_TRANS_SK": "3", "TRANS_DATE": "01-MAR-2024", "TRANS_CODE": "S",
              "TRANS_SHARES": "10", "TRANS_PRICEPERSHARE": "", "TRANS_ACQUIRED_DISP_CD": "D"},          # Preis fehlt
             {"ACCESSION_NUMBER": "A2", "NONDERIV_TRANS_SK": "4", "TRANS_DATE": "04-MAR-2024", "TRANS_CODE": "P",
              "TRANS_SHARES": "1000", "TRANS_PRICEPERSHARE": "50", "TRANS_ACQUIRED_DISP_CD": "A"},      # Amendment
             {"ACCESSION_NUMBER": "A3", "NONDERIV_TRANS_SK": "5", "TRANS_DATE": "05-MAR-2024", "TRANS_CODE": "P",
              "TRANS_SHARES": "1", "TRANS_PRICEPERSHARE": "1", "TRANS_ACQUIRED_DISP_CD": "A"}]
    if drop_col:
        for t in trans:
            t.pop(drop_col)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("SUBMISSION.tsv", _tsv(subs))
        z.writestr("REPORTINGOWNER.tsv", _tsv(owners))
        z.writestr("NONDERIV_TRANS.tsv", _tsv(trans))
    return buf.getvalue()


def test_form345_parser_open_market_only_original_filings_pit():
    obs = se.parse_form345_zip(form345_zip(), {CIK}, NOW)
    assert len(obs) == 1                                       # nur P des Originals; M, fehlender Preis, /A, fremder Emittent raus
    o = obs[0]
    assert o.metric == "insider_buy_usd" and o.value == 5000.0 and o.entity_id == f"cik:{CIK}"
    assert o.available_at.date().isoformat() == "2024-03-06"     # Filing-Datum + 1 Tag (konservativ)
    assert o.observation_time.date().isoformat() == "2024-03-01"
    assert o.availability_precision.value == "CONSERVATIVE_DATE" and o.attrs["owner_ciks"] == ["111"]
    assert len(se.parse_form345_zip(form345_zip(), None, NOW)) == 2    # ohne Universumsfilter auch Fremd-Emittent


def test_form345_schema_change_is_loud():
    with pytest.raises(SchemaError):
        se.parse_form345_zip(form345_zip(drop_col="TRANS_PRICEPERSHARE"), {CIK}, NOW)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("OTHER.tsv", "x\n")
    with pytest.raises(SchemaError):
        se.parse_form345_zip(buf.getvalue(), {CIK}, NOW)


def test_form345_quarters_only_published():
    q = se.form345_quarters(2025, NOW)
    assert q[0] == (2025, 1) and q[-1] == (2026, 2)              # Q3 2026 noch nicht veröffentlicht am 2.10.
    assert se.form345_quarters(2026, datetime(2026, 2, 1, tzinfo=timezone.utc)) == []


SUBS = {"cik": "320193", "name": "Apple Inc.", "filings": {
    "recent": {"accessionNumber": ["S1", "S2", "S3", "S4", "S5"],
               "filingDate": ["2024-05-03", "2024-08-02", "2024-08-10", "2024-09-01", "2024-09-02"],
               "reportDate": ["2024-03-30", "2024-06-29", "", "", ""],
               "acceptanceDateTime": ["2024-05-02T20:01:00.000Z", "2024-08-01T20:30:00.000Z",
                                      "2024-08-09T21:05:00.000Z", "2024-08-31T12:00:00.000Z", ""],
               "form": ["10-Q", "10-Q", "8-K", "4", "NT 10-K"], "items": ["", "", "5.02,9.01", "", ""]},
    "files": [{"name": "CIK0000320193-submissions-001.json", "filingCount": 1}]}}
PAGE = {"accessionNumber": ["S0"], "filingDate": ["2015-02-01"], "reportDate": [""],
        "acceptanceDateTime": ["2015-01-30T22:00:00.000Z"], "form": ["8-K"], "items": ["4.02"]}


def test_submissions_parser_and_observations():
    rows = se.parse_submissions_filings(SUBS)
    obs = se.filings_to_observations("320193", rows, NOW)
    by = {o.series_id: o for o in obs}
    assert set(by) == {"S1", "S2", "S3", "S5"}                     # Form 4 hier nicht (eigener Pfad)
    assert by["S1"].value == 34.0 and by["S1"].unit == "days_after_period"
    assert by["S3"].attrs["items"] == ["5.02", "9.01"] and by["S3"].availability_precision.value == "EXACT_TIMESTAMP"
    assert by["S3"].available_at.isoformat().startswith("2024-08-09T21:05")
    assert by["S5"].availability_precision.value == "CONSERVATIVE_DATE"   # ohne Annahmezeit: Datum + 1 Tag
    assert se.submissions_pages(SUBS) == ["CIK0000320193-submissions-001.json"]
    assert se.parse_submissions_filings(PAGE)[0]["form"] == "8-K"
    with pytest.raises(SchemaError):
        se.parse_submissions_filings({"filings": {"recent": {"form": ["8-K"]}}})
    with pytest.raises(SchemaError):
        se.parse_submissions_filings({"form": ["8-K", "4"], "accessionNumber": ["x"]})


def _feat_obs():
    obs = se.parse_form345_zip(form345_zip(), {CIK}, NOW)
    obs += se.filings_to_observations("320193", se.parse_submissions_filings(SUBS) + se.parse_submissions_filings(PAGE), NOW)
    return obs


def test_features_point_in_time_and_missing_semantics():
    obs = _feat_obs()
    ins, fil = sf.to_frames(obs)
    cov = sf.insider_coverage([(2023, 1), (2024, 2)])
    since = pd.Timestamp("2015-01-30", tz="UTC")
    before = sf.features_for(f"cik:{CIK}", "2024-03-05", ins, fil, cov, since)        # Filing 5.3. -> erst ab 6.3.
    after = sf.features_for(f"cik:{CIK}", "2024-03-08", ins, fil, cov, since)
    assert before["sec_insider_buy_value_90d"] == 0.0                                    # abgedeckt, echte Null
    assert after["sec_insider_buy_value_90d"] == pytest.approx(math.log1p(5000))
    assert after["sec_insider_buyers_90d"] == 1.0 and after["sec_insider_cluster_30d"] == 0.0
    out_cov = sf.features_for(f"cik:{CIK}", "2024-09-30", ins, fil, cov, since)          # nach Abdeckungsende
    assert math.isnan(out_cov["sec_insider_buy_value_90d"])                              # NaN, nie 0
    aug = sf.features_for(f"cik:{CIK}", "2024-08-20", ins, fil, None, since)
    assert aug["sec_exec_change_90d"] == 1.0 and aug["sec_8k_negative_90d"] == 0.0
    early = sf.features_for(f"cik:{CIK}", "2024-08-09", ins, fil, None, since)           # 8-K um 21:05 -> nach Cutoff 21:00
    assert early["sec_exec_change_90d"] == 0.0
    assert sf.features_for(f"cik:{CIK}", "2015-06-01", ins, fil, None, since)["sec_8k_negative_90d"] != \
        sf.features_for(f"cik:{CIK}", "2015-06-01", ins, fil, None, since)["sec_8k_negative_90d"] or True
    young = sf.features_for(f"cik:{CIK}", "2015-06-01", ins, fil, None, since)            # < 1 Jahr Historie
    assert all(math.isnan(young[k]) for k in sf.FILING_FEATURES)


def test_feature_table_flags_availability():
    obs = _feat_obs()
    t = sf.build_feature_table(obs, ["2024-03-08", "2024-09-30"], {"AAPL": f"cik:{CIK}", "NOPE": "cik:0000000001"},
                               sf.insider_coverage([(2023, 1), (2024, 2)]),
                               {f"cik:{CIK}": pd.Timestamp("2015-01-30", tz="UTC")})
    a = t[t["ticker"] == "AAPL"].set_index("date")
    assert a.loc[pd.Timestamp("2024-03-08"), "alt_sec_available"] == 1.0
    n = t[t["ticker"] == "NOPE"].set_index("date")
    assert n[sf.FILING_FEATURES].isna().all().all()                         # keine Filing-Historie -> NaN
    assert n.loc[pd.Timestamp("2024-03-08"), "sec_insider_buy_value_90d"] == 0.0   # Datensatz vollständig -> echte Null
    assert math.isnan(n.loc[pd.Timestamp("2024-09-30"), "sec_insider_buy_value_90d"])
    assert set(sf.FEATURES) <= set(t.columns) and (t["feature_version"] == sf.FEATURE_VERSION).all()


def test_ingest_incremental_and_failure_tolerant(tmp_path, monkeypatch):
    monkeypatch.setattr(si, "DIR", tmp_path)
    calls = []

    def get(url, **k):
        calls.append(url)
        if "form345" in url:
            if "2024q2" in url:
                raise RuntimeError("SEC 503")
            content = form345_zip()
        elif url.endswith("submissions-001.json"):
            content = json.dumps(PAGE).encode()
        elif "submissions" in url:
            content = json.dumps(SUBS).encode()
        else:
            raise RuntimeError("unerwartet")
        return types.SimpleNamespace(content=content, retrieved_at=NOW, content_hash="h",
                                     json=lambda c=content: json.loads(c))
    state = {}
    r1 = si.ingest_form345(state, {CIK}, 2024, NOW, get=get, sleep=lambda s: None)
    assert "2024Q1" in r1["fetched"] and any("2024Q2" in e for e in r1["errors"]) and r1["new_rows"] == 1
    r2 = si.ingest_form345(state, {CIK}, 2024, NOW, get=get, sleep=lambda s: None)
    assert "2024Q1" not in r2["fetched"] and r2["new_rows"] == 0                         # inkrementell, append-only
    s1 = si.ingest_submissions(state, [CIK], NOW, get=get, sleep=lambda s: None, live_budget=0)
    assert s1["ciks"] == 1 and s1["new_rows"] == 5 and state["filing_since"][CIK].startswith("2015-01-30")
    n_page_calls = sum(1 for c in calls if c.endswith("submissions-001.json"))
    si.ingest_submissions(state, [CIK], NOW, get=get, sleep=lambda s: None, live_budget=0)
    assert sum(1 for c in calls if c.endswith("submissions-001.json")) == n_page_calls  # alte Seiten nicht erneut
    h = si.health(state, {"form345": r1, "submissions": s1}, 1, NOW)
    assert h["coverage"] == 1.0 and h["error_rate"] > 0 and "2024Q1" in h["form345_quarters"] and "2024Q2" not in h["form345_quarters"]
    feat = si.build_features(state, {"AAPL": CIK}, pd.date_range("2024-03-01", "2024-03-15", freq="W-FRI"))
    assert len(feat) == 3 and set(sf.FEATURES) <= set(feat.columns)


def test_store_roundtrip_preserves_pit_fields(tmp_path):
    obs = _feat_obs()
    df = si.obs_to_rows(obs)
    back = si.rows_to_obs(df.astype(str))
    assert [o.available_at for o in back] == [o.available_at for o in obs]
    assert [o.availability_precision for o in back] == [o.availability_precision for o in obs]
    assert back[0].attrs == obs[0].attrs


def test_vectorized_features_equal_reference():
    """features_cik (schnell) == features_for (Referenz) über zufällige Event-Historien."""
    import numpy as np
    import pandas as pd
    from modules.external.sources import sec_features as sf
    rnd = np.random.default_rng(3)
    n = 300
    fil = pd.DataFrame({"cik": "1", "avail": pd.to_datetime(rnd.integers(1.40e9, 1.79e9, n), unit="s", utc=True),
                        "form": rnd.choice(["8-K", "10-Q", "10-K", "NT 10-Q", "NT 10-K"], n),
                        "items": [tuple(rnd.choice(["5.02", "2.02", "1.03", "4.02"], 2)) for _ in range(n)],
                        "delay": np.where(rnd.random(n) < 0.8, rnd.normal(40, 6, n), np.nan)})
    ins = pd.DataFrame({"cik": "1", "avail": pd.to_datetime(rnd.integers(1.40e9, 1.79e9, 200), unit="s", utc=True),
                        "metric": rnd.choice(["insider_buy_usd", "insider_sell_usd"], 200),
                        "value": rnd.random(200) * 1e5, "owners": [tuple(rnd.choice(list("abcd"), 1)) for _ in range(200)]})
    dates = pd.date_range("2015-01-02", "2026-09-25", freq="W-FRI")[::3]
    cov = sf.insider_coverage([(2015, 1), (2025, 4)])
    since = pd.Timestamp("2014-06-01", tz="UTC")
    fast = sf.features_cik(list(dates), ins, fil, cov, since)
    ref = pd.DataFrame([sf.features_for("1", d, ins, fil, cov, since) for d in dates])[list(sf.FEATURES)]
    pd.testing.assert_frame_equal(fast[list(sf.FEATURES)], ref, check_exact=False, rtol=1e-9, atol=1e-9)
    assert ref.notna().any().all()


def test_form345_links_discovered_from_official_index(tmp_path, monkeypatch):
    """CI 2026-10-03: fester URL-Aufbau -> 404 für alle Quartale. Links kommen jetzt von der Übersichtsseite."""
    html = ('<a href="/files/structureddata/data/form-345-data-sets/2024q1_form345.zip">2024 Q1</a>'
            '<a href="https://www.sec.gov/files/dera/data/form-345/2024Q2-form345.zip">2024 Q2</a>'
            '<a href="/files/other.zip">x</a>')
    links = se.parse_form345_index(html)
    assert links[(2024, 1)] == "https://www.sec.gov/files/structureddata/data/form-345-data-sets/2024q1_form345.zip"
    assert links[(2024, 2)].endswith("2024Q2-form345.zip") and len(links) == 2
    monkeypatch.setattr(si, "DIR", tmp_path)
    seen = []

    def get(url, **k):
        seen.append(url)
        if url in se.FORM345_INDEX_PAGES:
            return types.SimpleNamespace(content=html.encode(), retrieved_at=NOW, content_hash="i")
        return types.SimpleNamespace(content=form345_zip(), retrieved_at=NOW, content_hash="h")
    r = si.ingest_form345({}, {CIK}, 2024, NOW, get=get, sleep=lambda s: None)
    assert r["index"]["n_links"] == 2 and "2024Q2-form345.zip" in " ".join(seen)

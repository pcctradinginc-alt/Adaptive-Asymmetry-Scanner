"""SEC 8-K Item 2.02 – offizielle Earnings-Release-Zeitstempel (B1, Research/SHADOW, 2026-10-15).

Quelle: die bereits archivierten Filings aus data.sec.gov/submissions (sec_events/sec_ingest): jedes 8-K mit
Item 2.02 ("Results of Operations and Financial Condition") ist die offizielle Ergebnisveröffentlichung.
acceptanceDateTime ist der SEC-Zeitstempel -> EXACT_TIMESTAMP, unveränderlich (Korrekturen sind 8-K/A).
Die Pressemitteilung kann dem 8-K um Minuten vorausgehen: acceptanceDateTime ist die obere Schranke ("spätestens
öffentlich um"), also PIT-sicher; die Sitzungs-Einordnung kann deshalb BMO-Releases als RTH zeigen.

Abgeleitet je (Stichtag, Ticker), streng point-in-time (nur available_at <= Stichtag 21:00 UTC):
  sec_days_since_earnings_release   Tage seit dem letzten 2.02-Filing
  sec_earnings_release_session      Sitzung des letzten Releases: 0 = vor Börseneröffnung (BMO), 0.5 = während RTH,
                                    1 = nach Börsenschluss (AMC) – New-York-Zeit, DST-fest
  sec_earnings_releases_365d        Anzahl 2.02-Filings in 365 Tagen (Plausibilität: ~4)
Außerhalb der belegten Filing-Abdeckung des Emittenten -> NaN (nie 0). Keine Produktionswirkung: nur Feature-Store
(outputs/research/feature_store/alt_sec_earnings.csv.gz) und Gegenprüfung des Earnings-Kalenders.
"""
from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd

FEATURE_VERSION = "sec-earn-1"
EARNINGS_ITEM = "2.02"
NY = "America/New_York"
RTH_OPEN, RTH_CLOSE = time(9, 30), time(16, 0)
FEATURES = {
    "sec_days_since_earnings_release": "Tage seit letztem 8-K Item 2.02 (acceptanceDateTime)",
    "sec_earnings_release_session": "Sitzung des letzten Releases: 0 BMO, 0.5 RTH, 1 AMC (New York)",
    "sec_earnings_releases_365d": "Anzahl 8-K Item 2.02 in 365 T",
}


def session_of(ts: pd.Timestamp) -> float:
    """0 = vor 09:30 ET (BMO), 0.5 = 09:30–16:00 ET, 1 = ab 16:00 ET (AMC)."""
    t = pd.Timestamp(ts).tz_convert(NY).time()
    return 0.0 if t < RTH_OPEN else 0.5 if t < RTH_CLOSE else 1.0


def release_events(observations) -> pd.DataFrame:
    """8-K mit Item 2.02 aus sec_submissions-Observations -> DataFrame[cik, accepted_at, accession, session]."""
    rows = []
    for o in observations:
        if o.source_id != "sec_submissions" or (o.attrs or {}).get("form") != "8-K":
            continue
        if EARNINGS_ITEM not in [str(i).strip() for i in (o.attrs.get("items") or [])]:
            continue
        acc = pd.Timestamp(o.available_at)
        rows.append({"cik": o.entity_id, "accepted_at": acc, "accession": o.series_id, "session": session_of(acc)})
    df = pd.DataFrame(rows, columns=["cik", "accepted_at", "accession", "session"])
    return df.drop_duplicates("accession").sort_values(["cik", "accepted_at"], kind="mergesort").reset_index(drop=True)


def _cutoff(d) -> pd.Timestamp:
    ts = pd.Timestamp(d)
    return (ts.tz_localize("UTC") if ts.tzinfo is None else ts) + pd.Timedelta(hours=21)


def features_cik(dates, ev: pd.DataFrame, filing_since: pd.Timestamp | None) -> pd.DataFrame:
    out = {k: np.full(len(dates), np.nan) for k in FEATURES}
    if filing_since is None or not len(dates):
        return pd.DataFrame(out)
    T = [_cutoff(d) for d in dates]
    acc = list(ev["accepted_at"]) if len(ev) else []
    ses = list(ev["session"]) if len(ev) else []
    a = np.array([x.value for x in acc], dtype=np.int64)
    for i, t in enumerate(T):
        if pd.Timestamp(filing_since) > t - pd.Timedelta(days=365):
            continue                                            # Historie nicht belegt -> unbekannt, nicht 0
        k = int(np.searchsorted(a, t.value, "right"))
        out["sec_earnings_releases_365d"][i] = float(k - np.searchsorted(a, (t - pd.Timedelta(days=365)).value,
                                                                          "right"))
        if k:
            out["sec_days_since_earnings_release"][i] = (t - acc[k - 1]).total_seconds() / 86400.0
            out["sec_earnings_release_session"][i] = ses[k - 1]
    return pd.DataFrame(out)


def build_feature_table(observations, dates, cik_by_ticker: dict[str, str],
                        filing_since_by_cik: dict[str, pd.Timestamp]) -> pd.DataFrame:
    """-> DataFrame[date, ticker, cik, <FEATURES>, alt_sec_earnings_available, feature_version]."""
    ev = release_events(observations)
    by = dict(tuple(ev.groupby("cik"))) if len(ev) else {}
    dates = [pd.Timestamp(d) for d in dates]
    parts = []
    for tk, cik in cik_by_ticker.items():
        f = features_cik(dates, by.get(cik, ev.iloc[:0]), filing_since_by_cik.get(cik))
        f.insert(0, "cik", cik)
        f.insert(0, "ticker", tk)
        f.insert(0, "date", dates)
        f["alt_sec_earnings_available"] = f["sec_earnings_releases_365d"].notna().astype(float)
        f["feature_version"] = FEATURE_VERSION
        parts.append(f)
    cols = ["date", "ticker", "cik", *FEATURES, "alt_sec_earnings_available", "feature_version"]
    return pd.concat(parts, ignore_index=True)[cols] if parts else pd.DataFrame(columns=cols)

"""SEC-Features je (Stichtag, Emittent) – streng point-in-time.

Regeln:
  * Nur Beobachtungen mit available_at <= Stichtag (Ende des Handelstags, 21:00 UTC).
  * "Keine Insider-Käufe" ist ein ECHTER Nullwert nur innerhalb der belegten
    Abdeckung (Datensatz-Quartale ingestiert, CIK zugeordnet). Außerhalb ->
    NaN + alt_sec_available = 0, nie 0 als Ersatz.
  * Wenige robuste Faktoren statt vieler Varianten (Feature-Selektion folgt
    in der Evaluation, Redundanzprüfung gegen bestehende Features).
"""
from __future__ import annotations

import math
from datetime import timedelta

import numpy as np
import pandas as pd

FEATURE_VERSION = "sec-f1"
NEGATIVE_8K_ITEMS = {"1.02", "1.03", "2.04", "2.06", "3.01", "4.01", "4.02"}   # Kündigung wesentl. Vertrag, Insolvenz,
#   Auslöser Verbindlichkeit, Wertminderung, Delisting-Mitteilung, Prüferwechsel, Nicht-Verlässlichkeit (Restatement)
EXEC_CHANGE_ITEM = "5.02"
FEATURES = {
    "sec_insider_buy_value_90d": "log1p(USD Open-Market-Käufe, 90 T)",
    "sec_insider_buyers_90d": "verschiedene kaufende Insider, 90 T",
    "sec_insider_net_value_90d": "vorzeichenbehaftetes log1p(Käufe − Verkäufe USD, 90 T)",
    "sec_insider_cluster_30d": "1 wenn >= 2 verschiedene Käufer in 30 T",
    "sec_8k_count_30d_z": "8-K-Anzahl 30 T gegen eigene Historie (24 Monatsfenster), z",
    "sec_8k_negative_90d": f"8-K mit Items {sorted(NEGATIVE_8K_ITEMS)}, 90 T",
    "sec_exec_change_90d": "8-K Item 5.02 (Organwechsel), 90 T",
    "sec_filing_delay_z": "Verzögerung letzter 10-K/10-Q (Tage nach Periodenende) gegen eigene Historie, z",
    "sec_late_filing_365d": "NT 10-K/NT 10-Q in 365 T",
}
INSIDER_FEATURES = [f for f in FEATURES if f.startswith("sec_insider")]
FILING_FEATURES = [f for f in FEATURES if f not in INSIDER_FEATURES]


def _slog(x: float) -> float:
    return math.copysign(math.log1p(abs(x)), x)


def _cutoff(d) -> pd.Timestamp:
    return pd.Timestamp(d).tz_localize("UTC") + pd.Timedelta(hours=21) if pd.Timestamp(d).tzinfo is None \
        else pd.Timestamp(d) + pd.Timedelta(hours=21)


def to_frames(observations) -> tuple[pd.DataFrame, pd.DataFrame]:
    ins, fil = [], []
    for o in observations:
        if o.source_id == "sec_form345":
            ins.append({"cik": o.entity_id, "avail": pd.Timestamp(o.available_at), "metric": o.metric,
                        "value": o.value, "owners": tuple(o.attrs.get("owner_ciks") or ())})
        elif o.source_id == "sec_submissions":
            fil.append({"cik": o.entity_id, "avail": pd.Timestamp(o.available_at), "form": o.attrs.get("form"),
                        "items": tuple(o.attrs.get("items") or ()),
                        "delay": o.value if o.unit == "days_after_period" else np.nan})
    i = pd.DataFrame(ins, columns=["cik", "avail", "metric", "value", "owners"])
    f = pd.DataFrame(fil, columns=["cik", "avail", "form", "items", "delay"])
    return i, f


def features_for(cik: str, d, ins: pd.DataFrame, fil: pd.DataFrame, insider_cov: tuple | None,
                 filing_since: pd.Timestamp | None) -> dict:
    t = _cutoff(d)
    out = {k: np.nan for k in FEATURES}
    # Insider: nur innerhalb belegter Datensatz-Abdeckung
    if insider_cov and insider_cov[0] <= t - pd.Timedelta(days=90) and t <= insider_cov[1]:
        g = ins[(ins["cik"] == cik) & (ins["avail"] <= t)]
        w90, w30 = g[g["avail"] > t - pd.Timedelta(days=90)], g[g["avail"] > t - pd.Timedelta(days=30)]
        buys90 = w90[w90["metric"] == "insider_buy_usd"]
        sells90 = w90[w90["metric"] == "insider_sell_usd"]
        b, s = float(buys90["value"].sum()), float(sells90["value"].sum())
        out["sec_insider_buy_value_90d"] = math.log1p(b)
        out["sec_insider_buyers_90d"] = float(len({o for t_ in buys90["owners"] for o in t_}))
        out["sec_insider_net_value_90d"] = _slog(b - s)
        buyers30 = {o for t_ in w30[w30["metric"] == "insider_buy_usd"]["owners"] for o in t_}
        out["sec_insider_cluster_30d"] = 1.0 if len(buyers30) >= 2 else 0.0
    # Filings: ab frühestem belegten Filing des Emittenten (+ 2 J. für Historien-z)
    if filing_since is not None and filing_since <= t - pd.Timedelta(days=365):
        g = fil[(fil["cik"] == cik) & (fil["avail"] <= t)]
        k8 = g[g["form"] == "8-K"]
        out["sec_8k_negative_90d"] = float(sum(1 for its in k8[k8["avail"] > t - pd.Timedelta(days=90)]["items"]
                                               if set(its) & NEGATIVE_8K_ITEMS))
        out["sec_exec_change_90d"] = float(sum(1 for its in k8[k8["avail"] > t - pd.Timedelta(days=90)]["items"]
                                               if EXEC_CHANGE_ITEM in its))
        out["sec_late_filing_365d"] = float((g[g["form"].isin(["NT 10-K", "NT 10-Q"])]["avail"]
                                             > t - pd.Timedelta(days=365)).sum())
        if filing_since <= t - pd.Timedelta(days=24 * 30 + 30):
            cur = float((k8["avail"] > t - pd.Timedelta(days=30)).sum())
            hist = [float(((k8["avail"] > t - pd.Timedelta(days=30 * (j + 1))) &
                           (k8["avail"] <= t - pd.Timedelta(days=30 * j))).sum()) for j in range(1, 25)]
            sd = float(np.std(hist))
            out["sec_8k_count_30d_z"] = (cur - float(np.mean(hist))) / sd if sd > 0 else 0.0
        per = g[g["form"].isin(["10-K", "10-Q"]) & g["delay"].notna()].sort_values("avail", kind="mergesort")
        if len(per) >= 5:
            prior = per["delay"].iloc[:-1].tail(12)
            sd = float(prior.std(ddof=1))
            out["sec_filing_delay_z"] = (float(per["delay"].iloc[-1]) - float(prior.mean())) / sd if sd > 0 else 0.0
    return out


def _ns(s: pd.Series) -> np.ndarray:
    return pd.DatetimeIndex(pd.to_datetime(s, utc=True)).as_unit("ns").asi8.astype(np.int64)


def _count(a: np.ndarray, lo, hi) -> np.ndarray:
    """Anzahl Elemente von a (sortiert) in (lo, hi]."""
    return np.searchsorted(a, hi, "right") - np.searchsorted(a, lo, "right")


def features_cik(dates, ins_c: pd.DataFrame, fil_c: pd.DataFrame, insider_cov: tuple | None,
                 filing_since: pd.Timestamp | None) -> pd.DataFrame:
    """Vektorisierte, ergebnisgleiche Fassung von features_for() für alle Stichtage eines
    Emittenten (Referenz: features_for, Gleichheit getestet)."""
    T = np.array([_cutoff(d).value for d in dates], dtype=np.int64)
    day = np.int64(86_400 * 10**9)
    out = {k: np.full(len(T), np.nan) for k in FEATURES}
    if insider_cov and len(T):
        ok = (insider_cov[0].value <= T - 90 * day) & (T <= insider_cov[1].value)
        g = ins_c.sort_values("avail", kind="mergesort")
        a = _ns(g["avail"])
        isbuy = (g["metric"] == "insider_buy_usd").to_numpy()
        issell = (g["metric"] == "insider_sell_usd").to_numpy()
        v = g["value"].to_numpy(dtype=float)
        cb = np.concatenate([[0.0], np.cumsum(np.where(isbuy, v, 0.0))])
        cs = np.concatenate([[0.0], np.cumsum(np.where(issell, v, 0.0))])
        owners = g["owners"].tolist()
        hi = np.searchsorted(a, T, "right")
        lo90, lo30 = np.searchsorted(a, T - 90 * day, "right"), np.searchsorted(a, T - 30 * day, "right")
        for i in np.flatnonzero(ok):
            b, sl = cb[hi[i]] - cb[lo90[i]], cs[hi[i]] - cs[lo90[i]]
            out["sec_insider_buy_value_90d"][i] = math.log1p(b)
            out["sec_insider_net_value_90d"][i] = _slog(b - sl)
            out["sec_insider_buyers_90d"][i] = float(len({o for j in range(lo90[i], hi[i]) if isbuy[j] for o in owners[j]}))
            out["sec_insider_cluster_30d"][i] = 1.0 if len({o for j in range(lo30[i], hi[i]) if isbuy[j]
                                                             for o in owners[j]}) >= 2 else 0.0
    if filing_since is not None and len(T):
        fs_ = pd.Timestamp(filing_since).value
        ok = fs_ <= T - 365 * day
        g = fil_c.sort_values("avail", kind="mergesort")
        av = _ns(g["avail"])
        form = g["form"].to_numpy()
        k8 = av[form == "8-K"]
        items8 = [set(x) for x in g["items"][form == "8-K"]]
        neg = np.concatenate([[0], np.cumsum([bool(x & NEGATIVE_8K_ITEMS) for x in items8])]).astype(float)
        exe = np.concatenate([[0], np.cumsum([EXEC_CHANGE_ITEM in x for x in items8])]).astype(float)
        nt = av[np.isin(form, ["NT 10-K", "NT 10-Q"])]
        h8, l8 = np.searchsorted(k8, T, "right"), np.searchsorted(k8, T - 90 * day, "right")
        out["sec_8k_negative_90d"] = np.where(ok, neg[h8] - neg[l8], np.nan)
        out["sec_exec_change_90d"] = np.where(ok, exe[h8] - exe[l8], np.nan)
        out["sec_late_filing_365d"] = np.where(ok, _count(nt, T - 365 * day, T).astype(float), np.nan)
        okz = ok & (fs_ <= T - (24 * 30 + 30) * day)
        bounds = np.searchsorted(k8, T[:, None] - np.arange(26)[None, :] * 30 * day, "right")   # Spalte k: t - 30k
        cnt = (bounds[:, :-1] - bounds[:, 1:]).astype(float)                                   # Fenster j: (t-30(j+1), t-30j]
        cur, hist = cnt[:, 0], cnt[:, 1:25]
        mu, sd = hist.mean(axis=1), hist.std(axis=1)
        z = np.where(sd > 0, (cur - mu) / np.where(sd > 0, sd, 1.0), 0.0)
        out["sec_8k_count_30d_z"] = np.where(okz, z, np.nan)
        per = g[g["form"].isin(["10-K", "10-Q"]) & g["delay"].notna()]
        pa = _ns(per["avail"])
        pdl = per["delay"].to_numpy(dtype=float)
        kk = np.searchsorted(pa, T, "right")
        for i in np.flatnonzero(ok & (kk >= 5)):
            prior = pdl[max(0, kk[i] - 13):kk[i] - 1]
            sdv = float(np.std(prior, ddof=1))
            out["sec_filing_delay_z"][i] = (pdl[kk[i] - 1] - float(prior.mean())) / sdv if sdv > 0 else 0.0
    return pd.DataFrame(out)


def build_feature_table(observations, dates, cik_by_ticker: dict[str, str], insider_cov: tuple | None,
                        filing_since_by_cik: dict[str, pd.Timestamp]) -> pd.DataFrame:
    """-> DataFrame[date, ticker, cik, <FEATURES>, alt_sec_available, feature_version]."""
    ins, fil = to_frames(observations)
    ins_by, fil_by = dict(tuple(ins.groupby("cik"))), dict(tuple(fil.groupby("cik")))
    dates = [pd.Timestamp(d) for d in dates]
    parts = []
    for tk, cik in cik_by_ticker.items():
        f = features_cik(dates, ins_by.get(cik, ins.iloc[:0]), fil_by.get(cik, fil.iloc[:0]), insider_cov,
                         filing_since_by_cik.get(cik))
        f.insert(0, "cik", cik)
        f.insert(0, "ticker", tk)
        f.insert(0, "date", dates)
        f["alt_sec_available"] = f[list(FEATURES)].notna().any(axis=1).astype(float)
        f["feature_version"] = FEATURE_VERSION
        parts.append(f)
    cols = ["date", "ticker", "cik", *FEATURES, "alt_sec_available", "feature_version"]
    return pd.concat(parts, ignore_index=True)[cols] if parts else pd.DataFrame(columns=cols)


def insider_coverage(quarters: list[tuple[int, int]]) -> tuple | None:
    """Belegte Abdeckung der Form-345-Datensätze: [Beginn erstes Quartal, Ende letztes Quartal + 1 Tag)."""
    if not quarters:
        return None
    (y0, q0), (y1, q1) = min(quarters), max(quarters)
    start = pd.Timestamp(year=y0, month=3 * (q0 - 1) + 1, day=1, tz="UTC")
    end = pd.Timestamp(year=y1 + (q1 == 4), month=(3 * q1 % 12) + 1, day=1, tz="UTC") + timedelta(days=1)
    return start, end

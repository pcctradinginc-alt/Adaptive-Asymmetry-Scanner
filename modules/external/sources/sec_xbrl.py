"""SEC XBRL companyfacts – point-in-time Fundamentaldaten (Research/SHADOW, nie Produktion).

    python -m modules.external.sources.sec_xbrl [--budget 800] [--features-only]

Quelle: https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json (offiziell, kostenlos,
SEC Fair Access <= 10 Anfragen/s, User-Agent mit Kontakt). Jede Zahl trägt das Filing,
in dem sie berichtet wurde (accn, form, filed). Eine spätere Korrektur (10-K/A, Restatement
im Folgejahr) erscheint als EIGENE Zeile mit späterem `filed`.

Point-in-Time / Revisionen:
  * Jede (Konzept, Periode, Filing)-Zahl ist eine eigene Beobachtung,
    available_at = filed + 1 Tag (CONSERVATIVE_DATE; `filed` ist ein Datum), vintage_time = filed.
  * Zum Stichtag t gilt je Periode der Wert aus dem JÜNGSTEN Filing mit available_at <= t
    – also genau das, was ein Investor damals kannte. Heutige revidierte Werte werden nie
    rückwirkend verwendet.
  * Q4 = Geschäftsjahr − (Q1+Q2+Q3), nur wenn alle Teile zum Stichtag bekannt sind.
  * Fehlt ein Wert oder ist er veraltet (letzte Periode > MAX_STALE_DAYS vor t) -> NaN.

Rohdaten: companyfacts ist je Firma mehrere MB groß. Gespeichert werden die normalisierten
Beobachtungen der verwendeten Konzepte (append-only, je Zeile alle PIT-Felder und der
payload_hash des Abrufs) plus je Abruf request/retrieved_at/payload_hash im State. Der
vollständige Roh-Payload wird nicht eingecheckt (Größe); der Hash belegt, was gelesen wurde.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from modules.external.http import SchemaError
from modules.external.pit import AvailabilityPrecision, Observation

log = logging.getLogger(__name__)

PARSER_VERSION = "sec-xbrl-1"
FEATURE_VERSION = "xbrl-f1"
URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
DIR = Path("outputs/external_data/sec")
STORE = DIR / "xbrl.csv.gz"
STATE = DIR / "xbrl_state.json"
HEALTH = DIR / "xbrl_health.json"
FEATURE_PATH = Path("outputs/research/feature_store/alt_xbrl.csv.gz")
SEC_MIN_INTERVAL = 0.12
MAX_STALE_DAYS = 200
FORMS = {"10-K", "10-Q", "10-K/A", "10-Q/A", "20-F", "40-F"}

# Konzept -> (Taxonomie, Kandidaten in Prioritätsreihenfolge, Einheit, Art)
CONCEPTS = {
    "revenue": ("us-gaap", ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet",
                            "RevenueFromContractWithCustomerIncludingAssessedTax"], "USD", "duration"),
    "eps_diluted": ("us-gaap", ["EarningsPerShareDiluted", "EarningsPerShareBasic"], "USD/shares", "duration"),
    "net_income": ("us-gaap", ["NetIncomeLoss"], "USD", "duration"),
    "cfo": ("us-gaap", ["NetCashProvidedByUsedInOperatingActivities"], "USD", "duration"),
    "assets": ("us-gaap", ["Assets"], "USD", "instant"),
    "shares": ("dei", ["EntityCommonStockSharesOutstanding"], "shares", "instant"),
}
FEATURES = {
    "xbrl_rev_yoy": "log(Umsatz letztes Quartal / Vorjahresquartal), Stand zum Stichtag",
    "xbrl_sue": "EPS-Überraschung: (EPS_q − EPS_q-4) / sd der letzten 8 saisonalen Differenzen",
    "xbrl_accruals": "(Nettoergebnis − operativer Cashflow) / Bilanzsumme, letztes Geschäftsjahr",
    "xbrl_asset_growth": "log(Bilanzsumme / Bilanzsumme ~1 Jahr zuvor)",
    "xbrl_share_change": "log(Aktienanzahl / Aktienanzahl ~1 Jahr zuvor) (Emission > 0, Rückkauf < 0)",
}


def _date(s) -> datetime | None:
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def parse_companyfacts(payload: dict, cik: str, retrieved_at: datetime, content_hash: str = "") -> list[Observation]:
    """-> Beobachtungen der verwendeten Konzepte (je Konzept der erste vorhandene Kandidat)."""
    if not isinstance(payload, dict) or "facts" not in payload:
        raise SchemaError(f"companyfacts: 'facts' fehlt (Schlüssel {sorted(payload)[:6] if isinstance(payload, dict) else type(payload)})")
    c10 = str(int(cik)).zfill(10)
    out = []
    for key, (tax, names, unit, kind) in CONCEPTS.items():
        facts = (payload["facts"] or {}).get(tax) or {}
        tag = next((n for n in names if n in facts and unit in ((facts[n] or {}).get("units") or {})), None)
        if tag is None:
            continue
        for f in facts[tag]["units"][unit]:
            if not isinstance(f, dict) or "val" not in f or "end" not in f or "filed" not in f:
                raise SchemaError(f"companyfacts {tag}: Fakt ohne val/end/filed")
            if f.get("form") not in FORMS:
                continue
            end, filed = _date(f["end"]), _date(f["filed"])
            start = _date(f.get("start")) if kind == "duration" else None
            if end is None or filed is None or (kind == "duration" and start is None):
                continue
            accn = f.get("accn") or ""
            out.append(Observation(
                source_id="sec_companyfacts", dataset=key, entity_id=f"cik:{c10}", metric=key,
                series_id=f"xbrl:{c10}:{key}:{(start.date().isoformat() if start else '')}:{end.date().isoformat()}:{accn}",
                value=float(f["val"]), unit=unit, observation_time=end, available_at=filed + timedelta(days=1),
                retrieved_at=retrieved_at, availability_precision=AvailabilityPrecision.CONSERVATIVE_DATE,
                parser_version=PARSER_VERSION, vintage_time=filed, payload_hash=content_hash,
                attrs={"tag": tag, "start": start.date().isoformat() if start else None, "end": end.date().isoformat(),
                       "form": f.get("form"), "fy": f.get("fy"), "fp": f.get("fp"), "accn": accn}))
    return drop_repeats(out)


def drop_repeats(obs: list[Observation]) -> list[Observation]:
    """Spätere Filings wiederholen Vorperioden (Vergleichszahlen). Unveränderte Wiederholungen
    tragen keine neue Information (der Wert zum Stichtag bleibt gleich) -> nur Erstmeldung
    und echte Revisionen (abweichender Wert) behalten."""
    keep, last = [], {}
    for o in sorted(obs, key=lambda o: (o.metric, o.attrs.get("start") or "", o.attrs["end"], o.available_at)):
        k = (o.metric, o.attrs.get("start"), o.attrs["end"])
        if k in last and last[k] == o.value:
            continue
        last[k] = o.value
        keep.append(o)
    return keep


# ── Features (je CIK; Wissenstand ändert sich nur an Filing-Tagen) ─────────────
def _known(df: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """Je (metric, start, end) der Wert des jüngsten Filings mit avail <= t."""
    k = df[df["avail"] <= t]
    if k.empty:
        return k
    return k.sort_values("avail", kind="mergesort").drop_duplicates(["metric", "start", "end"], keep="last")


def _quarters(k: pd.DataFrame, metric: str) -> pd.Series:
    """Quartalswerte je Periodenende (3-Monats-Fakten + Q4 aus GJ − 9M YTD bzw. − Q1..Q3)."""
    d = k[k["metric"] == metric]
    if d.empty:
        return pd.Series(dtype=float)
    st, en, va = d["start"].to_numpy(), d["end"].to_numpy(), d["value"].to_numpy(dtype=float)
    days = (en - st) / np.timedelta64(1, "D")
    q: dict = {}
    for e, v, n in zip(en, va, days):
        if 80 <= n <= 100:
            q[e] = v
    nine = {s_: v for s_, v, n in zip(st, va, days) if 260 <= n <= 285}
    for s_, e, v, n in zip(st, en, va, days):
        if not (350 <= n <= 380) or e in q:
            continue
        if s_ in nine:
            q[e] = v - nine[s_]
            continue
        parts = [x for x in q if s_ < x < e - np.timedelta64(60, "D")]
        if len(parts) == 3:
            q[e] = v - sum(q[x] for x in parts)
    if not q:
        return pd.Series(dtype=float)
    keys = sorted(q)
    return pd.Series([q[x] for x in keys], index=pd.DatetimeIndex(keys, tz="UTC") if pd.DatetimeIndex(keys).tz is None
                     else pd.DatetimeIndex(keys))


def _instants(k: pd.DataFrame, metric: str) -> pd.Series:
    d = k[k["metric"] == metric].set_index("end")["value"]
    return d[~d.index.duplicated(keep="last")].sort_index()


def _yoy(s: pd.Series, log_ratio: bool = True) -> tuple[float, pd.Timestamp | None]:
    """-> (Wert, Periodenende des jüngsten Werts) – Staleness prüft der Aufrufer."""
    if s.empty:
        return np.nan, None
    last_end, last = s.index[-1], s.iloc[-1]
    prev = s[(s.index <= last_end - pd.Timedelta(days=340)) & (s.index >= last_end - pd.Timedelta(days=390))]
    if prev.empty:
        return np.nan, last_end
    p = prev.iloc[-1]
    if not log_ratio:
        return float(last - p), last_end
    return (float(math.log(last / p)) if last > 0 and p > 0 else np.nan), last_end


def _sue(q: pd.Series) -> tuple[float, pd.Timestamp | None]:
    if len(q) < 6:
        return np.nan, None
    t = q.index.values
    v = q.to_numpy(dtype=float)
    lo = np.searchsorted(t, t - np.timedelta64(390, "D"), "left")
    hi = np.searchsorted(t, t - np.timedelta64(340, "D"), "right")
    ok = hi > lo                                          # Vorjahresquartal über das Datum (Lücken-robust)
    diffs = v[ok] - v[hi[ok] - 1]
    if len(diffs) < 5 or not ok[-1]:
        return np.nan, q.index[-1]
    hist = diffs[-9:-1]
    sd = float(hist.std(ddof=1)) if len(hist) >= 4 else 0.0
    return (float(diffs[-1] / sd) if sd > 0 else np.nan), q.index[-1]


def _accruals(k: pd.DataFrame) -> tuple[float, pd.Timestamp | None]:
    d = k[k["metric"].isin(["net_income", "cfo"])]
    d = d.assign(days=(d["end"] - d["start"]).dt.days)
    d = d[(d["days"] >= 350) & (d["days"] <= 380)]
    if d.empty:
        return np.nan, None
    end = d["end"].max()
    ni, cfo = d[(d["metric"] == "net_income") & (d["end"] == end)], d[(d["metric"] == "cfo") & (d["end"] == end)]
    a = _instants(k, "assets")
    a = a[a.index == end]
    if ni.empty or cfo.empty or a.empty or a.iloc[-1] <= 0:
        return np.nan, end
    return float((ni.iloc[-1]["value"] - cfo.iloc[-1]["value"]) / a.iloc[-1]), end


MAX_AGE = {"xbrl_rev_yoy": MAX_STALE_DAYS, "xbrl_sue": MAX_STALE_DAYS, "xbrl_asset_growth": MAX_STALE_DAYS,
           "xbrl_share_change": MAX_STALE_DAYS, "xbrl_accruals": 450}


def state_features(k: pd.DataFrame) -> dict[str, tuple[float, pd.Timestamp | None]]:
    return {"xbrl_rev_yoy": _yoy(_quarters(k, "revenue")), "xbrl_sue": _sue(_quarters(k, "eps_diluted")),
            "xbrl_accruals": _accruals(k), "xbrl_asset_growth": _yoy(_instants(k, "assets")),
            "xbrl_share_change": _yoy(_instants(k, "shares"))}


def features_cik(obs: pd.DataFrame, dates) -> pd.DataFrame:
    """obs: DataFrame[metric, start, end, avail, value] einer Entität. -> je Stichtag FEATURES.
    Der Wissensstand ändert sich nur an Filing-Tagen: Features je Wissensstand einmal,
    danach je Stichtag nur Staleness (Periodenende zu alt -> NaN)."""
    T = pd.DatetimeIndex([pd.Timestamp(d).tz_localize("UTC") if pd.Timestamp(d).tzinfo is None
                          else pd.Timestamp(d).tz_convert("UTC") for d in dates]).normalize() + pd.Timedelta(hours=21)
    out = pd.DataFrame(np.nan, index=range(len(T)), columns=list(FEATURES))
    if obs.empty:
        return out
    obs = obs.sort_values("avail")
    points = pd.DatetimeIndex(pd.to_datetime(obs["avail"], utc=True).unique()).sort_values()
    idx = points.searchsorted(T, side="right") - 1
    states = {j: state_features(_known(obs, points[j])) for j in np.unique(idx[idx >= 0])}
    tv = T.values
    for f in FEATURES:
        vals = np.full(len(T), np.nan)
        for j, st in states.items():
            v, ref = st[f]
            if ref is None or not np.isfinite(v):
                continue
            m = (idx == j) & ((tv - np.datetime64(ref.tz_convert(None))) <= np.timedelta64(MAX_AGE[f], "D"))
            vals[m] = v
        out[f] = vals
    return out


def build_feature_table(rows: pd.DataFrame, dates, cik_by_ticker: dict[str, str]) -> pd.DataFrame:
    """rows: Store-Zeilen (entity_id, metric, value, available_at, attrs). -> [date, ticker, cik, FEATURES, alt_xbrl_available]."""
    dates = [pd.Timestamp(d) for d in dates]
    cols = ["date", "ticker", "cik", *FEATURES, "alt_xbrl_available", "feature_version"]
    if len(rows):
        at = rows["attrs"].map(lambda a: json.loads(a) if isinstance(a, str) else (a or {}))
        obs = pd.DataFrame({"cik": rows["entity_id"].astype(str).str.replace("cik:", "", regex=False),
                            "metric": rows["metric"], "value": pd.to_numeric(rows["value"], errors="coerce"),
                            "avail": pd.to_datetime(rows["available_at"], utc=True),
                            "start": pd.to_datetime(at.map(lambda a: a.get("start")), utc=True),
                            "end": pd.to_datetime(at.map(lambda a: a.get("end")), utc=True)}).dropna(subset=["value"])
    else:
        obs = pd.DataFrame(columns=["cik", "metric", "value", "avail", "start", "end"])
    by = dict(tuple(obs.groupby("cik"))) if len(obs) else {}
    parts = []
    for tk, cik in cik_by_ticker.items():
        c10 = str(int(cik)).zfill(10)
        f = features_cik(by.get(c10, obs.iloc[:0]), dates)
        f.insert(0, "cik", c10)
        f.insert(0, "ticker", tk)
        f.insert(0, "date", dates)
        f["alt_xbrl_available"] = f[list(FEATURES)].notna().any(axis=1).astype(float)
        f["feature_version"] = FEATURE_VERSION
        parts.append(f)
    return pd.concat(parts, ignore_index=True)[cols] if parts else pd.DataFrame(columns=cols)


# ── Ingest (inkrementell, budgetiert) ───────────────────────────────────────
def due_ciks(state: dict, ciks: list[str], last_report: dict[str, str]) -> list[str]:
    """Erst nie abgerufene, dann solche mit neuem 10-K/10-Q seit dem letzten Abruf."""
    fetched = state.get("ciks") or {}
    never = [c for c in ciks if c not in fetched]
    stale = [c for c in ciks if c in fetched and last_report.get(c, "") > fetched[c].get("retrieved_at", "")]
    return never + sorted(stale, key=lambda c: fetched[c].get("retrieved_at", ""))


def ingest(state: dict, ciks: list[str], last_report: dict[str, str], get, sleep=time.sleep, budget: int = 800) -> dict:
    from modules.entity_resolution.sources import SEC_HEADERS
    from modules.external.sources.sec_ingest import merge_store, obs_to_rows
    res = {"requests": 0, "new_rows": 0, "errors": [], "schema_errors": 0, "not_found": 0, "budget_exhausted": False}
    frames = []
    st = state.setdefault("ciks", {})
    for c in due_ciks(state, ciks, last_report):
        if res["requests"] >= budget:
            res["budget_exhausted"] = True
            break
        c10 = str(int(c)).zfill(10)
        try:
            r = get(URL.format(cik=c10), headers=SEC_HEADERS, timeout=60)
            res["requests"] += 1
            obs = parse_companyfacts(r.json(), c10, r.retrieved_at, r.content_hash)
            frames.append(obs_to_rows(obs))
            st[c] = {"retrieved_at": r.retrieved_at.isoformat(), "payload_hash": r.content_hash, "n_obs": len(obs),
                     "request": URL.format(cik=c10)}
        except SchemaError as e:
            res["schema_errors"] += 1                     # Schema-Drift: laut melden, nichts raten
            res["errors"].append(f"schema {c10}: {e}"[:300])
        except Exception as e:  # noqa: BLE001
            res["requests"] += 1
            if "404" in str(e):
                res["not_found"] += 1                     # kein XBRL-Filer (z.B. nur Altfilings)
                st[c] = {"retrieved_at": datetime.now(timezone.utc).isoformat(), "not_found": True}
            else:
                res["errors"].append(f"{c10}: {e!r}"[:300])
        finally:
            sleep(SEC_MIN_INTERVAL)
    if frames:
        _, res["new_rows"] = merge_store(STORE, pd.concat(frames, ignore_index=True))
    return res


def health(state: dict, res: dict, n_ciks: int, now: datetime) -> dict:
    from modules.external.sources.sec_ingest import read_store
    st = read_store(STORE)
    ok = [v for v in (state.get("ciks") or {}).values() if not v.get("not_found")]
    return {"source_id": "sec_companyfacts", "checked_at": now.isoformat(), "parser_version": PARSER_VERSION,
            "last_observation": str(st["available_at"].max()) if len(st) else None, "n_observations": int(len(st)),
            "ciks_with_facts": len(ok), "universe_ciks": n_ciks,
            "coverage": round(len(ok) / n_ciks, 3) if n_ciks else 0.0,
            "requests": res.get("requests"), "budget_exhausted": res.get("budget_exhausted"),
            "schema_errors": res.get("schema_errors"), "not_found": res.get("not_found"),
            "error_rate": round(len(res.get("errors") or []) / max(1, res.get("requests") or 1), 3),
            "errors_sample": (res.get("errors") or [])[:5]}


def last_reports() -> dict[str, str]:
    """Jüngstes 10-K/10-Q je CIK aus dem SEC-Filing-Store (für inkrementelle Abrufe)."""
    from modules.external.sources.sec_ingest import read_store
    fil = read_store(DIR / "filings.csv.gz")
    if not len(fil):
        return {}
    f = fil[fil["attrs"].astype(str).str.contains('"form": "(?:10-K|10-Q)', regex=True)]
    return {str(k).replace("cik:", ""): str(v) for k, v in f.groupby("entity_id")["available_at"].max().items()}


def main(argv=None) -> int:
    from modules.entity_resolution.store import DEFAULT_PATH as ENTITY_PATH, EntityStore
    from modules.external.http import fetch
    from modules.external.sources.sec_ingest import read_store
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=800)
    ap.add_argument("--features-only", action="store_true")
    args = ap.parse_args(argv)
    now = datetime.now(timezone.utc)
    store = EntityStore(ENTITY_PATH)
    ident = {r.ticker: r.cik for r in store.records if r.usage == "research_ticker_identity" and r.valid_to is None}
    if not ident:
        log.error("sec_xbrl: keine Entity-Map – zuerst python -m modules.entity_resolution.build")
        return 1
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    if not args.features_only:
        res = ingest(state, sorted(set(ident.values())), last_reports(), fetch, budget=args.budget)
        DIR.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps(state, indent=1, sort_keys=True))
        HEALTH.write_text(json.dumps(health(state, res, len(set(ident.values())), now), indent=1))
        print(json.dumps({k: (v if k != "errors" else len(v)) for k, v in res.items()}))
    dates = pd.date_range("2015-01-02", now.date(), freq="W-FRI")
    feat = build_feature_table(read_store(STORE), dates, ident)
    FEATURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    feat.to_csv(FEATURE_PATH, index=False, compression="gzip")
    print(f"XBRL-Features: {len(feat)} Zeilen, verfügbar {feat['alt_xbrl_available'].mean():.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

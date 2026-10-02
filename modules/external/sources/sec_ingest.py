"""SEC Deep Events – inkrementeller Ingest + Feature-Store (Research/SHADOW).

    python -m modules.external.sources.sec_ingest [--start-year 2014] [--live-budget 400]

Speicher (kompakt, weil ~10^5 Ereignisse; jede Zeile behält ALLE PIT-Felder):
  outputs/external_data/sec/insider/<Jahr>.csv.gz   Open-Market-Transaktionen (partitioniert)
  outputs/external_data/sec/filings/<Jahr>.csv.gz   8-K / 10-K / 10-Q / NT-Filings
  outputs/external_data/sec/state.json       ingestierte Quartale (+Hash), Abdeckung
  outputs/external_data/sec/health.json      Source Health
  outputs/research/feature_store/alt_sec.csv.gz  Features je (Stichtag, Ticker) – im CI-Lauf
                                                 neu berechnet, nicht eingecheckt (.gitignore)
Ausfall einer Quelle -> Features dieser Quelle NaN (alt_sec_available=0),
niemals Ersatzwerte.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from modules.entity_resolution.sources import SEC_HEADERS
from modules.entity_resolution.store import DEFAULT_PATH as ENTITY_PATH, EntityStore
from modules.external.http import fetch
from modules.external.pit import AvailabilityPrecision, Observation
from modules.external.sources import sec_events as se
from modules.external.sources import sec_features as sf

log = logging.getLogger(__name__)
DIR = Path("outputs/external_data/sec")
FEATURE_PATH = Path("outputs/research/feature_store/alt_sec.csv.gz")
SCHEMA_VERSION = "sec-store-1"
SEC_MIN_INTERVAL = 0.12
COLS = ["source_id", "series_id", "entity_id", "metric", "value", "unit", "observation_time", "available_at",
        "retrieved_at", "availability_precision", "parser_version", "payload_hash", "attrs"]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def obs_to_rows(obs: list[Observation]) -> pd.DataFrame:
    return pd.DataFrame([{c: (json.dumps(getattr(o, c), sort_keys=True) if c == "attrs" else
                              (getattr(o, c).isoformat() if hasattr(getattr(o, c), "isoformat") else
                               (getattr(o, c).value if hasattr(getattr(o, c), "value") else getattr(o, c))))
                          for c in COLS} for o in obs], columns=COLS)


def rows_to_obs(df: pd.DataFrame) -> list[Observation]:
    out = []
    for r in df.itertuples(index=False):
        out.append(Observation(source_id=r.source_id, dataset="", series_id=r.series_id, entity_id=r.entity_id,
                               metric=r.metric, value=None if pd.isna(r.value) else float(r.value), unit=r.unit,
                               observation_time=r.observation_time, available_at=r.available_at,
                               retrieved_at=r.retrieved_at, availability_precision=r.availability_precision,
                               parser_version=r.parser_version, payload_hash=r.payload_hash,
                               attrs=json.loads(r.attrs) if isinstance(r.attrs, str) else {}))
    return out


def _store_dir(path: Path) -> Path:
    """insider.csv.gz -> insider/ (je Jahr der Verfügbarkeit eine Datei: nur das
    laufende Jahr ändert sich, ältere Partitionen bleiben byte-identisch)."""
    return path.parent / path.name.replace(".csv.gz", "")


def read_store(path: Path) -> pd.DataFrame:
    d = _store_dir(path)
    parts = sorted(d.glob("*.csv.gz")) if d.exists() else []
    return pd.concat([pd.read_csv(f, dtype=str) for f in parts], ignore_index=True) if parts \
        else pd.DataFrame(columns=COLS)


def merge_store(path: Path, new: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Append-only: vorhandene series_id bleiben unverändert (erste Kenntnis gilt)."""
    old = read_store(path)
    new = new.drop_duplicates("series_id") if len(new) else new
    add = new[~new["series_id"].isin(set(old["series_id"]))] if len(new) else new
    if len(add):
        add = add.astype(str)
        d = _store_dir(path)
        d.mkdir(parents=True, exist_ok=True)
        for year, g in add.groupby(add["available_at"].str[:4]):
            f = d / f"{year}.csv.gz"
            prev = pd.read_csv(f, dtype=str) if f.exists() else pd.DataFrame(columns=COLS)
            pd.concat([prev, g], ignore_index=True).sort_values("series_id").to_csv(
                f, index=False, compression={"method": "gzip", "mtime": 0})
    out = pd.concat([old, add], ignore_index=True) if len(add) else old
    return out, len(add)


def ingest_form345(state: dict, ciks: set[str], start_year: int, now: datetime, get=fetch, sleep=time.sleep) -> dict:
    done = state.setdefault("form345_quarters", {})
    res = {"fetched": [], "errors": [], "new_rows": 0}
    frames = []
    for y, q in se.form345_quarters(start_year, now):
        key = f"{y}Q{q}"
        if key in done:
            continue
        try:
            r = get(se.FORM345_URL.format(year=y, q=q), headers=SEC_HEADERS, timeout=120)
            obs = se.parse_form345_zip(r.content, ciks, r.retrieved_at)
            frames.append(obs_to_rows(obs))
            done[key] = {"content_hash": r.content_hash, "n_obs": len(obs), "retrieved_at": r.retrieved_at.isoformat()}
            res["fetched"].append(key)
        except Exception as e:  # noqa: BLE001 – Quartal fehlt -> Abdeckung endet davor, nichts erfinden
            res["errors"].append(f"{key}: {e!r}")
        sleep(SEC_MIN_INTERVAL)
    if frames:
        _, res["new_rows"] = merge_store(DIR / "insider.csv.gz", pd.concat(frames, ignore_index=True))
    return res


def ingest_submissions(state: dict, cik_list: list[str], now: datetime, get=fetch, sleep=time.sleep,
                       live_budget: int = 400) -> dict:
    """Alle Filing-Seiten je CIK (Historie) + live Form 4 nach Ende der Bulk-Abdeckung (budgetiert)."""
    from modules.alpha_sources import parse_form4_xml
    seen_pages = state.setdefault("submission_pages", {})
    since = state.setdefault("filing_since", {})
    res = {"ciks": 0, "errors": [], "new_rows": 0, "live_form4": 0, "live_complete": True}
    filing_frames, live_frames = [], []
    bulk_end = sf.insider_coverage([tuple(int(x) for x in k.split("Q")) for k in state.get("form345_quarters", {})])
    budget = live_budget
    for cik in cik_list:
        c10 = str(int(cik)).zfill(10)
        try:
            main = get(se.SUBMISSIONS_URL.format(name=f"CIK{c10}.json"), headers=SEC_HEADERS)
            sleep(SEC_MIN_INTERVAL)
            payload = main.json()
            rows = se.parse_submissions_filings(payload)
            for page in se.submissions_pages(payload):
                if page in seen_pages.get(c10, []):
                    continue                                       # ältere Seiten ändern sich nicht
                p = get(se.SUBMISSIONS_URL.format(name=page), headers=SEC_HEADERS)
                sleep(SEC_MIN_INTERVAL)
                rows += se.parse_submissions_filings(p.json())
                seen_pages.setdefault(c10, []).append(page)
            obs = se.filings_to_observations(c10, rows, main.retrieved_at)
            filing_frames.append(obs_to_rows(obs))
            if obs:
                first = min(o.available_at for o in obs).isoformat()
                since[c10] = min(since.get(c10, first), first)
            res["ciks"] += 1
            # live Form 4 nach Bulk-Ende (EXACT_TIMESTAMP aus acceptanceDateTime)
            if bulk_end is not None:
                for r in rows:
                    if r.get("form") != "4" or not r.get("acceptanceDateTime"):
                        continue
                    acc_t = pd.Timestamp(r["acceptanceDateTime"])
                    if acc_t.tzinfo is None:
                        acc_t = acc_t.tz_localize("UTC")
                    if acc_t < bulk_end[1]:
                        continue
                    if budget <= 0:
                        res["live_complete"] = False
                        break
                    budget -= 1
                    acc = r["accessionNumber"]
                    try:
                        # Index der Filing-Dokumente ist nicht nötig: das Roh-XML trägt den Namen des primaryDocument
                        idx = get(f"https://www.sec.gov/Archives/edgar/data/{int(c10)}/{acc.replace('-', '')}/{acc}.txt",
                                  headers=SEC_HEADERS)
                        sleep(SEC_MIN_INTERVAL)
                        txt = idx.content.decode("utf-8", errors="replace")
                        xml = txt[txt.find("<ownershipDocument>"):txt.find("</ownershipDocument>") + 20]
                        parsed = parse_form4_xml(xml)
                    except Exception as e:  # noqa: BLE001
                        res["errors"].append(f"form4 {acc}: {e!r}")
                        res["live_complete"] = False
                        continue
                    for k, tx in enumerate(parsed["transactions"]):
                        if tx["code"] not in se.OPEN_MARKET_CODES or tx["shares"] is None or tx["price"] is None:
                            continue
                        live_frames.append(obs_to_rows([Observation(
                            source_id="sec_form345", dataset="form4_live", series_id=f"{acc}#live{k}",
                            entity_id=f"cik:{c10}", metric=se.OPEN_MARKET_CODES[tx["code"]],
                            value=round(tx["shares"] * tx["price"], 2), unit="USD",
                            observation_time=se.parse_date(tx["date"]) or acc_t.to_pydatetime(),
                            available_at=acc_t.to_pydatetime(), retrieved_at=idx.retrieved_at,
                            availability_precision=AvailabilityPrecision.EXACT_TIMESTAMP,
                            parser_version=se.PARSER_VERSION, payload_hash=idx.content_hash,
                            attrs={"accession": acc, "owner_ciks": parsed["owners"], "code": tx["code"],
                                   "shares": tx["shares"], "price": tx["price"]})]))
                        res["live_form4"] += 1
        except Exception as e:  # noqa: BLE001
            res["errors"].append(f"submissions {c10}: {e!r}")
    if filing_frames:
        _, res["new_rows"] = merge_store(DIR / "filings.csv.gz", pd.concat(filing_frames, ignore_index=True))
    if live_frames:
        merge_store(DIR / "insider.csv.gz", pd.concat(live_frames, ignore_index=True))
    if res["live_complete"] and bulk_end is not None:
        state["live_insider_through"] = now.isoformat()
    return res


def health(state: dict, results: dict, n_ciks: int, now: datetime) -> dict:
    ins = read_store(DIR / "insider.csv.gz")
    fil = read_store(DIR / "filings.csv.gz")
    errs = sum(len(r.get("errors", [])) for r in results.values())
    calls = max(1, len(results.get("form345", {}).get("fetched", [])) + results.get("submissions", {}).get("ciks", 0) + errs)
    return {"source_id": "sec_deep_events", "schema_version": SCHEMA_VERSION, "checked_at": now.isoformat(),
            "last_success": now.isoformat() if errs < calls else state.get("last_success"),
            "last_observation": max([x for x in (ins.get("available_at", pd.Series(dtype=str)).max() if len(ins) else None,
                                                fil.get("available_at", pd.Series(dtype=str)).max() if len(fil) else None)
                                     if isinstance(x, str)], default=None),
            "coverage_ciks_with_filings": len(state.get("filing_since", {})), "universe_ciks": n_ciks,
            "coverage": round(len(state.get("filing_since", {})) / n_ciks, 3) if n_ciks else 0.0,
            "form345_quarters": sorted(state.get("form345_quarters", {})),
            "live_insider_through": state.get("live_insider_through"),
            "error_rate": round(errs / calls, 3), "errors_sample": [e for r in results.values()
                                                                    for e in r.get("errors", [])][:20]}


def build_features(state: dict, cik_by_ticker: dict[str, str], dates) -> pd.DataFrame:
    obs = []
    for f in ("insider.csv.gz", "filings.csv.gz"):
        df = read_store(DIR / f)
        if len(df):
            obs += rows_to_obs(df)
    quarters = [tuple(int(x) for x in k.split("Q")) for k in state.get("form345_quarters", {})]
    cov = sf.insider_coverage(quarters)
    if cov and state.get("live_insider_through"):
        cov = (cov[0], max(cov[1], pd.Timestamp(state["live_insider_through"])))
    since = {f"cik:{k}": pd.Timestamp(v) for k, v in state.get("filing_since", {}).items()}
    return sf.build_feature_table(obs, dates, {t: f"cik:{c}" for t, c in cik_by_ticker.items()}, cov, since)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=2014)
    ap.add_argument("--live-budget", type=int, default=400)
    ap.add_argument("--features-only", action="store_true")
    args = ap.parse_args(argv)
    now = _now()
    store = EntityStore(ENTITY_PATH)
    ident = {r.ticker: r.cik for r in store.records if r.usage == "research_ticker_identity" and r.valid_to is None}
    if not ident:
        log.error("sec_ingest: keine Entity-Map – zuerst python -m modules.entity_resolution.build")
        return 1
    state_path = DIR / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    results = {}
    if not args.features_only:
        results["form345"] = ingest_form345(state, set(ident.values()), args.start_year, now)
        results["submissions"] = ingest_submissions(state, sorted(set(ident.values())), now,
                                                    live_budget=args.live_budget)
        state["last_success"] = now.isoformat()
        DIR.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=1, sort_keys=True))
        (DIR / "health.json").write_text(json.dumps(health(state, results, len(set(ident.values())), now), indent=1))
    dates = pd.date_range("2015-01-02", now.date(), freq="W-FRI")
    feat = build_features(state, ident, dates)
    FEATURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    feat.to_csv(FEATURE_PATH, index=False, compression="gzip")
    print(json.dumps({k: {kk: (vv if kk != "errors" else len(vv)) for kk, vv in v.items()} for k, v in results.items()},
                     indent=1), f"\nFeatures: {len(feat)} Zeilen, verfügbar {feat['alt_sec_available'].mean():.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

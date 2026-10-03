"""TED-Zuschläge: Ingestion (budgetiert, inkrementell) + Feature-Tabelle.

    python -m modules.external.sources.ted_ingest [--start-year 2014] [--budget 1500] [--features-only] [--probe-only]

1. Feld-Probe je Jahr: Anteil der Ergebnis-Bekanntmachungen mit Gewinnernamen. Erst ab dem
   ersten Jahr, ab dem dieser Anteil durchgehend >= PROBE_MIN_SHARE ist, gelten fehlende
   Treffer als echte Null (vorher NaN).
2. Je Entität (Entity-Map, research_ticker_identity) und Jahr: Suche nach dem Firmennamen,
   alle Seiten (max. MAX_PAGES, sonst Jahr "unvollständig"), strenger Namensabgleich.
   Vergangene, vollständige Jahre werden nie erneut abgefragt; das laufende Jahr bei jedem Lauf.
3. Event-Speicher outputs/external_data/ted/awards/<jahr>.csv.gz (append-only, wie SEC).
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from modules.entity_resolution.store import DEFAULT_PATH as ENTITY_PATH
from modules.entity_resolution.store import EntityStore, normalize_name
from modules.external.http import fetch
from modules.external.sources import ted_events as te
from modules.external.sources import ted_features as tf
from modules.external.sources.sec_ingest import merge_store, obs_to_rows, read_store

log = logging.getLogger(__name__)
DIR = Path("outputs/external_data/ted")
STORE = DIR / "awards.csv.gz"
FEATURE_PATH = Path("outputs/research/feature_store/alt_ted.csv.gz")
MIN_INTERVAL = 0.6
PROBE_MIN_SHARE = 0.5
HEADERS = {"Accept": "application/json", "Content-Type": "application/json"}


def _post(query_body: dict):
    return fetch(te.SEARCH_URL, method="POST", json_body=query_body, headers=HEADERS, timeout=60)


def probe(state: dict, start_year: int, now: datetime, post=_post, sleep=time.sleep) -> dict:
    cov = state.setdefault("field_coverage", {})
    res = {"errors": []}
    for y in range(start_year, now.year + 1):
        if str(y) in cov and y < now.year:
            continue
        try:
            r = post(te.body(te.PROBE_TEMPLATE.format(start=f"{y}0101", end=f"{y}1231"), limit=100))
            notices, _ = te.parse_search(r.json())
            share = te.probe_winner_share(notices)
            cov[str(y)] = None if share is None else round(share, 3)
        except Exception as e:  # noqa: BLE001 – Probe-Fehler -> Abdeckung unbekannt (NaN), nie 0
            res["errors"].append(f"probe {y}: {e!r}"[:400])
        sleep(MIN_INTERVAL)
    state["field_start_year"] = field_start(cov, now.year)
    res["field_start_year"] = state["field_start_year"]
    return res


def field_start(cov: dict, current_year: int) -> int | None:
    """Erstes Jahr, ab dem bis heute jedes Jahr Gewinnernamen zuverlässig liefert."""
    start = None
    for y in range(current_year, 1990, -1):
        v = cov.get(str(y))
        if v is None or v < PROBE_MIN_SHARE:
            break
        start = y
    return start


def ingest(state: dict, entities: list[tuple[str, str, list[str]]], now: datetime, post=_post,
           sleep=time.sleep, budget: int = 1500) -> dict:
    """entities: [(cik, Suchname, Aliasnamen)]."""
    res = {"requests": 0, "new_rows": 0, "errors": [], "entity_years_done": 0, "truncated": 0, "budget_exhausted": False}
    fs = state.get("field_start_year")
    if fs is None:
        res["errors"].append("keine Feld-Abdeckung (Probe) -> keine Abfragen")
        return res
    ent_state = state.setdefault("entities", {})
    frames = []
    for ent in entities:
        cik, name, aliases = ent[:3]
        subs = ent[3] if len(ent) > 3 else None
        years = ent_state.setdefault(cik, {}).setdefault("years", {})
        for y in range(fs, now.year + 1):
            info = years.get(str(y))
            if info and info.get("complete") and y < now.year:
                continue
            notices, page, total, ok = [], 1, None, True
            while True:
                if res["requests"] >= budget:
                    res["budget_exhausted"] = True
                    break
                try:
                    r = post(te.body(te.query_for(name, y), page=page))
                    res["requests"] += 1
                    batch, total = te.parse_search(r.json())
                except Exception as e:  # noqa: BLE001
                    res["errors"].append(f"{cik} {y}: {e!r}"[:400])
                    ok = False
                    break
                finally:
                    sleep(MIN_INTERVAL)
                notices += batch
                if not batch or page * te.PAGE_LIMIT >= total:
                    break
                if page >= te.MAX_PAGES:
                    res["truncated"] += 1
                    ok = False
                    break
                page += 1
            if res["budget_exhausted"]:
                return _finish(res, frames)
            obs = te.notices_to_observations(cik, notices, aliases, now, sub_aliases=subs)
            if obs:
                frames.append(obs_to_rows(obs))
            years[str(y)] = {"complete": ok, "fetched_at": now.isoformat(), "n_notices": len(notices),
                             "n_matched": len(obs)}
            res["entity_years_done"] += 1
    return _finish(res, frames)


def _finish(res: dict, frames: list) -> dict:
    if frames:
        _, res["new_rows"] = merge_store(STORE, pd.concat(frames, ignore_index=True))
    return res


def entities_from_store(store: EntityStore) -> tuple[list[tuple], dict[str, str]]:
    """-> [(cik, Suchname, Aliasnamen, [(Tochtername, valid_from, valid_to)])], {ticker: cik}."""
    ents, ident = {}, {}
    subs: dict[str, set] = {}
    for r in store.records:
        if r.usage == "research_ticker_identity" and r.valid_to is None and r.cik:
            ident[r.ticker] = r.cik
            names = [r.canonical_name] + list(r.aliases or [])
            ents.setdefault(r.cik, (normalize_name(r.canonical_name) or r.canonical_name, set()))[1].update(names)
        elif r.usage == "exposure_subsidiary" and (r.parent_entity or "").startswith("cik:") \
                and r.mapping_confidence in ("HIGH", "MEDIUM"):
            for a in {r.canonical_name, *(r.aliases or [])}:
                subs.setdefault(r.parent_entity[4:], set()).add((a, r.valid_from, r.valid_to))
    return [(c, n, sorted(a), sorted(subs.get(c, set()), key=str)) for c, (n, a) in sorted(ents.items())], ident


def health(state: dict, res: dict, now: datetime) -> dict:
    ents = state.get("entities") or {}
    fs = state.get("field_start_year")
    done = sum(1 for e in ents.values() if fs is not None and all(
        (e.get("years") or {}).get(str(y), {}).get("complete") for y in range(fs, now.year + 1)))
    st = read_store(STORE)
    last = str(st["available_at"].max()) if len(st) else None
    return {"source_id": "ted_awards", "checked_at": now.isoformat(), "field_start_year": fs,
            "last_observation": last, "n_awards": int(len(st)),
            "field_coverage": state.get("field_coverage"), "entities_complete": done, "entities_total": len(ents),
            "requests": res.get("requests"), "truncated": res.get("truncated"),
            "budget_exhausted": res.get("budget_exhausted"),
            "error_rate": round(len(res.get("errors") or []) / max(1, res.get("requests") or 1), 3),
            "errors_sample": (res.get("errors") or [])[:5]}


def build_features(state: dict, ident: dict[str, str], dates) -> pd.DataFrame:
    years = {cik: e.get("years") or {} for cik, e in (state.get("entities") or {}).items()}
    return tf.build_feature_table(read_store(STORE), dates, ident, years, state.get("field_start_year"))


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=2014)
    ap.add_argument("--budget", type=int, default=1500)
    ap.add_argument("--features-only", action="store_true")
    ap.add_argument("--probe-only", action="store_true")
    args = ap.parse_args(argv)
    now = datetime.now(timezone.utc)
    ents, ident = entities_from_store(EntityStore(ENTITY_PATH))
    if not ents:
        log.error("ted_ingest: keine Entity-Map – zuerst python -m modules.entity_resolution.build")
        return 1
    state_path = DIR / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    res = {}
    if not args.features_only:
        res = probe(state, args.start_year, now)
        print("TED-Feldabdeckung je Jahr:", json.dumps(state.get("field_coverage")), "Start:", state.get("field_start_year"),
              "Probe-Fehler:", res["errors"][:3])
        if not args.probe_only:
            res.update(ingest(state, ents, now, budget=args.budget))
        DIR.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=1, sort_keys=True))
        (DIR / "health.json").write_text(json.dumps(health(state, res, now), indent=1))
    feat = build_features(state, ident, pd.date_range("2015-01-02", now.date(), freq="W-FRI"))
    FEATURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    feat.to_csv(FEATURE_PATH, index=False, compression="gzip")
    print(json.dumps({k: (v if k != "errors" else v[:3]) for k, v in res.items()}, default=str)[:1500],
          f"\nFeatures: {len(feat)} Zeilen, verfügbar {feat['alt_ted_available'].mean() if len(feat) else 0:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

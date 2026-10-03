"""Entity-Map aufbauen/aktualisieren (inkrementell, budgetiert).

    python -m modules.entity_resolution.build [--gleif-budget 150]

1. SEC company_tickers.json -> Ticker/CIK/Name (alle börsennotierten Registranten).
2. Für das Research-Universum (PIT, inkl. entfernter Titel, sofern bei der SEC
   noch geführt) die SEC-Submissions (Name, frühere Namen, SIC, Sitz, Website).
3. GLEIF: LEI-Zuordnung + Eltern, nur für noch nicht zugeordnete Entitäten,
   höchstens `gleif_budget` Firmen je Lauf (GLEIF erlaubt ~60 Anfragen/min).
Ergebnis: outputs/entity/entity_map.jsonl (append-only) + entity_report.json.
Netzfehler einer Quelle stoppen nie den Lauf; sie landen im Report.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from modules.entity_resolution import sources as src
from modules.entity_resolution.store import DEFAULT_PATH, EntityStore

log = logging.getLogger(__name__)
REPORT = Path("outputs/entity/entity_report.json")
SEC_MIN_INTERVAL = 0.12          # SEC Fair Access: <= 10 Anfragen/s
GLEIF_MIN_INTERVAL = 1.05        # <= ~57 Anfragen/min


def build(tickers_wanted: list[str], store: EntityStore, today: str, *, fetch_tickers=src.fetch_sec_tickers,
          fetch_submissions=src.fetch_sec_submissions, fetch_gleif=src.fetch_gleif_by_name,
          fetch_parent=src.fetch_gleif_parent, gleif_budget: int = 150, sleep=time.sleep) -> dict:
    rep = {"date": today, "sec": {}, "gleif": {}, "errors": []}
    try:
        rows = fetch_tickers()
    except Exception as e:  # noqa: BLE001 – Quelle aus, Lauf geht weiter
        rep["errors"].append(f"sec_tickers: {e!r}")
        return rep
    wanted = {t.upper().replace(".", "-") for t in tickers_wanted}
    sel = [r for r in rows if r["ticker"] in wanted]
    rep["sec"] = {"tickers_total": len(rows), "wanted": len(wanted), "mapped": len(sel),
                  "unmapped": sorted(wanted - {r["ticker"] for r in sel})[:200]}
    subs = {}
    for r in sel:
        try:
            subs[r["cik"]] = src.parse_submissions_entity(fetch_submissions(r["cik"]))
        except Exception as e:  # noqa: BLE001
            rep["errors"].append(f"sec_submissions {r['ticker']}: {e!r}")
        sleep(SEC_MIN_INTERVAL)
    changes = {"opened": 0, "replaced": 0, "unchanged": 0}
    sec_recs = src.sec_records(sel, subs)
    for rec in sec_recs:
        changes[store.upsert(rec, today)] += 1
    rep["sec"]["changes"] = changes
    # GLEIF nur für PIT-Datensätze ohne offene LEI-Zuordnung
    done = {r.entity_id for r in store.records if r.mapping_source == "gleif_name_match" and r.valid_to is None}
    todo = [r for r in sec_recs if r.usage == "pit" and r.entity_id not in done][:gleif_budget]
    g = {"attempted": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "with_parent": 0}
    for rec in todo:
        g["attempted"] += 1
        try:
            cands = fetch_gleif(rec.canonical_name)
            sleep(GLEIF_MIN_INTERVAL)
            m = src.match_lei(rec.aliases or [rec.canonical_name], cands, country=rec.country or "US")
            g[m["confidence"]] += 1
            if m["confidence"] == "LOW":
                continue                                    # nie als Zuordnung speichern
            direct = fetch_parent(m["lei"], "direct")
            sleep(GLEIF_MIN_INTERVAL)
            ultimate = fetch_parent(m["lei"], "ultimate")
            sleep(GLEIF_MIN_INTERVAL)
            for gr in src.gleif_records(rec, m, direct, ultimate):
                store.upsert(gr, today)
                if gr.mapping_source == "gleif_parent":
                    g["with_parent"] += 1
        except Exception as e:  # noqa: BLE001
            rep["errors"].append(f"gleif {rec.ticker}: {e!r}")
    g["remaining"] = max(0, len([r for r in sec_recs if r.usage == "pit"]) - len(done) - g["attempted"])
    rep["gleif"] = g
    return rep


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--gleif-budget", type=int, default=150)
    args = ap.parse_args(argv)
    from modules.universe import get_universe, research_universe
    tickers = set(get_universe())
    try:
        tickers |= set(research_universe("2014-01-01")["tickers"])
    except Exception as e:  # noqa: BLE001 – ohne Historie nur heutiges Universum
        log.warning(f"entity build: PIT-Universum nicht verfügbar ({e})")
    store = EntityStore(DEFAULT_PATH)
    rep = build(sorted(tickers), store, src.utc_today(), gleif_budget=args.gleif_budget)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in rep.items() if k != "errors"}, indent=1), f"\nFehler: {len(rep['errors'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

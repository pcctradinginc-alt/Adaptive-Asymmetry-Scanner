"""Einmal-Migration (Audit 2026-09-27): Eurostat-Archivzeilen, deren
available_at die Datensatz-Update-Zeit ist, waren für die GESAMTE Historie
als EXACT_TIMESTAMP etikettiert. Exakt ist das nur für die jüngste Periode
je Abruf (gleiches available_at); ältere Perioden -> CONSERVATIVE_DATE.
Ändert ausschließlich availability_precision -- nie Werte, available_at
oder Vintages. Idempotent."""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path("outputs/external_data/normalized")
SOURCES = ["eurostat_sentiment", "eurostat_industrial_production", "eurostat_road_freight"]


def migrate(root: Path = ROOT) -> dict:
    changed = {}
    for sid in SOURCES:
        files = sorted((root / sid).glob("*.jsonl"))
        rows = {f: [json.loads(l) for l in f.read_text().splitlines() if l.strip()] for f in files}
        latest_by_avail = defaultdict(str)
        for rs in rows.values():
            for r in rs:
                latest_by_avail[r["available_at"]] = max(latest_by_avail[r["available_at"]], r["observation_time"])
        n = 0
        for f, rs in rows.items():
            dirty = False
            for r in rs:
                if r["availability_precision"] == "EXACT_TIMESTAMP" and \
                        r["observation_time"] < latest_by_avail[r["available_at"]]:
                    r["availability_precision"] = "CONSERVATIVE_DATE"
                    n += 1
                    dirty = True
            if dirty:
                f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs))
        changed[sid] = n
    return changed


if __name__ == "__main__":
    print(migrate())

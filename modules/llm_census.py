"""LLM-Zensus: zufällige Tiefenanalyse ohne Gates (Aktionsplan 2026-11-09, Punkt 6, nur Research).

Problem: Impact, Surprise und Mismatch – die Kernvariablen der Underreaction-Hypothese – existieren nur für die
höchstens 3 Kandidaten je Lauf, die alle Gates bis zur Deep Analysis passieren (economics-first). Ob diese
Variablen überhaupt Vorhersagekraft haben, ist so nicht schätzbar (Selektion vor dem Messpunkt).

Verfahren:
  * Je Lauf k Kandidaten aus ALLEN Hard-Filter-Überlebenden (nach FinBERT, vor jedem weiteren Gate), gleichverteilt
    ohne Zurücklegen, deterministisch über das Datum (ergebnisunabhängig); Einschlusswahrscheinlichkeit π = k/N
    wird je gezogener Zeile protokolliert (llm_census).
  * Ausgewertet am Laufende (nach allen Champion-Entscheidungen, mit eigenem Zeitbudget): Deep Analysis (gleiche
    Prompts, Temperatur 0, Cache), Mismatch inklusive der gerichteten Varianten. Ergebnisse NUR mit Präfix
    census_* im Kandidaten-Ledger – nie das Feld direction, nie im Trade-, Snapshot- oder Score-Pfad.
  * Bericht (report): Rangkorrelation von census_impact / census_mismatch* mit der richtungsbereinigten
    Underlying-Rendite (ret_20d/ret_45d · Thesenrichtung) auf der Zufallsstichprobe; wegen gleicher π sind
    einfache Mittel erwartungstreu für die Hard-Filter-Population. Beschreibend, keine Bestätigung.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import time
from datetime import date, datetime, timezone
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

CONFIG = Path("config/llm_census.yaml")
OUT = Path("outputs/research/llm_census.json")
POLICY = "uniform-without-replacement-v1"
DEFAULTS = {"enabled": True, "execution": "external", "k": 5, "budget_s": 420, "min_reserve_s": 120}
HORIZONS = (20, 45)


def settings(path: Path = CONFIG) -> dict:
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        raw = {}
    return {**DEFAULTS, **{k: v for k, v in raw.items() if k in DEFAULTS}}


def select(candidates: list[dict], day: str, k: int) -> tuple[list[dict], dict]:
    """(gezogene Kandidaten als tiefe Kopie, Design). Reihenfolge per sha1(Datum:Ticker) – reproduzierbar und
    unabhängig von jedem Ergebnis."""
    pool = [c for c in candidates if isinstance(c, dict) and c.get("ticker")]
    n = len(pool)
    kk = max(0, min(int(k), n))
    order = sorted(pool, key=lambda c: hashlib.sha1(f"census:{day}:{c['ticker']}".encode()).hexdigest())
    design = {"policy": POLICY, "day": str(day), "n": n, "k": kk, "pi": (kk / n) if n else 0.0}
    return [copy.deepcopy(c) for c in order[:kk]], design


def _record(a: dict, design: dict, scorer) -> dict:
    da = a.get("deep_analysis") or {}
    res = scorer._score(a)
    sh = scorer.shadow.get(a["ticker"]) or {}
    rec = {"census_impact": da.get("impact"), "census_surprise": da.get("surprise"),
           "census_direction": da.get("direction"), "census_bear_case_severity": da.get("bear_case_severity"),
           "census_mismatch": ((res or {}).get("features") or {}).get("mismatch", sh.get("mismatch_raw")),
           "llm_census": {**design, "status": "analyzed"}}
    for k in ("mismatch_dir_48h", "mismatch_dir_news", "move_since_news"):
        if k in sh:
            rec[f"census_{k}"] = sh[k]
    return rec


def run(picks: list[dict], design: dict, deadline: float | None, analyzer=None, scorer_cls=None,
        note=None) -> dict:
    """Deep Analysis + Mismatch für die gezogenen Kandidaten; Ergebnis nur mit census_*-Präfix in den Ledger."""
    if not picks:
        return {"status": "empty", **design}
    if note is None:
        from modules import candidate_ledger
        note = candidate_ledger.note
    if analyzer is None:
        from modules.deep_analysis import DeepAnalysis
        analyzer = DeepAnalysis()
    if scorer_cls is None:
        from modules.mismatch_scorer import MismatchScorer
        scorer_cls = MismatchScorer
    analyses = analyzer.run(picks, deadline=deadline) or []
    done = set()
    scorer = scorer_cls()
    for a in analyses:
        try:
            note(a["ticker"], **_record(a, design, scorer))
            done.add(a["ticker"])
        except Exception as e:  # noqa: BLE001 – Research-Messung, nie lauf-brechend
            log.warning(f"LLM-Zensus: {a.get('ticker')} nicht notiert ({e})")
    skipped = set(getattr(analyzer, "skipped_for_time", []) or [])
    for c in picks:
        if c["ticker"] not in done:
            why = "skipped_time" if c["ticker"] in skipped else "no_analysis"
            note(c["ticker"], llm_census={**design, "status": why})
    return {"status": "ok", **design, "analyzed": len(done), "not_analyzed": len(picks) - len(done)}


def deadline_for(run_deadline: float | None, reserve_s: float, cfg: dict) -> float:
    """Eigenes Zeitbudget am Laufende; lässt immer min_reserve_s für Ledger, Bericht und Mail übrig."""
    now = time.monotonic()
    hard = (run_deadline + reserve_s - cfg["min_reserve_s"]) if run_deadline else now + cfg["budget_s"]
    return min(now + cfg["budget_s"], hard)


def _spearman(a: list[float], b: list[float]) -> float | None:
    import numpy as np
    if len(a) < 5:
        return None
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0:
        return None
    return round(float(np.corrcoef(ra, rb)[0, 1]), 4)


def report(rows: list[dict], as_of: date | None = None) -> dict:
    """Beschreibende Population-Auswertung der Zensus-Zeilen (gleiche π -> einfache Mittel erwartungstreu)."""
    cen = [r for r in rows if isinstance(r, dict)
           and ((r.get("features") or {}).get("llm_census") or {}).get("status") == "analyzed"]
    out = {"policy": POLICY, "as_of": str(as_of) if as_of else None, "analyzed_rows": len(cen),
           "days": len({r.get("date") for r in cen}), "production_influence": "NONE", "horizons": {}}
    for h in HORIZONS:
        pts = []
        for r in cen:
            f, oc = r.get("features") or {}, r.get("outcomes") or {}
            ret = oc.get(f"ret_{h}d")
            if not isinstance(ret, (int, float)):
                continue
            s = -1.0 if str(f.get("census_direction") or "").upper() == "BEARISH" else 1.0
            pts.append((f, s * float(ret)))
        res = {"n": len(pts), "mean_dir_return": round(sum(y for _, y in pts) / len(pts), 6) if pts else None}
        for k in ("census_impact", "census_mismatch", "census_mismatch_dir_48h", "census_mismatch_dir_news"):
            xy = [(float(f[k]), y) for f, y in pts if isinstance(f.get(k), (int, float))]
            res[f"ic_{k}"] = _spearman([x for x, _ in xy], [y for _, y in xy]) if xy else None
        out["horizons"][f"{h}d"] = res
    out["note"] = ("Zufallsstichprobe der Hard-Filter-Population (π je Zeile protokolliert); Rangkorrelation mit der "
                   "richtungsbereinigten Underlying-Rendite; beschreibend, keine Bestätigung.")
    return out


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["report"])
    ap.add_argument("--as-of", default=None)
    args = ap.parse_args(argv)
    from modules.challenger import load_ledger_rows
    as_of = date.fromisoformat(args.as_of) if args.as_of else datetime.now(timezone.utc).date()
    rep = report(load_ledger_rows(), as_of)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("analyzed_rows", "days")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
